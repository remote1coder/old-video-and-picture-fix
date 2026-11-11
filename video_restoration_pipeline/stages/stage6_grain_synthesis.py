#!/usr/bin/env python3
"""
Grain Synthesis (Video) - Stage 6
Adds film grain to video after all processing is complete.

v6.3 Compliance:
- Positioned after all restoration (stage2-5) and before final encoding (stage7)
- Input source STRICT validation:
  * face_restore enabled -> MUST receive stage5 output
  * face_restore disabled -> MUST receive stage4 output
  * STRICTLY FORBIDDEN to receive stage3 output directly
- GrainFactory3 with g1str/g2str/g3str (havsfunc r33)
- 16bit processing space (YUV420P16)
- No taps/shutter/correlation (not GrainFactory3 params)
"""

import argparse
import sys
import os
import json
import subprocess
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Tuple
import glob

SCRIPT_DIR = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))

from logger import get_logger  # noqa: E402
from checkpoint import wsl_redirect_path, select_media_candidates  # FIX-P9-3 / R3-01

try:  # NEW-V-06: 取消/中断守卫（KeyboardInterrupt 继承 BaseException，
    # 本文件所有 `except Exception` 都抓不到它——不装守卫时 Ctrl+C 会把
    # 半截 mkv 留在 temp 里被下游 stage 复用）
    import cancel_guard as _cancel_guard
except Exception:  # pragma: no cover - 缺失时不阻断 stage，仅失去该保护
    _cancel_guard = None


def _vs_deterministic(params) -> bool:
    """本次运行是否要求确定性渲染（`vs.deterministic`）。"""
    try:
        from vs_render import wants_deterministic
        return bool(wants_deterministic(params))
    except Exception:                 # pragma: no cover
        return False


def _determinism_block(deterministic: bool) -> str:
    """生成 vpy 里的确定性前置代码（`core.num_threads = 1` 等）。

    必须放在 `core = vs.core` 之后、**任何 filter 创建之前**。
    stage6 用的是 `haf.GrainFactory3`（含 `temp_avg` 时域平均），
    与 stage1 的 mvtools 同属"可能受并行影响"的一类，所以走同一开关。
    """
    if not deterministic:
        return ""
    try:
        from vs_render import determinism_preamble as _dp
        lines = _dp({"vs_deterministic": True})
    except Exception:                 # pragma: no cover
        return ""
    return "".join(line + "\n" for line in lines)


def _build_vs_script(input_source: str, vs_format: str,
                     g1str: float, g2str: float, g3str: float,
                     addgrainc_ratio: float,
                     use_addgrainc: bool = False,
                     addgrainc_after_gf3: bool = False,
                     temp_avg: float = 0.0,
                     deterministic: bool = False,
                     seed: Optional[int] = None) -> str:
    """Build VapourSynth grain synthesis script.

    Module-level function for multiprocessing spawn compliance (CLAUDE.md §5).

    FIX-P9-4: per v7.32 §3.10.4 / v6.32 §7.17, AddGrainC has two modes:
    - use_addgrainc=True, addgrainc_after_gf3=False: standalone fallback
      (GrainFactory3 completely unavailable)
    - addgrainc_after_gf3=True: supplement mode, stacks AFTER GrainFactory3
      to enhance temporal continuity. Intensity ≤20% of g1str. Disabled by default.

    R22（N22 实测）：`seed=None` 时传 `-1`，即 `haf.GrainFactory3` 的默认
    **每次运行随机**。实测同一输入连跑两遍：step1–step4 逐像素相同
    （PSNR inf），**step6 的 Y 平面不同**（psnr_y 51.9，U/V 相同），
    成品跟着不同（53 dB）。`grain_seed` 这个参数名此前只出现在
    `stage_deps` 的依赖表里、**从未被读取**（"声明的参数没人用"病族），
    于是"可重复"这件事在 stage6 之后就断了。现在给它一条真实通路。
    """
    if use_addgrainc and not addgrainc_after_gf3:
        # Standalone AddGrainC fallback (GrainFactory3 completely unavailable)
        var = max(0.0, g1str * addgrainc_ratio)
        uvar = max(0.0, g2str * addgrainc_ratio)
        vvar = max(0.0, g3str * addgrainc_ratio)
        return (
            f"import vapoursynth as vs\n"
            f"core = vs.core\n"
            + _determinism_block(deterministic) +
            f"clip = core.ffms2.Source(source={repr(input_source)})\n"
            f"clip = core.resize.Bicubic(clip, format={vs_format})\n"
            f"clip = core.grain.Add(clip, var={var}, uvar={uvar}, "
            f"vvar={vvar}, constant=False{_seed_arg(seed, False)})\n"
            f"clip.set_output()\n"
        )

    # GrainFactory3 main path
    script = (
        f"import vapoursynth as vs\n"
        f"core = vs.core\n"
        + _determinism_block(deterministic) +
        f"import havsfunc as haf\n"
        f"clip = core.ffms2.Source(source={repr(input_source)})\n"
        f"clip = core.resize.Bicubic(clip, format={vs_format})\n"
        f"clip = haf.GrainFactory3(clip, g1str={g1str}, "
        f"g2str={g2str}, g3str={g3str}, temp_avg={temp_avg}"
        f"{_seed_arg(seed, True)})\n"
    )
    if addgrainc_after_gf3:
        # FIX-P9-4: Stack AddGrainC after GrainFactory3 for temporal continuity
        # per v7.32 §3.10.4: intensity ≤20% of g1str, disabled by default
        var = max(0.0, g1str * addgrainc_ratio)
        uvar = max(0.0, g2str * addgrainc_ratio)
        vvar = max(0.0, g3str * addgrainc_ratio)
        script += (
            f"clip = core.grain.Add(clip, var={var}, uvar={uvar}, "
            f"vvar={vvar}, constant=False{_seed_arg(seed, False)})\n"
        )
    script += "clip.set_output()\n"
    return script


def _seed_arg(seed, _for_gf3):
    """把颗粒随机种子拼成 vapoursynth 脚本片段。

    `seed is None` → 空串（用滤镜默认 `-1`，即每次随机，与历史行为一致）。
    给了整数就写成 `, seed=N` —— `haf.GrainFactory3` 与 `core.grain.Add`
    都接受这个关键字。
    """
    if seed is None:
        return ""
    try:
        return ", seed=" + str(int(seed))
    except (TypeError, ValueError):
        return ""


class _TailLog:
    """边排空边落盘的**有界**日志容器。

    审核 R18-A07：
    > stderr 虽被并发排空，却持续保存在列表，长任务的日志内存没有上限。

    旧实现 `_sinks = {"enc": [], "vs": []}` 把子进程的**全部** stderr
    留在内存里；4K 长片 + 详细日志可以到几百 MB。

    现在：字节流直接写进本 attempt 的日志文件（**不丢**），
    内存只保留最后 ``limit`` 字节（用于错误提示）。
    """

    __slots__ = ("path", "limit", "total", "_tail", "_fh")

    def __init__(self, path: Optional[str] = None, limit: int = 65536) -> None:
        self.path = path
        self.limit = int(limit)
        self.total = 0
        self._tail = bytearray()
        self._fh = None
        if path:
            try:
                self._fh = open(path, "ab")
            except OSError:
                self._fh = None

    def write(self, chunk) -> None:
        if not chunk:
            return
        self.total += len(chunk)
        if self._fh is not None:
            try:
                self._fh.write(chunk)
                self._fh.flush()
            except OSError:
                self._fh = None        # 落盘失败不该拖垮管道
        self._tail.extend(chunk)
        over = len(self._tail) - self.limit
        if over > 0:
            del self._tail[:over]

    def tail_bytes(self) -> bytes:
        return bytes(self._tail)

    def tail_text(self) -> str:
        return self.tail_bytes().decode("utf-8", errors="replace")

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None


def _sha256_file(path: str, chunk: int = 1 << 20):
    """文件的 SHA-256；读不出来返回 None。

    R22-A06：降级侧车要绑定**产物内容**。本文件原来没有这个工具，
    直接调用会 `NameError` —— 与 R20 stage7 缺 `_sha256_text`、
    stage3 缺 `_sha256_file` 是同一类问题（**语法通过 ≠ 能跑**）。
    """
    import hashlib
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while True:
                b = fh.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except OSError:
        return None


@dataclass
class GrainPipelineResult:
    """stage6 颗粒合成管道的**结构化**结果（审核 R19-B04）。

    旧实现返回 ``(int, str)`` 混装 tuple，调用方按两段字符串解包后
    再做 ``"GrainFactory3" in err_str`` —— ``err_str`` 实为整数 84，
    触发 ``TypeError: argument of type 'int' is not iterable``。

    字段说明：

    - ``ok``：整体是否成功。
    - ``vspipe_rc`` / ``ffmpeg_rc``：**各自**的 returncode（未启动则为 None）。
    - ``failed_stage``：``""`` / ``"start"`` / ``"vspipe"`` / ``"ffmpeg"`` /
      ``"wait"`` / ``"exception"``，用于区分"插件不存在"与"编码器出错"。
    - ``cancelled``：是否被取消/中断。**取消绝不能当成插件失败去回退**。
    - ``log_paths``：本次 attempt 的 stderr 落盘路径。
    - ``stderr_tail``：有界尾部（合并 vspipe + ffmpeg）。
    - ``vspipe_tail`` / ``encoder_tail``：**分开**保存两侧尾部（R21-C01）。
      合并尾部无法回答"这句报错是生产者说的还是编码器说的"，
      而"首发原因"正是回退判定的唯一依据。
    - ``detail``：人类可读摘要。
    """

    ok: bool = False
    vspipe_rc: Optional[int] = None
    ffmpeg_rc: Optional[int] = None
    failed_stage: str = ""
    #: 失败发生在 **Popen 之前/之中**（进程根本没跑起来）。
    #: 与 `failed_stage` 分开：后者说"哪一步"，这个说"有没有真的启动"。
    failed_at_start: bool = False
    #: **谁先退出**："vspipe" / "ffmpeg" / ""（同轮轮询同时看到 → 顺序不明）。
    #: R22-A06：20ms 轮询得到的是**观察顺序**，不是严格因果；同一轮里两个都
    #: 死了就不能宣称"生产者首发"。空串表示"顺序不可辨"，一律保守处理。
    first_exit: str = ""
    cancelled: bool = False
    log_paths: Tuple[str, ...] = field(default_factory=tuple)
    stderr_tail: str = ""
    vspipe_tail: str = ""
    encoder_tail: str = ""
    detail: str = ""

    #: 插件/滤镜缺失的特征词（**只在生产者侧尾部里找**）
    #:
    #: R21-C01 实测补充：本机（`tools_r4/vs_portable/r72`）未装 GrainFactory3 时，
    #: 真实报错是
    #:   `AttributeError: No attribute with the name grainfactory exists.
    #:    Did you mistype a plugin namespace or forget to install a plugin?`
    #: —— 里面**没有** `grainfactory3` 这个词。旧表只认
    #: `("grainfactory3", "haf.", "havsfunc")`，于是真实的插件缺失
    #: **一次也匹配不上**，只能掉到最后那条隐式成功。
    #: 另外 `haf.` 也匹配不到 `NameError: name 'haf' is not defined`。
    # R22-A06：**收紧**。审核受控例："编码器报磁盘无空间、生产者留下
    # 无关的 AttributeError" 时，R21 的泛化词 `no attribute` / `importerror`
    # 仍会判成插件缺失，从而触发一次不该发生的回退。
    # 现在只认**指向具体插件/模块**的措辞：
    #   - `grainfactory`（本机真实报错的原文，见上）
    #   - `havsfunc` / `haf.` / `name 'haf'`（havsfunc 缺失或未导入）
    #   - `not found in the plugin` / `unknown function`（VS 插件注册错误）
    # 裸的 `no attribute` / `importerror` **不再**算证据 —— 它们太泛。
    _PLUGIN_TOKENS: Tuple[str, ...] = (
        "grainfactory3", "grainfactory", "havsfunc",
        "haf.", "name 'haf'",
        "not found in the plugin", "unknown function")

    def is_plugin_capability_failure(self) -> bool:
        """是否属于"插件能力缺失"，可按配置尝试 AddGrainC 回退。

        审核 R20-C01 要求区分**首发原因**与**连带失败**：

        - 插件缺失 → vspipe 立即失败 → ffmpeg 拿到空输入，通常也非零退出。
          此时 ffmpeg 的 returncode 是**后果**，不能拿它否掉插件判定
          （旧实现要求 ``ffmpeg_rc in (None, 0)``，于是真实的插件缺失
          反而永远走不到回退）。
        - 独立编码/磁盘错误 → ``failed_stage == "ffmpeg"``，
          且生产者侧尾部**没有**插件特征 → 绝不回退。

        Returns:
            bool: 首发原因是插件能力缺失时为 True。
        """
        if self.ok or self.cancelled:
            return False

        # ① 进程根本没起来（vspipe/ffmpeg 可执行文件缺失、脚本路径错）
        #    → **不算插件能力问题**：AddGrainC 回退同样要跑这两个可执行文件，
        #      换插件救不了环境问题，只会把真原因盖住。
        if self.failed_at_start:
            return False

        _prod = (self.vspipe_tail or "").lower()
        _hit = any(k in _prod for k in self._PLUGIN_TOKENS)

        # ② 生产者侧失败 → 看它自己的尾部
        if self.failed_stage == "vspipe":
            return _hit
        # ③ 编码器失败：只有**确认生产者先退出**的级联才算插件问题。
        #    R22-A06：审核指出"一次轮询同时看到两个进程结束时，固定先遍历
        #    vspipe，于是把它记在前面" —— 那是观察顺序，不是因果。
        #    所以 `first_exit` 为空（顺序不明）时一律**不认**插件失败。
        if self.failed_stage in ("ffmpeg", "wait"):
            return _hit and self.first_exit == "vspipe"
        return False

    def is_cascading_encoder_failure(self) -> bool:
        """编码器失败是否只是生产者失败的**连带**后果。

        用于日志归因：报错要指向首发原因，而不是把"ffmpeg rc=1"
        当成根因反复排查。
        """
        if self.ok or self.cancelled:
            return False
        return (self.failed_stage == "ffmpeg"
                and self.vspipe_rc not in (None, 0))


def _reap_pipeline(procs, threads, logger=None, label: str = "stage6") -> None:
    """收尾一组**可能部分启动**的进程。本函数不抛异常。

    顺序很重要。R21-C01 调整了它（审核：'停止子进程前先等待日志线程，
    停止后没有完整再次收束'）：

    1. **先停进程** terminate → 有界 wait → kill 残留 → 再 wait。
       先 join 线程的话，只要还有一个子进程在往管道里写，join 就要
       空等满超时；terminate 之后管道自然 EOF，线程随即退出。
    2. **再 join 排空线程**（此时读到 EOF，正常返回；仍有超时兜底）。
    3. 关 stdout/stderr/stdin 句柄。
    4. 解取消守卫登记 —— 必须先停进程再注销，否则会遗留写入者。

    Raises:
        不抛。收尾阶段抛异常只会掩盖真正的原因。
    """
    live = []
    for p in procs or []:
        try:
            if p is not None and p.poll() is None:
                live.append(p)
        except BaseException:
            continue

    for p in live:
        try:
            p.terminate()
        except BaseException:
            pass
    for p in live:
        try:
            p.wait(timeout=10)
        except BaseException:
            try:
                p.kill()
            except BaseException:
                pass
            try:
                p.wait(timeout=5)
            except BaseException:
                if logger:
                    try:
                        logger.warning(
                            "STAGE6_REAP_INCOMPLETE",
                            "A pipeline process did not exit even after kill",
                            pid=getattr(p, "pid", None), label=label)
                    except BaseException:
                        pass

    # ---- 停完进程再 join：此时管道 EOF，线程不会卡在 read 上 ----
    for t in threads or []:
        try:
            t.join(timeout=10)
        except BaseException:
            pass

    for p in procs or []:
        if p is None:
            continue
        for attr in ("stdout", "stderr", "stdin"):
            h = getattr(p, attr, None)
            if h is None:
                continue
            try:
                if not h.closed:
                    h.close()
            except BaseException:
                pass

    if _cancel_guard is not None:
        for p in procs or []:
            if p is None:
                continue
            try:
                _cancel_guard.unwatch_process(p)
            except BaseException:
                pass


def _run_grain_pipeline(vs_script_str: str, vs_path: str,
                        vspipe_cmd: list, ffmpeg_cmd: list,
                        output_path: str, logger,
                        fallback_label: str = "",
                        inactivity_s: float = 1800.0) -> tuple:
    """Run grain synthesis pipeline (vspipe → ffmpeg FFV1).

    Module-level function for multiprocessing spawn compliance (CLAUDE.md §5).

    Returns:
        ``GrainPipelineResult``。**不是** ``(status, detail)`` 元组 ——
        R19-B04 起改为结构化对象，R21-C01 补上两侧分离的尾部。
        ``result.ok`` 为真才算成功；失败时 ``failed_stage`` 指出首发位置。

    R19 重写要点（审核 R18-A07）：

    1. **每创建一个进程就立刻放进 `procs`** —— 第二个 Popen 失败时，
       第一个也一定会被收尾（旧实现直接 return，vspipe 变孤儿，
       继续把半截数据写回 output_path）。
    2. **统一 finally** 覆盖"启动 / 等待 / 取消 / 异常"四条路径。
    3. **检查每一个进程的状态**，不只看 ffmpeg
       （旧实现：生产者 rc=9 + 编码器 rc=0 → 返回 `(0, '')` 假成功）。
    4. 失败时**清掉半成品** —— 下游不该看到半截 mkv。
    5. stderr 落盘 + 内存只留有界尾部（:class:`_TailLog`）。
    """
    with open(vs_path, "w", encoding="utf-8") as f:
        f.write(vs_script_str)

    procs: list = []            # 已创建的**全部**进程（含部分启动）
    threads: list = []
    vspipe_proc = None
    ffmpeg_proc = None
    status = None
    detail = ""
    #: 失败发生在哪一类（start/vspipe/ffmpeg/exception/cancel）。
    #: 取消与异常要分开：取消**不能**被当成插件失败去回退。
    _fail_stage = ""
    _is_cancel = False
    #: 是否"进程根本没起来"（Popen 抛异常）。与 `_fail_stage` 分开记。
    _at_start_failure = False
    #: 当前正在启动/等待哪一步。失败时写进 detail，
    #: 让调用方（和测试）能区分"第一步就起不来"与"第二步失败但第一步要回收"。
    _step = "vspipe"

    _base = output_path or "stage6"
    enc_log = _TailLog(_base + ".enc.log")
    vs_log = _TailLog(_base + ".vs.log")

    #: 谁先退出（"vspipe" / "ffmpeg"）。两边都非零时用它归因首发原因。
    _exit_order: list = []
    import threading as _threading_exit
    _exit_mon_stop = _threading_exit.Event()
    _exit_mon = None

    try:
        # ---- ① 第一个进程：创建后**立刻**纳入管理 ----
        vspipe_proc = subprocess.Popen(
            vspipe_cmd, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, shell=False)
        procs.append(vspipe_proc)

        # ---- ② 第二个进程：失败时 vspipe 已在 procs 里，finally 会回收 ----
        _step = "ffmpeg"
        ffmpeg_proc = subprocess.Popen(
            ffmpeg_cmd, stdin=vspipe_proc.stdout,
            stderr=subprocess.PIPE, shell=False)
        procs.append(ffmpeg_proc)
        vspipe_proc.stdout.close()
        _step = "wait"

        # NEW-V-06/RV-12：两个子进程都必须登记——取消时只杀 Python stage
        # 的话，vspipe/ffmpeg 会继续把半截数据写回 output_path（守卫刚删掉
        # 的文件被孤儿进程重新创建，取消就白做了）。
        if _cancel_guard is not None:
            _cancel_guard.watch_process(vspipe_proc)
            _cancel_guard.watch_process(ffmpeg_proc)

        # ---- ③ 并发排空（同时落盘，内存有界）----
        import threading

        def _pump(pipe, sink):
            if pipe is None:
                return
            try:
                while True:
                    chunk = pipe.read(65536)
                    if not chunk:
                        break
                    sink.write(chunk)
            except BaseException:
                pass

        for _pipe, _sink in ((getattr(ffmpeg_proc, "stderr", None), enc_log),
                             (getattr(vspipe_proc, "stderr", None), vs_log)):
            _t = threading.Thread(target=_pump, args=(_pipe, _sink),
                                 daemon=True)
            _t.start()
            threads.append(_t)

        # ---- ③b 退出顺序监视（R21-C01）----
        # 两边都非零时，"谁先退出"是唯一可靠的归因证据。
        # 20ms 轮询、两个进程都退出即停；由 finally 里的 stop/join 收束。
        def _watch_exits():
            while not _exit_mon_stop.is_set():
                _found = []
                for _p, _tag in ((vspipe_proc, "vspipe"),
                                 (ffmpeg_proc, "ffmpeg")):
                    try:
                        if _p is not None and _p.poll() is not None \
                                and _tag not in _exit_order \
                                and _tag not in _found:
                            _found.append(_tag)
                    except BaseException:
                        pass
                if _found:
                    # R22-A06：**同一轮**里看到两个都退出 = 顺序不可辨。
                    # 旧实现固定先遍历 vspipe，于是把它记成"首发" ——
                    # 那是遍历顺序，不是因果。这里显式记成空串。
                    _exit_order.append(_found[0] if len(_found) == 1 else "")
                if len([x for x in _exit_order if x != ""]) >= 2 or \
                        _exit_order.count("") >= 1 and len(_exit_order) >= 2:
                    return
                _exit_mon_stop.wait(0.02)

        _exit_mon = threading.Thread(target=_watch_exits, daemon=True)
        _exit_mon.start()
        threads.append(_exit_mon)

        # ---- ④ 按心跳等（不设总时长）----
        _hb = (lambda: (os.path.getsize(output_path)
                        if output_path and os.path.exists(output_path)
                        else 0))
        if _cancel_guard is not None and hasattr(_cancel_guard,
                                                 "wait_with_heartbeat"):
            _cancel_guard.wait_with_heartbeat(
                [ffmpeg_proc, vspipe_proc], heartbeat=_hb,
                inactivity_timeout=inactivity_s,
                logger=logger, label="stage6-grain")
        else:  # pragma: no cover - 守卫缺失时退回有界等待
            ffmpeg_proc.wait(timeout=inactivity_s)
            vspipe_proc.wait(timeout=inactivity_s)
    except BaseException as exc:
        # 不在这里 return：交给 finally 统一收尾，再在下面判状态
        status = 84
        _exc_name = type(exc).__name__
        _is_cancel = (isinstance(exc, (KeyboardInterrupt, SystemExit))
                      or _exc_name in ("CancelledError", "Cancelled",
                                       "StageCancelled"))
        # ---- R21-C01：Popen 抛异常 = **启动失败**，与"该阶段跑失败"分开 ----
        # `failed_stage` 仍记为 `_step`（诊断需要知道是第几步），
        # 另用 `failed_at_start` 表示"进程根本没跑起来"。
        # 旧实现两者都没有，于是"第二个进程命令不存在"会被当成编码器跑失败。
        _at_start = _step in ("vspipe", "ffmpeg")
        _at_start_failure = _at_start
        _fail_stage = _step or "exception"
        _where = f"{_step or 'unknown'}{' (start)' if _at_start else ''}"
        detail = (f"grain pipeline failed at {_where}"
                  f"{' (cancelled)' if _is_cancel else ''}: "
                  f"{_exc_name}: {exc}")
        if logger:
            try:
                logger.error("STAGE6_PIPELINE_ERROR", detail,
                             stage6_error_type=type(exc).__name__)
            except BaseException:
                pass
    finally:
        # 先让监视线程停手，再收尾进程：否则它会把"我们自己 terminate 掉的"
        # 也记成退出顺序，污染首发归因。
        try:
            _exit_mon_stop.set()
            if _exit_mon is not None:
                _exit_mon.join(timeout=2)
        except BaseException:
            pass
        _reap_pipeline(procs, threads, logger=logger, label="stage6-grain")
        enc_log.close()
        vs_log.close()

    # ---- ⑤ 判定：检查**每一个**参与进程 ----
    vs_rc = None if vspipe_proc is None else vspipe_proc.returncode
    enc_rc = None if ffmpeg_proc is None else ffmpeg_proc.returncode
    _logs = tuple(x for x in (getattr(enc_log, "path", None),
                             getattr(vs_log, "path", None)) if x)
    # R21-C01：两侧尾部**分开**保存。合并后无法回答"这句报错是谁说的"，
    # 而"首发原因"正是插件回退与编码失败的唯一区分依据。
    _vs_tail = vs_log.tail_text()[-800:]
    _enc_tail = enc_log.tail_text()[-800:]
    _tail = (_vs_tail + "\n" + _enc_tail).strip()

    stage_name = ""
    if status is None:
        if vspipe_proc is None or ffmpeg_proc is None:
            status, stage_name = 84, "start"
            detail = ("grain pipeline did not start: "
                      f"vspipe={'ok' if vspipe_proc else 'missing'}, "
                      f"ffmpeg={'ok' if ffmpeg_proc else 'missing'}")
        elif vs_rc != 0 and enc_rc not in (None, 0):
            # ---- R21-C01：两边都非零时按**实测退出顺序**归因 ----
            # 消费者先死 → 生产者 stdout 变断管，通常跟着非零退出。
            # 只看 vspipe 的 rc 会把"编码器先失败"错记成"插件失败"，
            # 进而误触发 AddGrain 回退（等于用换插件掩盖磁盘/编码错误）。
            # 这里用监视线程记下的**谁先退出**判定，不依赖关键词表。
            _first = _exit_order[0] if _exit_order else None
            if _first == "ffmpeg":
                status, stage_name = 84, "ffmpeg"
                detail = (f"both processes failed; the encoder exited first "
                          f"(vspipe rc={vs_rc}, ffmpeg rc={enc_rc}): "
                          f"{enc_log.tail_text()[-800:]}")
            elif _first == "vspipe":
                status, stage_name = 84, "vspipe"
                detail = (f"both processes failed; the producer exited first "
                          f"(vspipe rc={vs_rc}, ffmpeg rc={enc_rc}): "
                          f"{vs_log.tail_text()[-800:]}")
            elif (vs_log.tail_text() or "").strip():
                # 同一轮轮询里双双退出、顺序不可辨 → 退回"谁留下了报错"
                status, stage_name = 84, "vspipe"
                detail = (f"both processes failed; exit order indistinguishable, "
                          f"the producer left a message "
                          f"(vspipe rc={vs_rc}, ffmpeg rc={enc_rc}): "
                          f"{vs_log.tail_text()[-800:]}")
            else:
                status, stage_name = 84, "ffmpeg"
                detail = (f"both processes failed; exit order indistinguishable "
                          f"and the producer stayed silent "
                          f"(vspipe rc={vs_rc}, ffmpeg rc={enc_rc}): "
                          f"{enc_log.tail_text()[-800:]}")
        elif vs_rc != 0:
            status, stage_name = 84, "vspipe"
            detail = (f"producer (vspipe) failed rc={vs_rc}: "
                      f"{vs_log.tail_text()[-800:]}")
        elif enc_rc != 0:
            status, stage_name = 84, "ffmpeg"
            detail = (f"encoder (ffmpeg) failed rc={enc_rc}: "
                      f"{enc_log.tail_text()[-800:]}")
        else:
            status, stage_name = 0, ""
            detail = ""
    else:
        # 异常分支已设过 status=84 与 _fail_stage（保留"哪一步"）
        stage_name = _fail_stage or "exception"

    # ---- ⑥ 失败时清掉半成品 ----
    if status != 0 and output_path:
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except OSError:
            pass

    if status == 0:
        if logger:
            try:
                if fallback_label:
                    logger.info("STAGE6_DONE",
                                f"Grain synthesis complete ({fallback_label})",
                                output=output_path)
                else:
                    logger.info("STAGE6_DONE", "Grain synthesis complete",
                                output=output_path)
            except BaseException:
                pass
        return GrainPipelineResult(ok=True, vspipe_rc=vs_rc,
                                   ffmpeg_rc=enc_rc, log_paths=_logs,
                                   stderr_tail=_tail)

    return GrainPipelineResult(
        ok=False, vspipe_rc=vs_rc, ffmpeg_rc=enc_rc,
        failed_stage=stage_name,
        failed_at_start=bool(_at_start_failure),
        # 空串 = 同轮同时退出/顺序不可辨 → 调用方不得宣称"生产者首发"
        first_exit=str((_exit_order[0] if _exit_order else "") or ""),
        cancelled=_is_cancel,
        log_paths=_logs, stderr_tail=_tail,
        vspipe_tail=_vs_tail, encoder_tail=_enc_tail, detail=detail)


def get_video_dimensions(input_source: str) -> tuple:
    """Get video width, height, fps using ffprobe."""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate",
             "-of", "csv=p=0", input_source],
            capture_output=True, text=True, errors="replace", timeout=30, shell=False
        )
        if r.returncode == 0:
            parts = r.stdout.strip().split(",")
            if len(parts) >= 3:
                width = int(parts[0])
                height = int(parts[1])
                fps_parts = parts[2].split("/")
                fps = (float(fps_parts[0]) / float(fps_parts[1])
                       if len(fps_parts) == 2 else float(parts[2]))
                return width, height, fps
    except Exception:
        pass
    return 720, 480, 29.97


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--job-id", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--params", default="")
    # FIX-R6-07: main.py 统一用 `--params-file`（避免超长命令行/转义问题），
    # 但本 stage 原来只认 `--params` → argparse "unrecognized arguments"
    # 直接退 2。**批处理下 stage6 从未被执行过**（实测 2026-10-18）。
    parser.add_argument("--params-file", default="")
    # FIX-R6-07: main.py 统一给**每个** stage 传 `--cleanup-intermediates`
    # （params.cleanup_intermediates 默认 True）。stage6 原来不认它 →
    # argparse "unrecognized arguments" → sys.exit(2)，2 不在项目退出码表
    # 内。**批处理下 stage6 从未被执行过**（实测 2026-10-18）。
    # 本 stage 不自行删除上游产物（清理由 main.py 统一收口），故只接受不动作。
    parser.add_argument("--cleanup-intermediates", action="store_true")
    args = parser.parse_args()

    params = {}
    if getattr(args, "params_file", None):
        import json
        try:
            with open(args.params_file, "r", encoding="utf-8") as f:
                params = json.load(f)
        except Exception:
            params = {}
    elif args.params:
        import json
        try:
            params = json.loads(args.params)
        except Exception:
            pass

    # Use get_logger with proper parameters from config
    log_dir = params.get("log_dir", "./logs")
    logger = get_logger(log_dir=log_dir, job_id=args.job_id)

    # NEW-V-06: 注册取消守卫（video main.py 认 rc=88 = EXIT_INTERRUPTED）。
    if _cancel_guard is not None:
        _cancel_guard.install_cancel_guard(stage=6, logger=logger)

    # FIX-P9-3: WSL2 temp_dir redirect per v7.32 §3.10.10 / v6.32 §7.3
    temp_dir_raw = params.get("temp_dir", "./temp")
    no_wsl_redirect = params.get("no_wsl_redirect", False)
    _, _, temp_dir, _ = wsl_redirect_path(
        output_path="", temp_dir=temp_dir_raw,
        no_wsl_redirect=no_wsl_redirect, logger=logger)
    job_id = args.job_id
    # Use args.input stem for pattern matching (consistent with other stages)
    input_name = Path(args.input).stem

    # FIX-W9-09: log ai_upscale_banned flag per checkproject rule F. Grain
    # synthesis is not AI so the flag doesn't change grain behavior; logged
    # for traceability. When banned, F9-02 fallback ensures stage4 input.
    ai_banned = params.get("ai_upscale_banned", False)
    if ai_banned:
        logger.info("STAGE6_BAN_FLAG",
                    "ai_upscale_banned=true noted (grain synthesis unaffected; "
                    "input sourced from stage4 per F9-02 fallback)")

    # Determine if face restore was enabled
    # Fix: ai_enabled is passed as flat key "ai_enabled" in params, not as nested "ai" dict
    face_restore_enabled = bool(params.get("face_restore", ""))
    ai_enabled = params.get("ai_enabled", False)

    # Input source STRICT validation per v6.3 §3.10.24:
    # - face_restore enabled -> MUST receive stage5 output
    # - face_restore disabled -> MUST receive stage4 output
    # - STRICTLY FORBIDDEN to receive stage3 output directly
    input_source = None
    expected_stage = None

    if ai_enabled and face_restore_enabled:
        # FIX-F9-02: face_restore enabled but stage5 may have produced no
        # output (stage5 exits 3 when no static face segments — a common
        # case). Per v7.32 §3.10.24, fall back to stage4 output instead of
        # aborting the whole pipeline with return 82.
        expected_stage = "step5_face"
        # ID-04（第七轮）：精确产物优先（同一约定，贯穿 3→4→5→6→7）
        _pinned = str(params.get("upstream_artifact") or "")
        if _pinned:
            if not os.path.exists(_pinned):
                logger.error("STAGE6_PINNED_INPUT_MISSING",
                             "Orchestrator pinned an upstream artifact that "
                             "does not exist; refusing to silently glob",
                             pinned=_pinned)
                return 82
            input_source = _pinned
            logger.info("STAGE6_INPUT_PINNED",
                        "Using the exact upstream artifact", input=_pinned)
            cands = None
        else:
            pattern5 = f"{job_id}_{input_name}_step5_face*"
            # R3-01：只接受媒体产物，排除同前缀侧车
            cands = select_media_candidates(glob.glob(os.path.join(temp_dir, pattern5)))
            if cands:
                input_source = str(cands[0])
        if not input_source:
            # Fallback to stage4 output
            pattern4 = f"{job_id}_{input_name}_step4_postprocess*"
            cands4 = select_media_candidates(
                glob.glob(os.path.join(temp_dir, pattern4)))
            if cands4:
                input_source = str(cands4[0])
                expected_stage = "step4_postprocess(fallback)"
                logger.warning("STAGE6_FALLBACK_STEP4",
                               "face_restore enabled but no stage5 output "
                               "(stage5 likely skipped — no static face segments), "
                               "falling back to stage4 output per v7.32 §3.10.24")
            else:
                logger.error("STAGE6_INPUT_ERROR",
                            f"Neither stage5 nor stage4 output found. "
                            f"Patterns: {pattern5} / {pattern4}",
                            temp_dir=temp_dir)
                return 82
    else:
        # Face restore disabled: MUST use stage4 output
        expected_stage = "step4_postprocess"
        pattern = f"{job_id}_{input_name}_step4_postprocess*"
        # R3-01：只接受媒体产物，排除同前缀侧车
        cands = select_media_candidates(glob.glob(os.path.join(temp_dir, pattern)))
        if cands:
            input_source = str(cands[0])
        if not input_source:
            # R4（关键修复）：**AI 关闭时 main 根本不会调度 stage3/stage4**
            # （见 main.py 里 stages 的构造：stage3/4 仅在 ai_enabled 时 append）。
            # 此时传统链的最后产物是 **stage2** 的输出，而这里原来只认
            # stage4、没有回退，于是 AI 关闭跑视频链必然
            # "No stage4 output found" -> return 82，整链断在这里。
            pattern2 = f"{job_id}_{input_name}_step2_traditional*"
            cands2 = select_media_candidates(
                glob.glob(os.path.join(temp_dir, pattern2)))
            if cands2:
                input_source = str(cands2[0])
                expected_stage = "step2_traditional(fallback: AI disabled)"
                logger.warning("STAGE6_FALLBACK_STEP2",
                               "stage4 absent because AI stages are disabled; "
                               "using stage2 output as the traditional-chain "
                               "result instead")
            else:
                logger.error("STAGE6_INPUT_ERROR",
                            f"No stage4 output found (and no stage2 fallback). "
                            f"Patterns: {pattern} / {pattern2}",
                            temp_dir=temp_dir)
                return 82

    if not input_source:
        logger.error("STAGE6_NO_INPUT", "No valid input found",
                     expected_stage=expected_stage)
        return 82

    # FIX-R5-06（关键）: vspipe 会**临时把工作目录切到 vpy 脚本所在目录**
    # —— R55 起的默认行为，官方文档原文（vspipe 的 `-c/--preserve-cwd` 项）:
    #   "Don't temporarily change the working directory to the script
    #    directory (R54 compatibility)"
    # 所以 vpy 里 `core.ffms2.Source(source=...)` 的相对路径是按 **vpy 所在
    # 目录**解析的，而不是进程 cwd。上面 glob 出来的 input_source 是相对
    # temp_dir 的（如 `temp\job_001_clip15_step2_traditional.mkv`），写进
    # `temp\job_001_clip15_grain.vpy` 后被解析成
    # `temp\temp\job_001_clip15_step2_traditional.mkv` ->
    # `vapoursynth.Error: Index: Can't open ...`（实测复现：同一个文件换成
    # 绝对路径立刻能打开，前向/后向斜杠、加不加 `./` 都一样失败）。
    # ffprobe 用绝对路径同样无害，故统一在这里转绝对路径。
    input_source = os.path.abspath(input_source)

    logger.info("STAGE6_INPUT", f"Using {expected_stage} output",
                input=input_source)

    # FIX-P9-11: input file integrity check via ffprobe per v6.32 §7.9
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height",
             "-of", "json", input_source],
            capture_output=True, text=True, errors="replace", timeout=30, shell=False)
        if r.returncode != 0:
            logger.error("STAGE6_INPUT_CORRUPT",
                        f"Input file is not a valid video: {input_source}",
                        ffprobe_err=r.stderr[:300] if r.stderr else "")
            return 86
        probe_info = json.loads(r.stdout)
        streams = probe_info.get("streams", [])
        if not streams or "width" not in streams[0]:
            logger.error("STAGE6_INPUT_CORRUPT",
                        f"Input file has no valid video stream: {input_source}")
            return 86
    except subprocess.TimeoutExpired:
        logger.error("STAGE6_INPUT_CORRUPT",
                    f"ffprobe timed out on input: {input_source}")
        return 86
    except Exception as e:
        logger.error("STAGE6_INPUT_CORRUPT",
                    f"Cannot probe input file: {e}")
        return 86

    # FIX-P9-4: output filename must include param_hash per v7.3 §3.10.19
    # FIX-2026-10-20: merge_params() 从不注入 params["param_hash"]，原来退化成
    # job_id（与 stage5 同族）—— 改 g1str/g2str/g3str 等颗粒参数不改变文件名，
    # 陈旧产物会被复用。改为就地计算，以本 stage 的实际输入 input_source
    # 为身份基准（注意 input_source 可能来自回退链，与 args.input 不同）。
    from checkpoint import compute_param_hash
    param_hash = compute_param_hash(input_source, params)
    # FIX-R5-06: 同 input_source，输出路径也转绝对路径。ffmpeg 按进程 cwd
    # 解析相对路径本来是对的，但转绝对后可彻底摆脱 cwd 依赖（vspipe 的
    # chdir 不影响 ffmpeg，这里是防御性的）。
    output_path = os.path.abspath(
        os.path.join(
            temp_dir,
            f"{job_id}_{input_name}_step6_grain_p{param_hash}.mkv"
        )
    )

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        return 3

    # NEW-V-06: 写任何字节之前登记产物——取消时删掉半截 mkv，
    # 否则上面的 exists()>0 缓存检查会把它当成功产物复用。
    if _cancel_guard is not None:
        _cancel_guard.watch_outputs(output_path, logger=logger, stage=6)

    # Load grain parameters from config
    config_path = args.config
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)
    except Exception as e:
        logger.error("STAGE6_CONFIG", f"Cannot load config: {e}")
        return 81

    # FIX-NEW-V-09 / R12: 合并 params 侧的 grain_factory3 覆盖。
    # 原实现只读 cfg["grain_factory3"]（config.yaml），于是
    # user_overrides / manifest 里的 grain_factory3 **完全无效**——
    # 实测三型矩阵（dv g1str=0.5 / hdv 0.8 / temp_avg=1）全部落回
    # analog_tape 默认 1.0，且日志不报警（rc 仍为 0）。
    # 优先级：params（覆盖）> cfg（配置），与项目其余参数一致。
    _cfg_grain = dict(cfg.get("grain_factory3", {}) or {})
    _params_grain = params.get("grain_factory3")
    if isinstance(_params_grain, dict):
        _cfg_grain.update(_params_grain)
        logger.info("STAGE6_GRAIN_OVERRIDE",
                    "Applied grain_factory3 overrides from params",
                    keys=sorted(_params_grain.keys()))
    grain_cfg = _cfg_grain
    if not grain_cfg.get("enabled", True):
        logger.info("STAGE6_SKIP", "Grain synthesis disabled")
        return 0

    # FIX-P9-6: Read addgrainc_enabled and addgrainc_ratio from config
    # Per v7.3 §3.10.24: AddGrainC default disabled, intensity ≤20% of g1str
    addgrainc_enabled = grain_cfg.get("addgrainc_enabled", False)
    addgrainc_ratio = grain_cfg.get("addgrainc_ratio", 0.2)

    # FIX-P9-5: explicit detected_class → grain config key mapping per v7.32 §3.2
    # detected_class values may be uppercase (ANALOG_TAPE, DV_DIGITAL, HDV_AVCHD)
    _DETECTED_CLASS_TO_GRAIN_KEY = {
        "analog_tape": "analog_tape",
        "dv_digital": "dv_digital",
        "hdv_avchd": "hdv_avchd",
    }
    material_type_raw = params.get("detected_class", "analog_tape").lower()
    material_key = _DETECTED_CLASS_TO_GRAIN_KEY.get(material_type_raw, "analog_tape")
    material_grain = dict(grain_cfg.get(material_key, {}) or {})
    if material_key != material_type_raw:
        logger.warning("STAGE6_GRAIN_MAPPING",
                      f"Unknown material type '{material_type_raw}', "
                      f"defaulting to '{material_key}'")

    # FIX-NEW-V-09 / R12: **扁平**覆盖（直接给 g1str/g2str/g3str/temp_avg，
    # 不带分型子键）必须**盖过** config 的分型值，否则"params 里有值、
    # vpy 里还是默认值"。
    # 优先级：params 扁平覆盖 > cfg 分型子键 > cfg 扁平键 > 内置默认。
    # （原实现写成 "扁平键只在分型子键缺失时才补"，而 config 的分型
    # 子键永远存在，于是扁平覆盖 100% 失效——实测 dv 0.5/hdv 0.8 全落回 1.0。）
    _FLAT_GRAIN_KEYS = ("g1str", "g2str", "g3str", "temp_avg")
    _flat_override = {
        k: params["grain_factory3"][k]
        for k in _FLAT_GRAIN_KEYS
        if isinstance(params.get("grain_factory3"), dict)
        and k in params["grain_factory3"]
    }
    for _k in _FLAT_GRAIN_KEYS:
        if _k not in _flat_override and _k in grain_cfg:
            _flat_override[_k] = grain_cfg[_k]
    material_grain.update(_flat_override)

    g1str = material_grain.get("g1str", 1.0)
    g2str = material_grain.get("g2str", 0.2)
    g3str = material_grain.get("g3str", 0.2)
    # FIX-P1-13: pass temp_avg to GrainFactory3 (r33 legal param, controls
    # temporal correlation; default 0 = disabled, AddGrainC fallback used then).
    temp_avg = material_grain.get("temp_avg", grain_cfg.get("temp_avg", 0.0))

    # Validate no illegal params
    if not grain_cfg:
        logger.error("STAGE6_CONFIG", "grain_factory3 config missing")
        return 81

    logger.info("STAGE6_START", "Grain synthesis", input=input_source,
                g1str=g1str, g2str=g2str, g3str=g3str,
                temp_avg=temp_avg, material_key=material_key)

    # Get video dimensions for vspipe
    width, height, fps = get_video_dimensions(input_source)

    # V-P-046 fix: 16bit processing space
    # Generate VapourSynth script
    # v6.3: Only g1str/g2str/g3str are valid GrainFactory3 params.
    vs_bit_depth = params.get("vs_bit_depth", 16)
    if vs_bit_depth == 16:
        vs_format = "vs.YUV420P16"
        pix_fmt = "yuv420p16le"
    elif vs_bit_depth == 10:
        vs_format = "vs.YUV420P10"
        pix_fmt = "yuv420p10le"
    else:
        vs_format = "vs.YUV420P8"
        pix_fmt = "yuv420p"
    # R22 §5.2：色度跟工作表示走。
    # **这里必须同时改 vs_format**：vspipe 的输出格式由脚本决定，
    # 只改 ffmpeg 侧 `-pix_fmt` 会变成"把 420 转成 444 再落盘"——
    # 标签变了、细节没回来，正是审核禁止的"静默转换后仍称 444"。
    try:
        from work_format import (boundary_pix_fmt as _bpf,
                                 set_bit_depth as _sbd,
                                 vs_format_of as _vsf)
        _g_fmt = _sbd(
            _bpf(params, "stage6_out", default="yuv420p10le"),
            vs_bit_depth if vs_bit_depth in (8, 10, 16) else 16)
        _g_vs = _vsf(_g_fmt)
        if _g_vs:
            pix_fmt = _g_fmt
            vs_format = _g_vs
    except Exception:                                  # pragma: no cover
        pass

    # FIX-R5-06: vpy 路径同样转绝对路径。vspipe 找到脚本后会 chdir 到脚本
    # 所在目录，路径确定后这个 chdir 目标才可控（脚本里已无相对路径依赖）。
    vs_path = os.path.abspath(
        os.path.join(temp_dir, f"{job_id}_{input_name}_grain.vpy"))
    # FIX-P9-4: pass addgrainc_after_gf3 so AddGrainC can stack after GrainFactory3
    vs_script = _build_vs_script(
        input_source, vs_format, g1str, g2str, g3str,
        addgrainc_ratio, use_addgrainc=False,
        addgrainc_after_gf3=addgrainc_enabled,
        temp_avg=temp_avg,
        deterministic=_vs_deterministic(params),
        # R22：`grain_seed` 给了才固定随机种子（不给 = 历史行为，每次随机）
        seed=params.get("grain_seed"))
    # FIX-P9-11: removed redundant write here — _run_grain_pipeline writes vs_path internally

    # Fix: Use vspipe directly (standard VapourSynth CLI tool)
    vspipe_cmd = [
        "vspipe", "-c", "y4m",
        vs_path, "-"
    ]

    # V-P-048 fix: lossless FFV1 intermediate instead of lossy SVT-AV1
    ffmpeg_cmd = [
        "ffmpeg", "-hide_banner", "-v", "error", "-y",
        "-i", "-",
        # R15-04：y4m 管道本身是定帧率声明，但上游若是 VFR 素材，
        # 这里再加一道 passthrough，保证不因 vsync 重整而丢帧。
        "-fps_mode", "passthrough",
        "-c:v", "ffv1",
        "-pix_fmt", pix_fmt,
        output_path
    ]

    result = _run_grain_pipeline(
        vs_script, vs_path, vspipe_cmd, ffmpeg_cmd,
        output_path, logger,
        # 卡死阈值与阶段级看门狗同源：只看"产出是否停止增长"，不设总时长。
        inactivity_s=float(params.get("stage_inactivity_timeout", 1800.0)))

    # ---- R21-C01：结果分派抽成**可测函数** ----
    # 原实现把 5 条返回路径散在 main() 尾部，其中两条直接走到函数末尾
    # （隐式 None → sys.exit(None) = 0 = "成功"）。抽出来后既能逐条单测，
    # 也不会因为 `if/elif` 少一个分支就静默变成成功。
    def _run_fallback() -> "GrainPipelineResult":
        addgrain_script = _build_vs_script(
            input_source, vs_format,
            g1str * 0.5, g2str * 0.5, g3str * 0.5,
            addgrainc_ratio, use_addgrainc=True,
            temp_avg=temp_avg,
            deterministic=_vs_deterministic(params),
            seed=params.get("grain_seed"))
        return _run_grain_pipeline(
            addgrain_script, vs_path, vspipe_cmd, ffmpeg_cmd,
            output_path, logger, fallback_label="AddGrainC",
            inactivity_s=float(params.get("stage_inactivity_timeout", 1800.0)))

    rc, _degraded = dispatch_result(
        result, addgrainc_enabled=addgrainc_enabled, logger=logger,
        run_fallback=_run_fallback)
    if rc == 0:
        # NEW-V-06: 产物已完整落盘，取消守卫不再有权删它（RV-12）。
        if _cancel_guard is not None:
            _cancel_guard.release_outputs()
        if _degraded is not None:
            _write_degraded_note(output_path, result, _degraded,
                                 g1str, g2str, g3str, addgrainc_ratio, logger)
    return rc


def dispatch_result(result: "GrainPipelineResult", addgrainc_enabled: bool,
                    logger, run_fallback=None) -> Tuple[int, Optional["GrainPipelineResult"]]:
    """把一次 stage6 管道结果分派成**退出码**（R21-C01 抽成可测函数）。

    覆盖全部返回路径，**不允许隐式成功**：

    | 输入 | 退出码 | 副作用 |
    |---|---|---|
    | ``result.ok`` | 0 | — |
    | ``result.cancelled`` | 88 | **绝不**回退 |
    | 插件能力失败 + 回退禁用 | 84 | 记录 STAGE6_GF3_FAILED |
    | 插件能力失败 + 允许回退，回退成功 | 0 | 返回降级结果 |
    | 插件能力失败 + 允许回退，回退失败 | 84 | 记录两次失败 |
    | 其他一切失败（编码器/磁盘/异常） | 84 | 记录 STAGE6_FAIL |

    Args:
        result: 首次（GrainFactory3）管道结果。
        addgrainc_enabled: 配置是否允许 AddGrainC 独立回退。
        logger: 日志器。
        run_fallback: 执行回退的可调用对象，返回新的 ``GrainPipelineResult``。
            为 None 且需要回退时按"回退失败"处理（不会静默成功）。

    Returns:
        ``(exit_code, degraded_result_or_None)``。第二项非 None 表示
        **这一次是降级产物**，调用方必须落侧车记录。
    """
    if result.ok:
        return 0, None

    if result.cancelled:
        if logger:
            logger.error("STAGE6_CANCELLED",
                         "Grain synthesis was cancelled; no plugin fallback "
                         "attempted",
                         failed_stage=result.failed_stage,
                         vspipe_rc=result.vspipe_rc,
                         ffmpeg_rc=result.ffmpeg_rc)
        return 88, None

    if result.is_plugin_capability_failure():
        if not addgrainc_enabled:
            if logger:
                logger.critical(
                    "STAGE6_GF3_FAILED",
                    "GrainFactory3 failed. AddGrainC is disabled by config. "
                    "GrainFactory3 produces multi-scale structured grain; "
                    "AddGrainC fallback would produce lower quality grain.",
                    failed_stage=result.failed_stage,
                    detail=(result.detail or "")[:500])
            return 84, None

        if logger:
            logger.warning(
                "STAGE6_FALLBACK",
                "GrainFactory3 is unavailable; running the AddGrainC fallback "
                "at halved intensity (lower quality, declared)",
                first_stage=result.failed_stage,
                vspipe_rc=result.vspipe_rc,
                ffmpeg_rc=result.ffmpeg_rc,
                vspipe_tail=(result.vspipe_tail or "")[-400:])
        if run_fallback is None:            # 调用方没给回退实现 → 不得算成功
            if logger:
                logger.error("STAGE6_FALLBACK_UNAVAILABLE",
                             "AddGrainC fallback was required but no fallback "
                             "runner was supplied; failing instead of "
                             "pretending success")
            return 84, None

        second = run_fallback()
        # R22-A06：回退期间被取消 → 必须保持**取消语义**（88）。
        # 审核实测：首次插件失败后回退结果 cancelled=True，dispatch 却返回 84。
        # 取消不是"失败"，上层要靠 88 把半成品清掉并停下，
        # 走 84 会被当成"可以继续/重试"的普通失败。
        if getattr(second, "cancelled", False):
            if logger:
                logger.error(
                    "STAGE6_CANCELLED_DURING_FALLBACK",
                    "The AddGrainC fallback was cancelled; reporting the "
                    "cancel exit code rather than a generic failure",
                    first_stage=result.failed_stage)
            return 88, None
        if getattr(second, "ok", False):
            return 0, second
        if logger:
            logger.error(
                "STAGE6_FAIL",
                "Grain synthesis failed and the AddGrainC fallback also failed",
                first_stage=result.failed_stage,
                first_vspipe_rc=result.vspipe_rc,
                fallback_stage=getattr(second, "failed_stage", ""),
                fallback_vspipe_rc=getattr(second, "vspipe_rc", None),
                fallback_ffmpeg_rc=getattr(second, "ffmpeg_rc", None),
                fallback_detail=(getattr(second, "detail", "") or "")[:500])
        return 84, None

    # ---- 其余失败必须有非零返回 ----
    # 旧实现走到这里就"函数结束"→ 隐式返回 None → `sys.exit(None)` = 0，
    # 也就是**普通编码失败被当成成功**。这是审核 R20-C01 的第二条。
    if logger:
        logger.error("STAGE6_FAIL",
                     "Grain synthesis failed (no plugin fallback applies)",
                     failed_stage=result.failed_stage,
                     vspipe_rc=result.vspipe_rc,
                     ffmpeg_rc=result.ffmpeg_rc,
                     cascading_from_producer=result.is_cascading_encoder_failure(),
                     detail=(result.detail or "")[:800])
    return 84, None


def _write_degraded_note(output_path: str, first: "GrainPipelineResult",
                         second: "GrainPipelineResult",
                         g1str: float, g2str: float, g3str: float,
                         addgrainc_ratio: float, logger) -> None:
    """把"这一代是降级产物"落成侧车文件（R21-C01）。

    审核要求：回退成功必须验证实际输出，**并记录降级**。
    只写日志不够 —— 日志会轮转，而读报告的人需要知道"这份 step6 用的不是
    GrainFactory3"。侧车写失败只告警，不改变阶段结论（产物本身没坏）。

    Args:
        output_path: 本次产物路径（侧车与它同名 + `.degraded.json`）。
        first: 首次（GrainFactory3）失败结果，用于记录首发原因。
        second: 回退（AddGrainC）成功结果。
        g1str / g2str / g3str: 配置的原始颗粒强度（回退按 0.5 施加）。
        addgrainc_ratio: 配置的 AddGrainC 比例。
    """
    try:
        import json as _json
        _note = {
            "schema": "stage6-degraded/1",
            "output": os.path.abspath(output_path),
            "output_size": (os.path.getsize(output_path)
                            if os.path.exists(output_path) else 0),
            # R22-A06：侧车必须绑定**内容**，否则"这份降级说明属于哪一份
            # 产物"只能靠文件名猜（同大小替换 / 重跑覆盖都会错配）。
            "output_sha256": _sha256_file(output_path) or "",
            "degraded": True,
            "plugin_used": "AddGrainC",
            "plugin_missing": "GrainFactory3",
            "intensity": {
                "configured": {"g1str": g1str, "g2str": g2str,
                               "g3str": g3str},
                "applied_scale": 0.5,
                "addgrainc_ratio": addgrainc_ratio,
                "note": "standalone fallback uses halved intensity "
                        "(v6.32 §7.17 / FIX-P9-9)",
            },
            "first_failure": {
                "stage": first.failed_stage,
                "vspipe_rc": first.vspipe_rc,
                "ffmpeg_rc": first.ffmpeg_rc,
                "detail": (first.detail or "")[:800],
            },
            "fallback": {
                "stage": second.failed_stage,
                "vspipe_rc": second.vspipe_rc,
                "ffmpeg_rc": second.ffmpeg_rc,
            },
            "recorded_at": __import__("datetime").datetime.now()
                                 .astimezone().isoformat(),
        }
        _tmp = output_path + ".degraded.json.tmp"
        with open(_tmp, "w", encoding="utf-8") as f:
            _json.dump(_note, f, ensure_ascii=False, indent=2)
        os.replace(_tmp, output_path + ".degraded.json")
    except Exception as exc:            # 诊断失败不改变阶段结论
        try:
            logger.warning("STAGE6_DEGRADED_NOTE_FAIL",
                           "Could not write the degraded-output sidecar; the "
                           "degradation is only in the log", error=str(exc))
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
