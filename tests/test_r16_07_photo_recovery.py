# -*- coding: utf-8 -*-
"""R16-07：照片恢复与取消必须诚实、可再恢复。

审核确认三处：

1. `stage6:822` 回滚仍 `copyfile(backup, formal_path)` —— 复制中断会把正式
   文件留成半文件，下次既不等于旧摘要也不等于新摘要，只能人工。
2. `stage6:440` 恢复入口看到 `journal.phase=COMMITTED` 就**先删** recovery
   记录，之后才校验路径与整代摘要，也没核对 attempt。
3. `stage6:1950` 取消 lambda 忽略回滚返回值，`CancelGuard` 还吞 hook 异常。

本文件用源码守卫锁住这三条（照片 stage6 的执行路径需要完整编码上下文，
端到端负例成本高，见台账残留）。
"""
import os
import re
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_S6 = os.path.join(_ROOT, "photo_restoration_pipeline", "stages",
                   "stage6_encode.py")


def _src():
    return open(_S6, encoding="utf-8", errors="replace").read()


def test_rollback_restores_via_private_staging():
    """回滚必须"目标侧私有暂存 → 核对摘要 → replace"，不能直写正式路径。"""
    src = _src()

    assert ".restore-{os.getpid()}" in src, \
        "回滚必须有 attempt/进程私有的暂存路径"
    assert "staged restore refused" in src, \
        "暂存内容与 old_digest 不符时必须拒绝"
    assert "os.replace(_stage, path)" in src, \
        "核对通过后要用 replace 原子落位"

    # 反例：不允许再直接 copyfile 到正式路径
    bad = re.search(r"shutil\.copyfile\(backup,\s*path\)", src)
    assert bad is None, "仍在直接 copyfile 到正式路径（中断会留半文件）"


def test_recovery_record_cleared_only_after_full_generation_check():
    """解除 recovery 记录前必须核验全组摘要与 attempt。"""
    src = _src()

    assert "_gen_ok" in src, "必须有'整代完整'的判定变量"
    assert "recovery record belongs to another attempt" in src, \
        "必须核对 attempt 归属"
    # 源码里这条消息被拆成两个相邻字面量，只查前缀
    assert "target does not match the recorded" in src, \
        "必须逐个目标核对新代摘要"

    # 结构：`if _gen_ok:` 里才允许 os.remove(recovery record)
    # 结构：删记录必须在 `if _gen_ok:` 之后、其 `else:` 之前
    _gi = src.find("if _gen_ok:")
    _ri = src.find("os.remove(_rec_path_)")
    assert _gi != -1 and _ri != -1, "缺少 _gen_ok 判定或删记录调用"
    assert _gi < _ri, "删记录必须发生在 _gen_ok 判定之后"
    assert "else:" not in src[_gi:_ri], \
        "删记录落在了 `if _gen_ok:` 的 else 分支里（语义反了）"


def test_cancel_hook_reads_and_persists_rollback_result():
    """取消钩子必须读取回滚结果并持久记录。"""
    src = _src()

    assert "_cancel_rollback" in src, "必须是有名字的钩子，便于读取结果"
    assert "STAGE6_CANCEL_ROLLBACK_RESULT" in src, "必须记录回滚结果"
    assert "STAGE6_CANCEL_ROLLBACK_UNRECORDED" in src, \
        "诊断写失败也要如实报出"
    assert 'cancel_rollback"' in src, "结果要写进 journal（持久化）"

    # 反例：不允许再用忽略返回值的 lambda
    bad = re.search(r"add_cancel_hook\(\s*\n?\s*lambda:", src)
    assert bad is None, "取消钩子仍是忽略返回值的 lambda"


# ------------------------------------------------ B11/B12/B14 真实函数

def _load_s6():
    """导入照片 stage6（只导入定义，不执行 main）。"""
    import importlib
    p = os.path.join(_ROOT, "photo_restoration_pipeline", "stages")
    if p not in sys.path:
        sys.path.insert(0, p)
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    return importlib.import_module("stage6_encode")


def _w(p, data):
    with open(p, "wb") as f:
        f.write(data)
    return p


def _dig(p):
    import hashlib
    with open(p, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


class _L:
    def __init__(self):
        self.ev = []

    def info(self, e, *a, **k):
        self.ev.append(e)

    def warning(self, e, *a, **k):
        self.ev.append(e)

    def error(self, e, *a, **k):
        self.ev.append(e)


def test_b11_missing_old_digest_blocks_the_whole_rollback(tmp_path):
    """B11：某个目标缺 `old_digest` → **全组**预检拒绝，一个文件都不改。"""
    s6 = _load_s6()
    good = _w(str(tmp_path / "a.bin"), b"GOOD-CURRENT")
    bad = _w(str(tmp_path / "b.bin"), b"BAD-CURRENT")
    bk_good = _w(str(tmp_path / "bk_a"), b"OLD-A")
    bk_bad = _w(str(tmp_path / "bk_b"), b"OLD-B")

    snap = {
        good: {"state": s6.BACKUP_OK, "backup": bk_good, "old_digest": _dig(bk_good),
               "backup_digest": _dig(bk_good)},
        # old_digest 缺失 → 无从核对，必须拒绝整组
        bad: {"state": s6.BACKUP_OK, "backup": bk_bad, "old_digest": None,
              "backup_digest": _dig(bk_bad)},
    }
    log = _L()
    res = s6._rollback_publish(snap, log)

    assert res["fully_restored"] is False, res
    assert res["restored"] == [], res
    # 一个正式文件都没改
    assert open(good, "rb").read() == b"GOOD-CURRENT"
    assert open(bad, "rb").read() == b"BAD-CURRENT"
    assert "STAGE6_ROLLBACK_PRECHECK_FAILED" in log.ev, log.ev


def test_b11_unreadable_backup_blocks_the_whole_rollback(tmp_path):
    """B11：备份丢失/不可读 → 全组预检拒绝。"""
    s6 = _load_s6()
    tgt = _w(str(tmp_path / "a.bin"), b"CURRENT")
    snap = {
        tgt: {"state": s6.BACKUP_OK, "backup": str(tmp_path / "does-not-exist"),
              "old_digest": "deadbeef", "backup_digest": "deadbeef"},
    }
    log = _L()
    res = s6._rollback_publish(snap, log)
    assert res["fully_restored"] is False, res
    assert open(tgt, "rb").read() == b"CURRENT"
    assert "STAGE6_ROLLBACK_PRECHECK_FAILED" in log.ev, log.ev


def test_b12_successful_rollback_restores_exact_bytes(tmp_path):
    """B12 正向：备份与摘要都可信时，恢复出**逐字节相同**的旧代。"""
    s6 = _load_s6()
    old = b"OLD-GENERATION-BYTES"
    tgt = _w(str(tmp_path / "a.bin"), b"NEW-GENERATION")
    bk = _w(str(tmp_path / "bk"), old)
    d = _dig(bk)
    log = _L()
    res = s6._rollback_publish(
        {tgt: {"state": s6.BACKUP_OK, "backup": bk, "old_digest": d,
               "backup_digest": d}}, log)
    assert res["fully_restored"] is True, res
    assert open(tgt, "rb").read() == old
    # 不留下暂存文件
    left = [n for n in os.listdir(str(tmp_path)) if ".restore-" in n]
    assert not left, left


def test_b12_corrupted_backup_never_overwrites_the_target(tmp_path):
    """B12 反向：备份内容与摘要不符 → **不许**覆盖目标。"""
    s6 = _load_s6()
    tgt = _w(str(tmp_path / "a.bin"), b"CURRENT-GOOD")
    bk = _w(str(tmp_path / "bk"), b"TAMPERED")
    snap = {
        tgt: {"state": s6.BACKUP_OK, "backup": bk,
              "old_digest": "0" * 64,          # 与 TAMPERED 不符
              "backup_digest": _dig(bk)},      # 备份自身摘要自洽
    }
    log = _L()
    res = s6._rollback_publish(snap, log)
    assert res["fully_restored"] is False, res
    assert open(tgt, "rb").read() == b"CURRENT-GOOD", "坏备份盖掉了好文件"


def test_b14_absent_target_is_not_recreated(tmp_path):
    """B14：旧代本来不存在 → 不得凭备份凭空造出来。"""
    s6 = _load_s6()
    tgt = str(tmp_path / "never_existed.bin")
    bk = _w(str(tmp_path / "bk"), b"SOMETHING")
    snap = {
        tgt: {"state": s6.BACKUP_OK, "backup": bk, "old_digest": None,
              "backup_digest": _dig(bk), "existed": False},
    }
    log = _L()
    res = s6._rollback_publish(snap, log)
    # 无论如何都不该在"旧代不存在"的位置凭空写文件
    assert not os.path.exists(tgt), "旧代不存在却被恢复了"
    assert res["fully_restored"] is False or res.get("removed") == [], res


# ------------------------------------------------ B13 取消钩子失败必须留痕

def test_b13_cancel_hook_failure_is_logged_not_swallowed():
    """取消钩子抛异常时：**必须留痕**，且不影响后续钩子。

    审核原话："`CancelGuard` 也吞 hook 异常" —— 原来
    `except BaseException: continue`，取消时的回滚失败了日志里一个字都没有。
    """
    import importlib
    if _ROOT not in sys.path:
        sys.path.insert(0, _ROOT)
    cg = importlib.import_module("cancel_guard")

    class L:
        def __init__(self):
            self.ev = []

        def error(self, e, *a, **k):
            self.ev.append((e, k))

        def warning(self, e, *a, **k):
            self.ev.append((e, k))

        info = warning

    guard = cg.CancelGuard()
    guard._logger = L()
    ran = []

    def boom():
        raise RuntimeError("rollback exploded")

    def good():
        ran.append("good")

    guard.add_cancel_hook(boom)
    guard.add_cancel_hook(good)
    guard._run_cancel_hooks()      # 不得向外抛

    assert ran == ["good"], "前面的钩子失败不该挡住后面的"
    events = [e for e, _ in guard._logger.ev]
    assert "CANCEL_HOOK_FAILED" in events, events
    payload = dict(guard._logger.ev)["CANCEL_HOOK_FAILED"]
    assert "RuntimeError" in payload.get("error", ""), payload
