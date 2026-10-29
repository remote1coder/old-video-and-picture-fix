# -*- coding: utf-8 -*-
"""R14-06：有效路径必须在首次落盘之前定清楚并预检。

**背景**

2026-10-28 的事故：复制了别人的 manifest、用 `--output` 指向隔离目录，
结果仍然写回原目录、覆盖了既有验收产物。原因在 `main.py`：

```python
params = merge_params(...)                    # CLI --output → output_dir
params["output_path"] = job.get("output_path")  # job 又覆盖最终媒体
log_dir = params.get("log_dir", "./logs")       # 日志再取一次
```

`--output` 只换了 `output_dir`，**没换** `output_path`。

执行指令 §4 要求把有效路径物化并预检，本文件验证 `shared_modules/run_plan.py`。

**注意**：这里测的是**路径解析与预检**，不是发布事务。它只回答
"这些值是不是我以为的那些值、会不会写到不该写的地方"。
"""
import os
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import run_plan as rp  # noqa: E402


def _job(**kw):
    base = {"job_id": "job_001", "input_path": "", "output_path": ""}
    base.update(kw)
    return base


def _touch(path):
    """造一个极小的输入文件（预检要求输入真实存在）。"""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"fixture")
    return str(path)


# ---------------------------------------------------------------- 路径归一

def test_resolve_path_resolves_existing_ancestor(tmp_path):
    """不存在的尾部按最深已存在祖先解析（父目录是 junction 时也能发现）。"""
    real = tmp_path / "real_dir"
    real.mkdir()
    got = rp.resolve_path(str(real / "not_yet" / "file.mkv"))
    assert got is not None
    # resolve_path 保留原始大小写（消费用）；比较时用 path_key
    assert rp.path_key(str(real)) in rp.path_key(got)
    # 尾部未被丢弃
    assert got.endswith(os.path.join("not_yet", "file.mkv"))


def test_is_within_does_not_use_bare_startswith(tmp_path):
    """`out2` 不是 `out` 的子路径 —— 裸 startswith 会误判。"""
    out = tmp_path / "out"
    out.mkdir()
    out2 = tmp_path / "out2"
    out2.mkdir()
    assert rp.is_within(str(out / "a.mkv"), str(out)) is True
    assert rp.is_within(str(out2 / "a.mkv"), str(out)) is False


def test_is_within_handles_case_and_drive_mismatch(tmp_path):
    """大小写差异不误判；跨盘符时返回 False 而不是抛异常。"""
    p = str(tmp_path / "Sub" / "a.mkv")
    assert rp.is_within(p, str(tmp_path).upper()) is True
    # 不同盘符：commonpath 会抛 ValueError，必须被吞成 False
    assert rp.is_within("Z:\\x\\y.mkv", "C:\\z") is False


# ---------------------------------------------------------------- 六种场景

def test_stale_manifest_external_output_is_blocked(tmp_path):
    """旧 manifest 带外部绝对输出 + 给了授权根 → 必须阻断。"""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "elsewhere" / "c_restored.mkv"
    (tmp_path / "elsewhere").mkdir()

    plan = rp.resolve_run_plan(
        _job(input_path=_touch(tmp_path / "in.mp4"), output_path=str(outside)),
        {"temp_dir": str(allowed / "tmp"), "log_dir": str(allowed / "logs")},
        write_roots=[str(allowed)])

    assert plan.ok is False, "外部输出必须被拦下"
    assert any("授权写根之外" in e for e in plan.errors), plan.errors
    assert plan.get("output_path") == rp.resolve_path(str(outside))
    assert plan.source_of("output_path") == rp.SOURCE_JOB


def test_relocate_output_overrides_job_paths(tmp_path):
    """显式 `--relocate-output` 把媒体与四个侧车全部改到新目录。"""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    old = tmp_path / "old" / "c_restored.mkv"

    plan = rp.resolve_run_plan(
        _job(input_path=_touch(tmp_path / "in.mp4"), output_path=str(old)),
        {"temp_dir": str(allowed / "tmp"), "log_dir": str(allowed / "logs")},
        write_roots=[str(allowed)], relocate_output=str(allowed))

    assert plan.ok is True, plan.errors
    assert plan.get("output_path") == rp.resolve_path(
        str(allowed / "c_restored.mkv"))
    assert plan.source_of("output_path") == rp.SOURCE_CLI_RELOCATE
    for key in ("verify_path", "fullchain_path", "status_path",
                "contract_path", "journal_path"):
        assert rp.is_within(plan.get(key), str(allowed)), (key, plan.get(key))


def test_input_equals_output_is_blocked(tmp_path):
    """输入=输出必须拦住，否则会覆盖原始素材。"""
    same = tmp_path / "clip.mkv"
    same.write_bytes(b"x")
    plan = rp.resolve_run_plan(
        _job(input_path=str(same), output_path=str(same)),
        {"temp_dir": str(tmp_path / "t"), "log_dir": str(tmp_path / "l")})
    assert plan.ok is False
    assert any("同一个路径" in e or "同名同目录" in e for e in plan.errors)


def test_sidecar_collision_between_jobs(tmp_path):
    """两个 job 的媒体与侧车撞到同一路径 → 跨 job 冲突。"""
    a = rp.resolve_run_plan(
        _job(job_id="a", input_path=_touch(tmp_path / "a.mp4"),
             output_path=str(tmp_path / "out.mkv")),
        {"temp_dir": str(tmp_path / "t"), "log_dir": str(tmp_path / "l")})
    b = rp.resolve_run_plan(
        _job(job_id="b", input_path=_touch(tmp_path / "b.mp4"),
             output_path=str(tmp_path / "out.mkv")),
        {"temp_dir": str(tmp_path / "t2"), "log_dir": str(tmp_path / "l2")})

    assert a.ok and b.ok
    conflicts = rp.detect_cross_job_conflicts([a, b])
    assert conflicts, "同名输出必须被识别为跨 job 冲突"
    assert any("out.mkv" in c for c in conflicts)


def test_temp_and_log_outside_root_are_blocked(tmp_path):
    """temp/log 越界也要拦（它们同样会写盘）。"""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    plan = rp.resolve_run_plan(
        _job(input_path=_touch(tmp_path / "in.mp4"),
             output_path=str(allowed / "o.mkv")),
        {"temp_dir": str(tmp_path / "outside_tmp"),
         "log_dir": str(tmp_path / "outside_logs")},
        write_roots=[str(allowed)])

    assert plan.ok is False
    msgs = " ".join(plan.errors)
    assert "temp_dir" in msgs and "log_dir" in msgs, plan.errors


def test_wsl_copyback_outside_root_is_blocked(tmp_path):
    """WSL 最终回拷目标同样属于写路径，也要在授权根内。"""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    plan = rp.resolve_run_plan(
        _job(input_path=str(tmp_path / "in.mp4"),
             output_path=str(allowed / "o.mkv"),
             original_output_path=str(tmp_path / "linux" / "o.mkv")),
        {"temp_dir": str(allowed / "t"), "log_dir": str(allowed / "l")},
        write_roots=[str(allowed)])

    assert plan.ok is False
    assert any("wsl_copyback_path" in e for e in plan.errors), plan.errors


# ---------------------------------------------------------------- 来源记录

def test_sources_are_recorded_for_every_path(tmp_path):
    """每个值都要能解释"听谁的"——这是本次事故的直接教训。"""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    plan = rp.resolve_run_plan(
        _job(input_path=_touch(tmp_path / "in.mp4"),
             output_path=str(allowed / "o.mkv"),
             temp_dir=str(allowed / "job_tmp")),
        {"temp_dir": str(allowed / "cli_tmp"), "log_dir": str(allowed / "l")},
        write_roots=[str(allowed)])

    assert plan.ok is True, plan.errors
    # job 级优先于 params（保留既有逐 job 配置）
    assert plan.source_of("temp_dir") == rp.SOURCE_JOB
    assert "job_tmp" in plan.get("temp_dir")
    # log_dir 有独立语义：只有显式标记为 CLI 来源时才记 cli，
    # 否则视为配置值（R15-01 之后不再从 job.log_path 推断）
    assert plan.source_of("log_dir") in (rp.SOURCE_CONFIG, rp.SOURCE_CLI)
    d = plan.as_dict()
    assert all("source" in v for v in d["paths"].values())


def test_log_dir_source_distinguishes_cli_and_config(tmp_path):
    """CLI 给的 log_dir 与配置给的必须能区分开。"""
    allowed = tmp_path / "allowed"
    allowed.mkdir()
    base = _job(input_path=_touch(tmp_path / "in" / "c.mp4"),
                output_path=str(allowed / "o.mkv"))

    p_cfg = rp.resolve_run_plan(
        base, {"temp_dir": str(allowed / "t"), "log_dir": str(allowed / "l")},
        write_roots=[str(allowed)])
    assert p_cfg.source_of("log_dir") == rp.SOURCE_CONFIG

    p_cli = rp.resolve_run_plan(
        base, {"temp_dir": str(allowed / "t"), "log_dir": str(allowed / "l"),
               "_log_dir_from_cli": True},
        write_roots=[str(allowed)])
    assert p_cli.source_of("log_dir") == rp.SOURCE_CLI


def test_no_write_roots_only_warns(tmp_path):
    """不给授权根时不阻断（保持向后兼容），但必须留痕。"""
    plan = rp.resolve_run_plan(
        _job(input_path=_touch(tmp_path / "in.mp4"),
             output_path=str(tmp_path / "o.mkv")),
        {"temp_dir": str(tmp_path / "t"), "log_dir": str(tmp_path / "l")})
    assert plan.ok is True
    assert any("未提供授权写根" in w for w in plan.warnings)


def test_missing_output_is_an_error(tmp_path):
    """既没 job 输出也没 output_dir → 明确报错，不猜默认文件名。"""
    plan = rp.resolve_run_plan(
        _job(input_path=str(tmp_path / "in.mp4")),
        {"temp_dir": str(tmp_path / "t"), "log_dir": str(tmp_path / "l")})
    assert plan.ok is False
    assert any("输出路径无法确定" in e for e in plan.errors)


def test_missing_input_is_an_error(tmp_path):
    plan = rp.resolve_run_plan(
        _job(output_path=str(tmp_path / "o.mkv")),
        {"temp_dir": str(tmp_path / "t"), "log_dir": str(tmp_path / "l")})
    assert plan.ok is False
    assert any("输入路径缺失" in e for e in plan.errors)


# ---------------------------------------------------------------- 生产接入

def test_process_job_blocks_before_any_write(tmp_path, capsys):
    """生产 `process_job` 在**落盘之前**拦下越界输出。

    这是 2026-10-28 那次误覆盖的直接回归：旧 manifest 带外部绝对输出，
    只换 `--output` 不会改变 `output_path`。现在给了授权根就必须拦，
    而且**一个字节都不能写**。
    """
    sys.path.insert(0, os.path.join(_ROOT, "video_restoration_pipeline"))
    import main as vmain

    allowed = tmp_path / "allowed"
    allowed.mkdir()
    outside = tmp_path / "OUTSIDE"
    inp = tmp_path / "in" / "c.mp4"
    inp.parent.mkdir(parents=True)
    inp.write_bytes(b"fixture")

    class _Log:
        def __init__(self):
            self.events = []

        def info(self, ev, *a, **k):
            self.events.append(("info", ev))

        def warning(self, ev, *a, **k):
            self.events.append(("warn", ev))

        def error(self, ev, *a, **k):
            self.events.append(("error", ev))

        def debug(self, *a, **k):
            pass

        def has(self, ev):
            return any(e == ev for _l, e in self.events)

    log = _Log()
    job = {"job_id": "j1", "input_path": str(inp),
           "output_path": str(outside / "c_restored.mkv")}
    args = {"output": None, "allow_write_root": [str(allowed)],
            "relocate_output": None, "no_wsl_redirect": True}

    rc, msg = vmain.process_job(job, {}, args, log)

    assert rc == 82, f"越界必须先拦，实际 rc={rc} msg={msg}"
    # R15-03：计划阶段**零落盘** —— 被拦下的 job 不写自己的日志，
    # 诊断走 stderr（缓冲 logger），不会在授权根外造目录。
    err = capsys.readouterr().err
    assert "RUNPLAN_BLOCKED" in err, err[:400]
    assert "RUNPLAN_ABORT" in err, err[:400]
    assert not outside.exists(), "拦截后不得创建任何输出目录"
    assert not (outside / "c_restored.mkv").exists()
    # 也没有任何 job 日志落到外面
    assert not (tmp_path / "logs").exists(), \
        "预检未通过时不应创建日志目录"


def test_process_job_relocate_redirects_outputs(tmp_path):
    """显式 `--relocate-output` 让输出真的搬到新目录（来源可追溯）。"""
    plan = rp.resolve_run_plan(
        {"job_id": "j1", "input_path": _touch(tmp_path / "in" / "c.mp4"),
         "output_path": str(tmp_path / "old" / "c_restored.mkv")},
        {"temp_dir": str(tmp_path / "t"), "log_dir": str(tmp_path / "l")},
        relocate_output=str(tmp_path / "new"))
    assert plan.ok is True, plan.errors
    assert rp.path_key(plan.get("output_path")) == rp.path_key(
        str(tmp_path / "new" / "c_restored.mkv"))
    assert plan.source_of("output_path") == rp.SOURCE_CLI_RELOCATE
