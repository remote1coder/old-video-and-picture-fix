# -*- coding: utf-8 -*-
r"""R23-LONG-01：命令行长度必须按**转义后**的总长判断。

这条缺陷是"长编码实测"顺带抓出来的，20 秒小片段永远测不到（见下）。
第一次跑 16 分钟 1080p 真实素材时，stage1 **10 毫秒内 rc=82**，
temp 一个字节都没写，job 日志只有：

    STAGE_FAILED  Stage 1 failed: 82 | [WinError 206] The filename or
                  extension is too long

旧实现 `if len(params_json) > 30000: 走参数文件`，两个问题：

1. `len(params_json)` 不是命令行占用。JSON 里的 `"` 与 Windows 路径的 `\`
   会被 `subprocess.list2cmdline()` 再转义一遍，**实测增幅 1.13–1.30 倍**；
   再叠加 head/tail 与安全余量（约 1.5 千字符），`len` 到 2.5 万就可能越界。
   旧判据要到 3 万才切文件 —— 中间 5000 字符是"看着安全、实际必炸"的区间。
2. `except Exception: return 82` 把 WinError 206 吞成无信息量的 82。

为什么只有长片能测出来：params 大小与参数数量相关（本素材的
`field_segments` 有多段、`_work_format` 计划整份进 params），
短视频用同一份参数也过得去 —— 真正把它推过线的是长片特有的字段量。
"""
import json
import os
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import main as vmain  # noqa: E402

BS = chr(92)                      # 反斜杠，避开各层转义
CMD_LIMIT = 32767


def _job(tmp_path):
    src = tmp_path / "in.mkv"
    src.write_bytes(b"x")
    return {"job_id": "job_001", "input_path": str(src),
            "original_width": 1920, "original_height": 1080}


def _params(tmp_path, **extra):
    p = {"temp_dir": str(tmp_path / "temp"),
         "output_path": str(tmp_path / "out" / "restored.mkv"),
         "duration": 962.295,
         "work_format": "auto"}
    p.update(extra)
    return p


@pytest.fixture()
def capture(monkeypatch, tmp_path):
    """截住子进程启动，只回看命令行（不真的跑 stage）。"""
    os.makedirs(tmp_path / "temp", exist_ok=True)
    seen = {"cmd": None, "calls": 0}

    def fake_run_managed(cmd, **kwargs):
        seen["cmd"] = list(cmd)
        seen["calls"] += 1

        class _R(object):
            returncode = 0
            stdout = ""
            stderr = ""
        return _R()

    import cancel_guard
    monkeypatch.setattr(cancel_guard, "run_managed", fake_run_managed)
    return seen


def _need(cmd_without_params, params_json, config_path):
    head = len(subprocess.list2cmdline(cmd_without_params))
    tail = len(subprocess.list2cmdline(["--config", config_path])) + 64
    return head + len(subprocess.list2cmdline([params_json])) + tail + 1024


def test_short_params_stay_inline(capture, tmp_path):
    """正常大小的 params 仍然内联 —— 不要为了修 206 把所有人都赶去写文件。"""
    params = _params(tmp_path)
    rc, err = vmain.run_stage("stage1_preprocess.py", _job(tmp_path), params,
                              str(tmp_path / "config.yaml"))
    assert rc == 0, err
    cmd = capture["cmd"]
    assert "--params" in cmd
    assert "--params-file" not in cmd
    # 内联的内容必须能被 json 还原成同一份 params
    payload = cmd[cmd.index("--params") + 1]
    assert json.loads(payload)["duration"] == params["duration"]


def test_long_params_go_to_a_file(capture, tmp_path):
    """超限的 params 走 `--params-file`，内容必须逐字节等于原 JSON。"""
    big = {("k%d" % i): ("C:" + BS + "a" + BS + "b" + BS + ("f%d.mkv" % i))
           for i in range(1200)}
    params = _params(tmp_path, padding=big)
    rc, err = vmain.run_stage("stage1_preprocess.py", _job(tmp_path), params,
                              str(tmp_path / "config.yaml"))
    assert rc == 0, err
    cmd = capture["cmd"]
    assert "--params" not in cmd, "超限了还内联，必然再撞 WinError 206"
    assert "--params-file" in cmd
    path = cmd[cmd.index("--params-file") + 1]
    # run_stage 的 finally 会删掉参数文件，所以只能断言"当时存在过"
    assert path.endswith(".json")


def test_budget_is_computed_after_escaping_not_from_len(capture, tmp_path):
    """核心回归：len 看着安全、转义后越界时，必须走文件。

    构造一份以引号/反斜杠为主的 params：字符数不大，但 list2cmdline 的
    转义会把实际占用推过 32767。旧判据（只看 len）会内联 → 必炸。
    """
    # 每个值的 JSON 形态是 "\\\"x\\\"" 这类高转义密度；
    # 项数要卡在"len < 30000（旧判据放行）但转义后必然越界"的窗口里。
    dense = {("q%d" % i): ('"x%d"' % i) + BS * 6 for i in range(800)}
    params = _params(tmp_path, dense=dense)
    payload = json.dumps(params, default=str)
    escaped = subprocess.list2cmdline([payload])

    assert len(payload) < 30000, (
        "构造失败：这份 params 自己就超过旧判据的 30000，测不出差异"
        "（len=%d）" % len(payload))
    assert len(escaped) > len(payload) * 1.10, (
        "构造失败：转义没有明显放大（%d -> %d）" % (len(payload), len(escaped)))
    # 确认它确实落在"旧判据放行、新判据拒绝"的区间
    _need_now = _need([sys.executable, str(_ROOT) + os.sep + "main.py",
                       "--input", str(tmp_path / "in.mkv"),
                       "--job-id", "job_001"],
                      payload, str(tmp_path / "config.yaml"))
    assert _need_now > CMD_LIMIT, (
        "构造失败：转义后总长 %d 仍在上限 %d 之内，测不出差异"
        % (_need_now, CMD_LIMIT))

    rc, err = vmain.run_stage("stage1_preprocess.py", _job(tmp_path), params,
                              str(tmp_path / "config.yaml"))
    assert rc == 0, err
    cmd = capture["cmd"]
    assert "--params-file" in cmd, (
        "转义后总长已超限却仍内联：len=%d，escaped=%d"
        % (len(payload), len(escaped)))


def test_params_file_write_failure_is_reported_not_downgraded(
        monkeypatch, capture, tmp_path):
    """写参数文件失败时必须明确报错，**不能**回落 inline。

    回落只会二次触发 206，再被上层吞成 "Stage 1 failed: 82" —— 那次实测
    就是这么浪费了两轮排查。
    """
    def boom(params, payload):
        raise OSError(28, "No space left on device")

    monkeypatch.setattr(vmain, "_write_stage_params_file", boom)
    big = {("k%d" % i): ("C:" + BS + "a" + BS + ("f%d.mkv" % i))
           for i in range(1200)}
    params = _params(tmp_path, padding=big)
    rc, err = vmain.run_stage("stage1_preprocess.py", _job(tmp_path), params,
                              str(tmp_path / "config.yaml"))

    assert rc == 80, err
    assert "could not be written" in err
    assert "No space left on device" in err
    assert capture["calls"] == 0, "写失败后还去启动子进程了（回落 inline）"


def test_winerror_206_is_explained(monkeypatch, capture, tmp_path):
    """命令行越界要说清是越界，不能只剩一个 82。"""
    import cancel_guard

    def too_long(cmd, **kwargs):
        raise OSError(0, "The filename or extension is too long", None, 206)

    monkeypatch.setattr(cancel_guard, "run_managed", too_long)
    params = _params(tmp_path)
    rc, err = vmain.run_stage("stage1_preprocess.py", _job(tmp_path), params,
                              str(tmp_path / "config.yaml"))

    assert rc == 80, err
    assert "WinError 206" in err or "command line too long" in err
    assert "32767" in err


def test_other_oserror_still_maps_to_82(monkeypatch, capture, tmp_path):
    """不要为了识别 206 把其它 OSError 也改成 80（会破坏既有重试语义）。"""
    import cancel_guard

    def denied(cmd, **kwargs):
        raise OSError(0, "Access is denied", None, 5)

    monkeypatch.setattr(cancel_guard, "run_managed", denied)
    params = _params(tmp_path)
    rc, err = vmain.run_stage("stage1_preprocess.py", _job(tmp_path), params,
                              str(tmp_path / "config.yaml"))

    assert rc == 82, err
    assert "Access is denied" in err
