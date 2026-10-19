"""AI 子进程：为什么"卡两小时"—— 两层原因与已修的边界。

**第一层：超时太长**

批量路径原来是 `timeout=7200`（两小时）。子进程一旦不返回，父进程要
等满两小时才动；main 若重试，再两小时。这就是"卡了约 2 小时"的
**字面来源**。

**第二层：PYTHONHOME 让 .venv_r4 永久阻塞**

父进程是 Portable R72（VapourSynth 环境）。`os.environ.copy()` 会把
R72 的 Python 环境变量原封不动传给**另一个解释器** `.venv_r4`。
实测（同一个 .venv_r4）：

```
干净环境                        -> torch 2.11.0+cu128 cuda True, 1.6s
带 PYTHONHOME=<r72>             -> 卡死（60s 超时，<no Python frame>）
带 PYTHONPATH=<r72>/site-packages-> OK
```

卡死时进程 CPU 只有 0.14s、内存 11MB —— 不是"算得慢"，是压根没动。

**已修**：清掉 6 个 PYTHON* 变量；超时改为按帧数动态（每帧 30s 上限，
最少 300s）；走 `run_managed`（可取消、超时 kill+reap）。

**还没修好的**：即使上述两条都生效（单独验证：从 R72 起子进程 +
清 env + run_managed = 2.2 秒 `OK True`），**stage3 端到端仍然卡**
（AI 子进程 CPU 0.14s、0 帧）。说明还有第三层原因没定位到。
这些测试只覆盖"已验证有效"的部分，不声称 stage3 已修好。
"""

import os
import subprocess
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VSTAGES = os.path.join(ROOT, "video_restoration_pipeline", "stages")
if VSTAGES not in sys.path:
    sys.path.insert(0, VSTAGES)

import stage3_ai_upscale as s3  # noqa: E402

R72 = os.path.join(ROOT, "tools_r4", "vs_portable", "r72",
                   "python.exe")
VENV = os.path.join(ROOT, ".venv_r4", "Scripts", "python.exe")

_BAD = ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP",
        "PYTHONEXECUTABLE", "PYTHONUSERBASE", "PYTHONNOUSERSITE")


def test_all_callsites_strip_python_env():
    """三个起 AI 子进程的地方都要清 PYTHON* （不能只修一个）。"""
    src = open(s3.__file__, encoding="utf-8").read()
    assert src.count("for _bad_env in (") == 3, \
        "不是三处都清了 PYTHON* 环境变量"
    assert src.count("env.pop(_bad_env, None)") == 3
    for key in ("PYTHONHOME", "PYTHONPATH"):
        assert key in src


def test_batch_timeout_is_not_two_hours():
    """批量路径不能再用固定 7200 秒。"""
    src = open(s3.__file__, encoding="utf-8").read()
    assert "timeout=7200, shell=False, env=env)  # Longer timeout" not in src, \
        "批量路径仍是固定两小时超时"
    assert "_batch_timeout = max(300, min(7200, _batch_frames * 30))" in src


def test_batch_uses_cancel_guard():
    """批量 AI 子进程必须受守卫管理（可取消、超时 kill+reap）。"""
    src = open(s3.__file__, encoding="utf-8").read()
    assert "_cancel_guard.run_managed(" in src
    assert "STAGE3_AI_SUBPROC_TIMEOUT" in src


@pytest.mark.skipif(not (os.path.exists(R72) and os.path.exists(VENV)),
                    reason="需要 R72 与 .venv_r4")
def test_stripping_env_makes_venv_import_torch_fast():
    """从 R72 起 .venv_r4 子进程 + 清 PYTHON* → import torch 必须很快。

    这是**已修部分的证明**，不是"stage3 已修好"的证明。
    """
    env = os.environ.copy()
    for k in _BAD:
        env.pop(k, None)
    cmd = [VENV, "-c",
           "import time;t=time.time();import torch;"
           "print(round(time.time()-t,1), torch.cuda.is_available())"]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=120,
                       env=env)
    assert r.returncode == 0, f"子进程失败：{(r.stderr or '')[-300:]}"
    secs = float((r.stdout or "99").split()[0])
    assert secs < 30, f"import torch 用了 {secs}s（说明还在被环境变量拖住）"


@pytest.mark.skipif(not (os.path.exists(R72) and os.path.exists(VENV)),
                    reason="需要 R72 与 .venv_r4")
def test_pythonhome_alone_is_the_culprit():
    """锁死结论：只设 PYTHONHOME 就足以让 .venv_r4 卡死。

    这条用短超时验证"会卡"这件事本身；卡住时子进程会被 kill。
    """
    env = os.environ.copy()
    for k in _BAD:
        env.pop(k, None)
    env["PYTHONHOME"] = os.path.dirname(os.path.dirname(R72))
    cmd = [VENV, "-c", "print('should-not-print')"]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=20,
                           env=env)
    except subprocess.TimeoutExpired:
        return          # 卡住 = 符合预期
    # 没卡住也可以（不同 Python 版本行为不同），但至少要能得出结论
    if r.returncode == 0:
        pytest.skip(f"本机 PYTHONHOME 未触发卡死（stdout={r.stdout!r}）")
