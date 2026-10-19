"""ID-07（第七轮）：实际执行指纹 —— 记录"到底加载了什么"。

**为什么配置不够**

配置里写 `upscale_model: realesr-general-x4v3` 说明不了执行事实：

* **同名权重换内容**：文件还是那个路径，内容已经换了（重下、被覆盖）；
* **后端不同**：同一个模型名可能走 ncnn-vulkan、走 Python API、
  或退到 CPU —— 三者的输出可能不一样；
* **工具版本不同**：torch / realesrgan / CUDA 版本变了，结果可能变；
* **降级**：OOM 之后自动换小模型/换后端，配置里看不出来。

任务书 §2.3：「模型名称也不足以说明执行……后端与解析后的资源应来自
执行记录，不是配置猜测」。

**本模块只做"取指纹"，不做决策。** 取不到就返回 None，绝不猜。
"""

from __future__ import annotations

import hashlib
import os
import platform
import subprocess
import sys
from typing import Any, Dict, List, Optional

__all__ = [
    "EXEC_SCHEMA",
    "file_fingerprint",
    "probe_ai_backend",
    "build_exec_fingerprint",
    "exec_sidecar_path",
    "write_exec_sidecar",
    "read_exec_sidecar",
]

EXEC_SCHEMA = "stage-exec/1"


def _sha256(path: str, chunk: int = 1 << 20) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            while True:
                b = fh.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except (OSError, FileNotFoundError):
        return None


def file_fingerprint(path: Optional[str]) -> Optional[Dict[str, Any]]:
    """权重/可执行文件的指纹：路径 + 大小 + 内容摘要。

    **必须有内容摘要**：只比路径和大小，同名等长替换完全发现不了。
    """
    if not path:
        return None
    p = str(path)
    if not os.path.exists(p):
        return {"path": p, "exists": False, "size": None, "sha256": None}
    try:
        st = os.stat(p)
    except OSError:
        return {"path": p, "exists": False, "size": None, "sha256": None}
    return {
        "path": os.path.abspath(p),
        "exists": True,
        "size": int(st.st_size),
        "sha256": _sha256(p),
    }


_PROBE_CODE = r"""
import json, sys
out = {}
try:
    import torch
    out["torch"] = torch.__version__
    out["cuda_available"] = bool(torch.cuda.is_available())
    out["cuda_version"] = getattr(torch.version, "cuda", None)
    if torch.cuda.is_available():
        try:
            out["gpu_name"] = torch.cuda.get_device_name(0)
        except Exception:
            out["gpu_name"] = None
except Exception as e:
    out["torch_error"] = f"{type(e).__name__}: {e}"
try:
    import realesrgan
    out["realesrgan"] = getattr(realesrgan, "__version__", "unknown")
except Exception as e:
    out["realesrgan_error"] = f"{type(e).__name__}: {e}"
out["python"] = sys.version.split()[0]
print(json.dumps(out))
"""


def probe_ai_backend(ai_python: Optional[str],
                     timeout: int = 60) -> Dict[str, Any]:
    """在**真正的 AI 解释器**里探测 torch/CUDA/realesrgan 版本。

    任务书提醒「不要只问编排器 Python 里装了什么」—— 编排器是
    Portable R72，根本没有 torch；真正的 AI 子进程是另一个解释器。
    所以这里显式用 `ai_python` 起子进程去问。

    探测失败返回 `{"probe_error": ...}`，**不编造版本号**。
    """
    py = ai_python or sys.executable
    if not py or not os.path.exists(py):
        return {"probe_error": f"interpreter not found: {py}"}
    try:
        r = subprocess.run([py, "-c", _PROBE_CODE],
                           capture_output=True, text=True, errors="replace",
                           timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"probe_error": "probe timed out", "interpreter": py}
    except Exception as exc:
        return {"probe_error": f"{type(exc).__name__}: {exc}",
                "interpreter": py}
    if r.returncode != 0:
        return {"probe_error": f"probe exited {r.returncode}",
                "stderr_tail": (r.stderr or "")[-600:],
                "interpreter": py}
    import json as _json
    try:
        data = _json.loads((r.stdout or "").strip().splitlines()[-1])
    except Exception:
        return {"probe_error": "unparsable probe output",
                "stdout_tail": (r.stdout or "")[-400:],
                "interpreter": py}
    data["interpreter"] = py
    return data


def build_exec_fingerprint(model: Optional[str],
                           model_path: Optional[str],
                           ai_python: Optional[str] = None,
                           backend: Optional[str] = None,
                           scale: Optional[int] = None,
                           fallback_reason: Optional[str] = None,
                           logger=None) -> Dict[str, Any]:
    """组装一份"实际执行"记录。

    `backend` 由调用方按**真实选择**填写（`ncnn-vulkan` / `python-api` /
    `python-api-cpu` / `dat-python`），不是配置推断。
    """
    fp: Dict[str, Any] = {
        "schema": EXEC_SCHEMA,
        "requested_model": model,
        "requested_scale": scale,
        "backend": backend,
        "model_weights": file_fingerprint(model_path),
        "fallback_reason": fallback_reason,
        "host": {
            "platform": platform.platform(),
            "machine": platform.machine(),
        },
    }
    try:
        fp["tools"] = probe_ai_backend(ai_python)
    except Exception as exc:            # 探测本身失败不能中断主流程
        fp["tools"] = {"probe_error": f"{type(exc).__name__}: {exc}"}
        if logger:
            try:
                logger.warning("EXEC_FINGERPRINT_PROBE_FAIL", str(exc))
            except Exception:
                pass
    return fp


def exec_sidecar_path(artifact_path: str) -> str:
    return str(artifact_path) + ".exec.json"


def write_exec_sidecar(artifact_path: str, payload: Dict[str, Any]) -> bool:
    """把执行指纹写到产物旁边；失败返回 False（不抛）。"""
    import json as _json
    path = exec_sidecar_path(artifact_path)
    tmp = f"{path}.tmp-{os.getpid()}"
    try:
        with open(tmp, "w", encoding="utf-8") as fh:
            _json.dump(payload, fh, ensure_ascii=False, sort_keys=True,
                       indent=2)
        os.replace(tmp, path)
        return True
    except OSError:
        try:
            if os.path.exists(tmp):
                os.unlink(tmp)
        except OSError:
            pass
        return False


def read_exec_sidecar(artifact_path: str) -> Optional[Dict[str, Any]]:
    import json as _json
    path = exec_sidecar_path(artifact_path)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            doc = _json.load(fh)
    except (OSError, ValueError):
        return None
    if not isinstance(doc, dict) or doc.get("schema") != EXEC_SCHEMA:
        return None
    return doc
