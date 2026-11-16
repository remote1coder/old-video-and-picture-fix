# -*- coding: utf-8 -*-
"""`photo_model_worker` 的客户端：起进程、发请求、收结果。

## 为什么要有这一层

主流程跑在一个解释器里（没有 torch），模型必须跑在另一个（`.venv_r4`）。
本类把"进程 + JSON 行协议"包成一个对象，让 `photo_chain` 只需要
`client.upscale(png, ...)`。

## 三条必须守住的事

1. **模型复用**：worker 起来后模型常驻。实测第一次 `upscale` 往返 3.9s
   （含加载），第二次 **0.0s**。
2. **协议通道干净**：worker 启动时把 fd 1 dup 出来当数据通道、
   fd 1 指向 stderr —— 否则 `realesrgan` 的 print 会污染 JSON 行。
3. **必须能收尸**：`close()` 发 `shutdown`；进程不听话就 terminate →
   kill。跑批几十张图时，漏一个 worker 就是 3 GB 显存不释放。
"""
from __future__ import annotations

import atexit
import base64
import json
import os
import subprocess
import sys
import threading
import time
from typing import Any, Dict, Optional

__all__ = ["ModelWorkerClient", "find_ai_python"]


def find_ai_python() -> str:
    """找带 torch 的解释器（项目里是 `.venv_r4`）。"""
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.dirname(here)
    for rel in (os.path.join(".venv_r4", "Scripts", "python.exe"),
                os.path.join(".venv_r4", "bin", "python")):
        p = os.path.join(root, rel)
        if os.path.isfile(p):
            return p
    # 环境变量兜底（与 ai_runtime 的约定一致）
    env = os.environ.get("WB_AI_PYTHON") or ""
    if env and os.path.isfile(env):
        return env
    return sys.executable


class ModelWorkerClient:
    """一个常驻模型 worker 的句柄。线程安全（内部一把锁串行化请求）。"""

    def __init__(self, *, python: str = "", worker: str = "",
                 model_dir: str = "", gpu_id: int = 0,
                 startup_timeout: float = 120.0) -> None:
        self.python = python or find_ai_python()
        if not worker:
            worker = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  "photo_model_worker.py")
        self.worker = worker
        self.model_dir = model_dir
        self.gpu_id = int(gpu_id)
        self.startup_timeout = float(startup_timeout)
        self._proc: Optional[subprocess.Popen] = None
        self._err_file = None
        self._err_path = ""
        self._lock = threading.Lock()
        self._closed = False
        atexit.register(self.close)

    # ---------------------------------------------------------------- 生命周期
    def _ensure(self) -> subprocess.Popen:
        if self._closed:
            raise RuntimeError("client is closed")
        if self._proc is not None and self._proc.poll() is None:
            return self._proc
        # **stderr 落文件，不能用 PIPE**：
        #   * 挂 PIPE 不读 → 缓冲区（64 KiB）满了 worker 就阻塞在写上；
        #   * 而且它继承了父进程的 stdout —— 父进程若用
        #     `subprocess.run(capture_output=True)` 等"写端全关"，
        #     worker 不退就永远等不完（实测：pytest 里直接 SIGTERM）。
        #
        # 落文件同时解决了诊断：崩溃时能读到尾部。
        import tempfile as _tf
        self._err_file = _tf.NamedTemporaryFile(
            prefix="photo_worker_", suffix=".err", delete=False)
        self._err_path = self._err_file.name
        self._proc = subprocess.Popen(
            [self.python, self.worker],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._err_file,
            text=True, encoding="utf-8", bufsize=1, shell=False)
        return self._proc

    def _stderr_tail(self, n: int = 500) -> str:
        """读 worker 的 stderr 尾部（诊断用）。"""
        try:
            self._err_file.flush()
        except Exception:
            pass
        try:
            with open(self._err_path, "r", encoding="utf-8",
                      errors="replace") as f:
                return f.read()[-n:]
        except Exception:
            return ""

    def close(self) -> None:
        self._closed = True
        p = self._proc
        self._proc = None
        if p is None:
            return
        try:
            if p.poll() is None:
                p.stdin.write(json.dumps({"op": "shutdown"}) + "\n")
                p.stdin.flush()
                p.wait(timeout=10)
        except Exception:
            try:
                p.terminate()
                p.wait(timeout=10)
            except Exception:
                try:
                    p.kill()
                except Exception:
                    pass
        for _s in (p.stdin, p.stdout):
            try:
                _s.close()
            except Exception:
                pass
        try:
            if self._err_file is not None:
                self._err_file.close()
        except Exception:
            pass
        try:
            if self._err_path and os.path.exists(self._err_path):
                os.remove(self._err_path)
        except Exception:
            pass

    def __enter__(self):
        return self

    def __exit__(self, *a):
        self.close()

    # ---------------------------------------------------------------- 请求
    def _call(self, req: Dict[str, Any],
              timeout: float = 600.0) -> Dict[str, Any]:
        with self._lock:
            p = self._ensure()
            try:
                p.stdin.write(json.dumps(req) + "\n")
                p.stdin.flush()
            except Exception as exc:
                # 进程已经死了：读出 stderr 尾巴再报错（没有它没法诊断）
                tail = self._stderr_tail()
                self._proc = None
                return {"ok": False,
                        "error": "worker stdin closed: %s | stderr=%s"
                                 % (exc, tail)}
            line = ""
            try:
                line = p.stdout.readline()
            except Exception as exc:
                return {"ok": False, "error": "read failed: %s" % exc}
            if not line:
                tail = self._stderr_tail()
                self._proc = None
                return {"ok": False,
                        "error": "worker produced no response (crashed?) "
                                 "stderr=%s" % tail}
            try:
                return json.loads(line)
            except Exception as exc:
                return {"ok": False,
                        "error": "worker returned non-JSON (%s): %r"
                                 % (exc, line[:120])}

    def ping(self) -> Dict[str, Any]:
        return self._call({"op": "ping"}, timeout=30.0)

    def upscale(self, png: bytes, *, model: str = "realesr-general-x4v3",
                scale: int = 2, half: bool = True,
                timeout: float = 900.0) -> Dict[str, Any]:
        """超分。返回 `{"ok", "png", "width", "height", "elapsed", ...}`。

        `png` 是**解码后的字节**（不是 base64）—— 让调用方不必关心编码。
        """
        r = self._call({
            "op": "upscale",
            "png_b64": base64.b64encode(png).decode("ascii"),
            "model": model, "scale": int(scale), "half": bool(half),
            "gpu_id": self.gpu_id,
            "model_dir": self.model_dir,
        }, timeout=timeout)
        if r.get("ok") and r.get("png_b64"):
            r["png"] = base64.b64decode(r["png_b64"])
            r.pop("png_b64", None)
        return r

    def face(self, png: bytes, *, upscale: int = 2,
             only_center_face: bool = False,
             timeout: float = 900.0) -> Dict[str, Any]:
        r = self._call({
            "op": "face",
            "png_b64": base64.b64encode(png).decode("ascii"),
            "upscale": int(upscale),
            "only_center_face": bool(only_center_face),
            "gpu_id": self.gpu_id,
            "model_dir": self.model_dir,
        }, timeout=timeout)
        if r.get("ok") and r.get("png_b64"):
            r["png"] = base64.b64decode(r["png_b64"])
            r.pop("png_b64", None)
        return r
