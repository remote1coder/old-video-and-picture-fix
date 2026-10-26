# -*- coding: utf-8 -*-
"""零落盘流式 AI 脚本的三条数据契约回归。

对应核查报告 STR-01 / STR-02 / STR-03：

- STR-01：第三方 `print` 不能污染二进制 stdout。
  本地 realesrgan 在 utils.py 里有 `print('\\tInput is a 16-bit image')`
  与 `print('\\tTile i/n')`，直接写 stdout 会与帧数据混流。
- STR-02：暗帧（最大值 ≤256）会被 realesrgan 按亮度误判成 8bit 返回 uint8，
  那一帧只写约定字节数的一半 → 下游全部错位。必须规整成 uint16 再写。
- STR-03：ffmpeg 用 rgb48le（RGB），而 realesrgan 的 enhance() 接收/返回 BGR。
  不转换就会把红蓝通道颠倒喂给模型。

本测试用**假模型**跑真实生成的脚本，检查进出字节与通道，不依赖 GPU 与权重。
"""
import os
import subprocess
import sys
import textwrap

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SHARED = os.path.join(_ROOT, "shared_modules")
sys.path.insert(0, _SHARED)

from stream_ai_template import build_stream_ai_script  # noqa: E402

IN_W, IN_H, SCALE = 8, 6, 2
IN_BYTES = IN_W * IN_H * 3 * 2
OUT_BYTES = (IN_W * SCALE) * (IN_H * SCALE) * 3 * 2


def _fake_pkg(tmp_path):
    """造一个假的 realesrgan / basicsr / torch 包，记录模型实际收到的东西。"""
    pkg = tmp_path / "fakepkg"
    pkg.mkdir()
    (pkg / "torch.py").write_text(
        "def __getattr__(name):\n"
        "    if name == 'cuda':\n"
        "        class _C:\n"
        "            @staticmethod\n"
        "            def is_available():\n"
        "                return False\n"
        "            @staticmethod\n"
        "            def empty_cache():\n"
        "                pass\n"
        "        return _C\n"
        "    raise AttributeError(name)\n",
        encoding="utf-8")

    bs = pkg / "basicsr"
    bs.mkdir()
    (bs / "__init__.py").write_text("", encoding="utf-8")
    arch = bs / "archs"
    arch.mkdir()
    (arch / "__init__.py").write_text("", encoding="utf-8")
    (arch / "rrdbnet_arch.py").write_text(
        "class RRDBNet:\n"
        "    def __init__(self, **kw):\n"
        "        self.kw = kw\n",
        encoding="utf-8")
    (arch / "srvgg_arch.py").write_text(
        "class SRVGGNetCompact:\n"
        "    def __init__(self, **kw):\n"
        "        self.kw = kw\n",
        encoding="utf-8")

    rg = pkg / "realesrgan"
    rg.mkdir()
    (rg / "__init__.py").write_text(
        "from .model import RealESRGANer\n", encoding="utf-8")
    # 假模型：故意 print 到文本 stdout，并把收到的输入记录到文件
    (rg / "model.py").write_text(textwrap.dedent("""
        import os, sys
        import numpy as np

        class RealESRGANer:
            def __init__(self, scale=None, model_path=None, model=None,
                         half=False, gpu_id=0):
                self.scale = scale
                self.out_dtype = np.uint16
                # 故意的第三方文本输出 —— STR-01 的复现条件
                print('\\tInput is a 16-bit image')
                print('\\tTile 1/1')
                sys.stdout.flush()

            def enhance(self, img, outscale=1):
                # 记录模型实际收到的通道，供断言（STR-03）
                np.save(os.environ['FAKE_IN'], img[0, 0].astype(np.uint16))
                h, w = img.shape[:2]
                out = np.zeros((h * outscale, w * outscale, 3),
                               dtype=self.out_dtype)
                # 标记值要对该 dtype 合法（uint8 时 65535 会溢出）
                peak = np.iinfo(self.out_dtype).max
                out[..., 0] = 1
                out[..., 2] = peak
                return out, None
    """), encoding="utf-8")
    return str(pkg)


def _run_script(tmp_path, in_frames, out_dtype="uint16"):
    pkg = _fake_pkg(tmp_path)
    script = build_stream_ai_script(
        IN_W, IN_H, SCALE, "dummy.pth", "realesr-general-x4v3", 0)
    sp = tmp_path / "ai.py"
    sp.write_text(script, encoding="utf-8")
    rec = tmp_path / "model_in.npy"
    env = dict(os.environ)
    env["PYTHONPATH"] = pkg + os.pathsep + env.get("PYTHONPATH", "")
    env["FAKE_IN"] = str(rec)
    if out_dtype == "uint8":
        # 让假模型返回 uint8，模拟 realesrgan 对暗帧的 8bit 分支
        (tmp_path / "fakepkg" / "realesrgan" / "model.py").write_text(
            (tmp_path / "fakepkg" / "realesrgan" / "model.py")
            .read_text(encoding="utf-8")
            .replace("self.out_dtype = np.uint16", "self.out_dtype = np.uint8"),
            encoding="utf-8")
    p = subprocess.run([sys.executable, str(sp)],
                       input=b"".join(in_frames),
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                       env=env, timeout=120)
    return p, (str(rec) if rec.exists() else None)


def _frame(r=1000, g=2000, b=30000):
    import numpy as np
    a = np.zeros((IN_H, IN_W, 3), dtype=np.uint16)
    a[..., 0], a[..., 1], a[..., 2] = r, g, b
    return np.ascontiguousarray(a).tobytes()


def test_stdout_is_pure_binary_str01(tmp_path):
    """第三方 print 必须去 stderr，stdout 只能有帧字节。"""
    p, _ = _run_script(tmp_path, [_frame() for _ in range(3)])
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")[:300]
    assert len(p.stdout) == 3 * OUT_BYTES, (
        f"stdout 字节数异常: {len(p.stdout)} != {3 * OUT_BYTES}")
    err = p.stderr.decode("utf-8", "replace")
    assert "Input is a 16-bit image" in err or "Tile 1/1" in err, \
        "第三方 print 应出现在 stderr（证明被重定向了）"
    assert "FRAMES=3" in err


def test_model_receives_bgr_str03(tmp_path):
    """管道是 RGB，模型必须收到 BGR（首像素 R/B 互换）。"""
    p, rec = _run_script(tmp_path, [_frame(r=1111, g=2222, b=3333)])
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")[:300]
    import numpy as np
    got = np.load(rec)
    # 送入时 (r=1111, b=3333)；模型应看到 (3333, 2222, 1111)
    assert int(got[0]) == 3333, f"模型收到的第0通道={got[0]}，应为 B=3333"
    assert int(got[2]) == 1111, f"模型收到的第2通道={got[2]}，应为 R=1111"


def test_dark_uint8_output_is_rescaled_str02(tmp_path):
    """模型返回 uint8（暗帧误判）时，仍必须按 uint16 字节数写出。"""
    p, _ = _run_script(tmp_path, [_frame(r=1, g=2, b=3)], out_dtype="uint8")
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")[:400]
    assert len(p.stdout) == OUT_BYTES, (
        f"暗帧写出字节数错误: {len(p.stdout)} != {OUT_BYTES}（会破坏帧边界）")
    err = p.stderr.decode("utf-8", "replace")
    assert "rescaled to uint16" in err, "应对 uint8 返回做 16bit 规整并留痕"


def test_frame_alignment_across_mixed_frames(tmp_path):
    """明暗交替多帧：总字节必须严格等于 帧数 × 每帧字节。"""
    frames = [_frame(r=50000, g=1000, b=200),
              _frame(r=1, g=2, b=3),
              _frame(r=40000, g=30000, b=20000)]
    p, _ = _run_script(tmp_path, frames)
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")[:300]
    assert len(p.stdout) == 3 * OUT_BYTES
    assert "FRAMES=3" in p.stderr.decode("utf-8", "replace")
