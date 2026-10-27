# -*- coding: utf-8 -*-
"""零落盘流式 AI 脚本的数据契约回归。

- **STR-01**：第三方 `print` 不能污染二进制 stdout。
  本地 realesrgan 的 utils.py 里有 `print('\tInput is a 16-bit image')`
  与 `print('\tTile i/n')`，直接写 stdout 会与帧数据混流。
- **STR-03**：ffmpeg 用 rgb48le（RGB），RealESRGAN 的 enhance() 收/返回 BGR。
  两次方向错误会部分抵消，所以只比"均值排序"证明不了问题 —— 这里查**逐通道**。
- **R12-09**（本轮核心）：RealESRGAN 用 `np.max(img) > 256` 猜位深。
  暗的 uint16 帧（峰值 ≤256）被当成 8bit **除以 255**，模型看到的输入放大
  257 倍；事后把 uint8 输出乘 257 **救不了已经发生的推理**。
  修法是绕开那段猜测：固定按 65535 归一、按 65535 量化。

本测试用**假模型**跑真实生成的脚本，检查模型实际收到的张量值与进出字节，
不依赖 GPU 与权重。
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
    # 假模型：故意 print 到文本 stdout，并把 pre_process 收到的张量存盘
    (rg / "model.py").write_text(textwrap.dedent("""
        import os, sys
        import numpy as np

        class RealESRGANer:
            \"\"\"三段式替身：pre_process / process / post_process。

            R12-09 之后模板不再调 enhance()，而是直接调这三段，
            所以替身必须提供同样的接口。
            \"\"\"

            def __init__(self, scale=None, model_path=None, model=None,
                         half=False, gpu_id=0):
                self.scale = scale
                self.half = half
                self.tile_size = 0          # 0 = 不走 tile_process
                self.output = None
                # 故意的第三方文本输出 —— STR-01 的复现条件
                print('\\tInput is a 16-bit image')
                print('\\tTile 1/1')
                sys.stdout.flush()

            def pre_process(self, img):
                # img 是 RGB float32 [0,1] HWC —— 正是"模型看到的东西"
                np.save(os.environ['FAKE_IN'], img[0, 0].astype(np.float64))
                self.img = img

            def process(self):
                s = self.scale
                out = np.repeat(np.repeat(self.img, s, axis=0), s, axis=1)
                out = out.copy()
                # 固定图案，便于验证 RGB->BGR 的写出顺序
                out[..., 0] = 1.0
                out[..., 1] = 0.5
                out[..., 2] = 0.0
                self.output = out

            def tile_process(self):
                self.process()

            def post_process(self):
                return self.output
    """), encoding="utf-8")
    return str(pkg)


def _run_script(tmp_path, in_frames):
    pkg = _fake_pkg(tmp_path)
    script = build_stream_ai_script(
        IN_W, IN_H, SCALE, "dummy.pth", "realesr-general-x4v3", 0)
    sp = tmp_path / "ai.py"
    sp.write_text(script, encoding="utf-8")
    rec = tmp_path / "model_in.npy"
    env = dict(os.environ)
    env["PYTHONPATH"] = pkg + os.pathsep + env.get("PYTHONPATH", "")
    env["FAKE_IN"] = str(rec)
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


def test_model_receives_original_rgb_order_str03(tmp_path):
    """管道是 RGB；模型经 BGR→RGB 往返后应收到**原始 RGB 顺序**。

    这一条能抓住"少翻一次"和"多翻一次"两种错误 —— 两次翻转会部分抵消，
    所以只比通道均值排序是证明不了的。
    """
    p, rec = _run_script(tmp_path, [_frame(r=1111, g=2222, b=3333)])
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")[:300]
    import numpy as np
    got = np.load(rec)                      # RGB float [0,1]
    # 反归一化回 16bit 整数便于比较
    px = np.round(got * 65535.0).astype(np.int64)
    assert int(px[0]) == 1111, f"模型收到的 R={px[0]}，应为 1111"
    assert int(px[1]) == 2222, f"模型收到的 G={px[1]}，应为 2222"
    assert int(px[2]) == 3333, f"模型收到的 B={px[2]}，应为 3333"


def test_dark_frame_normalised_by_65535_r12_09(tmp_path):
    """暗帧必须按 **65535** 归一，不能按 255（旧实现放大 257 倍）。"""
    p, rec = _run_script(tmp_path, [_frame(r=3, g=2, b=1)])
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")[:300]
    import numpy as np
    got = np.load(rec)
    # 若按 255 归一：3/255 ≈ 0.01176；按 65535 归一：3/65535 ≈ 4.578e-5
    assert got[0] < 1e-4, f"R 通道归一化错误: {got[0]}（应按 65535）"
    assert abs(float(got[0]) - 3 / 65535.0) < 1e-9, got[0]
    assert abs(float(got[1]) - 2 / 65535.0) < 1e-9, got[1]
    assert abs(float(got[2]) - 1 / 65535.0) < 1e-9, got[2]


def test_bright_and_dark_frames_share_one_scale_r12_09(tmp_path):
    """同一尺度：亮帧与暗帧必须用**同一个分母**。

    旧实现按每帧自身的最大值分别决定 255 / 65535，于是同一段视频里
    明暗帧的归一化尺度不一致 —— 模型的输入亮度会跳变。
    """
    p, rec = _run_script(tmp_path, [_frame(r=60000, g=60000, b=60000),
                                    _frame(r=200, g=200, b=200)])
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")[:300]
    assert len(p.stdout) == 2 * OUT_BYTES
    import numpy as np
    # 记录的是**最后一帧**（暗帧）
    got = np.load(rec)
    assert abs(float(got[0]) - 200 / 65535.0) < 1e-9, \
        f"暗帧未按 65535 归一: {got[0]}"


def test_frame_alignment_across_mixed_frames(tmp_path):
    """明暗交替多帧：总字节必须严格等于 帧数 × 每帧字节。"""
    frames = [_frame(r=50000, g=1000, b=200),
              _frame(r=1, g=2, b=3),
              _frame(r=40000, g=30000, b=20000)]
    p, _ = _run_script(tmp_path, frames)
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")[:300]
    assert len(p.stdout) == 3 * OUT_BYTES
    assert "FRAMES=3" in p.stderr.decode("utf-8", "replace")


def test_output_is_written_as_rgb(tmp_path):
    """模型返回 RGB 图案，写出的字节必须是 RGB 顺序（不是 BGR）。

    替身固定返回 R=1.0 / G=0.5 / B=0.0，所以首像素应写成
    (65535, 32768, 0) 的 RGB 三元组。
    """
    p, _ = _run_script(tmp_path, [_frame()])
    assert p.returncode == 0, p.stderr.decode("utf-8", "replace")[:300]
    import numpy as np
    out = np.frombuffer(p.stdout, dtype=np.uint16).reshape(
        IN_H * SCALE, IN_W * SCALE, 3)
    assert int(out[0, 0, 0]) == 65535, f"R={out[0,0,0]}"
    assert abs(int(out[0, 0, 1]) - 32768) <= 1, f"G={out[0,0,1]}"
    assert int(out[0, 0, 2]) == 0, f"B={out[0,0,2]}"
