"""色彩保真探测：源与成品的 RGB 均值对比（视频链 stage7 / 照片链 stage6 共用）。

**为什么需要它**

2026-10-20 的照片链 G6 实跑挖出一个缺陷：8-bit 输入的成品是**纯黑图**
（mean 0.03），而 rc=0、契约、CICP、`avifdec --info` 头部核验**全部正常**。
对照组（普通 sRGB JPEG）同样全黑 —— 说明这不是宽色域特有问题。
根因是 G'MIC `-o X.tif,uint16` 只改容器位深不拉伸值域，R16 的修法
（`media_io` 逐通道自适应读取）覆盖不到 G'MIC 自己。

那次能抓到它，靠的是人工拿 `avifdec` 解出成品算 mean。本模块把这一步
变成**发布前的例行守卫**，让"整体崩坏"（黑/白/内容丢失）不可能再带着
rc=0 流出去。

**这个阈值抓什么、不抓什么**

默认 60.0（0-255 的 RGB 通道均值最大绝对差）是**刻意宽松**的。实测三类：

| 场景 | 源 mean | 成品 mean | 最大通道差 |
|---|---|---|---|
| 纯黑成品 | 116.65 | 0.03 | **116.6** → 拦 |
| ProPhoto → sRGB 转换 | 116.65 | 143.30 | **26.7** → 放行 |
| 普通 sRGB 正常处理 | 107.27 | 106.30 | **1.0** → 放行 |

也就是说它专门抓"整体崩坏"，**不抓细微色偏**。细微色偏（第六轮实测的
R/B 比累积漂移约 5%）需要针对素材标定更严的值，那属于另一个议题。
把阈值定严会让宽色域素材和 AI 处理后的素材频繁误报，反而没人看日志。

**探测方法**

- 采样点按**时长比例**（10% / 50% / 90%），不用固定秒数 —— 几秒的短片在
  5/30/90 秒一个点都取不到，实测会让探测静默返回 None、守卫形同虚设。
- 像素用 `-f rawvideo -pix_fmt rgb24` 直接拿，不经过 PIL/cv2
  （R16 病族：读法不对会让数值完全错）。
- 探测失败返回 None，由调用方记 `*_SKIPPED` —— 不假装通过，也不阻断。
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional

#: 默认阈值：0-255 的 RGB 通道均值最大绝对差。见模块 docstring 的标定表。
COLOR_DRIFT_DEFAULT_THRESHOLD = 60.0

#: 采样位置，按时长比例取。
COLOR_PROBE_FRACTIONS = (0.1, 0.5, 0.9)


def probe_duration(path: str, logger=None) -> float:
    """读容器时长（秒）。失败返回 0.0。"""
    try:
        r = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
            capture_output=True, text=True, errors="replace", timeout=60,
            shell=False)
        if r.returncode == 0 and (r.stdout or "").strip():
            return float(r.stdout.strip())
    except Exception as e:
        if logger:
            logger.warning("COLOR_PROBE_ERROR", f"duration probe failed: {e}")
    return 0.0


def probe_rgb_mean(path: str, logger=None) -> Optional[List[float]]:
    """按时长比例抽 3 帧，算 RGB 三通道均值（0-255）。

    失败（文件读不出 / 一个点都没抽到）返回 None，由调用方决定放行。
    """
    try:
        import numpy as np
        dur = probe_duration(path, logger)
        offsets = ([f * dur for f in COLOR_PROBE_FRACTIONS] if dur > 0
                   else [0.0])
        per_frame: List[List[float]] = []
        for off in offsets:
            cmd = ["ffmpeg", "-hide_banner", "-v", "error",
                   "-ss", f"{max(0.0, off):.3f}", "-i", str(path),
                   "-frames:v", "1", "-vf", "scale=160:-2",
                   "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
            try:
                r = subprocess.run(cmd, capture_output=True, timeout=120,
                                   shell=False)
            except Exception:
                continue
            if r.returncode != 0 or not r.stdout:
                continue
            buf = np.frombuffer(r.stdout, dtype=np.uint8)
            n = (buf.size // 3) * 3
            if n <= 0:
                continue
            px = buf[:n].reshape(-1, 3).astype(np.float64)
            per_frame.append([float(px[:, i].mean()) for i in range(3)])
        if not per_frame:
            return None
        k = len(per_frame)
        return [sum(f[i] for f in per_frame) / k for i in range(3)]
    except Exception as e:
        if logger:
            logger.warning("COLOR_PROBE_ERROR", str(e))
        return None


def check_color_drift(source_path: str, output_path: str,
                      threshold: float = COLOR_DRIFT_DEFAULT_THRESHOLD,
                      enforce: bool = False,
                      logger=None,
                      src_mean: Optional[List[float]] = None,
                      out_mean: Optional[List[float]] = None) -> Dict[str, Any]:
    """比对源与成品的 RGB 均值，返回**可直接核验**的结论。

    两侧的探测结果可以外部传入（调用方已经探过，避免重复起 ffmpeg）。
    返回 dict：

    ```
    {
      "status":   "ok" | "drift" | "skipped",
      "delta":    最大通道绝对差（skipped 时为 None）,
      "threshold":…, "enforce":…,
      "src_mean": […], "out_mean": […],
    }
    ```

    只判"是不是超了阈值"，**不决定**调用方该失败还是只告警 —— 那是
    `enforce` 的事，由调用方读 `status` 后自行处理。
    """
    src = src_mean if src_mean is not None else probe_rgb_mean(source_path, logger)
    out = out_mean if out_mean is not None else probe_rgb_mean(output_path, logger)
    if not src or not out or any(v <= 0 for v in list(src) + list(out)):
        return {"status": "skipped", "delta": None, "threshold": threshold,
                "enforce": enforce, "src_mean": src, "out_mean": out}
    delta = max(abs(src[i] - out[i]) for i in range(3))
    return {
        "status": "drift" if delta > threshold else "ok",
        "delta": delta,
        "threshold": threshold,
        "enforce": enforce,
        "src_mean": [round(v, 2) for v in src],
        "out_mean": [round(v, 2) for v in out],
        # R/B 比也记下来：整体亮度没变但色调翻了，mean 差值可能很小
        "src_r_over_b": round(src[0] / src[2], 4) if src[2] else None,
        "out_r_over_b": round(out[0] / out[2], 4) if out[2] else None,
    }
