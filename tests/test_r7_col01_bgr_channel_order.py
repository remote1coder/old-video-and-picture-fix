"""COL-01（第七轮）：RealESRGANer 的通道约定核对。

**官方接口约定（本机安装版实测源码）**

```python
# 输入端（img_mode == 'RGB' 分支）
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)      # 把入参当作 BGR
# 输出端
output_img = np.transpose(output_img[[2, 1, 0], :, :], (1, 2, 0))  # RGB→BGR
```

即：**输入 BGR、输出 BGR**。

**旧代码**

```python
img = Image.open(fp).convert('RGB')            # RGB
output, _ = upsampler.enhance(np.array(img))   # 传 RGB（被当 BGR）
Image.fromarray(output).save(...)              # 把返回的 BGR 当 RGB 存
```

两次错配**恰好抵消** → 最终颜色看着是对的，但**模型实际处理的是
R/B 互换的图**。

**实测（纯色块，左右不对称）**

| 对象 | 左半 RGB | 右半 RGB |
|---|---|---|
| 输入 | (220,20,20) 红 | (20,20,220) 蓝 |
| 旧调用 | (219,19,19) 红 | (21,19,223) 蓝 |
| 新调用 | (223,19,21) 红 | (19,19,219) 蓝 |

两者都**没有颠倒**——所以不能把它当成"红蓝对调"的可见 bug，也不能
据此说"模型本身偏色"。这是一个**违反官方约定的接口误用**，修的是
"模型到底收到了什么"，不是"最终颜色对不对"。
"""

import os
import sys

import conftest  # noqa: F401  (sys.path bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VSTAGES = os.path.join(ROOT, "video_restoration_pipeline", "stages")
if VSTAGES not in sys.path:
    sys.path.insert(0, VSTAGES)


def _stage3_src():
    return open(os.path.join(VSTAGES, "stage3_ai_upscale.py"),
                encoding="utf-8").read()


def test_both_callsites_convert_rgb_to_bgr_before_enhance():
    """两个调用点（单窗/批量）都必须显式 RGB→BGR。"""
    src = _stage3_src()
    assert src.count("_rgb[:, :, ::-1].copy()") == 2, \
        "不是两个调用点都做了 RGB→BGR 适配"


def test_both_callsites_convert_output_back_to_rgb():
    """官方返回 BGR，保存前必须转回 RGB。"""
    src = _stage3_src()
    assert src.count("output[:, :, ::-1].copy()") == 2, \
        "输出没有转回 RGB"


def test_no_bare_rgb_enhance_left():
    """不允许再出现"PIL RGB 直传"的旧写法。"""
    src = _stage3_src()
    assert "enhance(\"\n                \"np.array(img), outscale=scale)" not in src
    assert "np.array(img), outscale=scale" not in src, \
        "仍有把 PIL RGB 直传 enhance 的调用"


def test_documented_contract_in_source():
    """源码里要写明官方约定，避免下次又"凭直觉"改回去。"""
    src = _stage3_src()
    assert "COL-01" in src
    assert "输入 BGR" in src or "BGR" in src
