"""B7：色彩探测必须能从中间产物回退到原始输入，否则成品偏色。

`stage7` 的 FIX-P7-3 把色彩探测对象从 `args.input`（原始）改成了
`input_source`（上游产物），理由是"转换应基于实际编码的视频"。
这个理由在中间产物**保留**色彩元数据时成立——但实测不成立：

```
原始输入 real2s.mov       -> ('limited', 'BT709', True,  True)
中间产物 step6_grain.mkv  -> ('limited', 'NTSC',  False, False)
```

中间产物是 stage1-6 用 ffv1 生成的 mkv，**丢失了 color_primaries**。
于是 stage7 判定"未知"，按 v7.32 §3.10.3 默认 NTSC/BT.601，执行

```
STAGE7_COLORSPACE  BT.601(NTSC)->BT.709 zscale: matrixin=smpte170m
```

——把本来就是 BT.709 的素材当成 601 转了一遍，**成品整体偏紫**
（对比图肉眼可见）。

修法：探测 `input_source` 后，若 range/primaries 任一未知，
回退探测 `args.input`；只有两者都拿不到才用保守默认。
"""

import os
import shutil
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_stage7():
    vstages = os.path.join(ROOT, "video_restoration_pipeline", "stages")
    if vstages not in sys.path:
        sys.path.insert(0, vstages)
    import stage7_encode as s7  # noqa: E402
    return s7


def test_original_input_reports_bt709():
    """真实素材的原始输入应能读出 BT709 + limited。"""
    src = r"E:/codex/_repair_r4_20261013/real_run/input/real2s.mov"
    if not os.path.exists(src):
        pytest.skip("缺少测试素材")
    s7 = _load_stage7()
    rng, std, rk, pk = s7._get_source_color_info(src)
    assert rk and pk, "原始输入必须能读出 range/primaries"
    assert std == "BT709", f"源是 bt709，实际判为 {std}"


def test_intermediate_loses_metadata():
    """反证：中间产物确实读不到 primaries（这是必须回退的前提）。"""
    import glob
    cands = glob.glob(
        r"E:/codex/_repair_r4_20261013/real_run/temp/*step[56]_*")
    if not cands:
        pytest.skip("缺少中间产物样本")
    s7 = _load_stage7()
    _rng, _std, _rk, pk = s7._get_source_color_info(cands[0])
    assert pk is False, (
        "本测试的前提是中间产物丢元数据；若 ffv1 已能保留，"
        "说明上游修好了，可考虑简化回退逻辑")


def test_fallback_logic_picks_original_when_intermediate_unknown():
    """回退规则：未读到的项用原始输入补齐，已读到的不覆盖。"""
    # 复刻 stage7 的回退逻辑
    def merge(mid, orig):
        _mr, _ms, mrk, mpk = mid
        _orr, _os_, ork, opk = orig
        rng, std, rk, pk = _mr, _ms, mrk, mpk
        if not (rk and pk):
            if not rk and ork:
                rng, rk = _orr, True
            if not pk and opk:
                std, pk = _os_, True
        return rng, std, rk, pk

    # 中间产物全丢 → 全部用原始
    assert merge(("limited", "NTSC", False, False),
                 ("limited", "BT709", True, True)) == \
        ("limited", "BT709", True, True)
    # 中间产物已知 → 不被原始覆盖
    assert merge(("full", "PAL", True, True),
                 ("limited", "BT709", True, True)) == \
        ("full", "PAL", True, True)
    # 原始也未知 → 保持保守默认
    assert merge(("limited", "NTSC", False, False),
                 ("limited", "NTSC", False, False)) == \
        ("limited", "NTSC", False, False)


def test_stage7_source_has_fallback():
    """静态守卫：stage7 必须保留回退分支。"""
    s7 = _load_stage7()
    src = open(s7.__file__, encoding="utf-8").read()
    assert "STAGE7_COLOR_FALLBACK_ORIGINAL" in src, \
        "缺少色彩回退日志 —— 中间产物丢元数据时会再次偏色"
