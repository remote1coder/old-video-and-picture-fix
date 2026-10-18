"""B7 关键缺陷：stage5 的 step3 存在性门禁漏掉 `_forced_upscale` 命名。

stage3 在 **force 模式**下会把产物改名：

```python
# stage3_ai_upscale.py 「V-P-031 fix: force_allow suffix」
if combined_banned and force_allow:
    inter_name = f"{job_id}_{stem}_step3_forced_upscale_p{param_hash}"
```

三个下游消费者里，**两个已修、一个漏了**：

| 消费者 | 是否匹配 `_forced_upscale` |
|---|---|
| `stage4_postprocess.py` | ✅ 同时匹配两种（注释 FIX-P8-2） |
| `stage7_encode.py` | ✅ 媒体白名单含两种 |
| **`stage5_ai_face.py`** | ❌ **只匹配 `step3_upscale*`** |

后果：1080p 素材默认被判 `ai_upscale_banned`，用户必须用
`force_allow_banned_upscale: true` 才能超分 → 产物名变成
`step3_forced_upscale` → stage5 找不到 → rc=82 →
`STALE_INTERMEDIATES_PURGED` → 整链从 stage0 重跑。
**每跑一次崩一次，永远到不了 stage6/7。**

实测（2026-10-19，2 秒 1080i）：
```
03:26:24 STAGE4_DONE
03:26:25 STAGE5_ORDER_ERROR  AI upscale enabled but no stage3 output found
03:26:31 STALE_INTERMEDIATES_PURGED
```
"""

import glob
import os

import conftest  # noqa: F401  (sys.path bootstrap)

import checkpoint as ck  # noqa: E402


def _touch(d, names):
    for n in names:
        with open(os.path.join(str(d), n), "wb") as fh:
            fh.write(b"x" * 10)


def _find_stage3(temp_dir, job_id, stem):
    """复刻 stage5 修好后的双 pattern 查找。"""
    patterns = [
        f"{job_id}_{stem}_step3_upscale*",
        f"{job_id}_{stem}_step3_forced_upscale*",
    ]
    return ck.select_media_candidates(
        [p for pat in patterns
         for p in glob.glob(os.path.join(str(temp_dir), pat))])


def test_plain_name_is_found(tmp_path):
    _touch(tmp_path, ["job_001_x_step3_upscale_pabc.mkv"])
    assert _find_stage3(tmp_path, "job_001", "x")


def test_forced_name_is_found(tmp_path):
    """反例：force 模式下产物叫 step3_forced_upscale —— 必须也能找到。"""
    _touch(tmp_path, ["job_001_x_step3_forced_upscale_pabc.mkv"])
    assert _find_stage3(tmp_path, "job_001", "x"), \
        "force 模式产物找不到 —— stage5 会报 STAGE5_ORDER_ERROR"


def test_old_single_pattern_would_miss_forced(tmp_path):
    """反证：只匹配 `step3_upscale*` 时确实找不到 forced 产物。"""
    _touch(tmp_path, ["job_001_x_step3_forced_upscale_pabc.mkv"])
    old = ck.select_media_candidates(
        glob.glob(os.path.join(str(tmp_path), "job_001_x_step3_upscale*")))
    assert not old, "旧 pattern 本应匹配不到，说明测试素材不对"


def test_sidecars_are_filtered(tmp_path):
    """侧车仍必须被过滤（R3-01 的既有保证不能被破坏）。"""
    _touch(tmp_path, [
        "job_001_x_step3_forced_upscale_pabc.mkv",
        "job_001_x_step3_forced_upscale_pabc.mkv.identity.json",
    ])
    got = _find_stage3(tmp_path, "job_001", "x")
    assert len(got) == 1 and str(got[0]).endswith(".mkv")
