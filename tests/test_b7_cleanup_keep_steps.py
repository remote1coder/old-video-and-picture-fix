"""B7 附带缺陷：cleanup_old_intermediates 在"同一步多产物"时误删上一步。

实测（2026-10-19，2 秒 1080i 真实素材）：

```
03:26:24 STAGE4_DONE
03:26:25 STAGE5_ORDER_ERROR  AI upscale enabled but no stage3 output found
03:26:25 STAGE_FAILED        Stage 5 failed: 82
03:26:31 STALE_INTERMEDIATES_PURGED  Purged stale intermediates for fresh run
→ 整链从 stage0 重跑
```

根因（`checkpoint.py` 旧实现）：

```python
steps_to_keep = {s for s, _ in intermediates[:keep_steps]}
```

它假设**每个 step 只有一个产物**。一旦某一步产出多个文件
（stage4 在分窗口/多变体时会），保留名额被同一步占满：

```
intermediates = [(4,a),(4,b),(4,c),(3,d),(2,e),(1,f)]
keep_steps=2, current_step=4
  → steps_to_keep = {4}
  → step3/2/1 全被删（step < current_step 且不在 {4}）
```

下游 stage5 需要 step3/step4 作为输入，于是找不到 → rc=82 →
整链重试。**等于每跑一次都在 stage5 崩一次、再从头重来**。

修法：保留集合按 **step 号范围**算
（`current_step - keep_steps + 1 .. current_step` 内的所有 step）。
"""

import os

import conftest  # noqa: F401  (sys.path bootstrap)

import checkpoint as ck  # noqa: E402


def _touch(d, names):
    for n in names:
        with open(os.path.join(str(d), n), "wb") as fh:
            fh.write(b"x" * 10)


def test_multiple_artifacts_in_same_step_do_not_evict_previous_step(tmp_path):
    """反例：step4 三个变体 → step3 必须保留。"""
    d = tmp_path
    _touch(d, [
        "job_001_x_step4_postprocess_p1.mkv",
        "job_001_x_step4_postprocess_p2.mkv",
        "job_001_x_step4_postprocess_p3.mkv",
        "job_001_x_step3_upscale_p1.mkv",
        "job_001_x_step2_traditional.mkv",
        "job_001_x_step1_preprocess_p1.mkv",
    ])
    ck.cleanup_old_intermediates(str(d), "job_001", keep_steps=2,
                                 current_step=4)
    left = sorted(os.listdir(str(d)))
    assert any("step3" in f for f in left), \
        "step3 被误删 —— stage5 将找不到 stage3 产物（B7 实测症状）"
    assert any("step4" in f for f in left)
    # step2 不在 [3,4] 范围 → 应删除
    assert not any("step2" in f for f in left)
    assert not any("step1" in f for f in left)


def test_keep_steps_3_keeps_three_step_numbers(tmp_path):
    d = tmp_path
    _touch(d, [
        "job_001_x_step5_a.mkv",
        "job_001_x_step4_a.mkv",
        "job_001_x_step3_a.mkv",
        "job_001_x_step2_a.mkv",
        "job_001_x_step1_a.mkv",
    ])
    ck.cleanup_old_intermediates(str(d), "job_001", keep_steps=3,
                                 current_step=5)
    left = sorted(os.listdir(str(d)))
    assert any("step5" in f for f in left)
    assert any("step4" in f for f in left)
    assert any("step3" in f for f in left)
    assert not any("step2" in f for f in left)


def test_current_step_none_still_uses_file_count(tmp_path):
    """向后兼容：current_step=None 时仍按"文件个数"保留（job 启动清场）。"""
    d = tmp_path
    _touch(d, [
        "job_001_x_step4_a.mkv",
        "job_001_x_step3_a.mkv",
        "job_001_x_step2_a.mkv",
        "job_001_x_step1_a.mkv",
    ])
    ck.cleanup_old_intermediates(str(d), "job_001", keep_steps=2,
                                 current_step=None)
    left = sorted(os.listdir(str(d)))
    assert len(left) == 2, f"应只保留 2 个，实际 {left}"
