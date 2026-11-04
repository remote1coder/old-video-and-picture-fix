# -*- coding: utf-8 -*-
"""第十八轮 P0-C / R17-04：TimelinePlan 与人工 FPS 的身份贯通。

审核点名的两处：

1. **缺身份旧计划仍被接受**。`read_plan` 原来只比对"双方都有值"的项：

```python
if _want and got.get(_k) and str(got[_k]) != str(_want):
    return None
```

于是**根本没有 identity 字段的旧计划**被判成"没有冲突"。同 job_id 复用 temp
时，上一轮的"源已验证"会被当成本次结论（审核探针确认）。

2. **`fps_interpret` 不在 CORE_QUALITY_PARAMS**。固定输入下 25 / 30000/1001 /
50 得到**同一个 param_hash**，于是"改了帧率解释却复用旧中间产物"在文件名
层面看不出来。

本轮修法：
- `read_plan`：调用方期望了某维度 → 计划**必须**有且相等；缺项即拒绝。
- `fps_interpret` 进入 `CORE_QUALITY_PARAMS`。

> 审核同时指出"更完整的 stage identity 能补偿一部分，因此不能据此断言所有
> resume 都必然误复用"。所以本轮**保留**既有的 stage identity 补偿，只是不再
> 依赖补偿来掩盖缺口；四处身份（文件名 hash、执行侧车、TimelinePlan、
> 最终 marker）的统一审查仍未完成。
"""
import json
import os
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import timeline_plan as tp  # noqa: E402
import checkpoint as ck  # noqa: E402


def _write_plan(temp_dir, job_id, identity=None, schema=None):
    doc = {"schema": schema or tp.SCHEMA, "job_id": job_id,
           "source_timeline": "cfr"}
    if identity is not None:
        doc["identity"] = identity
    with open(os.path.join(str(temp_dir), f"{job_id}_timeline_plan.json"),
              "w", encoding="utf-8") as f:
        json.dump(doc, f)


IDENT = {"source_input": "abc123", "param_hash": "ph1", "attempt_id": "a-1"}


def test_R17_04_plan_without_identity_is_rejected(tmp_path):
    """**核心缺陷**：旧计划没有 identity 字段 → 必须拒绝，不能当"无冲突"。"""
    _write_plan(tmp_path, "job1", identity=None)
    assert tp.read_plan(str(tmp_path), "job1", identity=IDENT) is None


def test_R17_04_plan_missing_one_expected_key_is_rejected(tmp_path):
    """期望了 attempt_id，计划里却没有 → 不可比，必须拒绝。"""
    _write_plan(tmp_path, "job2",
                identity={"source_input": "abc123", "param_hash": "ph1"})
    assert tp.read_plan(str(tmp_path), "job2", identity=IDENT) is None


def test_R17_04_plan_with_mismatched_identity_is_rejected(tmp_path):
    """身份对不上 → 拒绝（原有行为不能退化）。"""
    _write_plan(tmp_path, "job3", identity=dict(IDENT, param_hash="OTHER"))
    assert tp.read_plan(str(tmp_path), "job3", identity=IDENT) is None


def test_R17_04_matching_plan_is_accepted(tmp_path):
    """完全对得上 → 接受（防止修过头把正常路径也拒了）。"""
    _write_plan(tmp_path, "job4", identity=dict(IDENT))
    got = tp.read_plan(str(tmp_path), "job4", identity=IDENT)
    assert got is not None and got.get("job_id") == "job4"


def test_R17_04_caller_without_identity_still_reads(tmp_path):
    """调用方没给期望身份时保持原语义（不是所有入口都有期望）。"""
    _write_plan(tmp_path, "job5", identity=None)
    assert tp.read_plan(str(tmp_path), "job5") is not None


def test_R17_04_wrong_schema_still_rejected(tmp_path):
    _write_plan(tmp_path, "job6", identity=dict(IDENT), schema="other-schema")
    assert tp.read_plan(str(tmp_path), "job6", identity=IDENT) is None


# ------------------------------------------------ param_hash 覆盖 fps_interpret

def test_R17_04_fps_interpret_changes_param_hash(tmp_path):
    """25 / 30000-1001 / 50 必须产出**不同**的 param_hash。

    修复前它们完全一样，于是"改了帧率解释"在文件名层面无法体现。
    """
    src = tmp_path / "in.mp4"
    src.write_bytes(b"fake-media-bytes")

    base = {"deinterlace": "QTGMC"}
    h25 = ck.compute_param_hash(str(src), dict(base, fps_interpret="25"))
    h2997 = ck.compute_param_hash(str(src), dict(base, fps_interpret="30000/1001"))
    h50 = ck.compute_param_hash(str(src), dict(base, fps_interpret="50"))
    hnone = ck.compute_param_hash(str(src), dict(base))

    assert len({h25, h2997, h50, hnone}) == 4, \
        f"fps_interpret 未进入身份: {[h25, h2997, h50, hnone]}"


def test_R17_04_fps_interpret_is_in_core_quality_params():
    """结构守卫：键必须在白名单里，否则 compute_param_hash 会忽略它。"""
    assert "fps_interpret" in ck.CORE_QUALITY_PARAMS
