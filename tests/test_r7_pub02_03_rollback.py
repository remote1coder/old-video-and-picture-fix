"""PUB-02/03（第七轮）：回滚必须报可核验结果；失败 attempt 不得覆盖有效代状态。

**PUB-02 旧行为**（静态确认，`stage6_encode._rollback_publish`）：

```python
except OSError as e:
    if logger:
        logger.error("STAGE6_ROLLBACK_FAIL", ...)   # 逐项吞掉，返回 None
...
_rollback_publish(snapshot, logger)
logger.warning("STAGE6_PUBLISH_ROLLBACK",
               "Previous generation restored; ...")   # 无条件宣布成功
```

用户会收到"已完整恢复"的结论，而磁盘上可能只恢复了一部分。

**PUB-03 旧行为**：

```python
_fail(logger, status_path, ...)   # 无条件把固定 status 写成 FAILED
```

`status_path = publish_output_path + ".status.json"` 指向**当前有效代**。
回滚成功后旧成品其实完好，却被打上 FAILED —— 下次运行会误判为需要重跑，
甚至触发清理删除它。

新行为：
* `_rollback_publish` 返回 `{restored, removed, failed, impossible, fully_restored}`；
* 完全恢复 → `preserve_valid_status=True`，固定 status 不动，失败记 attempt sidecar；
* 未完全恢复 → `RECOVERY_REQUIRED`（此时覆写固定 status 是正确的）。
"""

import json
import os
import sys

import conftest  # noqa: F401  (sys.path bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PSTAGES = os.path.join(ROOT, "photo_restoration_pipeline", "stages")
if PSTAGES not in sys.path:
    sys.path.insert(0, PSTAGES)

import stage6_encode as s6  # noqa: E402


def _ok_backup_dir(tmp_path, names):
    """造一套"每个目标都有可用备份"的 snapshot。"""
    snap = {}
    for n in names:
        target = tmp_path / n
        backup = tmp_path / f"{n}.pubbak"
        target.write_bytes(b"NEW-GARBAGE")
        backup.write_bytes(b"OLD-GOOD")
        snap[str(target)] = {"state": s6.BACKUP_OK, "backup": str(backup)}
    return snap


def test_rollback_returns_structured_result(tmp_path):
    """回滚必须返回可核验结构，而不是 None。"""
    snap = _ok_backup_dir(tmp_path, ["a.avif", "a.tif"])
    res = s6._rollback_publish(snap, None)
    assert isinstance(res, dict)
    for k in ("restored", "removed", "failed", "impossible", "fully_restored"):
        assert k in res, f"缺少结果字段 {k}"
    assert res["fully_restored"] is True
    assert len(res["restored"]) == 2
    # 内容确实回到旧值
    assert (tmp_path / "a.avif").read_bytes() == b"OLD-GOOD"
    assert (tmp_path / "a.tif").read_bytes() == b"OLD-GOOD"


def test_rollback_reports_partial_failure(tmp_path):
    """某个目标恢复失败 → fully_restored 必须为 False（不能假报成功）。"""
    target = tmp_path / "a.avif"
    target.write_bytes(b"NEW")
    # backup 指向不存在的文件 → 走 OSError 分支
    snap = {str(target): {"state": s6.BACKUP_OK,
                          "backup": str(tmp_path / "missing.pubbak")}}
    res = s6._rollback_publish(snap, None)
    assert res["fully_restored"] is False
    assert res["failed"], "应记录 failed 明细"


def test_rollback_marks_backup_failed_as_impossible(tmp_path):
    """备份本身失败 → impossible，且**不能删除**目标（删了就销毁旧产物）。"""
    target = tmp_path / "a.tif"
    target.write_bytes(b"UNTOUCHED")
    snap = {str(target): {"state": s6.BACKUP_FAILED, "backup": None}}
    res = s6._rollback_publish(snap, None)
    assert res["impossible"] == [str(target)]
    assert res["fully_restored"] is False
    assert target.read_bytes() == b"UNTOUCHED", "备份失败时不该动目标"


def test_rollback_removes_freshly_created_target(tmp_path):
    """提交前不存在、本次新建的目标 → 回滚时删除。"""
    target = tmp_path / "fresh.avif"
    target.write_bytes(b"NEW")
    snap = {str(target): {"state": s6.BACKUP_ABSENT, "backup": None}}
    res = s6._rollback_publish(snap, None)
    assert not target.exists()
    assert res["removed"] == [str(target)]
    assert res["fully_restored"] is True


def test_fail_preserves_valid_status_sidecar(tmp_path):
    """PUB-03：preserve_valid_status=True 时，固定 status 不被覆写成 FAILED。"""
    status = tmp_path / "restored.avif.status.json"
    status.write_text(json.dumps({"status": "SUCCESS", "stage": 6}),
                      encoding="utf-8")

    s6._fail(None, str(status), "contract", "boom",
             paths=[], preserve_valid_status=True)

    after = json.loads(status.read_text(encoding="utf-8"))
    assert after["status"] == "SUCCESS", \
        "有效成品的 status 被失败 attempt 覆写了 —— PUB-03 退化"
    # 失败记录应落在 attempt 级 sidecar
    attempts = list(tmp_path.glob("*.attempt-*.json"))
    assert attempts, "缺少 attempt 级失败记录"
    rec = json.loads(attempts[0].read_text(encoding="utf-8"))
    assert rec["status"] == "FAILED"
    assert rec["failed_operation"] == "contract"


def test_fail_writes_separate_recovery_record(tmp_path):
    """R14-02：回滚未完成 → 写**独立恢复记录**，不覆写固定 status。

    旧预期是"这次覆写是对的"。但它让一次失败的尝试改写了上一代的
    磁盘事实；正式状态应当只随完整代提交。现在：
    * 固定 status 原样保留；
    * `<status>.recovery.json` 记录 RECOVERY_REQUIRED + 回滚明细；
    * `has_unresolved_recovery()` 供消费方判断是否可用。
    """
    status = tmp_path / "restored.avif.status.json"
    status.write_text(json.dumps({"status": "SUCCESS"}), encoding="utf-8")
    before = status.read_bytes()

    detail = {"restored": [], "removed": [], "impossible": ["x"],
              "failed": [{"path": "y", "error": "EACCES"}],
              "fully_restored": False}
    s6._fail(None, str(status), "status", "io", paths=[],
             recovery_required=True, rollback_detail=detail)

    assert status.read_bytes() == before, "固定 status 不得被改写"
    assert json.loads(status.read_text(encoding="utf-8"))["status"] == "SUCCESS"

    rec_path = s6.recovery_record_path(str(status))
    with open(rec_path, encoding="utf-8") as f:
        after = json.load(f)
    assert after["status"] == "RECOVERY_REQUIRED"
    assert after["prior_status"] == "SUCCESS"
    assert after["rollback"]["fully_restored"] is False
    assert after["rollback"]["failed"][0]["error"] == "EACCES"
    assert s6.has_unresolved_recovery(str(status)) is True


def test_caller_no_longer_claims_unconditional_restore():
    """静态守卫：调用方不能再用"无条件宣布恢复成功"的措辞。"""
    src = open(s6.__file__, encoding="utf-8").read()
    assert "Previous generation restored; no mixed-generation" not in src, \
        "仍有无条件宣布恢复成功的旧文案"
    assert "_rb[\"fully_restored\"]" in src, "调用方未按回滚结果分流"
    assert "STAGE6_RECOVERY_REQUIRED" in src
