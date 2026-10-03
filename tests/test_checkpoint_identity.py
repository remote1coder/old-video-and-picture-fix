"""NEW-V-01: checkpoint 身份哈希在缺少硬依赖 xxhash 时的静默降级。

背景（实测，见 修复清单.md NEW-V-01）：
    shared_modules/checkpoint.py::compute_param_hash 原本用 `except Exception: pass`
    包裹 get_file_xxhash()。xxhash 未安装时该调用抛 ImportError，被一并吞掉，
    于是 file_hash_prefix 恒为 ""、stat_ok 恒为 False，身份退化为
    「文件大小 + mtime + 文件名」。实测同名/同尺寸/同 mtime 但内容不同的两个文件
    会算出同一个 param_hash → 中间产物被错误复用（静默的假缓存命中）。

修复后契约：
    * xxhash 缺失 = 硬依赖缺失 → 快速失败（抛 ImportError，携带安装指引）。
    * 真实 I/O 异常（权限/文件消失/路径不可读）→ 允许降级，但必须打 warning 留痕。
"""

import importlib
import importlib.util
import os
import sys
import tempfile

import pytest

from checkpoint import compute_param_hash  # noqa: E402  (conftest 已注入 sys.path)


class _BlockXxhash:
    """上下文管理器：让 `import xxhash` 在本进程内抛出 ImportError。"""

    def __init__(self) -> None:
        self._finder = None

    def find_module(self, *args, **kwargs):  # legacy API, 兼容旧导入钩子
        return None

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "xxhash" or fullname.startswith("xxhash."):
            raise ImportError("blocked by test: xxhash is unavailable")
        return None

    def __enter__(self):
        sys.modules.pop("xxhash", None)
        self._finder = self
        sys.meta_path.insert(0, self)
        return self

    def __exit__(self, *exc):
        try:
            sys.meta_path.remove(self)
        except ValueError:
            pass
        sys.modules.pop("xxhash", None)
        return False


def _make_twin(tmp_path, payload_a: bytes, payload_b: bytes):
    """造两个同名、同尺寸、同 mtime、内容不同的文件（放在不同子目录）。"""
    d1 = tmp_path / "run1"
    d2 = tmp_path / "run2"
    d1.mkdir()
    d2.mkdir()
    a = d1 / "clip.bin"
    b = d2 / "clip.bin"
    a.write_bytes(payload_a)
    b.write_bytes(payload_b)
    frozen = 1_600_000_000
    os.utime(a, (frozen, frozen))
    os.utime(b, (frozen, frozen))
    return str(a), str(b)


def test_new_v01_content_is_part_of_identity():
    """装了 xxhash 时，内容必须参与身份计算 —— 同名同尺寸同 mtime 不得碰撞。"""
    if importlib.util.find_spec("xxhash") is None:
        pytest.skip("xxhash 未安装，无法验证正常路径（CI 需 pip install xxhash）")

    with tempfile.TemporaryDirectory() as td:
        import pathlib

        a, b = _make_twin(pathlib.Path(td), b"A" * 4096, b"B" * 4096)
        assert compute_param_hash(a, {}) != compute_param_hash(b, {}), (
            "内容未参与身份：同名/同尺寸/同 mtime 的不同文件碰撞成同一个 param_hash"
        )


def test_new_v01_missing_xxhash_fails_loudly_not_silently():
    """缺 xxhash 必须抛错，不得静默产出弱哈希。"""
    with tempfile.TemporaryDirectory() as td:
        import pathlib

        a, b = _make_twin(pathlib.Path(td), b"A" * 4096, b"B" * 4096)

        # 先证明：若按旧逻辑放行，这两个文件会碰撞（这是修复要消灭的行为）。
        weak = None
        with _BlockXxhash():
            try:
                weak = (compute_param_hash(a, {}), compute_param_hash(b, {}))
            except ImportError:
                weak = None
        assert weak is None or weak[0] != weak[1], (
            f"回归失败：缺少 xxhash 时仍产出了身份哈希 {weak}，"
            "同名/同尺寸/同 mtime 的不同文件会碰撞"
        )

        # 再证明：现在的行为是快速失败，且错误信息带安装指引。
        with _BlockXxhash():
            with pytest.raises(ImportError) as excinfo:
                compute_param_hash(a, {})
        assert "xxhash is required" in str(excinfo.value), (
            "错误信息必须指明缺失的依赖，便于直接修复"
        )


def test_new_v01_io_errors_still_degrade_with_a_warning(caplog):
    """真实 I/O 异常（而非缺依赖）仍允许降级，但必须留痕，不能无声无息。"""
    with tempfile.TemporaryDirectory() as td:
        # 目录在 Windows 上无法以文件方式打开 → PermissionError/OSError
        digest = compute_param_hash(td, {})
        assert isinstance(digest, str) and len(digest) == 16
        assert any(
            "degraded" in r.getMessage() for r in caplog.records
        ), "I/O 降级必须打 warning 留痕，不能静默 pass"
