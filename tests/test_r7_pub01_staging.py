"""PUB-01（第七轮）：视频编码必须写 attempt 暂存，失败/取消不碰旧成品。

旧行为（静态确认）：

```python
video_cmd.extend([..., "-y", output_path])          # 直写最终成品
_cancel_guard.watch_outputs(output_path, stage=7)   # delete_output 默认 True
```

`watch()` 的 `delete_output=True` 意味着取消时**删除被监控路径**。
而该路径上可能正是上一次成功发布的成品 → 一次失败的 attempt 能毁掉
已有的好结果。

新行为：
* 字节先写 `<output>.attempt-<pid>-<ms>.partial`；
* 取消守卫只监控这个暂存文件；
* 编码 + 非空 + A/V 同步校验通过后，才 `os.replace` 到最终路径；
* 失败/取消只删暂存，`output_path` 全程未被写。
"""

import os
import subprocess
import sys

import conftest  # noqa: F401  (sys.path bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VSTAGES = os.path.join(ROOT, "video_restoration_pipeline", "stages")
if VSTAGES not in sys.path:
    sys.path.insert(0, VSTAGES)

import stage7_encode as s7  # noqa: E402


def test_stage7_does_not_write_final_path_directly():
    """静态守卫：ffmpeg 输出必须是 _publish_tmp，不能是 output_path。"""
    src = open(s7.__file__, encoding="utf-8").read()
    assert '"-y", _publish_tmp' in src, \
        "ffmpeg 仍在直写最终路径 —— PUB-01 退化"
    assert '"-y", output_path' not in src, \
        "仍有 -y output_path 直写最终成品"


def test_watch_outputs_targets_staging_not_final():
    """静态守卫：取消守卫必须监控暂存文件（注释里引用旧行为不算）。"""
    src = open(s7.__file__, encoding="utf-8").read()
    code_lines = [ln for ln in src.splitlines()
                  if ln.lstrip().startswith(("_cancel_guard", "guard", "watch_outputs"))]
    code = "\n".join(code_lines)
    assert "watch_outputs(_publish_tmp" in code, \
        "取消守卫没有监控暂存文件"
    assert "watch_outputs(output_path" not in code, \
        "取消守卫仍在监控最终成品（取消会删掉旧结果）"


def test_discard_only_removes_the_given_staging_file(tmp_path):
    """_discard_attempt_tmp 只删指定暂存，不动同目录的旧成品。"""
    final = tmp_path / "movie.mkv"
    final.write_bytes(b"OLD-GOOD-RESULT")
    staging = tmp_path / "movie.attempt-1-2.mkv"
    staging.write_bytes(b"half-written-attempt")

    s7._discard_attempt_tmp(str(staging), None)

    assert not staging.exists(), "暂存应被删除"
    assert final.read_bytes() == b"OLD-GOOD-RESULT", \
        "旧成品被误删/误改 —— 这正是 PUB-01 要防的"


def test_discard_is_noop_when_staging_missing(tmp_path):
    """暂存不存在时不能抛异常（首次运行取消的常见场景）。"""
    s7._discard_attempt_tmp(str(tmp_path / "nope.attempt-1-2.mkv"), None)


def test_same_volume_detection(tmp_path):
    """同目录的两个路径必在同一卷。"""
    a = tmp_path / "a.bin"
    b = tmp_path / "sub" / "b.bin"
    b.parent.mkdir()
    a.write_bytes(b"x")
    b.write_bytes(b"y")
    assert s7._same_volume(str(a), str(b)) is True


def test_publish_via_replace_is_atomic_and_keeps_old_on_failure(tmp_path):
    """模拟发布：成功后旧结果被新结果替换，失败时旧结果仍在。"""
    final = tmp_path / "out.mkv"
    final.write_bytes(b"OLD")

    # 场景 A：暂存有效 → 发布成功
    ok_tmp = tmp_path / "out.attempt-ok.mkv"
    ok_tmp.write_bytes(b"NEW-VALID")
    os.replace(str(ok_tmp), str(final))
    assert final.read_bytes() == b"NEW-VALID"

    # 场景 B：暂存失败被丢弃 → 成品保持上一次的值
    bad_tmp = tmp_path / "out.attempt-bad.mkv"
    bad_tmp.write_bytes(b"HALF")
    s7._discard_attempt_tmp(str(bad_tmp), None)
    assert final.read_bytes() == b"NEW-VALID", \
        "失败的 attempt 不该影响已发布的成品"


def test_staging_name_is_attempt_unique_and_keeps_extension(tmp_path):
    """暂存名必须带 attempt 标识，**且保留扩展名**。

    踩坑记录：最初命名成 `xxx.mkv.attempt-1-2.partial`，ffmpeg 靠扩展名
    推断 muxer，直接报 "Unable to find a suitable output format" → rc=84。
    真实编码验证时才发现（纯静态测试抓不到）。
    """
    src = open(s7.__file__, encoding="utf-8").read()
    assert ".attempt-{_attempt_id}{_out_ext}" in src, \
        "暂存名丢了扩展名 —— ffmpeg 会无法推断 muxer"
    assert 'os.path.splitext(output_path)' in src
    assert ".partial" not in src.split("def _same_volume")[0].split("_publish_tmp =")[-1], \
        "暂存名不应以 .partial 结尾"
