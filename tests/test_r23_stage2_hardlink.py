# -*- coding: utf-8 -*-
r"""R23-LONG-02：stage2 不该为换个文件名再写一遍 82 GB。

stage2 是**检查型**阶段（idet 判逐行 + 帧率校验），一个像素都不改。
产物的唯一作用是给下游一个符合 `step2` 命名的路径 —— main 用
`{job}_{stem}_step2*` glob 校验，stage3 也可能 glob 兜底。

旧实现 `shutil.copyfile`：16 分钟 1080p 的 stage1 是 **82.4 GB**，
等于整条链凭空多一份全片写入。改用硬链接（同一 inode 的第二个名字）
后毫秒完成、零额外空间。

四条判据缺一不可：

1. 同卷真的走了硬链接（inode 相同、nlink=2）—— 不能只是"看起来快了"；
2. 硬链接不可用时**回退复制**且内容正确 —— 优化失败不能让 stage 挂掉；
3. 删掉源之后目标仍完整 —— 硬链接语义下这是安全的，必须实测钉死；
4. 下游契约不变（可解码、大小一致）—— 省 I/O 不能把产物变成不可用。
"""
import os
import shutil
import subprocess
import sys

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules"),
           os.path.join(_ROOT, "video_restoration_pipeline"),
           os.path.join(_ROOT, "video_restoration_pipeline", "stages")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import stage2_traditional_check as s2  # noqa: E402


def _fake_payload(tmp_path, n=1024):
    src = tmp_path / "stage1.mkv"
    src.write_bytes(b"\xAB" * n)
    return src, tmp_path / "step2.mkv"


def test_same_volume_uses_a_hard_link_not_a_copy(tmp_path):
    """同卷必须走硬链接：inode 相同、链接数为 2。"""
    src, dst = _fake_payload(tmp_path)
    before = shutil.disk_usage(str(tmp_path))

    how = s2._materialise_step2(str(src), str(dst))

    assert how == "linked", "同卷却没走硬链接（磁盘不够快时会白写一遍全片）"
    assert dst.read_bytes() == src.read_bytes()
    st_src = os.stat(str(src))
    st_dst = os.stat(str(dst))
    assert st_src.st_ino == st_dst.st_ino, "inode 不同 = 真的复制了一份"
    assert st_dst.st_nlink == 2
    # 硬链接不占额外空间：已用字节不应增加一个文件大小
    after = shutil.disk_usage(str(tmp_path))
    assert after.used - before.used < 4096


def test_falls_back_to_copy_when_link_is_unavailable(tmp_path, monkeypatch):
    """跨卷/不支持硬链接时必须老实复制，内容不能变。"""

    def cross_device(s, d):
        raise OSError(0, "The system cannot move the file to a different disk "
                         "drive", None, 17)

    monkeypatch.setattr(s2.os, "link", cross_device)
    src, dst = _fake_payload(tmp_path, 4096)

    how = s2._materialise_step2(str(src), str(dst))

    assert how == "copied"
    assert dst.read_bytes() == src.read_bytes()
    assert os.stat(str(src)).st_ino != os.stat(str(dst)).st_ino


def test_hardlinked_artifact_survives_source_removal(tmp_path):
    """删掉 stage1 不影响 step2 —— 这是硬链接比复制更好的地方，必须实测。"""
    src, dst = _fake_payload(tmp_path, 2048)
    assert s2._materialise_step2(str(src), str(dst)) == "linked"

    os.unlink(str(src))

    assert dst.is_file(), "删源后目标没了 —— 这不是硬链接"
    assert dst.read_bytes() == b"\xAB" * 2048


def test_downstream_contract_is_unchanged(tmp_path):
    """产物仍能被 ffprobe 解码、大小与源一致（省 I/O 不能改契约）。"""
    if shutil.which("ffmpeg") is None or shutil.which("ffprobe") is None:
        pytest.skip("需要 ffmpeg/ffprobe")

    src = tmp_path / "stage1.mkv"
    r = subprocess.run(
        ["ffmpeg", "-y", "-v", "error", "-f", "lavfi",
         "-i", "testsrc2=s=64x48:r=10:d=0.3", "-c:v", "ffv1",
         "-pix_fmt", "yuv420p16le", str(src)],
        capture_output=True, timeout=120)
    assert r.returncode == 0, (r.stderr or b"")[-300:]

    dst = tmp_path / "step2.mkv"
    assert s2._materialise_step2(str(src), str(dst)) == "linked"

    assert os.path.getsize(str(dst)) == os.path.getsize(str(src))
    from checkpoint import verify_ffprobe_decodeable
    assert verify_ffprobe_decodeable(str(dst)), "硬链接产物 ffprobe 解不开"


def test_copy_is_only_the_fallback_in_source():
    """静态守住：不许再改回"无条件 copyfile"。

    这个断言故意做得很窄 —— 只检查"os.link 出现在 copyfile 之前"，
    防止有人图省事把硬链接分支删掉而测试全绿。
    """
    path = os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                        "stage2_traditional_check.py")
    with open(path, encoding="utf-8") as fh:
        src = fh.read()
    assert "os.link(src, dst)" in src
    assert "_shutil.copyfile(src, dst)" in src
    assert src.index("os.link(src, dst)") < src.index("_shutil.copyfile(src, dst)"), \
        "copyfile 跑到 os.link 前面了 —— 又变成无条件复制"
