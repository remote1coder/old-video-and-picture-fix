# -*- coding: utf-8 -*-
"""S1→S3 管道的**第四进程**（vspipe）必须与另外三个同等待遇。

## 为什么要单独立测试

stage3 的三进程链（解码→AI→编码）在 R19/R21 被反复修过：谁先死怎么归因、
失败怎么回收、取消怎么杀。现在要接上游 vspipe，**不能**只是"多起一个进程"：

* 不登记到 cancel_guard → 取消后 vspipe 变孤儿，继续往管道里写；
* 不放进等待列表 → 它卡住时父进程以为活干完了；
* 不放进 `_reap_stream_pipeline` → 失败路径下句柄泄漏；
* 不参与 rc 判定 → 生产者失败、三个消费者都 rc=0，会被判成功
  （与"vspipe rc=9 + 编码器 rc=0 报成功"是同一个病）。

这条测试逐项检查这些点，防止以后有人只在 Popen 那里加一行了事。
"""
import io
import os

import conftest  # noqa: F401

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_S3 = os.path.join(ROOT, "video_restoration_pipeline", "stages",
                   "stage3_ai_upscale.py")


def _src():
    with io.open(_S3, encoding="utf-8", errors="replace") as f:
        return f.read()


def test_src_process_is_created_before_the_decoder():
    """生产者必须先起：先起解码器会让它立刻读到 EOF。"""
    s = _src()
    i_src = s.index("src = subprocess.Popen(list(stream_src_cmd)")
    i_dec = s.index("dec = subprocess.Popen(")
    assert i_src < i_dec, "vspipe 必须在解码器之前启动"


def test_decoder_reads_the_pipe_not_a_file_in_stream_mode():
    s = _src()
    i = s.index("if stream_src_cmd:")
    block = s[i:i + 700]
    assert '"yuv4mpegpipe"' in block
    assert '"pipe:0"' in block
    # `-f` 必须在 `-i` 之前
    assert block.index('"yuv4mpegpipe"') < block.index('"pipe:0"')


def test_src_is_registered_for_cancel_wait_reap_and_rc():
    s = _src()
    # 取消守卫登记
    assert "for _p in (src, dec, ai, enc):" in s
    # 等待列表包含它（列表推导式形式）
    assert "for p in (src, dec, ai, enc) if p is not None" in s
    # 统一收尾包含它（注意 `_reap_stream_pipeline` 有**定义**和**调用**两处，
    # 要取带进程列表的那处调用，不能撞上函数定义）
    assert "_reap_stream_pipeline(\n            [p for p in (src, dec, ai, enc)" in s
    # rc 判定包含它
    assert "_src_rc" in s and "_src_rc == 0" in s
    # 失败信息要把 src 的 rc 打出来（否则归因不到生产者）
    assert "rc src=" in s


def test_default_path_is_unchanged():
    """不传 stream_src_cmd 时必须还是读文件的老路径（旧行为不变）。"""
    s = _src()
    i = s.index("if stream_src_cmd:")
    j = s.index("else:", i)
    block = s[j:j + 400]
    assert '"-i", str(input_video)' in block
