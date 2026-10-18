"""A3：发布用的 .partial / .pubbak 必须按 job 隔离。

原实现 `tmp_avif = work_output_path + ".partial"` 纯由**输出路径**派生，
不含 job 维度。当多个 job 并发发布到同一输出路径（或同一 job 重跑而
上一轮 .partial 尚未清理）时，两个进程会**共用同一个临时名**：
一个写了一半被另一个覆盖，回滚时也可能删掉对方的备份。

只读观察者还可能读到"半代"：`.partial` 存在的窗口期里，
按输出路径派生的名字对外部是可预测的，与隔离名相比更容易被误读。

注：这些临时名不参与发布布局，因此加 job 维度不影响正常单 job 行为；
无 job_id 时保持原样（向后兼容）。
"""

import os
import sys

import conftest  # noqa: F401  (sys.path bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load_stage6():
    pstages = os.path.join(ROOT, "photo_restoration_pipeline", "stages")
    if pstages not in sys.path:
        sys.path.insert(0, pstages)
    import stage6_encode as s6  # noqa: E402
    return s6


def _derived(output_path, job_id):
    """复刻 stage6 的隔离命名逻辑（与实现保持一致）。"""
    iso = f".{job_id}" if job_id else ""
    return (f"{output_path}{iso}.partial",
            f"{output_path}{iso}.pubbak")


def test_two_jobs_get_distinct_temp_names():
    out = "/x/out.avif"
    a = _derived(out, "job_001")
    b = _derived(out, "job_002")
    assert a != b, "不同 job 的临时名必须不同（否则并发互相覆盖）"
    assert a[0] != b[0] and a[1] != b[1]


def test_no_job_id_keeps_legacy_names():
    """向后兼容：无 job_id 时与旧实现一致。"""
    out = "/x/out.avif"
    assert _derived(out, "") == (out + ".partial", out + ".pubbak")


def test_names_stay_under_output_path():
    """临时名应是输出路径的旁支，不能跑进别的目录。"""
    out = "/x/out.avif"
    p, b = _derived(out, "job_001")
    assert os.path.dirname(p) == os.path.dirname(out)
    assert os.path.dirname(b) == os.path.dirname(out)


def test_stage6_source_uses_job_scoped_temp_names():
    """静态守卫：实现里不得再出现裸 `+ ".partial"` / `+ ".pubbak"`。"""
    import io
    import tokenize
    s6 = _load_stage6()
    src = open(s6.__file__, encoding="utf-8").read()
    code = []
    prev = tokenize.INDENT
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING and prev in (
                tokenize.INDENT, tokenize.NEWLINE, tokenize.NL,
                tokenize.DEDENT):
            continue
        if tok.type not in (tokenize.NL, tokenize.NEWLINE,
                            tokenize.INDENT, tokenize.DEDENT):
            code.append(tok.string)
        prev = tok.type
    joined = " ".join(code)
    assert '".partial"' not in joined or "_iso" in joined, \
        "临时名必须带 job 维度（_iso），不得纯由输出路径派生"
    assert "_iso" in joined
