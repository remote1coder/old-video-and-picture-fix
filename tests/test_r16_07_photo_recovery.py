# -*- coding: utf-8 -*-
"""R16-07：照片恢复与取消必须诚实、可再恢复。

审核确认三处：

1. `stage6:822` 回滚仍 `copyfile(backup, formal_path)` —— 复制中断会把正式
   文件留成半文件，下次既不等于旧摘要也不等于新摘要，只能人工。
2. `stage6:440` 恢复入口看到 `journal.phase=COMMITTED` 就**先删** recovery
   记录，之后才校验路径与整代摘要，也没核对 attempt。
3. `stage6:1950` 取消 lambda 忽略回滚返回值，`CancelGuard` 还吞 hook 异常。

本文件用源码守卫锁住这三条（照片 stage6 的执行路径需要完整编码上下文，
端到端负例成本高，见台账残留）。
"""
import os
import re

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_S6 = os.path.join(_ROOT, "photo_restoration_pipeline", "stages",
                   "stage6_encode.py")


def _src():
    return open(_S6, encoding="utf-8", errors="replace").read()


def test_rollback_restores_via_private_staging():
    """回滚必须"目标侧私有暂存 → 核对摘要 → replace"，不能直写正式路径。"""
    src = _src()

    assert ".restore-{os.getpid()}" in src, \
        "回滚必须有 attempt/进程私有的暂存路径"
    assert "staged restore refused" in src, \
        "暂存内容与 old_digest 不符时必须拒绝"
    assert "os.replace(_stage, path)" in src, \
        "核对通过后要用 replace 原子落位"

    # 反例：不允许再直接 copyfile 到正式路径
    bad = re.search(r"shutil\.copyfile\(backup,\s*path\)", src)
    assert bad is None, "仍在直接 copyfile 到正式路径（中断会留半文件）"


def test_recovery_record_cleared_only_after_full_generation_check():
    """解除 recovery 记录前必须核验全组摘要与 attempt。"""
    src = _src()

    assert "_gen_ok" in src, "必须有'整代完整'的判定变量"
    assert "recovery record belongs to another attempt" in src, \
        "必须核对 attempt 归属"
    # 源码里这条消息被拆成两个相邻字面量，只查前缀
    assert "target does not match the recorded" in src, \
        "必须逐个目标核对新代摘要"

    # 结构：`if _gen_ok:` 里才允许 os.remove(recovery record)
    # 结构：删记录必须在 `if _gen_ok:` 之后、其 `else:` 之前
    _gi = src.find("if _gen_ok:")
    _ri = src.find("os.remove(_rec_path_)")
    assert _gi != -1 and _ri != -1, "缺少 _gen_ok 判定或删记录调用"
    assert _gi < _ri, "删记录必须发生在 _gen_ok 判定之后"
    assert "else:" not in src[_gi:_ri], \
        "删记录落在了 `if _gen_ok:` 的 else 分支里（语义反了）"


def test_cancel_hook_reads_and_persists_rollback_result():
    """取消钩子必须读取回滚结果并持久记录。"""
    src = _src()

    assert "_cancel_rollback" in src, "必须是有名字的钩子，便于读取结果"
    assert "STAGE6_CANCEL_ROLLBACK_RESULT" in src, "必须记录回滚结果"
    assert "STAGE6_CANCEL_ROLLBACK_UNRECORDED" in src, \
        "诊断写失败也要如实报出"
    assert 'cancel_rollback"' in src, "结果要写进 journal（持久化）"

    # 反例：不允许再用忽略返回值的 lambda
    bad = re.search(r"add_cancel_hook\(\s*\n?\s*lambda:", src)
    assert bad is None, "取消钩子仍是忽略返回值的 lambda"
