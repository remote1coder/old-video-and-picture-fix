# -*- coding: utf-8 -*-
"""stage3 生成的 AI 子进程脚本必须是合法 Python。

为什么要有这个测试
------------------
stage3 的 AI 脚本是**字符串拼接**出来的（f.write("..." "..." ...)）。
这类代码最容易出现"stage3 自身能编译、但拼出来的脚本是坏的"——
2026-10-25 实测连抓两个：
    1. `os.environ['WB_FRAME_DTYPE'])` 多了一个 `)` → unterminated string
    2. `else:` 后面跟着**顶层**的 `try:` → expected an indented block
两个 bug 都只有把拼接结果 compile() 一遍才能发现。

做法：从 stage3 源码里抽出那段 f.write(...) 的实参，用假文件对象真执行一遍
拿到拼好的脚本文本，再 compile()。同时断言两条路径（共享内存 / 原 PNG）
都存在，且默认（无环境变量）走 PNG。
"""
import io
import os
import re

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_STAGE3 = os.path.join(_ROOT, "video_restoration_pipeline", "stages",
                       "stage3_ai_upscale.py")


def _extract_generated_script():
    src = io.open(_STAGE3, encoding="utf-8", errors="replace").read()
    # 锚点必须唯一。原先锚 `_SHM_OK = False`，但 stage3 里现在有**两段**
    # 生成脚本（单张路径 / 批量路径），两段都带这个守卫；`src.index()` 命中
    # 第一段，抽出来的是单张脚本 —— 断言就以"PNG 路径被改掉了"的形式失败。
    # `WB_FRAME_SHM_NAME` 只在批量（PNG/共享内存）脚本里出现，用它做锚。
    i = src.index("WB_FRAME_SHM_NAME")
    j = src.rindex("f.write(", 0, i)          # 向前找包含它的那个 f.write(
    k, depth = j + len("f.write("), 1
    while depth:
        if src[k] == "(":
            depth += 1
        elif src[k] == ")":
            depth -= 1
        k += 1
    block = src[j + len("f.write("):k - 1]

    class _Fake:
        def __init__(self):
            self.buf = []

        def write(self, s):
            self.buf.append(s)

    env = {"f": _Fake(), "scale": 2, "model_path": "/x/y.pth",
           "input_path": "/in", "output_dir": "/out", "gpu_id": 0,
           "arch_code": "ARCH = None", "shared_dir": "/shared",
           "native_scale": 4, "model": "realesr-general-x4v3"}
    for _ in range(40):                        # 缺什么补什么
        try:
            exec("f.write(" + block + ")", env)
            break
        except NameError as e:
            env[re.search(r"name '(\w+)'", str(e)).group(1)] = "X"
    else:                                     # pragma: no cover
        raise AssertionError("无法求值 f.write 块（缺变量补不齐）")
    return "".join(env["f"].buf)


def test_generated_script_compiles():
    script = _extract_generated_script()
    try:
        compile(script, "<generated_ai_script>", "exec")
    except SyntaxError as e:
        lines = script.split("\n")
        lo, hi = max(0, e.lineno - 6), min(len(lines), e.lineno + 5)
        ctx = "\n".join("  %4d| %s" % (n + 1, lines[n])
                        for n in range(lo, hi))
        raise AssertionError(
            "生成的 AI 脚本语法错误: %s (line %s)\n%s" % (e.msg, e.lineno, ctx))


def test_generated_script_has_both_paths():
    s = _extract_generated_script()
    assert "_SHM_OK = False" in s, "缺少共享内存分支的守卫"
    assert "SharedMemory(name=_IN)" in s, "缺少共享内存挂载"
    # R22：PNG 路径的 glob 现在先落到 `_inputs` 再判空 —— 字面量
    # `for fp in sorted(glob.glob(` 已经不成立。断言改成守**行为**：
    # 仍然要 glob 输入目录、仍然要逐帧循环、且**空目录必须大声失败**。
    assert "glob.glob(" in s and "os.path.join(input_path, '*.png')" in s, \
        "原 PNG 路径被改掉了（不再 glob 输入目录）"
    assert "for fp in _inputs:" in s, "原 PNG 路径的逐帧循环被改掉了"
    assert "STAGE3_AI_NO_INPUT" in s and "sys.exit(84)" in s, \
        "空输入目录必须失败（曾经零产出报成功）"
    assert "sys.exit(0)" in s, "共享内存路径应清理后提前退出"


def test_generated_script_shm_is_env_gated():
    """没设 WB_FRAME_SHM_NAME 时必须走 PNG —— 回退不能被意外禁用。"""
    s = _extract_generated_script()
    assert "os.environ.get('WB_FRAME_SHM_NAME')" in s
    # 守卫必须包在 try 里：变量缺失/非法 -> _SHM_OK 保持 False
    assert "_SHM_OK = False" in s.split("try:")[0] or \
           s.index("_SHM_OK = False") < s.index("_SHM_OK = True")
