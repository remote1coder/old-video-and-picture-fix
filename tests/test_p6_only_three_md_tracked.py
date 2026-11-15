# -*- coding: utf-8 -*-
"""守卫：仓库里**只允许**跟踪那几本长期文档，其余 md 一律不入库。

## 为什么需要这条

2026-11-16：`tests/probes/README.md` 被误推到远程。

根因不是"我忘了加 ignore"，而是 `.gitignore` 写的是：

```
*.md
!README.md        ← 不带前导斜杠
```

gitignore 的规则是**不带斜杠的模式匹配任意层级**。于是任何子目录里的
`README.md` 都被 `!README.md` 放行了，`tests/probes/README.md` 就这样
被带进了提交。

已改成 `!/README.md`（只匹配仓库根），并把那个文件 `git rm --cached`
（磁盘上保留）。

## 这条测试守什么

**只看"实际被跟踪的文件"**，不看 .gitignore 的写法 ——
因为写法可以有很多种，而"远程上有什么"只有一种事实。

一旦有人新增了别的 md（或被 ignore 规则意外放行），这里立刻失败，
并提示"要入库必须先改 .gitignore 并在本白名单登记"。

## 白名单为什么是这几个

用户明确指定：远程只放行 `README.md`、`PROJECT_MANUAL.md`、`architecture.md`。
交接文档、台账（LEDGER/CLOSURE/PROGRESS）、审计、缺陷清单一律不入库。
"""
import os
import subprocess

ALLOWED = {
    "README.md",
    "readme.md",
    "PROJECT_MANUAL.md",
    "architecture.md",
}

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _tracked_md():
    out = subprocess.run(
        ["git", "ls-files", "*.md"],
        cwd=ROOT, capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        # 不是在 git 工作区里跑（例如源码被打包到别处）——这条无从判断
        return None
    return [p.strip() for p in out.stdout.splitlines() if p.strip()]


def test_only_allowed_md_files_are_tracked():
    tracked = _tracked_md()
    if tracked is None:
        import pytest
        pytest.skip("not a git worktree")
    extra = sorted(set(tracked) - ALLOWED)
    assert not extra, (
        "有不该入库的 md 被跟踪了：\n  " + "\n  ".join(extra) +
        "\n\n这些会随 push 上远程。处理方式："
        "\n  1) git rm --cached <文件>   （磁盘上保留）"
        "\n  2) 确认 .gitignore 用的是 `!/README.md` 这种**根目录限定**写法"
        "\n  3) 如果它确实该入库，先在本测试的 ALLOWED 里登记并说明理由")


def test_subdir_readme_is_ignored():
    """`tests/probes/README.md` 这类子目录 README 必须被忽略。

    直接问 git 的判定，而不是读 .gitignore 文本 —— 后者写法太多。
    """
    target = "tests/probes/README.md"
    if not os.path.exists(os.path.join(ROOT, target)):
        import pytest
        pytest.skip("file not present")
    out = subprocess.run(
        ["git", "check-ignore", "-q", target],
        cwd=ROOT, capture_output=True, timeout=60)
    if out.returncode not in (0, 1):
        import pytest
        pytest.skip("git check-ignore unavailable")
    assert out.returncode == 0, (
        f"{target} 没有被忽略 —— 它会被再次推上远程。"
        "检查 .gitignore 是否用了 `!/README.md`（不带斜杠会匹配任意层级）")
