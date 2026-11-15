#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把仓库自带的 git 钩子接到 `.git/hooks/` 下。

## 为什么需要它

`.git/hooks/` 不进版本控制，所以钩子没法随仓库分发。新克隆之后必须
手动装一次，否则 `commit_msg_guard.py` 那套检查等于不存在 ——
而"以为有检查、其实没装"比"没有检查"更危险。

## 装什么

* `commit-msg` → 调 `scripts/commit_msg_guard.py`
  （拦住会暴露开发机细节的提交说明）。

## 用法

    python scripts/install_git_hooks.py          # 安装/覆盖
    python scripts/install_git_hooks.py --check  # 只检查是否已装

## 注意

幂等：重复跑只会覆盖成当前版本。如果 `.git/hooks/commit-msg` 已存在且
**不是**本脚本装的（没有标记行），默认**不覆盖**并提示，避免踩掉别人的钩子。
"""
import io
import os
import stat
import sys

MARKER = "# installed-by: scripts/install_git_hooks.py"

HOOKS = {
    "commit-msg": """#!/bin/sh
{marker}
# 检查提交说明里有没有会暴露开发机细节的写法。
# 详见 scripts/commit_msg_guard.py 顶部的说明。
ROOT=$(git rev-parse --show-toplevel) || exit 0
for PY in python3 python py; do
    if command -v "$PY" >/dev/null 2>&1; then
        exec "$PY" "$ROOT/scripts/commit_msg_guard.py" "$1"
    fi
done
exit 0
""",
}


def _repo_root():
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.dirname(here)


def main(argv):
    root = _repo_root()
    hooks_dir = os.path.join(root, ".git", "hooks")
    if not os.path.isdir(hooks_dir):
        print("找不到 .git/hooks —— 这不是 git 工作区？", file=sys.stderr)
        return 2
    check_only = "--check" in argv
    rc = 0
    for name, tpl in HOOKS.items():
        dest = os.path.join(hooks_dir, name)
        body = tpl.format(marker=MARKER)
        exists = os.path.exists(dest)
        ours = False
        if exists:
            try:
                ours = MARKER in io.open(dest, encoding="utf-8",
                                         errors="replace").read()
            except Exception:
                ours = False
        if check_only:
            state = "已装" if (exists and ours) else ("存在但不是本脚本装的" if exists else "未装")
            print(f"  {name}: {state}")
            if not (exists and ours):
                rc = 1
            continue
        if exists and not ours:
            print(f"  {name}: 已存在且不是本脚本装的，跳过（避免覆盖别人的钩子）",
                  file=sys.stderr)
            rc = 1
            continue
        with io.open(dest, "w", encoding="utf-8", newline="\n") as f:
            f.write(body)
        os.chmod(dest, os.stat(dest).st_mode | stat.S_IEXEC | stat.S_IXGRP
                 | stat.S_IXOTH)
        print(f"  {name}: 已安装")
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
