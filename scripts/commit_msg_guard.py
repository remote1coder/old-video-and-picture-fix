#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""commit message 守卫：拦住会暴露开发机细节的写法。

## 为什么要有它

`git log` 是**公开**的 —— 推上去之后任何人都能看到每一条提交说明。
但写起来太顺手，很容易被当成内部笔记：

* 写下开发机的绝对路径（盘符或家目录开头的长路径）；
* 写下只在本机存在的临时目录名；
* 写下"哪些文件该入库、要不要推远程"这类**工作约定**；
* 写下对话过程。

这几类信息与代码行为无关，却把开发环境暴露了。人的自觉不可靠
（这类写法能长期存在而没人发现），所以做成机械检查，在 `commit-msg` 钩子里跑。

## 拦什么（只用**通用模式**，不枚举任何具体名字）

1. **绝对路径**：Windows 盘符（`X:\\` / `X:/`）、常见 POSIX 根目录
   （`/usr` `/home` `/mnt` `/tmp` …）。仓库相对路径不受影响。
2. **带日期的下划线目录**：形如 `_<name>_YYYYMMDD` —— 个人工作区的典型命名。
3. **工作约定**：含"只放行 N 个 / 只该有 N 个 / 不入库 / 不推远程"等表述。
4. **对话过程**：含"用户要求 / 用户问 / 用户原话"等。
5. **正文过长**：超过 4000 字符 —— 正常技术说明用不了这么多，
   超长通常意味着把内部文档整段搬了进来。

## 不拦什么

* 仓库**相对路径**（`shared_modules/x.py`、`tests/test_y.py`）—— 必要信息；
* 实测数字（`26 分钟 → 20.9 分钟`、`PSNR y:48.71`）—— 这是证据；
* 提交哈希 —— 但**引用前必须核实**，脚本查不了真假。

## 装法

`scripts/install_git_hooks.py` 把它接到 `.git/hooks/commit-msg`。
钩子不进版本控制，新克隆需跑一次安装脚本。

## 手动检查

    python scripts/commit_msg_guard.py <file-with-message>
"""
import io
import re
import sys

RULES = [
    ("绝对路径（Windows 盘符）",
     re.compile(r"[A-Za-z]:[\\/](?![/\\])[^\s\"'`)]{2,}")),
    ("绝对路径（POSIX 根目录）",
     re.compile(r"(?<![\w.])/(?:usr|home|mnt|media|tmp|var|opt|srv|root|"
                r"Users|Volumes|private|e|d|f|g)(?:/[^\s\"'`)]{2,})")),
    ("带日期的临时目录名（下划线 + YYYYMMDD）",
     re.compile(r"(?<![\w])_[A-Za-z][A-Za-z0-9_]{1,30}_\d{8}[A-Za-z0-9]*")),
    ("工作约定（不该出现在公开记录里）",
     re.compile(r"只(?:放行|该有|允许)\s*\d*\s*个?\s*[A-Za-z*]|不入库|"
                r"不推(?:远程|送)|不进版本控制")),
    ("对话过程叙述",
     re.compile(r"用户(?:要求|问|原话|指出|让|说)|对话(?:中|里)")),
]

MAX_BODY = 4000


def check(text: str):
    """返回 ``[(说明, 命中片段, 行号)]``；空列表表示通过。"""
    problems = []
    lines = text.splitlines()
    body = [(i + 1, ln) for i, ln in enumerate(lines)
            if not ln.lstrip().startswith("#")]
    for desc, pat in RULES:
        for lineno, ln in body:
            for m in pat.finditer(ln):
                problems.append((desc, m.group(0)[:60], lineno))
    total = sum(len(ln) for _, ln in body)
    if total > MAX_BODY:
        problems.append((f"正文过长（{total} 字符 > {MAX_BODY}）", "", 0))
    return problems


def main(argv):
    if len(argv) < 2:
        print("usage: commit_msg_guard.py <commit-msg-file>", file=sys.stderr)
        return 2
    text = io.open(argv[1], encoding="utf-8", errors="replace").read()
    problems = check(text)
    if not problems:
        return 0
    print("\ncommit message 被拦下：它会随 git log 公开。\n", file=sys.stderr)
    seen = set()
    for desc, frag, lineno in problems:
        if (desc, frag) in seen:
            continue
        seen.add((desc, frag))
        loc = f"第 {lineno} 行" if lineno else ""
        print(f"  x {desc} {loc}", file=sys.stderr)
        if frag:
            print(f"      {frag}", file=sys.stderr)
    print("""
改法：
  * 路径写**仓库相对路径**，不要写盘符或家目录；
  * 只在本机存在的目录，说"临时目录/证据目录"即可；
  * 工作约定不写进 commit；
  * 正文讲"改了什么 / 为什么 / 影响"，不搬内部文档。
""", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
