# -*- coding: utf-8 -*-
"""R22-A09 第二批：**硬退出**矩阵（六个语义点）。

## 为什么必须用 `os._exit`

`sys.exit` 会执行 `finally` / `atexit` / 上下文管理器的 `__exit__` ——
那是**优雅退出**，测不到"进程在写删中途消失"。真实故障（断电、OOM killer、
蓝屏）不会给任何清理机会。`os._exit()` 直接调系统 `_exit()`，跳过全部清理，
这才是要模拟的形态。

## 注入方式（不改生产代码）

参考 BMAD-METHOD issue #3112 的崩溃复现法：在**子进程**里包装目标函数，
执行到指定点 `os._exit`，然后跑真实的 `publisher.publish()`。
被测代码一行不改。

| 点 | 锚 | 判定 |
|---|---|---|
| `after_backup` | `shutil.copyfile` | 目标名含 `.pubbak-` |
| `after_media` | `os.replace` | 目标以 `.mkv` 结尾 |
| `after_verify` | `os.replace` | 目标以 `.verify.json` 结尾 |
| `after_marker` | `os.replace` | 目标以 `.done.json` 结尾 |
| `before_committed` | `write_journal` | payload 的 phase == COMMITTED，**写之前** |
| `after_committed` | `write_journal` | 同上，**写成功之后** |

## 断言什么

审核要求："结果应为完整旧代、完整新代，或明确 `recovery_required` 且保留现场"，
并且"按整代哈希核验，而非只看退出码"。

所以每个点都核：

1. 子进程确实死在目标点（rc=97 + `hit.json` 的 `point`）
2. `hit.json` 里记了**死时每个对象的哈希**（子进程死前自己 dump）
3. `recover` 之后每个对象要么是完整旧代、要么是完整新代
4. 判 `recovery_required` 时，所有对象哈希与死时**逐字节相同**（一个都没动）
5. journal 终态与结论自洽（ROLLED_BACK / COMMITTED / RECOVERY_REQUIRED）
"""
import hashlib
import json
import os
import subprocess
import sys

import pytest

import conftest  # noqa: F401

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for _p in (_ROOT, os.path.join(_ROOT, "shared_modules")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import publisher as P  # noqa: E402
import publish_journal as J  # noqa: E402


class _Log:
    def __init__(self):
        self.events = []

    def _rec(self, ev, msg="", **kw):
        self.events.append((ev, msg, kw))

    info = warning = error = critical = _rec

    def names(self):
        return [e[0] for e in self.events]


def _sha(path):
    if not os.path.exists(path):
        return "<absent>"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def _w(path, data):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    mode = "wb" if isinstance(data, bytes) else "w"
    with open(path, mode, **({} if mode == "wb" else {"encoding": "utf-8"})) as f:
        f.write(data)
    return path


#: 子进程脚本：包装三个原语，到点 `os._exit`，死前把自己看到的现场 dump 出来。
_CHILD = r'''
import hashlib, json, os, shutil, sys

cfg = json.load(open(os.environ["R22_CFG"], encoding="utf-8"))
POINT = cfg["point"]
HIT = cfg["hit"]

for p in (cfg["root"], os.path.join(cfg["root"], "shared_modules")):
    if p not in sys.path:
        sys.path.insert(0, p)

import publisher as P
import publish_journal as J


def _sha(path):
    if not os.path.exists(path):
        return "<absent>"
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def _die(tag):
    """死前记下**每个对象的当前哈希** —— 父进程据此判断"有没有被动过"。"""
    snap = {name: _sha(p) for name, p in cfg["targets"].items()}
    try:
        doc = J.read_journal(cfg["journal"]) or {}
        phase = str(doc.get("phase") or "")
    except Exception:
        phase = "<unreadable>"
    with open(HIT, "w", encoding="utf-8") as f:
        json.dump({"point": POINT, "at": tag, "pid": os.getpid(),
                   "snapshot": snap, "journal_phase": phase},
                  f, ensure_ascii=False)
    # os._exit：不执行 finally / atexit / 上下文管理器退出
    os._exit(97)


# ---- ① 备份复制完成 ----
_real_copy = shutil.copyfile


def _copy(src, dst, *a, **k):
    r = _real_copy(src, dst, *a, **k)
    if POINT == "after_backup" and ".pubbak-" in str(dst):
        _die("backup:" + os.path.basename(str(dst)))
    return r


shutil.copyfile = _copy

# ---- ②③④ 晋升 ----
_real_replace = os.replace


def _replace(src, dst, *a, **k):
    r = _real_replace(src, dst, *a, **k)
    d = str(dst)
    if d.endswith(".journal.json"):
        return r
    if POINT == "after_media" and d.endswith(".mkv"):
        _die("promote:media")
    if POINT == "after_verify" and d.endswith(".verify.json"):
        _die("promote:verify")
    if POINT == "after_marker" and d.endswith(".done.json"):
        _die("promote:marker")
    return r


os.replace = _replace

# ---- ⑤⑥ COMMITTED ----
_real_wj = J.write_journal


def _wj(path, payload):
    phase = str((payload or {}).get("phase") or "")
    if phase == "COMMITTED" and POINT == "before_committed":
        _die("before_committed")
    ok = _real_wj(path, payload)
    if phase == "COMMITTED" and POINT == "after_committed":
        _die("after_committed")
    return ok


# publisher 是模块级 `from publish_journal import write_journal`，两个都要换
J.write_journal = _wj
P.write_journal = _wj

arts = [P.Artifact(n, cfg["staging"][n], cfg["targets"][n])
        for n in cfg["order"]]
res = P.publish(arts, cfg["journal"], cfg["backup"], "att1",
                decision={"publish": True},
                allowed_output_roots=[cfg["allowed_root"]])
print("PUBLISH_OK", res.ok, res.errors)
sys.exit(0 if res.ok else 3)
'''

POINTS = ["after_backup", "after_media", "after_verify", "after_marker",
          "before_committed", "after_committed"]


def _build_case(tmp_path):
    """造一**旧代**（目标路径上已有内容）+ 一份**新代** staging。"""
    out_dir = tmp_path / "out"
    out_dir.mkdir(parents=True, exist_ok=True)
    media = str(out_dir / "clip_restored.mkv")
    verify = media + ".verify.json"
    fullchain = media + ".fullchain.json"
    marker = media + ".done.json"

    # 旧代：三个目标都在，内容各不相同
    _w(media, b"OLD-MEDIA")
    _w(verify, json.dumps({"media_sha256": "old"}))
    _w(fullchain, json.dumps({"ok": "old"}))
    # marker 原本不存在（首次发布）—— 这样"原本不存在"的删除路径也被覆盖

    stage = tmp_path / "staging"
    s_media = _w(str(stage / "m.bin"), b"NEW-MEDIA")
    s_verify = _w(str(stage / "v.json"),
                  json.dumps({"media_sha256": _sha(str(stage / "m.bin"))}))
    s_fc = _w(str(stage / "f.json"), json.dumps({"ok": "new"}))
    s_mk = _w(str(stage / "mk.json"), json.dumps({"impl": "new"}))

    return {
        "out_dir": str(out_dir),
        "targets": {"media": media, "verify": verify,
                    "fullchain": fullchain, "marker": marker},
        "order": ["media", "verify", "fullchain", "marker"],
        "staging": {"media": s_media, "verify": s_verify,
                    "fullchain": s_fc, "marker": s_mk},
        "journal": media + ".journal.json",
        "backup": str(tmp_path / "backup"),
        "root": _ROOT,
    }


def _run_child(tmp_path, case, point):
    cfg = dict(case)
    cfg["point"] = point
    cfg["hit"] = str(tmp_path / "hit.json")
    cfg["allowed_root"] = case["out_dir"]
    cfg_path = str(tmp_path / "cfg.json")
    _w(cfg_path, json.dumps(cfg, ensure_ascii=False))
    script = str(tmp_path / "child.py")
    _w(script, _CHILD)

    env = dict(os.environ)
    env["R22_CFG"] = cfg_path
    proc = subprocess.run([sys.executable, script], capture_output=True,
                          text=True, errors="replace", timeout=180, env=env)
    hit = {}
    if os.path.exists(cfg["hit"]):
        try:
            hit = json.load(open(cfg["hit"], encoding="utf-8"))
        except Exception:
            hit = {}
    return proc, hit


@pytest.mark.parametrize("point", POINTS)
def test_hard_exit_leaves_a_provable_generation(tmp_path, point):
    """六个点各自硬退出 → 恢复后必须是完整旧代 / 完整新代 / 明确保留现场。"""
    case = _build_case(tmp_path)
    before = {n: _sha(p) for n, p in case["targets"].items()}
    # ---- R24-09：新代的期望摘要必须在**事务开始前**取 ----
    # `publish` 用 `os.replace(staging, final)` 转正，事务跑完 staging 就没了；
    # 事后去算 `_sha(staging)` 只能得到 `<absent>`，精确 oracle 会退化成
    # "什么也没比"。审核要求的"预先生成完整旧/新代期望值"正是这个意思。
    new_expected = {n: _sha(p) for n, p in case["staging"].items()}

    proc, hit = _run_child(tmp_path, case, point)

    # ---- ① 子进程必须真的死在目标点（不是被别的原因带走的）----
    assert proc.returncode == 97, (
        f"{point}: 期望 rc=97（os._exit），实际 {proc.returncode}\n"
        f"stdout={proc.stdout[-500:]}\nstderr={proc.stderr[-800:]}")
    assert hit.get("point") == point, hit
    assert hit.get("at"), "死前没有记下触发点"

    # 死时现场（子进程自己 dump 的）
    at_death = hit.get("snapshot") or {}
    assert set(at_death) == set(case["targets"]), at_death

    # ---- ② 跑生产 recover ----
    log = _Log()
    rep = P.recover(case["journal"], case["backup"], "att2", logger=log,
                    allowed_output_roots=[case["out_dir"]])
    after = {n: _sha(p) for n, p in case["targets"].items()}

    # ---- ③ 结论只能是三种之一 ----
    if rep is None:
        # `after_backup` 是天生的**恢复责任之外**：备份已经复制完，
        # 但 PREPARED 还没写 —— 没有 journal 就没有事务，recover 无从下手。
        # 这不是缺陷：目标一个字节都没动（晋升还没开始），
        # 只留下一份孤儿备份（无害，属于"备份目录清理"的范畴）。
        # 断言它的真实形态：完整旧代 + 没有半成品 + 有孤儿备份。
        assert point == "after_backup", (
            f"{point}: journal 都不存在，说明注入点太早：{hit}")
        for name in ("media", "verify", "fullchain"):
            assert after[name] == before[name], (
                f"{point}: 没有事务但 {name} 被改了")
        assert after["marker"] == "<absent>"
        orphans = ([f for f in os.listdir(case["backup"])
                    if ".pubbak-" in f]
                   if os.path.isdir(case["backup"]) else [])
        if point == "after_backup":
            assert orphans, "标称备份完成，却没有孤儿备份可清"
        return

    if rep.get("ok") is True:
        gen = str(rep.get("generation") or "")
        if gen in ("all_old", "identical") or rep.get("action") in (
                "rolled_back", "confirmed_rolled_back"):
            # 完整旧代：每个目标要么回到旧内容，要么本来就是旧的
            for name in ("media", "verify", "fullchain"):
                assert after[name] == before[name], (
                    f"{point}: 判成完整旧代，但 {name} 不是旧内容")
            # marker 原本不存在 → 回滚后必须仍不存在
            assert after["marker"] == "<absent>", (
                f"{point}: 旧代没有 marker，回滚后却出现了 {after['marker']}")
        elif gen in ("all_new",) or rep.get("action") in (
                "rolled_forward", "recorded_committed"):
            # ---- R24-09 / R23-A11：完整新代必须是**精确的新代** ----
            # 旧断言只要求 `after != before`（"不同于旧"）—— 随机损坏的新内容
            # 也满足它，于是"六点精确新旧哈希"这句话没有依据。
            # 这里改成与 `case["staging"]` 里那份**预先生成的新代内容**逐一对齐：
            # 每个目标的摘要必须**恰好等于**新代那一份。
            for name in ("media", "verify", "fullchain", "marker"):
                _want = new_expected[name]
                assert after[name] == _want, (
                    f"{point}: 判成完整新代，但 {name} 不是新代内容\n"
                    f"  期望 {_want}\n  实际 {after[name]}\n"
                    f"  旧代 {before.get(name)}")
        else:
            # ---- B08 / O01：成功必须落在**已知终态**上 ----
            # 旧写法只有 "all_old/identical" 与 "all_new" 两个分支，
            # 一个 ok=True 但 generation/action 都不认识的返回值
            # **两个分支都不进**，于是这个点什么都不检查 —— 直接放行。
            raise AssertionError(
                f"{point}: recovery reported success with an unknown "
                f"generation={gen!r} / action={rep.get('action')!r}; "
                f"a success must name a known terminal state "
                f"(report={rep})")
    else:
        # recovery_required：**保留现场**，一个字节都不能动
        assert rep.get("ok") is False
        assert after == at_death, (
            f"{point}: 判成保留现场，但文件被改过\n"
            f"死时={at_death}\n恢复后={after}")

    # ---- ④ journal 终态与结论自洽 ----
    doc = J.read_journal(case["journal"]) or {}
    phase = str(doc.get("phase") or "")
    assert phase in ("COMMITTED", "ROLLED_BACK", "RECOVERY_REQUIRED",
                     "PREPARED", "COMMITTING"), (
        f"{point}: journal 停在了意外的 phase={phase!r}")
    if rep.get("ok") is True:
        assert phase in ("COMMITTED", "ROLLED_BACK"), (
            f"{point}: 报成功但 journal 是 {phase}")


def test_backup_point_really_died_after_the_copy(tmp_path):
    """`after_backup` 点必须发生在**备份已存在**之后（否则测的不是这个点）。"""
    case = _build_case(tmp_path)
    proc, hit = _run_child(tmp_path, case, "after_backup")
    assert proc.returncode == 97
    backups = [f for f in os.listdir(case["backup"])
               if ".pubbak-" in f] if os.path.isdir(case["backup"]) else []
    assert backups, "标称'备份完成后'，但备份目录是空的"
    # 备份里必须是**旧内容**
    assert _sha(os.path.join(case["backup"], backups[0])) == \
        _sha(case["targets"]["media"])
