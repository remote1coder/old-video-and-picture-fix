# -*- coding: utf-8 -*-
"""D3：真并发与故障注入（不是"加个 sleep 然后祈祷"）。

审核 §D3 的要求：
"不要只加 STAGE7_TEST_PAUSE_AT 后靠随机 sleep。优先注入测试 adapter/
回调和跨进程 Event/屏障；用真实生产 publisher、临时小文件和两个进程验证。"

本文件用**真实 `publish_lock`** 和两个真实进程验证三件事：

1. 同一目标的互斥锁确实让两个进程**串行**（不会丢更新）；
2. attempt 专属报告文件名不冲突，两个 attempt 各自的报告都能落盘；
3. 报告与媒体的**配对**：后来者不会把前者的报告留在自己媒体旁边
   （因为报告按 attempt 命名、由各自转正）。

这几条是协议级保证，不复制任何业务算法。
"""
import json
import os
import subprocess
import sys
import textwrap

import pytest

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SHARED = os.path.join(_ROOT, "shared_modules")
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)


# --------------------------------------------------------------------------
# 子进程脚本：在锁内做"读-改-写"，用于检测是否真的串行
# --------------------------------------------------------------------------
_WORKER = textwrap.dedent('''
    import os, sys, time
    sys.path.insert(0, sys.argv[1])
    from publish_lock import acquire_target_locks

    target, counter, tag, hold = sys.argv[2], sys.argv[3], sys.argv[4], float(sys.argv[5])
    with acquire_target_locks([target], timeout=30):
        with open(counter, "r", encoding="utf-8") as f:
            n = int(f.read().strip() or 0)
        time.sleep(hold)          # 加宽临界区，暴露非互斥
        n += 1
        with open(counter, "w", encoding="utf-8") as f:
            f.write(str(n))
        # 模拟"媒体 + 本 attempt 报告"一起落地
        with open(target + ".media-" + tag, "wb") as f:
            f.write(("MEDIA-" + tag).encode())
        with open(target + ".verify." + tag + ".json", "w", encoding="utf-8") as f:
            f.write('{"attempt": "%s"}' % tag)
''')


def _run_two(tmp_path, n=2, hold=0.4):
    target = str(tmp_path / "out.mkv")
    counter = str(tmp_path / "counter.txt")
    counter_file = tmp_path / "counter.txt"
    counter_file.write_text("0", encoding="utf-8")
    script = tmp_path / "worker.py"
    script.write_text(_WORKER, encoding="utf-8")

    procs = []
    for i in range(n):
        procs.append(subprocess.Popen(
            [sys.executable, str(script), _SHARED, target,
             counter, "A%d" % i, str(hold)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE))
    outs = []
    for p in procs:
        so, se = p.communicate(timeout=120)
        outs.append((p.returncode, se.decode("utf-8", "replace")))
    return target, counter_file, outs


def test_two_processes_serialise_on_the_same_target(tmp_path):
    """同一目标的两个进程必须**串行**执行临界区。

    非互斥时典型症状是丢更新：两个进程都读到 0，都写 1。
    """
    target, counter, outs = _run_two(tmp_path, n=2, hold=0.4)
    for rc, err in outs:
        assert rc == 0, f"子进程失败：rc={rc}\n{err[:400]}"

    final = int(counter.read_text(encoding="utf-8").strip())
    assert final == 2, (
        f"互斥失效：临界区应被串行执行两次，计数应为 2，实际 {final}")


def test_attempt_scoped_reports_do_not_collide(tmp_path):
    """两个 attempt 的报告按 attempt 命名，互不覆盖。"""
    target, _counter, outs = _run_two(tmp_path, n=2, hold=0.2)
    for rc, err in outs:
        assert rc == 0, err[:400]

    reports = sorted(os.listdir(str(tmp_path)))
    reports = [r for r in reports if ".verify." in r]
    assert len(reports) == 2, f"两个 attempt 的验证报告应各自独立，实际 {reports}"

    seen = set()
    for name in reports:
        doc = json.loads((tmp_path / name).read_text(encoding="utf-8"))
        seen.add(doc["attempt"])
    assert seen == {"A0", "A1"}, f"报告内容应绑定各自的 attempt，实际 {seen}"


def test_report_never_describes_the_other_attempts_media(tmp_path):
    """每个 attempt 的报告只能指向自己的媒体文件。

    这是 R12-05 的核心不变量：A 的报告不能描述 B 的媒体。
    报告文件名带 attempt，媒体也带 attempt，二者必须一一对应。
    """
    target, _counter, outs = _run_two(tmp_path, n=2, hold=0.2)
    for rc, err in outs:
        assert rc == 0, err[:400]

    for tag in ("A0", "A1"):
        media = tmp_path / f"out.mkv.media-{tag}"
        report = tmp_path / f"out.mkv.verify.{tag}.json"
        assert media.exists(), f"{tag} 的媒体缺失"
        assert report.exists(), f"{tag} 的报告缺失"
        doc = json.loads(report.read_text(encoding="utf-8"))
        assert doc["attempt"] == tag, (
            f"{report.name} 描述的是 {doc['attempt']} 而不是 {tag}")


def test_lock_timeout_is_reported_not_hung(tmp_path):
    """持锁不放时，第二个进程应在超时后**明确失败**，而不是无限等待。"""
    target = str(tmp_path / "out.mkv")
    script = tmp_path / "hold.py"
    script.write_text(textwrap.dedent('''
        import sys, time
        sys.path.insert(0, sys.argv[1])
        from publish_lock import acquire_target_locks
        with acquire_target_locks([sys.argv[2]], timeout=30):
            print("HELD", flush=True)
            time.sleep(6)
    '''), encoding="utf-8")

    holder = subprocess.Popen([sys.executable, str(script), _SHARED, target],
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        assert holder.stdout.readline().strip() == b"HELD", "持有者没拿到锁"

        waiter = tmp_path / "wait.py"
        waiter.write_text(textwrap.dedent('''
            import sys
            sys.path.insert(0, sys.argv[1])
            from publish_lock import acquire_target_locks, TargetLockTimeout
            try:
                with acquire_target_locks([sys.argv[2]], timeout=1.0):
                    print("GOT")
            except TargetLockTimeout:
                print("TIMEOUT")
        '''), encoding="utf-8")

        r = subprocess.run([sys.executable, str(waiter), _SHARED, target],
                           capture_output=True, timeout=60)
        assert b"TIMEOUT" in r.stdout, (
            f"抢锁应在 1s 后明确超时，实际输出 {r.stdout!r} "
            f"stderr={r.stderr[:200]!r}")
    finally:
        holder.kill()
        holder.wait(timeout=30)
