"""PUB-04（第七轮）：发布目标的互斥锁。

**为什么不能"整个目标集合一把锁"**

用户的任务书指出：不能只以整个集合摘要作锁。反例：

```
进程 A 发布 {output.avif, output.tif}  → 集合摘要 hash({avif,tif}) = L1
进程 B 发布 {output.avif}              → 集合摘要 hash({avif})      = L2
```

`L1 != L2`，两个进程各自拿到"不同的锁"，都认为自己独占 —— 但它们
**都要写同一个 `output.avif`**。集合摘要把"互斥"错误地绑定到了
"完全相同的集合"，而真正需要互斥的是**任意重叠的目标**。

**做法**：为每个目标路径建一把独立的锁文件（`<target>.lock`，与目标
同目录），按**字典序固定顺序**逐个获取。任何重叠的目标集合必然在某个
共同路径上抢同一把锁，因此正确互斥；固定顺序避免死锁。

锁文件与目标同目录是**必需的**：若放到 temp_dir，不同 job 的
temp_dir 不同，锁文件位置也不同，跨 job 就完全不互斥了。

**平台**：Windows 用 `msvcrt.locking`（字节区间锁），POSIX 用
`fcntl.flock`。都不依赖第三方包。

**已知限制（不隐瞒）**
* 锁的是"本组件自己的锁文件"，不会阻止完全绕过本组件的其他程序；
* 网络文件系统上的语义取决于底层实现，本组件不做跨主机保证；
* 进程被 `kill -9` 时 OS 会释放文件锁，但 `.lock` 文件本身会留下
  （无害，下次复用）。
"""

from __future__ import annotations

import errno
import hashlib
import os
import time
from contextlib import contextmanager
from typing import Iterable, List, Optional, Sequence, Tuple

__all__ = [
    "TargetLockTimeout",
    "acquire_target_locks",
    "lock_path_for",
    "normalize_targets",
]

# Windows 上 msvcrt 只支持对已存在文件的字节锁；用 1 字节表示"独占"。
_LOCK_BYTES = 1


class TargetLockTimeout(RuntimeError):
    """在超时时间内没能拿到全部目标锁。"""

    def __init__(self, targets: Sequence[str], timeout: float):
        self.targets = list(targets)
        self.timeout = timeout
        super().__init__(
            f"could not acquire publish locks within {timeout:.1f}s for: "
            + ", ".join(self.targets))


def normalize_targets(targets: Iterable[Optional[str]]) -> List[str]:
    """规范化并排序目标路径。

    * 转绝对路径（`os.path.abspath`）+ `normcase`（Windows 大小写不敏感）；
    * 去重；
    * **按确定顺序排序** —— 固定顺序是避免死锁的唯一手段。
    """
    seen = set()
    out: List[str] = []
    for t in targets:
        if not t:
            continue
        # realpath 会解析符号链接/junction；发布目标是用户可见路径，
        # 用 abspath 保留原样，但用 normcase 消除大小写差异。
        key = os.path.normcase(os.path.abspath(str(t)))
        if key in seen:
            continue
        seen.add(key)
        out.append(key)
    out.sort()
    return out


def lock_path_for(target: str) -> str:
    """锁文件路径：与目标同目录的 `<name>.lock`。

    同目录是**必需**的（见模块 docstring）：不同 job 的 temp_dir 不同，
    锁文件若放 temp_dir 就无法跨 job 互斥。

    路径过长时（Windows MAX_PATH）退化为目录内的哈希名。
    """
    abs_target = os.path.abspath(target)
    cand = abs_target + ".lock"
    if len(cand) < 240:
        return cand
    d = os.path.dirname(abs_target)
    h = hashlib.sha1(abs_target.encode("utf-8")).hexdigest()[:16]
    return os.path.join(d, f".publish-{h}.lock")


def _try_lock(fh) -> bool:
    """尝试对已打开的文件句柄取独占锁；取不到返回 False。"""
    try:
        import msvcrt  # Windows
    except ImportError:
        msvcrt = None  # type: ignore

    if msvcrt is not None:
        try:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, _LOCK_BYTES)
            return True
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EDEADLK, errno.EAGAIN, 13, 36):
                return False
            raise
    else:
        import fcntl
        try:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError as exc:
            if exc.errno in (errno.EACCES, errno.EAGAIN, 11, 13):
                return False
            raise


def _unlock(fh) -> None:
    try:
        import msvcrt
    except ImportError:
        msvcrt = None  # type: ignore
    try:
        if msvcrt is not None:
            fh.seek(0)
            msvcrt.locking(fh.fileno(), msvcrt.LK_UNLCK, _LOCK_BYTES)
        else:
            import fcntl
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
    except Exception:
        # 释放失败不该把成功发布变成失败；OS 会在 close 时回收
        pass


@contextmanager
def acquire_target_locks(targets: Iterable[Optional[str]],
                         timeout: float = 30.0,
                         poll_interval: float = 0.05,
                         logger=None):
    """按固定顺序获取**全部**目标锁；任一拿不到就整体回滚并抛超时。

    用法：

    ```python
    with acquire_target_locks([avif_path, tiff_path], timeout=30) as locked:
        # locked 是规范化后的目标列表；此时已独占这些路径
        ...执行发布...
    ```

    **要么全拿到，要么一个都不持有** —— 避免"只锁了一半"导致的
    半互斥（那比不锁更难排查）。
    """
    ordered = normalize_targets(targets)
    held: List[Tuple[str, object]] = []
    deadline = time.monotonic() + max(0.0, float(timeout))

    try:
        for target in ordered:
            lp = lock_path_for(target)
            try:
                os.makedirs(os.path.dirname(lp) or ".", exist_ok=True)
                fh = open(lp, "a+b")
            except OSError as exc:
                if logger:
                    logger.error("PUBLISH_LOCK_OPEN_FAIL",
                                 f"cannot open lock file {lp}: {exc}",
                                 path=lp)
                raise
            got = False
            while True:
                if _try_lock(fh):
                    got = True
                    break
                if time.monotonic() >= deadline:
                    break
                time.sleep(poll_interval)
            if not got:
                fh.close()
                raise TargetLockTimeout(ordered, timeout)
            held.append((target, fh))
            if logger:
                logger.info("PUBLISH_LOCK_ACQUIRED",
                            "publish lock acquired", path=target, lock=lp)

        yield ordered
    finally:
        # 逆序释放（与获取顺序相反）
        for target, fh in reversed(held):
            try:
                _unlock(fh)
            finally:
                try:
                    fh.close()
                except Exception:
                    pass
            if logger:
                logger.info("PUBLISH_LOCK_RELEASED",
                            "publish lock released", path=target)
