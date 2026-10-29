# -*- coding: utf-8 -*-
"""帧率/倍率的**有理数**解析（R14-08）。

**为什么要单独一个模块**

`stage3` 原来用 `float()` 解析用户给的帧率：

```python
_manual_fps = float(_manual_fps)          # "30000/1001" → ValueError
except (TypeError, ValueError):
    _manual_fps = 0.0                     # → 被当成"没指定"
```

于是输入最标准的 NTSC 帧率 `30000/1001` 反而被判成"用户没给帧率"，
直接 `STAGE3_FPS_UNAVAILABLE` → 退出 86。

而 `float("23.976")` 之后再反推有理数，会累积误差：
`23.976` 不是 `24000/1001`（差约 1.5e-5），长片会漂。

**做法**：用 `fractions.Fraction` 精确表示，全程不经过浮点。

支持：
- ``"30000/1001"``（分数）
- ``"25"``、``"30"``（整数）
- ``"23.976"``（小数 → 精确的 23976/1000，不做"猜标准值"）
- 前后空白

拒绝（给出具体原因，不是笼统失败）：
- 空值、无法解析、零分母、负数、零
- ``NaN`` / ``Inf``
- 超出合理范围（默认 1–240 fps，可调）
"""
from __future__ import annotations

from fractions import Fraction
from typing import Optional

#: 合理帧率范围。低于 1 或高于 240 基本是填错/单位错。
DEFAULT_MIN_RATE = 1.0
DEFAULT_MAX_RATE = 240.0


class RateSpecError(ValueError):
    """帧率规格不合法。``reason`` 是给用户看的具体原因。"""

    def __init__(self, spec, reason: str):
        super().__init__(f"{spec!r}: {reason}")
        self.spec = spec
        self.reason = reason


class Rate:
    """一个精确的有理数帧率 + 它的来源。"""

    __slots__ = ("num", "den", "raw", "source")

    def __init__(self, num: int, den: int, raw: str, source: str = "user"):
        if den == 0:
            raise RateSpecError(raw, "zero denominator")
        if num <= 0 or den < 0:
            raise RateSpecError(raw, "non-positive rate")
        self.num = int(num)
        self.den = int(den)
        self.raw = str(raw)
        self.source = source

    @property
    def value(self) -> float:
        """浮点近似 —— **只用于比较/显示**，不要拿它再反推有理数。"""
        return self.num / self.den

    def as_ffmpeg(self) -> str:
        """给 ffmpeg 的 `-framerate` / `-r` 用。整数也写成 `n/1`。"""
        return f"{self.num}/{self.den}"

    def __eq__(self, other) -> bool:
        if isinstance(other, Rate):
            # 交叉相乘比较，避免浮点
            return self.num * other.den == other.num * self.den
        return NotImplemented

    def __hash__(self) -> int:
        f = Fraction(self.num, self.den)
        return hash((f.numerator, f.denominator))

    def __repr__(self) -> str:      # pragma: no cover - 调试用
        return f"Rate({self.num}/{self.den}, {self.source!r})"


def parse_rate(spec, source: str = "user",
               min_rate: float = DEFAULT_MIN_RATE,
               max_rate: float = DEFAULT_MAX_RATE) -> Rate:
    """把用户/配置给的帧率规格解析成 :class:`Rate`。

    Args:
        spec: 字符串或数字。``"30000/1001"`` / ``"25"`` / ``23.976`` / ``25``。
        source: 来源标签（``user`` / ``manifest`` / ``probe`` …），
            会跟着结果走，便于报告里说明"这个值是听谁的"。
        min_rate / max_rate: 合理范围；超出即拒绝并说明。

    Returns:
        :class:`Rate`。

    Raises:
        RateSpecError: 规格不合法。消息里带具体原因。
    """
    if spec is None or (isinstance(spec, str) and not spec.strip()):
        raise RateSpecError(spec, "empty value")

    raw = str(spec).strip()
    # NaN / Inf 在 Fraction 里会抛 ValueError 或 OverflowError，
    # 这里先显式挡掉，好给出更准确的原因。
    low = raw.lower()
    if low in ("nan", "+nan", "-nan"):
        raise RateSpecError(spec, "NaN is not a frame rate")
    if low in ("inf", "+inf", "-inf", "infinity"):
        raise RateSpecError(spec, "Infinity is not a frame rate")

    try:
        frac = Fraction(raw)
    except (ValueError, ZeroDivisionError) as exc:
        if isinstance(exc, ZeroDivisionError) or "zero" in str(exc).lower():
            raise RateSpecError(spec, "zero denominator") from exc
        raise RateSpecError(
            spec, "not a number or a fraction like 30000/1001") from exc

    if frac.denominator == 0:            # pragma: no cover - Fraction 已挡
        raise RateSpecError(spec, "zero denominator")
    if frac.numerator <= 0:
        raise RateSpecError(spec, "rate must be positive")

    rate = Rate(frac.numerator, frac.denominator, raw, source)
    v = rate.value
    if v < min_rate or v > max_rate:
        raise RateSpecError(
            spec, f"rate {v:.4f} is outside the supported range "
                  f"[{min_rate}, {max_rate}]")
    return rate


def parse_rate_or_none(spec, source: str = "user",
                       min_rate: float = DEFAULT_MIN_RATE,
                       max_rate: float = DEFAULT_MAX_RATE
                       ) -> Optional[Rate]:
    """同 :func:`parse_rate`，但不合法时返回 ``None``。

    只给"可选覆盖"场景用；需要告诉用户为什么不行的地方请直接用
    :func:`parse_rate` 并捕获 :class:`RateSpecError`。
    """
    try:
        return parse_rate(spec, source=source, min_rate=min_rate,
                          max_rate=max_rate)
    except RateSpecError:
        return None
