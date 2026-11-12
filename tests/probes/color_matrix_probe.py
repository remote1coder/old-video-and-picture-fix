# -*- coding: utf-8 -*-
"""N21：4:2:0 / 4:4:4 × bt601 / bt709 × full / limited 的颜色往返误差。

## 这个探针回答什么

审核 R19 §4.2：

> 不要把 420 中间格式当成已定结论。

现行中间格式固定 `yuv420p16le`。这是一次**信息有损**的降采样，而规范里
从来没写过"420 到底损失了多少、在什么条件下可以接受"。本探针给出**数字**，
让"420 现状"和"验收结论"能分开写。

## 方法（不涉及 AI、不涉及业务链路，纯 ffmpeg）

1. 造参考：`smptebars` 渲成 **rgb24** FFV1（无子采样），这就是"真值"。
2. 对每个组合：
   - 真值 RGB → YUV（指定 matrix / range）→ 16 位 FFV1 中间件（420 或 444）
   - 中间件 → RGB（**同一组** matrix / range）
   - 与真值逐帧比 PSNR（在 RGB 域，三个平面各自报）
3. 另测**标签错配**：按 A 编码、按 B 解码 —— 这是"偏色"最常见的来源，
   量出它比子采样误差大多少。

## 判据

- `444 + 同参数往返` 必须是 `inf`（无损）—— 否则说明命令本身有问题，
  整轮结论作废；
- 420 的损失用 **RGB 域 PSNR** 表达（dB），并换算出 8-bit 码值当量；
- 标签错配的损失**必须显著大于**同参数 420 的损失，否则说明"标签"在这条
  路径上不起作用，得回去查 scale 参数是否真的生效。

## 用法

    python tests/probes/color_matrix_probe.py
    python tests/probes/color_matrix_probe.py --work E:\\..\\n21 --size 1280x720
"""
from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

#: 有损组合：chroma 子采样 × 矩阵 × 量化范围
#: 8 种基本组合：chroma × matrix × range 全交叉。
#: R21-C09：审核指出旧表缺 BT.601 **full** 与 444 的对应组合。
COMBOS = [
    ("444", "bt709", "limited"),
    ("444", "bt709", "full"),
    ("444", "bt601", "limited"),
    ("444", "bt601", "full"),
    ("420", "bt709", "limited"),
    ("420", "bt709", "full"),
    ("420", "bt601", "limited"),
    ("420", "bt601", "full"),
]

#: 标签错配：按 (编码) 写、按 (解码) 读
MISMATCH = [
    ("bt709", "limited", "bt601", "limited"),
    ("bt601", "limited", "bt709", "limited"),
    ("bt709", "limited", "bt709", "full"),
]


def sh(cmd, timeout=600):
    return subprocess.run(cmd, capture_output=True, text=True,
                          errors="replace", timeout=timeout, shell=False)


def _need_tool(name):
    try:
        r = sh([name, "-version"], timeout=30)
    except Exception:
        return False
    return r.returncode == 0


def make_reference(work, size, frames):
    """造一段**已知 RGB 真值**的 FFV1（rgb24，无子采样）。"""
    out = os.path.join(work, "ref_rgb.mkv")
    r = sh(["ffmpeg", "-hide_banner", "-v", "error", "-y",
            "-f", "lavfi",
            "-i", f"smptebars=size={size}:rate=30:duration={frames / 30.0:.4f}",
            # 再叠一层渐变，让色度不是"只有几块纯色"——纯色块会掩盖子采样误差
            "-vf", "format=rgb24,geq=r='r(X,Y)':g='g(X,Y)*0.7+70*X/W':"
                   "b='b(X,Y)*0.7+70*Y/H'",
            "-c:v", "ffv1", "-pix_fmt", "rgb24", out])
    if r.returncode != 0:
        raise RuntimeError(f"造参考失败: {(r.stderr or '')[:300]}")
    return out


def to_yuv(src, dst, chroma, matrix, rng):
    """RGB 真值 → YUV 中间件（16 位 FFV1）。"""
    pix = "yuv420p16le" if chroma == "420" else "yuv444p16le"
    vf = (f"scale=out_color_matrix={matrix}:out_range={rng},"
          f"format={pix}")
    r = sh(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", src,
            "-vf", vf, "-c:v", "ffv1", "-pix_fmt", pix, dst])
    if r.returncode != 0:
        raise RuntimeError(f"{chroma}/{matrix}/{rng} 编码失败: "
                           f"{(r.stderr or '')[:300]}")
    return dst


def back_to_rgb(src, dst, matrix, rng):
    """YUV 中间件 → RGB（用**指定**参数解释）。"""
    vf = (f"scale=in_color_matrix={matrix}:in_range={rng},format=rgb24")
    r = sh(["ffmpeg", "-hide_banner", "-v", "error", "-y", "-i", src,
            "-vf", vf, "-c:v", "ffv1", "-pix_fmt", "rgb24", dst])
    if r.returncode != 0:
        raise RuntimeError(f"{matrix}/{rng} 解码失败: {(r.stderr or '')[:300]}")
    return dst


def psnr_rgb(a, b, expected_frames=None):
    """RGB 域逐帧 PSNR **与逐帧 MSE**（R21-C09 重写）。

    旧实现的四处硬伤（审核逐条点名）：

    1. 不看 `returncode` —— 比较进程失败仍然继续解析空输出；
    2. 不检查每行是否**每个平面都有值** —— 缺一个平面照样算平均；
    3. 不检查帧号是否 1..N 连续唯一 —— 重复行/缺行不被发现；
    4. 统计用"先平均 dB 再反推 RMSE" —— 那等价于对 MSE 作几何平均，
       与"先累计平方误差再求总体 RMSE"不是一回事。

    现在返回逐帧 `mse_*`，由 `aggregate()` 用**总平方误差 / 总样本数**
    求总体统计量。

    Returns:
        ``{rc, frames, rows, problems, seq}``；``problems`` 非空即不可采信。
    """
    r = sh(["ffmpeg", "-hide_banner", "-v", "info", "-i", a, "-i", b,
            "-lavfi", "[0:v]format=rgb24[x];[1:v]format=rgb24[y];"
                      "[x][y]psnr=stats_file=-", "-f", "null", "-"])
    problems = []
    rows = []
    for ln in (r.stdout or "").splitlines():
        if not ln.startswith("n:"):
            continue
        row = {}
        for tok in ln.split():
            if ":" in tok:
                k, v = tok.split(":", 1)
                row[k] = v
        rows.append(row)

    if r.returncode != 0:
        problems.append(f"comparison rc={r.returncode}: "
                        f"{(r.stderr or '')[-200:]}")

    seq = []
    for i, row in enumerate(rows):
        try:
            seq.append(int(row.get("n")))
        except (TypeError, ValueError):
            problems.append(f"row {i + 1} has no parsable frame number")
        for p in ("r", "g", "b"):
            if not row.get(f"mse_{p}") and not row.get(f"psnr_{p}"):
                problems.append(f"row {i + 1} lacks plane {p}")
                break
    if seq and seq != list(range(1, len(seq) + 1)):
        problems.append("frame numbers are not 1..N contiguous "
                        "(duplicate/missing rows are not evidence)")
    if expected_frames is not None and len(rows) != int(expected_frames):
        problems.append(f"compared {len(rows)} of {expected_frames} frames")
    if not rows:
        problems.append("no psnr rows at all")

    return {"rc": r.returncode, "frames": len(rows), "rows": rows,
            "seq": seq, "problems": problems}


def _raw_rgb48(path):
    """把视频解成 rgb48le 原始字节（**不做**任何 8-bit 归一）。

    注意：这里**不能**用 `sh()` —— 它是 text 模式（`text=True`），
    二进制帧会被当字符串返回，`array.frombytes` 立刻 TypeError。
    这条曾经让"精确比较"在真正跑起来时直接崩（不是判错，是崩）。
    """
    r = subprocess.run(["ffmpeg", "-hide_banner", "-v", "error", "-i",
                        str(path), "-f", "rawvideo", "-pix_fmt", "rgb48le",
                        "-"], capture_output=True, timeout=600, shell=False)
    if r.returncode != 0:
        return None, ("decode rc=%s: %s"
                      % (r.returncode,
                         (r.stderr or b"")[-200:].decode("utf-8", "replace")))
    return r.stdout or b"", ""


def rgb48_exact_diff(a, b):
    """**16-bit 精确比较**：两侧都解成 rgb48le，逐样本比。

    审核 R22-B11 的真实反例：两张 16×16 的 16-bit PPM 只差一个 R 码值，
    `psnr` 滤镜把两边都 `format=rgb24` 之后得到全 inf —— 于是"原始 16-bit
    相同"被判成立。**在 rgb24 下相同是可能的，但不能据此宣称 16-bit 相同。**

    Returns:
        ``{ok, samples, differing, max_code_diff, problems}``；
        ``ok=True`` 才表示"在这条表示上逐样本相同"。
    """
    out = {"ok": False, "samples": 0, "differing": 0, "max_code_diff": 0,
           "problems": []}
    da, ea = _raw_rgb48(a)
    db, eb = _raw_rgb48(b)
    if da is None or db is None:
        out["problems"].append(f"cannot decode: {ea or eb}")
        return out
    if not da or not db:
        out["problems"].append("decoded stream is empty")
        return out
    if len(da) != len(db):
        out["problems"].append(
            f"byte length differs: {len(da)} vs {len(db)} (frame count or "
            f"size differs; an exact comparison is impossible)")
        return out
    import array
    aa = array.array("H")
    aa.frombytes(da)
    bb = array.array("H")
    bb.frombytes(db)
    if sys.byteorder != "little":       # pragma: no cover - Windows 都是 LE
        aa.byteswap()
        bb.byteswap()
    diff = 0
    mx = 0
    for x, y in zip(aa, bb):
        if x != y:
            diff += 1
            d = abs(int(x) - int(y))
            if d > mx:
                mx = d
    out.update(samples=len(aa), differing=diff, max_code_diff=mx,
               ok=(diff == 0))
    return out


def _num(tok, *, zero_means=None):
    """解析统计值。

    R22-A08：**MSE 与 PSNR 不能共用 inf 语义**。审核的反例是
    `mse_r: inf` 被按"PSNR 正无穷 = 零误差"的规则转成 None，再转 0，
    最后判 OK —— 一个明显异常的数值变成了"完美"。

    Args:
        tok: 文本值。
        zero_means: ``None`` 表示"这个字段的正无穷 = 0"（只有 PSNR 这样）；
            ``"bad"`` 表示"正无穷是非法值"（MSE 就是这样）。

    Returns:
        float / ``None``（表示零误差）/ ``"bad"``。
    """
    t = str(tok or "").strip().lower()
    if t in ("inf", "+inf", "infinity", "+infinity"):
        return None if zero_means is None else "bad"
    if t in ("-inf", "-infinity"):
        return "bad"
    try:
        v = float(t)
    except (TypeError, ValueError):
        return "bad"
    if not math.isfinite(v) or v < 0:
        return "bad"
    return v


def _mse_from_psnr(psnr):
    """由 PSNR(8-bit 满量程) 反推 MSE —— 显示精度丢掉的小值靠这一步找回。

    审核的真实反例：两张 64×64 PPM 只差一个像素 R 100→101，
    正确 MSE = 1/4096 ≈ 0.000244；ffmpeg 日志写的是 `mse_r:0.00`
    （只有 2 位小数）、`psnr_r:84.25`。把 "0.00" 当精确零就会得出
    PSNR=inf —— 抹掉真实差异。用 psnr 反推：255²/10^(84.25/10) ≈ 0.000244。
    """
    if psnr is None:
        return 0.0
    return (255.0 ** 2) / (10.0 ** (float(psnr) / 10.0))


def aggregate(res):
    """用**总平方误差 / 总样本数**求各通道 MSE、RMSE、PSNR（R21-C09）。

    为什么不能先平均 dB：dB 是对数域，平均 dB 再反推 RMSE 相当于对 MSE
    作几何平均；动态素材里个别极差帧会被系统性低估。

    R22-A08 增加三件事：

    1. MSE 字段的正无穷判**非法**（不是零误差）；
    2. 当 MSE 显示为 0 但 PSNR 有限时，**用 PSNR 反推 MSE** ——
       显示精度下的 "0.00" 不能当精确零；
    3. 输出 `exact_zero`：只有**每一帧** PSNR 都是 inf 才为真。
       "精确相等"只能由它来判，不能由汇总出的 mse==0 反推。
    """
    out = {"frames": res.get("frames", 0), "planes": {},
           "problems": list(res.get("problems") or [])}
    rows = res.get("rows") or []
    for p in ("r", "g", "b"):
        mses = []
        bad = 0
        exact = True
        reconstructed = 0
        for row in rows:
            m = _num(row.get(f"mse_{p}"), zero_means="bad")
            ps = _num(row.get(f"psnr_{p}"))
            if m == "bad" or ps == "bad":
                bad += 1
                exact = False
                continue
            if ps is None:
                # R23-B11：PSNR=inf 与"MSE 是一个正数"互相矛盾。
                # 旧实现无条件按"零误差"记 0，于是伪造的
                # `mse_r:1 psnr_r:inf` 被判成 exact_zero=True。
                if m is not None and float(m) > 0:
                    out["problems"].append(
                        f"plane {p}: contradictory statistics "
                        f"(mse={row.get('mse_' + p)} but psnr=inf)")
                    bad += 1
                    exact = False
                    continue
                mses.append(0.0)
                continue
            exact = False
            if m is None or float(m) == 0.0:
                # MSE 显示成 0（或缺失）但 PSNR 有限 → 用 PSNR 反推
                mses.append(_mse_from_psnr(ps))
                reconstructed += 1
            else:
                mses.append(float(m))
        if bad or not mses:
            out["problems"].append(
                f"plane {p}: {bad} unparsable/illegal value(s), "
                f"{len(mses)} usable")
            out["planes"][p] = {"mse": None, "rmse": None, "psnr": None,
                                "frames": len(mses), "exact_zero": False}
            continue
        mse = sum(mses) / len(mses)          # 每帧像素数相同 → 样本均值
        rmse = mse ** 0.5
        psnr = ("inf" if mse == 0
                else round(10.0 * math.log10((255.0 ** 2) / mse), 4))
        out["planes"][p] = {
            "mse": round(mse, 9), "rmse": round(rmse, 4),
            "psnr": psnr, "frames": len(mses),
            # 只有全部帧 PSNR=inf 才是"精确相等"
            "exact_zero": bool(exact),
            # 有多少帧的 MSE 是靠 PSNR 反推回来的（显示精度不够）
            "mse_from_psnr_frames": reconstructed,
        }
    return out


def evaluate(results, expected_frames=None, expected_combos=None):
    """把一轮 N21 的原始结果判成 OK / SUSPECT / INVALID（R21-C09）。

    审核："必须先修退出码、完整帧/平面、有限数及错误传播，
    才能让新数据承担验收责任。"

    Returns:
        ``(verdict, reasons)``。``verdict`` ∈ ``OK`` / ``SUSPECT`` / ``INVALID``。
    """
    reasons = []
    for row in results.get("roundtrip", []) + results.get("mismatch", []):
        if row.get("error"):
            reasons.append(f"{row.get('combo')}: {row['error']}")
        for p in row.get("problems") or []:
            reasons.append(f"{row.get('combo')}: {p}")
    if reasons:
        return "INVALID", reasons

    rt = results.get("roundtrip", [])
    if expected_combos is not None and len(rt) != expected_combos:
        reasons.append(f"only {len(rt)} of {expected_combos} combos ran")
        return "INVALID", reasons

    # ---- R23-B11：组合必须是**唯一身份**的完整集合 ----
    # 审核反例：8 份同名、零帧、空平面的 420 组合 + 空 mismatch，
    # `evaluate(expected_frames=6, expected_combos=8)` 仍然 OK ——
    # 旧实现只看"条数对不对"，不看这些条目到底是什么。
    _ids = [str(r.get("combo") or "") for r in rt]
    if any(not i for i in _ids):
        reasons.append("a roundtrip row has no combo identity")
    if len(set(_ids)) != len(_ids):
        reasons.append("duplicate combo identities (the same combo was "
                       "counted more than once)")
    for row in rt:
        _pl = row.get("planes") or {}
        for p in ("r", "g", "b"):
            _p = _pl.get(p) or {}
            if _p.get("mse") is None:
                reasons.append(f"{row.get('combo')}: plane {p} has no usable "
                               f"measurement")
                continue
            if expected_frames is not None and \
                    int(_p.get("frames") or 0) != int(expected_frames):
                reasons.append(
                    f"{row.get('combo')}: plane {p} compared "
                    f"{_p.get('frames')} of {expected_frames} frames")
        # 声称"精确相等"必须有 16-bit 原始比较支撑
        _claims_exact = any(((_pl.get(p) or {}).get("exact_zero"))
                            for p in ("r", "g", "b"))
        if _claims_exact:
            _ex = row.get("exact16") or {}
            if not _ex:
                reasons.append(
                    f"{row.get('combo')}: claims exact equality but no "
                    f"16-bit comparison was performed")
            elif not _ex.get("ok"):
                reasons.append(
                    f"{row.get('combo')}: claims exact equality but the "
                    f"16-bit comparison differs "
                    f"({_ex.get('differing')} sample(s), max code diff "
                    f"{_ex.get('max_code_diff')})")
    if reasons:
        return "INVALID", reasons

    # ---- 错配组合必须**真的检出**差异 ----
    for row in results.get("mismatch", []):
        _g = ((row.get("planes") or {}).get("g") or {})
        if float(_g.get("mse") or 0.0) <= 0.0:
            reasons.append(
                f"{row.get('combo')}: the mismatched decode produced no "
                f"measurable difference; the case did not exercise anything")
    if reasons:
        return "INVALID", reasons

    # ---- 444 往返：按**误差预算**判，不是"必须精确相等" ----
    # R22-A08：审核指出旧判据把所有 RGB→YUV444→RGB 往返都当成"同表示无操作"。
    # 它们实际经过矩阵、范围、量化和舍入 —— 精确相等是**可选的更好结果**，
    # 不是唯一正确答案。这里给一个明确预算（8-bit 码值 RMSE ≤ 1.0），
    # 超出才说明转换链路不对。
    _RT_BUDGET_RMSE = 1.0
    for row in rt:
        if row.get("chroma") != "444":
            continue
        for p in ("r", "g", "b"):
            _pl = row["planes"][p]
            if _pl.get("mse") is None:
                reasons.append(
                    f"{row['combo']}: 444 roundtrip has no usable {p} MSE "
                    f"(unparsable/illegal values)")
                continue
            if float(_pl.get("rmse") or 0.0) > _RT_BUDGET_RMSE:
                reasons.append(
                    f"{row['combo']}: 444 roundtrip exceeds the error budget "
                    f"on {p} (RMSE {_pl.get('rmse')} > {_RT_BUDGET_RMSE})")
    if reasons:
        return "INVALID", reasons

    if not any(r.get("chroma") == "420" for r in rt):
        return "INVALID", ["no 420 combo ran; N21's question is unanswered"]
    return "OK", []


def db_to_8bit(db):
    """PSNR(dB, 8-bit 满量程) → 等效 RMSE 的 8-bit 码值。"""
    if db is None or db == "inf":
        return 0.0
    return round(255.0 * (10 ** (-db / 20.0)), 4)


def main():
    ap = argparse.ArgumentParser(description="N21 colour round-trip matrix")
    ap.add_argument("--work", default=None,
                    help="输出根（默认在系统临时目录下 mkdtemp，绝不覆盖已有目录）")
    ap.add_argument("--size", default="640x360")
    ap.add_argument("--frames", type=int, default=10)
    args = ap.parse_args()

    for t in ("ffmpeg", "ffprobe"):
        if not _need_tool(t):
            print(f"  [错误] 找不到 {t}")
            return 2

    if args.work:
        work = os.path.abspath(args.work)
        if os.path.isdir(work) and os.listdir(work):
            print(f"  [错误] --work 已存在且非空，拒绝覆盖：{work}")
            return 2
        os.makedirs(work, exist_ok=True)
    else:
        work = tempfile.mkdtemp(prefix="n21_colour_")
    print(f"  work  : {work}")
    print(f"  size  : {args.size}   frames: {args.frames}")

    ref = make_reference(work, args.size, args.frames)
    print(f"  ref   : {ref}")

    results = {"work": work, "reference": ref, "roundtrip": [],
               "mismatch": []}

    # ---- ① 同参数往返 ----
    for chroma, matrix, rng in COMBOS:
        tag = f"{chroma}_{matrix}_{rng}"
        mid = os.path.join(work, f"mid_{tag}.mkv")
        back = os.path.join(work, f"back_{tag}.mkv")
        try:
            to_yuv(ref, mid, chroma, matrix, rng)
            back_to_rgb(mid, back, matrix, rng)
            m = aggregate(psnr_rgb(ref, back, expected_frames=args.frames))
        except Exception as exc:
            print(f"  [失败] {tag}: {exc}")
            results["roundtrip"].append({"combo": tag, "error": str(exc)})
            continue
        # R23-B11：**16-bit 精确比较**。`psnr` 滤镜两边都降到 rgb24，
        # 那只能证明"在 8-bit 表示上相同"，不能证明原始高位深相同。
        _ex = rgb48_exact_diff(ref, back)
        row = {"combo": tag, "chroma": chroma, "matrix": matrix,
               "range": rng, "exact16": _ex, **m,
               "rmse8": {p: (None if m["planes"][p]["rmse"] is None
                             else round(m["planes"][p]["rmse"], 4))
                         for p in ("r", "g", "b")}}
        results["roundtrip"].append(row)
        print("  {:<22} r={:<10} g={:<10} b={:<10}".format(
            tag,
            str(m["planes"]["r"]["psnr"])[:8],
            str(m["planes"]["g"]["psnr"])[:8],
            str(m["planes"]["b"]["psnr"])[:8]))

    # ---- ② 标签错配（按编码 / 按解码）----
    for enc_m, enc_r, dec_m, dec_r in MISMATCH:
        tag = f"enc{enc_m}_{enc_r}__dec{dec_m}_{dec_r}"
        mid = os.path.join(work, f"mid_mm_{tag}.mkv")
        back = os.path.join(work, f"back_mm_{tag}.mkv")
        try:
            to_yuv(ref, mid, "444", enc_m, enc_r)
            back_to_rgb(mid, back, dec_m, dec_r)
            m = aggregate(psnr_rgb(ref, back, expected_frames=args.frames))
        except Exception as exc:
            print(f"  [失败] {tag}: {exc}")
            results["mismatch"].append({"combo": tag, "error": str(exc)})
            continue
        row = {"combo": tag, "encode": [enc_m, enc_r],
               "decode": [dec_m, dec_r], **m,
               "rmse8": {p: (None if m["planes"][p]["rmse"] is None
                             else round(m["planes"][p]["rmse"], 4))
                         for p in ("r", "g", "b")}}
        results["mismatch"].append(row)
        print("  {:<34} r={:<10} g={:<10} b={:<10}".format(
            tag,
            str(m["planes"]["r"]["psnr"])[:8],
            str(m["planes"]["g"]["psnr"])[:8],
            str(m["planes"]["b"]["psnr"])[:8]))

    # ---- ③ 判据（R21-C09：改用统一的 evaluate）----
    print("\n  === 判定 ===")
    verdict, reasons = evaluate(results, expected_frames=args.frames,
                                expected_combos=len(COMBOS))
    if verdict == "OK":
        # 错配应当**明显大于**同参数 420 的损失；否则说明标签没真正生效
        base420 = None
        for r in results["roundtrip"]:
            if (r.get("chroma") == "420" and r.get("matrix") == "bt709"
                    and r.get("range") == "limited" and not r.get("error")):
                base420 = r["planes"]["g"]["mse"]
        for r in results["mismatch"]:
            if r.get("error") or base420 in (None, 0):
                continue
            mse = r["planes"]["g"]["mse"]
            if isinstance(mse, (int, float)) and mse <= base420:
                verdict = "SUSPECT"
                reasons.append(
                    f"{r['combo']} 的 G 通道 MSE({mse}) 不高于同参数 420"
                    f"({base420}) —— 标签可能没生效，需复查 scale 参数")
    print(f"    verdict = {verdict}")
    for x in reasons:
        print(f"      - {x}")

    out = os.path.join(work, "n21_result.json")
    results["verdict"] = verdict
    results["reasons"] = reasons
    with open(out, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"    json    = {out}")
    # 退出码约定（R21-C09）：0=OK，2=SUSPECT（结论不可用于验收），1=INVALID
    return {"OK": 0, "SUSPECT": 2}.get(verdict, 1)


if __name__ == "__main__":
    sys.exit(main())
