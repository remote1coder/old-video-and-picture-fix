#!/usr/bin/env python3
"""Media I/O contracts: high-bit-depth image I/O + avifenc capability model.

FIX-V01/V02/V05: the encode boundary must agree with the encoder's documented
contract instead of relying on file extensions, exit codes or optimistic
defaults. Facts verified against the libavif manual (doc/avifenc.1.md) and the
libavif 1.4.0/1.4.1 changelog on 2026-10-02:

* avifenc accepts **only** jpg / jpeg / png / y4m input (no TIFF).
* ``--depth D[,EXTENSION]``: D must be 8, 10 or 12. A 16-bit result is only
  reachable through a bit-depth *extension* image; the supported combinations
  are (8,8), (12,4) and (12,8).
* Bit-depth extension (AVIF v1.2 Sample Transform) requires libavif >= 1.4.0;
  decoding the 16-bit result back requires a decoder that enables sample
  transforms (``avifdec --sato`` / ``--depth 16``).
* ``--icc FILENAME`` attaches an ICC payload; ``--cicp P/T/M`` uses ``2`` for
  "unspecified".

The module is dependency-optional: tifffile/imagecodecs/Pillow are used when
importable, otherwise pure-python writers are used. Every write reports the
method actually used, so callers record facts rather than intentions.
"""

from __future__ import annotations

import os
import re
import struct
import subprocess
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# --- avifenc contract constants (verified against libavif manual) -----------
AVIFENC_SUPPORTED_INPUT_EXTS = frozenset({".jpg", ".jpeg", ".png", ".y4m"})

# NEW-V-04: 中间产物目录里现在会有 .status.json / .cache.json 之类的侧车文件，
# glob 形如 `step5_postprocess*` 会把它们一起选进来，导致"选中的输入"其实是
# 一个 JSON 文本。候选必须先按真实图像扩展名过滤。
IMAGE_EXTS = frozenset({
    ".tif", ".tiff", ".png", ".jpg", ".jpeg", ".webp", ".bmp", ".y4m",
})
AVIFENC_BASE_DEPTHS = frozenset({8, 10, 12})
AVIFENC_DEPTH_EXTENSIONS = frozenset({(8, 8), (12, 4), (12, 8)})
#: libavif version that made the --depth extension (Sample Transform) stable
AVIFENC_DEPTH_EXTENSION_MIN_VERSION = (1, 4, 0)
AVIFENC_VALID_YUV = frozenset({"auto", "400", "420", "422", "444"})
AVIFENC_VALID_RANGE = frozenset({"full", "limited", "f", "l"})
AVIFENC_DEFAULT_DEPTH_EXTENSION = (12, 8)  # keeps 12-bit compatibility + 16-bit decode

PNG_COLOR_TYPE = {1: 0, 2: 4, 3: 2, 4: 6}  # channels -> PNG color type


class UnsupportedAvifConfig(ValueError):
    """Raised when a requested encode setting is not supported by avifenc."""


class MediaWriteError(RuntimeError):
    """Raised when an image cannot be written at the required precision."""


# --- image read -------------------------------------------------------------

def read_image_array(path: str) -> Tuple[Any, Dict[str, Any]]:
    """Read an image as a numpy array, preserving uint16 when present.

    Returns (array, meta). ``meta`` carries the reader actually used and any
    embedded ICC profile. tifffile is preferred because Pillow cannot represent
    multi-channel 16-bit data (its ``I;16`` mode is single-channel only).
    """
    meta: Dict[str, Any] = {"reader": None, "icc_profile": None,
                            "mode": None, "path": str(path)}
    try:
        import numpy as np  # noqa: F401
        import tifffile  # type: ignore
        arr = tifffile.imread(path)
        meta["reader"] = "tifffile"
        meta["mode"] = str(getattr(arr, "dtype", ""))
        return arr, meta
    except Exception:
        pass

    from PIL import Image  # noqa: WPS433 (optional dependency)
    with Image.open(path) as im:
        meta["mode"] = im.mode
        meta["icc_profile"] = im.info.get("icc_profile")
        import numpy as np
        arr = np.array(im)
    meta["reader"] = "pillow"
    return arr, meta


def _ensure_uint(arr: Any) -> Any:
    """Normalise to uint8/uint16 (float images are scaled to their range)."""
    import numpy as np
    if arr.dtype == np.uint8 or arr.dtype == np.uint16:
        return arr
    if arr.dtype.kind == "f":
        lo = float(arr.min()) if arr.size else 0.0
        hi = float(arr.max()) if arr.size else 1.0
        if lo >= 0.0 and hi <= 1.0:
            scaled = arr * 65535.0
        else:
            scaled = np.clip(arr, 0, 65535)
        return np.clip(scaled, 0, 65535).astype(np.uint16)
    if arr.dtype == np.int16 or arr.dtype == np.int32:
        return np.clip(arr, 0, 65535).astype(np.uint16)
    return np.clip(arr, 0, 65535).astype(np.uint16)


def read_png(path: str) -> Any:
    """Minimal PNG reader used to verify what we wrote (no external deps).

    Supports the subset our writer emits: 8/16-bit, color type 0/2/4/6,
    filter type 0 or adaptive filters (all 5 filter types), no interlace.
    """
    import numpy as np
    data = Path(path).read_bytes()
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG file")
    pos = 8
    idat = bytearray()
    width = height = bitdepth = colortype = None
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if ctype == b"IHDR":
            width, height, bitdepth, colortype, _comp, _filt, interlace = \
                struct.unpack(">IIBBBBB", body)
            if interlace != 0:
                raise ValueError("interlaced PNG not supported")
        elif ctype == b"IDAT":
            idat.extend(body)
        elif ctype == b"IEND":
            break
        pos += 12 + length
    if None in (width, height, bitdepth, colortype):
        raise ValueError("PNG header missing")
    raw = zlib.decompress(bytes(idat))
    channels = {0: 1, 2: 3, 3: 1, 4: 2, 6: 4}[colortype]
    bytes_per_sample = bitdepth // 8 or 1
    if bitdepth not in (8, 16):
        raise ValueError(f"unsupported PNG bit depth {bitdepth}")
    stride_bytes = width * channels * bytes_per_sample
    # FIX-NEW-V-02: PNG 滤波是 **逐字节** 定义的（PNG 规范 §6 滤波）：
    #   bpp = channels * bytes_per_sample（**字节**偏移），
    #   Sub:  Recon(x) = Filt(x) + Recon(x - bpp)   —— 每一步都 mod 256，
    #         低字节产生的进位 **丢弃**，绝不进到高字节。
    # 旧实现按 16-bit 字（mod 65536）做反滤波，进位被传播到高字节，
    # 于是每个「低字节相加溢出」的样本都会多出 +256。
    # 对自家 writer（filter 0）无感，但读 libpng/avifdec 产出的图会整幅偏色。
    bpp = channels * bytes_per_sample
    out = np.zeros((height, stride_bytes), dtype=np.uint8)
    prev = np.zeros(stride_bytes, dtype=np.uint8)
    p = 0
    for y in range(height):
        ftype = raw[p]
        p += 1
        line = np.frombuffer(raw, dtype=np.uint8,
                             count=stride_bytes, offset=p).copy()
        p += stride_bytes
        if ftype == 0:  # None
            pass
        elif ftype == 1:  # Sub —— 每条字节链独立累加，mod 256
            for r in range(bpp):
                line[r::bpp] = np.cumsum(line[r::bpp], dtype=np.uint8)
        elif ftype == 2:  # Up
            line += prev
        elif ftype == 3:  # Average
            for j in range(stride_bytes):
                a = int(line[j - bpp]) if j >= bpp else 0
                line[j] = (int(line[j]) + ((a + int(prev[j])) >> 1)) & 0xFF
        elif ftype == 4:  # Paeth
            for j in range(stride_bytes):
                a = int(line[j - bpp]) if j >= bpp else 0
                b = int(prev[j])
                c = int(prev[j - bpp]) if j >= bpp else 0
                pa, pb, pc = abs(b - c), abs(a - c), abs(a + b - 2 * c)
                pred = a if (pa <= pb and pa <= pc) else (b if pb <= pc else c)
                line[j] = (int(line[j]) + pred) & 0xFF
        else:
            raise ValueError(f"unsupported PNG filter {ftype}")
        out[y] = line
        prev = line
    # 采样重组：大端 16-bit 或 8-bit
    flat = out.reshape(height, width, channels, bytes_per_sample)
    if bitdepth == 16:
        arr = (flat[:, :, :, 0].astype(np.uint16) << 8) | \
            flat[:, :, :, 1].astype(np.uint16)
    else:
        arr = flat[:, :, :, 0]
    if channels == 1:
        arr = arr[:, :, 0]
    return arr


def write_png(path: str, arr: Any) -> str:
    """Write a PNG preserving uint8/uint16 precision. Returns the writer name.

    Pillow cannot write multi-channel 16-bit PNGs, so 16-bit arrays go through
    an internal zlib/struct writer. 8-bit arrays use Pillow when available.
    """
    import numpy as np
    arr = _ensure_uint(arr)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[2] not in PNG_COLOR_TYPE:
        raise MediaWriteError(f"unsupported channel count {arr.shape[2]}")
    if arr.dtype == np.uint16:
        return _write_png16_internal(path, arr)
    try:
        from PIL import Image
        if arr.shape[2] == 1:
            Image.fromarray(arr[:, :, 0], mode="L").save(path, format="PNG")
        elif arr.shape[2] == 3:
            Image.fromarray(arr, mode="RGB").save(path, format="PNG")
        else:
            Image.fromarray(arr, mode="RGBA").save(path, format="PNG")
        return "pillow"
    except Exception:
        return _write_png16_internal(path, arr.astype(np.uint16))


def _write_png16_internal(path: str, arr: Any) -> str:
    """Write a 16-bit PNG (filter 0, no interlace) with stdlib zlib only."""
    import numpy as np
    arr = np.ascontiguousarray(arr.astype(">u2"))
    h, w, c = arr.shape
    raw = bytearray()
    for y in range(h):
        raw.append(0)
        raw.extend(arr[y].tobytes())

    def chunk(tag: bytes, payload: bytes) -> bytes:
        return (struct.pack(">I", len(payload)) + tag + payload +
                struct.pack(">I", zlib.crc32(tag + payload) & 0xFFFFFFFF))

    ihdr = struct.pack(">IIBBBBB", w, h, 16, PNG_COLOR_TYPE[c], 0, 0, 0)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) +
           chunk(b"IDAT", zlib.compress(bytes(raw), 6)) + chunk(b"IEND", b""))
    Path(path).write_bytes(png)
    return "internal-png16"


# --- TIFF master ------------------------------------------------------------

def write_tiff(path: str, arr: Any) -> str:
    """Write a TIFF preserving uint8/uint16. Returns the writer actually used.

    tifffile is used when importable; otherwise an internal uncompressed TIFF
    writer is used so that a real 16-bit master is still produced (Pillow
    cannot write multi-channel 16-bit TIFF at all).
    """
    import numpy as np
    arr = _ensure_uint(arr)
    try:
        import tifffile  # type: ignore
        tifffile.imwrite(path, arr, compression="zlib")
        return "tifffile"
    except Exception:
        pass
    return _write_tiff_internal(path, arr)


def _write_tiff_internal(path: str, arr: Any) -> str:
    """Write an uncompressed, little-endian, uncompressed TIFF (RGB/RGBA/gray)."""
    import numpy as np
    arr = np.ascontiguousarray(arr)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    h, w, c = arr.shape
    if c not in (1, 3, 4):
        raise MediaWriteError(f"unsupported channel count {c}")
    if arr.dtype == np.uint8:
        raw = arr.tobytes()
        bits = 8
    elif arr.dtype == np.uint16:
        raw = arr.astype("<u2").tobytes()
        bits = 16
    else:
        raise MediaWriteError(f"unsupported dtype {arr.dtype}")

    entries: List[Tuple[int, int, int, bytes]] = []

    def _short(v: int) -> bytes:
        return struct.pack("<H", v)

    def _long(v: int) -> bytes:
        return struct.pack("<I", v)

    photometric = 1 if c == 1 else 2  # 1=BlackIsZero, 2=RGB
    tags: List[Tuple[int, int, int]] = [
        (256, 3, w),            # ImageWidth
        (257, 3, h),            # ImageLength
        (259, 3, 1),            # Compression = none
        (262, 3, photometric),  # PhotometricInterpretation
        (277, 3, c),            # SamplesPerPixel
        (278, 4, h),            # RowsPerStrip
        (284, 3, 1),            # PlanarConfiguration = chunky
    ]
    if c == 4:
        tags.append((339, 3, 2))  # ExtraSamples: unassociated alpha

    ifd_size = 2 + 12 * (len(tags) + 3) + 4  # + BitsPerSample/StripOffsets/ByteCounts
    header_size = 8
    bits_array_offset = header_size + ifd_size
    data_offset = bits_array_offset + (2 * c)
    data_offset += (4 - data_offset % 4) % 4  # keep word alignment
    byte_count = len(raw)

    for tag, ttype, value in tags:
        if ttype == 3:
            entries.append((tag, ttype, 1, _short(value).ljust(4, b"\x00")))
        else:
            entries.append((tag, ttype, 1, _long(value)))
    entries.append((258, 3, c, _long(bits_array_offset)))              # BitsPerSample
    entries.append((273, 4, 1, _long(data_offset)))                    # StripOffsets
    entries.append((279, 4, 1, _long(byte_count)))                     # StripByteCounts
    entries.sort(key=lambda e: e[0])

    out = bytearray()
    out.extend(b"II" + struct.pack("<H", 42) + struct.pack("<I", header_size))
    out.extend(struct.pack("<H", len(entries)))
    for tag, ttype, count, value in entries:
        out.extend(struct.pack("<HHI", tag, ttype, count) + value)
    out.extend(struct.pack("<I", 0))  # next IFD offset
    out.extend(b"\x00" * (bits_array_offset - len(out)))
    for _ in range(c):
        out.extend(_short(bits))
    out.extend(b"\x00" * (data_offset - len(out)))
    out.extend(raw)
    Path(path).write_bytes(bytes(out))
    return "internal-tiff"


def embed_icc_in_tiff(path: str, icc_bytes: bytes) -> bool:
    """Attach an ICC profile to an existing TIFF (tag 34675) when possible.

    Returns True when the profile was embedded by a known-good writer; False
    means the caller must record that no profile is attached.
    """
    try:
        import tifffile  # type: ignore
        import numpy as np
        arr = tifffile.imread(path)
        tifffile.imwrite(path, arr, compression="zlib",
                         extratags=[(34675, 1, len(icc_bytes), icc_bytes, True)])
        return True
    except Exception:
        return False


# --- avifenc capability / command ------------------------------------------

@dataclass
class AvifencCapability:
    """Result of probing the installed avifenc."""

    available: bool = False
    version: Tuple[int, int, int] = (0, 0, 0)
    raw: str = ""
    binary: str = "avifenc"

    @property
    def supports_depth_extension(self) -> bool:
        return (self.available and
                self.version >= AVIFENC_DEPTH_EXTENSION_MIN_VERSION)


def detect_avifenc_capability(binary: str = "avifenc") -> AvifencCapability:
    """Probe avifenc --version. Never raises; reports (0,0,0) when absent."""
    try:
        r = subprocess.run([binary, "--version"], capture_output=True,
                           text=True, timeout=15, shell=False)
    except Exception:
        return AvifencCapability(binary=binary)
    if r.returncode != 0:
        return AvifencCapability(binary=binary, raw=(r.stdout or "")[:200])
    out = (r.stdout or "") + (r.stderr or "")
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", out)
    if not m:
        return AvifencCapability(available=True, binary=binary, raw=out[:200])
    return AvifencCapability(
        available=True,
        version=(int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)),
        raw=out.strip()[:200], binary=binary)


def resolve_depth_spec(requested_depth: int,
                       requested_extension: Optional[int],
                       capability: AvifencCapability,
                       allow_downgrade: bool = False) -> Tuple[str, int, Dict[str, Any]]:
    """Map a requested bit depth onto an avifenc-supported ``--depth`` value.

    Returns ``(depth_arg, effective_decode_depth, info)``.
    Raises UnsupportedAvifConfig when the request is impossible and downgrade
    is not authorised — silently truncating 16 to 12 while still labelling the
    output "16-bit archival" is exactly the V01 defect.
    """
    info: Dict[str, Any] = {"requested_depth": requested_depth,
                            "requested_extension": requested_extension}
    try:
        depth = int(requested_depth)
    except (TypeError, ValueError) as exc:
        raise UnsupportedAvifConfig(f"invalid depth {requested_depth!r}") from exc

    if depth in AVIFENC_BASE_DEPTHS and requested_extension in (None, 0):
        info.update({"mode": "base", "downgraded": False})
        return str(depth), depth, info

    # A bit-depth extension image is requested (either explicitly, or by asking
    # for a 16-bit result): primary depth + extension bits reach 16 bits on
    # decode — see avifSampleTransformRecipe combinations (8,8) (12,4) (12,8).
    if depth == 16 or requested_extension not in (None, 0):
        if depth == 16:
            ext = int(requested_extension) if requested_extension else None
            combo = None
            if ext:
                combo = next((c for c in sorted(AVIFENC_DEPTH_EXTENSIONS,
                                                reverse=True) if c[1] == ext),
                             None)
            if combo is None:
                combo = AVIFENC_DEFAULT_DEPTH_EXTENSION
        else:
            combo = (depth, int(requested_extension))
        if combo not in AVIFENC_DEPTH_EXTENSIONS:
            raise UnsupportedAvifConfig(
                f"unsupported depth/extension combination {combo}; avifenc "
                f"supports {sorted(AVIFENC_DEPTH_EXTENSIONS)}")
        if not capability.supports_depth_extension:
            if not allow_downgrade:
                raise UnsupportedAvifConfig(
                    "a 16-bit AVIF result requires the avifenc --depth "
                    f"extension (libavif >= "
                    f"{'.'.join(map(str, AVIFENC_DEPTH_EXTENSION_MIN_VERSION))}); "
                    f"detected {capability.version or 'unknown'}. Set "
                    "allow_depth_downgrade=true to encode the primary depth "
                    "only (the output then IS 12-bit and must be recorded "
                    "as such).")
            info.update({"mode": "downgraded-primary", "downgraded": True,
                         "reason": "avifenc lacks depth-extension support",
                         "requested_combo": list(combo)})
            return str(combo[0]), combo[0], info
        info.update({"mode": "depth-extension", "downgraded": False,
                     "primary_depth": combo[0], "extension_depth": combo[1],
                     "decode_requires": "avifdec --sato (or --depth 16)"})
        return f"{combo[0]},{combo[1]}", 16, info

    raise UnsupportedAvifConfig(
        f"depth {depth} is not supported by avifenc (allowed base depths: "
        f"{sorted(AVIFENC_BASE_DEPTHS)}; 16 requires a depth extension)")


def _validate_choice(value: str, allowed: frozenset, name: str) -> str:
    if str(value).lower() not in allowed:
        raise UnsupportedAvifConfig(
            f"invalid {name}: {value!r} (allowed: {sorted(allowed)})")
    return str(value).lower()


def _validate_cicp(cicp: str) -> str:
    parts = str(cicp).split("/")
    if len(parts) != 3 or not all(p.strip().lstrip("-").isdigit() for p in parts):
        raise UnsupportedAvifConfig(
            f"invalid cicp: {cicp!r} (expected P/T/M, use 2 for unspecified)")
    return "/".join(parts)


def _validate_speed(speed: Any) -> str:
    s = str(speed)
    if s in ("default", "d"):
        return s
    try:
        iv = int(s)
    except (TypeError, ValueError) as exc:
        raise UnsupportedAvifConfig(f"invalid speed {speed!r}") from exc
    if not 0 <= iv <= 10:
        raise UnsupportedAvifConfig(f"speed out of range 0..10: {iv}")
    return str(iv)


def resolve_jobs_value(jobs: str, capability: AvifencCapability) -> Optional[str]:
    """Resolve --jobs per avifenc version (FIX-P12-10 behaviour preserved)."""
    if jobs != "all":
        return str(jobs)
    if capability.version >= (1, 1, 0):
        return "all"
    if capability.version >= (0, 9, 2):
        return str(min(os.cpu_count() or 1, 32))
    return None


def build_avifenc_cmd(input_path: str, output_path: str, params: Dict[str, Any],
                      use_rgb444: bool = False,
                      capability: Optional[AvifencCapability] = None,
                      icc_path: Optional[str] = None) -> Tuple[List[str], Dict[str, Any]]:
    """Build a validated avifenc command.

    Returns (cmd, contract) where ``contract`` records what will actually be
    encoded (not what the user hoped for). Raises UnsupportedAvifConfig for
    unsupported/invalid combinations so the caller can fail the stage instead
    of passing a command the encoder will reject.
    """
    capability = capability or detect_avifenc_capability()
    input_ext = Path(input_path).suffix.lower()
    if input_ext not in AVIFENC_SUPPORTED_INPUT_EXTS:
        raise UnsupportedAvifConfig(
            f"avifenc does not accept {input_ext or 'extension-less'} input; "
            f"supported: {sorted(AVIFENC_SUPPORTED_INPUT_EXTS)} "
            "(convert with prepare_encoder_input first)")
    src = params.get("avif_rgb444_params", {}) if use_rgb444 else params.get("avif_params", {})
    src = src or {}
    if use_rgb444:
        depth = src.get("depth", 16)
        yuv = src.get("yuv", "444")
        cicp = src.get("cicp", "1/13/0")
        lossless = bool(src.get("lossless", True))
    else:
        depth = src.get("depth", 10)
        yuv = src.get("yuv", "420")
        cicp = src.get("cicp", "1/13/1")
        lossless = bool(src.get("lossless", False))
    rng = src.get("range", "limited")
    # NEW-V-05（真机实测 avifenc 1.4.2）：lossless 模式下 `--range limited`
    # 会被直接拒绝：stderr = "Range has to be full in lossless mode."
    # 归档档（rgb444）默认就是 lossless，因此必须强制 full，否则整条 stage6
    # 在真实编码器上必然 exit 84 —— 单测里的 stub 编码器永远不会暴露这一点。
    if lossless and str(rng).lower() != "full":
        rng = "full"
    range_forced = lossless and str(src.get("range", "limited")).lower() != "full"
    speed = src.get("speed", 4)
    jobs = src.get("jobs", "all")
    extension = src.get("depth_extension")
    allow_downgrade = bool(params.get("allow_depth_downgrade", False))

    depth_arg, effective_depth, depth_info = resolve_depth_spec(
        depth, extension, capability, allow_downgrade=allow_downgrade)

    cmd = ["avifenc",
           "--depth", depth_arg,
           "--yuv", _validate_choice(yuv, AVIFENC_VALID_YUV, "yuv"),
           "--range", _validate_choice(rng, AVIFENC_VALID_RANGE, "range"),
           "--cicp", _validate_cicp(cicp),
           "--speed", _validate_speed(speed)]
    if lossless:
        cmd.append("--lossless")
    jobs_val = resolve_jobs_value(jobs, capability)
    if jobs_val is not None:
        cmd.extend(["--jobs", jobs_val])
    if icc_path:
        cmd.extend(["--icc", str(icc_path)])
    cmd.extend([str(input_path), str(output_path)])

    contract = {
        "input_path": str(input_path),
        "input_ext": Path(input_path).suffix.lower(),
        "archival_rgb444": bool(use_rgb444),
        "depth_arg": depth_arg,
        "requested_depth": depth,
        "effective_decode_depth": effective_depth,
        "depth_info": depth_info,
        "yuv": str(yuv),
        "range": str(rng),
        "range_forced_full_by_lossless": range_forced,
        "cicp": str(cicp),
        "speed": str(speed),
        "lossless": lossless,
        "icc_attached": bool(icc_path),
        "avifenc_version": ".".join(map(str, capability.version)),
        "avifenc_available": capability.available,
    }
    return cmd, contract


# --- encoder input adaptation ----------------------------------------------

def prepare_encoder_input(path: str, workdir: str,
                          logger: Any = None,
                          forbid_lossy_input: bool = True) -> Dict[str, Any]:
    """Return an avifenc-acceptable input path (FIX-V05).

    TIFF (the photo pipeline's default stage5 output) is NOT an avifenc input
    format. Conversion keeps the real pixel precision: 16-bit data is written
    as a 16-bit PNG (never silently downgraded to 8-bit JPEG/PNG).
    """
    p = Path(path)
    ext = p.suffix.lower()
    result: Dict[str, Any] = {
        "path": str(path), "converted": False, "source_ext": ext,
        "source_dtype": None, "verification": None, "writer": None,
        "reason": None,
    }
    if ext in AVIFENC_SUPPORTED_INPUT_EXTS:
        if forbid_lossy_input and ext in (".jpg", ".jpeg"):
            result["reason"] = "lossy input accepted by avifenc but lossy"
        arr, meta = read_image_array(path)
        result["source_dtype"] = str(getattr(arr, "dtype", ""))
        result["verification"] = "passthrough"
        return result

    arr, meta = read_image_array(path)
    result["source_dtype"] = str(getattr(arr, "dtype", ""))
    os.makedirs(workdir, exist_ok=True)
    out_path = str(Path(workdir) / f"{p.stem}_enc_input.png")
    writer = write_png(out_path, arr)
    result.update({"path": out_path, "converted": True, "writer": writer})

    # Verify the converted input actually round-trips at the same precision.
    verify_arr = None
    verify_via = None
    try:
        import imagecodecs  # type: ignore  (libpng-backed, independent decoder)
        verify_arr = imagecodecs.imread(out_path)
        verify_via = "imagecodecs(libpng)"
    except Exception:
        try:
            verify_arr = read_png(out_path)
            verify_via = "internal-png-reader"
        except Exception:
            verify_arr = None
    if verify_arr is None:
        raise MediaWriteError(
            f"encoder input conversion could not be verified: {out_path}")
    ok = (tuple(verify_arr.shape) == tuple(arr.shape) and
          verify_arr.dtype == _ensure_uint(arr).dtype and
          bool((verify_arr == _ensure_uint(arr)).all()))
    result["verification"] = f"{verify_via}:{'exact' if ok else 'MISMATCH'}"
    if not ok:
        raise MediaWriteError(
            f"encoder input conversion changed pixel data ({result['verification']})")
    if logger:
        logger.info("STAGE6_INPUT_ADAPTER",
                    f"Converted {ext or 'unknown'} -> PNG for avifenc",
                    source=path, converted=out_path, writer=writer,
                    dtype=result["source_dtype"],
                    verification=result["verification"])
    return result
