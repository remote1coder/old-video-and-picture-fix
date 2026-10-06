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


class MediaPrecisionError(RuntimeError):
    """RV-03: raised when a file's real precision cannot be preserved.

    Pillow represents a 16-bit RGB PNG/TIFF as ``mode='RGB'`` backed by uint8
    (the high byte of each sample). Treating that as "the file is 8-bit" is
    exactly the silent precision loss this module exists to prevent, so the
    read fails loudly unless the caller explicitly authorises a downgrade.
    """


# --- ICC extraction (RV-04: colour provenance must survive every boundary) --

def _png_icc_profile(path: str) -> Optional[bytes]:
    """Return the ICC payload of a PNG's ``iCCP`` chunk (zlib-compressed)."""
    try:
        data = Path(path).read_bytes()
    except Exception:
        return None
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        return None
    pos = 8
    while pos + 8 <= len(data):
        length = struct.unpack(">I", data[pos:pos + 4])[0]
        ctype = data[pos + 4:pos + 8]
        body = data[pos + 8:pos + 8 + length]
        if ctype == b"iCCP":
            try:
                # iCCP = profile name (1..79 bytes, NUL-terminated) +
                #        compression method (1 byte) + compressed profile
                nul = body.find(b"\x00")
                if nul < 0:
                    return None
                compressed = body[nul + 2:]
                return zlib.decompress(compressed)
            except Exception:
                return None
        if ctype == b"IEND":
            break
        pos += 12 + length
    return None


def _png_bit_depth(path: str) -> Optional[int]:
    """Bit depth declared in the PNG IHDR (used for the precision guard)."""
    try:
        with open(path, "rb") as f:
            head = f.read(26)
        if head[:8] != b"\x89PNG\r\n\x1a\n":
            return None
        return int(head[24])
    except Exception:
        return None


def _tiff_read(path: str) -> Tuple[Any, Optional[bytes]]:
    """Read a TIFF with tifffile, returning (array, icc_bytes or None).

    RV-04: the ORIGINAL reader took ``tifffile.imread`` only and returned
    ``icc_profile=None``, so a TIFF carrying a real ICC produced a
    profile-less archival master. Tag 34675 is the ICC profile tag.
    """
    import tifffile  # type: ignore
    icc: Optional[bytes] = None
    try:
        with tifffile.TiffFile(path) as tf:
            try:
                arr = tf.asarray()
            except Exception:
                arr = tifffile.imread(path)
            for page in getattr(tf, "pages", [])[:1]:
                try:
                    tag = page.tags.get(34675)
                except Exception:
                    tag = None
                if tag is not None:
                    value = tag.value
                    if isinstance(value, (bytes, bytearray)):
                        icc = bytes(value)
                    elif isinstance(value, str):
                        icc = value.encode("latin-1", "replace")
    except Exception:
        arr = tifffile.imread(path)
    return arr, icc


# --- image read -------------------------------------------------------------

def read_image_array(path: str,
                     allow_precision_downgrade: bool = False) -> Tuple[Any, Dict[str, Any]]:
    """Read an image as a numpy array, preserving uint16 when present.

    RV-03: the reader is chosen **per format and bit depth**, not by "whatever
    import happens to succeed". tifffile handles TIFF (and carries its ICC),
    imagecodecs (libpng) handles 16-bit PNG; Pillow is only used when the file
    really is 8-bit, because Pillow's ``I;16`` is single-channel only and a
    16-bit RGB image comes back as uint8 truncated to the high byte.

    RV-04: the embedded ICC profile (TIFF tag 34675 / PNG iCCP chunk) and where
    it came from are returned in ``meta`` so every downstream boundary can keep
    colour provenance consistent with the pixels.

    Returns (array, meta). Raises :class:`MediaPrecisionError` when the file's
    real bit depth cannot be preserved, unless
    ``allow_precision_downgrade=True`` (the caller must then record it).
    """
    meta: Dict[str, Any] = {"reader": None, "icc_profile": None,
                            "mode": None, "path": str(path),
                            "icc_source": None, "bit_depth": None,
                            "precision_downgraded": False}
    ext = Path(path).suffix.lower()
    errors: List[str] = []

    # 1) format-aware high-precision readers
    if ext in (".tif", ".tiff"):
        try:
            arr, icc = _tiff_read(path)
            meta.update({"reader": "tifffile", "mode": str(getattr(arr, "dtype", "")),
                         "icc_profile": icc, "icc_source": "tiff-tag-34675" if icc else None})
            meta["bit_depth"] = 16 if str(getattr(arr, "dtype", "")) == "uint16" else 8
            return arr, meta
        except Exception as e:
            errors.append(f"tifffile: {type(e).__name__}: {e}")
        try:
            import imagecodecs  # type: ignore
            arr = imagecodecs.imread(path)
            icc = _png_icc_profile(path)  # not applicable, kept for symmetry
            meta.update({"reader": "imagecodecs",
                         "mode": str(getattr(arr, "dtype", "")),
                         "icc_profile": icc,
                         "icc_source": "png-iccP" if icc else None})
            meta["bit_depth"] = 16 if str(getattr(arr, "dtype", "")) == "uint16" else 8
            return arr, meta
        except Exception as e:
            errors.append(f"imagecodecs: {type(e).__name__}: {e}")
    elif ext == ".png":
        declared_depth = _png_bit_depth(path)
        meta["bit_depth"] = declared_depth
        try:
            import imagecodecs  # type: ignore  (libpng-backed, keeps uint16)
            arr = imagecodecs.imread(path)
            if declared_depth == 16 and str(getattr(arr, "dtype", "")) != "uint16":
                raise MediaPrecisionError(
                    f"{path}: IHDR declares 16-bit but the reader returned "
                    f"{getattr(arr, 'dtype', '?')}")
            icc = _png_icc_profile(path)
            meta.update({"reader": "imagecodecs(libpng)",
                         "mode": str(getattr(arr, "dtype", "")),
                         "icc_profile": icc,
                         "icc_source": "png-iccP" if icc else None})
            return arr, meta
        except MediaPrecisionError:
            raise
        except Exception as e:
            errors.append(f"imagecodecs: {type(e).__name__}: {e}")

    # 2) Pillow fallback — only acceptable when the file really is 8-bit
    try:
        from PIL import Image  # noqa: WPS433 (optional dependency)
        with Image.open(path) as im:
            meta["mode"] = im.mode
            icc = im.info.get("icc_profile")
            declared = meta.get("bit_depth")
            if declared is None and ext in (".tif", ".tiff"):
                # RV-03: 没有能读 16-bit TIFF 的依赖时，不能"静默塌成 8-bit"
                raise MediaPrecisionError(
                    f"{path}: no high-bit-depth TIFF reader available "
                    f"({'; '.join(errors) or 'tifffile/imagecodecs missing'}); "
                    "refusing to return 8-bit data for a possibly 16-bit file")
            import numpy as np
            arr = np.array(im)
            got_depth = 16 if str(getattr(arr, "dtype", "")) == "uint16" else 8
            if declared and declared > 8 and got_depth == 8:
                if not allow_precision_downgrade:
                    raise MediaPrecisionError(
                        f"{path}: declared {declared}-bit but only an 8-bit "
                        "representation is available; pass "
                        "allow_precision_downgrade=True to accept the loss "
                        "explicitly")
                meta["precision_downgraded"] = True
            meta.update({"reader": "pillow", "icc_profile": icc,
                         "icc_source": "pillow-info" if icc else None,
                         "bit_depth": declared or got_depth})
            return arr, meta
    except MediaPrecisionError:
        raise
    except Exception as e:
        errors.append(f"pillow: {type(e).__name__}: {e}")

    raise MediaPrecisionError(
        f"cannot read {path} at its real precision ({'; '.join(errors)})")


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


def write_png(path: str, arr: Any, icc: Optional[bytes] = None) -> str:
    """Write a PNG preserving uint8/uint16 precision. Returns the writer name.

    Pillow cannot write multi-channel 16-bit PNGs, so 16-bit arrays go through
    an internal zlib/struct writer. 8-bit arrays use Pillow when available.

    RV-04: ``icc`` is embedded as an ``iCCP`` chunk so the colour provenance
    survives the intermediate that is handed to the encoder.
    """
    import numpy as np
    arr = _ensure_uint(arr)
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.shape[2] not in PNG_COLOR_TYPE:
        raise MediaWriteError(f"unsupported channel count {arr.shape[2]}")
    if arr.dtype == np.uint16:
        return _write_png16_internal(path, arr, icc=icc)
    try:
        from PIL import Image
        if arr.shape[2] == 1:
            img = Image.fromarray(arr[:, :, 0], mode="L")
        elif arr.shape[2] == 3:
            img = Image.fromarray(arr, mode="RGB")
        else:
            img = Image.fromarray(arr, mode="RGBA")
        kwargs = {"format": "PNG"}
        if icc:
            kwargs["icc_profile"] = icc
        img.save(path, **kwargs)
        return "pillow"
    except Exception:
        return _write_png16_internal(path, arr.astype(np.uint16), icc=icc)


def _write_png16_internal(path: str, arr: Any,
                          icc: Optional[bytes] = None) -> str:
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
    chunks = [chunk(b"IHDR", ihdr)]
    if icc:
        # iCCP: profile name + NUL + compression method(0) + zlib-compressed ICC
        chunks.append(chunk(b"iCCP", b"ICC Profile\x00\x00" +
                            zlib.compress(bytes(icc), 6)))
    chunks.append(chunk(b"IDAT", zlib.compress(bytes(raw), 6)))
    chunks.append(chunk(b"IEND", b""))
    Path(path).write_bytes(b"\x89PNG\r\n\x1a\n" + b"".join(chunks))
    return "internal-png16"


# --- TIFF master ------------------------------------------------------------

def write_tiff(path: str, arr: Any, icc: Optional[bytes] = None) -> str:
    """Write a TIFF preserving uint8/uint16. Returns the writer actually used.

    tifffile is used when importable; otherwise an internal uncompressed TIFF
    writer is used so that a real 16-bit master is still produced (Pillow
    cannot write multi-channel 16-bit TIFF at all).

    RV-04: ``icc`` (when given) is embedded as tag 34675 so the master keeps
    the profile that actually describes its pixels.
    """
    import numpy as np
    arr = _ensure_uint(arr)
    try:
        import tifffile  # type: ignore
        if icc:
            tifffile.imwrite(path, arr, compression="zlib",
                             extratags=[(34675, 1, len(icc), icc, True)])
        else:
            tifffile.imwrite(path, arr, compression="zlib")
        return "tifffile"
    except Exception:
        pass
    return _write_tiff_internal(path, arr, icc=icc)


def _write_tiff_internal(path: str, arr: Any,
                         icc: Optional[bytes] = None) -> str:
    """Write an uncompressed, little-endian TIFF (gray/RGB/RGBA).

    RV-05 fixes (independent decoder reproduction):
    * ``BitsPerSample`` for a single-channel image is ONE SHORT and must be
      stored **inline** in the IFD entry — the old code always wrote an offset,
      so readers decoded 134 (the offset's low bytes) as the sample width.
    * Alpha is ``ExtraSamples`` = **338**; 339 is ``SampleFormat``. Using 339
      with value 2 declared a signed int format, so RGBA came back as int16.
    * ``SampleFormat`` is now written explicitly as 1 (unsigned) for 16-bit.
    """
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

    def _short(v: int) -> bytes:
        return struct.pack("<H", v)

    def _long(v: int) -> bytes:
        return struct.pack("<I", v)

    photometric = 1 if c == 1 else 2  # 1=BlackIsZero, 2=RGB
    # tag -> (type, count, value)；value 可以是 int / list[int] / bytes
    tag_spec: Dict[int, Tuple[int, int, Any]] = {
        256: (3, 1, w),            # ImageWidth
        257: (3, 1, h),            # ImageLength
        258: (3, c, [bits] * c),   # BitsPerSample
        259: (3, 1, 1),            # Compression = none
        262: (3, 1, photometric),  # PhotometricInterpretation
        273: (4, 1, 0),            # StripOffsets（稍后填真实偏移）
        277: (3, 1, c),            # SamplesPerPixel
        278: (4, 1, h),            # RowsPerStrip
        279: (4, 1, 0),            # StripByteCounts（稍后填）
        284: (3, 1, 1),            # PlanarConfiguration = chunky
    }
    if c == 4:
        tag_spec[338] = (3, 1, 2)  # ExtraSamples: unassociated alpha (NOT 339)
    if bits == 16:
        tag_spec[339] = (3, 1, 1)  # SampleFormat = 1 (unsigned integer)
    if icc:
        tag_spec[34675] = (1, len(icc), bytes(icc))  # ICCProfile (UNDEFINED)

    # 类型宽度：UNDEFINED(1)=1, SHORT(3)=2, LONG(4)=4
    type_size = {1: 1, 3: 2, 4: 4}

    # 超过 4 字节的值必须放在 IFD 之外（TIFF 规范：entry 的 value 字段只有 4 字节）
    external = sorted(tag for tag, (ttype, count, _v) in tag_spec.items()
                      if count * type_size[ttype] > 4)

    ifd_size = 2 + 12 * len(tag_spec) + 4
    header_size = 8
    offsets: Dict[int, int] = {}
    cursor = header_size + ifd_size
    for tag in external:
        ttype, count, _v = tag_spec[tag]
        offsets[tag] = cursor
        size = count * type_size[ttype]
        cursor += size + ((-size) % 4)  # 保持字对齐
    data_offset = cursor
    tag_spec[273] = (4, 1, data_offset)
    tag_spec[279] = (4, 1, len(raw))

    def _pack(ttype: int, value: Any) -> bytes:
        vals = value if isinstance(value, list) else [value]
        if ttype == 3:
            return b"".join(_short(int(v)) for v in vals)
        if ttype == 4:
            return b"".join(_long(int(v)) for v in vals)
        return bytes(value)

    out = bytearray()
    out.extend(b"II" + struct.pack("<H", 42) + struct.pack("<I", header_size))
    out.extend(struct.pack("<H", len(tag_spec)))
    for tag in sorted(tag_spec):
        ttype, count, value = tag_spec[tag]
        if tag in offsets:
            field = _long(offsets[tag])
        else:
            field = _pack(ttype, value).ljust(4, b"\x00")[:4]
        out.extend(struct.pack("<HHI", tag, ttype, count) + field)
    out.extend(struct.pack("<I", 0))  # next IFD offset

    # 外置数据区
    for tag in external:
        _ttype, _count, value = tag_spec[tag]
        payload = _pack(_ttype, value)
        out.extend(b"\x00" * (offsets[tag] - len(out)))
        out.extend(payload)

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
                      icc_path: Optional[str] = None,
                      cicp_override: Optional[str] = None) -> Tuple[List[str], Dict[str, Any]]:
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
    # RV-02: 色彩决议给出的 CICP 覆盖值必须真正进入命令行 —— ORIGINAL 只在
    # sidecar 里记了 2/2/0，实际命令仍是配置里的 1/13/0，于是"降级"这个决定
    # 从来没有落到产物上。2 = unspecified（libavif 手册）。
    cicp_override_applied: Optional[str] = None
    if cicp_override:
        cicp = str(cicp_override)
        cicp_override_applied = cicp
    speed = src.get("speed", 4)
    jobs = src.get("jobs", "all")
    extension = src.get("depth_extension")
    allow_downgrade = bool(params.get("allow_depth_downgrade", False))

    depth_arg, effective_depth, depth_info = resolve_depth_spec(
        depth, extension, capability, allow_downgrade=allow_downgrade)

    # NEW-RV-03（真机实测 avifenc 1.4.2，_lossless_ 分支）：
    #   `--cicp` 的 matrix 在 lossless 模式下只允许 identity(0) / YCgCo-Re / YCgCo-Ro。
    #   实测 stderr: "Matrix coefficients have to be identity, YCgCo-Re, or
    #   YCgCo-Ro in lossless mode."（rc=1）
    #   归档档（rgb444 + lossless）因此必须把矩阵显式落成 identity，否则即便
    #   "未指定"这种最诚实的标签也会被编码器拒绝。与 NEW-V-05 的
    #   range=full 同理：这类约束只有真编码器能暴露。
    matrix_forced_identity = False
    if lossless:
        parts = str(cicp).split("/")
        if len(parts) == 3 and parts[2].strip() != "0":
            parts[2] = "0"
            cicp = "/".join(parts)
            matrix_forced_identity = True

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
        "cicp_override": cicp_override_applied,
        "cicp_source": "color-resolution" if cicp_override_applied else "config",
        "matrix_forced_identity_by_lossless": matrix_forced_identity,
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
    # RV-04: 源 ICC 必须跟着像素一起进入编码器输入，否则"带 ICC 的素材"在
    # 这一跳就把色彩来源丢了。
    source_icc = meta.get("icc_profile")
    result["icc_carried"] = bool(source_icc)
    result["icc_source"] = meta.get("icc_source")
    os.makedirs(workdir, exist_ok=True)
    out_path = str(Path(workdir) / f"{p.stem}_enc_input.png")
    writer = write_png(out_path, arr, icc=source_icc)
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
