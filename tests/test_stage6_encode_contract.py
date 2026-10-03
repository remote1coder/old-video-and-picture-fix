"""V02/V04 regression: photo stage6 colour + archival-master contracts."""

import json
import os

import numpy as np
import pytest
from PIL import Image

import conftest  # noqa: F401
import stage6_encode as s6
import media_io


class _Log:
    """Minimal logger stub capturing events."""

    def __init__(self):
        self.events = []

    def _rec(self, name, msg, **kw):
        self.events.append((name, msg))

    info = warning = error = debug = _rec


def _working_png(tmp_path, with_icc=False, size=(8, 8)):
    p = tmp_path / "working.png"
    img = Image.fromarray(
        np.linspace(0, 255, size[0] * size[1] * 3, dtype=np.uint8)
        .reshape(size[0], size[1], 3), mode="RGB")
    kwargs = {}
    if with_icc:
        from PIL import ImageCms
        kwargs["icc_profile"] = ImageCms.ImageCmsProfile(
            ImageCms.createProfile("sRGB")).tobytes()
    img.save(p, format="PNG", **kwargs)
    return str(p)


# --- V04 --------------------------------------------------------------------

def test_v04_pillow_cannot_create_the_wide_gamut_profiles():
    """Root-cause evidence: on-the-fly ADBE/ROMM creation is unsupported."""
    from PIL import ImageCms
    for name in ("ADBE", "ROMM"):
        with pytest.raises(Exception):
            ImageCms.createProfile(name)
    ImageCms.createProfile("sRGB")  # sRGB is supported


def test_v04_wide_gamut_without_profile_blocks_the_stage(tmp_path):
    work = _working_png(tmp_path)
    log = _Log()
    color = s6.resolve_color_space(work, {"input_color_space": "adobergb"},
                                   str(tmp_path), "j1", "clip", log)
    assert color["status"] == s6.COLOR_FAILED
    assert any(name == "STAGE6_COLOR_FAILED" for name, _ in log.events)


def test_v04_degraded_encode_marks_cicp_unspecified(tmp_path):
    work = _working_png(tmp_path)
    log = _Log()
    color = s6.resolve_color_space(
        work, {"input_color_space": "adobergb",
               "allow_unconverted_colorspace": True},
        str(tmp_path), "j2", "clip", log)
    assert color["status"] == s6.COLOR_DEGRADED
    assert color["cicp_override"] == "2/2/0", "must not claim sRGB primaries"
    assert any(name == "STAGE6_COLOR_DEGRADED" for name, _ in log.events)


def test_v04_real_profile_is_used_for_conversion(tmp_path):
    """With a real (openable) profile the pixels are actually transformed.

    Limitation recorded in the ledger: the profile used here is an sRGB stand-in
    because no Adobe RGB profile is distributed with this environment; the
    assertion covers the control flow (open → transform → new file), not colour
    accuracy of Adobe RGB.
    """
    from PIL import ImageCms
    icc_path = tmp_path / "source.icc"
    icc_path.write_bytes(ImageCms.ImageCmsProfile(
        ImageCms.createProfile("sRGB")).tobytes())
    work = _working_png(tmp_path)
    log = _Log()
    color = s6.resolve_color_space(
        work, {"input_color_space": "adobergb",
               "source_icc_path": str(icc_path)},
        str(tmp_path), "j3", "clip", log)
    assert color["status"] == s6.COLOR_CONVERTED, log.events
    assert color["working_path"].endswith("_srgb.png")
    assert os.path.exists(color["working_path"])
    assert color["precision"] == "8-bit"  # honest about CMS precision


def test_v04_unknown_source_is_recorded_not_guessed(tmp_path):
    work = _working_png(tmp_path)  # no embedded ICC, no metadata
    log = _Log()
    color = s6.resolve_color_space(work, {}, str(tmp_path), "j4", "clip", log)
    assert color["status"] == s6.COLOR_UNKNOWN
    assert any(name == "STAGE6_COLOR_UNKNOWN" for name, _ in log.events)

    strict = s6.resolve_color_space(work, {"require_known_color_space": True},
                                    str(tmp_path), "j5", "clip", _Log())
    assert strict["status"] == s6.COLOR_FAILED


def test_v04_srgb_declared_source_passes_through(tmp_path):
    work = _working_png(tmp_path, with_icc=True)
    color = s6.resolve_color_space(work, {"input_color_space": "srgb"},
                                   str(tmp_path), "j6", "clip", _Log())
    assert color["status"] == s6.COLOR_NOT_NEEDED
    assert color["working_path"] == work


# --- V02 --------------------------------------------------------------------

def test_v02_master_records_real_precision_instead_of_claiming_16(tmp_path):
    work = _working_png(tmp_path)  # 8-bit source
    out = str(tmp_path / "master.tif")
    info = s6.write_archival_master(work, out, {"photo_bit_depth": 16}, _Log())
    assert info["written"] is True
    assert info["actual_bit_depth"] == 8
    assert info["precision_loss"] is True
    assert info["icc"] in ("none", "source")

    tifffile = pytest.importorskip("tifffile")
    written = tifffile.imread(out)
    assert written.dtype == np.uint8, "must not fake a 16-bit master"


def test_v02_16bit_source_keeps_16bit(tmp_path):
    tifffile = pytest.importorskip("tifffile")
    src = tmp_path / "src16.tif"
    arr = np.zeros((6, 6, 3), dtype=np.uint16)
    arr[..., 0] = np.arange(36, dtype=np.uint16).reshape(6, 6) * 100
    media_io.write_tiff(str(src), arr)
    out = str(tmp_path / "master16.tif")
    info = s6.write_archival_master(str(src), out, {"photo_bit_depth": 16},
                                    _Log())
    assert info["actual_bit_depth"] == 16
    assert info["precision_loss"] is False
    assert tifffile.imread(out).dtype == np.uint16


def test_v02_no_system_icc_is_pasted_on(tmp_path):
    """A profile may only be attached when it belongs to these pixels."""
    work = _working_png(tmp_path)  # no ICC in file
    out = str(tmp_path / "master_noicc.tif")
    info = s6.write_archival_master(work, out, {"photo_bit_depth": 16}, _Log())
    assert info["icc"] == "none"
    tifffile = pytest.importorskip("tifffile")
    # tag 34675 (ICC profile) must be absent
    with tifffile.TiffFile(out) as tf:
        tags = [t.code for p in tf.pages for t in p.tags.values()]
    assert 34675 not in tags
