"""照片分类：DPI 分支短路 `elif model or make` —— 真实相机照片全过不了分类。

**这个 bug 是我在实测照片链身份记录时撞上的**

我给测试图加了 `Make=Canon / Model=Canon EOS 5D`，本以为会判
`DIGITAL_PHOTO / 0.90`，结果拿到 `UNKNOWN / 0.0`，主流程直接
`LOW_CONFIDENCE_SKIP` 跳过整张图。

**根因：`elif` 短路 + 分支内没有兜底**

```python
if scanner or "scan" in software:       # 0.95
    ...
elif result.get("dpi"):                 # ← 只要有 DPI 就进来（72 也算）
    if dpi_discrete_match:              # 300/600/... 才匹配
        ... 0.80
    # ← 不匹配时什么都不做，直接结束
elif model or make:                     # ← 永远走不到
    ... 0.90
```

**后果比"一个边角 case"严重**

| 输入 | 修复前 | 修复后 |
|---|---|---|
| 相机 + DPI=72（最常见） | **UNKNOWN 0.0 → 跳过** | DIGITAL_PHOTO 0.90 |
| 相机 + DPI=300（打印用） | SCAN_PRINT 0.80（误判） | DIGITAL_PHOTO 0.90 |
| 无型号 + DPI=300 | SCAN_PRINT 0.80 | SCAN_PRINT 0.80 |
| 无型号 + DPI=72 | UNKNOWN 0.0 | DIGITAL_PHOTO 0.60 |

也就是说：**任何带常见 DPI 的数码相机照片都过不了分类**。

**修法**：把"是否标准扫描 DPI"收进条件本身，并要求"没有相机型号"
才按扫描件处理（扫描件通常不写 Make/Model）。

**已知局限**：某些扫描仪会写 `Make=EPSON`，这类仍会被判 DIGITAL_PHOTO。
不加品牌白名单是因为 Canon/HP 既做相机也做扫描仪，白名单会引入新误判。
"""

import os
import subprocess
import shutil
import sys

import pytest

import conftest  # noqa: F401  (sys.path bootstrap)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PHOTO = os.path.join(ROOT, "photo_restoration_pipeline")
if PHOTO not in sys.path:
    sys.path.insert(0, PHOTO)

import classify  # noqa: E402

HAS_EXIFTOOL = bool(shutil.which("exiftool"))
HAS_FFMPEG = bool(shutil.which("ffmpeg"))


def _make_jpg(path, dpi, make=None, model=None, software=None):
    if not (HAS_FFMPEG and HAS_EXIFTOOL):
        pytest.skip("需要 ffmpeg + exiftool")
    r = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi",
                        "-i", "testsrc2=s=1600x1200", "-frames:v", "1",
                        "-q:v", "2", "-y", str(path)], capture_output=True)
    if r.returncode != 0:
        pytest.skip("生成测试图失败")
    args = ["exiftool", f"-XResolution={dpi}", f"-YResolution={dpi}",
            "-ResolutionUnit=inches", "-overwrite_original"]
    if make:
        args += [f"-Make={make}"]
    if model:
        args += [f"-Model={model}"]
    if software:
        args += [f"-Software={software}"]
    args.append(str(path))
    subprocess.run(args, capture_output=True)


# ---------------------------------------------------------------------------
# 四种组合 —— 这是 bug 的直接证据
# ---------------------------------------------------------------------------

def test_camera_photo_with_common_dpi_is_not_unknown(tmp_path):
    """相机照片 + DPI=72（最常见）：必须判 DIGITAL_PHOTO，不能 UNKNOWN。

    修复前这条会得到 UNKNOWN / 0.0，导致整张图被跳过。
    """
    p = tmp_path / "cam72.jpg"
    _make_jpg(p, 72, make="Canon", model="Canon EOS 5D")
    r = classify.classify_photo(str(p))
    assert r["detected_class"] == "DIGITAL_PHOTO", (
        f"相机照片被判成 {r['detected_class']} —— "
        "DPI 分支又短路了 model/make 判断")
    assert r["confidence"] >= 0.90


def test_camera_photo_with_print_dpi_is_not_scan(tmp_path):
    """相机照片 + DPI=300：不能被误判成扫描件。"""
    p = tmp_path / "cam300.jpg"
    _make_jpg(p, 300, make="Canon", model="Canon EOS 5D")
    r = classify.classify_photo(str(p))
    assert r["detected_class"] == "DIGITAL_PHOTO", (
        f"300 DPI 的相机照片被判成 {r['detected_class']}")
    assert r["confidence"] >= 0.90


def test_no_camera_model_with_scanner_dpi_is_scan(tmp_path):
    """无相机型号 + 标准扫描 DPI：仍应判扫描件。"""
    p = tmp_path / "scan300.jpg"
    _make_jpg(p, 300)
    r = classify.classify_photo(str(p))
    assert r["detected_class"] == "SCAN_PRINT"
    assert abs(r["confidence"] - 0.80) < 1e-6


def test_no_camera_model_with_screen_dpi_falls_back_to_resolution(tmp_path):
    """无型号 + 72 DPI：走分辨率兜底（1600x1200 → DIGITAL_PHOTO 0.60）。"""
    p = tmp_path / "unknown72.jpg"
    _make_jpg(p, 72)
    r = classify.classify_photo(str(p))
    assert r["detected_class"] == "DIGITAL_PHOTO"
    assert abs(r["confidence"] - 0.60) < 1e-6


def test_scan_software_keyword_still_wins(tmp_path):
    """Software 含 "scan" 仍应走最强的 0.95 分支。"""
    p = tmp_path / "scanword.jpg"
    _make_jpg(p, 300, make="Canon", model="EOS 5D",
              software="Scanned by ScanWizard")
    r = classify.classify_photo(str(p))
    assert r["detected_class"] == "SCAN_PRINT"
    assert abs(r["confidence"] - 0.95) < 1e-6


# ---------------------------------------------------------------------------
# 辅助函数与源码守卫
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("dpi,expected", [
    (300, True), (600, True), (1200, True), (2400, True),
    (302, True),        # 容差 5 内
    (72, False), (96, False), (144, False), (None, False), (0, False),
])
def test_is_scanner_dpi(dpi, expected):
    assert classify._is_scanner_dpi(dpi) is expected


def test_dpi_branch_does_not_short_circuit_model_check():
    """源码守卫：DPI 分支必须带 `_is_scanner_dpi` 且排除相机型号。"""
    src = open(classify.__file__, encoding="utf-8").read()
    assert "_is_scanner_dpi(result[\"dpi\"])" in src, (
        "DPI 分支又退回成 `elif result.get('dpi'):` —— 会重新短路 model/make")
    assert "and not (model or make)" in src
    # 不能再出现"进分支后什么都不做"的写法。
    # 注意只查**代码行**：docstring/注释里引用旧写法是允许的。
    code_lines = [ln.strip() for ln in src.splitlines()
                  if ln.strip() and not ln.strip().startswith("#")]
    assert not any(ln.startswith('elif result.get("dpi"):')
                   for ln in code_lines), (
        "DPI 分支又退回成裸 elif —— 会重新短路 model/make")
