#!/usr/bin/env python3
"""
Dependency Pre-check Module
Validates external tools, VapourSynth plugins, Python packages, AI model weights.
Returns exit code 80 if any required dependency is missing.
"""

import sys
import shutil
import subprocess
import importlib.util
from pathlib import Path
from typing import List, Dict, Tuple, Optional


def run_cmd(cmd: List[str], capture: bool = True, timeout: int = 30) -> Tuple[int, str, str]:
    """Run a command with shell=False."""
    try:
        r = subprocess.run(cmd, capture_output=capture, text=True,
                          timeout=timeout, shell=False)
        return r.returncode, r.stdout, r.stderr
    except FileNotFoundError:
        return -1, "", f"Command not found: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return -1, "", "Command timed out"


def check_external_tools() -> List[str]:
    """Check external tools: ffmpeg, ffprobe, vapoursynth, avifenc, exiftool, gmic."""
    required = {
        "ffmpeg": "FFmpeg (>=4.4 required for bandreject width_type=h)",
        "ffprobe": "FFprobe (stream analysis)",
        "avifenc": "avifenc (AVIF encoding for photos)",
        "exiftool": "ExifTool (metadata reading for photos)",
        "gmic": "G'MIC (photo processing)",
    }
    missing = []
    for tool, desc in required.items():
        if not shutil.which(tool):
            missing.append(f"  - {tool}: {desc}")

    # VapourSynth: check vspipe first, fallback to python import
    if not shutil.which("vspipe"):
        try:
            import vapoursynth as vs  # noqa: F401
        except ImportError:
            missing.append("  - vapoursynth: VapourSynth not installed (vspipe not found, python import failed)")

    return missing


def check_ffmpeg_version(min_version: str = "4.4") -> Tuple[bool, str]:
    """Check FFmpeg version >= min_version."""
    code, out, err = run_cmd(["ffmpeg", "-version"])
    if code != 0:
        return False, "FFmpeg not found"
    first_line = out.splitlines()[0] if out else ""
    # Parse version string like "ffmpeg version 4.4.2"
    import re
    m = re.search(r'version\s+(\d+\.\d+(?:\.\d+)?)', first_line)
    if not m:
        return False, f"Cannot parse FFmpeg version from: {first_line}"
    version = m.group(1)
    # Compare major.minor
    def v2tuple(v: str):
        parts = v.split(".")
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    if v2tuple(version) >= v2tuple(min_version):
        return True, f"FFmpeg {version} (>= {min_version})"
    return False, f"FFmpeg {version} (requires >= {min_version})"


def check_ffmpeg_filters() -> List[str]:
    """Check required FFmpeg filters are available."""
    required_filters = ["afftdn", "bandreject", "arnndn", "compand", "asetpts"]
    code, out, _ = run_cmd(["ffmpeg", "-filters"])
    if code != 0:
        return ["  - Cannot list FFmpeg filters"]
    missing = []
    for f in required_filters:
        if f not in out:
            missing.append(f"  - FFmpeg filter '{f}' not available")
    return missing


def check_vs_plugins() -> List[str]:
    """Check VapourSynth plugins via vs.core.get_plugins()."""
    missing = []
    try:
        import vapoursynth as vs
        core = vs.core
        plugins = core.get_plugins()
        # Build lowercase plugin name set
        loaded = set()
        for pid, pinfo in plugins.items():
            name = pinfo.get("name", "").lower()
            namespace = pinfo.get("namespace", "").lower()
            loaded.add(name)
            loaded.add(namespace)

        # Required plugins with aliases
        required = {
            "QTGMC": ["qtgmc"],
            "TemporalDegrain2": ["temporaldegrain2"],
            "VIVTC": ["vivtc"],
            "Deblock_QED": ["deblock_qed", "deblock"],
            "KNLMeansCL": ["knlmeanscl"],
            "GrainFactory3": ["grainfactory3"],
            "MVTools2": ["mvtools2", "mv"],
            "MaskTools2": ["masktools2", "mt"],
            "NNEDI3": ["nnedi3"],
            "NNEDI3CL": ["nnedi3cl"],
            "RgTools": ["rgtools"],
            "Zs_RF_Shared": ["zs_rf_shared"],
            "TCanny": ["tcanny"],
            "havsfunc": ["havsfunc"],
            "FFT3DFilter": ["fft3dfilter", "neo-fft3d"],
            "AddGrainC": ["addgrainc"],
            "RemoveGrain": ["removegrain"],
            "DFTTest": ["dftest"],
        }

        for plugin, aliases in required.items():
            found = any(a in loaded for a in aliases)
            if not found:
                missing.append(f"  - VS Plugin '{plugin}' (tried: {aliases})")
    except ImportError:
        missing.append("  - vapoursynth Python package not installed")
    except Exception as e:
        missing.append(f"  - Error checking VS plugins: {e}")

    return missing


def check_havsfunc_version() -> Tuple[bool, str]:
    """Check havsfunc version is r33."""
    try:
        import havsfunc as haf
        # Try to get version
        version = getattr(haf, "__version__", None)
        if version is None:
            # Version cannot be detected; require explicit r33
            import inspect
            src_file = inspect.getfile(haf)
            return False, (f"havsfunc found at {src_file} but version cannot be detected. "
                          f"Please install r33: pip install 'havsfunc==33'")
        if "33" in str(version):
            return True, f"havsfunc r{version}"
        return False, f"havsfunc version {version} (requires r33, install: pip install 'havsfunc==33')"
    except ImportError:
        return False, "havsfunc not installed (pip install 'havsfunc==33')"
    except Exception as e:
        return False, f"Error checking havsfunc: {e}"


def check_vs_minimal_calls() -> List[str]:
    """Execute minimal parameter set test calls for core filters."""
    errors = []
    try:
        import vapoursynth as vs
        import havsfunc as haf
        core = vs.core

        # Test 1: QTGMC InputType=1 with BlankClip YUV420P16
        try:
            clip1 = core.std.BlankClip(format=vs.YUV420P16, width=720, height=480, length=30)
            result1 = haf.QTGMC(clip1, InputType=1, Preset="Fast")
            if result1 is None:
                errors.append("  - QTGMC InputType=1 returned None")
        except Exception as e:
            errors.append(f"  - QTGMC InputType=1 test failed: {e}")

        # Test 2: QTGMC InputType=0 with field-based clip
        try:
            clip2 = core.std.BlankClip(format=vs.YUV420P16, width=720, height=480,
                                       fpsnum=30000, fpsden=1001, length=30)
            clip2 = clip2.std.SetFieldBased(2)  # TFF
            result2 = haf.QTGMC(clip2, InputType=0, TFF=True, Preset="Fast")
            if result2 is None:
                errors.append("  - QTGMC InputType=0 returned None")
        except Exception as e:
            errors.append(f"  - QTGMC InputType=0 test failed: {e}")

        # Test 3: TemporalDegrain2
        try:
            clip3 = core.std.BlankClip(format=vs.YUV420P16, width=720, height=480, length=60)
            result3 = haf.TemporalDegrain2(clip3, degrainTR=1, grainLevel=1)
            if result3 is None:
                errors.append("  - TemporalDegrain2 returned None")
        except Exception as e:
            errors.append(f"  - TemporalDegrain2 test failed: {e}")

        # Test 4: GrainFactory3
        try:
            clip4 = core.std.BlankClip(format=vs.YUV420P16, width=720, height=480, length=30)
            result4 = haf.GrainFactory3(clip4, g1str=1.0, g2str=0.2, g3str=0.2)
            if result4 is None:
                errors.append("  - GrainFactory3 returned None")
        except TypeError as e:
            if "unexpected keyword" in str(e):
                errors.append(f"  - GrainFactory3 parameter error (need r33): {e}")
            else:
                errors.append(f"  - GrainFactory3 TypeError: {e}")
        except Exception as e:
            errors.append(f"  - GrainFactory3 test failed: {e}")

        # Test 5: sRestore
        try:
            clip5 = core.std.BlankClip(format=vs.YUV420P16, width=720, height=480, length=60)
            result5 = haf.srestore(clip5, frate=23.976)
            if result5 is None:
                errors.append("  - sRestore returned None")
        except Exception as e:
            errors.append(f"  - sRestore test failed (optional): {e}")

    except ImportError as e:
        errors.append(f"  - Cannot import for minimal calls: {e}")
    except Exception as e:
        errors.append(f"  - Unexpected error in minimal calls: {e}")

    return errors


def check_python_packages() -> List[str]:
    """Check required Python packages."""
    required = [
        ("vapoursynth", "VapourSynth Python bindings"),
        ("cv2", "opencv-python"),
        ("numpy", "NumPy"),
        ("pandas", "Pandas"),
        ("yaml", "PyYAML"),
        ("PIL", "Pillow"),
        ("requests", "requests"),
        ("rich", "rich (TUI)"),
    ]
    missing = []
    for mod_name, desc in required:
        spec = importlib.util.find_spec(mod_name)
        if spec is None:
            missing.append(f"  - {desc}: not installed")
    return missing


def check_face_detection_packages() -> List[str]:
    """Check face detection Python packages."""
    required = ["retina_face", "ultralytics", "mtcnn"]
    missing = []
    for pkg in required:
        spec = importlib.util.find_spec(pkg)
        if spec is None:
            missing.append(f"  - {pkg}: not installed (pip install {pkg})")
    return missing


def check_ai_models(models_dir: str = "./models") -> List[str]:
    """Check AI model weights exist. Returns missing model descriptions."""
    models_path = Path(models_dir).resolve()
    required_models = {
        "RealESRGAN_x2plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases",
        "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases",
        "GFPGANv1.4.pth": "https://github.com/TencentARC/GFPGAN/releases",
        "DAT_x2.pth": "https://github.com/ckkelvinchan/DAT/releases",
        "DAT_x4.pth": "https://github.com/ckkelvinchan/DAT/releases",
    }
    missing = []
    for model_name, url in required_models.items():
        model_path = models_path / model_name
        if not model_path.exists():
            missing.append(f"  - {model_name}: not found in {models_path} (download: {url})")
    return missing


def check_arnndn_models(config: Optional[Dict] = None) -> List[str]:
    """Check arnndn model files for FFmpeg <4.4 fallback path."""
    if config is None:
        config = {}
    models_dir = config.get("arnndn_model_dir", "./models")
    models_path = Path(models_dir)
    missing = []
    for fname in ["arnndn_50Hz.pb", "arnndn_60Hz.pb"]:
        fpath = models_path / fname
        if not fpath.exists():
            missing.append(
                f"  - {fname}: not found (download: https://github.com/NullModel/arnndn-models)"
            )
    return missing


def run_all_checks(config: Optional[Dict] = None) -> Tuple[bool, List[str]]:
    """Run all dependency checks. Returns (all_ok, list_of_issues)."""
    issues = []

    # 1. External tools
    ext = check_external_tools()
    if ext:
        issues.append("Missing external tools:")
        issues.extend(ext)

    # 2. FFmpeg version
    ok, msg = check_ffmpeg_version()
    if not ok:
        issues.append(f"FFmpeg version check: {msg}")
    else:
        # Check filters only if FFmpeg is available
        filt = check_ffmpeg_filters()
        if filt:
            issues.append("Missing FFmpeg filters:")
            issues.extend(filt)

    # 3. VS Plugins
    vs = check_vs_plugins()
    if vs:
        issues.append("Missing VapourSynth plugins:")
        issues.extend(vs)

    # 4. havsfunc version
    ok, msg = check_havsfunc_version()
    if not ok:
        issues.append(f"havsfunc: {msg}")

    # 5. VS minimal calls
    calls = check_vs_minimal_calls()
    if calls:
        issues.append("VapourSynth filter test failures:")
        issues.extend(calls)

    # 6. Python packages
    py = check_python_packages()
    if py:
        issues.append("Missing Python packages:")
        issues.extend(py)

    # 7. Face detection packages (if AI enabled)
    if config and config.get("ai", {}).get("enabled", False):
        face = check_face_detection_packages()
        if face:
            issues.append("Missing face detection packages (AI enabled):")
            issues.extend(face)

    # 8. AI models (if AI enabled)
    if config and config.get("ai", {}).get("enabled", False):
        models = check_ai_models(config.get("model_dir", "./models"))
        if models:
            issues.append("Missing AI model weights:")
            issues.extend(models)

    # 9. arnndn models
    arnndn = check_arnndn_models(config.get("audio", {}) if config else {})
    if arnndn:
        issues.append("Missing arnndn models (audio fallback):")
        issues.extend(arnndn)

    return len(issues) == 0, issues


def main():
    """CLI entry for dependency check."""
    import yaml  # type: ignore[import-untyped]

    config_path = None
    for p in ["./config.yaml", "../config.yaml"]:
        if Path(p).exists():
            config_path = p
            break

    config = None
    if config_path:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)

    ok, issues = run_all_checks(config)

    print("=" * 60)
    print("Dependency Check Results")
    print("=" * 60)
    if ok:
        print("All required dependencies are satisfied.")
    else:
        print(f"Found {len(issues)} issue(s):")
        for issue in issues:
            print(issue)
        print("\nInstall missing dependencies and retry.")
        sys.exit(80)


if __name__ == "__main__":
    main()
