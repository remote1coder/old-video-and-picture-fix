#!/usr/bin/env python3
"""
Dependency Pre-check Module
Validates external tools, VapourSynth plugins, Python packages, AI model weights.
Returns exit code 80 if any required dependency is missing.

v7.3 Compliance:
- Pipeline-type-aware checks (photo skips video-specific deps)
- Checks AI CLI tools (not just model files)
- G'MIC filter validation
- GPU backend extension detection (CUDA → ROCm → MPS → CPU)
- PyTorch spawn compliance pre-check
"""

import sys
import shutil
import subprocess
import importlib.util
import re
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


def check_external_tools(pipeline_type: str = "photo") -> List[str]:
    """Check external tools. Photo skips video-specific tools."""
    if pipeline_type == "photo":
        required = {
            "ffmpeg": "FFmpeg (metadata reading for photos)",
            "ffprobe": "FFprobe (stream analysis for photos)",
            "avifenc": "avifenc (AVIF encoding for photos)",
            "exiftool": "ExifTool (metadata reading for photos)",
            "gmic": "G'MIC (photo processing)",
        }
    else:
        required = {
            "ffmpeg": "FFmpeg (>=4.4 required for bandreject width_type=h)",
            "ffprobe": "FFprobe (stream analysis)",
            "exiftool": "ExifTool (metadata reading)",
            "gmic": "G'MIC (filter processing)",
        }
    missing = []
    for tool, desc in required.items():
        if not shutil.which(tool):
            missing.append(f"  - {tool}: {desc}")

    # VapourSynth: only for video pipeline
    if pipeline_type != "photo":
        if not shutil.which("vspipe"):
            try:
                import vapoursynth as vs  # noqa: F401
            except ImportError:
                missing.append("  - vapoursynth: VapourSynth not installed (vspipe not found, python import failed)")

    return missing


def check_avifenc_version() -> Tuple[bool, str, str]:
    """Check avifenc version and determine appropriate --jobs argument.
    
    Per v6.32 §7.29 / v7.32 §3.10.20:
    - libavif <0.9.2: --jobs not supported
    - 0.9.2-1.0.x: --jobs requires numeric value (e.g., --jobs 4)
    - 1.1.0+: --jobs all supported (and is default)
    
    Returns (ok, version_str, jobs_argument).
    """
    code, out, err = run_cmd(["avifenc", "--version"])
    if code != 0:
        # Try --help as fallback for older versions
        code2, out2, _ = run_cmd(["avifenc", "--help"])
        if code2 != 0:
            return False, "unknown", ""
        # Version unknown; assume recent enough for --jobs all
        return True, "unknown (assumed >=1.1.0)", "all"
    
    version_line = out.splitlines()[0] if out else ""
    # Parse version: avifenc output usually contains version number
    m = re.search(r'(\d+\.\d+(?:\.\d+)?)', version_line)
    if not m:
        return True, version_line.strip() or "unknown", "all"
    
    version = m.group(1)
    def v2tuple(v: str):
        parts = v.split(".")
        return tuple(int(p) for p in parts[:3])
    vt = v2tuple(version)
    
    if vt < (0, 9, 2):
        return True, version, ""  # --jobs not supported
    elif vt < (1, 1, 0):
        # Numeric jobs required
        import os
        nproc = os.cpu_count() or 1
        return True, version, str(min(nproc, 32))
    else:
        return True, version, "all"


def check_ffmpeg_version(min_version: str = "4.4") -> Tuple[bool, str]:
    """Check FFmpeg version >= min_version."""
    code, out, err = run_cmd(["ffmpeg", "-version"])
    if code != 0:
        return False, "FFmpeg not found"
    first_line = out.splitlines()[0] if out else ""
    m = re.search(r'version\s+(\d+\.\d+(?:\.\d+)?)', first_line)
    if not m:
        return False, f"Cannot parse FFmpeg version from: {first_line}"
    version = m.group(1)
    def v2tuple(v: str):
        parts = v.split(".")
        return (int(parts[0]), int(parts[1]) if len(parts) > 1 else 0)
    if v2tuple(version) >= v2tuple(min_version):
        return True, f"FFmpeg {version} (>= {min_version})"
    return False, f"FFmpeg {version} (requires >= {min_version})"


def check_ffmpeg_filters(pipeline_type: str = "photo") -> List[str]:
    """Check required FFmpeg filters. Photo only checks minimal set."""
    if pipeline_type == "photo":
        return []  # Photos don't use FFmpeg filters
    required_filters = ["afftdn", "bandreject", "arnndn", "compand", "asetpts", "zscale", "atempo"]
    code, out, _ = run_cmd(["ffmpeg", "-filters"])
    if code != 0:
        return ["  - Cannot list FFmpeg filters"]
    missing = []
    # Parse filter list for exact filter name matches
    # ffmpeg -filters output format:  ... filter_name ...
    available = set()
    for line in out.splitlines():
        parts = line.strip().split()
        if len(parts) >= 2:
            # Filter names are in the second column (after the flags like "A..T")
            fname = parts[1].strip()
            available.add(fname)
    for f in required_filters:
        if f not in available:
            missing.append(f"  - FFmpeg filter '{f}' not available")
    return missing


def check_ffmpeg_filters_minimal_calls(pipeline_type: str = "photo") -> List[str]:
    """Execute minimal parameter set test calls for critical FFmpeg filters.
    
    Per v7.32 §3.6: must verify filters work with actual parameters, not just names.
    """
    if pipeline_type == "photo":
        return []
    errors = []
    # Use a 1-second testsrc to avoid needing real input files
    test_src = "testsrc=duration=1:size=320x240:rate=1"
    
    # Test 1: bandreject with width_type=h (critical for hum removal)
    code, _, err = run_cmd([
        "ffmpeg", "-f", "lavfi", "-i", f"sine=frequency=50:duration=1",
        "-af", "bandreject=f=50:width_type=h:width=2",
        "-f", "null", "-"
    ], timeout=15)
    if code != 0:
        errors.append(f"  - bandreject test failed: {err[:200]}")
    
    # Test 2: afftdn (noise reduction)
    code, _, err = run_cmd([
        "ffmpeg", "-f", "lavfi", "-i", f"sine=frequency=1000:duration=1",
        "-af", "afftdn=nf=-25",
        "-f", "null", "-"
    ], timeout=15)
    if code != 0:
        errors.append(f"  - afftdn test failed: {err[:200]}")
    
    # Test 3: asetpts=PTS (audio sync)
    code, _, err = run_cmd([
        "ffmpeg", "-f", "lavfi", "-i", f"sine=frequency=1000:duration=1",
        "-af", "asetpts=PTS",
        "-f", "null", "-"
    ], timeout=15)
    if code != 0:
        errors.append(f"  - asetpts test failed: {err[:200]}")
    
    # Test 4: atempo (speed change)
    code, _, err = run_cmd([
        "ffmpeg", "-f", "lavfi", "-i", f"sine=frequency=1000:duration=1",
        "-af", "atempo=1.0",
        "-f", "null", "-"
    ], timeout=15)
    if code != 0:
        errors.append(f"  - atempo test failed: {err[:200]}")
    
    # Test 5: zscale (color space conversion)
    code, _, err = run_cmd([
        "ffmpeg", "-f", "lavfi", "-i", test_src,
        "-vf", "zscale=matrixin=bt470bg:matrix=bt709:rangein=limited:range=limited",
        "-frames:v", "1",
        "-f", "null", "-"
    ], timeout=15)
    if code != 0:
        errors.append(f"  - zscale test failed: {err[:200]}")
    
    return errors


def check_vs_plugins(pipeline_type: str = "photo") -> List[str]:
    """Check VapourSynth plugins. Skipped for photo pipeline."""
    if pipeline_type == "photo":
        return []
    missing = []
    try:
        import vapoursynth as vs
        core = vs.core
        plugins = core.get_plugins()
        loaded = set()
        for pid, pinfo in plugins.items():
            name = pinfo.get("name", "").lower()
            namespace = pinfo.get("namespace", "").lower()
            loaded.add(name)
            loaded.add(namespace)

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
        version = getattr(haf, "__version__", None)
        if version is None:
            import inspect
            src_file = inspect.getfile(haf)
            return False, (f"havsfunc found at {src_file} but version cannot be detected. "
                          f"Please install r33: pip install 'havsfunc==33'")
        # Strict match: r33, 33, 33.0, but not 133, 33.1, or 33a
        vstr = str(version)
        if re.match(r"^r?33($|\.0*$)", vstr):
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

        # FIX-P1-1: DepanStabilize minimal call test per v7.32 §3.6
        try:
            clip3b = core.std.BlankClip(format=vs.YUV420P16, width=720, height=480, length=60)
            # mvtools2 registers as core.mv namespace in VS
            result3b = core.mv.DepanStabilize(clip3b, dxmax=10, dymax=10)
            if result3b is None:
                errors.append("  - DepanStabilize returned None")
        except Exception as e:
            errors.append(f"  - DepanStabilize test failed: {e}")

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


def check_python_packages(pipeline_type: str = "photo") -> List[str]:
    """Check required Python packages."""
    if pipeline_type == "photo":
        required = [
            ("numpy", "NumPy"),
            ("yaml", "PyYAML"),
            ("PIL", "Pillow"),
            ("scipy", "SciPy (for Stage 4 region separation)"),
        ]
    else:
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


def check_face_detection_packages(config: Optional[Dict] = None) -> List[str]:
    """Check face detection Python packages (if AI enabled).
    
    Per v7.32 §3.6: Must confirm 'retina-face' (PyPI package name),
    'ultralytics', 'mtcnn' are installed.
    Note: The Python import module name for retina-face is 'retinaface'.
    """
    if config and not config.get("ai", {}).get("enabled", False):
        return []
    # (module_name, pip_package_name)
    required = [
        ("retinaface", "retina-face"),
        ("ultralytics", "ultralytics"),
        ("mtcnn", "mtcnn"),
    ]
    missing = []
    for mod_name, pip_name in required:
        spec = importlib.util.find_spec(mod_name)
        if spec is None:
            missing.append(f"  - {pip_name}: not installed (pip install {pip_name})")
    return missing


def check_ai_cli_tools() -> List[str]:
    """Check AI CLI tools are available (not just model files)."""
    missing = []
    # Check realesrgan-ncnn-vulkan
    if not shutil.which("realesrgan-ncnn-vulkan"):
        spec = importlib.util.find_spec("realesrgan")
        if spec is None:
            missing.append("  - realesrgan: neither realesrgan-ncnn-vulkan binary nor Python package found")
        else:
            try:
                import realesrgan  # noqa: F401
            except ImportError:
                missing.append("  - realesrgan Python package found but RealESRGANer not importable")

    # Check GFPGAN
    gfpgan_ok = False
    if shutil.which("inference_gfpgan.py"):
        gfpgan_ok = True
    else:
        spec = importlib.util.find_spec("gfpgan")
        if spec is not None:
            try:
                import gfpgan  # noqa: F401
                gfpgan_ok = True
            except ImportError:
                pass
    if not gfpgan_ok:
        missing.append("  - gfpgan: neither inference_gfpgan.py nor GFPGANer API available")

    # Check CodeFormer
    cf_ok = False
    if shutil.which("inference_codeformer.py"):
        cf_ok = True
    else:
        # Try importing codeformer through basicsr
        try:
            import basicsr.archs.codeformer_arch  # noqa: F401
            cf_ok = True
        except ImportError:
            pass
    if not cf_ok:
        missing.append("  - codeformer: neither inference_codeformer.py nor basicsr.archs.codeformer importable")

    return missing


def check_ai_models(models_dir: str = "./models", skip_md5_check: bool = False) -> List[str]:
    """Check AI model weights exist and validate MD5 checksums.
    
    Per v7.32 §3.6: Each model must have MD5 + download URL;
    version mismatch rejects loading.
    
    Supports external MD5 manifest (md5_manifest.json) in models_dir
    to enable checksum validation without modifying source code.
    If no MD5 is configured (built-in or external), an ERROR is issued
    unless --skip-md5-check is used.
    """
    models_path = Path(models_dir).resolve()
    
    # Load external MD5 manifest if present
    external_md5 = {}
    manifest_path = models_path / "md5_manifest.json"
    if manifest_path.exists():
        try:
            import json
            with open(manifest_path, "r", encoding="utf-8") as f:
                external_md5 = json.load(f)
        except Exception:
            pass
    
    required_models = {
        "RealESRGAN_x2plus.pth": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases",
            "md5": None,  # Fill with actual MD5 or use external manifest
        },
        "RealESRGAN_x4plus.pth": {
            "url": "https://github.com/xinntao/Real-ESRGAN/releases",
            "md5": None,
        },
        "GFPGANv1.4.pth": {
            "url": "https://github.com/TencentARC/GFPGAN/releases",
            "md5": None,
        },
        "DAT_x2.pth": {
            "url": "https://github.com/ckkelvinchan/DAT/releases",
            "md5": None,
        },
        "DAT_x4.pth": {
            "url": "https://github.com/ckkelvinchan/DAT/releases",
            "md5": None,
        },
        "CodeFormer.pth": {
            "url": "https://github.com/sczhou/CodeFormer/releases",
            "md5": None,
        },
    }
    missing = []
    for model_name, info in required_models.items():
        model_path = models_path / model_name
        if not model_path.exists():
            missing.append(f"  - {model_name}: not found in {models_path} (download: {info['url']})")
            continue
        
        expected_md5 = info.get("md5") or external_md5.get(model_name)
        if expected_md5:
            import hashlib
            h = hashlib.md5()
            with open(model_path, "rb") as f:
                while chunk := f.read(65536):
                    h.update(chunk)
            actual_md5 = h.hexdigest()
            if actual_md5 != expected_md5:
                missing.append(f"  - {model_name}: MD5 mismatch (expected {expected_md5}, got {actual_md5})")
        else:
            # FIX-P1-6: No MD5 configured → ERROR per v7.32 §3.6
            if skip_md5_check:
                missing.append(
                    f"  - {model_name}: WARNING - MD5 checksum not configured. "
                    f"Create {manifest_path} with {{'{model_name}': '<md5_hash>'}} to enable verification"
                )
            else:
                missing.append(
                    f"  - {model_name}: ERROR - MD5 verification not configured. "
                    f"Create {manifest_path} or use --skip-md5-check to bypass"
                )
    return missing


def check_arnndn_models(config: Optional[Dict] = None, ffmpeg_version_ok: bool = True) -> List[str]:
    """Check arnndn model files for FFmpeg <4.4 fallback path.
    
    Also checks when audio is disabled but FFmpeg <4.4 (bandreject unavailable).
    """
    if config is None:
        config = {}
    
    # Skip only if FFmpeg >=4.4 AND audio not enabled
    if ffmpeg_version_ok:
        audio_cfg = config.get("audio", {})
        if audio_cfg.get("enabled", False) is False:
            return []
    
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


def check_gmic_filters() -> List[str]:
    """Check G'MIC supports required filters.
    
    BUG-062 fix: Use per-filter exact query (gmic -h filtername) instead of
    substring matching in the full help text to avoid false positives.
    """
    required_filters = ["repair", "deblock", "deband", "film_grain"]
    missing = []
    for f in required_filters:
        try:
            r = subprocess.run(["gmic", "-h", f],
                              capture_output=True, text=True, timeout=10, shell=False)
            # If returncode != 0 or stdout is empty/just error, filter is unavailable
            if r.returncode != 0 or not r.stdout.strip() or "unknown command" in r.stdout.lower():
                missing.append(f"  - G'MIC filter '{f}' not available")
        except Exception:
            missing.append(f"  - Cannot verify G'MIC filter '{f}'")
    return missing


def check_gpu_backends() -> Tuple[bool, List[str]]:
    """Check GPU backend availability: CUDA → ROCm → MPS → CPU.
    
    Returns (has_gpu, messages).
    """
    messages = []
    try:
        import torch
        if torch.cuda.is_available():
            messages.append(f"  - CUDA: available (device: {torch.cuda.get_device_name(0)})")
            return True, messages
        if getattr(torch.version, "hip", None) is not None:
            messages.append("  - ROCm: available")
            return True, messages
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            messages.append("  - MPS (Apple Silicon): available")
            return True, messages
        messages.append("  - No GPU backend found (CUDA/ROCm/MPS)")
        return False, messages
    except ImportError:
        messages.append("  - PyTorch not installed, cannot detect GPU backends")
        return False, messages


def check_spawn_compliance(target_script: Optional[str] = None,
                             related_modules: Optional[List[str]] = None) -> List[str]:
    """Check PyTorch spawn compliance.
    
    Per v7.32 §3.10.18:
    1. Main script all top-level executable code in if __name__ == '__main__':
    2. Imported modules: only defs/classes/constants at top level
    3. Target function must be module-level global function
    
    ISSUE-066 fix: also checks related_modules (e.g., main.py, stage scripts)
    for top-level executable code.
    """
    issues = []
    
    def _check_single_file(path: str, label: str) -> None:
        if not path or not Path(path).exists():
            return
        content = Path(path).read_text(encoding="utf-8")
        lines = content.splitlines()
        in_main = False
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("if __name__"):
                in_main = True
            elif in_main:
                continue
            elif stripped and not stripped.startswith(("#", "import", "from", "def ", "class ", 
                                                        "__", "\"\"\"", "'''", " ", "\t")):
                if i > 20:  # Skip header comments
                    issues.append(f"  - {label}: potential top-level executable code at line {i+1}: {stripped[:60]}")
                    break
    
    # Check main script
    if target_script is None:
        target_script = sys.argv[0] if sys.argv else None
    if target_script:
        _check_single_file(target_script, Path(target_script).name)
    
    # ISSUE-066 fix: check related modules
    if related_modules:
        for mod_path in related_modules:
            _check_single_file(mod_path, Path(mod_path).name)
    else:
        # Default: discover project Python files adjacent to the main script
        if target_script:
            base_dir = Path(target_script).parent
            for py_file in base_dir.rglob("*.py"):
                # Skip __pycache__ and virtualenv paths
                if "__pycache__" in str(py_file):
                    continue
                if py_file.resolve() == Path(target_script).resolve():
                    continue
                _check_single_file(str(py_file), py_file.name)
    
    return issues


# Fix: Auto-detect pipeline type from config
def _detect_pipeline_type(config: Optional[Dict] = None) -> str:
    """Auto-detect pipeline type from config.
    
    Heuristic: if config has 'photo' section or lacks video-specific keys,
    treat as photo. Can be overridden by explicit pipeline_type.
    """
    if config is None:
        return "photo"
    if "pipeline_type" in config:
        return config["pipeline_type"]
    if "photo" in config and "audio" not in config:
        return "photo"
    audio_cfg = config.get("audio", {})
    if isinstance(audio_cfg, dict) and audio_cfg.get("enabled", False):
        return "video"
    return "photo"


def run_all_checks(config: Optional[Dict] = None,
                   pipeline_type: Optional[str] = None,
                   skip_md5_check: bool = False) -> Tuple[bool, List[str]]:
    """Run all dependency checks. Returns (all_ok, list_of_issues).
    
    Args:
        config: Loaded config dict (used for auto-detection)
        pipeline_type: Explicit pipeline type override ("photo" or "video")
    """
    issues = []
    
    if pipeline_type is None:
        pipeline_type = _detect_pipeline_type(config)

    # 1. External tools
    ext = check_external_tools(pipeline_type)
    if ext:
        issues.append("Missing external tools:")
        issues.extend(ext)

    # 1a. avifenc version + --jobs compatibility (ISSUE-SHARED-DC-001)
    if shutil.which("avifenc"):
        avif_ok, avif_ver, avif_jobs = check_avifenc_version()
        if not avif_ok:
            issues.append(f"avifenc version check failed: {avif_ver}")
        # Info message is not an issue; store for caller inspection if needed
        # (callers can inspect config['photo']['avif_params']['jobs'])

    # 2. FFmpeg version (video only, but also for photo metadata)
    ffmpeg_ok = True
    if pipeline_type != "photo":
        ok, msg = check_ffmpeg_version()
        if not ok:
            issues.append(f"FFmpeg version check: {msg}")
            ffmpeg_ok = False
        else:
            filt = check_ffmpeg_filters(pipeline_type)
            if filt:
                issues.append("Missing FFmpeg filters:")
                issues.extend(filt)

    # 3. VS Plugins (video only)
    if pipeline_type != "photo":
        vs = check_vs_plugins(pipeline_type)
        if vs:
            issues.append("Missing VapourSynth plugins:")
            issues.extend(vs)

        # 4. havsfunc version (video only)
        ok, msg = check_havsfunc_version()
        if not ok:
            issues.append(f"havsfunc: {msg}")

        # 5. VS minimal calls (video only)
        calls = check_vs_minimal_calls()
        if calls:
            issues.append("VapourSynth filter test failures:")
            issues.extend(calls)

    # 6. Python packages
    py = check_python_packages(pipeline_type)
    if py:
        issues.append("Missing Python packages:")
        issues.extend(py)

    # 7. Face detection packages (if AI enabled)
    if config and config.get("ai", {}).get("enabled", False):
        face = check_face_detection_packages(config)
        if face:
            issues.append("Missing face detection packages (AI enabled):")
            issues.extend(face)

    # 8. AI CLI tools (if AI enabled)
    if config and config.get("ai", {}).get("enabled", False):
        ai_cli = check_ai_cli_tools()
        if ai_cli:
            issues.append("Missing AI CLI tools:")
            issues.extend(ai_cli)

    # 9. AI models (if AI enabled)
    if config and config.get("ai", {}).get("enabled", False):
        models = check_ai_models(config.get("model_dir", "./models"), skip_md5_check=skip_md5_check)
        if models:
            issues.append("Missing AI model weights:")
            issues.extend(models)

    # 10. arnndn models (audio fallback, or FFmpeg <4.4)
    arnndn = check_arnndn_models(config, ffmpeg_version_ok=ffmpeg_ok)
    if arnndn:
        issues.append("Missing arnndn models (audio fallback):")
        issues.extend(arnndn)

    # 11. G'MIC filters (all pipelines)
    gmic = check_gmic_filters()
    if gmic:
        issues.append("G'MIC filter warnings:")
        issues.extend(gmic)

    # 12. GPU backends
    has_gpu, gpu_msgs = check_gpu_backends()
    if not has_gpu:
        issues.append("GPU backend check:")
        issues.extend(gpu_msgs)

    # 13. FFmpeg filter minimal calls (video only)
    if pipeline_type != "photo":
        ffmpeg_call_issues = check_ffmpeg_filters_minimal_calls(pipeline_type)
        if ffmpeg_call_issues:
            issues.append("FFmpeg filter test failures:")
            issues.extend(ffmpeg_call_issues)

    # 14. Spawn compliance (all pipelines, when multiprocessing used)
    # ISSUE-SHARED-DC-002 fix: Removed pipeline_type != "photo" restriction.
    # Photo pipeline also uses multiprocessing (--workers) and must comply.
    spawn_issues = check_spawn_compliance()
    if spawn_issues:
        issues.append("Spawn compliance warnings:")
        issues.extend(spawn_issues)

    return len(issues) == 0, issues


def main():
    """CLI entry for dependency check."""
    import yaml  # type: ignore[import-untyped]
    import argparse
    
    parser = argparse.ArgumentParser(description="Dependency Check")
    parser.add_argument("--config", "-c", help="Config file path")
    parser.add_argument("--pipeline-type", choices=["photo", "video", "auto"],
                       default="auto", help="Pipeline type (auto-detect by default)")
    parser.add_argument("--skip-md5-check", action="store_true",
                        help="Skip AI model MD5 verification (not recommended)")
    args = parser.parse_args()

    config = None
    if args.config and Path(args.config).exists():
        with open(args.config, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
    else:
        for p in ["./config.yaml", "../config.yaml"]:
            if Path(p).exists():
                with open(p, "r", encoding="utf-8") as f:
                    config = yaml.safe_load(f)
                break

    pipeline_type = None if args.pipeline_type == "auto" else args.pipeline_type

    ok, issues = run_all_checks(config, pipeline_type=pipeline_type, skip_md5_check=args.skip_md5_check)

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
