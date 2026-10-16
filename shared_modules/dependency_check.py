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
import ast
import os
import shutil
import subprocess
import importlib.util
import re
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any

SCRIPT_DIR = Path(__file__).parent.resolve()
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

# R2-08: `arnndn=m=...` 的模型路径必须经 filter_option_path 转义（见
# audio_utils 的 docstring）。此前该函数在本模块**被调用但未导入**，
# 一旦 models/std.rnnn 存在、真正进入该分支就抛
# `NameError: name 'filter_option_path' is not defined`（独立用例 A05 复现）。
from audio_utils import filter_option_path  # noqa: E402


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
        # FIX-P3-7: os now imported at module top per CLAUDE.md §4 (import order)
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
        # R4: 有些打包（如 gyan.dev 的 Windows 构建）的版本串是
        # `ffmpeg version 2023-11-13-git-67a2571a55-full_build-www.gyan.dev`
        # —— 用**发布日期**当版本号，上面 `\d+\.\d+` 匹配不到，
        # 结果是"Cannot parse FFmpeg version"并被当成依赖缺失（exit 80）。
        # 日期构建可以按年份判定：FFmpeg 4.4 发布于 2021 年，
        # 因此 2021 年及以后的构建必然满足 >= 4.4。
        md = re.search(r'version\s+(\d{4})-(\d{2})-(\d{2})', first_line)
        if md:
            year = int(md.group(1))
            stamp = "-".join(md.groups())
            if year >= 2021:
                return True, (f"FFmpeg date-based build {stamp} "
                              f"(>= 2021, satisfies >= {min_version}; exact "
                              "version string not exposed by this build)")
            return False, (f"FFmpeg date-based build {stamp} "
                           f"(requires >= {min_version})")
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
    # FIX-P3-4: Improved FFmpeg filter parsing per v7.32 §3.6
    # ORIGINAL: parsed parts[1] without verifying FLAGS column format,
    # could misinterpret header lines or description words as filter names.
    available = set()
    for line in out.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            # FFmpeg -filters format: FLAGS FILTER_NAME ...description
            # FLAGS is like "A..T", ".F.T", etc. (letters and dots only)
            first = parts[0]
            if re.match(r'^[A-Z\.]{1,5}$', first):
                available.add(parts[1])
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

    # FIX-P2-4: Test 6: compand (audio dynamic range compression)
    # per v7.32 §3.6
    # R4: 原测试参数 `-90/-60|-60/-30|-15/-15|-15/-15|-30/0` 的**输入值不是递增的**
    # （-30 排在 -15 之后），ffmpeg 直接报
    # "Transfer function input values must be increasing" → 自检永远失败。
    # 生产代码（audio_utils.py）用的是递增曲线且是有效的；
    # 自检应当验证**生产实际使用的**参数，否则等于没验证。
    code, _, err = run_cmd([
        "ffmpeg", "-f", "lavfi", "-i", "sine=frequency=1000:duration=1",
        "-af", "compand=0.3:1:-90/-900|-70/-70|-30/-9|0/-3:6:0:0:0.5",
        "-f", "null", "-"
    ], timeout=15)
    if code != 0:
        errors.append(f"  - compand test failed: {err[:200]}")

    # FIX-P2-4: Test 7: arnndn (only if model file available) per v6.32 §7.30
    # FIX-P1-8: arnndn uses std.rnnn (official .rnnn format, not .pb).
    arnndn_model = Path("./models/std.rnnn")
    if arnndn_model.exists():
        code, _, err = run_cmd([
            "ffmpeg", "-f", "lavfi", "-i", "sine=frequency=50:duration=1",
            "-af", f"arnndn=m={filter_option_path(arnndn_model)}",
            "-f", "null", "-"
        ], timeout=15)
        if code != 0:
            errors.append(f"  - arnndn test failed: {err[:200]}")

    return errors


# FIX-P2-4: Minimum VS plugin versions keyed by namespace/name alias per
# v7.32 §3.6. NOTE: VapourSynth's get_plugins() exposes an integer API
# "version" for most plugins, which cannot be reliably compared to a semver
# minimum. The comparison therefore ONLY fires when the plugin exposes a
# dotted version string (semver-like); bare integer versions are skipped to
# avoid false positives. Full enforcement awaits confirmation of the version
# field semantics (tracked as 待用户确认).
MIN_PLUGIN_VERSIONS: Dict[str, str] = {
    "mvtools2": "23",
    "nnedi3": "0.5",
    "tcanny": "1.2",
}


def _parse_version(v) -> Optional[Tuple[int, ...]]:
    """Parse a dotted version string into an int tuple. Returns None if not a
    dotted semver-like string. Bare integers (e.g. VS API version "3") are
    rejected to avoid confusing API version with release version."""
    if not isinstance(v, str):
        return None
    m = re.match(r'^\D*(\d+\.\d+(?:\.\d+)*)', v.strip())
    if not m:
        return None
    return tuple(int(p) for p in m.group(1).split("."))


def _ver_lt(a, b) -> bool:
    """True if dotted version a < dotted version b. Returns False if either
    side is not a dotted version string (conservative: never flags)."""
    pa = _parse_version(a)
    pb = _parse_version(b)
    if pa is None or pb is None:
        return False
    n = max(len(pa), len(pb))
    pa = pa + (0,) * (n - len(pa))
    pb = pb + (0,) * (n - len(pb))
    return pa < pb


def _iter_vs_plugins(core: Any) -> List[Dict[str, Any]]:
    """NEW-RV-02: enumerate loaded VapourSynth plugins across API generations.

    * VapourSynth R81: ``core.plugins()`` → iterable of ``Plugin`` objects with
      ``name`` / ``namespace`` / ``version`` attributes.
    * Older cores: ``core.get_plugins()`` → dict of ``{id: {...}}``.

    Returns a normalised list of dicts so the caller never has to care.
    """
    raw: Any = None
    if hasattr(core, "plugins"):
        try:
            raw = list(core.plugins())
        except Exception:
            raw = None
    if raw is None and hasattr(core, "get_plugins"):
        try:
            got = core.get_plugins()
            raw = list(got.values()) if isinstance(got, dict) else list(got)
        except Exception:
            raw = None
    if not raw:
        return []

    out: List[Dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append({
                "name": str(item.get("name", "")),
                "namespace": str(item.get("namespace", "")),
                "version": item.get("version"),
            })
        else:
            out.append({
                "name": str(getattr(item, "name", "") or ""),
                "namespace": str(getattr(item, "namespace", "") or ""),
                "version": getattr(item, "version", None),
            })
    return out


def check_vs_plugins(pipeline_type: str = "photo") -> List[str]:
    """Check VapourSynth plugins. Skipped for photo pipeline."""
    if pipeline_type == "photo":
        return []
    missing = []
    try:
        import vapoursynth as vs
        core = vs.core
        # NEW-RV-02（本机实测 VapourSynth R81）：``core.get_plugins()`` 已被
        # 移除，调用它抛 "No attribute with the name get_plugins exists..."，
        # 于是整个插件检查变成一条"检查出错"的阻断项 —— 真正的缺失插件反而
        # 一条都没列出来。R81 的 API 是 ``core.plugins()``，返回 Plugin 对象
        # 的可迭代对象（属性 name/namespace/version），不再是 {id: dict}。
        plugins = _iter_vs_plugins(core)
        loaded = set()
        # FIX-P2-4: capture version strings (best-effort) for min-version check
        plugin_versions: Dict[str, str] = {}
        for pinfo in plugins:
            name = str(pinfo.get("name", "")).lower()
            namespace = str(pinfo.get("namespace", "")).lower()
            loaded.add(name)
            loaded.add(namespace)
            ver = pinfo.get("version")
            if ver is not None:
                plugin_versions[name] = str(ver)
                plugin_versions[namespace] = str(ver)

        # R4: 对齐**现代 namespace**。这些别名是实测得到的
        # （VapourSynth R72 + 当前 vsrepo 包），不是凭印象改的：
        #   knlmeanscl -> knlm | addgrainc -> grain | removegrain -> rgvs
        #   dftest -> dfttest | rgtools -> rgvs/rgsf
        # 原来的名字在当前生态里一个都匹配不上，于是"明明装了却报缺失"。
        required = {
            "VIVTC": ["vivtc"],
            "Deblock_QED": ["deblock_qed", "deblock"],
            "KNLMeansCL": ["knlmeanscl", "knlm"],
            "MVTools2": ["mvtools2", "mv"],
            "NNEDI3": ["nnedi3"],
            "NNEDI3CL": ["nnedi3cl"],
            "RgTools": ["rgtools", "rgvs", "rgsf"],
            "TCanny": ["tcanny"],
            "FFT3DFilter": ["fft3dfilter", "neo-fft3d"],
            "AddGrainC": ["addgrainc", "grain"],
            "RemoveGrain": ["removegrain", "rgvs"],
            "DFTTest": ["dftest", "dfttest"],
        }

        # R4: 这几项是 **Python 脚本/函数**，不是 VS 插件 namespace，
        # 用 namespace 查永远查不到 —— 必须按模块 + 函数存在性检查。
        py_scripts = {
            "havsfunc": ("havsfunc", ["QTGMC", "GrainFactory3", "SMDegrain", "Stab"]),
            # R4: 默认 denoise="TemporalDegrain"（v1）由 fvsfunc 提供
            # （havsfunc r33 没有这个函数），所以它是默认路径的必需依赖。
            "fvsfunc": ("fvsfunc", ["TemporalDegrain"]),
            # R4: 默认 stabilize="DepanStabilize" 走 haf.Stab（内部即 mvtools depan）
            "Stab": ("havsfunc", ["Stab"]),
            "QTGMC": ("havsfunc", ["QTGMC"]),
            "GrainFactory3": ("havsfunc", ["GrainFactory3"]),
            "TemporalDegrain2": (
                # VS 版 TemporalDegrain2 由 **G41Fun** 提供（havsfunc 里没有，
                # 实测 havsfunc r33 只有 SMDegrain）；G41Fun 不在 vsrepo 上，
                # 需从 GitHub 获取。
                "G41Fun", ["TemporalDegrain2"]),
        }

        for plugin, aliases in required.items():
            found_alias = next((a for a in aliases if a in loaded), None)
            if not found_alias:
                missing.append(f"  - VS Plugin '{plugin}' (tried: {aliases})")
                continue
            # FIX-P2-4: best-effort minimum-version check (semver only).
            min_ver = next((MIN_PLUGIN_VERSIONS[a] for a in aliases
                            if a in MIN_PLUGIN_VERSIONS), None)
            if min_ver is not None:
                actual = plugin_versions.get(found_alias)
                if actual is not None and _ver_lt(actual, min_ver):
                    missing.append(
                        f"  - VS Plugin '{plugin}' version {actual} < {min_ver}"
                    )
        # R4: Python 脚本类依赖按模块检查（namespace 里根本不会有它们）
        for label, (mod_name, funcs) in py_scripts.items():
            try:
                mod = __import__(mod_name)
            except Exception:
                missing.append(
                    f"  - VS script '{label}' (python module '{mod_name}' "
                    f"not importable)")
                continue
            absent = [f for f in funcs if not hasattr(mod, f)]
            if absent:
                missing.append(
                    f"  - VS script '{label}' (module '{mod_name}' lacks "
                    f"function(s): {', '.join(absent)})")

    except ImportError:
        missing.append("  - vapoursynth Python package not installed")
    except Exception as e:
        missing.append(f"  - Error checking VS plugins: {e}")

    return missing


def check_havsfunc_version() -> Tuple[bool, str]:
    """Check havsfunc is r33 (r34 has incomplete function set).

    FIX-P1-6: r33 source has NO __version__ attribute, so the old
    getattr(haf, "__version__") always returned None → false failure.
    New approach per v7.32 §3.6:
    1. Primary: importlib.metadata.version("havsfunc") (PEP 396).
    2. Fallback: function-existence test (QTGMC/GrainFactory3/TemporalDegrain/
       srestore/DepanStabilize) — r34 lacks some of these, so a missing
       function excludes r34 without relying on __version__.
    Installation: pip install 'havsfunc==33' (r34 is NOT compatible).
    """
    try:
        import havsfunc as haf  # noqa: F401
    except ImportError:
        return False, "havsfunc not installed (pip install 'havsfunc==33')"
    except Exception as e:
        return False, f"Error importing havsfunc: {e}"

    # Primary: importlib.metadata
    version = None
    try:
        import importlib.metadata as _meta
        version = _meta.version("havsfunc")
    except Exception:
        version = None
    if version is not None:
        vstr = str(version)
        if re.match(r"^r?33($|\.0*$)", vstr):
            return True, f"havsfunc r{version}"
        return False, (f"havsfunc version {version} (requires r33; r34 has incomplete "
                       f"function set; install: pip install 'havsfunc==33')")

    # Fallback: function-existence test (r33 has __version__ absent, but has
    # the full function set; r34 lacks some functions).
    # R4: 判据必须换成 **r33 实际拥有** 的函数。
    # 实测 havsfunc r33：QTGMC / GrainFactory3 / SMDegrain / srestore 都有，
    # 但 **TemporalDegrain 与 DepanStabilize 并不存在**（r31 之后移除）。
    # 于是原判据连"正确版本 r33"都会判失败 —— 这不是放宽，是修正判据：
    # r34 移除了 99% 的函数（官方 release note），这四个同样不会有。
    required_funcs = ["QTGMC", "GrainFactory3", "SMDegrain", "srestore"]
    missing_funcs = [f for f in required_funcs if not hasattr(haf, f)]
    if missing_funcs:
        return False, (f"havsfunc missing functions {missing_funcs} (likely r34 which is "
                       f"incomplete; install r33: pip install 'havsfunc==33')")
    return True, "havsfunc r33 (verified via function presence; __version__ absent in r33)"


def check_vs_minimal_calls() -> Tuple[List[str], List[str]]:
    """Execute minimal parameter set test calls for core filters.

    FIX-P1-1: Returns (errors, warnings). sRestore is an OPTIONAL dependency
    per v7.32 §3.6 — its absence only disables IVTC auto-fallback and must
    NOT block startup. sRestore failures are therefore routed to warnings
    instead of errors. Core filters (QTGMC/TemporalDegrain/GrainFactory3/
    DepanStabilize) remain blocking errors.
    """
    errors = []
    warnings = []
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

        # Test 3: TemporalDegrain (r33 function name; no "2" suffix)
        # FIX-P1-9: r33 signature is TemporalDegrain(inpClip, sigma=16, blockWidth=16, ...)
        # — NO degrainTR/grainLevel params (those are AviSynth TemporalDegrain2).
        try:
            clip3 = core.std.BlankClip(format=vs.YUV420P16, width=720, height=480, length=60)
            result3 = haf.TemporalDegrain(clip3, sigma=16, blockWidth=16)
            if result3 is None:
                errors.append("  - TemporalDegrain returned None")
        except AttributeError:
            errors.append("  - TemporalDegrain not found (havsfunc may be r34 with "
                          "incomplete functions; install r33: pip install 'havsfunc==33')")
        except Exception as e:
            errors.append(f"  - TemporalDegrain test failed: {e}")

        # FIX-P1-10: DepanStabilize needs motion-vector data from MAnalyse/
        # MDepan first; calling it directly on a BlankClip fails (no motion
        # data) → false "dependency missing" report. The production pipeline
        # uses haf.DepanStabilize (havsfunc wrapper) which internally runs
        # MDepan. So here we only check attribute existence, not a full call.
        try:
            mv = getattr(core, "mv", None)
            if mv is None or not hasattr(mv, "DepanStabilize"):
                errors.append("  - DepanStabilize not found in core.mv "
                              "(mvtools plugin missing; haf.DepanStabilize needs it)")
        except Exception as e:
            errors.append(f"  - DepanStabilize attribute check failed: {e}")

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
        # FIX-P1-1: sRestore is OPTIONAL per v7.32 §3.6. Failure only disables
        # IVTC auto-fallback; it must NOT block startup. Route to warnings.
        try:
            clip5 = core.std.BlankClip(format=vs.YUV420P16, width=720, height=480, length=60)
            result5 = haf.srestore(clip5, frate=23.976)
            if result5 is None:
                warnings.append("  - sRestore returned None — IVTC auto-fallback disabled")
        except Exception as e:
            warnings.append(
                f"  - sRestore unavailable ({e}) — IVTC auto-fallback disabled per v7.32 §3.6"
            )

    except ImportError as e:
        errors.append(f"  - Cannot import for minimal calls: {e}")
    except Exception as e:
        errors.append(f"  - Unexpected error in minimal calls: {e}")

    return errors, warnings


def check_python_packages(pipeline_type: str = "photo") -> List[str]:
    """Check required Python packages."""
    if pipeline_type == "photo":
        required = [
            ("numpy", "NumPy"),
            ("yaml", "PyYAML"),
            ("PIL", "Pillow"),
            ("scipy", "SciPy (for Stage 5 region separation + Laplacian pyramid grain)"),
            # FIX-R11R-12: imageio + tifffile are imported by photo stage1/3/4/5
            # (16bit RGB TIFF I/O) per v6.32 §4.2.1 / §7.16. They must be in the
            # §3.6 dependency list so their absence is caught at startup, not at
            # runtime mid-pipeline. ORIGINAL: only scipy was listed.
            ("imageio", "imageio (16bit RGB TIFF fallback I/O per v6.32 §4.2.1)"),
            ("tifffile", "tifffile (16bit RGB TIFF read/write per v6.32 §4.2.1)"),
            ("xxhash", "xxhash (FIX-P2-9: required for checkpoint hash per v7.32 §3.10.19)"),
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
            ("xxhash", "xxhash (FIX-P2-9: required for checkpoint hash per v7.32 §3.10.19)"),
        ]
    missing = []
    for mod_name, desc in required:
        spec = importlib.util.find_spec(mod_name)
        if spec is None:
            missing.append(f"  - {desc}: not installed")
    return missing


_FACE_DETECT_ORDER = [
    ("retinaface", "retina-face", "RetinaFace"),
    ("ultralytics", "ultralytics", "YOLOv8-face"),
    ("mtcnn", "mtcnn", "MTCNN"),
]


def check_face_detection_packages(config: Optional[Dict] = None) -> Tuple[List[str], Optional[str]]:
    """Check face detection packages and determine default model.

    Per v7.32 §3.6: checks in RetinaFace → YOLOv8-face → MTCNN order,
    first available becomes default.

    Returns (missing_list, default_model_name).
    """
    if config and not config.get("ai", {}).get("enabled", False):
        return [], None

    missing = []
    default_model = None
    for mod_name, pip_name, display_name in _FACE_DETECT_ORDER:
        spec = importlib.util.find_spec(mod_name)
        if spec is None:
            missing.append(f"  - {pip_name}: not installed (pip install {pip_name})")
        elif default_model is None:
            default_model = display_name

    return missing, default_model


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


# FIX-P2-5: AI model registry extracted to module level so both check_ai_models
# and download_missing_models share one source of truth. Per v7.32 §3.6: model
# validation is file existence + size>0 (no MD5/SHA256 — spec deprecated hash
# checks). Auto-download validates the same way after fetching.
_AI_MODELS_REGISTRY: Dict[str, Dict[str, Any]] = {
    "realesr-general-x4v3.pth": {
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/tag/v0.2.5.0",
        "aliases": ["RealESRGAN_x2plus.pth", "RealESRGAN_x4plus.pth"],  # optional alternates
        "optional_alt": True,
    },
    "GFPGANv1.4.pth": {
        "url": "https://github.com/TencentARC/GFPGAN/releases",
    },
    "DAT_2_x2.pth": {
        "url": "https://github.com/zhengchen1999/DAT (Google Drive)",
        "aliases": ["DAT_x2.pth"],  # optional alternates
        "optional_alt": True,
    },
    "DAT_2_x4.pth": {
        "url": "https://github.com/zhengchen1999/DAT (Google Drive)",
        "aliases": ["DAT_x4.pth"],
        "optional_alt": True,
    },
    "codeformer.pth": {
        "url": "https://github.com/sczhou/CodeFormer (Google Drive)",
        "aliases": ["CodeFormer.pth", "codeformer_v0.1.pth",
                    "weights/CodeFormer/codeformer.pth"],
    },
    "detection_Resnet50_Final.pth": {
        "url": "facexlib weights (place in facexlib/weights/ or models/)",
        "optional": True,  # RetinaFace auto-downloads if absent
    },
}


def _is_direct_download_url(url: str) -> bool:
    """Heuristic: True if url looks like a direct file download (not a web page)."""
    if not isinstance(url, str) or not url.startswith(("http://", "https://")):
        return False
    low = url.lower()
    if "releases/download/" in low or "raw.githubusercontent.com" in low:
        return True
    return low.endswith((".pth", ".pt", ".onnx", ".bin", ".rnnn", ".caffemodel"))


def download_model(name: str, url: str, dest_dir: str = "./models") -> bool:
    """Download a model file. Per v7.32 §3.6: validate presence + size>0
    after download (no MD5/SHA256 checksum — spec deprecated hash checks).

    FIX-P2-5: Returns True only if the file exists and is non-zero after
    download. Uses requests (lazy import — only required when auto-download
    is actually used).
    """
    try:
        import requests  # type: ignore[import-untyped]
    except ImportError:
        print(f"  - Cannot auto-download {name}: 'requests' not installed")
        return False
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)
    target = dest / name
    try:
        r = requests.get(url, stream=True, timeout=300)
        r.raise_for_status()
        with open(target, "wb") as f:
            for chunk in r.iter_content(65536):
                f.write(chunk)
    except Exception as e:
        print(f"  - Failed to download {name} from {url}: {e}")
        return False
    # Presence + size>0 validation only (no checksum per spec)
    return target.exists() and target.stat().st_size > 0


def download_missing_models(models_dir: str = "./models") -> Tuple[int, List[str]]:
    """Download missing required AI models whose URL is a direct download link.

    FIX-P2-5: Per v7.32 §3.6 (auto-download support). Only attempts models
    with a direct file URL; GitHub release *page* URLs / Google Drive links
    are skipped (cannot be fetched programmatically) and reported as still
    missing. Returns (count_downloaded, still_missing_names).
    """
    models_path = Path(models_dir).resolve()
    downloaded = 0
    still_missing: List[str] = []
    for name, info in _AI_MODELS_REGISTRY.items():
        if info.get("optional") or info.get("optional_alt"):
            continue
        primary = models_path / name
        aliases = info.get("aliases", [])
        if primary.exists() and primary.stat().st_size > 0:
            continue
        if any((models_path / a).exists() and (models_path / a).stat().st_size > 0
               for a in aliases):
            continue
        url = info.get("url", "")
        if _is_direct_download_url(url):
            print(f"  - Auto-downloading {name} from {url} ...")
            if download_model(name, url, str(models_path)):
                downloaded += 1
                continue
        still_missing.append(name)
    return downloaded, still_missing


def check_ai_models(models_dir: str = "./models", skip_md5_check: bool = False) -> Tuple[List[str], List[str]]:
    """Check AI model weights exist (file presence + size>0).

    FIX-P1-7: Removed MD5 checksum validation per v7.32 §3.6 (official
    releases mostly provide SHA256, not MD5; hard-coding unverified MD5s
    caused false mismatches). Now validates file existence + size>0 only.
    The skip_md5_check param is retained for backward CLI compatibility
    but no longer affects validation (no checksum is ever computed).
    """
    models_path = Path(models_dir).resolve()
    del skip_md5_check  # retained in signature for CLI backward compat

    # FIX-P2-5: registry now module-level (_AI_MODELS_REGISTRY) so
    # download_missing_models shares it.
    required_models = _AI_MODELS_REGISTRY
    missing = []
    warnings = []

    for model_name, info in required_models.items():
        model_path = models_path / model_name
        actual_path = model_path

        # Check primary path, then aliases
        if not model_path.exists():
            aliases = info.get("aliases", [])
            found = False
            for alias in aliases:
                alias_path = models_path / alias
                if alias_path.exists():
                    actual_path = alias_path
                    found = True
                    break
            if not found:
                if info.get("optional"):
                    warnings.append(
                        f"  - {model_name}: not found (optional; {info['url']})"
                    )
                elif info.get("optional_alt"):
                    # Acceptable if at least one alternate exists; the primary
                    # missing is only a warning since an alias satisfies it.
                    warnings.append(
                        f"  - {model_name}: not found in {models_path} "
                        f"(optional; an alias may be present: {aliases}; "
                        f"download: {info['url']})"
                    )
                else:
                    alias_info = f" (also checked aliases: {aliases})" if aliases else ""
                    missing.append(
                        f"  - {model_name}: not found in {models_path}{alias_info} "
                        f"(download: {info['url']})"
                    )
                continue

        # Validate file size > 0 (presence-based check, no MD5)
        try:
            if actual_path.stat().st_size == 0:
                missing.append(f"  - {model_name}: exists but is 0 bytes (corrupt?)")
        except Exception as e:
            missing.append(f"  - {model_name}: cannot stat ({e})")
    return missing, warnings


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
    # FIX-P1-8: arnndn uses official std.rnnn (Xiph RNNoise original generic
    # model from richardpl/arnndn-models, .rnnn format — NOT .pb). The old
    # arnndn_50Hz.pb/arnndn_60Hz.pb files do not exist in the official repo.
    # std.rnnn is a single generic model (model names are not frequency-based).
    model_file = config.get("audio", {}).get("arnndn_model", "std.rnnn")
    fpath = models_path / model_file
    if not fpath.exists():
        missing.append(
            f"  - {model_file}: not found (download: https://github.com/richardpl/arnndn-models)"
        )
    return missing


def check_gmic_filters() -> List[str]:
    """Check G'MIC supports required filters.

    FIX-A1: G'MIC 3.7.6 has NO -deblock/-deband/-despeckle/-film_grain/
    -remove_scratches/-transfer_colors/-match_histogram commands (verified via
    `gmic -help`). Replaced with real commands:
    - denoise_haar: scratch/deband approximation (Haar wavelet denoise)
    - denoise_patchpca: deblock approximation (patch-PCA denoise)
    - transfer_histogram: color/histogram transfer
    Photo grain uses pure Python Laplacian pyramid.

    FIX-A1-2: `gmic -h <name>` returns rc=0 with empty stdout for ANY name
    (even non-existent), so it cannot detect command existence. Use
    `gmic -help` once to get the full command list, then regex-match
    `^\\s+<name>:` (the documented command-definition format).
    """
    required_filters = ["denoise_haar", "denoise_patchpca", "transfer_histogram"]
    missing = []
    try:
        # FIX-A1-3: gmic -help output contains non-ASCII bytes (e.g. 0x92)
        # that break the default GBK codec on Windows. Use errors="replace"
        # so decoding never raises; we only need ASCII command names.
        r = subprocess.run(["gmic", "-help"],
                          capture_output=True, timeout=30, shell=False)
        if r.returncode != 0:
            return [f"  - G'MIC -help failed (rc={r.returncode})"]
        help_text = r.stdout.decode("utf-8", errors="replace")
    except Exception as e:
        return [f"  - Cannot run gmic -help ({e})"]

    for f in required_filters:
        # Command definitions appear as "  <name>:" in gmic -help output.
        if not re.search(rf"^\s+{re.escape(f)}:", help_text, re.MULTILINE):
            missing.append(f"  - G'MIC command '{f}' not available")
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


class _SpawnTargetVisitor(ast.NodeVisitor):
    """AST visitor to find multiprocessing Process/Pool target= arguments.

    FIX-P3-2: Per v7.32 §3.10.18 constraint 3, spawn target must be a
    module-level global function. This visitor detects:
    - multiprocessing.Process(target=...)
    - multiprocessing.Pool(..., initializer=...)
    - mp.Process / ctx.Process (aliased imports)
    - get_context('spawn').Process(target=...)

    Invalid targets: lambda, attribute access (method), call expression,
    subscript, nested function definition.
    """

    def __init__(self, label: str, issues: List[str]):
        self.label = label
        self.issues = issues
        self._mp_names: List[str] = []  # Names that refer to multiprocessing

    def _describe_node(self, node: ast.AST) -> str:
        """Return a human-readable description of an AST node for error messages."""
        if isinstance(node, ast.Lambda):
            return "lambda"
        elif isinstance(node, ast.Attribute):
            return f"attribute/method access ({ast.dump(node)})"
        elif isinstance(node, ast.Call):
            return f"function call ({ast.dump(node)})"
        elif isinstance(node, ast.Subscript):
            return f"subscript ({ast.dump(node)})"
        elif isinstance(node, ast.Name):
            return f"name '{node.id}'"
        else:
            return f"{type(node).__name__} ({ast.dump(node)})"

    def _is_mp_call(self, node: ast.Call) -> bool:
        """Check if a Call node is a multiprocessing Process/Pool call."""
        # multiprocessing.Process(...)
        if isinstance(node.func, ast.Attribute):
            attr = node.func.attr
            if attr in ("Process", "Pool"):
                # Check if the value is 'multiprocessing' or an alias
                if isinstance(node.func.value, ast.Name):
                    if node.func.value.id in self._mp_names:
                        return True
                # get_context('spawn').Process(...)
                if isinstance(node.func.value, ast.Attribute):
                    if node.func.value.attr == "get_context":
                        return True
                # multiprocessing.get_context('spawn').Process(...)
                if isinstance(node.func.value, ast.Call):
                    callee = node.func.value.func
                    if (isinstance(callee, ast.Attribute)
                            and callee.attr == "get_context"):
                        return True
        return False

    def _check_target_arg(self, node: ast.Call, arg_name: str,
                          line: int) -> None:
        """Check a single keyword argument (target/initializer)."""
        for kw in node.keywords:
            if kw.arg != arg_name:
                continue
            value = kw.value
            # Valid: simple Name reference (module-level function)
            if isinstance(value, ast.Name):
                continue
            # Invalid: everything else
            desc = self._describe_node(value)
            self.issues.append(
                f"  - {self.label}: spawn constraint 3 violation at line "
                f"{line}: '{arg_name}=' is {desc}, must be module-level "
                f"global function per v7.32 §3.10.18"
            )

    def visit_Import(self, node: ast.Import) -> None:
        """Track 'import multiprocessing' and aliases."""
        for alias in node.names:
            if alias.name == "multiprocessing":
                self._mp_names.append(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        """Track 'from multiprocessing import Process/Pool'."""
        if node.module and "multiprocessing" in node.module:
            for alias in node.names:
                if alias.name in ("Process", "Pool"):
                    self._mp_names.append(alias.asname or alias.name)
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        """Check multiprocessing.Process/Pool calls for invalid targets."""
        if self._is_mp_call(node):
            line = getattr(node, "lineno", 0)
            self._check_target_arg(node, "target", line)
            self._check_target_arg(node, "initializer", line)
        # Also check direct Process(...)/Pool(...) when imported directly
        if isinstance(node.func, ast.Name):
            if node.func.id in self._mp_names:
                line = getattr(node, "lineno", 0)
                self._check_target_arg(node, "target", line)
                self._check_target_arg(node, "initializer", line)
        self.generic_visit(node)


def _check_spawn_targets_ast(path: str, label: str,
                              issues: List[str]) -> None:
    """AST-based check for spawn constraint 3: target must be module-level.

    FIX-P3-2: Uses ast.parse() to recursively analyze multiprocessing
    Process/Pool calls and verify their target= parameter is a simple
    module-level global function reference, not a lambda, method, or
    nested function. Per v7.32 §3.10.18 constraint 3.
    """
    if not path or not Path(path).exists():
        return
    try:
        # R4: 必须容错。实测扫描中遇到非 UTF-8 的 .py（byte 0xfc）会抛
        # UnicodeDecodeError，直接把 run_all_checks 整个带崩，
        # 表现为依赖检查失败 -> exit 80，真正的问题反而看不到。
        content = Path(path).read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(content, filename=path)
    except SyntaxError:
        # If file has syntax errors, skip AST analysis (line-based check
        # in _check_single_file will catch obvious issues)
        return
    except Exception:
        return

    visitor = _SpawnTargetVisitor(label, issues)
    visitor.visit(tree)


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

    # FIX-P3-2: Improved spawn compliance pre-check per v7.32 §3.10.18
    # ORIGINAL: Only checked a narrow set of prefixes, missed top-level
    # constant assignments (X = ...) and decorator lines.
    _VALID_TOP_LEVEL_PREFIXES = (
        "#", "import", "from", "def ", "class ",
        "__", '"""', "'''", " ", "\t", "@",
    )
    # FIX-P2-3: Distinguish constant assignments from executable assignments
    # per v7.32 §3.10.18 constraint 2
    _TOP_LEVEL_ASSIGNMENT = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*\s*=\s')
    _TOP_LEVEL_EXECUTABLE = re.compile(r'^[A-Za-z_][A-Za-z0-9_]*\s*=\s.*\(')

    def _check_single_file(path: str, label: str) -> None:
        if not path or not Path(path).exists():
            return
        # R4: 必须容错。实测扫描中遇到非 UTF-8 的 .py（byte 0xfc）会抛
        # UnicodeDecodeError，直接把 run_all_checks 整个带崩，
        # 表现为依赖检查失败 -> exit 80，真正的问题反而看不到。
        try:
            content = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return
        lines = content.splitlines()
        in_main = False
        # FIX-P4-2: Track triple-quoted string state instead of the old
        # `if i > 20` guard (which skipped the first 20 lines and missed
        # executable code appearing after a long module docstring, while
        # also falsely flagging docstring interior lines). Per v7.32 §3.10.18.
        in_triple = False
        triple_tok = ""
        for i, line in enumerate(lines):
            stripped = line.strip()
            if stripped.startswith("if __name__"):
                in_main = True
                continue
            if in_main:
                continue
            # Skip lines inside a multi-line triple-quoted string
            if in_triple:
                if triple_tok in line:
                    in_triple = False
                continue
            # Detect a triple-quoted string opening that does not close on
            # the same line (module docstring / multi-line constant).
            for tok in ('"""', "'''"):
                if tok in stripped:
                    if stripped.count(tok) < 2:
                        in_triple = True
                        triple_tok = tok
                    break
            if in_triple:
                continue
            if stripped and not stripped.startswith(_VALID_TOP_LEVEL_PREFIXES):
                # Allow top-level constant assignments (X = ...)
                # FIX-P2-3: But flag executable assignments (X = func()) per
                # v7.32 §3.10.18 constraint 2
                if _TOP_LEVEL_ASSIGNMENT.match(stripped):
                    # Flag executable assignments (X = func())
                    if _TOP_LEVEL_EXECUTABLE.match(stripped):
                        issues.append(
                            f"  - {label}: top-level executable assignment "
                            f"at line {i+1}: {stripped[:60]}"
                        )
                        break
                    continue
                issues.append(
                    f"  - {label}: potential top-level executable code "
                    f"at line {i+1}: {stripped[:60]}"
                )
                break

    def _check_file_full(path: str, label: str) -> None:
        """Run both line-based (constraint 1+2) and AST-based (constraint 3) checks."""
        _check_single_file(path, label)
        # FIX-P3-2: AST-based spawn target analysis per v7.32 §3.10.18 constraint 3
        _check_spawn_targets_ast(path, label, issues)

    # Check main script
    if target_script is None:
        target_script = sys.argv[0] if sys.argv else None
    if target_script:
        _check_file_full(target_script, Path(target_script).name)

    # ISSUE-066 fix: check related modules
    if related_modules:
        for mod_path in related_modules:
            _check_file_full(mod_path, Path(mod_path).name)
    else:
        # Default: discover project Python files adjacent to the main script
        if target_script:
            base_dir = Path(target_script).parent
            # FIX-P3-5: Exclude virtualenv / site-packages / test dirs so the
            # spawn scan does not flag third-party or test code as project
            # spawn targets per v7.32 §3.10.18.
            _EXCLUDE_DIRS = {"__pycache__", ".venv", "venv", "env",
                             "site-packages", "test"}
            for py_file in base_dir.rglob("*.py"):
                if any(part in _EXCLUDE_DIRS for part in py_file.parts):
                    continue
                if py_file.name.startswith("test_"):
                    continue
                if py_file.resolve() == Path(target_script).resolve():
                    continue
                _check_file_full(str(py_file), py_file.name)

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
                   skip_md5_check: bool = False) -> Tuple[bool, List[str], List[str]]:
    """Run all dependency checks. Returns (all_ok, list_of_issues).
    
    Args:
        config: Loaded config dict (used for auto-detection)
        pipeline_type: Explicit pipeline type override ("photo" or "video")
    """
    issues = []
    warnings = []

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

    # 2. FFmpeg version + filters
    # FIX-P3-5: Photo pipeline also needs FFmpeg/ffprobe for metadata reading
    # per v7.32 §3.6. Photo skips version requirement and audio filters.
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
    else:
        # Photo: verify FFmpeg/ffprobe exist (metadata reading required)
        # Version check is optional but logged for reference
        if not shutil.which("ffmpeg"):
            issues.append("FFmpeg not found (required for photo metadata reading)")
        if not shutil.which("ffprobe"):
            issues.append("FFprobe not found (required for photo stream analysis)")

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
        # FIX-P1-1: sRestore failures now non-blocking warnings per v7.32 §3.6
        calls, vs_warnings = check_vs_minimal_calls()
        if calls:
            issues.append("VapourSynth filter test failures:")
            issues.extend(calls)
        if vs_warnings:
            warnings.append("VapourSynth optional dependencies (non-blocking):")
            warnings.extend(vs_warnings)

    # 6. Python packages
    py = check_python_packages(pipeline_type)
    if py:
        issues.append("Missing Python packages:")
        issues.extend(py)

    # 7. Face detection packages (if AI enabled)
    # FIX-P1-3: Handle new Tuple return (missing_list, default_model)
    # per v7.32 §3.6: first available model becomes default
    if config and config.get("ai", {}).get("enabled", False):
        face_missing, face_default = check_face_detection_packages(config)
        if face_missing:
            issues.append("Missing face detection packages (AI enabled):")
            issues.extend(face_missing)
        if face_default:
            warnings.append(f"Face detection default model: {face_default}")

    # 8. AI CLI tools (if AI enabled)
    if config and config.get("ai", {}).get("enabled", False):
        ai_cli = check_ai_cli_tools()
        if ai_cli:
            issues.append("Missing AI CLI tools:")
            issues.extend(ai_cli)

    # 9. AI models (if AI enabled)
    if config and config.get("ai", {}).get("enabled", False):
        model_errors, model_warnings = check_ai_models(
            config.get("model_dir", "./models"), skip_md5_check=skip_md5_check
        )
        if model_errors:
            issues.append("Missing AI model weights:")
            issues.extend(model_errors)
        # FIX-P3-1-REV: Warnings are informational, do not block startup
        if model_warnings:
            warnings.append("AI model MD5 warnings (non-blocking):")
            warnings.extend(model_warnings)

    # 10. arnndn models (audio fallback, or FFmpeg <4.4)
    arnndn = check_arnndn_models(config, ffmpeg_version_ok=ffmpeg_ok)
    if arnndn:
        issues.append("Missing arnndn models (audio fallback):")
        issues.extend(arnndn)

    # 11. G'MIC filters (all pipelines)
    # FIX-P3-9: Label was "G'MIC filter warnings:" while the result was routed
    # to blocking issues — label/destination mismatch per v7.32 §3.6. G'MIC
    # filters (denoise_haar/denoise_patchpca/transfer_histogram) are required,
    # so this stays blocking with an accurate "Missing G'MIC filters:" label.
    gmic = check_gmic_filters()
    if gmic:
        issues.append("Missing G'MIC filters:")
        issues.extend(gmic)

    # 12. GPU backends
    # FIX-P1-2 / FIX-P3-8: GPU absence is non-blocking — CPU fallback is
    # allowed per v7.32 §3.5 (CUDA → ROCm → MPS → CPU auto-degrade). Only
    # block when AI is enabled AND PyTorch itself is missing (no backend at
    # all). AI disabled → pure WARNING.
    ai_enabled = bool(config and config.get("ai", {}).get("enabled", False))
    has_gpu, gpu_msgs = check_gpu_backends()
    if not has_gpu:
        if ai_enabled:
            # AI enabled: need at least CPU PyTorch to fall back to
            try:
                import torch  # noqa: F401
                warnings.append("GPU backend check (AI will run on CPU):")
                warnings.extend(gpu_msgs)
            except ImportError:
                issues.append("GPU backend check (AI enabled but PyTorch missing):")
                issues.extend(gpu_msgs)
        else:
            warnings.append("GPU backend check (AI disabled, not required):")
            warnings.extend(gpu_msgs)

    # 13. FFmpeg filter minimal calls (video only)
    if pipeline_type != "photo":
        ffmpeg_call_issues = check_ffmpeg_filters_minimal_calls(pipeline_type)
        if ffmpeg_call_issues:
            issues.append("FFmpeg filter test failures:")
            issues.extend(ffmpeg_call_issues)

    # 14. Spawn compliance (all pipelines, when multiprocessing used)
    # ISSUE-SHARED-DC-002 fix: Removed pipeline_type != "photo" restriction.
    # Photo pipeline also uses multiprocessing (--workers) and must comply.
    # FIX-P4-3: Label was "Spawn compliance warnings:" while routed to blocking
    # issues. Spawn violations are blocking (per v7.32 §3.10.18 they cause
    # runtime deadlocks), so the label is corrected to "violations". The
    # exit-code semantics (80 vs a dedicated 86) is a design decision tracked
    # as 待用户确认 (86=validation failure per §3.5 is the spec-aligned option).
    spawn_issues = check_spawn_compliance()
    if spawn_issues:
        issues.append("Spawn compliance violations:")
        issues.extend(spawn_issues)

    return len(issues) == 0, issues, warnings


def main():
    """CLI entry for dependency check."""
    import yaml  # type: ignore[import-untyped]
    import argparse
    
    parser = argparse.ArgumentParser(description="Dependency Check")
    parser.add_argument("--config", "-c", help="Config file path")
    parser.add_argument("--pipeline-type", choices=["photo", "video", "auto"],
                       default="auto", help="Pipeline type (auto-detect by default)")
    parser.add_argument("--skip-md5-check", action="store_true",
                        help="Skip AI model MD5 verification (no-op; MD5 checks removed per spec)")
    # FIX-P2-5: auto-download missing AI models per v7.32 §3.6
    parser.add_argument("--auto-download", action="store_true",
                        help="Auto-download missing AI models with direct URLs (per v7.32 §3.6)")
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

    # FIX-P2-5: attempt auto-download of missing AI models before the check.
    # Only acts when AI is enabled; GitHub release *page* / Google Drive URLs
    # are skipped (not direct files) and surface as still-missing in the check.
    if args.auto_download and config and config.get("ai", {}).get("enabled", False):
        models_dir = config.get("model_dir", "./models")
        print("Auto-downloading missing AI models...")
        dl_count, still_missing = download_missing_models(models_dir)
        print(f"  Downloaded {dl_count} model(s); {len(still_missing)} still "
              f"missing (need manual download): {still_missing}")

    ok, issues, warnings = run_all_checks(config, pipeline_type=pipeline_type, skip_md5_check=args.skip_md5_check)

    print("=" * 60)
    print("Dependency Check Results")
    print("=" * 60)
    if warnings:
        print(f"\n{len(warnings)} warning(s) (non-blocking):")
        for w in warnings:
            print(w)
    if ok:
        print("All required dependencies are satisfied.")
        return

    print(f"Found {len(issues)} issue(s):")
    for issue in issues:
        print(issue)
    print("\nInstall missing dependencies and retry.")

    # FIX-P4-3: Per v7.32 §3.5, spawn-compliance violations are validation
    # failures (exit 86), distinct from missing dependencies (exit 80). Spawn
    # violations cause runtime deadlocks (§3.10.18) and must be fixed in source,
    # not by installing packages. ORIGINAL: all issues mapped to exit 80, which
    # masked spawn violations as "install deps" misdiagnosis.
    has_spawn_violation = any(
        isinstance(i, str) and i.startswith("Spawn compliance violations:")
        for i in issues
    )
    if has_spawn_violation:
        sys.exit(86)
    sys.exit(80)


if __name__ == "__main__":
    main()
