# Video & Photo Restoration Pipeline

Hybrid restoration pipeline combining traditional signal processing with AI super-resolution for vintage video and photo restoration.

## Overview

This project provides two restoration pipelines:

- **Video Restoration Pipeline**: For analog tape, DV digital, and HDV/AVCHD video materials
- **Photo Restoration Pipeline**: For scanned prints and digital photos

Both pipelines follow a multi-stage architecture that preserves original quality while applying modern restoration techniques.

## Quick Start

### Prerequisites

- Python 3.9+
- FFmpeg 4.4+ (with `bandreject` filter supporting `width_type=h`)
- ExifTool
- G'MIC (for photo processing)
- VapourSynth (for video processing)
- havsfunc r33

### Installation

1. Clone the repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install AI model weights to `./models/`:
   - `RealESRGAN_x2plus.pth`
   - `RealESRGAN_x4plus.pth`
   - `GFPGANv1.4.pth`
   - `DAT_x2plus.pth`, `DAT_x4plus.pth`

4. Run dependency check:
   ```bash
   python shared_modules/dependency_check.py -c video_restoration_pipeline/config.yaml --pipeline-type video
   python shared_modules/dependency_check.py -c photo_restoration_pipeline/config.yaml --pipeline-type photo
   ```

### Usage

#### Video Restoration

```bash
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    -i input_video.avi \
    -o output.mkv
```

#### Photo Restoration

```bash
python photo_restoration_pipeline/main.py \
    -c photo_restoration_pipeline/config.yaml \
    -i input_photo.jpg \
    -o output.jpg
```

## Architecture

See [architecture.md](architecture.md) for detailed system architecture.

See [PROJECT_MANUAL.md](PROJECT_MANUAL.md) for complete operation manual.

## Directory Structure

```
├── video_restoration_pipeline/     # Video restoration stages
│   ├── classify.py                  # Video material classifier
│   ├── main.py                      # Video pipeline controller
│   ├── config.yaml                  # Video configuration
│   └── stages/                      # Video processing stages
│       ├── stage0_audio_extract.py
│       ├── stage1_preprocess.py
│       ├── stage2_traditional_check.py
│       ├── stage3_ai_upscale.py
│       ├── stage4_postprocess.py
│       ├── stage5_ai_face.py
│       ├── stage6_grain_synthesis.py
│       └── stage7_encode.py
│
├── photo_restoration_pipeline/     # Photo restoration stages
│   ├── classify.py                  # Photo material classifier
│   ├── main.py                      # Photo pipeline controller
│   ├── config.yaml                  # Photo configuration
│   └── stages/                      # Photo processing stages
│       ├── stage1_preprocess.py
│       ├── stage2_traditional_check.py
│       ├── stage3_ai_upscale.py
│       ├── stage4_postprocess.py
│       ├── stage5_ai_face.py
│       └── stage6_encode.py
│
├── shared_modules/                  # Shared modules
│   ├── checkpoint.py                # Checkpoint/resume support
│   ├── dependency_check.py          # Dependency validation
│   ├── logger.py                    # Structured JSON Lines logging
│   ├── audio_utils.py               # Audio analysis utilities
│   ├── manifest_generate.py         # Manifest generation
│   ├── manifest_editor.py           # Manifest editing
│   ├── manifest_loader.py           # Manifest loading
│   └── preview.py                   # Preview utilities
│
└── models/                          # AI model weights directory
```

## Version History

- **v6.3**: Technical specification for video restoration
- **v7.3**: Architecture specification with photo pipeline, TUI, manifest editor
- **v7.4+**: Hybrid pipeline with both video and photo restoration

## Code Quality & Audit Reports

Comprehensive per-file audit reports are available in the project root (e.g., `dependency_check-audit.md`, `video_stage1_preprocess-audit.md`, etc.). These documents detail compliance against the v6.32 and v7.32 specifications, identifying bugs, missing features, and improvement opportunities.

### Key Known Issues

| ID | Component | Severity | Summary |
|----|-----------|----------|---------|
| BUG-001 | shared_modules/checkpoint.py | **Critical** | `verify_ffprobe_decodeable` is referenced but not defined — will crash on FAILED task resume |
| BUG-002 | video/main.py | **Critical** | `_worker_wrapper` is a nested function — multiprocessing `spawn` will fail with `PicklingError` |
| BUG-003 | shared_modules/checkpoint.py | **Critical** | `is_retryable_error` is referenced by video/main.py but never defined |
| BUG-004 | shared_modules/manifest_loader.py | **Critical** | References `verify_ffprobe_decodeable` which does not exist |
| BUG-005 | shared_modules/checkpoint.py | **High** | `is_retryable_error` returns 88 (unknown exit code) for CUDA OOM, but the real exit code is 83 |
| BUG-006 | video/stage1_preprocess.py | **High** | `haf.QTGMC()` call is inside `elif field_order == "unknown"` — completely missing for `tt`/`bb` |
| BUG-007 | shared_modules/checkpoint.py | **High** | `grain_factory3` params missing `temp_avg`; `audio_override` location ambiguous |
| BUG-008 | video/stage1_preprocess.py | **High** | `validate_crop()` lacks <50% and min 240px dimension checks |
| BUG-009 | video/stage2_traditional_check.py | **High** | Progressive <90% only logs warning, no IVTC fallback triggered |
| BUG-010 | video/stage7_encode.py | **Medium** | WSL copyback returns exit code 82 even when file exists on Linux side |
| BUG-011 | video/stage5_ai_face.py | **Medium** | Frame-to-time assumes 25fps; hardcoded `libsvtav1 CRF24` ignores config |
| BUG-012 | video/stage6_grain_synthesis.py | **Medium** | `sys.path` missing `shared_modules` — logger import fails at runtime |
| BUG-013 | video/stage7_encode.py | **Medium** | SVT-AV1 param string built twice redundantly |
| BUG-014 | video/stage3_ai_upscale.py | **Medium** | Hardcoded `-framerate 25` — source fps not preserved |
| BUG-015 | shared_modules/preview.py | **Low** | Preview clips use 10s duration instead of spec-mandated 0.5s |

See `shixian.md` for the complete 48-item issue register with fixes.

## License

MIT License
