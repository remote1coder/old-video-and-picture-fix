# Project Architecture

## 1. Pipeline Controller

### 1.1 Entry Point
- `video_restoration_pipeline/main.py` — Video controller
- `photo_restoration_pipeline/main.py` — Photo controller

### 1.2 Parameter Merge Hierarchy
6-layer parameter cascading per v7.32 §3.1:

1. `global` defaults — base configuration
2. `paths[type]` — type-specific defaults (video: analog_tape/dv_digital/hdv_avchd; photo: scan_print/digital_photo)
3. `paths.audio_override` (video) / `photo.audio_override` (photo) — audio parameter overrides per material type
4. `recommended_params` — classifier recommendations (from manifest or classify.py)
5. `user_overrides` — manifest editor overrides
6. CLI arguments — highest priority

**注意**：当前实现中 `audio_override` 的层级位置存在歧义。photo 流水线将其视为 `recommended_params` 的一部分，而 video 流水线将其作为独立层级。这可能导致音频参数在 photo 模式下的合并行为与预期不符。

### 1.3 Core Workflow
- Parse CLI arguments
- Load and validate config (YAML)
- Initialize logger (per-job, structured JSON Lines)
- Classify material → get detected_class, confidence, params
- Merge 6-layer parameters
- Execute stages sequentially
- Each stage is a separate Python script call with subprocess.run()
- Optional: retry on transient errors

---

## 2. Stage System

### 2.1 Stage Scripts

#### Video Pipeline Stages
| Stage | File | Purpose | Input → Output |
|-------|------|---------|----------------|
| 0 | `stages/stage0_audio_extract.py` | Audio extraction | video → audio.flac |
| 1 | `stages/stage1_preprocess.py` | Preprocessing (QTGMC, de-block, etc.) | video → processed video |
| 2 | `stages/stage2_traditional_check.py` | Quality verification | processed video → verified |
| 3 | `stages/stage3_ai_upscale.py` | AI super-resolution | verified → upscaled |
| 4 | `stages/stage4_postprocess.py` | Post-processing | upscaled → refined |
| 5 | `stages/stage5_ai_face.py` | Face restoration | refined → face-restored |
| 6 | `stages/stage6_grain_synthesis.py` | Film grain synthesis | face-restored → grain-added |
| 7 | `stages/stage7_encode.py` | Final encoding | grain-added → final output |

#### Photo Pipeline Stages
| Stage | File | Purpose | Input → Output |
|-------|------|---------|----------------|
| 1 | `stages/stage1_preprocess.py` | Preprocessing (colorspace, scratch removal, deblock, deband) | photo → processed |
| 2 | `stages/stage2_traditional_check.py` | Quality verification | processed → verified |
| 3 | `stages/stage3_ai_upscale.py` | AI super-resolution | verified → upscaled |
| 4 | `stages/stage4_postprocess.py` | Post-processing | upscaled → refined |
| 5 | `stages/stage5_ai_face.py` | Face restoration | refined → face-restored |
| 6 | `stages/stage6_encode.py` | Final encoding | face-restored → final output |

### 2.2 Stage Communication
- Stages communicate via intermediate files in temp_dir
- File naming: `{job_id}_{input_name}_step{N}_{operation}_p{param_hash}.{ext}`
- param_hash = SHA-256 of file identity + quality params (v6.3 §3.7)

---

## 3. Classifier System

### 3.1 Video Classifier
- File: `video_restoration_pipeline/classify.py`
- Detects: ANALOG_TAPE, DV_DIGITAL, HDV_AVCHD
- Uses: ffprobe metadata, field order detection, PSF detection, SSIM analysis

### 3.2 Photo Classifier
- File: `photo_restoration_pipeline/classify.py`
- Detects: SCAN_PRINT, DIGITAL_PHOTO
- Uses: PIL metadata, EXIF (model/make/scanner), resolution heuristics

---

## 4. Shared Modules

All shared modules are in `shared_modules/` directory.

| Module | File | Purpose |
|--------|------|---------|
| Checkpoint | `checkpoint.py` | param_hash, intermediate naming, integrity verification |
| Logger | `logger.py` | Structured JSON Lines logging per job |
| Dependency Check | `dependency_check.py` | Pre-run validation of tools/libraries |
| Audio Utils | `audio_utils.py` | Hum detection, audio profiling |
| Manifest Loader | `manifest_loader.py` | Load/save YAML/JSONL manifests |
| Manifest Editor | `manifest_editor.py` | Edit manifest jobs |
| Manifest Generator | `manifest_generate.py` | Generate manifests from directories |
| Preview | `preview.py` | Preview frame extraction |

### 4.1 Python Path Setup
大多数 stage 脚本正确添加 shared_modules 到 sys.path：
```python
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "shared_modules"))
from logger import get_logger
from checkpoint import compute_param_hash
```

**⚠️ 已知问题**：`video_restoration_pipeline/stages/stage6_grain_synthesis.py` 的 sys.path 设置仅包含 `stages/` 目录，**未包含 `shared_modules/`，导致 `from logger import get_logger` 运行时导入失败。** 该问题需在 stage6 脚本开头添加正确的路径设置。

---

## 5. Configuration

### 5.1 Video Config
File: `video_restoration_pipeline/config.yaml`

Key sections:
- `global` — global defaults (crf, preset, tile_rows, tile_cols, retry_count)
- `paths` — per-type settings + audio_override
- `audio_override` — per-material audio params (hum_freq, hum_harmonics, denoise, etc.)
- `audio` — audio processing settings
- `grain_factory3` — film grain synthesis
- `ai` — AI upscaling configuration

### 5.2 Photo Config
File: `photo_restoration_pipeline/config.yaml`

Key sections:
- `global` — global defaults
- `paths` — per-type settings
- `photo` — photo-specific defaults
- `ai` — AI upscaling configuration

---

## 6. Logging

### 6.1 Log Format
Structured JSON Lines, one file per job:
```json
{"timestamp": "2024-01-01T12:00:00Z", "level": "INFO", "event": "STAGE_START", "message": "Starting stage 1", "job_id": "job_123"}
```

### 6.2 Log Rotation
Per-file: max 10000 lines, 5 backups. Large fields truncated with SHA-256 hash.

---

## 7. Checkpoint / Resume

### 7.1 param_hash
Deterministic hash based on:
- File identity: size, mtime, xxhash prefix (xxhash3_64)
- Quality params: crf, preset, AI model, deinterlace settings, etc.
- Excludes: job_id, output_path, log settings

**注意**：当前 `CORE_QUALITY_PARAMS` 白名单缺少 `qtgmc_input_type` 等参数，可能导致场序参数变更后仍命中旧的中间文件。

### 7.2 Intermediate Files
Naming: `{job_id}_{input_name}_step{N}_{operation}_p{param_hash}.{ext}`

### 7.3 Cleanup
- After each job: keep N most recent stages (configurable, **默认 5，规范要求 2**)
- Full cleanup on completion (configurable)

**⚠️ 已知问题**：
- `cleanup_old_intermediates()` 默认 `keep_steps=5`，但 v6.32 §7.5 要求保留最近 **2 个** stage
- `is_retryable_error()` 函数被多处引用但**未实现**，导致重试逻辑实际不可达
- `verify_ffprobe_decodeable()` 函数被 `manifest_loader.py` 引用但**未实现**，导致 FAILED 任务恢复时崩溃

---

## 8. Dependency Check

Run before first use:
```bash
python shared_modules/dependency_check.py -c video_restoration_pipeline/config.yaml --pipeline-type video
python shared_modules/dependency_check.py -c photo_restoration_pipeline/config.yaml --pipeline-type photo
```

Checks: FFmpeg version, VapourSynth plugins, Python packages, AI model files, G'MIC filters.

Returns exit code 80 if any required dependency is missing.

**注意**：当前依赖检查缺少以下规范要求的检测项：
- ROCm / MPS GPU 后端扩展检测
- FFmpeg 滤镜最小参数集测试调用（仅检查滤镜名存在性）
- 人脸检测 Python 包名一致性（代码检查 `retinaface`，规范可能要求 `retina-face`）
- `sRestore` 滤镜的显式独立校验（当前仅在 `check_vs_minimal_calls()` 中间接包含）
