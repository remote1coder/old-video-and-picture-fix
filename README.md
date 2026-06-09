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
- VapourSynth + havsfunc r33 (for video processing)
- avifenc (for photo AVIF output)

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
   - `DAT_x2.pth`, `DAT_x4.pth`

4. Run dependency check:
   ```bash
   python shared_modules/dependency_check.py -c video_restoration_pipeline/config.yaml --pipeline-type video
   python shared_modules/dependency_check.py -c photo_restoration_pipeline/config.yaml --pipeline-type photo
   ```

### Usage

#### Video Restoration

视频修复支持 **6 种运行模式**（v7.3 §3.3）：

```bash
# 模式 D（四步工作流）Step 1: 扫描输入目录生成分类清单
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    --scan -i /path/to/videos/

# 模式 D Step 2: 交互式审阅清单（调整参数覆盖）
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    --edit-manifest

# 模式 D Step 4 / 模式 F（批量执行，支持断点续传）
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    --mode batch --resume

# 模式 E（快速模式：扫描后立即执行，不保留清单）
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    --mode quick -i /path/to/videos/

# 模式 A（交互式单文件处理）
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    --mode interactive -i /path/to/single_video.avi

# 模式 C / 模式 B（预览模式：提取代表性帧和运动片段）
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    --mode preview -i /path/to/videos/

# 单 stage 调试运行
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    --stage 3 -i /path/to/video.avi
```

#### Photo Restoration

```bash
python photo_restoration_pipeline/main.py \
    -c photo_restoration_pipeline/config.yaml \
    -i input_photo.jpg \
    -o output.avif
```

**注意**: 照片修复默认输出 AVIF 格式（10bit YUV420 SDR）。如需 16bit RGB444 无损存档，在 config.yaml 中设置 `archival_mode: true`。

**已知限制**（详见 `bugs.md` / `shixian.md`）：
- AI 模型权重 MD5 校验支持外部清单 (`models/md5_manifest.json`)；若缺失，依赖检查会输出 WARNING 提示用户创建
- ~~视频人脸修复对非 25fps 素材的片段定位存在偏移风险~~ ✅ **已修复**：`stage5_ai_face.py` 已改为读取实际帧率
- ~~清单编辑器当前为命令行式，无 TUI 交互界面~~ ✅ **已修复**：`manifest_editor.py interactive` 子命令已提供基于 rich 的交互式 TUI（支持 y/e/s/p/q 快捷键审阅）


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
│   ├── main.py                      # Photo pipeline controller (single-file mode)
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

Comprehensive per-file audit reports are available in the project root (e.g., `photo_main-audit.md`, `checkpoint-audit.md`, `video_stage1_preprocess-audit.md`, etc.). These documents detail compliance against the v6.32 and v7.32 specifications, identifying bugs, missing features, and improvement opportunities.

### Key Known Issues (as of 2026-04-30)

| ID | Component | Severity | Summary | Status |
|----|-----------|----------|---------|--------|
| BUG-001 | shared_modules/checkpoint.py | **Fixed** | `verify_ffprobe_decodeable` was missing — now implemented | ✅ |
| BUG-002 | video/main.py | **Fixed** | `_worker_wrapper` nested function pickle issue — moved to module level | ✅ |
| BUG-003 | shared_modules/checkpoint.py | **Fixed** | `is_retryable_error` was missing — now implemented | ✅ |
| BUG-006 | video/stage1_preprocess.py | **Fixed** | QTGMC call was in wrong branch — swapped tt/bb and unknown | ✅ |
| BUG-011 | video/stage5_ai_face.py | **Fixed** | Hardcoded encoding params — now read from config | ✅ |
| BUG-012 | video/stage6_grain_synthesis.py | **Fixed** | `sys.path` missing shared_modules — corrected | ✅ |
| BUG-PHOTO-001 | photo/main.py | **Fixed** | `original_width/height` now injected at lines 335-336 | ✅ |
| BUG-PHOTO-002 | photo/main.py | **Fixed** | `--ai-upscale-factor` correctly maps to `upscale_scale` | ✅ |
| BUG-PHOTO-003 | photo/main.py | **Fixed** | `--ai-model` correctly maps to `upscale_model` | ✅ |
| BUG-PHOTO-004 | photo/main.py | **Fixed** | Retry interval now 5s/15s per spec | ✅ |
| BUG-PHOTO-005 | photo/main.py | **Fixed** | `--preview` calls `extract_preview_frames()` | ✅ |
| BUG-PHOTO-009 | photo/main.py | **Fixed** | `stage_idx` from `enumerate(stages)` | ✅ |
| BUG-PHOTO-013 | photo/stage3_ai_upscale.py | **Fixed** | 1080p check works with injected dimensions | ✅ |
| BUG-PHOTO-027 | shared_modules/checkpoint.py | **Fixed** | `wsl_copy_back` now uses `copyfile` + verify | ✅ |
| BUG-PHOTO-028 | shared_modules/checkpoint.py | **Fixed** | `gpu_memory_check` now queries available VRAM via nvidia-smi | ✅ |
| BUG-PHOTO-029 | shared_modules/checkpoint.py | **Fixed** | `CORE_QUALITY_PARAMS` includes photo params | ✅ |
| BUG-PHOTO-048 | photo/stage5_ai_face.py | **Fixed** | RetinaFace/MTCNN both apply confidence threshold | ✅ |
| BUG-PHOTO-050 | photo/stage5_ai_face.py | **Fixed** | CodeFormer Python fallback implements real inference | ✅ |
| BUG-SM-001 | shared_modules/checkpoint.py | **Critical** | `wsl_redirect_path` does not redirect `/mnt/` paths at all; direction reversed | ✅ **已修复** |
| BUG-SM-002 | shared_modules/checkpoint.py | **Critical** | `wsl_copy_back` missing MD5+size+ffprobe verification and atomicity | ✅ **已修复** |
| BUG-SM-003 | shared_modules/dependency_check.py | **Critical** | Missing CodeFormer model weight check | ✅ **已实现** |
| BUG-SM-004 | shared_modules/dependency_check.py | High | `check_havsfunc_version` substring match is imprecise | ✅ **已修复** |
| BUG-SM-005 | shared_modules/dependency_check.py | High | `check_codeformer` assumes basicsr implies CodeFormer | ✅ **已修复** |
| BUG-SM-006 | shared_modules/logger.py | High | Windows `msvcrt.locking` uses non-blocking `LK_NBLCK` | ✅ **已修复** |
| BUG-SM-007 | shared_modules/logger.py | High | Log rotation `rename` has multi-process race condition | ✅ **已修复**（timestamp-based 备份名避免竞争） |
| BUG-SM-008 | shared_modules/audio_utils.py | High | `arnndn` applied without converting to mono first | ✅ **已修复** |
| BUG-SM-009 | shared_modules/audio_utils.py | High | `arnndn` model not auto-selected by detected hum frequency | ✅ **已实现** |
| BUG-SM-010 | shared_modules/preview.py | High | Preview mode does not implement spec requirements (0.5s, WebP, comparison) | ✅ **大部分已实现** |
| BUG-SM-011 | shared_modules/manifest_loader.py | Medium | `_verify_photo_integrity` downgrades 16bit to 8bit | ✅ **已修复**（I;16 使用 `np.uint16` 保持精度） |
| BUG-SM-012 | shared_modules/manifest_generate.py | Medium | `audio_override` not merged into `recommended_params` | ✅ **已实现** |
| BUG-SM-013 | shared_modules/manifest_generate.py | Medium | `map_class_to_path_params` returns empty dict for photo classes | ✅ **已实现** |
| BUG-SM-014 | shared_modules/manifest_editor.py | Medium | Interactive TUI missing (only CLI subcommands) | ✅ **已实现** |
| BUG-SM-015 | shared_modules/checkpoint.py | Medium | `wsl_redirect_path` does not check `temp_dir` | ✅ **已修复** |
| BUG-SM-016 | shared_modules/checkpoint.py | Medium | `wsl_copy_back` cross-fs missing `.bak` backup | ✅ **已修复** |
| BUG-SM-017 | shared_modules/checkpoint.py | Low | `estimate_disk_space` method2 integer precision loss | ✅ **已修复** |
| BUG-SM-018 | shared_modules/dependency_check.py | Medium | `retinaface` vs `retina-face` package name mismatch | ✅ **已修复** |
| BUG-SM-019 | shared_modules/dependency_check.py | Medium | `check_ai_models` MD5 checksums all None | ✅ **已修复** |
| BUG-SM-020 | shared_modules/preview.py | Medium | `generate_preview_clip` no motion magnitude detection | ✅ **已修复** |
| BUG-SM-021 | shared_modules/preview.py | Medium | `generate_comparison_frames` video size mismatch | ✅ **已修复** |
| BUG-036 | video/stage6_grain_synthesis.py | Medium | `_run_grain_pipeline` 返回类型不一致（`0` vs tuple） | ✅ **已修复** |
| BUG-037 | video/config.yaml | Low | `audio_override` 缩进不一致导致 YAML 解析歧义 | ✅ **已修复** |
| BUG-038 | video/stage5_ai_face.py | Medium | 人脸片段提取硬编码 25fps | ✅ **已修复** |
| BUG-039 | video/stage5_ai_face.py | Low | 输出文件名仅使用 `job_id` 未嵌入 `param_hash` | ✅ **已修复** |
| BUG-040 | video/main.py | High | `_worker_wrapper` 传递 `argparse.Namespace` 导致 spawn 序列化失败 | ✅ **已修复** |
| BUG-041 | video/stage7_encode.py | Medium | `svtav1-params` 重复构建/追加 | ✅ **已修复** |
| BUG-014 | photo/main.py | Low | `force_traditional_first` 默认值逻辑错误（始终为 False） | ✅ **已修复** |
| BUG-023 | photo/stage6_encode.py | Medium | 中间文件缺失时静默回退到原始输入 | ✅ **已修复** |
| BUG-024 | photo/stage3_ai_upscale.py | Medium | `_resolve_input_source` 忽略 `ban_original`/`ban_intermediate` | ✅ **行为正确** |
| ISSUE-063 | shared_modules/checkpoint.py | Low | `gpu_memory_check` 文档未说明 `vram_required` 单位为 MiB | ✅ **已修复** |
| ISSUE-064 | shared_modules/dependency_check.py | Medium | `check_face_detection_packages` 未检查 `retina-face` 包 | ✅ **已修复** |
| ISSUE-069 | shared_modules/manifest_generate.py | Medium | 照片编码参数混入 `recommended_params` | ✅ **已修复** |

See `bugs.md` and `shixian.md` for the complete issue registers with fix ideas.

## License

MIT License


---

## Video Restoration Pipeline 审查更新 (2026-04-29 全面复审)

本次审查依据 v6.32_revised.md + v7.32_revised.md，对 `video_restoration_pipeline/` 全部 11 个文件 + `shared_modules/` 核心文件进行了纯手动代码复审。

### 审查结论

**视频流水线总体合规度约 85%**。核心工作流（单 worker 和 spawn 多 worker）均已可用，此前 Critical 缺陷（BUG-042 spawn 模式、BUG-043 resume 错误标记 DONE）已修复。大量历史问题已修复，文档已同步更新至最新状态。

### 已确认修复的历史问题（请勿重复报告）

| Bug ID | 原描述 | 状态 |
|--------|--------|------|
| BUG-VIDEO-001 | 缺少 `--mode/--scan/--edit-manifest/--resume` | ✅ **已修复** — main.py l.52-59 已实现六种模式和 resume |
| BUG-VIDEO-002 | `force_traditional_first` 未强制 | ✅ **已修复** — main.py l.420-432 已实施门控 |
| BUG-VIDEO-003 | WSL2 重定向仅在 stage7 | ✅ **已修复** — process_job 开头统一重定向 temp_dir+output_dir |
| BUG-VIDEO-004 | FAILED 降级逻辑缺失 | ✅ **已修复** — main.py l.617-643 已实现降级验证 |
| BUG-VIDEO-005 | 音频文件名不匹配 | ✅ **已修复** — stage0 统一输出 `.audio`，stage7 兼容旧命名 |
| BUG-VIDEO-007 | stage2 失败不阻止 AI | ✅ **已修复** — stage2 非 0/3 时返回 86 阻止后续阶段 |
| BUG-VIDEO-009 | stage3 硬编码 25fps | ✅ **已修复** — 通过 ffprobe 读取 r_frame_rate |
| BUG-VIDEO-010 | 缺少 Lanczos 二次缩放 | ✅ **已修复** — 已实现非整数倍目标 Lanczos 缩放 |
| BUG-VIDEO-012 | 缺少人脸稳定性检查 | ✅ **已修复** — 位移<5%、面积>15% 已检查 |
| BUG-VIDEO-014 | 缺少遮罩羽化 | ✅ **已修复** — Gaussian blur mask margin≥8px |
| BUG-VIDEO-015 | stage6 8bit 处理 | ✅ **已修复** — 使用 YUV420P16 + FFV1 |
| BUG-VIDEO-016 | stage6 有损 SVT-AV1 | ✅ **已修复** — 输出 FFV1 无损中间格式 |
| BUG-VIDEO-017 | 多声道未保留 | ✅ **已修复** — >2ch 自动切换 FLAC |
| BUG-019 | `audio_override` 位置错误 | ✅ **已修复** — 已移至 config 根级 |
| BUG-038 | stage5 硬编码 25fps | ✅ **已修复** — 读取实际帧率 |
| BUG-039 | stage5 未使用 param_hash | ✅ **已修复** — 文件名嵌入 param_hash |
| BUG-040 | spawn Namespace 序列化 | ✅ **已修复** — task_contexts 传入纯 dict |

### 本轮审查新增关键问题

| Bug ID | 位置 | 严重程度 | 描述 | 状态 |
|--------|------|----------|------|------|
| **BUG-042** | video/main.py | **Critical** | spawn 模式下 `args` 为 dict，但 `process_job()` 使用 `args.config` / `args.stage`（AttributeError），`workers>1` 时完全崩溃 | ✅ **已修复** |
| **BUG-043** | video/main.py | **Critical** | Resume 模式：PENDING 任务只要有任意 stepN 中间文件就标记 DONE，导致未完成的任务被跳过 | ✅ **已修复** |
| BUG-044 | video/main.py | Medium | FAILED 降级未定位失败 stage，从 stage 0 重新开始浪费计算 | ✅ **已修复** |
| BUG-045 | video/classify.py | Medium | `classify_video()` 未返回 `duration`，导致超时和磁盘估算使用 0 | ✅ **已修复** |
| BUG-046 | video/stage7 | Medium | 音频中间文件查找使用 glob 模式，可能匹配到旧任务残留文件 | ✅ **已修复** |
| BUG-047 | video/main.py | Medium | 缺少 `--ai-enabled` / `--ai-upscale-factor` / `--face-restore` CLI 参数 | ✅ **已修复** |
| ISSUE-V-S4-001 | video/stage4 | Medium | 全局 edgedetect+gblur+unsharp 未实现真正的局部遮罩区域分离处理 | ✅ **已修复**（工程折中） |
| ISSUE-V-S4-002 | video/stage4 | Medium | 未实现最终调色（lift/gamma/gain），仅使用简单 `eq` saturation/contrast | ✅ **已修复** |
| ISSUE-V-PREV-001 | shared/preview.py | Medium | VFR 素材使用 output seeking，时间位置不精确 | ✅ **已修复** |
| ISSUE-V-MGEN-001 | shared/manifest_generate.py | Medium | `classify_item()` 每个文件都重新 importlib 加载模块，大目录扫描极慢 | ✅ **已修复** |

完整的审查报告（每文件一个 audit 报告）和修复思路详见项目根目录：
- `video_main-audit.md`
- `video_config-audit.md`
- `video_classify-audit.md`
- `video_stage0_audio_extract-audit.md`
- `video_stage1_preprocess-audit.md`
- `video_stage2_traditional_check-audit.md`
- `video_stage3_ai_upscale-audit.md`
- `video_stage4_postprocess-audit.md`
- `video_stage5_ai_face-audit.md`
- `video_stage6_grain_synthesis-audit.md`
- `video_stage7_encode-audit.md`
- `shared_audio_utils-audit.md`
- `shared_checkpoint-audit.md`
- `shared_dependency_check-audit.md`
- `shared_logger-audit.md`
- `shared_manifest_generate-audit.md`
- `shared_preview-audit.md`

以及 `bugs.md`（含 BUG-042 ~ BUG-047）和 `shixian.md`（含 ISSUE-V-MAIN-001 ~ ISSUE-V-MGEN-003）。


---

## Shared Modules 审查更新 (2026-04-29 复审)

本次审查依据 v6.32_revised.md + v7.32_revised.md，对 `shared_modules/` 全部 9 个文件进行了纯手动代码复审。

### 审查结果概览

| 文件 | 状态 | 主要问题 |
|------|------|----------|
| `audio_utils.py` | ✅ 基本合规 | `sample_rate` 支持 `"auto"` 从实际音频读取；arnndn 路径单引号边缘情况 |
| `checkpoint.py` | ✅ 基本合规 | `wsl_redirect_path` 已检查 temp_dir；跨文件系统已备份旧文件；`estimate_disk_space` 精度已修复；`vram_required` 单位已明确（MiB） |
| `dependency_check.py` | ✅ 基本合规 | `retinaface` 包名提示已修正；MD5 校验已启用（支持外部清单）；spawn 合规检查范围基础实现 |
| `logger.py` | ✅ 基本合规 | 日志轮转理论竞争条件（实际影响极小） |
| `manifest_editor.py` | ✅ 基本合规 | TUI 参数展示过于简略 |
| `manifest_generate.py` | ✅ 基本合规 | 照片参数映射过于宽泛；audio_override 结构不一致 |
| `manifest_loader.py` | ✅ 基本合规 | 16bit 图像模式处理边缘情况 |
| `preview.py` | ✅ 基本合规 | 运动幅度检测（`detect_motion_segments`）、对比帧尺寸统一（`scale=640:-2`）、VFR 帧号精确提取均已实现 |

### 新增 Key Known Issues (shared_modules 复审)

| Bug ID | 位置 | 严重程度 | 描述 | 状态 |
|--------|------|----------|------|------|
| BUG-SHARED-001 | checkpoint.py | **Critical** | `wsl_copy_back()` 使用 `shutil.copy2`（应 copyfile），WSL2 NTFS 下可能 PermissionError | ✅ **已修复** |
| BUG-SHARED-002 | checkpoint.py | **Critical** | `wsl_redirect_path()` 对 `/mnt/` 路径不做重定向且方向相反，WSL2 I/O 瓶颈未解决 | ✅ **已修复** |
| BUG-SHARED-003 | checkpoint.py | **Critical** | `gpu_memory_check()` 使用总显存而非可用显存，其他进程占用时易 OOM | ✅ **已修复** |
| BUG-SHARED-016 | dependency_check.py | **Critical** | `check_ai_models()` 缺少 CodeFormer 权重检查 | ✅ **已实现** |
| BUG-SHARED-054 | manifest_editor.py | **Critical** | 交互式 TUI 完全缺失，仅命令行子命令 | ✅ **已实现** |
| BUG-SHARED-032 | audio_utils.py | High | arnndn 未先转 mono，某些 CPU 上 stereo 输入存在 bug | ✅ **已修复** |
| BUG-SHARED-033 | audio_utils.py | High | arnndn 模型未根据检测频率自动选择 50/60Hz | ✅ **已实现** |
| BUG-SHARED-026 | logger.py | High | Windows 下 `msvcrt.locking` 使用 `LK_NBLCK` 非阻塞锁，多进程可能失效 | ✅ **已修复** |
| BUG-SHARED-027 | logger.py | High | 日志轮转 `rename` 存在多进程竞争条件 | ✅ **已修复**（timestamp-based 备份名） |
| BUG-SHARED-040 | preview.py | High | 预览模式与规范差距极大（无 0.5s 片段、WebP、对比帧） | ✅ **大部分已实现** |
| BUG-SHARED-076 | audio_utils.py | Medium | `process_audio()` codec 默认 `"flac"` 与 spec `"aac"` 不一致 | ✅ **已修复**（默认 aac） |
| BUG-SHARED-077 | audio_utils.py | Medium | `noise_reduction` 配置键名与 spec `denoise` 不一致 | ✅ **已修复**（统一为 `denoise`） |
| BUG-SHARED-070 | checkpoint.py | Medium | `get_file_xxhash()` 使用 xxh3_64 而非 spec 要求的 xxhash64 | ✅ **已合规**（使用 `xxhash.xxh64()`） |
| BUG-SHARED-071 | checkpoint.py | Medium | 未实现 CUDA 路径分层探测 | ✅ **已实现** |
| BUG-SHARED-072 | checkpoint.py | Medium | 未实现 NVIDIA-SMI 显存解析跨平台兼容 | ✅ **已实现** |
| BUG-SM-015 | checkpoint.py | Medium | `wsl_redirect_path` 未检查 `temp_dir` | ✅ **已修复** |
| BUG-SM-016 | checkpoint.py | Medium | `wsl_copy_back` 跨文件系统未备份旧文件 | ✅ **已修复** |
| BUG-SM-017 | checkpoint.py | Low | `estimate_disk_space` method2 整数精度丢失 | ✅ **已修复** |
| BUG-SM-018 | dependency_check.py | Medium | `retinaface` vs `retina-face` 包名不匹配 | ✅ **已修复** |
| BUG-SM-019 | dependency_check.py | Medium | `check_ai_models` MD5 校验未启用 | ✅ **已修复** |
| BUG-SM-020 | preview.py | Medium | `generate_preview_clip` 未实现运动幅度检测 | ✅ **已修复** |
| BUG-SM-021 | preview.py | Medium | `generate_comparison_frames` 视频尺寸不匹配 | ✅ **已修复** |
| BUG-036 | video/stage6_grain_synthesis.py | Medium | `_run_grain_pipeline` 返回类型不一致 | ✅ **已修复** |
| BUG-037 | video/config.yaml | Low | `audio_override` 缩进不一致 | ✅ **已修复** |
| BUG-038 | video/stage5_ai_face.py | Medium | 人脸片段提取硬编码 25fps | ✅ **已修复** |
| BUG-039 | video/stage5_ai_face.py | Low | 输出文件名未嵌入 `param_hash` | ✅ **已修复** |
| BUG-040 | video/main.py | High | `_worker_wrapper` 传递 `argparse.Namespace` 导致 spawn 失败 | ✅ **已修复** |
| BUG-041 | video/stage7_encode.py | Medium | `svtav1-params` 重复构建 | ✅ **已修复** |
| BUG-014 | photo/main.py | Low | `force_traditional_first` 默认 False | ✅ **已修复** |
| BUG-023 | photo/stage6_encode.py | Medium | 中间文件缺失时静默回退 | ✅ **已修复** |
| BUG-024 | photo/stage3_ai_upscale.py | Medium | `ban_original`/`ban_intermediate` 参数未使用 | ✅ **行为正确** |
| ISSUE-063 | checkpoint.py | Low | `vram_required` 单位未说明 | ✅ **已修复** |
| ISSUE-064 | dependency_check.py | Medium | `retinaface` 包检查不完整 | ✅ **已修复** |
| ISSUE-069 | manifest_generate.py | Medium | 照片编码参数混入推荐参数 | ✅ **已修复** |

完整的审查报告（每文件一个 audit 报告）和修复思路详见：
- `audio_utils-audit.md`
- `checkpoint-audit.md`
- `dependency_check-audit.md`
- `logger-audit.md`
- `manifest_editor-audit.md`
- `manifest_generate-audit.md`
- `manifest_loader-audit.md`
- `preview-audit.md`

以及 `bugs.md` 和 `shixian.md`。

---

## Photo Restoration Pipeline 审查更新 (2026-04-29)

本次审查依据 v6.32_revised.md + v7.32_revised.md，对 `photo_restoration_pipeline/` 全部 9 个文件进行了纯手动代码审查。

### 照片流水线新增 Key Known Issues

| Bug ID | 位置 | 严重程度 | 描述 |
|--------|------|----------|------|
| ~~PHOTO-AUDIT-001~~ | photo/main.py | Medium | ~~缺少 manifest 工作流~~ | 📝 架构增强（`--input-dir` 已满足批量需求） |
| ~~PHOTO-AUDIT-002~~ | photo/main.py | Low | ~~`_check_disk_space()` 传入零尺寸~~ | ✅ **已修复** |
| PHOTO-AUDIT-003 | photo/main.py | Low | 无 per-stage 清理调用（仅在 job 开始时清理） |
| ~~PHOTO-AUDIT-004~~ | photo/classify.py | Low | ~~`native_dar` 输出浮点字符串~~ | ✅ **已修复** |
| ~~PHOTO-AUDIT-005~~ | photo/classify.py | Low | ~~`native_width/height` 未读取 EXIF 原始尺寸~~ | ✅ **已修复** |
| PHOTO-AUDIT-006 | photo/config.yaml | Low | 缺少 `model_dir` 配置选项；stage3 硬编码路径 |
| PHOTO-AUDIT-007 | photo/config.yaml | Low | `global.pix_fmt` 为视频专用参数，照片未使用 |
| ~~PHOTO-AUDIT-008~~ | photo/stage1_preprocess.py | Low | ~~通道校正使用均值/std 拉伸~~ | ✅ **已修复** |
| PHOTO-AUDIT-009 | photo/stage1_preprocess.py | Low | `extract_grain_pattern()` 使用 `img.mode` 决定 dtype，转换后可能不准确 |
| PHOTO-AUDIT-010 | photo/stage2_traditional_check.py | Low | SSIM 协方差 fallback 为近似算法，边缘情况可能与 skimage 精确值不同 |
| ~~PHOTO-AUDIT-011~~ | photo/stage3_ai_upscale.py | Medium | ~~DAT 模型支持不完整~~ | ✅ **已修复** |
| ~~PHOTO-AUDIT-012~~ | photo/stage3_ai_upscale.py | Low | ~~Python API fallback 单引号问题~~ | ✅ **已修复** |
| ~~PHOTO-AUDIT-013~~ | photo/stage3_ai_upscale.py | Low | ~~缺少 `model_dir` 配置~~ | ✅ **已修复** |
| ~~PHOTO-AUDIT-014~~ | photo/stage4_postprocess.py | Low | ~~颗粒合成在 stage4 执行~~ | ✅ **已修复** |
| PHOTO-AUDIT-015 | photo/stage4_postprocess.py | Low | `_add_grain_gmic()` strength×100 转换，G'MIC film_grain 参数范围待确认 |
| PHOTO-AUDIT-016 | photo/stage5_ai_face.py | Low | 颈部 ROI 为简化矩形，未考虑头部倾斜或肩部位置 |
| ~~PHOTO-AUDIT-017~~ | photo/stage5_ai_face.py | Low | ~~`face_restore` 为空时 stage5 仍执行~~ | ✅ **已修复**（代码已有提前返回） |
| PHOTO-AUDIT-018 | photo/stage6_encode.py | Low | PNG fallback 丢失 10bit/16bit 深度和 CICP 元数据 |
| ~~PHOTO-AUDIT-019~~ | photo/stage6_encode.py | Low | ~~avifenc 版本正则~~ | ✅ **已修复** |

### 已确认修复的历史问题

| Bug ID | 描述 | 状态 |
|--------|------|------|
| BUG-PHOTO-001 ~ BUG-PHOTO-013 | main.py 参数注入、CLI 映射、重试间隔、预览、stage_idx、1080p 检查 | ✅ 已修复 |
| BUG-PHOTO-048 | stage5 人脸检测器置信度阈值 | ✅ 已修复 |
| BUG-PHOTO-050 | CodeFormer Python fallback 真实推理 | ✅ 已修复 |
| Bug 10.6 ~ 10.11 | stage1 通道校正（Reinhard、IEC 61966-2-1、float32、四舍五入、阈值配置） | ✅ 已修复 |
| Bug 11.4 ~ 11.5 | stage3 目标分辨率检查、输入源验证 | ✅ 已修复 |
| Bug 11.12 ~ 11.13 | stage5 置信度阈值、CodeFormer fallback | ✅ 已修复 |
| Bug 11.14 ~ 11.25 | stage2 校验、stage4 局部遮罩、stage6 颗粒/16bit/FLIF/编码 | ✅ 已修复 |
| PHOTO-AUDIT-002, 004~005 | classify.py/native_dar/EXIF 尺寸、main.py 磁盘估算 | ✅ 已修复 |
| PHOTO-AUDIT-008 | stage1 通道校正改为独立去噪/色阶校正 | ✅ 已修复 |
| PHOTO-AUDIT-011~014 | stage3 DAT 支持、引号安全、model_dir、颗粒合成顺序 | ✅ 已修复 |
| PHOTO-AUDIT-019 | stage6 avifenc 版本正则增强 | ✅ 已修复 |
| P-1 ~ P-68 | shixian.md 照片流水线实现差距 | ✅ 已修复 |
| BUG-014 ~ BUG-041 | 本轮修复：spawn 安全、帧率、参数哈希、编码参数、回退逻辑等 | ✅ 已修复 |

完整的审查报告（每文件一个 audit 报告）详见：
- `photo_main-audit.md`
- `photo_classify-audit.md`
- `photo_config-audit.md`
- `photo_stage1-audit.md`
- `photo_stage2-audit.md`
- `photo_stage3-audit.md`
- `photo_stage4-audit.md`
- `photo_stage5-audit.md`
- `photo_stage6-audit.md`

以及 `bugs.md`（§13）和 `shixian.md`（§十三、§十四）。

### Shared Modules 第二轮复审（2026-04-29，纯手动逐文件审查）

本次审查对 `shared_modules/` 全部 9 个文件进行了纯手动逐条规范核对，依据 v6.32_revised.md + v7.3.md。

| Bug ID | 位置 | 严重程度 | 描述 | 状态 |
|--------|------|----------|------|------|
| BUG-058 | checkpoint.py | 🟡 Medium | `cleanup_old_intermediates()` 默认 `current_step=0` 导致清理永不执行 | ✅ **已修复** |
| BUG-059 | audio_utils.py | 🟢 Low | 多声道检测未覆盖 `5.1(side)` 等变体，可能导致声道丢失 | ✅ **已修复** |
| BUG-060 | manifest_loader.py | 🟢 Low | `_handle_failed_job` 假设 `shared_modules` 父目录为项目根，结构变化时 ImportError | ✅ **已修复** |
| BUG-061 | preview.py | 🟢 Low | `detect_motion_segments` 在输入目录（可能只读）创建临时目录 | ✅ **已修复** |
| BUG-062 | dependency_check.py | 🟢 Low | `check_gmic_filters` 子串匹配可能误报 | ✅ **已修复** |
| ISSUE-SHARED-CK-001 | checkpoint.py | Medium | `verify_frame_integrity` 默认 `bit_depth=8`，16bit 帧可能误判损坏 | ✅ **已修复** |
| ISSUE-SHARED-DC-001 | dependency_check.py | Medium | 未检查 avifenc 版本和 `--jobs` 支持（需 libavif 0.9.2+） | ✅ **已修复** |
| ISSUE-SHARED-DC-002 | dependency_check.py | Medium | spawn 合规检查未覆盖照片流水线 | ✅ **已修复** |
| ISSUE-SHARED-AU-001 | audio_utils.py | Low | `process_audio` 未前置检查音频流存在性，无音频流素材产生 ERROR | ✅ **已修复** |
| ISSUE-SHARED-ML-001 | manifest_loader.py | Medium | 照片完整性校验未实现多位置抽查（仅 PIL 加载检查） | ✅ **已修复** |
| ISSUE-SHARED-PV-001 | preview.py | Low | `generate_preview_clip` 两阶段 ffmpeg 效率低，可优化为单阶段 | ✅ **已修复** |

完整的审查报告详见 `bugs.md`（§"新增 Bug（2026-04-29 shared_modules 复审第二轮）"）和 `shixian.md`（§"shared_modules 复审第二轮 — 纯手动逐文件规范审查"）。
