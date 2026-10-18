# Video & Photo Restoration Pipeline

Hybrid restoration pipeline combining traditional signal processing with AI
super-resolution for vintage video and photo restoration.

## Overview

This project provides two restoration pipelines:

- **Video Restoration Pipeline** — analog tape, DV digital, HDV/AVCHD materials
- **Photo Restoration Pipeline** — scanned prints and digital photos

Both pipelines follow a multi-stage architecture that preserves original
quality while applying modern restoration techniques.

## Quick Start

### Prerequisites

- Python 3.9+（实测运行环境为 3.12）
- FFmpeg 4.4+（`bandreject` 需支持 `width_type=h`）
- ExifTool
- G'MIC（照片链）
- VapourSynth + havsfunc r33（视频链）
- avifenc（照片 AVIF 输出）

### Installation

1. Clone the repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install AI model weights to `./models/`:
   - `RealESRGAN_x2plus.pth` / `RealESRGAN_x4plus.pth`
   - `realesr-general-x4v3.pth`
   - `GFPGANv1.4.pth` / `codeformer.pth`
   - `DAT_x2.pth` / `DAT_x4.pth` / `DAT_2_*.pth`
   - `detection_Resnet50_Final.pth`（人脸检测）
4. Run the dependency check:
   ```bash
   python shared_modules/dependency_check.py -c video_restoration_pipeline/config.yaml --pipeline-type video
   python shared_modules/dependency_check.py -c photo_restoration_pipeline/config.yaml --pipeline-type photo
   ```

### Usage

#### Video Restoration

视频修复支持 **6 种运行模式**（`--mode` 的 6 个取值：
`scan` / `edit` / `batch` / `quick` / `interactive` / `preview`）：

```bash
# 模式 D（四步工作流）Step 1: 扫描输入目录生成分类清单
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    --scan -i /path/to/videos/ --output-manifest scan_manifest.yaml

# 模式 D Step 2: 交互式审阅清单（调整参数覆盖）
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    --edit-manifest --manifest scan_manifest.yaml

# 模式 D Step 4 / 模式 F（批量执行，支持断点续传）
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    --manifest scan_manifest.yaml --mode batch --resume

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

**AI 选项**（最高优先级，覆盖 config）：

```bash
--ai-enabled                          # 启用 AI 超分
--ai-upscale-factor {2,4}             # 放大倍数
--ai-model {realesr-general-x4v3, RealESRGAN_x2plus, RealESRGAN_x4plus,
            DAT_x2, DAT_x4, DAT_2_x2, DAT_2_x4}
--face-restore {GFPGAN,CodeFormer}    # 人脸修复模型
```

#### Photo Restoration

```bash
python photo_restoration_pipeline/main.py \
    -c photo_restoration_pipeline/config.yaml \
    -i input_photo.jpg \
    -o output.avif
```

照片修复默认输出 AVIF（10bit YUV420 SDR）。如需 16bit RGB444 无损存档，
在 `config.yaml` 中设置 `archival_mode: true`。

## 运行环境要点

两条链的解释器**不通用**，混用会直接失败：

| 链 | 解释器 | 原因 |
|---|---|---|
| 视频链编排 | `tools_r4/vs_portable/r72/python.exe` | VapourSynth R72（API3 插件全通）；不含 torch |
| 视频链 AI 推理 | 自动解析到 `.venv_r4` | stage3/stage5 以子进程调用；解析顺序 `WB_AI_PYTHON` → 当前解释器 → `.venv_r4`（`shared_modules/ai_runtime.py`） |
| 照片链 | `.venv_r4/Scripts/python.exe` | torch / basicsr / realesrgan / gfpgan / facexlib 齐全 |

- **PATH 必须包含子目录**：`tools_r4/gmic/gmic-*-cli-win64`、
  `tools_r4/exiftool/exiftool-*`（只加 `tools_r4` 会 `[WinError 2]`）。
- Windows/Git Bash 下 PATH 前置项要用 POSIX 形式（`/e/...`），
  写 `E:/...` 会被 MSYS 拆坏成 `E` + `/codex/...`。
- 退出码语义（按代码实际使用统计）：`80` 依赖缺失 / `81` 参数错误 /
  `82` 输入·IO·阶段失败 / `83` GPU / `84` 编码·IO / `85` 权限（保留未用）/
  `86` 校验·门禁失败 / `87` 资源不足·超时 / `88` 取消（仅编排层）。
  `3` 表示跳过（已存在、无需处理）。

### 跑批注意（重要）

- `--mode quick` **每次重新生成临时清单**：分类置信度 < 0.80 的素材会被
  标 SKIP，表现为 "executing 0 jobs" 且退出码 0（假成功）。
- 需要人工确认的素材走 `--scan` → `--edit-manifest` → `--mode batch`：
  把 `status` 置 `PENDING`，必要时用 `user_overrides` 调整门限
  （如 `deinterlace_progressive_threshold`）。
- 中途失败的任务会按 `last_completed_stage + 1` 自动续跑，不重复 AI 推理。

## Architecture

- [architecture.md](architecture.md) — 系统架构说明
- [PROJECT_MANUAL.md](PROJECT_MANUAL.md) — 操作手册

两份文档为**描述性材料**，如与代码不一致，以代码为准（见「版本与规范说明」）。

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
│       ├── stage2_ai_pre.py          # 单通道校正
│       ├── stage3_ai_upscale.py
│       ├── stage4_ai_face.py
│       ├── stage5_postprocess.py
│       ├── stage6_encode.py
│       └── stage2_traditional_check.py   # 历史遗留，不在默认链
│
├── shared_modules/                  # Shared modules
│   ├── checkpoint.py                # Checkpoint / resume support
│   ├── dependency_check.py          # Dependency validation
│   ├── logger.py                    # Structured JSON Lines logging
│   ├── audio_utils.py               # Audio analysis utilities
│   ├── manifest_generate.py         # Manifest generation
│   ├── manifest_editor.py           # Manifest editing
│   ├── manifest_loader.py           # Manifest loading
│   ├── preview.py                   # Preview utilities
│   ├── media_io.py                  # Media probing / IO helpers
│   ├── cancel_guard.py              # Cancellation / interrupt guard
│   ├── ai_runtime.py                # AI interpreter resolution
│   └── dat_runtime.py               # DAT model runtime
│
└── models/                          # AI model weights directory
```

## 版本与规范说明

**本仓库以代码为唯一事实来源**（`main.py` / `stages/` / `config.yaml` /
`shared_modules/`）。历史规范文档仅作背景参考，且**可能已落后于实现**：

- `v6.32.md` — 视频修复技术规范（历史版本）
- `v7.32.md` — 架构规范（照片链、清单编辑器、运行模式）

> 规范与代码不一致时，以代码行为为准。本 README 中的命令、参数、
> 退出码、目录结构均按当前代码实测核对；规范文档如需继续作为依据，
> 应先按代码修订。规范原文为本地文档，不入库。

## 问题台账与审计报告（仅本地，不入库）

缺陷清单、逐文件审计报告、修复台账与验收记录属于内部工作资料，
**不随仓库分发**，保存在项目工作目录本地：

- `bugs.md` — 缺陷登记册（含 BUG-xxx / ISSUE-xxx）
- `shixian.md` — 实现差距登记
- `修复清单.md` — 按轮次的修复台账与根因分析
- `MEDIA_ACCEPTANCE_PLAN.md` — 验收门与实测记录
- `*-audit.md` — 逐文件审查报告

如需查阅或提交问题，请直接在本地工作目录中查看上述文件。
