# Video & Photo Restoration Pipeline

[English](#english) &nbsp;|&nbsp; [简体中文](#chinese)

---

<a id="english"></a>

## English

Hybrid restoration pipeline combining traditional signal processing with AI
super-resolution for vintage video and photo restoration.

### Overview

This project provides two restoration pipelines:

- **Video Restoration Pipeline** — analog tape, DV digital, HDV/AVCHD materials
- **Photo Restoration Pipeline** — scanned prints and digital photos

Both pipelines follow a multi-stage architecture that preserves original
quality while applying modern restoration techniques.

### Quick Start

#### Prerequisites

- Python 3.9+ (tested on 3.12)
- FFmpeg 4.4+ (`bandreject` must support `width_type=h`)
- ExifTool
- G'MIC (photo pipeline)
- VapourSynth + havsfunc r33 (video pipeline)
- avifenc (photo AVIF output)

#### Installation

1. Clone the repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Install AI model weights into `./models/`:
   - `RealESRGAN_x2plus.pth` / `RealESRGAN_x4plus.pth`
   - `realesr-general-x4v3.pth`
   - `GFPGANv1.4.pth` / `codeformer.pth`
   - `DAT_x2.pth` / `DAT_x4.pth` / `DAT_2_*.pth`
   - `detection_Resnet50_Final.pth` (face detection)
4. Run the dependency check:
   ```bash
   python shared_modules/dependency_check.py -c video_restoration_pipeline/config.yaml --pipeline-type video
   python shared_modules/dependency_check.py -c photo_restoration_pipeline/config.yaml --pipeline-type photo
   ```

#### Usage

##### Video Restoration

The video pipeline supports **6 run modes** (the 6 values of `--mode`:
`scan` / `edit` / `batch` / `quick` / `interactive` / `preview`):

```bash
# Step 1: scan input directory and generate a classified manifest
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    --scan -i /path/to/videos/ --output-manifest scan_manifest.yaml

# Step 2: review the manifest (adjust status / parameter overrides)
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    --edit-manifest --manifest scan_manifest.yaml

# Step 3: batch execution (--resume skips finished jobs, resumes from checkpoint)
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    --manifest scan_manifest.yaml --mode batch --resume

# Quick mode: scan then execute immediately (temporary manifest, deleted on exit)
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    --mode quick -i /path/to/videos/

# Interactive single file
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    --mode interactive -i /path/to/single_video.avi

# Preview mode: extract representative frames / motion clips (no restoration)
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    --mode preview -i /path/to/videos/

# Single-stage debugging
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    --stage 3 -i /path/to/video.avi
```

**AI options** (highest priority, override config):

```bash
--ai-enabled                          # enable AI upscaling
--ai-upscale-factor {2,4}             # upscale factor
--ai-model {realesr-general-x4v3, RealESRGAN_x2plus, RealESRGAN_x4plus,
            DAT_x2, DAT_x4, DAT_2_x2, DAT_2_x4}
--face-restore {GFPGAN,CodeFormer}    # face restoration model
```

##### Photo Restoration

```bash
python photo_restoration_pipeline/main.py \
    -c photo_restoration_pipeline/config.yaml \
    -i input_photo.jpg \
    -o output.avif
```

Photo output defaults to AVIF (10-bit YUV420 SDR). For 16-bit RGB444 lossless
archival output, set `archival_mode: true` in `config.yaml`.

### Runtime Notes

The two pipelines use **different interpreters**; mixing them fails:

| Pipeline | Interpreter | Why |
|---|---|---|
| Video orchestration | `tools_r4/vs_portable/r72/python.exe` | VapourSynth R72 (all API3 plugins work); no torch |
| Video AI inference | auto-resolved to `.venv_r4` | stage3/stage5 spawn a subprocess; resolution order `WB_AI_PYTHON` → current interpreter → `.venv_r4` (`shared_modules/ai_runtime.py`) |
| Photo pipeline | `.venv_r4/Scripts/python.exe` | torch / basicsr / realesrgan / gfpgan / facexlib complete |

- **PATH must include the sub-directories** `tools_r4/gmic/gmic-*-cli-win64`
  and `tools_r4/exiftool/exiftool-*` (adding only `tools_r4` causes `[WinError 2]`).
- On Windows/Git Bash, use POSIX-style paths (`/e/...`) in PATH; `E:/...` gets
  split by MSYS into `E` + `/codex/...`.
- Exit codes (as actually used in code): `80` missing dependency / `81` bad
  argument / `82` input·IO·stage failure / `83` GPU / `84` encode·IO /
  `85` permission (reserved, unused) / `86` validation·gate failure /
  `87` resource exhausted·timeout / `88` cancelled (orchestrator only).
  `3` means skipped (already exists, nothing to do).

#### Batch Notes (important)

- `--mode quick` **regenerates a temporary manifest** on every run: materials
  with classification confidence < 0.80 are marked SKIP, which shows up as
  "executing 0 jobs" with exit code 0 (a false success).
- For materials that need human review use `--scan` → `--edit-manifest` →
  `--mode batch`: set `status` to `PENDING` and adjust thresholds via
  `user_overrides` when needed (e.g. `deinterlace_progressive_threshold`).
  The **photo** chain has a second gate: the job's own `confidence` must also
  be raised, otherwise `_run_single_job` skips it silently (returns 3).
- Failed jobs resume automatically from `last_completed_stage + 1`, so AI
  inference is not repeated.
- **Long clips (>15 min) take hours, not minutes.** A 16-minute 1080p clip
  needs roughly 4–5 h end to end; stage7's AV1 encode alone is ~3 h.
  Stall detection is a **no-progress watchdog**, not a time estimate:
  a stage is killed (exit `87`, log `STAGE_STALLED`) only when its output stops
  growing for `stage_inactivity_timeout` seconds (config, default `1800`).
  This is independent of resolution, machine speed and picture complexity.
  **The threshold must exceed the longest known silent interval** — at `600`
  it killed healthy stage7 encodes, because stage7 writes to `output/` as
  `{stem}.attempt-{pid}-{ts}.mkv` and a prefix built from the basename
  (*with* extension) never matched it, so the marker froze at the temp
  directory size. The prefix must be the **stem** (extension stripped).

#### Verifying output content (do not skip this)

An exit code of 0 does **not** mean the picture survived the pipeline. The
photo chain once delivered **all-black AVIFs** (386 bytes, pixel mean 0.0)
while every stage returned 0 and nothing was logged. Always inspect the
result itself:

| Check | How | Catches |
|---|---|---|
| Not a black frame | pixel `mean` / `std` of the decoded output | all-black output |
| Model/scale applied | log the vpy / `PARAM_OVERRIDE` | override silently ignored |
| Non-zero counters | blended frame count, faces detected | "0 frames" false pass |
| Artifact exists | glob + extension | naming mismatch |
| Intermediate TIFF readable | `tifffile` opens it directly | LZW (needs `imagecodecs`) |
| Third-party calls | `inspect.signature().bind()` | invalid kwargs |

**Never rely on rc + dimensions + bit depth alone** — that combination
passes an all-black output.

#### AI / face restore notes

- Face detection uses **facexlib**; the `RetinaFace` / `YOLOv8-face` / `MTCNN`
  packages named in the spec are not installed, so that fallback chain never
  matched anything. facexlib is not a peer backend — it is the detector
  bundled inside GFPGAN/CodeFormer.
- CodeFormer inference uses `basicsr.archs.codeformer_arch` (the `codeformer`
  package is absent, but the weights load with 0 missing / 0 unexpected keys).
- **G'MIC's `-o X.tif,uint16` writes a file that claims 16-bit but holds
  8-bit-range data.** Read such TIFFs with
  `media_io.load_image_rgb_uint8()` (tifffile + per-channel adaptive range);
  `PIL.Image.open(...).convert('RGB')` returns zeros.
- **Intermediate TIFFs must not be LZW.** Both `PIL.Image.save()` and
  `cv2.imwrite()` default to LZW for `.tif`, and reading LZW needs
  `imagecodecs` (not installed) — write with `media_io.write_tiff()`
  (zlib) or pass `[cv2.IMWRITE_TIFF_COMPRESSION, 8]`. Output **format must
  follow the output extension**: the video chain feeds `.png` names.
- Face restore works on real samples: on a frame extracted from
  `seg_face_static.mp4`, GFPGAN yields 2560×1920 and CodeFormer 1280×960.
  Note that facexlib detects **zero** faces across the 37 real photos.

### Architecture

- [architecture.md](architecture.md) — system architecture
- [PROJECT_MANUAL.md](PROJECT_MANUAL.md) — operation manual

These are **descriptive** documents; when they disagree with the code, the code
wins (see "Version & Specifications" below).

### Directory Structure

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
│       ├── stage2_ai_pre.py          # single-channel correction
│       ├── stage3_ai_upscale.py
│       ├── stage4_ai_face.py
│       ├── stage5_postprocess.py
│       └── stage6_encode.py
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
│   ├── dat_runtime.py               # DAT model runtime
│   ├── color_probe.py               # Colour-fidelity probe (stage6/stage7 shared)
│   ├── artifact_verify.py           # Output verification (frame count / order)
│   ├── exec_fingerprint.py          # Execution fingerprint for identity checks
│   ├── publish_journal.py           # Publish transaction journal (crash recovery)
│   ├── publish_lock.py              # Per-target publish lock
│   └── gpen/                        # GPEN architecture (upstream yangxy/GPEN)
│       └── face_model/              #   FullGenerator + op (pure-PyTorch fallback)
│
└── models/                          # AI model weights directory
```

### Colour Fidelity Guard

Both pipelines compare the **RGB channel means of the original input vs the
encoded output** before publishing, and record the result. Threshold is
`color_drift_threshold` (default `60.0`, in 0-255 units); set
`color_drift_enforce: true` to fail instead of only warning.

The default 60 is deliberately loose — it catches *whole-image collapse*
(black/white/lost content, which measures ~116 on a black frame) while letting
legitimate wide-gamut conversion through (~27 for ProPhoto→sRGB). Fine-grained
colour drift needs a per-material calibrated value; setting it too tight
produces frequent false positives and the log stops being read.

Photo pipeline config lives under `photo:`; video pipeline under `global:`.

### Artifact Verification

Before publishing, stage 7 runs **three independent gates**, each with its own
force-publish override:

| Gate | Compares | Override |
|---|---|---|
| A/V sync | inside the output file | `allow_sync_fail_publish` |
| Full-chain | **original input vs output** | `allow_verify_fail_publish` |
| Mechanical | step 6 artifact vs output | `allow_verify_fail_publish` |

A/V drift is judged **relative to the source** (`|output drift − source drift|`)
rather than against a fixed threshold — measured on 204 real phone clips,
a fixed threshold misjudged **31%** of them (the drift is inherent to old
sources), while the relative criterion dropped that to **1.5%**.

`full_chain_verify_mode` selects how strict the full-chain gate is:
`frames_only` (default, lowest false positives) / `relative` / `absolute`.
Results always land in `<output>.fullchain.json`; use
`shared_modules/fullchain_report.py <dir> --csv out.csv` to rank drift across
a batch.

Low-confidence classifications (`needs_confirmation`, <0.80) are skipped in
non-interactive batches. Confirm them with:

```bash
python shared_modules/manifest_editor.py <manifest> list --low-confidence
python shared_modules/manifest_editor.py <manifest> confirm --all-low-confidence
```

`stage_hard_timeout_hours` set to `0` disables the hard timeout entirely;
`stage_inactivity_timeout` set to `0` disables the inactivity watchdog.

### Version & Specifications

**The code is the single source of truth** (`main.py` / `stages/` /
`config.yaml` / `shared_modules/`). Historical specification documents are
background references only and **may lag behind the implementation**:

- `v6.32.md` — video restoration technical specification (historical)
- `v7.32.md` — architecture specification (photo pipeline, manifest editor, run modes)

> When a specification disagrees with the code, the code wins. Every command,
> flag, exit code and directory listing in this README was checked against the
> current code. Specification files are local-only and not tracked.

### Issue Registers (local only)

Defect logs, per-file audit reports, fix ledgers and acceptance records are
internal working documents and are **not distributed with the repository**:
`bugs.md`, `shixian.md`, `修复清单.md`, `MEDIA_ACCEPTANCE_PLAN.md`, `*-audit.md`.

---

<a id="chinese"></a>

## 简体中文

结合传统信号处理与 AI 超分的混合修复流水线，用于老视频与照片修复。

### 概览

本项目包含两条修复流水线：

- **视频修复流水线** — 模拟磁带、DV 数码带、HDV/AVCHD 素材
- **照片修复流水线** — 扫描件与数码照片

两条链都采用多阶段架构，在应用现代修复技术的同时尽量保留原始质量。

### 快速开始

#### 环境要求

- Python 3.9+（实测运行环境为 3.12）
- FFmpeg 4.4+（`bandreject` 需支持 `width_type=h`）
- ExifTool
- G'MIC（照片链）
- VapourSynth + havsfunc r33（视频链）
- avifenc（照片 AVIF 输出）

#### 安装

1. 克隆仓库
2. 安装 Python 依赖：
   ```bash
   pip install -r requirements.txt
   ```
3. 将 AI 模型权重放入 `./models/`：
   - `RealESRGAN_x2plus.pth` / `RealESRGAN_x4plus.pth`
   - `realesr-general-x4v3.pth`
   - `GFPGANv1.4.pth` / `codeformer.pth`
   - `DAT_x2.pth` / `DAT_x4.pth` / `DAT_2_*.pth`
   - `detection_Resnet50_Final.pth`（人脸检测）
4. 运行依赖检查：
   ```bash
   python shared_modules/dependency_check.py -c video_restoration_pipeline/config.yaml --pipeline-type video
   python shared_modules/dependency_check.py -c photo_restoration_pipeline/config.yaml --pipeline-type photo
   ```

#### 用法

##### 视频修复

视频修复支持 **6 种运行模式**（`--mode` 的 6 个取值：
`scan` / `edit` / `batch` / `quick` / `interactive` / `preview`）：

```bash
# Step 1：扫描输入目录，生成分类清单
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    --scan -i /path/to/videos/ --output-manifest scan_manifest.yaml

# Step 2：审阅清单（调整 status / 参数覆盖）
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    --edit-manifest --manifest scan_manifest.yaml

# Step 3：批量执行（--resume 跳过已完成任务，断点续传）
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    --manifest scan_manifest.yaml --mode batch --resume

# 快速模式：扫描后立即执行（清单为临时文件，退出即删）
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    --mode quick -i /path/to/videos/

# 交互式单文件
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    --mode interactive -i /path/to/single_video.avi

# 预览模式：提取代表性帧与运动片段（不执行修复）
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    --mode preview -i /path/to/videos/

# 单 stage 调试
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

##### 照片修复

```bash
python photo_restoration_pipeline/main.py \
    -c photo_restoration_pipeline/config.yaml \
    -i input_photo.jpg \
    -o output.avif
```

照片修复默认输出 AVIF（10bit YUV420 SDR）。如需 16bit RGB444 无损存档，
在 `config.yaml` 中设置 `archival_mode: true`。

### 运行环境要点

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

#### 跑批注意（重要）

- `--mode quick` **每次重新生成临时清单**：分类置信度 < 0.80 的素材会被
  标 SKIP，表现为 "executing 0 jobs" 且退出码 0（假成功）。
- 需要人工确认的素材走 `--scan` → `--edit-manifest` → `--mode batch`：
  把 `status` 置 `PENDING`，必要时用 `user_overrides` 调整门限
  （如 `deinterlace_progressive_threshold`）。
  **照片链还有第二道门限**：job 自身的 `confidence` 也必须一并调高，
  否则 `_run_single_job` 会静默跳过（返回 3）。
- 中途失败的任务会按 `last_completed_stage + 1` 自动续跑，不重复 AI 推理。
- **长片（>15 分钟）以小时计，不是分钟。** 16 分钟 1080p 全链约 4~5 小时，
  其中 stage7 的 AV1 编码单步就约 3 小时。判卡死用的是**无进展看门狗**
  而非时长估算：某 stage 的产出连续 `stage_inactivity_timeout` 秒不再增长
  才杀（config 可调，默认 `1800`），与分辨率、机器性能、画面复杂度无关。
  **阈值必须大于「最长的已知静默区间」** —— 设过 `600`，结果误杀了健康的
  stage7 编码：stage7 把成品写到 `output/` 下的
  `{stem}.attempt-{pid}-{ts}.mkv`，而用**含扩展名**的 basename 做前缀永远
  匹配不上它，标记于是冻结在 temp 目录大小。前缀必须用 **stem**（去扩展名）。

#### 必须核对产物内容（不要跳过）

**退出码 0 不代表画面活着。** 照片链曾交付**全黑 AVIF**（386 字节、
像素 mean 0.0），而全链 rc 一直是 0、没有任何日志报错。请务必检查成品本身：

| 核对对象 | 手段 | 能抓到什么 |
|---|---|---|
| 是否黑图 | 解码后看像素 `mean` / `std` | 全黑成品 |
| 模型/倍数是否真生效 | 查 vpy / `PARAM_OVERRIDE` | 覆盖被静默忽略 |
| 关键计数是否非零 | blend 帧数、检出人脸数 | "0 帧"假通过 |
| 产物是否存在 | glob 命中数 + 扩展名 | 命名约定不匹配 |
| 中间 TIFF 能否读 | `tifffile` 能直接打开 | LZW（需 `imagecodecs`） |
| 第三方调用 | `inspect.signature().bind()` | 非法 kwarg |

**只核对 rc + 尺寸 + 位深，会让全黑图"通过"验收。**

#### AI 与人脸修复要点

- 人脸检测用 **facexlib**。规范里那套 `RetinaFace → YOLOv8-face → MTCNN`
  回退链在本项目**三个包都没装**，恒为"全不可用"→ 检出零命中。
  facexlib 不是并列的第四个后端，而是 **GFPGAN/CodeFormer 自带的检测实现**。
- CodeFormer 推理走 `basicsr.archs.codeformer_arch`（`codeformer` 包不存在，
  但权重可零 missing / 零 unexpected 载入）。
- **G'MIC 的 `-o X.tif,uint16` 写出的文件声明 16-bit，实际却是 8-bit 量级。**
  这类 TIFF 必须用 `media_io.load_image_rgb_uint8()` 读（tifffile +
  逐通道值域自适应）；`PIL.Image.open(...).convert('RGB')` 只会读到全 0。
- **中间 TIFF 一律不得是 LZW。** `PIL.Image.save()` 与 `cv2.imwrite()`
  对 `.tif` 都默认选 LZW，而解 LZW 需要未安装的 `imagecodecs`——
  请用 `media_io.write_tiff()`（zlib），或给 cv2 传
  `[cv2.IMWRITE_TIFF_COMPRESSION, 8]`。**输出格式必须跟随输出扩展名**：
  视频链喂的是 `.png` 文件名。
- 人脸修复在真实样张上可用：从 `seg_face_static.mp4` 抽帧，
  GFPGAN 出 2560×1920、CodeFormer 出 1280×960。
  注意 facexlib 在**37 张真实照片上是零检出**。

### 架构

- [architecture.md](architecture.md) — 系统架构说明
- [PROJECT_MANUAL.md](PROJECT_MANUAL.md) — 操作手册

两份文档为**描述性材料**，如与代码不一致，以代码为准（见下方"版本与规范"）。

### 目录结构

```
├── video_restoration_pipeline/     # 视频修复流水线
│   ├── classify.py                  # 视频素材分类器
│   ├── main.py                      # 视频主控脚本
│   ├── config.yaml                  # 视频配置
│   └── stages/                      # 视频处理阶段
│       ├── stage0_audio_extract.py   # 音频提取
│       ├── stage1_preprocess.py      # 预处理（IVTC / QTGMC / 去噪）
│       ├── stage2_traditional_check.py  # 传统修复校验
│       ├── stage3_ai_upscale.py      # AI 超分（2×/4×）
│       ├── stage4_postprocess.py     # 选择性后处理（皮肤/头发保护）
│       ├── stage5_ai_face.py         # 人脸修复（GFPGAN / CodeFormer）
│       ├── stage6_grain_synthesis.py # 颗粒合成
│       └── stage7_encode.py          # 最终编码（AV1 + 音频）
│
├── photo_restoration_pipeline/     # 照片修复流水线
│   ├── classify.py                  # 照片分类器
│   ├── main.py                      # 照片主控脚本
│   ├── config.yaml                  # 照片配置
│   └── stages/                      # 照片处理阶段（默认链 6 步）
│       ├── stage1_preprocess.py      # 预处理（色彩空间/去划痕/去块/去色带）
│       ├── stage2_ai_pre.py          # AI 前处理：单通道校正
│       ├── stage3_ai_upscale.py      # AI 放大
│       ├── stage4_ai_face.py         # 人脸修复
│       ├── stage5_postprocess.py     # 后处理（局部遮罩 + 颗粒合成）
│       └── stage6_encode.py          # AVIF 最终编码
│
├── shared_modules/                  # 共享模块
│   ├── checkpoint.py                # 断点续传 / 参数哈希 / 身份校验
│   ├── dependency_check.py          # 依赖检查
│   ├── logger.py                    # 结构化 JSON Lines 日志
│   ├── audio_utils.py               # 音频分析
│   ├── manifest_generate.py         # 清单生成
│   ├── manifest_editor.py           # 清单编辑（含交互式 TUI）
│   ├── manifest_loader.py           # 清单加载
│   ├── preview.py                   # 预览工具
│   ├── media_io.py                  # 媒体探测 / IO 辅助
│   ├── cancel_guard.py              # 取消 / 中断守卫（rc=88）
│   ├── ai_runtime.py                # AI 子进程解释器解析
│   ├── dat_runtime.py               # DAT 模型运行时
│   ├── color_probe.py               # 色彩保真探测（stage6/stage7 共用）
│   ├── artifact_verify.py           # 产物核对（帧数 / 帧序 / 非黑图）
│   ├── exec_fingerprint.py          # 执行指纹（并入身份判据）
│   ├── publish_journal.py           # 发布事务 journal（硬杀后恢复）
│   ├── publish_lock.py              # 逐目标发布锁
│   └── gpen/                        # GPEN 架构（上游 yangxy/GPEN）
│       └── face_model/              #   FullGenerator + op（纯 PyTorch 回退）
│
└── models/                          # AI 模型权重目录
```

### 色彩保真校验

两条链在**发布前**比对「原始输入 vs 编码成品」的 RGB 三通道均值并记录结果。
阈值是 `color_drift_threshold`（默认 `60.0`，0-255 单位）；把
`color_drift_enforce` 设为 `true` 则超阈值时失败而不只是告警。

默认 60 是**刻意宽松**的 —— 它抓的是**整幅崩坏**（纯黑/纯白/内容丢失，
纯黑帧实测约 116），而合法的宽色域转换（ProPhoto→sRGB 实测约 27）会被放行。
细微色偏需要按素材单独标定；阈值设太紧会频繁误报，日志就没人看了。

照片链配置在 `photo:` 段，视频链在 `global:` 段。

同类开关：`require_16bit_color_conversion`（默认 `false`）。宽色域→sRGB 走
PIL ImageCms，而 Pillow 的 CMS 只支持 8-bit；置 `true` 会在这种降级发生时
直接失败（rc=86），而不是交出 8-bit 精度的母版。

### 成品核对体系

成品发布前跑**三道相互独立**的核对门，各有对应的强制出片开关：

| 门 | 对比对象 | 强制开关 |
|---|---|---|
| 音画同步 | 成品内部 | `allow_sync_fail_publish` |
| 全链核对 | **原始素材 vs 成品** | `allow_verify_fail_publish` |
| 机械核对 | step6 产物 vs 成品 | `allow_verify_fail_publish` |

音画漂移按**相对源片**判定（`|成品漂移 − 源片漂移|`），不用固定阈值 ——
实测 204 个真实手机素材，固定阈值误杀 **31%**（漂移是老素材自带的），
改用相对判据后降到 **1.5%**。

`full_chain_verify_mode` 决定全链核对的严格度：
`frames_only`（默认，误杀最低）/ `relative` / `absolute`。
结果一律写入 `<output>.fullchain.json`；
用 `shared_modules/fullchain_report.py <目录> --csv out.csv` 可按漂移排序筛查整批。

低置信度分类（`needs_confirmation`，<0.80）在非交互批处理中会被 SKIP，放行方式：

```bash
python shared_modules/manifest_editor.py <manifest> list --low-confidence
python shared_modules/manifest_editor.py <manifest> confirm --all-low-confidence
```

`stage_hard_timeout_hours` 设 `0` = 完全禁用硬超时；
`stage_inactivity_timeout` 设 `0` = 关闭看门狗。

### 版本与规范说明

**本仓库以代码为唯一事实来源**（`main.py` / `stages/` / `config.yaml` /
`shared_modules/`）。历史规范文档仅作背景参考，且**可能已落后于实现**：

- `v6.32.md` — 视频修复技术规范（历史版本）
- `v7.32.md` — 架构规范（照片链、清单编辑器、运行模式）

> 规范与代码不一致时，以代码行为为准。本 README 中的命令、参数、
> 退出码、目录结构均按当前代码实测核对；规范文档如需继续作为依据，
> 应先按代码修订。规范原文为本地文档，不入库。

### 问题台账与审计报告（仅本地，不入库）

缺陷清单、逐文件审计报告、修复台账与验收记录属于内部工作资料，
**不随仓库分发**，保存在项目工作目录本地：`bugs.md`、`shixian.md`、
`修复清单.md`、`MEDIA_ACCEPTANCE_PLAN.md`、`*-audit.md`。
