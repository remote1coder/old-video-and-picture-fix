# 项目操作手册 (PROJECT MANUAL)

> **目标读者**: 首次使用本项目的纯新手。本手册假设你熟悉基本的 Python 和命令行操作，但**不需要**视频处理专业知识。

---

## 1. 项目简介

本项目是**视频与照片混合修复流水线**，支持：
- **视频修复**：老录像带、DV数码带、HDV/AVCHD高清带的去噪、去隔行、AI放大、面部修复、颗粒合成
- **照片修复**：老照片扫描件、数码照片的去噪、去划痕、AI放大、面部修复、AVIF输出

流水线采用**传统信号处理 + AI超分**混合架构：
1. 先通过传统方法（去划痕、去块、去色带）修复基础画质
2. 再用 RealESRGAN/DAT 进行 AI 放大（仅低分辨率素材）
3. 最后可选 GFPGAN/CodeFormer 面部修复
4. 照片输出 AVIF 10bit SDR；可选 16bit RGB444 无损存档模式

---

## 2. 环境准备

### 2.1 操作系统
- **推荐**：Linux (Ubuntu 22.04 LTS) 或 WSL2 (Windows 10/11)
- **支持**：Windows 10/11 (Native, 部分功能受限)、macOS (Intel/Apple Silicon, 无 CUDA)
- **注意**：VapourSynth 和 CUDA 加速在 Linux/WSL2 下最稳定

### 2.2 必需软件

按顺序安装以下软件。每步安装后建议验证版本。

#### 1) Python 3.9+
```bash
# Ubuntu / WSL2
sudo apt update
sudo apt install python3 python3-pip python3-venv

# 验证
python3 --version   # 应显示 3.9.x 或更高
```

#### 2) FFmpeg 4.4+
```bash
# Ubuntu 22.04 (自带 4.4+)
sudo apt install ffmpeg

# 验证版本 (必须 >= 4.4，否则 bandreject width_type=h 不支持)
ffmpeg -version | head -1
```

#### 3) ExifTool
```bash
sudo apt install libimage-exiftool-perl
# 验证
exiftool -ver
```

#### 4) G'MIC
```bash
sudo apt install gmic
# 验证
gmic --version
```

#### 5) avifenc (照片修复必需)
```bash
# Ubuntu 22.04+
sudo apt install libavif-examples

# 验证
avifenc --version
```

#### 6) VapourSynth (仅视频修复需要)
**必须用 R72（最后完整支持 API3 的版本）**；`pip install vapoursynth`
默认会装最新版（R80+，仅 API4），API3 社区插件全部失效。

```bash
# 方案 A（本机已验证）：直接用 Portable R72
tools_r4/vs_portable/r72/python.exe -c "import vapoursynth as vs; print(vs.core)"

# 方案 B：自建环境时按 requirements.txt 钉版本
pip install "vapoursynth>=57,<73"
```

#### 7) havsfunc r33 (仅视频修复需要)
`havsfunc==33` 的依赖声明是 `VapourSynth (>=57)`（无上限），**直接 pip
装会把 VS 连带升到最新版**。正确顺序是先钉住 VS 再装：

```bash
pip install "vapoursynth>=57,<73"
pip install "havsfunc==33" --no-deps
pip install vsutil

# 验证（用 R72）
tools_r4/vs_portable/r72/python.exe -c "import havsfunc; print('havsfunc OK')"
```

### 2.3 Python 依赖

**以 `requirements.txt` 为准**（其中已固定 `vapoursynth>=57,<73` 并说明了
为什么不能升级、`havsfunc==33` 为什么不能直接装）。在项目根目录：

```bash
# 创建虚拟环境
python3 -m venv venv

# 激活 (Linux/Mac)
source venv/bin/activate

# 激活 (Windows CMD)
venv\Scripts\activate.bat

# 激活 (Windows PowerShell)
venv\Scripts\Activate.ps1

# 安装依赖
pip install -r requirements.txt
```

**本机实测环境（Windows，推荐直接复用）**：

| 用途 | 解释器 |
|---|---|
| 视频链编排（VapourSynth R72，API3 插件全通） | `tools_r4/vs_portable/r72/python.exe` |
| 视频链 AI 推理（torch/realesrgan/DAT/facexlib） | 自动解析到 `.venv_r4`（`shared_modules/ai_runtime.py`） |
| 照片链（全栈，含 CUDA） | `.venv_r4/Scripts/python.exe` |

两者均为 Python 3.12。**不要用 `.venv_r4` 跑视频链编排**：它的
VapourSynth 已被 pip 升到 R81（API4），API3 社区插件会全部加载失败。
PATH 需包含 `tools_r4/gmic/gmic-*-cli-win64` 与
`tools_r4/exiftool/exiftool-*` 子目录（只加 `tools_r4` 会 `[WinError 2]`）。

### 2.4 AI 模型下载

在项目根目录创建 `models/` 文件夹，下载以下模型：

```bash
mkdir models
cd models

# RealESRGAN (AI放大，默认 realesr-general-x4v3 通用修复版)
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
# 可选备用: realesr-general-wdn-x4v3.pth / RealESRGAN_x2plus.pth / RealESRGAN_x4plus.pth

# GFPGAN (面部修复)
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth

# DAT (AI放大备选，默认 DAT-2)
# 从 https://github.com/zhengchen1999/DAT 的 README Google Drive 链接下载 DAT_2_x2.pth 和 DAT_2_x4.pth
# （官方仓库是 zhengchen1999/DAT，模型在 Google Drive/百度网盘，不在 GitHub Releases）

# CodeFormer (面部修复备选)
# 从 https://github.com/sczhou/CodeFormer 的 Google Drive 下载 codeformer.pth

# arnndn (音频降噪，FFmpeg<4.4 降级用)
# 从 https://github.com/richardpl/arnndn-models 下载 std.rnnn 放入 models/

# RetinaFace 人脸检测权重
# detection_Resnet50_Final.pth 放入 models/ 或 facexlib/weights/（facexlib 缺失时自动下载）

cd ..
```

---

## 3. 项目结构

克隆或解压后，项目目录结构如下：

```
project_root/
├── video_restoration_pipeline/      # 视频修复流水线
│   ├── classify.py                  # 视频素材分类器
│   ├── main.py                      # 视频主控脚本
│   ├── config.yaml                  # 视频配置
│   └── stages/                      # 视频处理阶段
│       ├── stage0_audio_extract.py  # 音频提取
│       ├── stage1_preprocess.py     # 预处理（去噪、去隔行、QTGMC）
│       ├── stage2_traditional_check.py  # 质量验证
│       ├── stage3_ai_upscale.py     # AI放大
│       ├── stage4_postprocess.py    # 后处理
│       ├── stage5_ai_face.py        # 面部修复
│       ├── stage6_grain_synthesis.py # 颗粒合成
│       └── stage7_encode.py         # 最终编码
│
├── photo_restoration_pipeline/      # 照片修复流水线
│   ├── classify.py                  # 照片分类器
│   ├── main.py                      # 照片主控脚本（单文件 + --input-dir 批量；manifest 工作流已支持 --manifest/--resume）
│   ├── config.yaml                  # 照片配置
│   └── stages/                      # 照片处理阶段（默认链 6 步）
│       ├── stage1_preprocess.py     # 预处理（色彩空间、去划痕、去块、去色带）
│       ├── stage2_ai_pre.py         # AI 前处理：单通道校正
│       ├── stage3_ai_upscale.py     # AI放大
│       ├── stage4_ai_face.py        # 面部修复（GFPGAN/CodeFormer）
│       ├── stage5_postprocess.py    # 后处理（局部遮罩 + 颗粒合成 + 人像色彩匹配）
│       └── stage6_encode.py         # AVIF最终编码
│
├── shared_modules/                  # 共享模块（两边共用）
│   ├── checkpoint.py                # 断点续传 / 身份与参数哈希
│   ├── dependency_check.py          # 依赖检查
│   ├── logger.py                    # 结构化日志记录
│   ├── audio_utils.py               # 音频分析
│   ├── manifest_generate.py         # 清单生成
│   ├── manifest_editor.py           # 清单编辑（含交互式 TUI）
│   ├── manifest_loader.py           # 清单加载
│   ├── preview.py                   # 预览工具
│   ├── media_io.py                  # 媒体探测 / IO 辅助
│   ├── cancel_guard.py              # 取消 / 中断守卫（rc=88）
│   ├── ai_runtime.py                # AI 子进程解释器解析
│   └── dat_runtime.py               # DAT 模型运行时
│
├── models/                          # AI模型文件夹（需自行下载）
├── logs/                            # 日志输出（自动创建）
├── temp/                            # 中间文件（自动创建）
└── output/                          # 最终输出（自动创建）
```

---

## 4. 首次运行：依赖检查

在修复任何素材前，先运行依赖检查，确保环境完整：

### 4.1 视频修复依赖检查
```bash
# 激活虚拟环境（Linux/macOS: source venv/bin/activate；
# Windows: venv\Scripts\activate 或直接 .venv_r4\Scripts\python.exe）
# 运行依赖检查
python shared_modules/dependency_check.py \
    -c video_restoration_pipeline/config.yaml \
    --pipeline-type video
```

如果看到 `All required dependencies are satisfied.`，表示视频修复环境就绪。

如果看到错误，按提示安装缺失的依赖。

### 4.2 照片修复依赖检查
```bash
python shared_modules/dependency_check.py \
    -c photo_restoration_pipeline/config.yaml \
    --pipeline-type photo
```

**注意**: 照片修复不需要 VapourSynth 和 havsfunc，但需要 G'MIC 和 avifenc。

---

## 5. 配置文件

### 5.1 视频配置 (video_restoration_pipeline/config.yaml)

关键配置项：

```yaml
global:
  output_dir: "./output"
  temp_dir: "./temp"
  encoder: "svtav1"
  crf: 24                          # 视频质量 (越小越好，文件越大)
  preset: 4                        # 编码速度 (0-8, 越小越慢质量越好)
  pix_fmt: "yuv420p10le"
  color_range: "limited"
  platform: "auto"                 # auto / wsl2 / linux / windows
  gpu_priority: ["cuda", "rocm", "mps", "cpu"]
  cleanup_intermediates: true      # 完成后清理中间文件
  retry_count: 2                   # 失败重试次数
  log_dir: "./logs"

ai:
  enabled: false                   # 是否启用AI放大
  upscale_model: "realesr-general-x4v3"
  upscale_scale: 2                 # 放大倍数 (仅支持 2 或 4)
  face_restore: ""                 # GFPGAN 或 CodeFormer，空字符串表示不启用
  force_traditional_first: true    # 强制先执行传统修复
  force_allow_banned_upscale: false

audio:
  enabled: true
  hum_freq: 50                     # 电源哼声频率 (50或60)
  hum_harmonics: 3
```

### 5.2 照片配置 (photo_restoration_pipeline/config.yaml)

关键配置项：

```yaml
global:
  output_dir: "./output"
  temp_dir: "./temp"
  encoder: "avifenc"
  crf: 24
  preset: 4
  pix_fmt: "yuv420p10le"
  color_range: "limited"
  platform: "auto"
  gpu_priority: ["cuda", "rocm", "mps", "cpu"]
  cleanup_intermediates: true
  retry_count: 2
  log_dir: "./logs"

photo:
  max_long_edge: 3840              # AI超分禁令阈值
  max_short_edge: 2160
  dpi_threshold: 1200
  scratch_remove: true             # 去划痕
  scratch_remove_strength: 20
  deblock_strength: 25             # 去块强度
  deband_threshold: 1.2            # 去色带阈值
  grain_synthesis: true            # 颗粒合成
  grain_strength: 0.3
  grain_size: 1.0
  grain_roughness: 0.5
  archival_mode: false             # true = 输出 RGB444 16bit 无损 AVIF

ai:
  enabled: false
  upscale_model: "realesr-general-x4v3"
  upscale_scale: 2                 # 仅支持 2 或 4
  face_restore: ""                 # GFPGAN 或 CodeFormer
  force_traditional_first: true
  force_allow_banned_upscale: false
```

**重要**: `archival_mode: true` 时，输出为 RGB444 16bit 无损 AVIF（`--cicp 1/13/0`）。默认 `archival_mode: false` 输出 YUV420 10bit 有损 AVIF（`--cicp 1/13/1`）。

---

## 6. 视频修复操作

### 6.1 单文件修复

```bash
# 运行环境见 §2；Windows 本机实测用 Portable R72 + .venv_r4（见 README 运行环境要点）
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    -i input_video.avi \
    -o ./output \
    --ai-enabled \
    --ai-upscale-factor 2 \
    --face-restore GFPGAN
```

参数说明：
- `-c`: 配置文件路径
- `-i`: 输入视频文件
- `-o`: **输出目录**（不是文件名）。成品为
  `./output/<输入文件名去扩展名>_restored.mkv`
- `--ai-enabled`: 启用AI放大
- `--ai-upscale-factor`: AI放大倍数 (2或4)
- `--face-restore`: 面部修复模型 (GFPGAN或CodeFormer)

### 6.2 批量修复 (使用清单)

#### 步骤1：生成清单
```bash
# 视频清单
python shared_modules/manifest_generate.py \
    -c video_restoration_pipeline/config.yaml \
    -o ./batch_manifest.yaml \
    -i ./video_folder \
    --type video

# 照片清单
python shared_modules/manifest_generate.py \
    -c photo_restoration_pipeline/config.yaml \
    -o ./batch_manifest.yaml \
    -i ./photo_folder \
    --type photo
```

#### 步骤2：编辑清单（可选）

**命令行方式**（推荐）：
```bash
# 查看所有任务
python shared_modules/manifest_editor.py ./batch_manifest.yaml list

# 跳过某个任务
python shared_modules/manifest_editor.py ./batch_manifest.yaml set-status job_001 SKIP

# 为某个任务设置参数覆盖
python shared_modules/manifest_editor.py ./batch_manifest.yaml set-override job_001 crf 20
```

**Python 方式**（高级用户）：
```python
from shared_modules.manifest_editor import set_job_status, set_user_override

set_job_status("./batch_manifest.yaml", "job_001", "SKIP")
set_user_override("./batch_manifest.yaml", "job_001", "crf", "20")
```

#### 步骤3：批量运行
```bash
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    -m ./batch_manifest.yaml
```

### 6.3 不同素材类型的特殊参数

#### 老录像带 (ANALOG_TAPE)
```bash
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    -i tape.avi \
    -o ./output \
    --ai-enabled \
    --ai-upscale-factor 2
```

#### DV数码带 (DV_DIGITAL)
```bash
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    -i dv.mov \
    -o ./output \
    --ai-enabled \
    --ai-upscale-factor 2
```

#### HDV/AVCHD (HDV_AVCHD)
```bash
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    -i hdv.m2ts \
    -o ./output
```

**注意 1**: HDV/AVCHD 素材默认 **不建议** AI 放大（原生已 1080p），如需强制放大请在清单该 job 的 `user_overrides` 里设置 `force_allow_banned_upscale: true`（视频 main.py **没有** `--crf` / `--preset` 命令行参数，编码质量请改 config.yaml 的 `global.crf` / `global.preset`，或在清单 `user_overrides` 里覆盖）。

---

## 7. 照片修复操作

### 7.1 单文件修复（基础）

```bash
# 照片链用 .venv_r4（Windows 实测）：.venv_r4\Scripts\python.exe

python photo_restoration_pipeline/main.py \
    -c photo_restoration_pipeline/config.yaml \
    -i old_photo.jpg \
    -o restored_photo.avif
```

**输出说明**:
- 默认输出 AVIF 格式（10bit YUV420 SDR），体积小巧适合分享
- 若需最高质量存档，在 config.yaml 中设置 `archival_mode: true`，输出 RGB444 16bit 无损 AVIF

### 7.2 启用 AI 放大

```bash
python photo_restoration_pipeline/main.py \
    -c photo_restoration_pipeline/config.yaml \
    -i old_photo.jpg \
    -o restored_photo.avif \
    --ai-enabled \
    --ai-upscale-factor 2
```

**AI 放大限制**:
- 仅支持 2× 或 4× 整数倍放大
- 放大后目标分辨率不能超过 1920×1080（v6.3 规范要求）
- 原始长边 > 3840px 或短边 > 2160px 的素材会自动禁用 AI 放大
- 动漫模型 (`*anime*`) 被禁止用于真实照片

### 7.3 启用面部修复

```bash
python photo_restoration_pipeline/main.py \
    -c photo_restoration_pipeline/config.yaml \
    -i old_photo.jpg \
    -o restored_photo.avif \
    --ai-enabled \
    --ai-upscale-factor 2 \
    --face-restore GFPGAN
```

支持的面部修复模型：
- `GFPGAN`: 重建模糊人脸，恢复五官
- `CodeFormer`: 保真度更高，保留皱纹和年龄特征

**人脸修复的实测前提**（重要，直接影响你能否看到效果）：

- 人脸检测由 **facexlib** 完成。规范里列出的 `RetinaFace` / `YOLOv8-face` /
  `MTCNN` 三个独立包在本项目的环境里都没有安装——不要指望通过安装它们
  来"启用"人脸修复，facexlib 才是 GFPGAN/CodeFormer 自带的检测实现。
- **facexlib 在 37 张真实照片上检出 0 张脸**。如果你处理真实照片时日志出现
  `STAGE4_NO_FACE` 且流程跳过面部修复，这是检出为零、不是功能坏了。
  想验证人脸修复是否工作，请先用
  `E:\codex\_repair_r4_20261013\matrix\photo_opt\face_sample\face_sample.png`
  这类确含人脸的样张。
- 面部修复会让输出尺寸变化：GFPGAN 在 2× 超分基础上再放大人脸区域，
  样张上得到 2560×1920，而 CodeFormer 为 1280×960。这是预期行为，
  不是 bug。
- 若日志出现 `STAGE4_FAILED`（退出码 83），说明推理子进程失败；
  此时应查看该 job 日志中的 `GFPGAN/CodeFormer subprocess failed` 行，
  它会带出子进程的原始报错。

### 7.4 批量修复

**方式一：使用 `--input-dir`（推荐）**

照片 `main.py` 支持原生批量目录处理：

```bash
python photo_restoration_pipeline/main.py \
    -c photo_restoration_pipeline/config.yaml \
    --input-dir ./photos/ \
    -o ./output/ \
    --ai-enabled \
    --ai-upscale-factor 2
```

参数说明：
- `--input-dir`: 输入照片目录（支持 `.jpg`, `.jpeg`, `.png`, `.tiff`, `.tif`, `.bmp`, `.avif`）
- `-o`: 输出目录（每个文件生成同名 `.avif`）
- 支持 `--workers N` 并行处理（使用 `ProcessPoolExecutor`，spawn 安全）

**方式二：Shell 循环（兼容旧版）**

若需要更复杂的输出命名或与其他工具链集成：

```bash
for img in ./photos/*.jpg; do
    python photo_restoration_pipeline/main.py \
        -c photo_restoration_pipeline/config.yaml \
        -i "$img" \
        -o "./output/$(basename "$img" .jpg)_restored.avif" \
        --ai-enabled \
        --ai-upscale-factor 2
done
```

**清单系统**: 照片 `main.py` 已支持 `--manifest`/`--resume` 批量清单工作流（scan→execute→resume），与 video 流水线一致。见 ISSUE-P-MAIN-001。

---

## 8. 日志查看

### 8.1 日志位置
- 每任务一个日志文件：`logs/job_{job_id}.log`（例如 job_id=`job_001` →
  `logs/job_job_001.log`）
- 编排层汇总日志：`logs/general.log`
- 格式：JSON Lines（每行一条独立 JSON，**不是**单个 JSON 文档）

### 8.2 查看日志
```bash
# 跟踪最新日志
tail -f logs/job_*.log

# 逐行格式化（JSONL：每行一个 JSON，用 jq 最方便）
jq -c . logs/job_job_001.log | head -20

# 没有 jq 时：逐行解析
python -c "import sys,json;[print(json.loads(l)['event'], json.loads(l)['message']) for l in open('logs/job_job_001.log',encoding='utf-8') if l.strip()]"
```

### 8.3 日志字段示例
```json
{
  "timestamp": "2026-04-30T12:00:00+08:00",
  "level": "INFO",
  "event": "STAGE3_START",
  "message": "AI upscaling",
  "job_id": "photo_1234567890",
  "thread": "MainThread"
}
```

---

## 9. 问题与限制（本地文档）

缺陷登记、逐文件审计、修复台账与验收记录属于内部工作资料，**不随仓库分发**：

- `bugs.md` — 缺陷登记册（BUG-xxx / ISSUE-xxx）
- `shixian.md` — 实现差距登记
- `修复清单.md` — 按轮次的修复台账与根因分析
- `MEDIA_ACCEPTANCE_PLAN.md` — 验收门与实测记录

本手册只保留**当前代码的操作方式**；如遇具体报错，先查 §10「常见问题」，
再到上述本地文档中检索对应事件的根因。

---

## 10. 故障排除

### 10.1 常见问题

#### 问题1：FFmpeg版本过低
**症状**: bandreject width_type=h 不支持  
**解决**: 升级 FFmpeg 到 4.4+
```bash
sudo apt install ffmpeg  # Ubuntu 22.04+ 自带 4.4+
```

#### 问题2：VapourSynth插件缺失
**症状**: stage1_preprocess.py 报错 "VS Plugin QTGMC not available"  
**解决**: 不要用 `pip install havsfunc==33` —— 它会把 VapourSynth 连带升级到
R73+，API3 社区插件（MVTools2/NNEDI3/MaskTools2/KNLMeansCL/focus2/ffms2）
会全部加载失败（详见 README「运行环境要点」）。正确做法是使用本机已验证的
Portable R72 环境：

```bash
tools_r4/vs_portable/r72/python.exe video_restoration_pipeline/main.py --help
```

如需自建同版本环境，见 `requirements.txt` 中 `vapoursynth>=57,<73` 的注释。

#### 问题3：AI模型文件缺失
**症状**: stage3_ai_upscale.py 报错 "Model file not found"  
**解决**: 下载模型到 `models/` 目录
```bash
python shared_modules/dependency_check.py -c config.yaml
```

#### 问题4：内存不足 (OOM)
**症状**: AI放大阶段OOM  
**解决**:
- 降低AI放大倍数：`--ai-upscale-factor 2` (不用4)
- 使用CPU模式（较慢但省内存）
- 确保关闭其他占用显存的程序

#### 问题5：磁盘空间不足
**症状**: 中间文件写入失败  
**解决**:
- 清理 temp 目录：`rm -rf temp/*`
- 增加磁盘空间
- 确认 `cleanup_intermediates: true`（默认开启）

#### 问题6：avifenc 未安装（照片修复）
**症状**: stage6_encode.py 报错 "avifenc not found"  
**解决**:
```bash
sudo apt install libavif-examples
```

#### 问题7：G'MIC 滤镜失败（照片修复）
**症状**: 去划痕/去块/去色带未生效，日志无错误  
**解决**: 检查 G'MIC 版本和滤镜可用性：
```bash
gmic -h repair
gmic -h deblock
gmic -h deband
```

#### 问题8：WSL2 路径重定向不生效
**症状**: WSL2 下处理大文件时 temp/output 仍在 `/mnt/c/`，I/O 极慢  
**兜底解决**: 在 config.yaml 中显式将 `temp_dir` 和 `output_dir` 都设置为 Linux 原生分区路径：
```yaml
global:
  temp_dir: "/home/user/temp"
  output_dir: "/home/user/output"
```
注意：视频 `main.py` **没有** `--temp-dir` / `--output-dir` 命令行参数，
路径只能通过 config.yaml 或清单 job 的 `temp_dir`/`output_path` 指定；
照片 `main.py` 支持 `--temp-dir` / `--output`。

### 10.2 怎么确认这次修复真的成功了

退出码 0 **不等于画面活着**。本项目的照片链曾出现过"全链 rc=0、日志无报错、
但交付的 AVIF 是全黑图"的情况（成品仅 386 字节，像素均值 0.0）。
交付前请至少做一次像素级确认：

```bash
# 照片：解出成 PNG 看一眼，并看像素统计
python -c "
from PIL import Image; import numpy as np
Image.MAX_IMAGE_PIXELS = None
a = np.array(Image.open(r'你的输出.avif').convert('RGB'))
print(a.shape, 'mean', round(float(a.mean()),2), 'std', round(float(a.std()),2))
"
# mean 接近 0 → 成品是黑图，别当成成功
```

```bash
# 视频：抽第一帧看像素统计
ffmpeg -v error -i 输出.mkv -frames:v 1 -y /tmp/chk.png
python -c "
from PIL import Image; import numpy as np
a = np.array(Image.open('/tmp/chk.png').convert('RGB'))
print('mean', round(float(a.mean()),2), 'std', round(float(a.std()),2))
"
```

还要顺带确认：

| 确认什么 | 看哪里 |
|---|---|
| 参数真的生效了 | 日志里的 `PARAM_OVERRIDE`、生成的 `.vpy` |
| 人脸/帧数不是 0 | `STAGE4_FACE_DETECT`（faces found 数量）、`STAGE*_BLEND` 帧数 |
| 没有被静默跳过 | `STAGE4_NO_FACE_SKIP`、`SKIP` 状态作业 |
| 中间 TIFF 读得回来 | 用 `tifffile` 直接打开 |

**只核对 rc + 尺寸 + 位深是不够的**——那三项对全黑图一样"通过"。

### 10.3 获取帮助

```bash
# 查看主脚本帮助
python video_restoration_pipeline/main.py --help
python photo_restoration_pipeline/main.py --help

# 查看依赖检查帮助
python shared_modules/dependency_check.py --help
```

---

## 11. 进阶操作

### 11.1 断点续传

如果修复中断，重新运行相同命令即可。流水线会：
1. 计算 param_hash 匹配现有中间文件
2. 跳过已完成的 stage
3. 从断点继续

**限制**:
- 照片流水线支持单文件断点续传（通过中间文件存在性判断）
- 视频流水线支持清单级断点续传
- `param_hash` 的 `CORE_QUALITY_PARAMS` 白名单已覆盖照片与视频的质量参数

### 11.2 预览模式

**当前状态**: 预览模块 (`shared_modules/preview.py`) 已实现基础功能，但仍有差距：
- ✅ 视频静态帧提取（10%, 50%, 90% 位置）
- ✅ 0.5 秒固定时长 WebP 动画片段（动态帧数）
- ✅ 照片 thumbnail + center crop + detail crop
- ✅ 强制覆盖对比帧（side-by-side）
- ✅ 自动检测画面运动幅度并提取最剧烈的 3 段（`detect_motion_segments()` 已实现）
- ✅ VFR 素材使用帧号精确提取（`select=eq(n\,target_frame)`），时间位置精确

照片预览：
```bash
python photo_restoration_pipeline/main.py \
    -c photo_restoration_pipeline/config.yaml \
    -i input.jpg --preview
```

视频预览：
```bash
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    -i input.avi --preview
```

### 11.3 参数层叠示例

```bash
# CLI参数最高优先级，覆盖配置文件
python photo_restoration_pipeline/main.py \
    -c photo_restoration_pipeline/config.yaml \
    -i input.jpg \
    --crf 18 \              # 覆盖 global.crf
    --ai-enabled \          # 启用 AI
    --face-restore CodeFormer
```

### 11.4 WSL2 路径重定向

**WSL2 路径重定向**: 自动重定向已正常工作。将 `/mnt/c/temp` 自动重定向到 Linux 原生分区（如 `/home/user/temp`）。

如需手动指定路径，可在配置中设置：

```bash
# 在 config.yaml 中设置
# temp_dir: "/home/user/temp"
# output_dir: "/home/user/output"

# 或使用命令行覆盖
python photo_restoration_pipeline/main.py \
    -c config.yaml -i /mnt/c/photos/old.jpg \
    --temp-dir /home/user/temp \
    --output /home/user/output/restored.avif
```

如需禁用自动重定向（当前实际上等同于禁用错误的重定向）：
```bash
python photo_restoration_pipeline/main.py ... --no-wsl-redirect
```

**成品回迁注意事项**: `wsl_copy_back()` 已使用 `shutil.copyfile`（不复制扩展元数据）+ `os.utime()` 手动同步时间戳，在 WSL2 NTFS `metadata=disabled` 下正常工作。三重校验（MD5 + size + ffprobe）确保文件完整性。跨文件系统时自动回退到 copyfile，不触发 PermissionError。

---

## 12. 产物核对与人工确认

### 12.1 三道核对门

成品发布前会跑三道**相互独立**的核对，任一失败默认返回退出码 86：

| 门 | 判据 | 强制出片开关 |
|---|---|---|
| 音画同步 | 成品内部音视频区间 | `allow_sync_fail_publish` |
| 全链核对 | 原始素材 vs 成品 | `allow_verify_fail_publish` |
| 机械核对 | step6 产物 vs 成品 | `allow_verify_fail_publish` |

打开强制开关后仍会发布，但会记 ERROR 级日志，且问题写进
`<output>.fullchain.json` / `<output>.verify.json` 跟随文件 —— **不等于验过**。

### 12.2 全链核对三档模式

```yaml
full_chain_verify: true
full_chain_verify_mode: frames_only   # frames_only | relative | absolute
```

| 模式 | 帧数 | 时长/漂移 | 误杀风险 | 适用 |
|---|---|---|---|---|
| `frames_only`（默认） | 拦 | 只记录 | 低 | 日常批处理 |
| `relative` | 拦 | 拦（相对源片） | 中 | 想连漂移一起管 |
| `absolute` | 拦 | 拦 + 成品绝对漂移 | 高 | **体检**，配合强制出片 |

### 12.3 漂移筛查报告（体检）

```bash
python shared_modules/fullchain_report.py <输出目录> --csv drift.csv
```

按漂移排序，标注超过 ITU-R BT.1359-1 可接受线（185 ms）的份数。
`absolute` 档 + `allow_verify_fail_publish: true` 跑一遍，即可筛出
"哪些素材有漂移" 而不影响出片。

### 12.4 音画阈值

```yaml
sync_warn_ms_lag: 185      # 声音滞后告警
sync_fail_ms_lag: 400      # 声音滞后失败
sync_warn_ms_lead: 90      # 声音超前告警
sync_fail_ms_lead: 200     # 声音超前失败
```

依据 ITU-R BT.1359-1（可察觉 +45 / −125 ms，可接受 +90 / −185 ms），
此处取其约 2 倍，因为老素材可能自带原始偏移。设非正数退回默认值；
**不能设成 0 来关闭判定**。

### 12.5 低置信度素材的人工确认

分类器对置信度 <0.80 的素材置 `needs_confirmation`，非交互批处理会 SKIP。
三步放行：

```bash
# 1) 先看哪些会被拦
python shared_modules/manifest_editor.py <manifest> list --low-confidence

# 2) 人工核对类别后，一次确认全部（几百上千素材用这个）
python shared_modules/manifest_editor.py <manifest> confirm --all-low-confidence

# 3) 跑批次
python video_restoration_pipeline/main.py --manifest <manifest> \
  --config video_restoration_pipeline/config.yaml \
  --mode batch --output <输出目录> --no-wsl-redirect \
  --ai-enabled --ai-upscale-factor 2 --workers 1
```

`confirm` 会做四件事：清 `needs_confirmation`、把 confidence 至少抬到 0.80、
记 `manual_confirmed` / `confirmed_at` 便于追溯、并把 `SKIP` 复位为 `PENDING`。

### 12.6 超时与看门狗

```yaml
stage_hard_timeout_hours: 48        # 设 0 或负数 = 完全禁用硬超时
stage_inactivity_timeout: 1800      # 设 0 或负数 = 关闭看门狗
```

硬超时是**最终保险**，判死主要交给看门狗（stage 输出 N 秒无增长即杀）。
有素材要跑非常久时关掉硬超时即可；此时唯一防线是看门狗，
若看门狗也关了就是真的不限时，需自行承担风险。

### 12.7 AI 超分倍率选择

| 源分辨率 | 建议倍率 | 说明 |
|---|---|---|
| ≤ 640×368 | 2× | 4× 对 ANALOG_TAPE 被 v6.32 §4B 硬禁 |
| 544×960 ~ 800×480 | 2× | 4× 可用但输出体积与耗时暴涨 |
| ≥ 1280×720 | 2× | 4× 会超 2160p，实测 7.5 小时仍被看门狗杀死 |

### 12.8 帧传输：默认零落盘流式超分（`ai_stream_raw: true`）

**目的不是提速，是省固态硬盘寿命。**

#### 为什么要省

RGB48 PNG 单帧实测体积（随机噪声；真实画面会小些，但同一量级）：

| 分辨率 | 单帧 PNG | 2612 帧仅输出帧的写入 |
|---|---|---|
| 1280×720 | 5.28 MB | **13.5 GB** |
| 2560×1440 | 21.13 MB | **53.9 GB** |
| 5120×2880 | 84.53 MB | **215.6 GB** |

本批 204 个素材按 2× 估算 ≈ **11 TB** 写入（尚未计入输入帧与中间 mkv）。
这些帧"写完马上就删"，对固态硬盘是**纯粹的健康度损失**。

#### 实现：三进程管道，帧不进磁盘

```
ffmpeg(解码 rawvideo rgb48le) → AI(读stdin→超分→写stdout) → ffmpeg(ffv1) → step3.mkv
```

OS 负责背压，父进程无需协调。**没有双向管道，避免了"stderr=PIPE 未排空"这一类阻塞**（三个子进程都把 stderr 落到临时文件）。
但这不等于已证明所有退出路径都正确 —— 背压、短读、EOF 传播、
超时与取消仍需各自验证。
**不受 `ai_window_size` 限制，整段一次跑完。**
任何一步失败自动落回原 PNG 路径（记 `STAGE3_STREAM_FALLBACK`）。

#### 实测（1280×720 → 2560×1440，300 帧，2×）

| 路线 | 耗时 | rc | temp 帧文件 | 成品 |
|---|---|---|---|---|
| PNG | 18m36s | 0 | 约 7.7 GB | 2560×1440 / 300 帧 / 12.000s |
| 流式 | **16m34s** | 0 | **0** | 2560×1440 / 300 帧 / 12.000s |

结构比对（64×36 缩略）：相关系数 **0.999894**、平均差 0.26%
（差异来自 stage6 随机颗粒合成，非超分内容差异）。

> **口径限定（重要，不要过度解读）**
> - 上表「约 7.7 GB → 0」指的是**该样本 stage3 的临时帧 PNG 写入量**，
>   不是整个流程没有磁盘写入 —— FFV1 中间视频、后续成品与日志照样写盘。
> - 「temp 里 PNG 数为 0」是**任务结束时的目录快照**；它不能单独证明过程
>   中从未写过 PNG。要下结论需结合传输路径事件（STAGE3_STREAM_*）、
>   运行期写入计数与回退记录一起看。
> - 因此**不能**据此宣称"SSD 寿命已延长的实测结论"，只能说"该样本
>   stage3 的临时帧写入为 0"。
> - 另：两条路线的推理精度不同（PNG 脚本在 CUDA 可用时 `half=False`，
>   流式模板固定 `half=True`），所以这不是严格意义上的单变量对照。

#### 多素材验证（5/5 PASS，零回退，零帧落盘）

| 素材 | 耗时 | 成品 |
|---|---|---|
| 320×240 / 1721 帧 | 14m30s | 640×480 / 1721 帧 / 56.800s |
| 544×960 / 458 帧 | 16m20s | 1088×1920 / 458 帧 / 15.209s |
| 544×960 / 322 帧 | 10m39s | 1088×1920 / 322 帧 / 11.679s |
| 368×640 / 335 帧 | 7m35s | 736×1280 / 335 帧 / 13.959s |
| 1280×720 / 912 帧 | 38m43s | 2560×1440 / 912 帧 / 42.775s |

帧数全部精确一致、尺寸全部精确 2×、时长差 ≤ 23ms、
`STAGE3_STREAM_FALLBACK` 计数 0、`temp` 里 PNG 帧文件数 0。

#### 关闭方式

`ai_stream_raw: false` 即回到原 PNG 中转路径（AI 侧由 `ai_frame_transport` 决定用
PNG 还是共享内存环）。**共享内存环单独使用时并不省写入**——它的两端仍是 PNG
（抽帧写、drain 又写），等于一次写换成另一次写；要省必须用流式路径。

### 12.9 `ai_frame_transport`（共享内存环，非默认收益路径）

`ai_frame_transport` 取 `shared_memory` / `auto` / `png`，控制 **PNG 路径内部**
是否把 AI 侧的帧放进共享内存环。

实测结论：

- 只有**单窗口**（`len(window_dirs) == 1`）才会启用；多窗口自动跳过
  （子进程一次跑完所有窗口，父进程无法中途补帧）
- 环容量实测：640×368 约 2111 帧、1280×720 约 539 帧、1920×1080 约 239 帧
- 320×240 上 png 8m07s vs shm 8m58s（shm 慢约 10%）；该结论**不能外推**，
  因为小帧的 PNG 编解码很便宜，ring 的额外一跳被放大
- **净效应不省 SSD 写入**（两端仍是 PNG），故不作为省盘手段

内存不足时自动回退并记 `STAGE3_FRAME_RING_SKIP` / `STAGE3_FRAME_RING_FALLBACK`。

### 12.10 照片链 stage5 的性能陷阱（逐域全图扫描）

**症状**：单张照片跑批在 stage5 卡满 20 分钟（`timeout=1200s`）后超时，
重试 2 次仍失败，最终 rc=87，且**期间零日志输出**——看起来像卡死。

**根因**：`_detect_text_regions()` 对每个连通域做两次**全图**扫描。
AI 放大后的照片是 8000×6000（4800 万像素），实测连通域 **183,300** 个，
单次全图扫描 0.09s → 约 **268 分钟**。stage5 永远跑不完。

**判据（复现用）**：

```python
from scipy import ndimage
lab, n = ndimage.label(stroke_mask)   # n 会到十几万
```

只要 `n` 是五位数以上，任何 `labeled == i` 形式的逐域全图操作都不可行。

**正确写法**：用 `ndimage.find_objects()` 取 bbox，只在局部切片内操作：

```python
for i, sl in enumerate(ndimage.find_objects(labeled), start=1):
    if sl is None:
        continue
    sub = (labeled[sl] == i)      # 只在 bbox 内比较
    n_px = int(sub.sum())
    ...
    text_mask[sl][sub] = 1.0      # 基础切片是 view，能回写
```

**加速比是浮动的，不是固定上限。** 复杂度从 **O(n × N)** 降到
**O(N + Σ bbox面积)**（n = 连通域数，N = 全图像素），所以

```
加速比 ≈ 2n / (1 + Σbbox/N)
```

随连通域数与图像尺寸增长。实测多点（同一台机器，掩码全部逐位一致）：

| 场景 | 连通域 | 旧/次 | 新/次 | 加速比 |
|---|---|---|---|---|
| 240×180 全平 | 0 | 0.00ms | 0.003ms | **≈1×**（无收益） |
| 240×180 高频噪声 | 22 | 1.84ms | 0.058ms | 32× |
| 640×480 噪声 | 177 | 0.09s | — | 36× |
| 1000×750 噪声 | 443 | 0.55s | — | 121× |
| 2000×1500 噪声 | 1,947 | 10.86s | 0.02s | 599× |
| 8000×6000 真实照片 | 183,300 | ≈268 分钟 | 秒级 | 更高 |

要点：
- **小图/少连通域时收益很小甚至为零**（0 域时 ≈1×，只有 `find_objects`
  的固定开销）。所以不能把"680 倍"当成通用预期。
- 真正有意义的是**复杂度量级变化**：旧实现随域数线性爆炸，新实现不随
  域数线性增长 —— 这才是"永远不会跑不完"的保证。
- 真实照片端到端：修复前 1h05m/rc=87/无产物 → 修复后 **6m48s/rc=0**。

**通用规则**：任何"按连通域/按标签循环"的代码，在图像尺寸可能达到数千万
像素时，必须用 `find_objects` 之类的方式把操作限定在局部，禁止在循环体内
写 `arr == label` 或 `np.argwhere(arr == label)`。

### 12.11 照片链发布后的 `.pubbak` 备份清理

**症状**：照片正常跑完后，输出目录里留着一个 `<输出名>.<attempt>.pubbak`
备份目录，装着上一代的完整文件（avif + tif + encode_contract + status），
实测单张 **238 MB**。

**根因**：`_purge_publish_residue()` 只在两条**异常恢复**路径被调用
（上次已完整提交 / 回滚成功）；成功提交写完 `PHASE_COMMITTED` 后
直接 `return 0`，**从不清理**。稳态下恒留一份上一代备份。

**修法**：成功路径也清理，但 `keep_journal=True`：

| 对象 | 处置 | 理由 |
|---|---|---|
| `.pubbak` 里的备份文件 | 删 | 省 238 MB，目录干净 |
| `*.journal.json` | **留** | 同时是审计文件；下次启动靠它识别 `ALREADY_COMMITTED`，删了会重复发布 |

**注意**：清理只用 `os.rmdir`（仅空目录成功）+ 逐个 `os.remove`，
**不用 `rmtree`**。目录里若有 journal 没记录到的文件，整目录保留不动
—— 不猜、不盲删。

**验证**：跑前 `.pubbak` 1 个 / 238M → 跑后 0 个，rc=0。
输出目录最终只剩 avif、tif、journal、status、encode_contract 及各自的
0 字节 `.lock`（锁文件本身，正常保留）。

## 13. 目录名说明

**重要**：项目实际使用的目录名为：
- `video_restoration_pipeline/` — 视频修复流水线
- `photo_restoration_pipeline/` — 照片修复流水线
- `shared_modules/` — 共享模块（两边共用）

所有命令和代码引用必须使用上述实际目录名。


---

## 14. 历史排障记录（本地）

2026-04 及之后的逐文件审查、缺陷根因与修复过程已移出本手册，
保留在本地 `bugs.md` / `shixian.md` / `修复清单.md`（不入库）。
本手册只描述**当前代码**的操作方式与环境要点。
