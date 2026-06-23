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
```bash
pip install vapoursynth

# 验证
python3 -c "import vapoursynth as vs; print(vs.core)"
```

#### 7) havsfunc r33 (仅视频修复需要)
```bash
pip install "havsfunc==33"

# 验证
python3 -c "import havsfunc; print('havsfunc OK')"
```

### 2.3 Python 依赖

在项目根目录创建虚拟环境：

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

`requirements.txt` 应至少包含：
```
PyYAML>=6.0
Pillow>=9.0
numpy>=1.21
scipy>=1.7
opencv-python>=4.5
pandas
requests
rich>=13.0
vapoursynth>=60
havsfunc==33
```

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
│   ├── main.py                      # 照片主控脚本（单文件 + --input-dir 批量模式；manifest 工作流已支持 --manifest/--resume）
│   ├── config.yaml                  # 照片配置
│   └── stages/                      # 照片处理阶段
│       ├── stage1_preprocess.py     # 预处理（色彩空间、去划痕、去块、去色带）
│       ├── stage2_traditional_check.py  # 质量验证
│       ├── stage3_ai_upscale.py     # AI放大
│       ├── stage4_postprocess.py    # 后处理（区域分离、颗粒合成）
│       ├── stage5_ai_face.py        # 面部修复
│       └── stage6_encode.py         # AVIF最终编码
│
├── shared_modules/                  # 共享模块（两边共用）
│   ├── checkpoint.py                # 断点续传支持
│   ├── dependency_check.py          # 依赖检查
│   ├── logger.py                    # 结构化日志记录
│   ├── audio_utils.py               # 音频分析
│   ├── manifest_generate.py         # 清单生成
│   ├── manifest_editor.py           # 清单编辑
│   ├── manifest_loader.py           # 清单加载
│   └── preview.py                   # 预览工具
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
# 激活虚拟环境
source venv/bin/activate

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
source venv/bin/activate

python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    -i input_video.avi \
    -o output.mkv \
    --ai-enabled \
    --ai-upscale-factor 2 \
    --face-restore GFPGAN
```

参数说明：
- `-c`: 配置文件路径
- `-i`: 输入视频文件
- `-o`: 输出文件路径
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
    -o tape_restored.mkv \
    --ai-enabled \
    --ai-upscale-factor 2
```

#### DV数码带 (DV_DIGITAL)
```bash
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    -i dv.mov \
    -o dv_restored.mkv \
    --ai-enabled \
    --ai-upscale-factor 2
```

#### HDV/AVCHD (HDV_AVCHD)
```bash
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    -i hdv.m2ts \
    -o hdv_restored.mkv \
    --crf 26 \
    --preset 4
```

**注意**: HDV/AVCHD 素材默认 **不建议** AI 放大（原生已 1080p），如需强制放大请设置 `force_allow_banned_upscale: true`。

---

## 7. 照片修复操作

### 7.1 单文件修复（基础）

```bash
source venv/bin/activate

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
- 每任务一个日志文件：`logs/job_{job_id}.log`
- 格式：JSON Lines，每行一条记录

### 8.2 查看日志
```bash
# 查看最新日志
tail -f logs/job_*.log

# 格式化查看某任务日志
python -m json.tool logs/job_photo_1234567890.log
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

## 9. 已知问题与限制

### 9.1 关键运行时缺陷（生产环境使用前必读）

| 问题 | 影响 | 临时规避措施 |
|------|------|-------------|
| ~~照片 main.py 未传入 `original_width/height`~~ | ~~AI 超分 1080p 限制完全失效~~ | ✅ **已修复** |
| ~~照片 CLI `--ai-upscale-factor` 键名不匹配~~ | ~~CLI 指定的放大倍数被忽略~~ | ✅ **已修复** |
| ~~照片 CLI `--ai-model` 键名不匹配~~ | ~~CLI 指定的模型被忽略~~ | ✅ **已修复** |
| ~~checkpoint.py `wsl_copy_back` 使用 copy2~~ | ~~WSL2 下复制到 /mnt/ 可能 PermissionError~~ | ✅ **已修复**：改用 copyfile + 三重校验 |
| ~~checkpoint.py `gpu_memory_check` 用 total_memory~~ | ~~显存预检测不准确，可能运行时 OOM~~ | ✅ **已修复**：改用可用显存查询 |
| ~~照片 stage5 CodeFormer fallback 无操作~~ | ~~CodeFormer CLI 不可用时输出未修复原图~~ | ✅ **已修复**：已实现 CodeFormer Python API 推理（FaceRestoreHelper + torch 推理） |
| ~~照片 stage5 RetinaFace/MTCNN 未过滤低置信度~~ | ~~可能误检非人脸区域~~ | ✅ **已修复**：已正确应用阈值 |
| ~~视频 stage0/stage7 音频文件名不匹配~~ | ~~输出视频无音频~~ | ✅ **已修复**：stage0 统一输出 `.audio`，stage7 同时兼容旧命名 `.wav` fallback |
| ~~视频 stage3 硬编码 25fps~~ | ~~非 25fps 素材帧率错误~~ | ✅ **已修复**：stage3 通过 ffprobe 读取原始 `r_frame_rate` 并用于编码 |
| ~~视频 main.py `force_traditional_first` 未强制~~ | ~~stage2 失败仍进入 AI 阶段~~ | ✅ **已修复**：`process_job()` 中 stage2 非 0/3 时立即返回 86，阻止后续 AI 阶段执行 |
| ~~视频 main.py 缺少 `--resume`~~ | ~~清单无法断点续传~~ | ✅ **已修复**：`--resume` 参数已添加，支持对 PENDING 任务检测有效中间文件并跳过，FAILED 任务校验后降级为 PENDING |

### 9.2 规范偏离（不影响运行但影响质量）

| 问题 | 影响 |
|------|------|
| 照片流水线无批量/清单模式 | 无法断点续传和审阅多文件任务 |
| ~~照片 `--preview` 只打印计划不生成帧~~ | ~~无法预览效果~~ | ✅ **已修复**：预览模块已实现基础功能 |
| I;16 图像在 Stage1/4 被降级到 8bit | 16bit 输入精度丢失 |
| 半身人像色彩匹配未实现 | 面部修复后与身体可能有色彩断层 |
| FLIF 色差层提取是空实现 | 未实现真正的 FLIF 存档 |
| ~~照片重试间隔 2s/4s 而非规范 5s/15s~~ | ~~重试节奏过快~~ | ✅ **已修复**：已改为 5s/15s |

### 9.3 已修复的关键缺陷

以下问题在 2026-04-30 前的修复轮次中已解决：

- ✅ video stage1 QTGMC 逻辑反转（tt/bb 分支无调用）
- ✅ checkpoint.py 缺失 `is_retryable_error()` 和 `verify_ffprobe_decodeable()`
- ✅ video stage5 编码参数硬编码
- ✅ video stage6 `sys.path` 缺少 `shared_modules`
- ✅ `wsl_redirect_path()` 未检查 `platform`
- ✅ photo main.py 未定义退出码 88（已移除）
- ✅ video stage3 目标分辨率 ≤1080p 检查缺失
- ✅ video main.py `_worker_wrapper` 嵌套函数 spawn 下无法 pickle

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
**解决**: 安装插件
```bash
pip install "havsfunc==33"
```

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
~~**原因**: `wsl_redirect_path()` 仅检查 `output_path`，未检查 `temp_dir`。~~ ✅ **已修复**: `wsl_redirect_path()` 现在同时检查 `output_path` 和 `temp_dir`，两者任一在 `/mnt/` 下均自动重定向到 Linux 原生分区。
**兜底解决**: 在 config.yaml 中显式将 `temp_dir` 和 `output_dir` 都设置为 Linux 原生分区路径：
```yaml
global:
  temp_dir: "/home/user/temp"
  output_dir: "/home/user/output"
```
或运行脚本时指定：
```bash
python video_restoration_pipeline/main.py ... --temp-dir /home/user/temp --output-dir /home/user/output
```

#### 问题9：WSL2 复制成品触发 PermissionError
**症状**: 完成修复后 `wsl_copy_back()` 报错 PermissionError  
~~**原因**: 使用 `shutil.copy2` 复制 WSL2 NTFS 挂载文件~~ ✅ **已修复**: 改用 `shutil.copyfile` 并添加三重校验。跨文件系统时自动回退到 copyfile。
```bash
cp /home/user/output/restored.avif /mnt/c/output/
```

#### 问题10：日志文件多进程时损坏
**症状**: 并行处理多个文件后，日志文件中出现截断或交错的 JSON 行  
~~**原因**: Windows 下使用非阻塞文件锁 `LK_NBLCK`~~ ✅ **已修复**: 日志锁已改为 `LK_LOCK` 阻塞模式。多进程运行正常。

#### 问题11：依赖检查通过但运行时模型缺失
**症状**: `dependency_check.py` 全部通过，但 stage5 报 "CodeFormer model not found"  
~~**原因**: `check_ai_models()` 未检查 CodeFormer 权重~~ ✅ **已修复**: 依赖检查已包含 CodeFormer 权重验证。  
~~**注意**: `check_ai_models()` 支持外部 `md5_manifest.json` 清单启用 MD5 校验；内置模型表 MD5 为 `None` 时输出 WARNING 提示用户创建外部 manifest，不阻断执行。~~ ✅ **已修复**: `check_ai_models()` 现在自动检测同目录下的 `md5_manifest.json` 外部 MD5 清单并执行校验。无 MD5 配置时会输出明确警告提示用户配置。

#### 问题12：音频哼鸣未去除
**症状**: 修复后视频仍有 50Hz/60Hz 哼鸣  
~~**原因**: `arnndn` 模型未根据检测到的频率自动选择~~ ✅ **已修复**：已实现频率自动检测和 mono 转换。无需手动配置。

### 10.2 获取帮助

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
- ~~**⚠️ 重要**: `param_hash` 的 `CORE_QUALITY_PARAMS` 白名单缺少部分照片参数~~ ✅ **已修复**: 照片参数已加入白名单。

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

## 12. 目录名说明

**重要**：项目实际使用的目录名为：
- `video_restoration_pipeline/` — 视频修复流水线
- `photo_restoration_pipeline/` — 照片修复流水线
- `shared_modules/` — 共享模块（两边共用）

所有命令和代码引用必须使用上述实际目录名。


---

## 视频流水线排障补充（2026-04-29 审查新增）

### 问题13：视频修复后没有音频 / 音频未被处理
~~**原因**: `stage0_audio_extract.py` 输出文件名为 `{job_id}_{input_name}_audio_raw.wav`，但 `stage7_encode.py` 查找的是 `{job_id}_{input_name}.audio`（无扩展名）。名称不匹配导致 stage7 找不到音频文件。~~ ✅ **已修复**：stage0 输出统一为 `{job_id}_{input_name}.audio`，stage7 同时兼容旧命名 `.wav` 作为 fallback。

### 问题14：AI 超分后视频帧率变为 25fps
~~**原因**: `stage3_ai_upscale.py` 编码输出时硬编码 `-framerate 25`，未读取原始帧率。~~ ✅ **已修复**：stage3 通过 ffprobe 读取原始 `r_frame_rate` 并用于编码。非 25fps 素材（29.97/50/60fps）现已正确保留帧率。

### 问题15：WSL2 下视频修复 I/O 极慢
~~**原因**: `main.py` 未在 stage0-6 中处理 WSL2 路径重定向，临时文件仍在 `/mnt/`（Windows NTFS 挂载）下。只有 stage7 做了重定向。~~ ✅ **已修复**。在 `process_job` 开始时对 `temp_dir` 和 `output_dir` 调用 `wsl_redirect_path()`，所有 stage 统一使用重定向后的路径。

### 问题16：传统修复后仍进入 AI 阶段（即使失败）
~~**原因**: `main.py` 未检查 `force_traditional_first` 参数，stage2 返回非零后仍继续执行 stage3/4/5。~~ ✅ **已修复**：`process_job()` 中 stage2 非 0/3 时立即返回 86，阻止后续 AI 阶段执行。

### 问题17：视频修复清单无法断点续传
~~**原因**: `main.py` 缺少 `--resume` 参数，也没有 FAILED 任务降级逻辑。~~ ✅ **已修复**：`--resume` 参数已添加，支持对 PENDING 任务检测有效中间文件并跳过，FAILED 任务校验后降级为 PENDING。

### 问题18：人脸修复后画面有硬边界
~~**原因**: `stage5_ai_face.py` 将修复后的人脸帧直接复制替换到原视频，未做羽化（feathering）。~~ ✅ **已修复**：使用高斯模糊 mask，margin = max(8px, 5% min_dim) 实现羽化边缘混合。

### 问题19：GrainFactory3 颗粒效果不理想
~~**原因**: `stage6_grain_synthesis.py` 未转换到 16bit 空间，且直接编码为有损 SVT-AV1。~~ ✅ **已修复**：VS 脚本中使用 `vs.YUV420P16`，输出改为 FFV1 无损格式。同时增加 AddGrainC fallback。

### 问题20：多声道视频音频被混缩为立体声
~~**原因**: `stage7_encode.py` 固定使用 AAC 320k，未根据原始声道布局选择 FLAC 或 AAC 640k。~~ ✅ **已修复**：读取原始声道布局，>2 声道时自动切换为 FLAC 以保留完整布局。arnndn 降噪时 mono 转换后也恢复原始布局。


---

## shared_modules 排障补充（2026-04-29 审查新增）

### 问题21：依赖检查通过但运行时 AI 模型缺失
~~**原因**: `dependency_check.py` 的 `check_ai_models()` 未检查 CodeFormer 权重~~ ✅ **已修复**：依赖检查已包含 CodeFormer 权重验证。  
**注意**: `check_ai_models()` 支持外部 `md5_manifest.json` 清单启用 MD5 校验；内置模型表 MD5 为 `None` 时输出 WARNING 提示用户创建外部 manifest，不阻断执行。

### 问题22：WSL2 下成品文件复制失败 PermissionError
~~**原因**: `checkpoint.py` 的 `wsl_copy_back()` 使用 `shutil.copy2`~~ ✅ **已修复**：改用 `shutil.copyfile()` + `os.utime()` 手动同步时间戳。跨文件系统时自动回退到 copyfile。

### 问题23：Windows 下多进程日志文件损坏
~~**原因**: `logger.py` 使用 `msvcrt.LK_NBLCK` 非阻塞锁~~ ✅ **已修复**：已改为 `msvcrt.LK_LOCK` 阻塞锁。多进程运行正常。

### 问题24：音频处理后输出为 FLAC 而非 AAC
~~**原因**: `audio_utils.py` 的 `process_audio()` 默认 codec 为 `"flac"`~~ ✅ **已修复**：默认 codec 已改为 `"aac"`。

### 问题25：断点续传时修改照片参数未触发重新处理
~~**原因**: `checkpoint.py` 的 `CORE_QUALITY_PARAMS` 缺少照片专属参数~~ ✅ **已修复**：照片参数已加入 `CORE_QUALITY_PARAMS` 白名单。  
**注意**: 仍缺少少量参数（如 `enable_channel_correction`），修改这些参数不会触发 param_hash 更新。

### 问题26：清单编辑器无法交互式审阅
**原因**: `manifest_editor.py` 仅提供命令行子命令，无交互式 TUI。
**解决**: 使用 `rich` 库实现交互式审阅，或手动编辑 YAML 清单文件。

### 问题27：中间文件清理未生效（per-stage 清理）
**原因**: `checkpoint.py` 的 `cleanup_old_intermediates()` 默认 `current_step=0`，清理条件 `step < 0` 永远为 False。若调用方未显式传入 `current_step`，函数不会删除任何旧中间文件。
**解决**: 调用时显式传入当前 stage 编号，或将默认值改为 `sys.maxsize`。
**状态**: ✅ **已修复** — 默认值改为 `Optional[int] = None`，当为 `None` 时保留 `keep_steps` 个最新文件并删除其余所有（BUG-058）

### 问题28：多声道素材声道布局变体识别失败
**原因**: `audio_utils.py` 的 `get_audio_channel_layout()` 使用精确字符串匹配 `("5.1", "6.1", "7.1")`。ffprobe 可能返回 `5.1(side)`、`7.1(wide)` 等变体，导致这些素材未被识别为多声道，继续编码为 AAC 而非 FLAC。
**解决**: 改用 `channels >= 6` 判断多声道，或扩展正则匹配支持常见变体。
**状态**: ✅ **已修复** — `process_audio()` 改用 `channels >= 6` 判断多声道，不再依赖 `channel_layout` 精确字符串匹配（BUG-059）

### 问题29：FAILED 任务降级时 ImportError
**原因**: `manifest_loader.py` 的 `_handle_failed_job()` 通过 `sys.path.insert` 假设 `shared_modules` 父目录为项目根。若项目结构变化，导入 `shared_modules.checkpoint` 会失败，FAILED 降级逻辑失效。
**解决**: 使用 `importlib.util` 动态加载，或内联所需验证函数。
**状态**: ✅ **已修复** — `sys.path.insert` 目标改为 `shared_modules/` 自身，使用 `from checkpoint import ...` 直接同目录导入（BUG-060）

### 问题30：运动预览临时目录 PermissionError
**原因**: `preview.py` 的 `detect_motion_segments()` 在输入文件所在目录创建 `_motion_temp`。若该目录只读（如光盘挂载点），触发 PermissionError。
**解决**: 使用系统临时目录 `tempfile.mkdtemp(prefix="motion_")`。
**状态**: ✅ **已修复** — 改用 `tempfile.mkdtemp(prefix="motion_")` 创建系统临时目录（BUG-061）

### 问题31：GMIC 滤镜可用性误报
**原因**: `dependency_check.py` 的 `check_gmic_filters()` 在 `gmic -h` 输出中做子串匹配（如 `-repair` 可能匹配 `unrepair`）。
**解决**: 使用 `gmic -h repair` 精确查询单个滤镜。
**状态**: ✅ **已修复** — 对每个滤镜单独调用 `gmic -h <filtername>`，通过返回码精确判断（BUG-062）


---

## 视频流水线排障补充（2026-04-29 全面审查新增）

### 问题27：AI 阶段永不执行（即使 config.yaml 中启用了 AI）
~~**原因**: `main.py` 的 `merge_params()` 将 `config.ai.enabled` 注入为 flat key `"enabled"`，但 `process_job()` 和 stage3/stage5 查找的是 `"ai_enabled"`。由于 CLI 没有 `--ai-enabled` 参数，`params.get("ai_enabled", False)` 永远返回 `False`，导致 AI upscale 和人脸修复阶段永远不会被调度。~~ ✅ **已修复**：`merge_params()` 现在与 `audio_config` 处理方式一致，同步注入 `ai_{k}` 前缀版本（如 `ai_enabled`）。

### 问题28：人脸修复片段提取时间戳偏移（非 25fps 素材）
**原因**: `stage5_ai_face.py` 使用 `ffmpeg -ss {start_frame / 25.0}` 提取人脸片段，硬编码假设视频为 25fps。对于 29.97fps/50fps/60fps 素材，提取的时间戳会偏移，导致修复区域错位。
**解决**: 使用 `ffprobe` 读取实际帧率进行转换，或直接使用 `_get_frame_pts()` 的原始 PTS 数据计算时间戳。
**状态**: ✅ **已修复** — 已读取 `r_frame_rate` 并解析分数格式（如 30000/1001）（BUG-038）

### 问题29：人脸修复输出文件名缺少 param_hash
**原因**: `stage5_ai_face.py` 输出文件名为 `_step5_face_p{job_id}.mkv`，使用了 `job_id` 而非 `param_hash`。其他阶段（step1, step3 等）均使用 param_hash 实现 deterministic intermediate matching。缺少 param_hash 意味着仅调整 face_restore 相关参数时，文件名不变，可能被错误复用旧输出。
**解决**: 从 params 读取 `param_hash` 并嵌入文件名，改为 `_step5_face_p{param_hash}.mkv`。
**状态**: ✅ **已修复** — 输出文件名已改为 `_step5_face_p{param_hash}.mkv`（BUG-039）

### 问题30：最终编码 SVT-AV1 参数字符串冗余构建
**原因**: `stage7_encode.py` 中 `svtav1-params` 被构建两次：第一次在 `video_cmd` 初始化时仅含 tile 参数，第二次在颜色元数据确定后重新构建完整字符串并原地替换。功能正确但代码冗余。
**解决**: 延迟构建 `svt_params`，在所有参数确定后一次性追加到 `video_cmd`。
**状态**: ✅ **已修复** — 延迟构建 `svt_params`，所有参数确定后一次性 extend 到 `video_cmd`（BUG-041）

### 问题31：spawn 模式下 worker 参数序列化风险
**原因**: `main.py` 的 `_worker_wrapper` task dict 包含 `argparse.Namespace` 对象。虽然标准 Namespace 可被 pickle，但不够健壮，若被扩展可能引发序列化失败。
**解决**: 将 `args` 转为纯 dict (`vars(args)`) 后再序列化。
**状态**: ✅ **已修复** — task_contexts 中已传入纯 dict，但 `process_job()` 中仍使用 `args.config` / `args.stage`（AttributeError）。详见问题32。

### 问题32：并行处理（`--workers 2`）时立即崩溃 AttributeError
**原因**: `main.py` 的 `process_job()` 中直接使用 `args.config` 和 `args.stage`（Namespace 属性访问）。在 spawn 子进程中，`args` 已被转换为纯 dict，导致 `AttributeError: 'dict' object has no attribute 'config'`。
**症状**: 运行 `python video_restoration_pipeline/main.py --workers 2 ...` 后立即崩溃，单 worker（默认）正常。
**解决**: 
1. `process_job()` 开头添加 `if not isinstance(args, dict): args = vars(args)`
2. 将所有 `args.xxx` 访问改为 `args.get("xxx")`
3. `run_stage()` 接收 `config_path` 参数而非从 `args` 读取
**状态**: ✅ **已修复** — `process_job()` 已添加 `isinstance(args, dict)` 归一化，所有属性访问改为 `.get()`（BUG-042）

### 问题33：断点续传（`--resume`）跳过未完成的任务
**原因**: Resume 模式下，只要 PENDING 任务存在任意 stepN（如 step1）的有效中间文件，即被标记为 DONE。这导致仅完成了早期 stage 的任务被误认为全部完成。
**症状**: 中断在 stage 3 的任务，重新运行 `--resume` 后状态变为 DONE，但输出视频不存在或只完成了预处理。
**解决**: 逆向检查中间文件：从 step7 往下找第一个存在的中间文件。只有 step7 存在且有效时才标记 DONE；否则根据最高有效 stage 设置状态并从断点继续。
**状态**: ✅ **已修复** — resume 逻辑仅当 step7 输出有效时标记 DONE；否则从最高有效 stage 恢复（BUG-043）

### 问题34：视频分类后超时计算不准确
**原因**: `classify_video()` 未返回 `duration` 字段，导致 `main.py` 的 `_calculate_timeout()` 和 `_check_disk_space()` 使用默认值 0。超时仅为基础值 600 秒（10 分钟），对大文件可能不足。
**解决**: 在 `classify_video()` 中通过 ffprobe 读取 `format.duration` 并加入返回结果。
**状态**: ✅ **已修复** — `classify_video()` 已从 ffprobe format 读取 `duration` 并加入返回结果（BUG-045）

