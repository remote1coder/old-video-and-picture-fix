# 项目操作手册

## 1. 项目简介

本项目是**视频与照片混合修复流水线**，支持：
- **视频修复**：老录像带、DV数码带、HDV/AVCHD高清带的去噪、去隔行、AI放大、面部修复、颗粒合成
- **照片修复**：老照片扫描件、数码照片的去噪、去划痕、AI放大、面部修复

流水线采用**传统信号处理 + AI超分**混合架构，先通过QTGMC等传统方法去噪去隔行，再用RealESRGAN/DAT进行AI放大，最后可选GFPGAN/CodeFormer面部修复。

---

## 2. 环境准备

### 2.1 操作系统
- **推荐**：Linux (Ubuntu 20.04/22.04)
- **支持**：Windows 10/11 (需WSL2)、macOS (Intel/Apple Silicon)
- **注意**：本项目使用FFmpeg、VapourSynth等工具，Linux环境最稳定

### 2.2 必需软件

按顺序安装：

#### 1) Python 3.9+
```bash
# Ubuntu
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
# Ubuntu
sudo apt install gmic

# 或从官网下载最新版
# https://gmic.eu/download.html
```

#### 5) VapourSynth (仅视频修复需要)
```bash
# Ubuntu
sudo apt install vapoursynth-python3

# 或 pip 安装
pip install vapoursynth

# 验证
python3 -c "import vapoursynth; print(vs.core)"
```

#### 6) havsfunc r33 (仅视频修复需要)
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

# 激活
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安装依赖
pip install -r requirements.txt
```

`requirements.txt` 内容：
```
PyYAML>=6.0
Pillow>=9.0
numpy>=1.21
scipy>=1.7
cv2
pandas
requests
rich>=13.0
vapoursynth>=60
havsfunc==33
realesrgan-ncnn-vulkan
gfpgan
```

### 2.4 AI 模型下载

在项目根目录创建 `models/` 文件夹，下载以下模型：

```bash
mkdir models
cd models

# RealESRGAN (AI放大)
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.0/RealESRGAN_x2plus.pth
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.0/RealESRGAN_x4plus.pth

# GFPGAN (面部修复)
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth

# DAT (AI放大备选)
# 从 https://github.com/ckkelvinchan/DAT/releases 下载 DAT_x2plus.pth 和 DAT_x4plus.pth

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
│       ├── stage1_preprocess.py     # 预处理（去噪、去隔行）
│       ├── stage2_traditional_check.py # 质量验证
│       ├── stage3_ai_upscale.py     # AI放大
│       ├── stage4_postprocess.py    # 后处理
│       ├── stage5_ai_face.py        # 面部修复
│       ├── stage6_grain_synthesis.py # 颗粒合成
│       └── stage7_encode.py         # 最终编码
│
├── photo_restoration_pipeline/      # 照片修复流水线
│   ├── classify.py                  # 照片分类器
│   ├── main.py                      # 照片主控脚本
│   ├── config.yaml                  # 照片配置
│   └── stages/                      # 照片处理阶段
│       ├── stage1_preprocess.py     # 预处理
│       ├── stage2_traditional_check.py # 质量验证
│       ├── stage3_ai_upscale.py     # AI放大
│       ├── stage4_postprocess.py    # 后处理
│       ├── stage5_ai_face.py        # 面部修复
│       └── stage6_encode.py        # 最终编码
│
├── shared_modules/                  # 共享模块（两边共用）
│   ├── checkpoint.py                # 断点续传支持
│   ├── dependency_check.py          # 依赖检查
│   ├── logger.py                    # 日志记录
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

---

## 5. 配置文件

### 5.1 视频配置 (video_restoration_pipeline/config.yaml)

关键配置项：

```yaml
global:
  crf: 24                          # 视频质量 (越小越好，文件越大)
  preset: 4                        # 编码速度 (0-8, 越小越慢质量越好)
  svt_tile_rows: 2                 # SVT-AV1编码平铺行数
  svt_tile_cols: 2                 # SVT-AV1编码平铺列数
  parallel_workers: 4              # 并行任务数
  cleanup_intermediates: true      # 完成后清理中间文件
  retry_count: 2                   # 失败重试次数
  log_dir: "./logs"                # 日志目录
  temp_dir: "./temp"               # 中间文件目录
  output_dir: "./output"           # 输出目录

ai:
  enabled: true                    # 是否启用AI放大
  upscale_factor: 2                # 放大倍数 (2或4)
  model: "RealESRGAN_x2plus"       # AI模型名
  denoise_strength: 0.5            # 去噪强度
  face_restore: "GFPGAN"           # 面部修复模型 (GFPGAN或CodeFormer)

audio:
  enabled: true
  hum:
    enabled: true
    frequency: 50                  # 电源哼声频率 (50或60)
    harmonics: 3                   # 谐波数量
    dual: false                    # 双频哼声
```

### 5.2 照片配置 (photo_restoration_pipeline/config.yaml)

关键配置项：

```yaml
global:
  quality: 90                      # JPEG输出质量
  parallel_workers: 2
  log_dir: "./logs"
  temp_dir: "./temp"
  output_dir: "./output"

ai:
  enabled: true
  upscale_factor: 2
  model: "RealESRGAN_x2plus"
  denoise_strength: 0.5
  face_restore: ""
  face_restore_strength: 0.8
```

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
python shared_modules/manifest_generate.py \
    video_restoration_pipeline/config.yaml \
    ./batch_manifest.yaml \
    ./video_folder \
    --pipeline-type video
```

#### 步骤2：编辑清单（可选）
```python
from shared_modules.manifest_editor import ManifestEditor

editor = ManifestEditor("./batch_manifest.yaml")
editor.change_status("job_001", "SKIP")  # 跳过某个任务
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
    --deinterlace-preset "Slow" \
    --border-size 10 \
    --border-threshold 15
```

#### DV数码带 (DV_DIGITAL)
```bash
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    -i dv.mov \
    -o dv_restored.mkv \
    --ai-enabled
```

#### HDV/AVCHD (HDV_AVCHD)
```bash
python video_restoration_pipeline/main.py \
    -c video_restoration_pipeline/config.yaml \
    -i hdv.m2ts \
    -o hdv_restored.mkv \
    --ai-enabled \
    --crf 26 \
    --preset "Slower"
```

---

## 7. 照片修复操作

### 7.1 单文件修复

```bash
source venv/bin/activate

python photo_restoration_pipeline/main.py \
    -c photo_restoration_pipeline/config.yaml \
    -i old_photo.jpg \
    -o restored_photo.jpg \
    --ai-enabled \
    --ai-upscale-factor 2
```

### 7.2 批量修复

照片批量修复与视频类似，使用清单：

```bash
# 生成清单
python shared_modules/manifest_generate.py \
    photo_restoration_pipeline/config.yaml \
    ./photo_manifest.yaml \
    ./photo_folder \
    --pipeline-type photo

# 批量运行
python photo_restoration_pipeline/main.py \
    -c photo_restoration_pipeline/config.yaml \
    -m ./photo_manifest.yaml
```

---

## 8. 日志查看

### 8.1 日志位置
- 每任务一个日志文件：`logs/job_{job_id}.log`
- 格式：JSON Lines，每行一条记录

### 8.2 查看日志
```bash
# 查看最新日志
tail -f logs/job_*.log

# 搜索错误
python shared_modules/preview.py --search ERROR --log-dir ./logs
```

### 8.3 日志字段
```json
{
  "timestamp": "2024-01-01T12:00:00+00:00",
  "level": "INFO",
  "event": "STAGE_START",
  "message": "Starting stage 1",
  "job_id": "job_123"
}
```

---

## 9. 已知问题与限制

⚠️ **以下问题已通过代码审查识别，将在后续版本中修复。生产环境使用前请仔细阅读。**

### 9.1 关键运行时缺陷

| 问题 | 影响 | 临时规避措施 |
|------|------|-------------|
| `video_stage1_preprocess.py` 中 `haf.QTGMC()` 调用缺失 (`tt`/`bb` 场序) | 交错视频不会被去隔行，输出仍含梳状伪影 | 手动编辑生成的 `.vpy` 脚本，添加 `haf.QTGMC()` 调用 |
| `shared_modules/checkpoint.py` 缺少 `is_retryable_error()` | 批量任务在失败后无法自动重试 | 手动重新运行命令 |
| `shared_modules/checkpoint.py` 缺少 `verify_ffprobe_decodeable()` | FAILED 任务恢复时崩溃 | 手动删除对应中间文件重新运行 |
| `video_restoration_pipeline/main.py` `_worker_wrapper` 为嵌套函数 | Windows/macOS 多进程 `spawn` 启动失败 | 仅在 Linux 使用 `fork` 模式运行 |
| `video_restoration_pipeline/stages/stage6_grain_synthesis.py` `sys.path` 缺少 `shared_modules` | 运行时导入 logger 失败 | 手动添加 `sys.path` 或避免使用 grain synthesis |

### 9.2 规范偏离（不影响运行但影响质量）

| 问题 | 影响 |
|------|------|
| `video_stage3_ai_upscale.py` 硬编码 `-framerate 25` | 非 25fps 素材（如 29.97fps NTSC）输出帧率错误 |
| `video_stage7_encode.py` WSL copyback 失败时返回 82 | Linux 端文件可能仍保留，但 Windows 端无文件 |
| `video_stage5_ai_face.py` 编码参数硬编码 | 忽略 config 中的 CRF/preset 设置 |
| `shared_modules/preview.py` 预览片段时长 10 秒 | 与规范要求的 0.5 秒相差 20 倍，文件过大 |
| `photo_stage3_ai_upscale.py` `_resolve_input_source` 忽略 ban 参数 | AI 超分禁令被绕过，可能意外放大高分辨率素材 |
| `photo_stage1_preprocess.py` G'MIC 失败静默 | 去划痕等操作未执行也不会报错 |

---

## 10. 故障排除

### 10.1 常见问题

#### 问题1：FFmpeg版本过低
**症状**：bandreject width_type=h 不支持  
**解决**：升级 FFmpeg 到 4.4+
```bash
sudo apt install ffmpeg  # Ubuntu 22.04+ 自带 4.4+
```

#### 问题2：VapourSynth插件缺失
**症状**：stage1_preprocess.py 报错 "VS Plugin QTGMC not available"  
**解决**：安装插件
```bash
pip install vapoursynth
# 或手动安装插件到 VapourSynth 插件目录
```

#### 问题3：AI模型文件缺失
**症状**：stage3_ai_upscale.py 报错 "Model file not found"  
**解决**：下载模型到 `models/` 目录
```bash
# 查看缺失的模型
python shared_modules/dependency_check.py -c config.yaml
```

#### 问题4：内存不足
**症状**：AI放大阶段OOM  
**解决**：
- 减小并行任务数：`--parallel-workers 1`
- 降低AI放大倍数：`--ai-upscale-factor 2` (不用4)
- 使用CPU模式（较慢但省内存）

#### 问题5：磁盘空间不足
**症状**：中间文件写入失败  
**解决**：
- 清理temp目录
- 增加磁盘空间
- 关闭中间文件保留：`--no-cleanup` 的反面（默认开启清理）

### 9.2 获取帮助

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
1. 计算param_hash匹配现有中间文件
2. 跳过已完成的stage
3. 从断点继续

**限制**：当前断点续传存在以下限制：
- FAILED 任务恢复依赖 `verify_ffprobe_decodeable()`（该函数未实现），恢复时可能崩溃
- 建议 FAILED 任务直接删除对应中间文件后重新运行

### 11.2 预览模式

在正式修复前预览效果：

```bash
python video_restoration_pipeline/main.py \
    -c config.yaml -i input.avi \
    --preview 3  # 预览前3帧
```

**注意**：当前预览输出 PNG 帧和 H.264 片段（约 10 秒），与规范要求的 0.5 秒 WebP 动画有差距，预览文件可能较大。

### 11.3 自定义VapourSynth滤镜

在 `video_restoration_pipeline/config.yaml` 的 `paths` 段添加：

```yaml
paths:
  analog_tape:
    vs_filters: "my_custom_filter.vpy"
```

### 11.4 参数层叠示例

```bash
# CLI参数最高优先级，覆盖配置文件
python video_restoration_pipeline/main.py \
    -c config.yaml \
    -i input.avi \
    --crf 18 \              # 覆盖global.crf
    --ai-upscale-factor 4 \ # 覆盖ai.upscale_factor
    --face-restore CodeFormer  # 覆盖ai.face_restore
```

---

## 11. 目录名说明

**重要**：项目实际使用的目录名为：
- `video_restoration_pipeline/` — 视频修复流水线
- `photo_restoration_pipeline/` — 照片修复流水线
- `shared_modules/` — 共享模块（两边共用）

所有命令和代码引用必须使用上述实际目录名。
