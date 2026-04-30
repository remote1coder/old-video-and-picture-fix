# 老影像修复自动化脚本套件

## 项目简介

本项目提供**视频修复**和**照片修复**两套完整的自动化流水线，用于修复老旧的影像资料：

- **video_restoration_pipeline**: 处理模拟录像带（VHS/Hi8/Betamax）、DV/MiniDV、HDV/AVCHD
- **photo_restoration_pipeline**: 处理老照片扫描件、早期数码照片

核心修复哲学：**先传统、后 AI、再颗粒**

1. 先用 VapourSynth/FFmpeg/G'MIC 做传统修复（去隔行、去噪、去块、色域转换）
2. 再谨慎使用 AI 超分/人脸修复（仅整数倍、仅低分辨率、仅静态段落）
3. 最后在所有破坏性处理完成后添加颗粒还原胶片质感

## 系统要求

### 硬件
- **CPU**: 8核以上推荐，16核+可获得更好编码效率
- **内存**: 16GB 最低，32GB 推荐
- **GPU**: NVIDIA GTX 1060 6GB+（AI 超分可选，传统修复仅需 CPU）
- **磁盘**: 预留源文件 10-50 倍空间（取决于是否启用 AI 超分）

### 软件依赖

#### 外部工具
```bash
# Ubuntu/Debian 安装示例
sudo apt update
sudo apt install ffmpeg ffprobe vapoursynth avifenc exiftool gmic

# FFmpeg 需要 >= 4.4（支持 bandreject width_type=h）
ffmpeg -version

# G'MIC
sudo apt install gmic

# avifenc (libavif)
sudo apt install libavif-bin
```

#### VapourSynth 插件
```bash
# 通过 vsrepo 安装
python -m vsrepo install qtgmc temporaldegrain2 vivtc deblock_qed knlmeanscl
python -m vsrepo install mvtools2 masktools2 nnedi3 nnedi3cl rgtools tcanny
python -m vsrepo install addgrainc removegrain dftest

# 强制锁定 havsfunc r33
pip install 'havsfunc==33'
```

#### Python 包
```bash
pip install vapoursynth opencv-python numpy pandas pyyaml Pillow requests rich
pip install retina-face ultralytics mtcnn  # 人脸检测（可选）
pip install xxhash  # 快速哈希（可选，回退到 MD5）
```

#### AI 模型权重
```bash
mkdir -p models

# Real-ESRGAN
cd models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x4plus.pth

# GFPGAN
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth

# arnndn 模型（音频降噪 fallback）
wget https://github.com/NullModel/arnndn-models/raw/main/arnndn_50Hz.pb
wget https://github.com/NullModel/arnndn-models/raw/main/arnndn_60Hz.pb
```

## 快速开始

### 四步完整工作流

```bash
# Step 1: 依赖预检
cd video_restoration_pipeline
python -c "import sys; sys.path.insert(0, '../shared_utils'); from dependency_check import main; main()"

# Step 2: 生成清单
python -c "import sys; sys.path.insert(0, '../shared_utils'); from manifest_generate import generate_manifest; generate_manifest(['/mnt/videos'], 'batch_manifest.yaml', 'video')"

# Step 3: 编辑清单（可选）
python -c "import sys; sys.path.insert(0, '../shared_utils'); from manifest_editor import list_jobs; list_jobs('batch_manifest.yaml')"

# Step 4: 批量处理
python main.py --manifest batch_manifest.yaml
```

### 两步简版（自动分类+处理）

```bash
# 视频修复
python video_restoration_pipeline/main.py --input /mnt/videos/family_tapes --output ./output

# 照片修复
python photo_restoration_pipeline/main.py --input /mnt/photos/old_album --output ./output
```

## 配置文件说明

### 全局参数 (global)
```yaml
global:
  output_dir: "./output"          # 输出目录
  temp_dir: "./temp"              # 临时文件目录
  encoder: "svtav1"               # 编码器: svtav1 / x265
  crf: 24                         # 质量参数 (越小越好)
  preset: 4                       # 编码速度/质量平衡
  pix_fmt: "yuv420p10le"          # 输出像素格式
  vs_bit_depth: 16                # VapourSynth 处理位深
  cleanup_intermediates: true     # 清理中间文件
  retry_count: 2                  # 失败重试次数
  no_wsl_redirect: false          # 禁用 WSL 路径重定向
```

### 素材类型参数 (paths)
```yaml
paths:
  analog_tape:
    ivtc: "VIVTC"                 # IVTC 方案
    deinterlace: "QTGMC"          # 去隔行方案
    denoise: "TemporalDegrain2"   # 去噪方案
    target_resolution: "720x480"  # 目标分辨率
  dv_digital:
    deblock: "Deblock_QED"        # DV 去块
  hdv_avchd:
    deblock: "DeblockPP7"         # HDV 去块
    target_resolution: "1920x1080"
```

### AI 参数 (ai)
```yaml
ai:
  enabled: false                  # 默认关闭 AI
  upscale_model: "RealESRGAN_x2plus"  # 超分模型
  upscale_scale: 2                # 放大倍数 (仅支持 2 或 4)
  face_restore: "GFPGAN"          # 人脸修复模型
  force_allow_banned_upscale: false  # 强制允许大尺寸超分
```

### 颗粒合成参数 (grain_factory3)
```yaml
grain_factory3:
  enabled: true
  analog_tape:
    g1str: 1.0                    # 亮度颗粒强度
    g2str: 0.2                    # 色度颗粒强度 (U)
    g3str: 0.2                    # 色度颗粒强度 (V)
  hdv_avchd:
    g1str: 0.8                    # HDV 颗粒稍弱
```

### 音频参数 (audio，视频专用)
```yaml
audio:
  enabled: true
  denoise: "afftdn"               # 全频降噪
  hum_freq: 50                    # 哼鸣频率 (50/60)
  codec: "aac"
  bitrate: "320k"
```

## 命令行参数

### 视频修复
```bash
python main.py --input /path/to/videos [options]

Options:
  --input, -i         输入目录（支持多个）
  --output, -o        输出目录
  --config, -c        配置文件路径 (默认: config.yaml)
  --manifest, -m      使用已有清单
  --preview           预览模式（仅生成预览帧）
  --stage N           仅执行第 N 个 stage
  --job-id ID         仅处理指定 job
  --gpu [auto|cuda|rocm|mps|cpu]  GPU 后端
  --keep-intermediates 保留中间文件
  --no-wsl-redirect   禁用 WSL 路径重定向
```

### 照片修复
```bash
python main.py --input /path/to/photos [options]
# 参数同视频修复
```

## 清单编辑

```bash
# 查看所有任务
python -m shared_utils.manifest_editor list batch_manifest.yaml

# 仅看跳过的任务
python -m shared_utils.manifest_editor list batch_manifest.yaml --status SKIP

# 修改任务状态
python -m shared_utils.manifest_editor set-status batch_manifest.yaml job_001 PENDING

# 设置参数覆盖
python -m shared_utils.manifest_editor set-override batch_manifest.yaml job_001 crf 20
```

## 批量挂机模式

```bash
# 使用 nohup 后台运行
nohup python video_restoration_pipeline/main.py \
    --input /mnt/videos/family_tapes \
    --output /mnt/output/restored_videos \
    > pipeline.log 2>&1 &

# 或使用 screen
tscreen -S restore
python video_restoration_pipeline/main.py --input /mnt/videos --output ./output
# Ctrl+A+D  detach
```

## 常见问题

### Q1: 依赖检查失败（退出码 80）
```bash
# 安装缺失的依赖
sudo apt install ffmpeg vapoursynth gmic libavif-bin exiftool
pip install 'havsfunc==33' pyyaml opencv-python
python -m vsrepo install qtgmc temporaldegrain2 vivtc
```

### Q2: AI 超分被跳过
- 检查清单中 `ai_upscale_banned` 是否为 `true`
- 如果素材尺寸超过阈值（视频 >=1080p, 照片长边 >3840px），AI 会被禁止
- 如需强制超分：设置 `force_allow_banned_upscale: true`（记录 CRITICAL 日志）

### Q3: 音画不同步
- 脚本自动进行 A/V 同步校验
- 若差值 >500ms 会标记 FAILED
- 检查 stage0 提取的原始音频是否完整

### Q4: WSL2 路径问题
- 脚本自动检测 WSL2 并将临时文件重定向到 Linux 原生分区
- 处理完成后自动复制回 `/mnt/` 路径
- 如遇权限问题，加 `--no-wsl-redirect`

### Q5: GPU OOM
- 脚本自动进行显存预检测
- 不足时自动降级: 4x -> 2x -> CPU
- 如需手动指定 GPU: `--gpu cuda` 或 `--gpu cpu`

### Q6: 退出码说明
| 退出码 | 含义 |
|--------|------|
| 0 | 成功 |
| 3 | 跳过（已存在） |
| 80 | 依赖缺失 |
| 81 | 参数错误 |
| 82 | IO 错误 |
| 83 | GPU 错误 |
| 84 | 编码错误 |
| 85 | 权限错误 |
| 86 | 校验失败 |
| 87 | 资源不足 |

## 项目结构

```
video_restoration_pipeline/     # 视频修复套件
  main.py                       # 总控脚本
  config.yaml                   # 配置文件
  classify.py                   # 素材分类器
  stages/
    stage0_audio_extract.py     # 音频提取
    stage1_preprocess.py        # 传统修复（VS 脚本生成）
    stage2_traditional_check.py # 结果校验
    stage3_ai_upscale.py        # AI 超分
    stage4_postprocess.py       # 后处理
    stage5_ai_face.py           # 人脸修复
    stage6_grain_synthesis.py   # 颗粒合成
    stage7_encode.py            # 最终编码+封装

photo_restoration_pipeline/     # 照片修复套件
  main.py                       # 总控脚本
  config.yaml                   # 配置文件
  classify.py                   # 素材分类器
  stages/
    stage1_preprocess.py        # 传统修复
    stage2_traditional_check.py # 结果校验
    stage3_ai_upscale.py        # AI 超分
    stage4_postprocess.py       # 后处理
    stage5_ai_face.py           # 人脸修复
    stage6_encode.py            # 最终编码（AVIF）

shared_utils/                   # 公共工具模块
  dependency_check.py           # 依赖预检
  checkpoint.py                 # 断点续传
  preview.py                    # 预览模式
  logger.py                     # 结构化日志
  audio_utils.py                # 音频处理
  manifest_generate.py          # 清单生成
  manifest_loader.py            # 清单加载
  manifest_editor.py            # 清单编辑

models/                         # AI 模型权重
  RealESRGAN_x2plus.pth
  RealESRGAN_x4plus.pth
  GFPGANv1.4.pth
  DAT_x2.pth
  DAT_x4.pth

architecture.md                 # 架构设计文档
README.md                       # 本文件
```

## 技术要点

### 16bit 处理空间
- 所有 VapourSynth 处理在 16bit（YUV420P16/RGB48）执行
- 最终编码前转为 10bit（YUV420P10）
- 16bit->10bit 在 YUV444 空间执行 error_diffusion 抖动

### 色域转换
- BT.601 -> BT.709 为强制步骤
- 区分 PAL (bt470bg) 和 NTSC (smpte170m)
- 区分 full range 和 limited range

### AI 超分禁令
- 基础禁令：以原始输入文件像素尺寸为永久基准
- 补充禁令：以 AI 步骤输入中间文件像素尺寸为基准
- 满足任一即触发禁令
- 可通过 `force_allow_banned_upscale=true` 覆盖

### 颗粒合成顺序
- 必须在所有降噪/去块/AI 超分**之后**执行
- 视频：GrainFactory3（stage6）
- 照片：G'MIC film_grain（stage6）
- 严禁在 stage1 添加颗粒（会被后续处理抹除）
