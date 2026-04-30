# 老影像修复自动化脚本套件 - 详细说明文档

## 1. 项目概述

本项目是一个专业的老旧影像资料自动化修复系统，采用**"先传统、后 AI、再颗粒"**的修复哲学，提供视频和照片两套独立的修复流水线。

### 1.1 核心修复哲学

1. **先传统**：使用 VapourSynth/FFmpeg/G'MIC 进行传统修复（去隔行、去噪、去块、色域转换、稳定）
2. **后 AI**：谨慎使用 AI 超分和人脸修复（仅整数倍放大、仅低分辨率素材、仅静态面部段落）
3. **再颗粒**：所有破坏性处理完成后，添加胶片颗粒还原质感

### 1.2 支持的素材类型

**视频修复 (video_restoration_pipeline)**
- 模拟录像带：VHS、Hi8、Betamax（720x480/576，场序）
- DV/MiniDV 数码录像（720x480/576，4:1:1/4:2:0）
- HDV/AVCHD 高清录像（1440x1080/1920x1080，MPEG-2/H.264）

**照片修复 (photo_restoration_pipeline)**
- 老照片扫描件（高分辨率，可能有划痕、褪色、色带）
- 早期数码照片（低分辨率，可能有压缩块、色带）

### 1.3 项目产出文件清单

| 文件 | 类型 | 说明 |
|------|------|------|
| architecture.md | 文档 | 架构设计文档 |
| README.md | 文档 | 快速使用手册 |
| PROJECT_MANUAL.md | 文档 | 本详细说明文档 |
| video_restoration_pipeline/config.yaml | 配置 | 视频套件配置 |
| video_restoration_pipeline/main.py | 脚本 | 视频套件总控 |
| video_restoration_pipeline/classify.py | 脚本 | 视频素材分类器 |
| video_restoration_pipeline/stages/stage0-7 | 脚本 | 视频处理各阶段 |
| photo_restoration_pipeline/config.yaml | 配置 | 照片套件配置 |
| photo_restoration_pipeline/main.py | 脚本 | 照片套件总控 |
| photo_restoration_pipeline/classify.py | 脚本 | 照片素材分类器 |
| photo_restoration_pipeline/stages/stage1-6 | 脚本 | 照片处理各阶段 |
| shared_utils/dependency_check.py | 脚本 | 依赖预检 |
| shared_utils/checkpoint.py | 脚本 | 断点续传 |
| shared_utils/preview.py | 脚本 | 预览模式 |
| shared_utils/logger.py | 脚本 | 结构化日志 |
| shared_utils/audio_utils.py | 脚本 | 音频处理 |
| shared_utils/manifest_generate.py | 脚本 | 清单生成 |
| shared_utils/manifest_loader.py | 脚本 | 清单加载 |
| shared_utils/manifest_editor.py | 脚本 | 清单编辑器 |

## 2. 系统架构详解

### 2.1 模块关系图

```
                    +-------------------+
                    | dependency_check  |  (启动时预检)
                    +---------+---------+
                              |
                              v
+---------------+    +--------+--------+    +---------------+
| config.yaml   |--->|    main.py      |--->| manifest_generate.py / manifest_loader.py |
| (参数配置)     |    |   (总控调度)     |    | (清单管理)     |
+---------------+    +--------+--------+    +---------------+
                              |
            +-----------------+-----------------+
            |                                   |
            v                                   v
   +--------+---------+               +--------+--------+
   | classify.py      |               | checkpoint.py   |
   | (素材分类)        |               | (断点续传)       |
   +--------+---------+               +--------+--------+
            |                                   |
            v                                   v
   +--------+---------+               +--------+--------+
   | stages/stage*.py |               | preview.py      |
   | (分步处理)        |               | (预览模式)       |
   +------------------+               +-----------------+
```

### 2.2 数据流图

#### 视频修复数据流

```
输入视频文件
    |
    v
classify.py (分类: 模拟带/DV/HDV, 场序检测, PSF检测)
    |
    v
manifest_generate.py (生成作业清单, AI禁令标记)
    |
    v
stage0_audio_extract.py (提取原始音频, 暂存)
    |
    v
stage1_preprocess.py (IVTC/去隔行/去噪/稳定/色域转换)
    |
    v
stage2_traditional_check.py (传统修复结果校验)
    |
    v
[AI启用?] --否--> stage6_grain_synthesis.py
    |
    是
    v
stage3_ai_upscale.py (AI超分, GPU显存预检测)
    |
    v
stage4_postprocess.py (Lanczos缩放/局部遮罩处理)
    |
    v
stage5_ai_face.py (AI人脸修复, 静态段落判定)
    |
    v
stage6_grain_synthesis.py (GrainFactory3颗粒合成)
    |
    v
stage7_encode.py (SVT-AV1编码+音频处理+封装)
    |
    v
输出 MKV 文件
```

#### 照片修复数据流

```
输入照片文件
    |
    v
classify.py (分类: 扫描件/数码照片, DPI检测)
    |
    v
manifest_generate.py (生成作业清单, AI禁令标记)
    |
    v
stage1_preprocess.py (去划痕/褪色还原/去色带)
    |
    v
stage2_traditional_check.py (传统修复结果校验)
    |
    v
[AI启用?] --否--> stage6_encode.py
    |
    是
    v
stage3_ai_upscale.py (AI超分, GPU显存预检测)
    |
    v
stage4_postprocess.py (Lanczos缩放/局部遮罩处理)
    |
    v
stage5_ai_face.py (AI人脸/全身修复)
    |
    v
stage6_encode.py (AVIF编码输出, G'MIC颗粒合成)
    |
    v
输出 AVIF/TIFF 文件
```

## 3. 各模块功能详解

### 3.1 公共工具模块 (shared_utils/)

#### dependency_check.py - 依赖预检
**功能**：启动时检查所有外部依赖是否就绪。  
**检查项**：
- 外部工具：ffmpeg (>=4.4)、ffprobe、vapourSynth、avifenc、exiftool、gmic
- FFmpeg 版本号（必须 >= 4.4 才能使用 `bandreject width_type=h`）
- FFmpeg 滤镜：afftdn、bandreject、arnndn
- VapourSynth 插件（完整列表）：QTGMC、TemporalDegrain2、VIVTC、Deblock_QED、KNLMeansCL、GrainFactory3、MVTools2、MaskTools2、NNEDI3/CL、RgTools 等
- havsfunc 版本（强制 r33）
- 最小参数集测试（生成 BlankClip 执行核心滤镜测试）
- Python 包：vapoursynth、cv2、numpy、pandas、yaml、PIL、requests、rich
- AI 模型权重文件（RealESRGAN、GFPGAN、DAT）
- arnndn 音频降噪模型

**退出码**：发现缺失依赖时退出码 80。

#### checkpoint.py - 断点续传
**核心功能**：
- **中间文件命名**：`{job_id}_{safe_name}_step{N}_{operation}_p{param_hash}.{ext}`
- **param_hash 计算**：SHA-256，基于文件标识（大小+mtime+xxhash64前16位）+ 核心画质参数（去噪强度、QTGMC preset、IVTC模式、色域矩阵、超分倍数、AI模型名、颗粒强度、色深、CRF/preset等）
- **帧数据完整性校验**：开头(0-5%)、中间(45-55%)、结尾(95-100%)三区间采样 + 时长>10分钟每10分钟额外采样
- **旧中间文件清理**：保留最近2个 stage 的中间文件
- **WSL2 路径重定向**：临时文件自动重定向到 Linux 原生分区，处理完成后回迁
- **错误分类**：判断错误是否可重试（GPU OOM 可重试，磁盘满不可重试）

#### logger.py - 结构化日志
**核心功能**：
- 每个 job 独立的日志文件
- JSON Lines 格式，多进程安全（fcntl 文件锁）
- 字段截断保护（防止单行超过1MB导致 JSON 解析失败）
- 参数覆盖事件记录（param_override 方法）
- ISO-8601 带时区时间戳

#### audio_utils.py - 音频处理
**核心功能**：
- 音频提取（无损复制）
- 哼鸣频率自动检测（50Hz/60Hz，FFT 能量比分析）
- 哼鸣滤波链构建（bandreject 基频+谐波，width_type=h）
- 全频降噪（afftdn / arnndn fallback）
- 音画同步校验（差值 <100ms OK, <300ms WARNING, >500ms ERROR）
- 音频变速（atempo 链式，支持 0.5-100x）

#### preview.py - 预览模式
**核心功能**：
- 均匀位置抽取6帧预览图
- 生成10秒预览片段（缩小到640px宽）

### 3.2 清单系统 (shared_utils/manifest_*.py)

#### generate.py - 清单生成
- 扫描输入目录，递归查找所有支持格式的文件
- 自动分类每个文件（ANALOG_TAPE/DV_DIGITAL/HDV_AVCHD 或 SCAN_PRINT/DIGITAL_PHOTO）
- 计算 AI 超分禁令标记
- 生成 YAML 或 JSON Lines 格式清单

#### loader.py - 清单加载
- 支持 YAML 和 JSON Lines 格式
- 按状态过滤（默认 PENDING + FAILED）
- FAILED 任务自动检查中间文件完整性（有效则降级为 PENDING，无效则删除）
- 按优先级排序

#### editor.py - 清单编辑器
- `list`: 列出任务（可选状态过滤）
- `set-status`: 修改任务状态
- `set-override`: 设置参数覆盖

### 3.3 视频套件 Stage 详解

| Stage | 脚本 | 核心功能 | 关键技术 |
|-------|------|----------|----------|
| 0 | stage0_audio_extract.py | 提取原始音频流 | `ffmpeg -vn -acodec copy` |
| 1 | stage1_preprocess.py | 传统修复（IVTC/去隔行/去噪/稳定/色域转换） | 生成 VapourSynth 脚本，16bit YUV420P16 处理 |
| 2 | stage2_traditional_check.py | 传统修复结果校验 | ffprobe 解码测试 + 帧数据完整性校验 |
| 3 | stage3_ai_upscale.py | AI 超分（可选） | Real-ESRGAN/DAT，仅2x/4x，GPU 显存预检测 |
| 4 | stage4_postprocess.py | 超分后处理 | Lanczos 缩放、16bit->10bit error diffusion |
| 5 | stage5_ai_face.py | AI 人脸修复（可选） | GFPGAN，仅静态段落（>=2s，置信度>=0.8） |
| 6 | stage6_grain_synthesis.py | 颗粒合成 | GrainFactory3（仅 g1str/g2str/g3str） |
| 7 | stage7_encode.py | 编码封装 | SVT-AV1 10bit，音频哼鸣去除+降噪，MKV 封装 |

### 3.4 照片套件 Stage 详解

| Stage | 脚本 | 核心功能 | 关键技术 |
|-------|------|----------|----------|
| 1 | stage1_preprocess.py | 传统修复 | G'MIC 去划痕/去块/去色带，单通道退化检测 |
| 2 | stage2_traditional_check.py | 结果校验 | 文件完整性 + PIL 可打开验证 |
| 3 | stage3_ai_upscale.py | AI 超分 | Real-ESRGAN，仅2x/4x |
| 4 | stage4_postprocess.py | 后处理 | Lanczos 缩放，局部遮罩 |
| 5 | stage5_ai_face.py | 人脸修复 | GFPGAN（照片无时间限制，全帧处理） |
| 6 | stage6_encode.py | 最终编码 | G'MIC 颗粒合成 + avifenc（10bit YUV420 或 16bit RGB444） |

## 4. 配置系统详解

### 4.1 参数优先级（从高到低）

1. 命令行参数（CLI flags）
2. user_overrides（清单中单个文件的覆盖参数）
3. paths.{detected_class}（素材类型专属参数）
4. global（全局默认参数）
5. 脚本内置默认值

### 4.2 关键配置项

#### grain_factory3（视频颗粒合成）
```yaml
grain_factory3:
  enabled: true
  analog_tape:
    g1str: 1.0    # 亮度颗粒强度
    g2str: 0.2    # U色度颗粒强度
    g3str: 0.2    # V色度颗粒强度
  hdv_avchd:
    g1str: 0.8    # HDV 颗粒稍弱
```

**重要约束**：GrainFactory3 仅允许传入 `g1str/g2str/g3str` 三个参数，严禁传入 `taps/shutter/correlation` 等未经验证的参数。

#### ai（AI 超分与人脸修复）
```yaml
ai:
  enabled: false                    # 默认关闭，需手动开启
  upscale_model: "RealESRGAN_x2plus" # 超分模型
  upscale_scale: 2                  # 仅支持 2 或 4
  face_restore: "GFPGAN"            # 人脸修复模型
  ban_original_input_pixels: true   # 以原始输入像素为基准判断禁令
  ban_intermediate_pixels: true     # 以AI步骤输入像素为基准判断禁令
```

#### audio（视频音频处理）
```yaml
audio:
  enabled: true
  hum_freq: 50                      # 哼鸣频率（50/60）
  hum_harmonics: 3                  # 谐波次数
  hum_bandwidth: 2                  # 频宽
  hum_width_type: "h"               # 必须设为 h（仅 FFmpeg>=4.4 支持）
  denoise: "afftdn"                 # 全频降噪
  codec: "aac"
  bitrate: "320k"
```

#### photo（照片专用）
```yaml
photo:
  max_long_edge: 3840               # AI 超分禁令阈值
  dpi_threshold: 1200               # 高 DPI 扫描件标记
  avif_params:
    cicp: "1/13/1"                  # BT.709 YUV420
  avif_rgb444_params:
    cicp: "1/13/0"                  # BT.709 RGB444 lossless
```

## 5. 使用方法

### 5.1 安装依赖

```bash
# 系统包 (Ubuntu/Debian)
sudo apt update
sudo apt install ffmpeg vapoursynth libavif-bin gmic exiftool

# FFmpeg 需 >= 4.4
ffmpeg -version  # 确认版本

# Python 包
pip install vapoursynth opencv-python numpy pandas pyyaml Pillow requests rich
pip install 'havsfunc==33'  # 强制 r33

# VapourSynth 插件
python -m vsrepo install qtgmc temporaldegrain2 vivtc deblock_qed
python -m vsrepo install knlmeanscl mvtools2 masktools2 nnedi3 addgrainc

# AI 模型 (放在 models/ 目录)
mkdir -p models
wget https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.1/RealESRGAN_x2plus.pth -P models/
wget https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.4.pth -P models/
```

### 5.2 四步完整工作流

```bash
# Step 1: 依赖预检
python -c "
import sys; sys.path.insert(0, 'shared_utils')
from dependency_check import run_all_checks
ok, issues = run_all_checks()
if not ok:
    for i in issues: print(i)
    exit(80)
print('All dependencies OK')
"

# Step 2: 生成清单
python -c "
import sys; sys.path.insert(0, 'shared_utils')
from manifest_generate import generate_manifest
import yaml
with open('video_restoration_pipeline/config.yaml') as f:
    config = yaml.safe_load(f)
generate_manifest(['/mnt/videos'], 'batch_manifest.yaml', config, 'video')
"

# Step 3: 查看和编辑清单（可选）
python -c "
import sys; sys.path.insert(0, 'shared_utils')
from manifest_editor import list_jobs
list_jobs('batch_manifest.yaml')
"

# Step 4: 批量处理
python video_restoration_pipeline/main.py --manifest batch_manifest.yaml
```

### 5.3 两步简版

```bash
# 视频修复（自动分类+处理）
python video_restoration_pipeline/main.py --input /mnt/videos --output ./output

# 照片修复
python photo_restoration_pipeline/main.py --input /mnt/photos --output ./output
```

### 5.4 高级用法

```bash
# 仅处理特定 stage
python video_restoration_pipeline/main.py --input /mnt/videos --stage 3

# 仅处理特定 job
python video_restoration_pipeline/main.py --manifest batch.yaml --job-id job_001

# 指定 GPU 后端
python video_restoration_pipeline/main.py --input /mnt/videos --gpu cuda

# 禁用 WSL 路径重定向
python video_restoration_pipeline/main.py --input /mnt/videos --no-wsl-redirect

# 保留中间文件
python video_restoration_pipeline/main.py --input /mnt/videos --keep-intermediates

# 预览模式（仅生成预览帧）
python video_restoration_pipeline/main.py --input /mnt/videos --preview

# 后台挂机
nohup python video_restoration_pipeline/main.py --input /mnt/videos > pipeline.log 2>&1 &
```

### 5.5 清单编辑

```bash
# 查看所有 PENDING 任务
python -c "
import sys; sys.path.insert(0, 'shared_utils')
from manifest_editor import list_jobs
list_jobs('batch_manifest.yaml', 'PENDING')
"

# 修改任务状态
python -c "
import sys; sys.path.insert(0, 'shared_utils')
from manifest_editor import set_job_status
set_job_status('batch_manifest.yaml', 'job_001', 'PENDING')
"

# 为特定任务覆盖参数
python -c "
import sys; sys.path.insert(0, 'shared_utils')
from manifest_editor import set_user_override
set_user_override('batch_manifest.yaml', 'job_001', 'crf', '20')
"
```

## 6. 退出码说明

| 退出码 | 含义 | 是否可重试 | 处理方式 |
|--------|------|-----------|---------|
| 0 | 成功 | - | 继续下一 stage |
| 3 | 跳过（中间文件已存在） | - | 继续下一 stage |
| 80 | 依赖缺失 | 否 | 安装缺失依赖后重新运行 |
| 81 | 参数错误 | 否 | 检查 config.yaml 和命令行参数 |
| 82 | IO 错误 | 是（非磁盘满） | 检查磁盘空间和文件权限 |
| 83 | GPU 错误 | 是 | 自动降级 4x->2x->CPU |
| 84 | 编码错误 | 是 | 检查编码器参数 |
| 85 | 权限错误 | 否 | 检查文件/目录权限 |
| 86 | 校验失败 | 否 | 删除损坏的中间文件重试 |
| 87 | 资源不足 | 是（非磁盘满） | 释放内存/磁盘空间 |

## 7. Bug 修复记录

### 三轮查错发现的 Bug 及修复

| 轮次 | Bug | 位置 | 修复方式 |
|------|-----|------|---------|
| 第1轮 | `compute_file_identity` 在文件不存在时崩溃 | checkpoint.py | 添加 try/except，返回安全默认值 |
| 第1轮 | `is_retryable_error` 磁盘满(IO错误+No space)被误判为可重试 | checkpoint.py | 调整检查顺序：stderr 关键字检查优先于 exit_code |
| 第3轮 | 12 个 stage 脚本缺少 `--cleanup-intermediates` 参数 | stages/*.py | 批量添加 argparse 参数 |

## 8. 安全与约束

- **所有外部命令调用均使用 `subprocess.run(..., shell=False)`**，防止命令注入
- **GrainFactory3 仅传入 g1str/g2str/g3str**，禁止未验证参数
- **AI 超分以原始输入像素为基准判断禁令**，防止过度放大
- **音频哼鸣滤波强制使用 `width_type=h`**，需 FFmpeg >= 4.4
- **GPU 显存预检测 + 自动降级策略**，防止 OOM 崩溃

## 9. 技术规格

- **处理位深**：VapourSynth 全程 16bit（YUV420P16/RGB48），编码前转 10bit
- **色域转换**：BT.601 -> BT.709 强制转换，区分 PAL/NTSC
- **中间文件格式**：FFV1 lossless (MKV 容器)
- **输出编码器**：SVT-AV1 10bit YUV420P10
- **输出容器**：MKV（视频）/ AVIF（照片）
- **日志格式**：JSON Lines，多进程安全
- **中间文件命名**：`{job_id}_{name}_step{N}_{op}_p{hash}.{ext}`
- **param_hash 算法**：SHA-256 前16位
- **断点保留**：最近2个 stage 的中间文件
