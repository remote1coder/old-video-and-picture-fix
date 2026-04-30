# 老影像修复自动化脚本 - 架构设计文档

## 1. 项目概述

本项目提供两套完整的自动化修复流水线：
- **video_restoration_pipeline**：处理视频（模拟录像带 / DV / HDV）
- **photo_restoration_pipeline**：处理照片（老照片扫描件 / 早期数码照片）

核心修复哲学：**先传统、后 AI、再颗粒**——先用 VapourSynth/FFmpeg 做传统修复，再谨慎使用 AI 超分/人脸修复，最后在所有破坏性处理完成后添加颗粒还原胶片质感。

## 2. 文件结构

```
video_restoration_pipeline/
├── main.py                    # 视频套件总控脚本
├── config.yaml                # 配置文件
├── stages/
│   ├── stage0_audio_extract.py       # 音频提取（视频处理前）
│   ├── stage1_preprocess.py          # 传统修复（去隔行/IVTC/去噪/稳定）
│   ├── stage2_traditional_check.py   # 传统修复结果校验
│   ├── stage3_ai_upscale.py          # AI 超分（可选）
│   ├── stage4_postprocess.py         # 超分后处理
│   ├── stage5_ai_face.py             # AI 人脸修复（可选）
│   ├── stage6_grain_synthesis.py     # 颗粒合成
│   └── stage7_encode.py              # 最终编码+音频处理+封装
├── classify.py                # 视频素材分类器

photo_restoration_pipeline/
├── main.py                    # 照片套件总控脚本
├── config.yaml                # 配置文件
├── stages/
│   ├── stage1_preprocess.py   # 传统修复（去划痕/褪色还原/去色带）
│   ├── stage2_traditional_check.py  # 传统修复结果校验
│   ├── stage3_ai_upscale.py   # AI 超分（可选）
│   ├── stage4_postprocess.py  # 超分后处理
│   ├── stage5_ai_face.py      # AI 人脸/全身修复（可选）
│   └── stage6_encode.py       # 最终编码输出（AVIF/TIFF）
├── classify.py                # 照片素材分类器

shared_utils/                  # 公共工具模块
├── dependency_check.py        # 依赖预检
├── checkpoint.py              # 断点续传
├── preview.py                 # 预览模式
├── logger.py                  # 结构化日志
├── audio_utils.py             # 音频处理工具
├── manifest_generate.py       # 清单生成
├── manifest_loader.py         # 清单加载
└── manifest_editor.py         # 清单编辑器（可选）

models/                        # AI 模型权重（用户自行下载放置）
├── RealESRGAN_x2plus.pth
├── RealESRGAN_x4plus.pth
├── GFPGANv1.4.pth
├── DAT_x2.pth
└── DAT_x4.pth
```

## 3. 模块关系图

```
                         +------------------+
                         |   dependency_check  |
                         |   (启动时预检)       |
                         +--------+---------+
                                  |
                                  v
+-------------------+    +--------+---------+    +------------------+
|  config.yaml      |--->|      main.py      |--->|  manifest_generate.py / manifest_loader.py |
|  (参数配置)        |    |   (总控调度)       |    |  (作业清单管理)   |
+-------------------+    +--------+---------+    +------------------+
                                  |
                    +-------------+-------------+
                    |                           |
                    v                           v
         +----------+----------+     +----------+----------+
         | classify.py          |     | checkpoint.py       |
         | (素材分类)            |     | (断点续传)           |
         +----------+----------+     +----------+----------+
                    |                           |
                    v                           v
         +----------+----------+     +----------+----------+
         | stages/stage*.py    |     | preview.py          |
         | (分步处理)           |     | (预览模式)           |
         +---------------------+     +---------------------+
```

## 4. 数据流图

### 视频修复数据流

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
    |       |
    |       v
    |   checkpoint.py (中间文件+param_hash)
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

### 照片修复数据流

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
stage1_preprocess.py (去划痕/褪色还原/去色带/单通道校正)
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

## 5. 调用顺序

### 视频套件 Stage 调用顺序

| Stage | 脚本 | 功能 | 输入 | 输出 |
|-------|------|------|------|------|
| 0 | stage0_audio_extract.py | 音频提取 | 原始视频 | 原始音频流(WAV/PCM) |
| 1 | stage1_preprocess.py | 传统修复 | 原始视频 | 修复后视频(YUV420P16) |
| 2 | stage2_traditional_check.py | 结果校验 | 修复后视频 | 校验报告 |
| 3 | stage3_ai_upscale.py | AI超分 | 修复后视频帧 | 超分后帧(RGB48) |
| 4 | stage4_postprocess.py | 后处理 | 超分后帧 | 处理后帧 |
| 5 | stage5_ai_face.py | 人脸修复 | 处理后帧 | 人脸修复帧 |
| 6 | stage6_grain_synthesis.py | 颗粒合成 | stage4/5输出 | 含颗粒视频 |
| 7 | stage7_encode.py | 编码封装 | 含颗粒视频+处理后音频 | MKV文件 |

### 关键约束

- **IVTC → 去隔行顺序**：stage1 中 IVTC 必须在 QTGMC 之前
- **AI超分 → 人脸修复顺序**：stage3 必须在 stage5 之前
- **所有处理 → 颗粒合成顺序**：stage6 必须在所有画质处理之后
- **stage0 仅提取音频，不做处理**
- **音频处理并入 stage7**

## 6. Manifest 机制详解

### 6.1 设计目的

Manifest（作业清单）是批量处理的核心状态管理器，实现：
- 批量任务的持久化跟踪
- 断点续传的状态记录
- AI超分禁令的永久硬标记
- 用户参数覆盖的追溯
- 多进程并行的任务分配

### 6.2 数据结构

```yaml
meta:
  generated_at: "ISO-8601时间戳"
  source_dirs: ["/path/to/input"]
  total_files: N
  pending: N
  done: N
  failed: N
  skipped: N

jobs:
  - job_id: "job_001"
    input_path: "/path/to/file.avi"
    detected_class: "ANALOG_TAPE"  # 或 DV_DIGITAL, HDV_AVCHD
    confidence: 0.92
    field_order: "tt"  # tt/bb/psf/progressive/unknown
    needs_ivtc: true
    is_psf: false
    status: "PENDING"  # PENDING/DONE/FAILED/SKIP
    priority: 0
    ai_upscale_banned: false  # 永久硬标记
    user_overrides: {}
    output_path: "./output/file_restored.mkv"
    log_path: "./logs/job_001.log"
    error_message: ""
    # 文件唯一标识（用于param_hash）
    file_size: 104857600
    file_mtime: 1714291200
    file_hash_prefix: "a1b2c3d4"
    # 像素尺寸基准
    original_width: 720
    original_height: 480
    native_width: 720
    native_height: 480
    native_dar: "4:3"
```

### 6.3 状态流转

```
                    +----------+
    +--------------->|  PENDING  |
    |                +----+-----+
    |                     |
    |                     | 开始处理
    |                     v
    |               +-----+------+
    |          +--->| IN_PROGRESS|
    |          |    +-----+------+
    |          |          |
    |          |          | 处理完成
    |          |          v
    |          |     +----+----+
    |          +-----|   DONE   |
    |          |     +---------+
    |          | 处理失败
    |          |
    |     +----+----+
    +-----|  FAILED  |
          +----+----+
               | 重试
               |
          +----+----+
          |  SKIP   |
          +---------+
```

### 6.4 参数优先级规则

**命令行参数 > user_overrides > paths（素材类型专属）> global > 脚本内置默认值**

### 6.5 AI超分禁令标记

- `manifest_generate.py` 生成时根据原始像素尺寸计算 `ai_upscale_banned`
- 后续所有 stage 强制读取该标记，**严禁仅在 AI 步骤做运行时校验**
- 用户可通过 `force_allow_banned_upscale=true` 覆盖（记录 CRITICAL 日志）

## 7. 断点续传机制

### 7.1 中间文件命名

```
{job_id}_{input_name}_step{N}_{operation}_p{param_hash}.{ext}
```

示例：`job_001_tape001_step2_denoise_p7a3f9e2b.mkv`

### 7.2 param_hash 计算

- 纳入输入文件唯一标识：文件大小 + 修改时间戳 + xxhash64前16位（>1GB分块采样）
- 纳入核心画质参数：去噪强度、QTGMC preset、IVTC模式、色域矩阵、超分倍数、AI模型名、颗粒强度、色深、CRF/preset、人脸修复参数等
- 排除：日志级别、输出路径、临时路径、并行线程数
- 算法：SHA-256 前 16 位

### 7.3 帧数据完整性校验

- 覆盖开头(0-5%)、中间(45-55%)、结尾(95-100%)三个区间
- 每区间至少抽取1帧解码为像素数组
- 校验：解码无错、像素非全0/全255、亮度均值在合法区间
- 时长>10分钟视频，每10分钟额外抽取1帧

## 8. 配置系统

### 8.1 分层结构

- **global**：全局默认参数
- **paths.analog_tape**：模拟录像带专属参数
- **paths.dv_digital**：DV数码录像专属参数
- **paths.hdv_avchd**：HDV/AVCHD专属参数
- **grain_factory3**：颗粒合成参数
- **ai**：AI超分/人脸修复参数
- **audio**：音频处理参数
- **photo**：照片处理参数

### 8.2 运行时合并流程

```
config.yaml (global)
    |
    v
config.yaml (paths.{detected_class})  <- 素材类型匹配
    |
    v
user_overrides (单个文件覆盖)
    |
    v
命令行参数 (最高优先级)
```

## 9. 关键设计决策

### 9.1 16bit 处理空间

- 所有 VapourSynth 处理在 16bit（YUV420P16/RGB48）执行
- 最终编码前转为 10bit
- 16bit→10bit 在 YUV444 空间执行 error_diffusion 抖动后再下采样为 YUV420

### 9.2 色域转换

- BT.601→BT.709 为强制步骤
- 区分 PAL/NTSC 和 full/limited range
- 黑边裁剪→色域转换→还原黑边（使用目标位深对应黑电平）

### 9.3 音频处理时序

- stage0 仅提取原始音频
- stage7 执行音频处理（去哼鸣→全频降噪）
- 去哼鸣必须先于全频降噪
- IVTC/去隔行不改变视频总时长，无需音频变速

### 9.4 WSL2 适配

- 临时路径自动重定向到 Linux 原生分区
- 成品文件回迁到用户原始 `/mnt/` 路径
- 使用 `shutil.copyfile`（非 copy2）避免 PermissionError
- 跨文件系统禁用 `os.rename()`

### 9.5 GPU 降级策略

- 优先级：CUDA → ROCm → MPS → CPU
- 显存预检测：可用显存 < 需求×1.2 时触发降级
- OOM 时：del model → empty_cache → 等待100ms → 降级重试
- AI 子进程使用 `multiprocessing.get_context('spawn')`

## 10. 退出码规范

| 退出码 | 含义 | 说明 |
|--------|------|------|
| 0 | 成功 | 正常完成 |
| 3 | 跳过/已存在 | 中间文件已存在且校验通过 |
| 80 | 依赖缺失 | 外部工具或 VS 插件未安装 |
| 81 | 参数错误 | 配置参数或命令行参数错误 |
| 82 | IO 错误 | 文件读写错误 |
| 83 | GPU 错误 | CUDA/ROCm/MPS 相关错误 |
| 84 | 编码错误 | 编码器返回错误 |
| 85 | 权限错误 | 文件权限不足 |
| 86 | 校验失败 | 中间文件或输出校验失败 |
| 87 | 资源不足 | 磁盘空间或内存不足 |
