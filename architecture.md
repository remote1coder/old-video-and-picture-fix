# Project Architecture

## 1. Pipeline Controller

### 1.1 Entry Point
- `video_restoration_pipeline/main.py` — Video controller (supports batch/manifest mode)
- `photo_restoration_pipeline/main.py` — Photo controller (single-file / `--input-dir` batch / `--manifest` + `--resume` 清单工作流)

### 1.2 Parameter Merge Hierarchy
6-layer parameter cascading per v7.32 §3.2:

1. `global` defaults — base configuration
2. `paths[type]` — type-specific defaults (video: analog_tape/dv_digital/hdv_avchd; photo: scan_print/digital_photo)
3. `audio_override[type]` — per-material audio parameter overrides
4. `recommended_params` — classifier recommendations (from manifest or classify.py)
5. `user_overrides` — manifest editor overrides
6. CLI arguments — highest priority

**注意**:
- `audio_override` 优先级低于 `paths[type]` 但高于 `global`，符合规范要求。
- `ai` 配置以扁平化方式注入 params（同时保留 `ai_{k}` 前缀版本），确保 stage 脚本兼容性。

### 1.3 Core Workflow
- Parse CLI arguments
- Load and validate config (YAML)
- Initialize logger (per-job, structured JSON Lines)
- Classify material → get detected_class, confidence, params
- Merge 6-layer parameters
- Execute stages sequentially
- Each stage is a separate Python script call with subprocess.run()
- Optional: retry on transient errors (photo: max 2 retries, intervals 5s/15s per spec)

---

## 2. Stage System

### 2.1 Stage Scripts

#### Video Pipeline Stages
| Stage | File | Purpose | Input → Output |
|-------|------|---------|----------------|
| 0 | `stages/stage0_audio_extract.py` | Audio extraction | video → audio `.mka`（`-c:a copy`，任意编解码） |
| 1 | `stages/stage1_preprocess.py` | Preprocessing (QTGMC, de-block, etc.) | video → processed video |
| 2 | `stages/stage2_traditional_check.py` | Quality verification | processed video → verified |
| 3 | `stages/stage3_ai_upscale.py` | AI super-resolution | verified → upscaled |
| 4 | `stages/stage4_postprocess.py` | Post-processing | upscaled → refined |
| 5 | `stages/stage5_ai_face.py` | Face restoration | refined → face-restored |
| 6 | `stages/stage6_grain_synthesis.py` | Film grain synthesis | face-restored → grain-added |
| 7 | `stages/stage7_encode.py` | Final encoding | grain-added → final output |

#### stage3 内部数据流：默认零落盘（`ai_stream_raw`）

stage3 有两种内部通路，由 `ai_stream_raw` 选择：

**默认 —— 三进程管道，帧不进磁盘：**

```
ffmpeg(-i stage2.mkv -f rawvideo -pix_fmt rgb48le -)   ← 解码
      → python(读 stdin 帧 → RealESRGAN → 写 stdout)   ← 超分
      → ffmpeg(-f rawvideo ... -i - -c:v ffv1 step3.mkv)  ← 编码
```

- OS 负责背压，父进程不参与协调；没有双向管道，不会死锁
- **不受 `ai_window_size` 限制**，整段一次跑完
- 失败自动落回下面的 PNG 通路（`STAGE3_STREAM_FALLBACK`）

**回退 —— PNG 中转（`ai_stream_raw: false`）：**

```
stage2.mkv → ffmpeg 抽帧 frame_%08d.png → AI 读写 PNG → ffmpeg 编 step3.mkv
```

这条通路每个窗口会写「输入帧 + 输出帧」两批 PNG，2560×1440 时约 21 MB/帧，
是纯粹的固态硬盘损耗。`ai_frame_transport`（`shared_memory` / `auto` / `png`）
只影响 AI 侧用环还是用文件，**两端仍是 PNG，因此不省写入**。


#### Photo Pipeline Stages
| Stage | File | Purpose | Input → Output |
|-------|------|---------|----------------|
| 1 | `stages/stage1_preprocess.py` | Preprocessing (colorspace, scratch removal, deblock, deband, channel correction) | photo → processed TIFF |
| 2 | `stages/stage2_ai_pre.py` | AI pre-processing (single-channel correction for AI upscale input) | processed → corrected |
| 3 | `stages/stage3_ai_upscale.py` | AI super-resolution (Real-ESRGAN/DAT/GPEN) | corrected → upscaled PNG/TIFF |
| 4 | `stages/stage4_ai_face.py` | Face restoration (GFPGAN/CodeFormer) + feather blend | upscaled → face-restored PNG |
| 5 | `stages/stage5_postprocess.py` | Post-processing (selective local processing, portrait color match, grain synthesis) | face-restored → refined PNG/TIFF |
| 6 | `stages/stage6_encode.py` | Final AVIF encoding (YUV420 10bit / RGB444 16bit lossless) | refined → final .avif |

**注意**: 旧版文件 `stage4_postprocess.py` 和 `stage5_ai_face.py` 已于 Round 12 清理删除。v7.32 §3.1 修正了照片管线阶段顺序：face restore (stage4) 先于 postprocess (stage5)，grain synthesis 移至 stage5。

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
- Returns: original_width, original_height, native_width, native_height, native_dar, dpi, ai_upscale_banned, file_hash_prefix
- **注意**: `original_width`/`original_height` 已正确注入 params，stage3 的 1080p 检查正常工作。

---

## 4. Shared Modules

All shared modules are in `shared_modules/` directory.

| Module | File | Purpose |
|--------|------|---------|
| Checkpoint | `checkpoint.py` | param_hash, intermediate naming, integrity verification, WSL redirect, GPU memory check |
| Logger | `logger.py` | Structured JSON Lines logging per job, multi-process safe |
| Dependency Check | `dependency_check.py` | Pre-run validation of tools/libraries |
| Audio Utils | `audio_utils.py` | Hum detection, audio profiling |
| Manifest Loader | `manifest_loader.py` | Load/save YAML/JSONL manifests |
| Manifest Editor | `manifest_editor.py` | Edit manifest jobs |
| Manifest Generator | `manifest_generate.py` | Generate manifests from directories |
| Preview | `preview.py` | Preview frame extraction |
| Media IO | `media_io.py` | Media probing / IO helpers; `read_image_array` (format-aware), `load_image_rgb_uint8` (per-channel adaptive TIFF reading), `write_tiff` (zlib) |
| Cancel Guard | `cancel_guard.py` | Cancellation / interrupt guard (rc=88), managed subprocess runner |
| AI Runtime | `ai_runtime.py` | Resolve AI subprocess interpreter (`WB_AI_PYTHON` → current → `.venv_r4`) |
| DAT Runtime | `dat_runtime.py` | DAT architecture configs + PyTorch inference (single source of truth); output format follows the output path extension |

### 4.1 Python Path Setup
每个 stage 脚本把 `shared_modules` 加入 `sys.path`，但**两条链的层级不同**，
表达方式也不同：

```python
# video_restoration_pipeline/stages/*.py —— SCRIPT_DIR 是 stages/ 子目录
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "shared_modules"))

# photo_restoration_pipeline/stages/*.py —— SCRIPT_DIR 是链目录本身，
# 因此项目根只往上走一层；写成 .parent.parent 会高一层
sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))
```

**这是跨链最易踩的坑**：两条链里同名表达式 `SCRIPT_DIR.parent.parent`
含义不同。照片链若照抄视频链的写法，模型路径会解析到项目根之外，
表现为"权重找不到"→ 推理失败。同理，`ai_spawn_isolation` 相关实现
在两条链里也各自指向各自的链目录。

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
- `paths` — per-type settings (scan_print, digital_photo)
- `photo` — photo-specific defaults (scratch_remove, deblock_strength, deband_threshold, grain_synthesis, grain_strength, grain_size, grain_roughness, avif_params, avif_rgb444_params, channel_correction_*)
- `grain_factory3` — `enabled: false` (photos don't use GrainFactory3; use Laplacian pyramid grain in stage5 instead)
- `audio` — `enabled: false` (photos don't process audio)
- `ai` — AI upscaling configuration

---

## 6. Logging

### 6.1 Log Format
Structured JSON Lines, one file per job:
```json
{"timestamp": "2026-04-30T12:00:00+08:00", "level": "INFO", "event": "STAGE_START", "message": "Starting stage 1", "job_id": "job_123"}
```

### 6.2 Log Rotation
Per-file: max 10000 lines, 5 backups. Large fields truncated with SHA-256 hash.

### 6.3 Multi-Process Safety
- Linux/macOS: `fcntl.flock` 文件锁
- Windows: `msvcrt.locking` 文件锁
- Fallback: 无锁（单进程场景）

**实现现状**:
- Windows 用 `msvcrt.locking` 阻塞锁 (`LK_LOCK`)；Linux/macOS 用 `fcntl.flock`
- 日志轮转使用 timestamp-based 备份名，避免多进程 rename 竞争
- `merge_params()` 记录所有参数覆盖事件（`PARAM_OVERRIDE`）
- 行数统计按实时文件行数，不依赖内存计数

---

## 7. Checkpoint / Resume

### 7.1 param_hash
Deterministic hash based on:
- File identity: size, mtime, xxhash prefix (xxhash3_64)
- Quality params: crf, preset, AI model, deinterlace settings, scratch_remove, grain params, etc.
- Excludes: job_id, output_path, log settings

**已知缺口**（低影响）: `CORE_QUALITY_PARAMS` 未包含
`enable_channel_correction`、`frc_target_fps`、`aspect_ratio_correction`，
改动这些参数不会改变 `param_hash`（可能复用旧中间产物）。

### 7.2 Intermediate Files
Naming: `{job_id}_{input_name}_step{N}_{operation}_p{param_hash}.{ext}`

### 7.3 Cleanup
- After each job: keep N most recent stages (`keep_steps` 默认 **2**)
- Full cleanup on completion (configurable)

**实现要点**:
- `get_file_xxhash()` 对 >1GB 文件按 head/middle/tail 各 1MB 采样
- `is_retryable_error()` 覆盖 Linux/Windows/macOS 的跨平台信号码
- `verify_frame_integrity()` 16bit limited-range 亮度阈值为 4096/60160
- `.avif` 已列入照片扩展名白名单

---

## 8. WSL2 适配

### 8.1 路径重定向
- `wsl_redirect_path()` 检测 WSL 环境并尝试重定向路径
- `--no-wsl-redirect` 可禁用自动重定向
- `platform` 参数控制重定向行为（`wsl2`/`auto` 允许重定向）

**实现现状**:
- `wsl_redirect_path()` 同时检查 `output_path` 与 `temp_dir`，两者任一在 `/mnt/` 下即重定向到 Linux 原生分区，返回 4 元组（原/新 output、原/新 temp）
- `wsl_copy_back()` 用 `copyfile` + `os.utime()`，三重校验（MD5 + size + ffprobe）+ 跨文件系统 `.bak` 备份 + 最多 3 次重试

### 8.2 CUDA 路径检测
WSL2 下 CUDA 存根可能位于多个路径。已检测 `/usr/local/cuda/lib64`、`/usr/lib/wsl/lib`、`/opt/cuda/lib64`；但仍需完善环境变量持久化和用户覆盖机制。

---

## 9. GPU 显存管理

### 9.1 显存预检测
- `gpu_memory_check()` 在 `checkpoint.py` 中提供
- 通过 `nvidia-smi` / `torch.cuda.mem_get_info()` 查询**可用**显存（非总显存）
- 支持 CUDA / ROCm / MPS 后端探测；参数名为 `vram_required_mb`（单位 MiB）

### 9.2 分级降级
- AI stage 中实现 4×→2×→CPU 降级逻辑
- 降级时记录 WARNING 日志

---

## 10. Dependency Check

Run before first use:
```bash
python shared_modules/dependency_check.py -c video_restoration_pipeline/config.yaml --pipeline-type video
python shared_modules/dependency_check.py -c photo_restoration_pipeline/config.yaml --pipeline-type photo
```

Checks: FFmpeg version, VapourSynth plugins, Python packages, AI model files, G'MIC filters.

Returns exit code 80 if any required dependency is missing.

**当前检查范围**（`dependency_check.py`）:
- 外部工具：FFmpeg≥4.4（bandreject `width_type=h`）、ExifTool、G'MIC、avifenc（含版本与 `--jobs`）
- Python 包：yaml/numpy/PIL/xxhash/rich 等；AI 栈以**子进程探针**在 AI 解释器内验证
- VapourSynth：最小调用集（含 `sRestore` 间接校验）
- AI 模型：逐个权重存在性 + 可选外部 `md5_manifest.json` 校验
- 人脸检测：facexlib（视频链唯一可用后端）
- spawn 合规：静态扫描（`if TYPE_CHECKING:` 白名单 + `# spawn-safe:` 抑制标记）

**已知限制**:
- G'MIC 仅检查存在性，未对 `-repair`/`-deblock`/`-deband` 做滤镜级探测
- `-film_grain` 不存在于 G'MIC CLI（照片 grain 已改为纯 Python 拉普拉斯金字塔）

---

## 11. 产物核对体系（stage7 三道门 + 全链核对）

成品发布前有三道**相互独立**的核对门，每道都可被对应开关强制绕过：

| 门 | 判据 | 失败退出码 | 强制开关 |
|---|---|---|---|
| 音画同步 | 成品**内部** 音视频区间（start / end / span） | 86 | `allow_sync_fail_publish` |
| 全链核对 | **原始素材 vs 成品**（帧数 / 时长 / 首末帧 / 黑图 / 逐帧对齐） | 86 | `allow_verify_fail_publish` |
| 机械核对 | **step6 产物 vs 成品**（编码这一跳） | 86 | `allow_verify_fail_publish` |

三处强制判断收敛为 `stage7_encode._force_publish_enabled(params, key)`，
并有结构性测试保证每道门都经由它（源码中出现 ≥4 次：定义 1 + 调用 3）。

### 11.1 音画判据：相对源片，不用绝对阈值

`_validate_av_sync()` 取 start / end / span 三项中绝对值最大者并**保留符号**，
方向决定阈值（ITU-R BT.1359-1 的可接受线 +90 / −185 ms，此处取其约 2 倍）：

    声音滞后：warn 185ms / fail 400ms
    声音超前：warn  90ms / fail 200ms

若提供了 `reference_path`（原始素材），判据改为
`| 成品漂移 − 源片漂移 |` —— 源片自带的漂移不算流程的问题。
实测 204 个真实素材：绝对阈值误杀率 31%，相对判据 1.5%。

四个阈值均可在 config 中调整（设非正数退回默认，**不允许关闭判定**）。

### 11.2 全链核对三档模式

`full_chain_verify_mode`：

| 模式 | 帧数 | 时长 / 漂移 | 误杀风险 |
|---|---|---|---|
| `frames_only`（默认） | 拦 | 只记录 | 低 |
| `relative` | 拦 | 拦（源片 vs 成品） | 中 |
| `absolute` | 拦 | 拦 + 成品自身绝对漂移 | 高 |

结果一律写入 `<output>.fullchain.json`，即使 `frames_only` 也记录漂移，
便于事后用 `shared_modules/fullchain_report.py` 做筛查（体检工具）。

帧数判据之所以可以默认硬拦：`artifact_verify.expected_frame_counts()`
已覆盖三种合法的帧数变化（IVTC 5→4、QTGMC divisor=1 翻倍、divisor=2 不变）。
**注意**：源帧数必须用 `-count_frames`（显示帧数），不能用 `-count_packets`
—— 后者对含负 PTS 参考帧的文件会高估（实测 651 vs 599）。

### 11.3 人工确认入口

分类器对低置信度素材（<0.80）置 `needs_confirmation`，`main.py` 门禁为：

    if job.get("needs_confirmation", False) or confidence < 0.80:
        # 非交互环境 → SKIP

`manifest_editor` 提供 `list --low-confidence` 与 `confirm`（支持
`--all-low-confidence` 批量）。确认时：清 `needs_confirmation`、
把 confidence 至少抬到 0.80、并把 `SKIP` 复位为 `PENDING`
（否则 scan 标的 SKIP 原样保留，批次会 "Processing 0 video jobs"）。

## 12. 已知架构级限制（当前）

- `CORE_QUALITY_PARAMS` 未包含 `enable_channel_correction` /
  `frc_target_fps` / `aspect_ratio_correction`：改这三项不会改变
  `param_hash`，可能复用旧中间产物（低影响）
- 照片链 `stage2_traditional_check.py` 曾是遗留死代码，已于 2026-10-20 删除
  （`git rm`；视频链同名 stage 是活跃的，与它无关）
- 视频链 AI 推理必须走外部解释器（`ai_runtime`），编排器 R72 无 torch
- 依赖检查的 spawn 合规扫描是静态文本匹配（`# spawn-safe:` 可人工抑制）
- WSL2 相关路径逻辑在本机（Windows 原生）不触发，属未实测分支
- **中间 TIFF 的压缩方式必须是 tifffile 可直读的**
  （`NONE` / `DEFLATE` / `ADOBE_DEFLATE`），不得用 LZW：
  LZW 解码依赖 `imagecodecs`，本项目未安装。
  注意 `PIL.Image.save()` 与 `cv2.imwrite()` 对 `.tif` **都默认 LZW**，
  须显式指定压缩参数，或改走 `media_io.write_tiff()`。
- **G'MIC 写出的 `uint16` TIFF 可能名实不符**（声明 16-bit、实际 8-bit
  量级）。读这类文件必须用 `media_io.load_image_rgb_uint8()`
  （tifffile + **逐通道**值域自适应）；按全局最大值判定会在
  "部分通道被放大"的中间产物上整体走 `/257`，把画面压成全黑。
- `facexlib` 是唯一可用的人脸检测后端（RetinaFace / YOLOv8-face / MTCNN
  三个独立包均未安装）。它是 GFPGAN/CodeFormer 的内置检测实现，
  不是并列备选。CodeFormer 架构取自 `basicsr.archs.codeformer_arch`。

## 13. 历史审查记录（本地）

2026-04 之后的逐文件审查、缺陷根因与修复过程已移出本文档，保存在
本地 `bugs.md` / `shixian.md` / `修复清单.md`（不入库）。
