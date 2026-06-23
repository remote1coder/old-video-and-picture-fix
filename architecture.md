# Project Architecture

## 1. Pipeline Controller

### 1.1 Entry Point
- `video_restoration_pipeline/main.py` — Video controller (supports batch/manifest mode)
- `photo_restoration_pipeline/main.py` — Photo controller (single-file mode + `--input-dir` batch mode; manifest workflow not yet implemented)

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
| 0 | `stages/stage0_audio_extract.py` | Audio extraction | video → audio.flac |
| 1 | `stages/stage1_preprocess.py` | Preprocessing (QTGMC, de-block, etc.) | video → processed video |
| 2 | `stages/stage2_traditional_check.py` | Quality verification | processed video → verified |
| 3 | `stages/stage3_ai_upscale.py` | AI super-resolution | verified → upscaled |
| 4 | `stages/stage4_postprocess.py` | Post-processing | upscaled → refined |
| 5 | `stages/stage5_ai_face.py` | Face restoration | refined → face-restored |
| 6 | `stages/stage6_grain_synthesis.py` | Film grain synthesis | face-restored → grain-added |
| 7 | `stages/stage7_encode.py` | Final encoding | grain-added → final output |

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

### 4.1 Python Path Setup
大多数 stage 脚本正确添加 shared_modules 到 sys.path：
```python
SCRIPT_DIR = Path(__file__).parent.resolve()
sys.path.insert(0, str(SCRIPT_DIR.parent.parent / "shared_modules"))
from logger import get_logger
from checkpoint import compute_param_hash
```

**⚠️ 已修复**: `video_restoration_pipeline/stages/stage6_grain_synthesis.py` 的 sys.path 设置已修正，现在正确包含 `shared_modules/`。

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

**历史问题状态**:
- ~~Windows 使用 `msvcrt.LK_NBLCK`（非阻塞锁）~~ ✅ **已修复**：改为 `LK_LOCK` 阻塞锁（BUG-004/BUG-025）。
- ~~日志轮转 rename 竞争~~ ✅ **已修复**：使用 timestamp-based 备份名。
- ~~`param_override()` 未被调用~~ ✅ **已修复**：`merge_params()` 中记录所有覆盖事件。
- ~~`self.line_count` 多进程计数~~ ✅ **已修复**：改用实时行数统计。
- `param_override()` source 参数未限制枚举值（BUG-016，🟢 低）

---

## 7. Checkpoint / Resume

### 7.1 param_hash
Deterministic hash based on:
- File identity: size, mtime, xxhash prefix (xxhash3_64)
- Quality params: crf, preset, AI model, deinterlace settings, scratch_remove, grain params, etc.
- Excludes: job_id, output_path, log settings

~~**⚠️ 已知问题**: `CORE_QUALITY_PARAMS` 白名单缺少部分照片参数~~ ✅ **已修复**: 照片参数已加入 `CORE_QUALITY_PARAMS`。  
**⚠️ 仍缺少**: `enable_channel_correction`、`frc_target_fps`、`aspect_ratio_correction` 等少量参数（🟢 低，不影响核心功能）。

### 7.2 Intermediate Files
Naming: `{job_id}_{input_name}_step{N}_{operation}_p{param_hash}.{ext}`

### 7.3 Cleanup
- After each job: keep N most recent stages (configurable, **默认 5，规范要求 2**）
- Full cleanup on completion (configurable)

**历史问题状态**:
- ~~`cleanup_old_intermediates()` 默认 `keep_steps=5`~~ ✅ **已修复**：默认值已调整为 2。
- ~~`wsl_redirect_path()` 对 `/mnt/` 前缀路径不做重定向~~ ✅ **已修复**：已实现正确的重定向（同时检查 output_path 和 temp_dir）。
- ~~`wsl_copy_back()` 使用 `shutil.copy2`~~ ✅ **已修复**：改用 `copyfile` + `os.utime()`，跨文件系统时备份旧文件为 `.bak`。
- ~~`verify_frame_integrity()` 16bit limited range 亮度校验值~~ ✅ **已修复**：阈值改为 4096/60160（BUG-005）。
- ~~`_read_frame_info()` 未将 `.avif` 列入照片扩展名~~ ✅ **已修复**。
- ~~`get_file_xxhash()` 对 >1GB 文件未按规范分块采样~~ ✅ **已修复**：采样 head/middle/tail 各 1MB。
- ~~`is_retryable_error()` 未处理跨平台信号码~~ ✅ **已修复**：覆盖 Linux/Windows/macOS 信号码（BUG-001）。
- ~~`estimate_disk_space()` 精度丢失~~ ✅ **已修复**：`int()` 移至最外层（BUG-027）。

---

## 8. WSL2 适配

### 8.1 路径重定向
- `wsl_redirect_path()` 检测 WSL 环境并尝试重定向路径
- `--no-wsl-redirect` 可禁用自动重定向
- `platform` 参数控制重定向行为（`wsl2`/`auto` 允许重定向）

**WSL2 重定向状态**: ✅ **已修复**。`wsl_redirect_path()` 同时检查 output_path 和 temp_dir；`wsl_copy_back()` 实现三重校验 + 备份 + 3 次重试。
- ✅ `wsl_redirect_path()` 已实现 output_path 和 temp_dir 重定向。
- ✅ `wsl_copy_back()` 已改用 `copyfile` 并实现三重校验 + 跨文件系统 `.bak` 备份。
- ✅ 所有调用方已同步更新为 4 元组返回值解包。

### 8.2 CUDA 路径检测
WSL2 下 CUDA 存根可能位于多个路径。已检测 `/usr/local/cuda/lib64`、`/usr/lib/wsl/lib`、`/opt/cuda/lib64`；但仍需完善环境变量持久化和用户覆盖机制。

---

## 9. GPU 显存管理

### 9.1 显存预检测
- `gpu_memory_check()` 在 `checkpoint.py` 中提供
- **历史问题状态**: ✅ 大部分已修复。
  - 已改用 nvidia-smi / `torch.cuda.mem_get_info()` 查询可用显存。
  - 已添加 ROCm/MPS 检测。
  - 已修复 "trt" 后端检查逻辑。
  - ~~⚠️ `vram_required` 参数单位未明确~~ ✅ **已修复**：参数已重命名为 `vram_required_mb`，文档字符串明确说明单位为 MiB。

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

**依赖检查状态**: 大部分历史问题已修复。
- ~~**CodeFormer 模型权重**~~ ✅ 已修复。
- ~~**havsfunc 版本匹配**~~ ✅ 已修复。
- ~~**CodeFormer 可用性**~~ ✅ 已修复。
- ~~**ROCm / MPS GPU 后端扩展检测**~~ ✅ 已修复。
- ~~**FFmpeg 滤镜最小参数集测试调用**~~ ✅ 已修复。
- ~~**arnndn 强制检查**~~ ✅ 已修复。
- ~~**photo 模式 ffmpeg/ffprobe**~~ ✅ 已修复。
- ~~**spawn 三约束预检**~~ ✅ 已修复（基础实现）。
- ~~**人脸检测 Python 包名一致性**: 仍使用 `retinaface`，如需严格匹配规范 `retina-face` 包名，可调整。~~ ✅ 已修复：错误提示已改为 `pip install retina-face`（PyPI 官方包名），模块检测仍使用 `retinaface`。
- **`sRestore` 滤镜的显式独立校验**: 当前仍在 `check_vs_minimal_calls()` 中间接包含。
- **G'MIC 具体滤镜**（`-repair`, `-deblock`, `-deband`）可用性验证: 已检查 G'MIC 存在性，但未对具体滤镜做最小参数集测试。
- **`-film_grain` 不存在于 G'MIC**: 经联网搜索验证（G'MIC 官方 `gui_filters.txt` 2026-06-11 更新），G'MIC CLI 中不存在 `-film_grain` 命令。实际存在的颗粒相关命令为 `mc_film_grain`（社区滤镜）、`-fx_simulate_grain`、`-fx_simulate_film`。Round 13 已将照片 grain 合成替换为纯 Python 拉普拉斯金字塔实现，不再依赖 G'MIC。
- **avifenc 版本解析和 `--jobs` 支持检测**（照片流程需要）。
- ~~**AI 模型 MD5 校验**: 未启用（所有模型 `md5` 字段为 `None`）。~~ ✅ 已修复：支持外部 `md5_manifest.json` 清单加载，无 MD5 时输出配置指导警告。

---

## 11. 已知架构级问题汇总

| ID | 问题 | 位置 | 影响 |
|----|------|------|------|
| BUG-PHOTO-001 | main.py 未传 original_width/height | photo/main.py | ✅ 已修复 |
| BUG-PHOTO-002/003 | CLI 键名不匹配 | photo/main.py + stage3 | ✅ 已修复 |
| BUG-PHOTO-027 | wsl_copy_back 用 copy2 非 copyfile | checkpoint.py | ✅ 已修复 |
| BUG-PHOTO-028 | gpu_memory_check 用 total_memory | checkpoint.py | ✅ 已修复 |
| BUG-PHOTO-029 | CORE_QUALITY_PARAMS 不完整 | checkpoint.py | ✅ 已修复 |
| BUG-PHOTO-030 | cleanup 保留 5 个 stage | checkpoint.py | ✅ 已修复 |
| BUG-PHOTO-048 | 人脸检测阈值未应用 | photo/stage5 | ✅ 已修复 |
| BUG-PHOTO-050 | CodeFormer fallback 无操作 | photo/stage5 | ✅ 已修复 |
| BUG-SM-001 | WSL `/mnt/` 路径不做重定向 | checkpoint.py | ✅ 已修复 |
| BUG-SM-002 | WSL copyback 缺三重校验 | checkpoint.py | ✅ 已修复 |
| BUG-SM-003 | 缺少 CodeFormer 模型检查 | dependency_check.py | ✅ 已修复 |
| BUG-SM-004 | havsfunc 版本子串匹配不严谨 | dependency_check.py | ✅ 已修复 |
| BUG-SM-005 | basicsr 假设即 CodeFormer | dependency_check.py | ✅ 已修复 |
| BUG-SM-006 | Windows 日志锁非阻塞 | logger.py | ✅ 已修复 |
| BUG-SM-007 | 日志轮转 rename 竞争 | logger.py | ✅ 已修复 |
| BUG-SM-008 | arnndn 未先转 mono | audio_utils.py | ✅ 已修复 |
| BUG-SM-009 | arnndn 模型未按频率自动选择 | audio_utils.py | ✅ 已修复 |
| BUG-SM-010 | 预览模式差距极大 | preview.py | ✅ 大部分已修复 |
| BUG-SM-011 | 照片完整性 16bit 降级 | manifest_loader.py | ✅ 无需修复 |
| BUG-SM-012 | audio_override 未合并 | manifest_generate.py | ✅ 已修复 |
| BUG-SM-013 | 照片类别推荐参数为空 | manifest_generate.py | ✅ 已修复 |
| BUG-SM-014 | 清单编辑器缺交互式 TUI | manifest_editor.py | ✅ 已修复 |
| P-45 | 缺少 ROCm/MPS 检测 | dependency_check.py | ✅ 已修复 |
| P-46 | 缺少 spawn 三约束预检 | dependency_check.py | ✅ 已修复 |
| P-47 | compand 动态范围压缩未实现 | audio_utils.py | ✅ 已修复 |
| P-54 | `_verify_photo_integrity` 16bit 降级 | manifest_loader.py | ✅ 无需修复 |
| P-59 | `classify_item` 动态 sys.path | manifest_generate.py | ✅ 已修复 |
| P-63 | `wsl_redirect_path` 对 `/mnt/` 无重定向 | checkpoint.py | ✅ 已修复 |
| P-66 | `is_retryable_error` 跨平台信号 | checkpoint.py | ✅ 已修复 |
| P-67 | `estimate_disk_space` 精度丢失 | checkpoint.py | ✅ 已修复 |
| P-68 | `gpu_memory_check` "trt" 缺乏依据 | checkpoint.py | ✅ 已修复 |
| **SM-ARCH-011** | WSL 重定向未检查 temp_dir | checkpoint.py | ✅ 已修复 |
| **SM-ARCH-012** | 跨文件系统未备份旧文件 | checkpoint.py | ✅ 已修复 |
| **SM-ARCH-013** | 人脸检测包名不匹配 | dependency_check.py | ✅ 已修复 |
| **SM-ARCH-014** | AI 模型 MD5 校验未启用 | dependency_check.py | ✅ 已修复 |
| **SM-ARCH-015** | 预览未实现运动幅度检测 | preview.py | ✅ 已修复 |
| **P12-12** | G'MIC `-film_grain` 不存在，照片 grain 合成从未生效 | photo/stage5_postprocess.py | ⚠️ Round 13 修复中 |


---

## Video Restoration Pipeline 架构审查更新（2026-04-29 全面复审）

> 本次审查对 video_restoration_pipeline/ 全部 11 个文件 + shared_modules/ 核心文件进行了纯手动复审，依据 v6.32_revised.md + v7.32_revised.md。

### 主控脚本 (main.py)

**已实现**:
- ✅ 六种运行模式（scan/edit/batch/quick/interactive/preview）
- ✅ 参数六层合并（global → paths → audio_override → recommended_params → user_overrides → CLI）
- ✅ `force_traditional_first` 门控（stage2 失败返回 86 阻止 AI）
- ✅ WSL2 路径重定向（process_job 开头统一处理 temp_dir + output_dir）
- ✅ FAILED 任务降级 + resume 断点续传
- ✅ PARAM_OVERRIDE 日志记录
- ✅ ProcessPoolExecutor 并行框架（spawn 模式）
- ✅ 预览模式（静态帧 + 0.5s WebP 动画）

**现存问题**:
- ~~🔴 **Critical**: spawn 模式下 `args` 为 dict，但 `process_job()` 使用 `args.config` / `args.stage` → AttributeError~~ ✅ **已修复**（BUG-042）
- ~~🔴 **Critical**: Resume 模式错误标记 DONE~~ ✅ **已修复**（BUG-043）
- ~~🟡 缺少 `--ai-enabled` / `--ai-upscale-factor` / `--face-restore` CLI 参数~~ ✅ **已修复**（BUG-047）
- ~~🟡 Preview 模式未生成 AI 放大前后对比帧~~ ✅ **已实现**（ISSUE-V-MAIN-002：Lanczos 模拟对比帧）
- ~~🟡 参数 JSON 命令行长度风险~~ ✅ **已修复**（ISSUE-V-MAIN-003：`--params-file` 回退）

### Stage 0: Audio Extraction

**状态**: ✅ 音频链已修复 — stage0 输出 `.audio`，stage7 兼容旧命名 `.wav` fallback。

### Stage 1: Preprocessing

**已实现**: VS 脚本生成完整（QTGMC, IVTC, 降噪, 去块, 色域转换, 黑边裁剪/还原, 16bit, PSF, 变场序分段）

**问题**:
- VFM-VDecimate 物理相邻性需确认（字符串拼接中间是否插入注释）
- sRestore 自动回退逻辑待验证

### Stage 2: Traditional Check

**状态**: ✅ 校验失败阻止 AI 阶段已生效。

### Stage 3: AI Upscale

**已实现**: 2x/4x 放大、256帧滑动窗口、OOM 降级（4x→2x→CPU）、原始帧率保留、Lanczos 二次缩放、≤1080p 检查。

**问题**:
- 帧提取精度待确认（是否为 16bit 或 8bit）
- Python API fallback 代码健壮性

### Stage 4: Postprocess

**问题**:
- ~~🟡 使用全局 edgedetect+gblur+unsharp，未实现真正的局部遮罩区域分离~~ ✅ **已修复**（工程折中）：`stage4_postprocess.py` 已实现基于 edgedetect+geq+gblur+maskedmerge 的局部遮罩处理框架（flat/detail 双分支 + ≥8px 羽化过渡）
- ~~🟡 缺少最终调色（lift/gamma/gain），仅使用简单 `eq`~~ ✅ **已修复**：`stage4_postprocess.py` 已使用 `colorbalance` 实现 lift/gamma/gain（rs/gs/bs + rm/gm/bm + rh/gh/bh），参数从 `params` 读取

### Stage 5: AI Face

**已实现**: 人脸稳定性（位移<5%、面积>15%）、静帧检测（SSIM>0.99）、羽化混合（Gaussian blur mask）、FFV1 中间格式、param_hash 文件名、实际帧率读取。

### Stage 6: Grain Synthesis

**已实现**: 16bit YUV420P16、GrainFactory3 + AddGrainC fallback、FFV1 中间格式。

### Stage 7: Encode

**已实现**: SVT-AV1 自适应 preset/CRF、音频处理链（bandreject→arnndn/afftdn→compand→asetpts）、多声道 FLAC 保留、AV 同步验证、WSL copyback + 原子重命名。

**问题**:
- 音频路径查找使用 glob 模式，可能匹配旧任务残留

### 已知视频架构级问题汇总（已更新）

| ID | 问题 | 位置 | 影响 | 状态 |
|----|------|------|------|------|
| V-ARCH-001 | ~~运行模式参数缺失~~ | video/main.py | — | ✅ 已修复 |
| V-ARCH-002 | ~~force_traditional_first 未强制~~ | video/main.py | — | ✅ 已修复 |
| V-ARCH-003 | ~~WSL2 重定向不完整~~ | video/main.py + stage7 | — | ✅ 已修复 |
| V-ARCH-004 | ~~音频链断裂~~ | stage0 + stage7 | — | ✅ 已修复 |
| V-ARCH-005 | 帧精度待确认 | stage3 | 8bit/16bit | ⚠️ 需验证 |
| V-ARCH-006 | ~~帧率丢失~~ | stage3 | — | ✅ 已修复 |
| V-ARCH-007 | 局部处理缺失 | stage4 | 全局滤镜代替遮罩 | ✅ 已修复（工程折中） |
| V-ARCH-008 | ~~人脸修复不完整~~ | stage5 | — | ✅ 已修复 |
| V-ARCH-009 | ~~颗粒合成位深错误~~ | stage6 | — | ✅ 已修复 |
| V-ARCH-010 | ~~中间格式有损~~ | stage5 + stage6 | — | ✅ 已修复 |
| V-ARCH-011 | spawn 模式 AttributeError | video/main.py | 并行处理崩溃 | ✅ 已修复 |
| V-ARCH-012 | Resume 错误标记 DONE | video/main.py | 任务被跳过 | ✅ 已修复 |
| V-ARCH-013 | 磁盘空间估算硬编码 fps | video/main.py | PAL 估算偏差 20% | 🟡 Medium |
| V-ARCH-014 | 分类器未返回 duration | video/classify.py | 超时/估算不准 | ✅ 已修复 |

完整的 per-file audit 报告详见项目根目录 `video_*-audit.md` 系列文件。


---

## Shared Modules 架构审查更新 (2026-04-29 复审)

### 新增 shared_modules 架构级问题汇总

| ID | 问题 | 位置 | 影响 | 状态 |
|----|------|------|------|------|
| SM-ARCH-001 | WSL2 重定向方向相反 | checkpoint.py | `/mnt/` 路径 I/O 性能瓶颈 | ✅ 已修复 |
| SM-ARCH-002 | WSL2 成品回迁无校验 | checkpoint.py | 文件损坏风险 | ✅ 已修复 |
| SM-ARCH-003 | GPU 显存检测使用总量 | checkpoint.py | 运行时 OOM | ✅ 已修复 |
| SM-ARCH-004 | 参数哈希缺少照片参数 | checkpoint.py | 断点续传精度错误 | ✅ 已修复 |
| SM-ARCH-005 | Windows 日志锁失效 | logger.py | 多进程日志损坏 | ✅ 已修复 |
| SM-ARCH-006 | 清单编辑无 TUI | manifest_editor.py | 批量审阅不可用 | ✅ 已修复 |
| SM-ARCH-007 | 预览模式差距极大 | preview.py | 无法评估运动修复效果 | ✅ 大部分已修复 |
| SM-ARCH-008 | 依赖检查缺 CodeFormer | dependency_check.py | 运行时模型缺失 | ✅ 已修复 |
| SM-ARCH-009 | 音频默认 codec 不一致 | audio_utils.py | 输出格式不符合预期 | ✅ 已修复 |
| SM-ARCH-010 | xxhash 算法不兼容 | checkpoint.py | 跨工具哈希值不一致 | ✅ 已合规 |
| SM-ARCH-011 | WSL 重定向未检查 temp_dir | checkpoint.py | 中间文件仍在 /mnt/ | ✅ 已修复 |
| SM-ARCH-012 | 跨文件系统未备份旧文件 | checkpoint.py | 覆盖异常时文件损坏 | ✅ 已修复 |
| SM-ARCH-013 | 人脸检测包名不匹配 | dependency_check.py | retinaface vs retina-face | ✅ 已修复 |
| SM-ARCH-014 | AI 模型 MD5 校验未启用 | dependency_check.py | 无法检测模型损坏 | ✅ 已修复 |
| SM-ARCH-015 | 预览未实现运动幅度检测 | preview.py | 无法提取最剧烈片段 | ✅ 已修复 |

### Shared Modules 问题分布（复审后）

| 模块 | Critical | High | Medium | Low | 合规度 |
|------|----------|------|--------|-----|--------|
| dependency_check.py | 0 | 1 | 3 | 2 | 82% |
| checkpoint.py | 0 | 2 | 4 | 3 | 75% |
| logger.py | 0 | 0 | 1 | 2 | 92% |
| audio_utils.py | 0 | 0 | 2 | 2 | 88% |
| manifest_generate.py | 0 | 0 | 2 | 2 | 90% |
| manifest_loader.py | 0 | 0 | 1 | 1 | 92% |
| manifest_editor.py | 0 | 0 | 1 | 1 | 90% |
| preview.py | 0 | 1 | 2 | 2 | 72% |

完整的 per-file audit 报告详见项目根目录 `*-audit.md` 系列文件。

### Shared Modules 第二轮复审新增架构级问题（2026-04-29）

| ID | 问题 | 位置 | 影响 | 状态 |
|----|------|------|------|------|
| SM-ARCH-016 | 中间文件清理默认参数失效 | checkpoint.py | 旧中间文件无限累积，磁盘空间耗尽 | ✅ 已修复 |
| SM-ARCH-017 | 高比特深帧完整性误判 | checkpoint.py | 16bit 中间文件被误判损坏，触发重复处理 | ✅ 已修复 |
| SM-ARCH-018 | avifenc 版本未预检 | dependency_check.py | 旧版 avifenc 传入 `--jobs` 报错 | ✅ 已修复 |
| SM-ARCH-019 | spawn 合规未覆盖照片 | dependency_check.py | 照片多进程运行时可能崩溃 | ✅ 已修复 |
| SM-ARCH-020 | 音频流存在性未预检 | audio_utils.py | 无音频流素材产生 ERROR 日志 | ✅ 已修复 |
| SM-ARCH-021 | 照片降级校验不完整 | manifest_loader.py | 损坏的照片中间文件无法被检测 | ✅ 已修复 |
| SM-ARCH-022 | 预览两阶段效率低 | preview.py | I/O 开销大，异常时残留临时文件 | ✅ 已修复 |

### Shared Modules 问题分布（第二轮复审后）

| 模块 | Critical | High | Medium | Low | 合规度 |
|------|----------|------|--------|-----|--------|
| checkpoint.py | 0 | 0 | 2 | 3 | ~84% |
| dependency_check.py | 0 | 0 | 1 | 3 | ~91% |
| logger.py | 0 | 0 | 1 | 2 | 92% |
| audio_utils.py | 0 | 0 | 1 | 3 | ~91% |
| manifest_generate.py | 0 | 0 | 2 | 2 | 90% |
| manifest_loader.py | 0 | 0 | 1 | 1 | ~93% |
| manifest_editor.py | 0 | 0 | 1 | 1 | 90% |
| preview.py | 0 | 0 | 1 | 3 | ~86% |
