# 项目 Bug 追踪 (bugs.md)

> 本文件记录代码审查中发现的纯代码缺陷（逻辑错误、崩溃风险、未定义引用等），与规范合规性问题分开记录。

---

## Bug 列表

### [BUG-001] checkpoint.py 未定义 `is_retryable_error` 函数（ImportError 风险）
- **位置**: `shared_modules/checkpoint.py`（缺失） / `video_restoration_pipeline/main.py` 第 360 行
- **严重性**: 🔴 高（导致崩溃）
- **描述**: video main.py 尝试 `from checkpoint import is_retryable_error`，但该函数在 checkpoint.py 中不存在。任何需要重试的作业都会导致 ImportError，整个 pipeline 崩溃。
- **复现**: 运行 video main.py，当某个 stage 返回非零退出码时触发。

### [BUG-002] manifest_loader.py 引用不存在的 `verify_ffprobe_decodeable`
- **位置**: `shared_modules/manifest_loader.py` 第 112-143 行 / `shared_modules/checkpoint.py`（缺失）
- **严重性**: 🔴 高（导致崩溃）
- **描述**: `_handle_failed_job` 尝试 `from checkpoint import verify_ffprobe_decodeable, verify_frame_integrity`，但 checkpoint.py 中没有 `verify_ffprobe_decodeable` 函数。加载包含 FAILED 任务的 manifest 时会触发 ImportError。
- **复现**: `python -c "from manifest_loader import load_manifest, filter_jobs; m=load_manifest('xxx.yaml'); filter_jobs(m)"`（manifest 中包含 FAILED 任务）

### [BUG-003] video main.py `_worker_wrapper` 嵌套函数无法被 spawn 序列化
- **位置**: `video_restoration_pipeline/main.py` 第 430-434 行
- **严重性**: 🟡 中（Windows/WSL 多进程失败）
- **描述**: `_worker_wrapper` 定义在 `main()` 函数内部，是嵌套函数。`multiprocessing.set_start_method("spawn")` 下，嵌套函数无法被 pickle 序列化，会导致 `AttributeError: Can't pickle local object`。
- **复现**: Windows 或 WSL2 下运行 `python video_restoration_pipeline/main.py --workers 2 ...`

### [BUG-004] logger.py Windows 分支 `msvcrt.locking` 使用 `LK_NBLCK` 导致非阻塞锁失效
- **位置**: `shared_modules/logger.py` 第 112-114 行
- **严重性**: 🟡 中（Windows 多进程日志混乱）
- **描述**: `msvcrt.locking` 使用 `LK_NBLCK`（非阻塞锁），若锁获取失败（文件已被其他进程锁定），函数立即返回但不报错，代码继续执行写入操作。这导致多进程并发写入时没有任何实际同步效果。
- **复现**: Windows 下多进程同时写入同一日志文件。

### [BUG-005] checkpoint.py `verify_frame_integrity` 16bit 亮度范围计算错误
- **位置**: `shared_modules/checkpoint.py` 第 330 行
- **严重性**: 🟡 中（误判合法帧）
- **描述**: 16bit limited range 的亮度范围使用 `2560, 37632`（基于 `16*160` 和 `235*160`），但正确的计算应为 `16*256=4096` 和 `235*256=60160`。错误的阈值会导致大量合法 16bit 帧被误判为损坏。
- **复现**: 处理 16bit 视频中间文件时，校验失败。

### [BUG-006] video main.py `args.config` 为 Path 对象时与 str 混用
- **位置**: `video_restoration_pipeline/main.py` 第 36-37 行
- **严重性**: 🟢 低
- **描述**: `parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))` 使用 `str()` 转换，但 `SCRIPT_DIR` 是 `Path` 对象，`SCRIPT_DIR / "config.yaml"` 也是 `Path`。这本身没问题，但其他位置（如 `load_config(args.config)`）期望 str，而 Path 也兼容。非严重问题。

### [BUG-007] photo main.py `_is_retryable_error` 包含未定义退出码 88
- **位置**: `photo_restoration_pipeline/main.py` 第 217 行
- **严重性**: 🟢 低
- **描述**: `retryable_codes = {82, 83, 84, 86, 87, 88}` 中 88 不在规范定义范围内。不会导致崩溃，但语义不清晰。

### [BUG-008] photo main.py stage 索引始终为 6
- **位置**: `photo_restoration_pipeline/main.py` 第 287 行
- **严重性**: 🟢 低
- **描述**: `_run_stage` 的 `stage_idx` 参数传入 `len([s for s, _ in stages])`，这始终等于 6，不是当前 stage 的实际索引。

### [BUG-009] video main.py `ai_enabled` vs `params.get("ai_enabled")` 键名不一致
- **位置**: `video_restoration_pipeline/main.py` 第 303 行
- **严重性**: 🟡 中
- **描述**: `params.get("ai_enabled", False)` 查找的是 `"ai_enabled"`，但 config.yaml 中 AI 启用键是 `ai.enabled`（嵌套）。merge_params 中将 ai_config 的项 flat 注入为 `params[k] = v`，所以 `ai_enabled` 实际上不会被注入（因为 config 中是 `enabled` 而非 `ai_enabled`）。这意味着 `ai_enabled` 检测可能始终为 False。
- **验证**: 需要确认 config.yaml 是否通过其他方式传递 `ai_enabled` 到 params。

### [BUG-010] audio_utils.py `arnndn=m='...'` 单引号在 FFmpeg 列表传参中的风险
- **位置**: `shared_modules/audio_utils.py` 第 257 行
- **严重性**: 🟢 低
- **描述**: `filters.append(f"arnndn=m='{arnndn_model}'")` 在路径中包含空格时，由于整个滤镜链作为单个字符串元素传给 `-af`，FFmpeg 内部解析可能会因单引号产生问题。虽然 `subprocess.run` 使用列表模式时已处理了 shell 转义，但 FFmpeg 的滤镜语法解析器对引号的处理仍需验证。

### [BUG-011] manifest_generate.py 并发安全风险
- **位置**: `shared_modules/manifest_generate.py` 第 186-187 行
- **严重性**: 🟢 低
- **描述**: YAML dump 未使用原子写入。多进程同时生成 manifest 到同一文件时可能产生损坏的 YAML。

### [BUG-012] checkpoint.py `get_file_xxhash` 缺少 xxhash 时的 MD5 回退未标注
- **位置**: `shared_modules/checkpoint.py` 第 28-42 行
- **严重性**: 🟢 低
- **描述**: 当 xxhash 模块不可用时回退到 MD5，但 param_hash 计算逻辑期望的是 xxhash64 前 16 位。MD5 和 xxhash64 的前 16 位碰撞概率和格式不同，应记录 WARNING 日志告知用户使用了回退哈希算法。

### [BUG-013] video classify.py `field_order` 返回值曾为 "mixed"（现已修复为 "unknown"）
- **位置**: `video_restoration_pipeline/classify.py` 第 229-235 行
- **严重性**: 🟢 低
- **描述**: 代码注释表明已修复 "mixed" → "unknown"，符合 v7.32 规范。当前代码正确。
- **状态**: ✅ 已修复，无需处理

### [BUG-014] photo main.py `args.force_traditional_first is None` 逻辑永远为 False
- **位置**: `photo_restoration_pipeline/main.py` 第 341 行
- **严重性**: 🟡 中
- **描述**: `parser.add_argument("--force-traditional-first", action="store_true", ...)` 定义了一个布尔标志，其值只能是 True 或 False，永远不会是 None。因此 `True if args.force_traditional_first is None else args.force_traditional_first` 实际上等价于 `args.force_traditional_first`（即默认 False）。这与注释说的 "Default force_traditional_first to True" 矛盾。
- **修复**: 应将 default 设为 True，或使用 `default=True`。

### [BUG-015] preview.py 未按规范实现 0.5 秒固定时长预览
- **位置**: `shared_modules/preview.py` 第 24-119 行
- **严重性**: 🟢 低
- **描述**: v7.32 §3.10.21 要求运动片段预览按 "固定时长 0.5 秒" 动态提取帧数（25fps→12帧，50fps→25帧，60fps→30帧）。当前 `extract_preview_frames` 使用固定 `num_frames=6`（或用户指定），没有根据 fps 动态计算。

### [BUG-016] logger.py `param_override` 未限制 source 有效值
- **位置**: `shared_modules/logger.py` 第 135-141 行
- **严重性**: 🟢 低
- **描述**: `source` 参数没有限制为规范要求的枚举值（user_overrides/paths/global）。调用方可能传入任意字符串。

---

*审查继续中...*


### [BUG-017] video stage1_preprocess.py — field_order 为 tt/bb 时未生成 QTGMC 调用
- **位置**: video_restoration_pipeline/stages/stage1_preprocess.py
- **严重性**: 🔴 高（导致严重质量问题）
- **描述**: 当 field_order 为 tt 或 bb 时，生成的 VapourSynth 脚本只添加了注释 # Deinterlace with QTGMC，但没有实际发出 haf.QTGMC() 调用。QTGMC 调用被错误地放在了 elif field_order == "unknown" 分支中。
- **复现**: 输入一个 field_order=tt 的视频，查看生成的 .vpy 脚本，其中不会有 QTGMC 调用。

### [BUG-018] video stage6_grain_synthesis.py — sys.path 未包含 shared_modules 导致 ImportError
- **位置**: video_restoration_pipeline/stages/stage6_grain_synthesis.py
- **严重性**: 🔴 高（运行时崩溃）
- **描述**: sys.path.insert(0, str(SCRIPT_DIR)) 只添加了 stages/ 目录，没有添加 ../shared_modules，导致 from logger import get_logger 运行时 ImportError。
- **复现**: 直接运行 stage6_grain_synthesis.py 即可触发。

### [BUG-019] video stage3_ai_upscale.py — 硬编码 -framerate 25 导致帧率错误
- **位置**: video_restoration_pipeline/stages/stage3_ai_upscale.py line 419
- **严重性**: 🟡 中（帧率错误）
- **描述**: 将超分后的 PNG 帧重新编码为 MKV 时，使用硬编码 -framerate 25，没有保留原始视频的帧率。
- **复现**: 输入 29.97fps 或 50fps 视频，观察输出帧率为 25fps。

### [BUG-020] checkpoint.py 未定义 is_retryable_error（重复确认）
- **位置**: shared_modules/checkpoint.py
- **严重性**: 🔴 高
- **描述**: video main.py 引用不存在的函数。
- **复现**: 任何 stage 返回非零退出码时触发 ImportError。

### [BUG-021] manifest_loader.py 引用不存在的 verify_ffprobe_decodeable（重复确认）
- **位置**: shared_modules/manifest_loader.py
- **严重性**: 🔴 高
- **描述**: 加载 FAILED 任务时触发 ImportError。
- **复现**: manifest 中包含 FAILED 状态时触发。

### [BUG-022] photo stage5_ai_face.py — run_codeformer fallback 仅复制输入
- **位置**: photo_restoration_pipeline/stages/stage5_ai_face.py
- **严重性**: 🟡 中（功能缺失）
- **描述**: run_codeformer 的 fallback 逻辑只是复制输入图像，没有执行 CodeFormer 修复。
- **复现**: 选择 CodeFormer 作为 face_restore，观察输出与输入相同。

### [BUG-023] photo stage6_encode.py — 未找到中间文件时静默回退到原始输入
- **位置**: photo_restoration_pipeline/stages/stage6_encode.py
- **严重性**: 🟡 中（输出未处理文件）
- **描述**: 若前面的 stage 未生成中间文件，stage6 静默回退到原始输入，用户可能得到未处理的文件。
- **复现**: 删除 temp 目录中的中间文件后运行 stage6。

### [BUG-024] photo stage3_ai_upscale.py — _resolve_input_source 忽略 ban 参数
- **位置**: photo_restoration_pipeline/stages/stage3_ai_upscale.py
- **严重性**: 🟡 中（禁令被绕过）
- **描述**: _resolve_input_source 接受 ban_original/ban_intermediate 但始终返回 stage1 输出。
- **复现**: 对已被 ban 的文件运行 stage3，仍然会执行超分。

### [BUG-025] logger.py Windows 文件锁定使用 LK_NBLCK 导致非阻塞锁失效
- **位置**: shared_modules/logger.py
- **严重性**: 🟡 中
- **描述**: msvcrt.locking 使用 LK_NBLCK，锁失败时仍继续写入。
- **复现**: Windows 下多进程同时写入同一日志文件。

### [BUG-026] video main.py _worker_wrapper 嵌套函数无法被 spawn 序列化
- **位置**: video_restoration_pipeline/main.py
- **严重性**: 🟡 中
- **描述**: _worker_wrapper 是嵌套函数，spawn 模式下无法 pickle。
- **复现**: Windows/WSL2 下使用 --workers 2 运行。

---

*Bug 列表更新完成。*
