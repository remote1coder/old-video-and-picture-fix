import sys
content = """

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
"""
with open(r'C:\code1\old-video-and-picture-fix\bugs.md', 'a', encoding='utf-8') as f:
    f.write(content)
print('Done')
