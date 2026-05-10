import sys
content = """
---

## 审查批次 2：video stage 文件

### [ISSUE-038] video stage1_preprocess.py — 当 field_order 为 tt/bb 时未生成 QTGMC 调用（严重）
- **规范依据**: v7.32 §3.10.1 / v6.32 §7.15
- **问题描述**: 子代理发现：当 field_order 为 tt/bb 时，生成的 VS 脚本只添加了注释，但没有实际发出 haf.QTGMC() 调用。QTGMC 调用被错误地放在了 elif field_order == "unknown" 分支中。这意味着真实隔行素材完全跳过去隔行处理。
- **影响**: 真实隔行视频不会被去隔行，输出仍为隔行格式，质量严重受损。
- **修复思路**: 将 QTGMC 调用移至 if field_order in ("tt", "bb") 分支内。

### [ISSUE-039] video stage1_preprocess.py — IVTC 代码片段与 QTGMC 顺序未做硬校验
- **规范依据**: v7.32 §3.10.13 / v6.32 §7.13
- **问题描述**: v6.32 要求 IVTC 代码片段必须物理上位于 QTGMC 之前，且 needs_ivtc: true 时若 QTGMC 先于 IVTC 必须拒绝执行并标记 FAILED。当前代码没有这种硬校验。
- **影响**: 可能产生错误的处理顺序。
- **修复思路**: 在生成 VS 脚本后，检查 IVTC 和 QTGMC 的行号顺序。

### [ISSUE-040] video stage1_preprocess.py — 黑边裁剪后未校验有效画面 >= 50% 且 >= 240px
- **规范依据**: v7.32 §3.10.3 / v6.32 §7.19
- **问题描述**: 虽然代码中有 validate_crop() 函数，但需要确认它是否实现了规范要求的：裁剪后有效画面 < 原始 50% 时标记 FAILED，以及有效画面宽/高均不得低于 240px。
- **影响**: 极端裁剪情况下可能输出无效画面。
- **修复思路**: 在 validate_crop 中增加这两个硬校验。

### [ISSUE-041] video stage2_traditional_check.py — 未在 progressive < 90% 时返回失败码
- **规范依据**: v7.32 §3.10.13 / v6.32 §7.13
- **问题描述**: 子代理发现：stage2 在 progressive percentage 低于阈值时只记录 WARNING 并返回 0，没有自动触发 IVTC 回退或返回失败码。规范要求当 TFF+BFF > 10% 且 Progressive < 90% 时自动触发回退逻辑。
- **影响**: 去隔行质量差时不会触发回退，直接进入后续 stage。
- **修复思路**: 当 progressive < 90% 时返回非零退出码（如 86 校验失败），或在 stage2 内直接触发 sRestore 回退。

### [ISSUE-042] video stage3_ai_upscale.py — 硬编码 -framerate 25 未保留原始帧率
- **规范依据**: v7.32 §3.10.1 / v6.32 视频阶段 3
- **问题描述**: 子代理发现 line 419 硬编码 -framerate 25，没有保留原始视频的帧率。规范要求保持原始帧率。
- **影响**: 非 25fps 素材（如 NTSC 29.97fps、PAL 50fps）会被错误地重采样为 25fps。
- **修复思路**: 从原始视频读取 fps 并传给 -framerate 参数。

### [ISSUE-043] video stage3_ai_upscale.py — GPU 显存检查未实现规范要求的分级降级
- **规范依据**: v7.32 §3.10.23 / v6.32 §7.14
- **问题描述**: 虽然代码中有 check_gpu_memory() 和 GPU OOM retry 逻辑，但需要确认是否完整实现了：del model + torch.cuda.empty_cache() + 100ms wait + 4x->2x->CPU 分级降级。
- **影响**: GPU OOM 时可能无法正确降级。
- **修复思路**: 对照规范检查并完善。

### [ISSUE-044] video stage5_ai_face.py — 静态段落判定未完全实现 v7.32 要求
- **规范依据**: v7.32 §3.10.5 / v6.32 4A
- **问题描述**: 子代理发现：frame-to-time conversion 假设 25fps 而非使用实际帧率。此外，虽然代码有 _get_frame_pts() 和 _estimate_motion()，但需要确认是否完整实现了：VFR PTS 累加 >=2s、PTS 异常处理（超过 2 倍帧间隔中断）、重复帧容错、低分辨率 fallback 阈值 0.6、全局运动排除（>3%）。
- **影响**: 静态段落判定可能不准确，导致运动时人脸修复产生闪烁。
- **修复思路**: 逐条对照规范 §3.10.5 检查并补充缺失逻辑。

### [ISSUE-045] video stage5_ai_face.py — 最终编码硬编码编码器参数
- **规范依据**: v7.32 §3.10.3 / v6.32 阶段 3
- **问题描述**: 子代理发现最终输出编码使用硬编码 libsvtav1 CRF 24 / yuv420p10le，没有读取 config 中的编码参数。
- **影响**: 用户配置的 CRF/preset/色深等参数被忽略。
- **修复思路**: 从 params 读取编码参数。

### [ISSUE-046] video stage6_grain_synthesis.py — sys.path 未包含 shared_modules
- **规范依据**: v7.32 文件结构
- **问题描述**: 子代理发现 sys.path.insert(0, str(SCRIPT_DIR)) 只添加了 stages/ 目录，没有添加 ../shared_modules，导致 from logger import get_logger 运行时失败。
- **影响**: stage6 无法记录日志，可能崩溃。
- **修复思路**: 修正 sys.path 插入路径为 SCRIPT_DIR.parent.parent / "shared_modules"。

### [ISSUE-047] video stage7_encode.py — SVT-AV1 参数字符串重复构建
- **规范依据**: —
- **问题描述**: 子代理发现 SVT-AV1 参数字符串被构建了两次（冗余）。虽然不影响功能，但代码质量下降。
- **影响**: 代码冗余。
- **修复思路**: 提取为变量复用。

### [ISSUE-048] video stage7_encode.py — WSL copyback 失败后返回 82 但 Linux 侧文件仍存在
- **规范依据**: v7.32 §3.10.10 / v6.32 §7.3
- **问题描述**: 子代理发现：WSL copyback 失败后返回 82，但此时编码成功的输出文件仍存在于 Linux 原生分区。规范要求 "保留 Linux 原生分区的源文件，不清理任何临时数据"，当前行为部分符合，但用户可能困惑。
- **影响**: 用户不知道成功文件在 Linux 侧的位置。
- **修复思路**: 在返回 82 前记录 ERROR 日志，明确告知 Linux 侧文件路径。

---

## 审查批次 3：photo stage 文件

### [ISSUE-049] photo stage1_preprocess.py — G'MIC 失败静默处理
- **规范依据**: v7.32 §3.10.16 / v6.32 7.7
- **问题描述**: 子代理发现 run_gmic 失败时静默继续执行（没有记录 stderr 或返回错误）。规范要求 G'MIC 调用使用 shell=False + 列表传参，且失败时应记录 ERROR。
- **影响**: G'MIC 处理失败时用户无从知晓。
- **修复思路**: 检查 G'MIC 返回码，非零时记录 ERROR 并返回失败码。

### [ISSUE-050] photo stage1_preprocess.py — 16bit 处理路径下采样到 8bit
- **规范依据**: v7.32 §3.10.6 / v6.32 7.16
- **问题描述**: 子代理发现 16-bit 路径 (I;16) 通过 p / 256 下采样到 8-bit 后处理，丢失了高比特深度精度。规范要求全程 16bit/32bit 浮点处理。
- **影响**: 16bit 照片处理精度下降，色带风险增加。
- **修复思路**: 使用 numpy float32/64 处理，避免提前下采样到 8bit。

### [ISSUE-051] photo stage2_traditional_check.py — 校验时强制转 8bit RGB 丢失精度
- **规范依据**: v7.32 §3.10.8 / v6.32 7.1
- **问题描述**: 子代理发现质量检查强制 img.convert("RGB")（8-bit），导致 16-bit 输入在验证阶段丢失精度。
- **影响**: 16bit 中间文件的校验不准确。
- **修复思路**: 根据实际位深进行校验，避免强制转 8bit。

### [ISSUE-052] photo stage3_ai_upscale.py — _resolve_input_source 忽略了 ban 参数
- **规范依据**: v7.32 §3.10.2 / v6.32 4B
- **问题描述**: 子代理发现 _resolve_input_source 接受 ban_original/ban_intermediate 参数但忽略它们，始终返回 stage1 输出。这导致 AI 超分禁令（如原始文件已 >=4K）无法生效。
- **影响**: 超分禁令被绕过。
- **修复思路**: 根据 ban 参数选择正确的输入源，若被 ban 则返回跳过码 3 或报错。

### [ISSUE-053] photo stage3_ai_upscale.py — DAT 模型使用 ncnn-vulkan 未经验证
- **规范依据**: v7.32 §3.10.2 / v6.32 4B
- **问题描述**: 子代理发现 run_realesrgan 对 DAT 模型使用与 Real-ESRGAN 相同的 ncnn-vulkan 参数，但没有验证 DAT 是否被 ncnn-vulkan 支持。规范提到 DAT 的 ONNX 导出兼容性参差不齐，推荐优先使用官方 PyTorch 脚本。
- **影响**: DAT 模型可能无法通过 ncnn-vulkan 正确加载。
- **修复思路**: 对 DAT 模型使用 PyTorch API 而非 ncnn-vulkan。

### [ISSUE-054] photo stage4_postprocess.py — 16bit 路径下采样到 8bit
- **规范依据**: v7.32 §3.10.6 / v6.32 照片阶段 1
- **问题描述**: 子代理发现 16-bit 路径 (I;16) 通过 p / 256 下采样到 8-bit，丢失高比特深度精度。
- **影响**: 同 ISSUE-050。
- **修复思路**: 使用 numpy float32/64 处理，保持 16bit 精度直到最终编码。

### [ISSUE-055] photo stage4_postprocess.py — _extract_flif_chrominance 为 placeholder 未实现
- **规范依据**: v7.32 架构要求
- **问题描述**: 子代理发现 _extract_flif_chrominance 只是保存 Cb/Cr 为 PNG，不是真正的 FLIF 格式。
- **影响**: 若后续 stage 依赖 FLIF chrominance，会失败。
- **修复思路**: 实现真正的 FLIF 提取，或移除该 placeholder。

### [ISSUE-056] photo stage5_ai_face.py — run_codeformer fallback 仅复制输入
- **规范依据**: v7.32 §3.10.5 / v6.32 4A
- **问题描述**: 子代理发现 run_codeformer 的 fallback 逻辑只是复制输入图像，没有执行任何修复。规范要求 CodeFormer 作为 GFPGAN 的备选，应实际执行修复。
- **影响**: 选择 CodeFormer 时实际上没有进行人脸修复。
- **修复思路**: 实现 CodeFormer 的 Python API 调用。

### [ISSUE-057] photo stage5_ai_face.py — 未将人脸框信息传递给修复引擎
- **规范依据**: v7.32 §3.10.5 / v6.32 4A
- **问题描述**: 子代理发现检测到的人脸边界框和数量没有传递给 run_gfpgan 或 run_codeformer。修复引擎可能重新检测或使用全图。
- **影响**: 修复引擎可能无法精准定位人脸区域。
- **修复思路**: 将检测到的人脸坐标传递给修复函数。

### [ISSUE-058] photo stage6_encode.py — 未找到中间文件时静默回退到原始输入
- **规范依据**: v7.32 §3.10.11 / v6.32 照片阶段 2
- **问题描述**: 子代理发现：若未找到处理后的中间文件，stage6 静默回退到原始输入文件 (args.input)，可能输出未处理的文件。
- **影响**: 用户可能误以为文件已处理，实际上得到的是原始文件。
- **修复思路**: 若中间文件缺失，返回错误码 82 而非静默回退。

---

*审查批次 1-3 完成。后续将创建 audit 报告并更新文档。*
"""
with open(r'C:\code1\old-video-and-picture-fix\shixian.md', 'a', encoding='utf-8') as f:
    f.write(content)
print('Done')
