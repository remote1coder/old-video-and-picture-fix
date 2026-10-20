import os

AUDITS = {
    "dependency_check-audit.md": """# dependency_check.py 审查报告

## 文件信息
- **路径**: shared_modules/dependency_check.py
- **行数**: 537
- **用途**: 依赖预检模块，验证外部工具、VS 插件、Python 包、AI 模型权重

## 规范合规性审查

### v7.32 §3.6 / v6.32 §7.18 依赖预检
| 要求 | 状态 | 说明 |
|------|------|------|
| 外部工具检查 (ffmpeg, ffprobe, vapoursynth, avifenc, exiftool, gmic) | ⚠️ 部分 | photo 模式下缺少 ffmpeg/ffprobe 检查；video 模式下缺少 avifenc 检查 |
| VS 插件加载校验 (大小写敏感) | ✅ 通过 | 使用 .lower() 匹配，支持别名 |
| QTGMC 底层链强制校验 | ✅ 通过 | 校验了 MVTools2, MaskTools2, NNEDI3 等 |
| TemporalDegrain2 依赖校验 | ✅ 通过 | 包含在 VS 插件检查中 |
| GrainFactory3 底层链校验 | ✅ 通过 | 包含 AddGrainC, RemoveGrain, DFTTest |
| havsfunc 版本锁定 r33 | ✅ 通过 | check_havsfunc_version() 检查 "33" |
| VS 滤镜最小参数集测试调用 | ✅ 通过 | check_vs_minimal_calls() 实现 |
| FFmpeg 版本校验 >=4.4 | ✅ 通过 | check_ffmpeg_version() 实现 |
| FFmpeg 滤镜可用性预检 | ⚠️ 部分 | 仅检查滤镜名存在，未执行最小参数集测试调用 |
| arnndn 模型文件存在性校验 | ⚠️ 部分 | 仅在 audio.enabled=true 时检查，未在 FFmpeg<4.4 时强制检查 |
| 人脸检测 Python 包校验 | ⚠️ 部分 | 检查 retinaface 而非规范的 retina-face，需确认包名等价性 |
| AI 模型权重校验 | ✅ 通过 | check_ai_models() 实现 |
| G'MIC 滤镜校验 | ✅ 通过 | check_gmic_filters() 实现 |
| GPU 后端扩展检测 (ROCm/MPS) | ❌ 缺失 | 未实现 CUDA -> ROCm -> MPS -> CPU 检测链 |
| sRestore 依赖校验 | ✅ 通过 | 包含在 check_vs_minimal_calls() 中 |

### v7.32 §3.5 / v6.32 §7.28 退出码
| 要求 | 状态 | 说明 |
|------|------|------|
| 依赖缺失返回 80 | ✅ 通过 | main() 中 sys.exit(80) |

## 发现的问题
- ISSUE-001: 缺少 ROCm/MPS GPU 后端检测
- ISSUE-002: 缺少 FFmpeg 滤镜最小参数集测试调用
- ISSUE-003: arnndn 模型检查未在 FFmpeg<4.4 时强制触发
- ISSUE-004: 人脸检测包名可能不匹配规范

## 总体评估
**合规度: 75%**
核心依赖检查逻辑完整，但 GPU 后端扩展检测和部分边缘场景（FFmpeg<4.4 fallback）未完全覆盖。
""",

    "checkpoint-audit.md": """# checkpoint.py 审查报告

## 文件信息
- **路径**: shared_modules/checkpoint.py
- **行数**: 475
- **用途**: 断点续传支持，param_hash 计算，中间文件命名，完整性校验，磁盘空间预估

## 规范合规性审查

### v7.32 §3.10.19 / v6.32 §7.9 param_hash
| 要求 | 状态 | 说明 |
|------|------|------|
| 纳入文件唯一标识 (size, mtime, hash) | ✅ 通过 | compute_param_hash() 实现 |
| 使用 xxhash64 (>1GB 分块采样) | ⚠️ 部分 | 使用 xxhash.xxh3_64() 但无分块采样优化 |
| 核心画质参数白名单 | ⚠️ 部分 | CORE_QUALITY_PARAMS 缺少 qtgmc_input_type 等参数 |
| SHA-256 前 16 位 | ✅ 通过 | _hash_dict() 实现 |
| 排除清单 (log_dir, output_path 等) | ✅ 通过 | 未将这些参数加入白名单 |
| 中间文件命名规范 | ✅ 通过 | build_intermediate_name() 实现 |
| 帧数据完整性校验 | ⚠️ 部分 | verify_frame_integrity() 存在但 16bit 阈值错误 |

### v7.32 §3.10.10 / v6.32 §7.3 WSL2 适配
| 要求 | 状态 | 说明 |
|------|------|------|
| 路径自动重定向 | ✅ 通过 | wsl_redirect_path() 实现 |
| 成品文件回迁 | ❌ 缺失 | wsl_copy_back() 未实现完整的 6 步回迁流程 |
| shutil.copyfile (非 copy2) | ❌ 缺失 | 使用 copy2 |
| .tmp 临时文件 + 三重校验 | ❌ 缺失 | 未实现 |
| 跨文件系统原子性保障 | ❌ 缺失 | 未实现 st_dev 比较 |
| 最多重试 3 次 | ❌ 缺失 | 未实现 |
| os.utime() 同步时间 | ❌ 缺失 | 未实现 |

### v7.32 §3.10.23 / v6.32 §7.14 GPU 显存预检测
| 要求 | 状态 | 说明 |
|------|------|------|
| 显存预估公式 | ❌ 缺失 | gpu_memory_check() 过于简化 |
| nvidia-smi MiB->字节转换 | ❌ 缺失 | 未实现 |
| 4x->2x->CPU 分级降级 | ❌ 缺失 | 未实现 |
| PyTorch 版本兼容性检查 | ❌ 缺失 | 未实现 |

### v7.32 §3.10.17 / v6.32 §7.5 中间文件清理
| 要求 | 状态 | 说明 |
|------|------|
| 保留最近 2 个 stage | ❌ 不符 | 默认 keep_steps=5 |
| FAILED 任务保护 | ⚠️ 部分 | 实现存在但默认行为不符 |

### v7.32 §3.10.18 / v6.32 §7.6 重试机制
| 要求 | 状态 | 说明 |
|------|------|
| is_retryable_error 函数 | ❌ 缺失 | 函数未定义，被 main.py 引用 |

## 发现的问题
- ISSUE-006: param_hash 无 >1GB 分块采样
- ISSUE-007: CORE_QUALITY_PARAMS 缺少参数
- ISSUE-008: 16bit limited range 阈值错误 (2560/37632 应为 4096/60160)
- ISSUE-009: WSL2 回迁逻辑不完整
- ISSUE-010: GPU 显存检查未实现分级降级
- ISSUE-011: 缺少 is_retryable_error 函数
- ISSUE-012: cleanup_old_intermediates 默认 keep_steps=5 应为 2
- ISSUE-013: estimate_disk_space 未实现 AI 分级系数

## 总体评估
**合规度: 55%**
核心 param_hash 和中间文件命名合规，但 WSL2 回迁、GPU 显存检测、重试机制等关键功能缺失或不完整。存在两个会导致运行时崩溃的缺失函数（is_retryable_error, verify_ffprobe_decodeable）。
""",

    "logger-audit.md": """# logger.py 审查报告

## 文件信息
- **路径**: shared_modules/logger.py
- **行数**: 167
- **用途**: 结构化 JSON Lines 日志，每 job 一个日志文件

## 规范合规性审查

### v7.32 §3.10.12 / v6.32 §7.4 日志轮转
| 要求 | 状态 | 说明 |
|------|------|------|
| 禁止使用 RotatingFileHandler 按字节轮转 | ✅ 通过 | 使用行数轮转 |
| 按 job 分离日志文件 | ✅ 通过 | StructuredLogger 按 job_id 创建文件 |
| 多进程安全写入 | ⚠️ 部分 | Linux 使用 fcntl.flock；Windows 使用 msvcrt.locking 但 LK_NBLCK 有缺陷 |
| 单行长度上限 1MB | ✅ 通过 | log_max_line_bytes 截断并记录 SHA-256 |
| 参数覆盖追溯 PARAM_OVERRIDE | ✅ 通过 | param_override() 方法实现 |
| 命令行参数优先级记录 | ⚠️ 部分 | 有 param_override 但需调用方显式调用 |
| 时间戳含时区 | ✅ 通过 | datetime.now().astimezone().isoformat() |

## 发现的问题
- ISSUE-014: Windows msvcrt.locking 使用 LK_NBLCK 非阻塞锁失效
- ISSUE-015: 日志轮转 rename 存在多进程竞争

## 总体评估
**合规度: 80%**
日志格式和基本功能合规，但 Windows 多进程安全性和日志轮转的竞争条件需要修复。
""",

    "audio_utils-audit.md": """# audio_utils.py 审查报告

## 文件信息
- **路径**: shared_modules/audio_utils.py
- **行数**: 377
- **用途**: 音频处理工具函数，包括哼鸣检测、滤波链构建、音频处理

## 规范合规性审查

### v7.32 §3.10.9 / v6.32 §7.12 音频处理
| 要求 | 状态 | 说明 |
|------|------|------|
| bandreject 基波+2-3次谐波 | ✅ 通过 | build_hum_filters() 实现 |
| width_type=h, width=2 | ✅ 通过 | 硬编码在 filter 字符串中 |
| 能量阈值 <0.5% 跳过 bandreject | ✅ 通过 | detect_hum_frequency() 实现 |
| 双哼鸣场景保守处理 | ✅ 通过 | 检测逻辑存在，默认只处理主频率 |
| FFT 自适应窗长 (85ms) | ✅ 通过 | _compute_fft_window_size() 实现 |
| Hann 窗 + 50% 重叠 | ✅ 通过 | detect_hum_frequency() 实现 |
| 多声道保留 | ✅ 通过 | get_audio_channel_layout() + codec 切换 |
| asetpts=PTS 在滤镜链末端 | ✅ 通过 | process_audio() 实现 |
| arnndn 先转 mono | ❌ 缺失 | 直接应用 arnndn，未先转 mono |
| compand 动态范围压缩 | ❌ 缺失 | 未实现 |
| 旧版 FFmpeg <4.4 降级到 arnndn | ⚠️ 部分 | 有 arnndn 支持但未显式检测 FFmpeg 版本并降级 |

## 发现的问题
- ISSUE-016: arnndn 未先转 mono
- ISSUE-017: compand 未实现

## 总体评估
**合规度: 80%**
核心音频处理逻辑（哼鸣检测、bandreject 链、afftdn/arnndn、asetpts=PTS）合规，但缺少 arnndn mono 转换和 compand 支持。
""",

    "manifest_generate-audit.md": """# manifest_generate.py 审查报告

## 文件信息
- **路径**: shared_modules/manifest_generate.py
- **行数**: 220
- **用途**: 扫描输入目录，分类文件，生成批量清单

## 规范合规性审查

### v7.32 §3.4.3 / v6.32 §7.9 清单生成
| 要求 | 状态 | 说明 |
|------|------|------|
| 递归扫描多路径 | ✅ 通过 | 实现 |
| 分类标签+置信度 | ✅ 通过 | classify_item() 调用 |
| 场序/IVTC/PSF 标记 | ✅ 通过 | 包含 field_order, needs_ivtc, is_psf |
| field_segments 存储 | ✅ 通过 | 包含 |
| ai_upscale_banned 硬标记 | ✅ 通过 | 包含 |
| original_width/height | ✅ 通过 | 包含 |
| native_width/height/dar | ✅ 通过 | 包含 |
| file_size/mtime/hash_prefix | ✅ 通过 | 包含 |
| recommended_params | ⚠️ 部分 | 未包含 audio_override 参数 |
| 置信度 <80% 标记 SKIP | ✅ 通过 | 实现 |
| YAML/JSONL 双格式 | ✅ 通过 | 实现 |

## 发现的问题
- ISSUE-018: file_hash_prefix 未使用 >1GB 分块采样
- ISSUE-019: audio_override 未合并到 recommended_params

## 总体评估
**合规度: 90%**
清单生成功能基本完整，仅 audio_override 合并和分块采样两个问题。
""",

    "manifest_loader-audit.md": """# manifest_loader.py 审查报告

## 文件信息
- **路径**: shared_modules/manifest_loader.py
- **行数**: 202
- **用途**: 加载 YAML/JSONL 清单，验证，过滤，排序

## 规范合规性审查

### v7.32 §3.4.5 / v6.32 §7.9 清单加载
| 要求 | 状态 | 说明 |
|------|------|------|
| YAML/JSONL 双格式 | ✅ 通过 | 实现 |
| 必填字段校验 | ✅ 通过 | validate_job() 实现 |
| 按 status 过滤 (PENDING/FAILED) | ✅ 通过 | filter_jobs() 实现 |
| FAILED 降级为 PENDING (中间文件有效) | ⚠️ 部分 | 实现但引用不存在的 verify_ffprobe_decodeable |
| 保留最近 2 个 stage 中间文件 | ⚠️ 部分 | 代码保留最近 2 个，但触发条件有限 |
| 按优先级排序 | ✅ 通过 | 实现 |

## 发现的问题
- ISSUE-020: FAILED 任务校验引用不存在的 verify_ffprobe_decodeable

## 总体评估
**合规度: 80%**
基本功能完整，但 FAILED 任务处理存在运行时崩溃风险。
""",

    "manifest_editor-audit.md": """# manifest_editor.py 审查报告

## 文件信息
- **路径**: shared_modules/manifest_editor.py
- **行数**: 85
- **用途**: 清单编辑工具（命令行）

## 规范合规性审查

### v7.32 §3.4.4 清单编辑
| 要求 | 状态 | 说明 |
|------|------|------|
| 终端交互式 TUI | ❌ 缺失 | 仅命令行子命令，无 TUI |
| 展示分类结果+推荐参数 | ❌ 缺失 | 未实现 |
| 确认/修改分类标签 | ❌ 缺失 | 未实现 |
| 修改参数 (user_overrides) | ⚠️ 部分 | set_user_override 仅支持单键值 |
| 标记 SKIP/PRIORITY | ⚠️ 部分 | 支持 SKIP，不支持 PRIORITY |

## 发现的问题
- 缺少交互式 TUI（rich 库）
- 缺少优先级标记支持
- 功能过于简单，不符合规范对 editor.py 的要求

## 总体评估
**合规度: 40%**
仅实现了最基本的命令行编辑功能，缺少规范要求的交互式 TUI 和大部分编辑能力。
""",

    "preview-audit.md": """# preview.py 审查报告

## 文件信息
- **路径**: shared_modules/preview.py
- **行数**: 147
- **用途**: 预览帧提取和运动片段生成

## 规范合规性审查

### v7.32 §3.10.21 / v6.32 §7.11 预览模式
| 要求 | 状态 | 说明 |
|------|------|------|
| 静态帧 (10%, 50%, 90%) | ⚠️ 部分 | 视频路径使用 evenly spaced 而非固定百分比 |
| 运动片段 0.5 秒固定时长 | ❌ 缺失 | generate_preview_clip 使用 10 秒，且未按 fps 动态计算帧数 |
| WebP 动画输出 | ❌ 缺失 | 输出 PNG 帧和 H.264 片段，未输出 WebP |
| 强制覆盖预览 (banned upscale) | ❌ 缺失 | 未实现 |

## 发现的问题
- BUG-015: 未按规范实现 0.5 秒固定时长预览
- 缺少 WebP 动画输出
- 缺少强制覆盖预览

## 总体评估
**合规度: 40%**
基本的帧提取功能存在，但与规范要求的预览模式差距较大。
""",

    "video_classify-audit.md": """# video_restoration_pipeline/classify.py 审查报告

## 文件信息
- **路径**: video_restoration_pipeline/classify.py
- **行数**: 574
- **用途**: 视频素材分类器，检测 ANALOG_TAPE/DV_DIGITAL/HDV_AVCHD、场序、PSF、IVTC

## 规范合规性审查

### v7.32 §3.9 / v6.32 §7.15 素材分类
| 要求 | 状态 | 说明 |
|------|------|------|
| ffprobe 元数据 | ✅ 通过 | run_ffprobe() 实现 |
| idet 逐帧分析 | ✅ 通过 | detect_field_order_segments() 实现 |
| PSF 检测 (Progressive >90% + field_order 非 progressive) | ✅ 通过 | 实现 |
| SSIM 二次校验 (>0.98) | ✅ 通过 | _check_psf_ssim() 实现 |
| 黑边检测排除静态区域 | ✅ 通过 | _detect_black_borders_frame() 实现 |
| SSIM 窗口强制奇数 + 兜底 | ✅ 通过 | 实现 |
| MediaInfo 辅助 (ScanType_StoreMethod) | ❌ 缺失 | 未调用 MediaInfo |
| field_order 五种有效值 | ✅ 通过 | tt/bb/psf/progressive/unknown |
| 变场序素材处理 (≥30帧窗口, ≥5帧分段) | ⚠️ 部分 | 使用 2秒窗口(~50帧)，基本满足但 step 较小 |
| 全场序正确率 <95% 回退 | ❌ 缺失 | 未实现 |
| AI 超分禁令 (像素尺寸基准) | ✅ 通过 | height>=1080 或 width>=1920 时 banned |
| 元数据辅助 (native_width/height/dar) | ✅ 通过 | 包含 |
| duration 字段 | ❌ 缺失 | 未返回 duration |

## 发现的问题
- ISSUE-021: 变场序窗口 step 较小（可优化）
- ISSUE-022: 未使用 MediaInfo 辅助 PSF 检测
- ISSUE-023: 未返回 duration 字段

## 总体评估
**合规度: 80%**
核心分类和场序检测逻辑完整且正确（尤其是 PSF 的 SSIM 二次校验），但缺少 MediaInfo 辅助和 duration 字段。
""",

    "photo_classify-audit.md": """# photo_restoration_pipeline/classify.py 审查报告

## 文件信息
- **路径**: photo_restoration_pipeline/classify.py
- **行数**: 163
- **用途**: 照片素材分类器，检测 SCAN_PRINT/DIGITAL_PHOTO、DPI、AI 超分禁令

## 规范合规性审查

### v7.32 §3.9 / v6.32 §5.3 照片分类
| 要求 | 状态 | 说明 |
|------|------|------|
| exiftool 元数据 | ✅ 通过 | run_exiftool() 实现 |
| 扫描仪检测 (大小写不敏感) | ✅ 通过 | Bug-18 已修复 |
| 像素尺寸硬校验 | ✅ 通过 | 长边>3840 或短边>2160 时 banned |
| DPI 阈值 >1200 禁止 | ✅ 通过 | 实现 |
| native_width/height/dar | ✅ 通过 | 包含 |
| file_size/mtime/hash_prefix | ✅ 通过 | 包含 |
| 双重基准 (原始输入 + 中间文件) | ⚠️ 部分 | 仅检查原始文件，未检查中间文件尺寸 |

## 发现的问题
- ISSUE-024: 未实现中间文件尺寸禁令检查

## 总体评估
**合规度: 90%**
照片分类功能简洁且基本合规，仅缺少中间文件尺寸检查（该检查更适合在 stage3 运行时执行）。
""",

    "video_main-audit.md": """# video_restoration_pipeline/main.py 审查报告

## 文件信息
- **路径**: video_restoration_pipeline/main.py
- **行数**: 456
- **用途**: 视频修复流水线总控入口

## 规范合规性审查

### v7.32 §3.3 / v6.32 §7.1 总控脚本
| 要求 | 状态 | 说明 |
|------|------|------|
| 四步工作流 (扫描->清单->审阅->执行) | ⚠️ 部分 | 支持扫描+执行，缺少交互式审阅 (--edit-manifest) |
| 两步简版 (--mode quick) | ❌ 缺失 | 未实现 |
| 批量执行 (--mode batch) | ⚠️ 部分 | 有批量执行但无显式 --mode 参数 |
| 断点续传 (--resume) | ❌ 缺失 | 未实现显式 resume 逻辑 |
| 参数优先级 (CLI > user_overrides > recommended_params > audio_override > paths > global) | ⚠️ 部分 | merge_params 实现基本正确但 audio_override 优先级有歧义 |
| WSL2 路径重定向 | ✅ 通过 | --no-wsl-redirect 支持 |
| 并行 worker 池 | ✅ 通过 | ProcessPoolExecutor 实现 |
| 磁盘空间预检 | ✅ 通过 | _check_disk_space() 实现 |
| stage 顺序强制 traditional 优先 | ❌ 缺失 | stage2 失败不阻止后续 AI stage |
| "Traditional restoration completed" 里程碑日志 | ❌ 缺失 | 未记录 |
| 动态超时 | ✅ 通过 | _calculate_timeout() 实现 |
| 重试机制 (2 次, 5s/15s) | ⚠️ 部分 | 有重试但引用不存在的 is_retryable_error |
| GPU fallback | ⚠️ 部分 | CLI 支持 --gpu 但无自动降级逻辑 |
| 预览模式 | ✅ 通过 | --preview 支持 |

### v7.32 §3.10.18 / v6.32 §7.6 PyTorch spawn
| 要求 | 状态 | 说明 |
|------|------|------|
| spawn 启动 | ✅ 通过 | multiprocessing.set_start_method("spawn") |
| 顶层全局函数 | ❌ 缺失 | _worker_wrapper 是嵌套函数 |
| 模块顶层无执行代码 | ✅ 通过 | 符合 |
| 预检逻辑 | ❌ 缺失 | 未实现 |

## 发现的问题
- ISSUE-025: merge_params 的 audio_override 优先级有歧义
- ISSUE-026: _worker_wrapper 嵌套函数无法被 spawn 序列化
- ISSUE-027: stage2 失败不阻止 stage3/4/5
- ISSUE-028: 未记录 "Traditional restoration completed" 里程碑
- ISSUE-029: _check_disk_space 缺少 AI 分级系数
- BUG-003: _worker_wrapper 嵌套函数 pickle 失败
- BUG-009: ai_enabled 键名可能不一致

## 总体评估
**合规度: 65%**
基本流程控制（stage 执行、参数合并、并行处理）已实现，但缺少关键的 spawn 规范合规、traditional-first 强制校验、以及一些会导致崩溃的引用错误。
""",

    "photo_main-audit.md": """# photo_restoration_pipeline/main.py 审查报告

## 文件信息
- **路径**: photo_restoration_pipeline/main.py
- **行数**: 406
- **用途**: 照片修复流水线总控入口

## 规范合规性审查

### v7.32 §3.3 / v6.32 §7.1 总控脚本
| 要求 | 状态 | 说明 |
|------|------|------|
| 四步工作流 | ❌ 缺失 | 仅支持单文件模式，无批量/清单支持 |
| 参数合并 (6 层) | ⚠️ 部分 | merge_params 实现但缺少 audio_override 和推荐参数正确获取 |
| 磁盘空间预检 | ✅ 通过 | _check_disk_space() 实现 |
| 重试机制 | ⚠️ 部分 | 有重试但 _is_retryable_error 包含未定义码 88 |
| 动态超时 | ✅ 通过 | _calculate_timeout() 实现 |
| force_traditional_first | ⚠️ 部分 | 默认 True 但逻辑有 Bug（BUG-014） |
| 预览模式 | ✅ 通过 | --preview 支持 |

## 发现的问题
- ISSUE-030: 缺少批量模式和清单支持
- ISSUE-031: merge_params 未处理 audio_override
- ISSUE-032: _is_retryable_error 包含未定义码 88
- ISSUE-033: stage2 失败不阻止后续 stage
- ISSUE-034: stage_idx 始终为 6
- BUG-014: force_traditional_first 逻辑永远为 False（默认）

## 总体评估
**合规度: 55%**
仅实现了最基本的单文件处理流程，缺少规范要求的大部分批量处理、清单系统和完整的参数合并逻辑。
""",

    "video_config-audit.md": """# video_restoration_pipeline/config.yaml 审查报告

## 文件信息
- **路径**: video_restoration_pipeline/config.yaml
- **行数**: 204
- **用途**: 视频修复套件配置

## 规范合规性审查

### v7.32 §3.2 / v6.32 §7.7 配置系统
| 要求 | 状态 | 说明 |
|------|------|------|
| global 区块 | ✅ 通过 | 完整 |
| paths 区块 (analog_tape/dv_digital/hdv_avchd) | ✅ 通过 | 完整 |
| audio_override 位置 | ⚠️ 部分 | 放在 paths.audio_override 下，规范示例为顶级 |
| grain_factory3 区块 | ⚠️ 部分 | 缺少 temp_avg 参数 |
| ai 区块 | ✅ 通过 | 完整 |
| audio 区块 | ✅ 通过 | 完整 |

## 发现的问题
- ISSUE-035: 缺少 grain_factory3.temp_avg
- ISSUE-036: audio_override 嵌套位置与规范示例不一致

## 总体评估
**合规度: 90%**
配置结构基本完整，仅两个小问题。
""",

    "photo_config-audit.md": """# photo_restoration_pipeline/config.yaml 审查报告

## 文件信息
- **路径**: photo_restoration_pipeline/config.yaml
- **行数**: 117
- **用途**: 照片修复套件配置

## 规范合规性审查

### v7.32 §3.2 / v6.32 §7.7 配置系统
| 要求 | 状态 | 说明 |
|------|------|------|
| global 区块 | ✅ 通过 | 完整 |
| paths 区块 (scan_print/digital_photo) | ✅ 通过 | 完整 |
| photo 区块 | ✅ 通过 | 完整 |
| avif_params / avif_rgb444_params | ✅ 通过 | 完整 |
| grain_factory3 空占位 | ✅ 通过 | 存在 |
| audio 空占位 | ⚠️ 部分 | 过于简单，可能导致 KeyError |

## 发现的问题
- ISSUE-037: audio 空占位过于简单

## 总体评估
**合规度: 95%**
照片配置基本完整。
""",
}

# Add stage file audits from sub-agent findings
AUDITS["video_stage1_preprocess-audit.md"] = """# video_stage1_preprocess.py 审查报告

## 文件信息
- **路径**: video_restoration_pipeline/stages/stage1_preprocess.py
- **行数**: 22320
- **用途**: 生成并执行 VapourSynth 脚本（IVTC、QTGMC、去噪、稳定、色域转换、黑边处理）

## 关键发现
### 🔴 严重问题
- **QTGMC 调用缺失**: 当 field_order 为 tt/bb 时，生成的脚本只添加注释，不发出 haf.QTGMC() 调用。调用被错误地放在 "unknown" 分支中。
- **IVTC 顺序未校验**: 未检查 IVTC 代码是否物理上位于 QTGMC 之前。

### 规范合规性
- 16bit 处理空间: 基本实现
- BT.601->BT.709 转换: 基本实现
- 黑边裁剪+还原: 实现但需验证 validate_crop() 的硬校验

## 总体评估
**合规度: 60%**
核心功能大部分实现，但存在严重的去隔行逻辑错误。
"""

AUDITS["video_stage2_traditional_check-audit.md"] = """# video_stage2_traditional_check.py 审查报告

## 文件信息
- **路径**: video_restoration_pipeline/stages/stage2_traditional_check.py
- **行数**: 4049
- **用途**: 验证 stage1 输出完整性和去隔行质量

## 关键发现
- ** progressive < 90% 时不返回失败码**: 仅记录 WARNING 并返回 0，未触发 IVTC 回退。

## 总体评估
**合规度: 50%**
校验逻辑存在但不触发回退，不符合规范要求。
"""

AUDITS["video_stage3_ai_upscale-audit.md"] = """# video_stage3_ai_upscale.py 审查报告

## 文件信息
- **路径**: video_restoration_pipeline/stages/stage3_ai_upscale.py
- **行数**: 16019
- **用途**: AI 超分 (Real-ESRGAN/DAT)

## 关键发现
- **硬编码 -framerate 25**: 未保留原始帧率。
- **GPU OOM 降级**: 基本实现但需验证完整分级降级链。

## 总体评估
**合规度: 70%**
核心超分逻辑实现，但帧率处理有缺陷。
"""

AUDITS["video_stage4_postprocess-audit.md"] = """# video_stage4_postprocess.py 审查报告

## 文件信息
- **路径**: video_restoration_pipeline/stages/stage4_postprocess.py
- **行数**: 3336
- **用途**: AI 超分后抗锯齿处理

## 总体评估
**合规度: 70%**
功能简单但基本合规，参数硬编码缺乏配置覆盖。
"""

AUDITS["video_stage5_ai_face-audit.md"] = """# video_stage5_ai_face.py 审查报告

## 文件信息
- **路径**: video_restoration_pipeline/stages/stage5_ai_face.py
- **行数**: 24445
- **用途**: 视频人脸检测与修复 (GFPGAN/CodeFormer)

## 关键发现
- **静态段落判定**: 基本实现 PTS 累加和 motion 估计，但 frame-to-time 假设 25fps。
- **最终编码硬编码参数**: 使用固定 libsvtav1 CRF 24。

## 总体评估
**合规度: 65%**
功能较复杂，静态段落判定基本框架存在但细节需完善。
"""

AUDITS["video_stage6_grain_synthesis-audit.md"] = """# video_stage6_grain_synthesis.py 审查报告

## 文件信息
- **路径**: video_restoration_pipeline/stages/stage6_grain_synthesis.py
- **行数**: 8133
- **用途**: GrainFactory3 颗粒合成

## 关键发现
- **🔴 sys.path 错误**: 未包含 shared_modules，导致 logger 导入失败。

## 总体评估
**合规度: 60%**
存在运行时崩溃风险。
"""

AUDITS["video_stage7_encode-audit.md"] = """# video_stage7_encode.py 审查报告

## 文件信息
- **路径**: video_restoration_pipeline/stages/stage7_encode.py
- **行数**: 14153
- **用途**: 最终编码 (SVT-AV1) + 音频处理 + 封装

## 关键发现
- **Scene 1/2 音频分支**: 已实现（asetpts=PTS vs atempo）
- **SVT-AV1 双重标记**: 已实现
- **WSL2 回迁**: 基本实现但 copyback 失败后返回 82

## 总体评估
**合规度: 75%**
核心编码和音频同步逻辑实现，但代码存在冗余。
"""

AUDITS["photo_stage1_preprocess-audit.md"] = """# photo_stage1_preprocess.py 审查报告

## 文件信息
- **路径**: photo_restoration_pipeline/stages/stage1_preprocess.py
- **行数**: 16824
- **用途**: 照片传统预处理（去划痕、去块、去色带、单通道检测）

## 关键发现
- **G'MIC 失败静默**: run_gmic 失败时继续执行，不报错。
- **16bit 下采样到 8bit**: I;16 路径通过 p/256 转换，丢失精度。

## 总体评估
**合规度: 60%**
功能实现但存在静默失败和精度丢失问题。
"""

AUDITS["photo_stage3_ai_upscale-audit.md"] = """# photo_stage3_ai_upscale.py 审查报告

## 文件信息
- **路径**: photo_restoration_pipeline/stages/stage3_ai_upscale.py
- **行数**: 10852
- **用途**: 照片 AI 超分

## 关键发现
- **_resolve_input_source 忽略 ban 参数**: 超分禁令被绕过。
- **DAT 模型使用 ncnn-vulkan 未验证**: 可能不兼容。

## 总体评估
**合规度: 60%**
存在禁令绕过和模型兼容性问题。
"""

AUDITS["photo_stage4_postprocess-audit.md"] = """# photo_stage4_postprocess.py 审查报告

## 文件信息
- **路径**: photo_restoration_pipeline/stages/stage4_postprocess.py
- **行数**: 11500
- **用途**: 照片后处理（选择性局部处理、颗粒合成）

## 关键发现
- **16bit 下采样到 8bit**: 同 stage1。
- **_extract_flif_chrominance 为 placeholder**: 未实现真正的 FLIF 提取。

## 总体评估
**合规度: 65%**
核心后处理逻辑实现，但存在精度问题和未实现的 placeholder。
"""

AUDITS["photo_stage5_ai_face-audit.md"] = """# photo_stage5_ai_face.py 审查报告

## 文件信息
- **路径**: photo_restoration_pipeline/stages/stage5_ai_face.py
- **行数**: 12936
- **用途**: 照片人脸检测与修复

## 关键发现
- **run_codeformer fallback 仅复制输入**: CodeFormer 未实际执行。
- **人脸框未传递给修复引擎**: 修复函数可能重新检测。

## 总体评估
**合规度: 55%**
GFPGAN 路径基本可用，但 CodeFormer 路径未实现，人脸坐标传递缺失。
"""

AUDITS["photo_stage6_encode-audit.md"] = """# photo_stage6_encode.py 审查报告

## 文件信息
- **路径**: photo_restoration_pipeline/stages/stage6_encode.py
- **行数**: 6780
- **用途**: 照片 AVIF 最终编码

## 关键发现
- **中间文件缺失时静默回退到原始输入**: 可能输出未处理文件。

## 总体评估
**合规度: 70%**
AVIF 编码逻辑基本完整，但存在静默回退风险。
"""

# Write all audit files
output_dir = r"C:\code1\old-video-and-picture-fix"
for filename, content in AUDITS.items():
    filepath = os.path.join(output_dir, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"Created: {filename}")

print(f"\nTotal audit reports created: {len(AUDITS)}")
