# checkpoint.py 审查报告

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
