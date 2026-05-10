# video_restoration_pipeline/main.py 审查报告

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
