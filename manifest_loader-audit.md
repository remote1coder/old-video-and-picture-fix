# manifest_loader.py 审查报告

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
