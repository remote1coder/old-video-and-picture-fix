# manifest_generate.py 审查报告

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
