# photo_restoration_pipeline/classify.py 审查报告

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
