# video_stage1_preprocess.py 审查报告

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
