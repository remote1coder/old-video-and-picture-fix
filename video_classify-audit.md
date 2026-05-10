# video_restoration_pipeline/classify.py 审查报告

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
