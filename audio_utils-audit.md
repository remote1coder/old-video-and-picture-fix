# audio_utils.py 审查报告

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
