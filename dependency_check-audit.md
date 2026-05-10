# dependency_check.py 审查报告

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
