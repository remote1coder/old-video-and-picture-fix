# 探针脚本（证据归档）

这些脚本是**一次性验证工具**，用于定位具体缺陷。归档在这里是为了让
"结论可复跑、可核对"，而不是只留一句注释。

原位置 `tools_r4/` 被 `.gitignore` 忽略（该目录放的是 VS 便携版等大文件），
所以把探针移到这里入库。运行时请把工作目录指向项目根。

| 脚本 | 用途 | 关键结论 |
|---|---|---|
| `g34_pixfmt_probe.py` | 3 帧短实验，定位 G34 融合的色度差异根因 | packed `rgb48le` 直接进链 → Y 43.19 / **U 12.97 / V 12.37**；补 `format=gbrp16le` → **全部 inf（完全相同）** |
| `g34_equiv.py` | 经典 vs 融合的整链路对照（默认多线程） | 差异落在 stage1 本底噪声内，**不能据此判等价** |
| `g34_equiv_det.py` | 同上，但开 `vs.deterministic: true` | **IDENTICAL**（60 帧，PSNR 全 inf，文件大小相同） |
| `vs_determinism_probe.py` | 逐段定位 VapourSynth 渲染的非确定源 | `source+resize` 确定 → `TemporalDegrain` 出现 0.005 LSB 差异 → `haf.Stab` 放大 400 倍；只有 `num_threads=1` 完全确定（4.78× 慢） |

## 复跑注意

- 都要用 `tools_r4/vs_portable/r72/python.exe`（自带 VapourSynth 生态），
  **不要**用 `.venv_r4`。
- 输出目录在 `E:/codex/_media_r17_20261031T0004/` 下，与仓库分离。
- `g34_equiv_det.py` 是单线程模式，60 帧 1080p→4K 约 18 分钟；
  想快就把素材帧数改小（`-frames:v 20`），但注意 `in/g34.mp4` 已存在时
  脚本不会重新生成素材。
