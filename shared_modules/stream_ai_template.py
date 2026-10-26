# -*- coding: utf-8 -*-
"""零落盘流式超分的 AI 子进程脚本模板。

为什么单独放在这里
------------------
原先这段脚本是作为**三引号字符串**内联在 stage3_ai_upscale.py 里的。
但 DEP_CHECK 会静态扫描 `stages/*.py` 的原始文本找"顶层可执行代码"，
它分不清字符串内外的区别，于是把模板里的
    os.environ.setdefault(...)
    IN_W, IN_H, SCALE = 1280, 720, 2
当成真的模块级语句，判定"spawn 合规性违规"→ 整个跑批 exit 80。

把模板挪到 shared_modules 下（不属于 stages/），DEP_CHECK 不再扫它。

数据契约（三条都是实测/核查踩出来的，改动前先看这里）
----------------------------------------------------
1. **stdout 必须是纯二进制**。本地安装的 realesrgan 在 utils.py 里有
   `print('\\tInput is a 16-bit image')`（输入最大值 >256 时触发）与
   `print('\\tTile i/n')`，会直接写进 stdout，与帧数据混流。
   → 模板在**导入任何第三方库之前**先把 fd 1 复制出来当数据通道，
     再把文本层 stdout 指向 stderr（STR-01）。

2. **每帧输出必须恒为 uint16，且字节数严格等于 OUT_BYTES**。
   realesrgan 用 `np.max(img) > 256` 推断位深：暗帧（最大值 ≤256）会走
   8 位分支返回 uint8，那一帧只写约定字节数的一半，下游会把后续帧拼进
   同一帧，整个流从此处起错位。→ 写入前核对 dtype/shape/nbytes，
   不一致就大声失败（exit 84），而不是静默写半帧（STR-02）。

3. **管道是 RGB，模型接口是 BGR**。ffmpeg 用 rgb48le，而 realesrgan 的
   `enhance()` 内部做 `cv2.COLOR_BGR2RGB`，即接收 BGR、返回 BGR。
   PNG 路径用 `cv2.imread`（BGR 内存序）天然匹配；流式路径不转换就会
   把红蓝通道颠倒喂给模型。→ 进出各做一次 `[:, :, ::-1]`（STR-03）。
"""


def build_stream_ai_script(in_w, in_h, scale, model_path, model_name,
                           gpu_id=0):
    """生成 AI 子进程脚本：stdin 读 raw 帧 -> 超分 -> stdout 写 raw 帧。

    Returns:
        str: 可直接交给 python 执行的脚本文本。
    """
    NL = chr(10)
    in_w, in_h, scale = int(in_w), int(in_h), int(scale)
    in_bytes = in_w * in_h * 3 * 2
    out_bytes = (in_w * scale) * (in_h * scale) * 3 * 2

    lines = [
        # ---- 0) 先把真正的 stdout 存成数据通道，再把文本 stdout 引到 stderr ----
        # 必须早于任何第三方 import：realesrgan 的 print 会污染二进制流。
        "import os, sys",
        "_data = os.fdopen(os.dup(1), 'wb', buffering=0)",
        "os.dup2(2, 1)",
        "",
        "import gc",
        "import numpy as np",
        "import torch",
        "from basicsr.archs.rrdbnet_arch import RRDBNet",
        "from realesrgan import RealESRGANer",
        "",
        f"IN_W, IN_H, SCALE = {in_w}, {in_h}, {scale}",
        f"IN_BYTES, OUT_BYTES = {in_bytes}, {out_bytes}",
        f"MODEL_PATH = {model_path!r}",
        f"MODEL_NAME = {model_name!r}",
        f"GPU_ID = {int(gpu_id)}",
        "",
        "",
        "def _arch(name):",
        "    n = name.lower()",
        '    if "realesr-general" in n:',
        "        from basicsr.archs.srvgg_arch import SRVGGNetCompact",
        "        return SRVGGNetCompact(num_in_ch=3, num_out_ch=3, num_feat=64,",
        "                               num_conv=32, upscale=4, act_type='prelu')",
        '    if "x2plus" in n:',
        "        return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,",
        "                       num_block=23, num_grow_ch=32, scale=2)",
        "    return RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64,",
        "                   num_block=23, num_grow_ch=32, scale=4)",
        "",
        "",
        "def _as_uint16(out):",
        "    \"\"\"规整成 uint16。realesrgan 对暗帧会误判成 8bit 返回 uint8。\"\"\"",
        "    if out.dtype != np.uint16:",
        "        v = out.astype(np.float32)",
        "        if out.dtype == np.uint8:",
        "            v = v * 257.0",
        "        out = np.clip(v, 0, 65535).astype(np.uint16)",
        "        sys.stderr.write('WARN: model returned '"
        "+ str(out.dtype) + '; rescaled to uint16' + chr(10))",
        "        sys.stderr.flush()",
        "    return np.ascontiguousarray(out)",
        "",
        "",
        # ---- 1) 加载模型 ----
        # RealESRGANer 的 scale 必须是模型原生倍数（x4v3=4、x2plus=2），
        # 传目标倍数会让 enhance 输出原生尺寸（面积 4 倍），帧数/时长全错。
        "_NATIVE = 2 if 'x2plus' in MODEL_NAME.lower() else 4",
        "upsampler = RealESRGANer(scale=_NATIVE, model_path=MODEL_PATH,",
        "                         model=_arch(MODEL_NAME), half=True,",
        "                         gpu_id=GPU_ID)",
        "_in = sys.stdin.buffer",
        "_out = _data",
        "_n = 0",
        "while True:",
        "    buf = _in.read(IN_BYTES)",
        "    if not buf or len(buf) < IN_BYTES:",
        "        break",
        # 管道 rgb48le -> 模型要 BGR，进模型前翻一次
        "    arr = np.frombuffer(buf, dtype=np.uint16).reshape(IN_H, IN_W, 3)",
        "    arr_bgr = np.ascontiguousarray(arr[:, :, ::-1])",
        "    try:",
        "        out_bgr, _ = upsampler.enhance(arr_bgr, outscale=SCALE)",
        "    except RuntimeError as e:",
        "        if 'out of memory' in str(e).lower():",
        "            sys.stderr.write('OOM' + chr(10))",
        "            sys.stderr.flush()",
        "            sys.exit(83)",
        "        raise",
        # 出模型后翻回 RGB，再强制 uint16 并核对字节数
        "    out = np.ascontiguousarray(out_bgr[:, :, ::-1])",
        "    out = _as_uint16(out)",
        "    if out.shape != (IN_H * SCALE, IN_W * SCALE, 3):",
        "        sys.stderr.write('BADFRAME shape=' + repr(out.shape) + chr(10))",
        "        sys.stderr.flush()",
        "        sys.exit(84)",
        "    if out.nbytes != OUT_BYTES:",
        "        sys.stderr.write('BADFRAME nbytes=' + str(out.nbytes)",
        " + ' want=' + str(OUT_BYTES) + chr(10))",
        "        sys.stderr.flush()",
        "        sys.exit(84)",
        "    _out.write(out.tobytes())",
        "    _out.flush()",
        "    _n += 1",
        "try:",
        "    del upsampler",
        "except NameError:",
        "    pass",
        "gc.collect()",
        "if torch.cuda.is_available():",
        "    torch.cuda.empty_cache()",
        "sys.stderr.write(('FRAMES=%d' % _n) + chr(10))",
        "sys.stderr.flush()",
    ]
    return NL.join(lines) + NL
