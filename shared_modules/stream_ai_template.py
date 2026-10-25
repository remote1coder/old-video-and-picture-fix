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
"""


def build_stream_ai_script(in_w, in_h, scale, model_path, model_name,
                           gpu_id=0):
    """生成 AI 子进程脚本：stdin 读 raw 帧 → 超分 → stdout 写 raw 帧。

    约束（都是实测踩过的）：
    - stdout 必须是**纯二进制**，任何 print 都会污染视频流；
      脚本内所有提示信息一律走 stderr。
    - RealESRGANer 的 scale 必须是**模型原生倍数**（x4v3=4、x2plus=2），
      传目标倍数会让 enhance 输出原生尺寸（面积 4 倍），帧数/时长全错。
    - 换行一律用 chr(10) 拼接，避免任何反斜杠转义在多层拼装中被吃掉。
    """
    NL = chr(10)

    lines = [
        "import sys, gc",
        "import numpy as np",
        "import torch",
        "from basicsr.archs.rrdbnet_arch import RRDBNet",
        "from realesrgan import RealESRGANer",
        "",
        f"IN_W, IN_H, SCALE = {int(in_w)}, {int(in_h)}, {int(scale)}",
        "IN_BYTES = IN_W * IN_H * 3 * 2",
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
        "_NATIVE = 2 if 'x2plus' in MODEL_NAME.lower() else 4",
        "upsampler = RealESRGANer(scale=_NATIVE, model_path=MODEL_PATH,",
        "                         model=_arch(MODEL_NAME), half=True,",
        "                         gpu_id=GPU_ID)",
        "_in = sys.stdin.buffer",
        "_out = sys.stdout.buffer",
        "_n = 0",
        "while True:",
        "    buf = _in.read(IN_BYTES)",
        "    if not buf or len(buf) < IN_BYTES:",
        "        break",
        "    arr = np.frombuffer(buf, dtype=np.uint16).reshape(IN_H, IN_W, 3)",
        "    try:",
        "        out, _ = upsampler.enhance(arr, outscale=SCALE)",
        "    except RuntimeError as e:",
        "        if 'out of memory' in str(e).lower():",
        "            sys.stderr.write('OOM' + chr(10))",
        "            sys.stderr.flush()",
        "            sys.exit(83)",
        "        raise",
        "    _out.write(np.ascontiguousarray(out).tobytes())",
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
