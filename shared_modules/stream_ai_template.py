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


def build_fixed_frame_script(frame_dir, in_w, in_h, out_w, out_h):
    """生成"**冻结帧源**"子进程脚本：忽略模型，按顺序吐出预置帧。

    R22 §5.1 的①②两组实验需要"同一份固定帧源送进两条后半链"。
    把它做成一个与 AI 脚本**接口完全一致**的替身：

    - 从 stdin 读 `in_w*in_h*3*2` 字节/帧（消费掉，维持背压）；
    - 从 stdout 写 `out_w*out_h*3*2` 字节/帧（内容来自 `frame_dir/*.raw`）。

    这样 stage3 的调用方**不需要知道**帧是 AI 产的还是预置的，
    经典物化与融合串流吃的是同一串字节。

    数据契约（与 AI 模板一致的三条里最重要的一条）：
    **stdout 必须是纯二进制** —— 先把 fd 1 复制成数据通道，
    再把文本 stdout 指向 stderr。这个替身脚本也会被 import 到
    realesrgan 吗？不会，但保持同一形状能让"换回真 AI"零改动。

    Args:
        frame_dir: 预置帧目录；按文件名排序取 `*.raw`。
        in_w / in_h: 输入帧尺寸（用于算每帧输入字节数）。
        out_w / out_h: 输出帧尺寸（必须与预置帧的实际字节数一致）。

    Returns:
        str: 可直接交给 python 执行的脚本文本。

    Raises:
        ValueError: 尺寸非法。
    """
    NL = chr(10)
    in_bytes = int(in_w) * int(in_h) * 3 * 2
    out_bytes = int(out_w) * int(out_h) * 3 * 2
    if in_bytes <= 0 or out_bytes <= 0:
        raise ValueError("fixed-frame script needs positive dimensions")

    lines = [
        "import os, sys",
        # 与 AI 模板同一条铁律：先抢下真正的 stdout，再把文本层指到 stderr
        "_data = os.fdopen(os.dup(1), 'wb', buffering=0)",
        "os.dup2(2, 1)",
        "",
        "import pathlib",
        f"FRAME_DIR = pathlib.Path({frame_dir!r})",
        f"IN_BYTES = {in_bytes}",
        f"OUT_BYTES = {out_bytes}",
        "",
        "frames = sorted(FRAME_DIR.glob('*.raw'))",
        "if not frames:",
        "    sys.stderr.write('FIXED_FRAME: no *.raw in %s' % FRAME_DIR)",
        "    sys.stderr.flush()",
        "    raise SystemExit(84)",
        "",
        "stdin = sys.stdin.buffer",
        "i = 0",
        "while True:",
        "    chunk = stdin.read(IN_BYTES)",
        "    if not chunk:",
        "        break",
        "    if len(chunk) != IN_BYTES:",
        "        sys.stderr.write(",
        "            'FIXED_FRAME: short read %d != %d' % (len(chunk), IN_BYTES))",
        "        sys.stderr.flush()",
        "        raise SystemExit(84)",
        "    if i >= len(frames):",
        "        # 输入还有帧、预置帧却用完了 —— 帧数不匹配，必须大声失败",
        "        sys.stderr.write(",
        "            'FIXED_FRAME: input has more frames than the frozen set '",
        "            '(%d); refusing to truncate' % len(frames))",
        "        sys.stderr.flush()",
        "        raise SystemExit(84)",
        "    payload = frames[i].read_bytes()",
        "    if len(payload) != OUT_BYTES:",
        "        sys.stderr.write(",
        "            'FIXED_FRAME: %s is %d bytes, expected %d'",
        "            % (frames[i].name, len(payload), OUT_BYTES))",
        "        sys.stderr.flush()",
        "        raise SystemExit(84)",
        "    _data.write(payload)",
        "    i += 1",
        "",
        "if i != len(frames):",
        "    sys.stderr.write(",
        "        'FIXED_FRAME: used %d of %d frozen frames' % (i, len(frames)))",
        "    sys.stderr.flush()",
        "    raise SystemExit(84)",
        "sys.stderr.write('FIXED_FRAME: wrote %d frames' % i)",
        "sys.stderr.flush()",
        "",
    ]
    return NL.join(lines)


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
        "def _to_numpy(t):",
        "    \"\"\"把 post_process() 的结果转成 [0,1] 的 HWC numpy。",
        "",
        "    真实 RealESRGANer 返回 torch 张量（.data.squeeze().float()",
        "    .cpu().clamp_().numpy()）；测试替身可能直接返回 ndarray。",
        "    两种都支持，避免模板把测试绑死在 torch 上。",
        "    \"\"\"",
        "    if hasattr(t, 'data') and hasattr(getattr(t, 'data'), 'cpu'):",
        "        a = t.data.squeeze().float().cpu().clamp_(0, 1).numpy()",
        "        if a.ndim != 3:",
        "            sys.stderr.write('BADTENSOR shape=' + repr(a.shape) + chr(10))",
        "            sys.stderr.flush()",
        "            sys.exit(84)",
        # 真实 RealESRGANer.post_process() 返回 **NCHW**；squeeze 掉 batch
        # 后是 **CHW**。必须转成 HWC，否则后面的 `[:, :, ::-1]` 翻的是宽度轴，
        # 而且 cv2.resize 会因维度对不上抛 cv2.error（真机实测把流式打挂）。
        "        return np.ascontiguousarray(np.transpose(a, (1, 2, 0)))",
        "    a = np.asarray(t, dtype=np.float32).squeeze()",
        "    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[2] not in (1, 3):",
        "        a = np.transpose(a, (1, 2, 0))      # 兜底：CHW -> HWC",
        "    return np.clip(a, 0.0, 1.0)",
        "",
        "",
        "def _enhance_u16(upsampler, arr_bgr, outscale):",
        "    \"\"\"固定的 uint16 推理入口：/65535 归一、x65535 量化。",
        "",
        "    R12-09：RealESRGAN 的 enhance() 用 `np.max(img) > 256` 猜位深 ——",
        "    暗的 uint16 帧（峰值 <=256）会被当成 8bit 除以 255，模型看到的",
        "    输入放大 257 倍。事后把 uint8 输出乘回 257 救不了**已经发生**",
        "    的推理。这里绕过那段判断：先固定按 65535 归一化，再直接调用",
        "    pre_process / tile_process / post_process 三段公开方法。",
        "    \"\"\"",
        "    import cv2, torch",
        "    _h_in, _w_in = arr_bgr.shape[0:2]",
        "    img = arr_bgr.astype(np.float32) / 65535.0   # 固定分母，不看最大值",
        "    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)",
        # 必须关掉 autograd：原 enhance() 带 @torch.no_grad()，这里若不加，
        # PyTorch 会为整张推理图保留中间激活 —— 实测直接把 CUDA 显存打爆
        # （AI 进程 exit 83 OOM，流式路径退回 PNG）。
        "    with torch.no_grad():",
        "        upsampler.pre_process(img)",
        "        if upsampler.tile_size > 0:",
        "            upsampler.tile_process()",
        "        else:",
        "            upsampler.process()",
        "        out = _to_numpy(upsampler.post_process())",
        # 注意：这里的 out 是 **HWC**（enhance() 里那行 `[[2,1,0], :, :]`
        # 是 CHW 写法）。在 HWC 上用行下标会把**行**翻过来，不是通道。
        "    out = out[:, :, ::-1]   # RGB -> BGR（HWC，按通道翻）",
        "    out = (out * 65535.0).round().astype(np.uint16)",
        # R12-09 修正：`[:, :, ::-1]` 是**负步长视图**，cv2.resize 只接受
        # 连续数组，否则抛 cv2.error（真机实测直接把流式路径打挂、退回 PNG）。
        "    out = np.ascontiguousarray(out)",
        "    if outscale is not None and outscale != float(upsampler.scale):",
        "        out = cv2.resize(out, (int(_w_in * outscale),",
        "                               int(_h_in * outscale)),",
        "                         interpolation=cv2.INTER_LANCZOS4)",
        "    return out, 'RGB'",
        "",
        "",
        "def _as_uint16(out):",
        "    \"\"\"严格校验输出契约；**不再静默转换**。",
        "",
        "    R12-09：`_enhance_u16` 已保证 uint16，这里只做核对。",
        "    若 dtype/shape 不符，说明适配层失效 —— 必须报错，",
        "    不能把'任意不符的输出'悄悄转成'合法帧'。",
        "    \"\"\"",
        "    if out.dtype != np.uint16:",
        "        sys.stderr.write('BADFRAME dtype=' + str(out.dtype) + chr(10))",
        "        sys.stderr.flush()",
        "        sys.exit(84)",
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
        # R12 边界：空读 = 上游正常结束；**非空但不足一帧 = 协议错误**。
        # 原实现把两者都当 EOF，残尾会被静默丢弃（帧数变少却 rc=0）。
        "    if not buf:",
        "        break",
        "    if len(buf) < IN_BYTES:",
        "        sys.stderr.write('SHORTFRAME got=' + str(len(buf))",
        "                         + ' expect=' + str(IN_BYTES) + chr(10))",
        "        sys.stderr.flush()",
        "        sys.exit(85)",
        # 管道 rgb48le -> 模型要 BGR，进模型前翻一次
        "    arr = np.frombuffer(buf, dtype=np.uint16).reshape(IN_H, IN_W, 3)",
        "    arr_bgr = np.ascontiguousarray(arr[:, :, ::-1])",
        "    try:",
        "        out_bgr, _ = _enhance_u16(upsampler, arr_bgr, SCALE)",
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
