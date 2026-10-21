#!/usr/bin/env python3
"""
Photo Restoration Pipeline - Main Controller
Single-file sequential execution mode.

v7.3 §1.1-1.6 Compliance:
- No multi-file parallel execution
- Stages execute sequentially per file
- Dynamic timeout based on image dimensions

# FIX-P10-1: Stage order corrected per v7.32 §3.1:
#   stage1_preprocess → stage2_ai_pre → stage3_ai_upscale →
#   stage4_ai_face → stage5_postprocess → stage6_encode
# FIX-P10-3: Face restore (stage4) now runs BEFORE postprocess (stage5)
# FIX-P10-4: Postprocess (stage5) no longer bound to AI switch
# FIX-P10-5: WSL2 copyback logic added
# FIX-P10-6: --scan, --edit-manifest, --mode run modes added
# FIX-P10-7: Manifest filter logic fixed (no SKIP in non-resume)
# FIX-P10-8: Disk space check uses shutil.disk_usage for Windows compat
# FIX-P10-10: force_traditional_first gate bound to stage1
# FIX-P10-11: --keep-intermediates replaces --cleanup-intermediates
# FIX-P10-12: Preview mode supports --input-dir
# FIX-P10-13: --face-restore no longer allows empty string
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


SCRIPT_DIR = Path(__file__).parent.resolve()


def _load_config(config_path: str) -> Dict[str, Any]:
    try:
        import yaml
        with open(config_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Config load error: {e}", file=sys.stderr)
        sys.exit(81)


def _deep_merge(base: Dict[str, Any],
                override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge override into base."""
    result = dict(base)
    for k, v in override.items():
        if (k in result and isinstance(result[k], dict)
                and isinstance(v, dict)):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def merge_params(config: Dict[str, Any],
                 detected_class: str,
                 cli_overrides: Dict[str, Any],
                 logger=None) -> Dict[str, Any]:
    """8-layer parameter cascading per v7.32 §3.2 + §3.10.12.

    # FIX-R10-W8: Updated docstring to document all merge layers
    # (audio_override + recommended_params were previously undocumented).
    # ORIGINAL: docstring omitted audio_override and recommended_params layers.
    Priority (low -> high):
      global -> photo -> paths[type] -> audio_override[type]
      -> ai/grain_factory3/audio (flattened) -> recommended_params
      -> user_overrides -> CLI

    Note: ai config flattened with both ai_{k} and {k} keys.
    Records PARAM_OVERRIDE events per v7.32 §3.10.12.
    """
    # Layer 1: global
    params: Dict[str, Any] = {}
    params = _deep_merge(params, config.get("global", {}))

    # FIX-P10-1: Layer 1.5 photo-specific defaults BEFORE paths
    # Per v7.32 §3.2: paths[type] must override general photo defaults.
    # ORIGINAL: photo was after paths, causing photo defaults to override
    # type-specific paths settings (e.g. paths.digital_photo.scratch_remove=false
    # would be overridden by photo.scratch_remove=true)
    params = _deep_merge(params, config.get("photo", {}))

    # Layer 2: paths (type-dependent, overrides photo defaults)
    paths_cfg = config.get("paths", {})
    if isinstance(paths_cfg, dict):
        type_key = (detected_class.lower()
                    if detected_class else "digital_photo")
        # FIX-2026-10-22：`user_overrides.path` 应能切换分型参数组。
        #
        # 起因（真实素材实测）：1346 张老照片是**用数码相机翻拍打印照片**
        # （FUJIFILM FinePix F10，带完整曝光参数），分类器据
        # `_looks_like_camera_capture()` 判成 DIGITAL_PHOTO / 0.90 ——
        # 而内容其实是需要修复的老照片。于是丢掉了一整套修复参数：
        #   scan_print    : scratch_remove=true/20、deblock=25、deband=1.2、grain=true
        #   digital_photo : scratch_remove=false/0、deblock=15、deband=0.8、grain=false
        # 即划痕修复与颗粒合成全部关闭，修复效果大打折扣。
        #
        # 自动识别"翻拍打印件"属于 recapture detection（需半色调网点/
        # 抖动/表面高光等特征 + 训练分类器），本轮不做；这里先提供
        # **确定性的人工开关**：manifest 里写
        #   user_overrides: {path: scan_print}
        # 即可强制走老照片参数组。
        #
        # 与视频链 FIX-NEW-V-10 同族：那边也有"path 应能覆盖查表键"的修复。
        # 注意必须在这里**预读**——user_overrides 在 Layer 5 才应用，
        # 而 paths 选组发生在 Layer 2。
        _uo_pre = (cli_overrides.get("user_overrides")
                   if isinstance(cli_overrides, dict) else None)
        if isinstance(_uo_pre, dict):
            _path_override = _uo_pre.get("path")
            if isinstance(_path_override, str) and _path_override.strip():
                _new_key = _path_override.strip().lower()
                if _new_key in paths_cfg and _new_key != type_key:
                    if logger:
                        msg = (f"path override: {type_key} -> {_new_key} "
                               f"(source=user_overrides)")
                        if hasattr(logger, "param_override"):
                            logger.param_override(
                                "path", type_key, _new_key,
                                source="user_overrides")
                        else:
                            logger.info("PARAM_OVERRIDE", msg)
                    type_key = _new_key
        if type_key in paths_cfg:
            type_paths = paths_cfg[type_key]
            if isinstance(type_paths, dict):
                if logger:
                    for k, v in type_paths.items():
                        old_v = params.get(k)
                        if old_v is not None and old_v != v:
                            if hasattr(logger, 'param_override'):
                                logger.param_override(
                                    k, old_v, v, source="paths")
                            else:
                                logger.info("PARAM_OVERRIDE",
                                           f"Key '{k}': {old_v} -> {v} "
                                           f"(source=paths)")
                params = _deep_merge(params, type_paths)

    # FIX-P10-4: Layer 2.5 audio_override per v7.32 §3.2
    # Priority: paths.audio_override > global.audio
    audio_override_cfg = config.get("audio_override", {})
    if isinstance(audio_override_cfg, dict) and type_key in audio_override_cfg:
        ao = audio_override_cfg[type_key]
        if isinstance(ao, dict):
            if logger:
                for k, v in ao.items():
                    old_v = params.get(k)
                    if old_v is not None and old_v != v:
                        if hasattr(logger, 'param_override'):
                            logger.param_override(
                                k, old_v, v,
                                source="audio_override")
                        else:
                            logger.info("PARAM_OVERRIDE",
                                       f"Key '{k}': {old_v} -> {v} "
                                       f"(source=audio_override)")
            params = _deep_merge(params, ao)

    # Layer 3: ai / grain_factory3 / audio (Bug 8.2 fix)
    # Flatten nested ai params into params for stage compatibility
    ai_cfg = config.get("ai", {})
    if isinstance(ai_cfg, dict):
        # Rename ai.enabled -> ai_enabled for stage script compatibility
        ai_flat = dict(ai_cfg)
        if "enabled" in ai_flat:
            ai_flat["ai_enabled"] = ai_flat.pop("enabled")
        # FIX-P10-6: Log PARAM_OVERRIDE for ai layer per v7.32 §3.10.12
        if logger:
            for k, v in ai_flat.items():
                old_v = params.get(k)
                if old_v is not None and old_v != v:
                    if hasattr(logger, 'param_override'):
                        logger.param_override(
                            k, old_v, v, source="ai_config")
                    else:
                        logger.info("PARAM_OVERRIDE",
                                   f"Key '{k}': {old_v} -> {v} "
                                   f"(source=ai_config)")
        # FIX-P10-7: Per v7.32 §3.2, create BOTH ai_{k} and {k} keys
        ai_prefixed = {}
        for k, v in list(ai_flat.items()):
            if not k.startswith("ai_"):
                ai_prefixed[f"ai_{k}"] = v
        params = _deep_merge(params, ai_flat)
        # ai_ prefixed keys override unprefixed if both exist
        for k, v in ai_prefixed.items():
            params[k] = v
    grain_cfg = config.get("grain_factory3", {})
    if isinstance(grain_cfg, dict):
        # FIX-P10-6: Log PARAM_OVERRIDE for grain layer
        if logger:
            for k, v in grain_cfg.items():
                old_v = params.get(k)
                if old_v is not None and old_v != v:
                    if hasattr(logger, 'param_override'):
                        logger.param_override(
                            k, old_v, v, source="grain_config")
                    else:
                        logger.info("PARAM_OVERRIDE",
                                   f"Key '{k}': {old_v} -> {v} "
                                   f"(source=grain_config)")
        params = _deep_merge(params, grain_cfg)
    audio_cfg = config.get("audio", {})
    if isinstance(audio_cfg, dict):
        # Rename audio.enabled -> audio_enabled for consistency
        audio_flat = dict(audio_cfg)
        if "enabled" in audio_flat:
            audio_flat["audio_enabled"] = audio_flat.pop("enabled")
        # FIX-P10-6: Log PARAM_OVERRIDE for audio layer
        if logger:
            for k, v in audio_flat.items():
                old_v = params.get(k)
                if old_v is not None and old_v != v:
                    if hasattr(logger, 'param_override'):
                        logger.param_override(
                            k, old_v, v, source="audio_config")
                    else:
                        logger.info("PARAM_OVERRIDE",
                                   f"Key '{k}': {old_v} -> {v} "
                                   f"(source=audio_config)")
        params = _deep_merge(params, audio_flat)

    # Layer 4: recommended params from classifier
    if "recommended_params" in cli_overrides:
        rec_params = cli_overrides.pop("recommended_params")
        for k, v in rec_params.items():
            old_v = params.get(k)
            if old_v is not None and old_v != v:
                if logger and hasattr(logger, 'param_override'):
                    logger.param_override(
                        k, old_v, v, source="recommended_params")
                elif logger:
                    logger.info("PARAM_OVERRIDE",
                               f"Key '{k}': {old_v} -> {v} "
                               f"(source=recommended_params)")
            params[k] = v

    # Layer 5: user overrides from manifest
    if "user_overrides" in cli_overrides:
        user_ov = cli_overrides.pop("user_overrides")
        for k, v in user_ov.items():
            old_v = params.get(k)
            if old_v is not None and old_v != v:
                if logger and hasattr(logger, 'param_override'):
                    logger.param_override(
                        k, old_v, v, source="user_overrides")
                elif logger:
                    logger.info("PARAM_OVERRIDE",
                               f"Key '{k}': {old_v} -> {v} "
                               f"(source=user_overrides)")
            params[k] = v

    # Layer 6: CLI
    cli_flat = dict(cli_overrides)
    for k, v in cli_flat.items():
        old_v = params.get(k)
        if old_v is not None and old_v != v:
            if logger and hasattr(logger, 'param_override'):
                logger.param_override(k, old_v, v, source="cli")
            elif logger:
                logger.info("PARAM_OVERRIDE",
                           f"Key '{k}': {old_v} -> {v} (source=cli)")
    params = _deep_merge(params, cli_overrides)

    return params


def _get_image_dimensions(input_path: str) -> tuple:
    """Get image width and height using PIL or ffprobe."""
    try:
        from PIL import Image
        with Image.open(input_path) as img:
            return img.size
    except Exception:
        try:
            r = subprocess.run(
                ["ffprobe", "-v", "error",
                 "-show_entries", "stream=width,height",
                 "-of", "csv=p=0", input_path],
                capture_output=True, text=True, errors="replace", timeout=10, shell=False
            )
            if r.returncode == 0:
                parts = r.stdout.strip().split(",")
                return int(parts[0]), int(parts[1])
        except Exception:
            pass
    # FIX-P10-12: Return (0, 0) to indicate unknown dimensions
    # Callers should use conservative estimate when (0, 0) returned
    return 0, 0


def _calculate_timeout(params: Dict[str, Any],
                       input_path: str,
                       is_ai_stage: bool = False) -> int:
    """Calculate dynamic timeout based on image dimensions.

    Base timeout: 10 minutes
    Additional: proportional to image pixel count
    Max: 20 minutes per photo (v7.32 §3.10.18 single-photo limit)
    Also respects 30MB file size threshold.
    """
    w, h = _get_image_dimensions(input_path)
    # FIX-P10-12: Use conservative 1920x1080 when dimensions unknown
    if w == 0 or h == 0:
        w, h = 1920, 1080
    pixel_count = w * h

    # Base: 10 minutes
    base = 600

    # Additional time based on pixel count
    # 1080p (~2M pixels) = +5 minutes for non-AI, +20 minutes for AI
    # 4K (~8M pixels) = +20 minutes for non-AI, +80 minutes for AI
    pixel_multiplier = 2.0 if is_ai_stage else 0.5
    additional = (pixel_count / (1920 * 1080)) * 300 * pixel_multiplier

    # v7.32 §3.10.18: single photo limit 20min (1200s)
    max_timeout = 1200

    # FIX-P10-8: Removed dead 30MB threshold code (was: if > 30MB: pass)
    # v7.32 §3.10.18 single-photo limit is always 20min regardless of file size
    timeout = min(base + additional, max_timeout)
    return int(timeout)


def _run_stage(script_name: str,
               job_id: str,
               input_path: str,
               config_path: str,
               params: Dict[str, Any],
               stage_idx: int,
               is_retry: bool = False,
               is_ai_stage: bool = False,
               script_dir: Optional[str] = None) -> tuple:
    """Run a single stage with dynamic timeout.

    Returns:
        (returncode, stderr) tuple per v6.3 §3.7 retry logic.
    """
    if script_dir is None:
        script_dir = SCRIPT_DIR / "stages"
    script_path = script_dir / script_name
    if not script_path.exists():
        print(f"Stage script not found: {script_path}", file=sys.stderr)
        return 81, ""

    # BUG-057 fix: if params JSON exceeds ~30000 chars,
    # write to temp file
    # R4: 重试标记放进 params（而不是未定义的 `--is-retry` 开关）——
    # 见下方 `_is_retry` 的说明，必须在 params_json 生成**之前**设置。
    if is_retry:
        try:
            params = dict(params or {})
            params["_is_retry"] = True
        except Exception:
            pass
    params_json = json.dumps(params)
    params_file_path = None
    if len(params_json) > 30000:
        import tempfile
        # FIX-P10-13: Use configured temp_dir for temp params file
        temp_base = str(Path(params.get("temp_dir", "./temp")))
        Path(temp_base).mkdir(parents=True, exist_ok=True)
        pf = tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False, encoding="utf-8",
            dir=temp_base
        )
        pf.write(params_json)
        pf.close()
        params_file_path = pf.name
        cmd = [
            sys.executable, str(script_path),
            "--job-id", job_id,
            "--input", input_path,
            "--config", config_path,
            "--params-file", params_file_path,
        ]
    else:
        cmd = [
            sys.executable, str(script_path),
            "--job-id", job_id,
            "--input", input_path,
            "--config", config_path,
            "--params", params_json,
        ]
    # R4: 上面的 params 已带 `_is_retry`，这里不需要再追加命令行开关。
    if is_retry:
        pass

    # Dynamic timeout calculation
    timeout = _calculate_timeout(
        params, input_path, is_ai_stage=is_ai_stage)

    # Process isolation per v7.3 §4.2
    env = os.environ.copy()
    env["PYTHONPATH"] = str(SCRIPT_DIR)

    print(f"[Stage {stage_idx}] Running: {script_name} "
          f"(timeout={timeout}s)")

    try:
        r = subprocess.run(cmd, timeout=timeout, env=env, shell=False,
                          capture_output=True, text=True, errors="replace")
        if r.returncode != 0 and r.stderr:
            print(f"[Stage {stage_idx}] stderr: {r.stderr[:500]}",
                  file=sys.stderr)
        return r.returncode, r.stderr
    except subprocess.TimeoutExpired as e:
        print(f"[Stage {stage_idx}] Timeout after {timeout}s",
              file=sys.stderr)
        return 87, str(e)
    except Exception as e:
        print(f"[Stage {stage_idx}] Error: {e}", file=sys.stderr)
        return 84, str(e)
    finally:
        # Clean up temp params file
        if params_file_path:
            try:
                os.unlink(params_file_path)
            except Exception:
                pass


def _check_disk_space(input_path: str,
                      params: Dict[str, Any],
                      logger=None) -> bool:
    """Check sufficient disk space with 2x safety margin.

    Delegates to checkpoint.check_wsl_space() which handles WSL2 redirect
    + disk estimation + /mnt/ fallback chain per v7.32 §3.10.10.

    # FIX-P10-8: Uses merged params (not raw config) to correctly
    # access upscale_scale. Uses shutil.disk_usage for Windows compat.
    """
    try:
        # Ensure is_photo flag is set for photo pipeline
        params["is_photo"] = True
        params["input_path"] = input_path

        # Read actual image dimensions for accurate estimation
        img_w, img_h = 0, 0
        try:
            from PIL import Image
            with Image.open(input_path) as img:
                img_w, img_h = img.size
        except Exception:
            pass
        if img_w > 0:
            params["original_width"] = img_w
            params["original_height"] = img_h

        sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))
        from checkpoint import check_wsl_space
        return check_wsl_space(params, logger=logger)
    except Exception as e:
        # FIX-P10-5: Per v7.32 §3.10.10: log warning instead of silent pass
        if logger:
            logger.warning("DISK_CHECK_SKIP",
                          f"Disk space check failed: {e}, "
                          f"proceeding with caution")
        return True


def _current_run_param_hashes(produced_paths) -> set:
    """本轮运行**实际产出过**的 param_hash 集合 —— 孤儿回收的判据基准。

    与视频链 `main._current_run_param_hashes` 同义：照片链产物命名
    `{job}_{stem}_step{N}_{op}_p{hash}.{ext}` 里的 hash 也是各 stage 以
    **自己的输入**为身份基准算的，同一轮里不同 step 的 hash 天然不同。
    所以判定"是否本轮产物"必须用集合，拿单个值去比会把上一步的合法
    产物误判成孤儿。
    """
    hashes: set = set()
    for p in produced_paths or []:
        m = re.search(r"_p([0-9a-f]+)\.", os.path.basename(str(p)))
        if m:
            hashes.add(m.group(1))
    return hashes


def _snapshot_job_files(temp_dir: str, job_id: str) -> Dict[str, tuple]:
    """拍下 temp 目录里本 job 现有文件的 {文件名: (mtime_ns, size)} 指纹。

    用于"执行前/执行后"做差集，识别本轮**真正新增或被改写**的产物
    （视频链 R6-10 同族做法）。返回空 dict 表示目录不可读。
    """
    out: Dict[str, tuple] = {}
    try:
        t = Path(temp_dir)
        if not t.is_dir():
            return out
        for p in t.iterdir():
            if not (p.is_file() and p.name.startswith(f"{job_id}_")):
                continue
            try:
                st = p.stat()
                out[p.name] = (st.st_mtime_ns, st.st_size)
            except OSError:
                continue
    except OSError:
        return {}
    return out


def cleanup_old_intermediates(temp_dir: str, job_id: str,
                              current_step=None,
                              current_run_hashes: Optional[set] = None) -> None:
    """Clean old intermediate files.

    FIX-NEW-V-12 / R10（与视频链 R6-10 同族）：必须传 current_step。
    共享实现 `checkpoint.cleanup_old_intermediates` 在 current_step=None
    时走"按 step 号降序取前 keep_steps 个"的分支——photo 链 keep_steps
    默认 2，于是每个 stage 跑完就把它**刚产出**的中间产物删掉
    （实测 stage3 的 step3_upscale 被删，stage4 找不到输入、
    人脸检测必然零命中）。

    FIX-2026-10-22：`current_run_hashes` 必须传。与视频链 `7a157d6` /
    R6-10 同族的那个病——`checkpoint.cleanup_old_intermediates` 的
    `current_param_hash` 形参**没有任何生产调用点传入**，孤儿回收逻辑
    是死代码：上一轮遗留的中间产物永不清理，跨轮次累积。
    """
    try:
        # FIX-2026-10-20: checkpoint 在 **shared_modules** 下，不在
        # photo_restoration_pipeline/ 下。原来插的是 SCRIPT_DIR，只是恰好靠
        # classify.py:450 把 shared_modules 塞进 sys.path 才没崩 —— 属隐式耦合。
        sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))
        from checkpoint import cleanup_old_intermediates as coi
        coi(temp_dir, job_id, current_step=current_step,
           current_run_hashes=current_run_hashes)
    except Exception as e:
        # 不能静默：清理失败会让陈旧中间产物一直堆积而无任何提示。
        print(f"[WARN] cleanup_old_intermediates failed: {e}",
              file=sys.stderr)


def _is_retryable_error(returncode: int, stderr: str = "") -> bool:
    """Check if error is retryable (transient).

    FIX-CK-R2-X1: Per round_2 CK-R2-X1, photo main.py kept a local duplicate of
    is_retryable_error that drifted from the checkpoint.py canonical version
    (which has the full disk-full short-circuit, signal-code handling, and 84
    OOM logic per v7.32 §3.10.18 / §3.7). Delegate to checkpoint.is_retryable_error
    so both pipelines share one implementation. The local stub is kept only as a
    thin wrapper for backward-compat with the existing call site (rc, stderr).
    """
    try:
        from checkpoint import is_retryable_error
        return is_retryable_error(returncode, stderr)
    except Exception:
        # Fallback to a conservative local impl if checkpoint import fails
        # (should not happen in normal operation - checkpoint is a hard dep).
        retryable_codes = {82, 83, 87}
        if returncode in retryable_codes:
            return True
        if returncode == 84:
            stderr_lower = (stderr or "").lower()
            if any(k in stderr_lower for k in (
                    "no space", "errno 28", "enospc")):
                return False
            if any(k in stderr_lower for k in (
                    "out of memory", "oom", "cuda")):
                return True
        return False


def _verify_stage_output(job_id: str, input_path: str,
                         params: Dict[str, Any],
                         logger=None) -> bool:
    """Verify stage1 output per v6.32 §7.1.

    Checks: 1) output file exists, 2) size > 0,
    3) image is decodable (PIL for photos).
    """
    temp_dir = Path(params.get("temp_dir", "./temp"))
    job_id_safe = job_id.replace("/", "_").replace("\\", "_")

    # Find stage1 output files matching naming convention
    candidates = list(temp_dir.glob(
        f"{job_id_safe}_*_step1_*_p*.*"))
    if not candidates:
        # Fallback: check for any recent file in temp_dir
        candidates = [f for f in temp_dir.iterdir()
                     if f.is_file() and job_id_safe in f.name
                     and f.stat().st_size > 0]

    if not candidates:
        if logger:
            logger.error("OUTPUT_MISSING",
                        f"No stage1 output found for job {job_id}")
        return False

    latest = max(candidates, key=lambda p: p.stat().st_mtime)

    if latest.stat().st_size == 0:
        if logger:
            logger.error("OUTPUT_EMPTY",
                        f"stage1 output is 0 bytes: {latest}")
        return False

    # Photo verification: use PIL
    try:
        from PIL import Image
        with Image.open(str(latest)) as img:
            img.verify()  # Verify integrity without full decode
        # Re-open for dimensions check
        with Image.open(str(latest)) as img:
            w, h = img.size
            if w == 0 or h == 0:
                raise ValueError("Image has zero dimensions")
    except Exception as e:
        if logger:
            logger.error("OUTPUT_CORRUPT",
                        f"stage1 output not decodable: {e}")
        return False

    if logger:
        logger.info("OUTPUT_VERIFIED",
                   f"stage1 output verified: {latest.name} "
                   f"({latest.stat().st_size} bytes, "
                   f"{w}x{h}px)")
    return True


def _run_single_job(job: Dict[str, Any],
                    config_path: str,
                    config: Dict[str, Any],
                    cli_params: Dict[str, Any]) -> int:
    """Run a single job with retry per v7.3 §4.5."""
    job_id = job.get("job_id", "unknown")
    input_path = job.get("input_path", "")
    detected_class = job.get("detected_class", "")
    confidence = job.get("confidence", 0)
    ai_upscale_banned = job.get("ai_upscale_banned", False)

    # FIX-P10-11: Use pathlib.Path consistently per CLAUDE.md §4
    if not Path(input_path).exists():
        print(f"[{job_id}] Input not found: {input_path}",
              file=sys.stderr)
        return 82

    # Per-job log
    log_dir = cli_params.get("log_dir", "./logs")
    os.makedirs(log_dir, exist_ok=True)

    # Initialize logger for PARAM_OVERRIDE recording
    # FIX-P11-3: MOVED before confidence check — original placement caused
    # NameError when logger.warning() was called before logger was defined
    # FIX-2026-10-20: logger 在 **shared_modules** 下，不在
    # photo_restoration_pipeline/ 下。原来插 SCRIPT_DIR 靠 classify 的
    # sys.path 副作用兜底，任何绕过 classify 的入口都会在这里
    # ModuleNotFoundError（进程直接崩，退出码 1 落在 80-113 之外）。
    sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))
    from logger import get_logger
    logger = get_logger(log_dir=log_dir, job_id=job_id)

    # FIX-P11-3: v7.32 §3.9: Low confidence handling
    # FIX-P11-12: UNKNOWN class requires explicit handling per v7.32 §3.9
    needs_confirmation = job.get("needs_confirmation", False)
    if detected_class == "UNKNOWN":
        is_interactive = cli_params.get("_mode") == "interactive"
        if not is_interactive:
            logger.warning("UNKNOWN_CLASS_SKIP",
                          f"Cannot classify image (UNKNOWN), auto-skipping",
                          job_id=job_id)
            return 3
        else:
            print(f"\n{'='*50}")
            print(f"  UNKNOWN CLASSIFICATION")
            print(f"  File: {input_path}")
            print(f"  The image could not be classified.")
            print(f"  Please specify class manually.")
            print(f"{'='*50}")
            confirm = input("Proceed with default settings? [y/N] ")
            if confirm.lower() != 'y':
                logger.info("USER_SKIP", "User declined unknown-class file")
                return 3
    elif confidence < 0.80 or needs_confirmation:
        is_interactive = cli_params.get("_mode") == "interactive"
        if not is_interactive:
            # Batch mode: auto-SKIP low confidence jobs
            logger.warning("LOW_CONFIDENCE_SKIP",
                          f"Confidence {confidence:.2f} < 0.80, "
                          f"auto-skipping", job_id=job_id)
            return 3
        else:
            # Interactive mode: force manual confirmation
            print(f"\n{'='*50}")
            print(f"  LOW CONFIDENCE DETECTION")
            print(f"  File: {input_path}")
            print(f"  Class: {detected_class}")
            print(f"  Confidence: {confidence:.2f} (threshold: 0.80)")
            print(f"  Reason: {job.get('skip_reason', 'Unknown')}")
            print(f"{'='*50}")
            confirm = input("Proceed with restoration? [y/N] ")
            if confirm.lower() != 'y':
                logger.info("USER_SKIP", "User declined low-confidence file")
                return 3

    # FIX-P1-11: Inject manifest job's recommended_params and user_overrides
    # into the merge so the 6-layer priority chain (Layer 4/5) actually
    # takes effect. Previously these were never passed → user edits in
    # --edit-manifest were silently dropped.
    merge_input = dict(cli_params)
    if job.get("recommended_params"):
        merge_input["recommended_params"] = job["recommended_params"]
    if job.get("user_overrides"):
        merge_input["user_overrides"] = job["user_overrides"]

    # Parameter merge (with override logging)
    params = merge_params(config, detected_class, merge_input, logger=logger)
    params["detected_class"] = detected_class
    params["confidence"] = confidence
    params["ai_upscale_banned"] = ai_upscale_banned
    params["job_id"] = job_id
    params["input_path"] = input_path
    # Inject classification metadata for stage scripts (Bug 11.1 fix)
    params["original_width"] = job.get("original_width", 0)
    params["original_height"] = job.get("original_height", 0)
    params["native_width"] = job.get("native_width", 0)
    params["native_height"] = job.get("native_height", 0)
    params["native_dar"] = job.get("native_dar", "")
    params["dpi"] = job.get("dpi")
    params["file_size"] = job.get("file_size", 0)
    params["file_mtime"] = job.get("file_mtime", 0)
    params["file_hash_prefix"] = job.get("file_hash_prefix", "")

    # FIX-R5-08: manifest 的 per-job output_path 必须进 params。
    # 原来 params["output_path"] 只来自 cli_params（--output，单文件模式
    # 下是完整文件路径），manifest 里明明有 per-job 的 output_path 却
    # 从未被消费 → **批量模式下所有 job 写同一个路径，互相覆盖**。
    # 单文件模式构造的 job 没有 output_path 字段，自然回落到 CLI 值。
    if job.get("output_path"):
        params["output_path"] = job["output_path"]

    # Add log config to params for stages
    params["log_dir"] = log_dir

    # FIX-P2-6/§3.10.10: Combined WSL2 redirect + disk space pre-check
    # check_wsl_space() handles redirect + space estimation + /mnt/ fallback
    # chain internally per v7.32 §3.10.10. No separate WSL redirect block needed.
    temp_dir = params.get("temp_dir", "./temp")

    # Clean intermediates
    # FIX-NEW-V-12 / R10: 全新运行先清空本 job 的陈旧中间产物
    # （与视频链 main.py 的 STALE_INTERMEDIATES_PURGED 同一处理），
    # 否则陈旧的高 step 产物会霸占 keep_steps 保留位。
    try:
        if params.get("cleanup_intermediates", True) and \
                not job.get("resume_from_stage"):
            # FIX-2026-10-20: checkpoint 在 shared_modules 下（同 449/568 两处）。
            sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))
            from checkpoint import cleanup_all_intermediates as _cai
            _purged = _cai(temp_dir, job_id)
            logger.info("STALE_INTERMEDIATES_PURGED",
                        "Purged stale intermediates for fresh run",
                        temp_dir=temp_dir, purged=_purged)
        else:
            cleanup_old_intermediates(temp_dir, job_id)
    except Exception as _e:
        # 原来静默 pass：陈旧产物清不掉时毫无提示，而它们的后果是
        # 上游产物被错误复用（比"没清理"本身更难排查）。
        print(f"[WARN] stale-intermediate cleanup failed: {_e}",
              file=sys.stderr)

    # Disk space check (WSL redirect handled inside check_wsl_space)
    # FIX-P10-8: Use merged params (not raw config) for upscale_scale
    if not _check_disk_space(input_path, params, logger=logger):
        return 87

    # Stage execution
    max_retries = params.get(
        "retry_count", config.get("global", {}).get("retry_count", 2))

    # FIX-P10-1: Stage order per v7.32 §3.1
    # FIX-P10-3: Face restore (stage4) before postprocess (stage5)
    stages = [
        ("stage1_preprocess.py", False),
        ("stage2_ai_pre.py", False),       # FIX-P10-2: single-channel correction
        ("stage3_ai_upscale.py", True),     # AI stage
        ("stage4_ai_face.py", True),        # AI stage, after upscale
        ("stage5_postprocess.py", False),   # Post-processing (not AI-gated)
        ("stage6_encode.py", False),
    ]

    # FIX-P10-10: force_traditional_first gate bound to stage1
    traditional_stage_completed = False

    # FIX-2026-10-22：本轮运行**跨 stage 累积**的实际产出路径列表。
    # 孤儿回收的判据基准。必须累积而非每次替换——只拍"本 stage 前"会漏掉
    # 本轮更早 stage 的产物（它们在快照时已存在，不在差集里），其 hash
    # 不在合法集合中会被孤儿回收误删。2026-10-22 端到端实测踩到并修正。
    _run_produced_paths: List[str] = []

    for stage_idx, (stage_script, is_ai_stage) in enumerate(stages):
        stage_name = stage_script.replace(".py", "")
        retry_count = 0

        # FIX-R10-M1: --stage single-stage mode per CLAUDE.md §0
        # When set, run only the requested stage (1-6); skip all others.
        cli_stage = cli_params.get("_stage")
        if cli_stage is not None and (stage_idx + 1) != cli_stage:
            continue

        # Conditional stages per v7.32 §3.1
        # FIX-P10-4: stage5_postprocess NOT bound to AI switch
        if stage_name == "stage3_ai_upscale":
            if not params.get("ai_enabled", False) or ai_upscale_banned:
                continue
        elif stage_name == "stage4_ai_face":
            if (not params.get("ai_enabled", False)
                    or not params.get("face_restore")):
                continue
        elif stage_name == "stage2_ai_pre":
            # Only runs when AI enabled
            if not params.get("ai_enabled", False):
                continue
        # stage5_postprocess always runs (FIX-P10-4)

        # FIX-P10-10: force_traditional_first gate bound to stage1
        # Block AI stages if stage1 failed
        # FIX-R10-W7: Include stage5_postprocess in the gate per v7.32 §3.10.8
        # (spec literally lists stage3/4/5). Defensive assertion only: when
        # stage1 fails, _verify_stage_output already returns 86 above, so
        # this branch is an invariant guard, not the primary gate.
        # ORIGINAL: gate covered only stage2/3/4, omitted stage5.
        force_traditional_first = params.get(
            "force_traditional_first", True)
        # FIX-R10-M1: In --stage single-stage mode the gate is bypassed — the
        # user explicitly requested one stage, so the traditional-first
        # invariant (which assumes a full run) must not block it.
        if (cli_stage is None and force_traditional_first
                and stage_name in ("stage2_ai_pre", "stage3_ai_upscale",
                                   "stage4_ai_face", "stage5_postprocess")):
            if not traditional_stage_completed:
                logger.error("TRADITIONAL_GATE_BLOCK",
                            "force_traditional_first=true but "
                            "traditional stage not completed, "
                            "blocking AI/post stages")
                print(f"[{job_id}] Traditional restoration gate "
                      f"blocked {stage_name}", file=sys.stderr)
                return 86

        while True:
            is_retry = retry_count > 0
            # FIX-2026-10-22：执行前快照本 job 在 temp 里的文件指纹，
            # 用于事后识别"本轮实际产出"——孤儿回收的判据基准。
            # 不用"事后 glob 全部"，否则上一轮遗留的陈旧产物也会被算进来，
            # 判据失效（这正是 current_param_hash 死代码的病根）。
            _pre_files = _snapshot_job_files(temp_dir, job_id)
            rc, stderr = _run_stage(
                stage_script, job_id, input_path,
                config_path, params, stage_idx,
                is_retry=is_retry, is_ai_stage=is_ai_stage
            )

            if rc == 0 or rc == 3:
                # 身份记录（第七轮）：与视频链用**同一套** `stage_identity`
                # + `stage_scope` schema，便于跨链追溯"这张图是哪套参数
                # 产出的"。
                #
                # 照片链**不做** plan/裁剪：单张处理是秒级，最小重算省不了
                # 什么；视频是数万帧小时级，才值得那套复杂度。
                try:
                    sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))
                    from stage_deps import photo_stage_identity, \
                        photo_stage_scope_keys
                    _stage_no = stage_idx + 1
                    _ident_path = Path(
                        params.get("temp_dir", "./temp")) / \
                        f"{job_id}_stage_identities.json"
                    _ident = {}
                    if _ident_path.exists():
                        try:
                            import json as _json
                            _ident = _json.loads(
                                _ident_path.read_text(encoding="utf-8"))
                        except Exception:
                            _ident = {}
                    import json as _json2
                    _ident[str(_stage_no)] = {
                        "stage": _stage_no,
                        "stage_name": stage_name,
                        "stage_identity": photo_stage_identity(
                            params, _stage_no),
                        "stage_scope": {
                            k: params.get(k) for k in
                            photo_stage_scope_keys(params, _stage_no)},
                        "rc": rc,
                    }
                    _ident_path.parent.mkdir(parents=True, exist_ok=True)
                    _ident_path.write_text(
                        _json2.dumps(_ident, ensure_ascii=False, indent=2),
                        encoding="utf-8")
                    logger.info("PHOTO_STAGE_IDENTITY",
                                "Recorded stage-local parameter identity",
                                stage=_stage_no, stage_name=stage_name,
                                sidecar=str(_ident_path))
                except Exception as _ie:
                    logger.warning("PHOTO_STAGE_IDENTITY_FAIL", str(_ie))

                # FIX-P10-2: v6.32 §7.1 output verification for stage1
                if stage_name == "stage1_preprocess":
                    output_ok = _verify_stage_output(
                        job_id, input_path, params, logger)
                    if output_ok:
                        traditional_stage_completed = True
                        logger.info("TRADITIONAL_GATE_PASS",
                                   "Traditional restoration completed, "
                                   "proceeding to AI stages")
                    else:
                        logger.error("TRADITIONAL_GATE_FAIL",
                                    "stage1 output verification failed, "
                                    "blocking AI stages")
                        return 86  # Validation failure
                # FIX-P3-6: Per v7.32 §3.8: Auto-generate forced-upscale
                # comparison frames after stage3 AI upscale
                if (stage_name == "stage3_ai_upscale"
                        and params.get("force_allow_banned_upscale")):
                    try:
                        sys.path.insert(
                            0, str(SCRIPT_DIR.parent / "shared_modules"))
                        from preview import auto_generate_comparison_if_needed
                        _temp_dir_p = Path(params.get("temp_dir", "./temp"))
                        _job_id_safe = job_id.replace("/", "_").replace(
                            "\\", "_")
                        _step3_cands = list(_temp_dir_p.glob(
                            f"{_job_id_safe}_*_step3_*_p*.*"))
                        if not _step3_cands:
                            _step3_cands = [
                                f for f in _temp_dir_p.iterdir()
                                if f.is_file()
                                and _job_id_safe in f.name
                                and "step3" in f.name
                                and f.stat().st_size > 0]
                        if _step3_cands:
                            _step3_cands.sort(
                                key=lambda p: p.stat().st_mtime,
                                reverse=True)
                            _step3_output = str(_step3_cands[0])
                            _comp_frames = auto_generate_comparison_if_needed(
                                original_path=input_path,
                                processed_path=_step3_output,
                                output_dir=params.get("output_dir",
                                                      "./output"),
                                config=config,
                                num_pairs=3
                            )
                            if _comp_frames:
                                logger.info(
                                    "PREVIEW_FORCED_UPSCALE",
                                    f"Generated {len(_comp_frames)} "
                                    f"comparison frames",
                                    frames=_comp_frames)
                    except Exception as e:
                        logger.warning(
                            "PREVIEW_FORCED_UPSCALE_FAIL",
                            f"Comparison frame generation failed: {e}")
                # BUG-053 fix: per-stage intermediate cleanup
                # FIX-NEW-V-12 / R10: 传 current_step（= 1-based stage 号），
                # 否则共享实现走 current_step=None 分支会删掉本 stage
                # 刚产出的中间产物（实测 stage3 的 step3_upscale 被删，
                # stage4 找不到输入 → 人脸检测必然零命中）。
                # FIX-2026-10-22: 再传 current_run_hashes（本轮实际产出的
                # hash 集合），否则"跨轮次/跨参数"的孤儿产物永不回收。
                # 差集为空（如 rc=3 跳过、本 stage 无产物）时传空集合，
                # 共享实现会跳过孤儿判定、只走 keep_steps，行为与旧版一致。
                _post_files = _snapshot_job_files(temp_dir, job_id)
                _new_this_stage = [
                    str(Path(temp_dir) / _n)
                    for _n, _sig in _post_files.items()
                    if _pre_files.get(_n) != _sig
                ]
                # 跨 stage 累积：判据必须覆盖本轮**所有**已跑 stage 的产物
                _run_produced_paths.extend(_new_this_stage)
                cleanup_old_intermediates(
                    temp_dir, job_id, current_step=stage_idx + 1,
                    current_run_hashes=_current_run_param_hashes(
                        _run_produced_paths))
                break  # Success or skip
            elif _is_retryable_error(rc, stderr):
                retry_count += 1
                if retry_count > max_retries:
                    print(f"[{job_id}] {stage_name} max retries exceeded",
                          file=sys.stderr)
                    return rc
                # Fixed retry intervals: 5s first, 15s second
                sleep_seconds = 5 if retry_count == 1 else 15
                logger.warning("STAGE_RETRY",
                              f"Retrying {stage_name} in {sleep_seconds}s "
                              f"(attempt {retry_count}/{max_retries})")
                time.sleep(sleep_seconds)
            else:
                # Non-retryable error
                # FIX-P10-3: Map out-of-range exit codes to 86 per v7.32 §3.5
                if rc < 80 or rc > 113:
                    logger.warning("INVALID_EXIT_CODE",
                                  f"{stage_name} returned exit code {rc} "
                                  f"(outside 80-113 range), mapping to 86")
                    print(f"[{job_id}] {stage_name} returned invalid "
                          f"exit code {rc}, mapped to 86",
                          file=sys.stderr)
                    return 86
                print(f"[{job_id}] {stage_name} non-retryable error "
                      f"(rc={rc})", file=sys.stderr)
                return rc

    # FIX-P10-3/FIX-P10-5: WSL2 copyback logic per v6.32 §7.3
    # After all stages succeed, copy final output back to
    # user's original path if WSL redirect was active
    # Key name fixed: checkpoint.py uses "original_output_dir"
    # Argument type fixed: iterate files in output_dir, pass file paths as strings
    if params.get("original_output_dir"):
        sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))
        from checkpoint import wsl_copy_back
        output_dir = Path(params["output_dir"])
        orig_dir = Path(params["original_output_dir"])
        orig_dir.mkdir(parents=True, exist_ok=True)
        # Iterate files in output_dir, copy each back individually
        for f in output_dir.iterdir():
            if f.is_file():
                target = orig_dir / f.name
                try:
                    wsl_copy_back(str(f), str(target), logger=logger)
                except Exception as e:
                    logger.error("WSL_COPY_BACK_FAIL",
                                f"Failed to copy back {f.name}: {e}")
                    return 82  # IO error

    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", "-c", required=True)
    parser.add_argument("--input", "-i", default=None,
                       help="Single input photo file")
    parser.add_argument("--input-dir", default=None,
                       help="Directory of photos for batch processing")
    parser.add_argument("--job-id", default=None)
    # FIX-R10-M1: --stage single-stage mode per CLAUDE.md §0
    parser.add_argument("--stage", type=int, default=None,
                       help="Run only the specified stage (1-6); "
                            "skips other stages")
    parser.add_argument("--output", "-o", default=None)
    parser.add_argument("--log-dir", default="./logs")
    parser.add_argument("--temp-dir", default="./temp")
    parser.add_argument("--no-wsl-redirect", action="store_true")

    # FIX-P10-11: Replace --cleanup-intermediates and --no-cleanup
    # with --keep-intermediates per v7.32 §3.10.17
    parser.add_argument("--keep-intermediates", action="store_true",
                       help="Keep intermediate files (default: cleanup)")

    parser.add_argument("--preview", action="store_true")
    # FIX-R10-M3: --preview-only per v7.32 §3.3 mode C (preview, no execution)
    parser.add_argument("--preview-only", action="store_true",
                       help="Preview mode only (no execution); "
                            "per v7.32 §3.3 mode C")
    parser.add_argument("--force-allow-banned-upscale", action="store_true")
    # FIX-P10-1: Use BooleanOptionalAction per v7.32 §3.2
    # ORIGINAL: store_true + default=True = always True (cannot disable)
    # Now: --force-traditional-first enables, --no-force-traditional-first disables
    parser.add_argument("--force-traditional-first",
                       action=argparse.BooleanOptionalAction,
                       default=True,
                       help="Force traditional processing before AI "
                            "(default: True, use --no-force-traditional-first "
                            "to disable)")
    # FIX-P10-10: Add DiffBIR per v6.32 §4.2.5
    # DiffBIR: diffusion-based face restoration for severely degraded faces
    parser.add_argument("--face-restore",
                       choices=["GFPGAN", "CodeFormer", "DiffBIR", "none"],
                       default=None)
    # Quality overrides
    parser.add_argument("--crf", type=int, default=None)
    parser.add_argument("--preset", type=int, default=None)
    parser.add_argument("--aq-mode", type=int, default=None)
    # AI params
    parser.add_argument("--ai-enabled", action="store_true")
    # FIX-R10-W6: Constrain upscale factor to 2 or 4 per v6.32 §4.2.5
    # (integer scales only). ORIGINAL: no choices, allowed 3 etc.
    parser.add_argument("--ai-upscale-factor", type=int, default=None,
                       choices=[2, 4],
                       help="AI upscale factor (2 or 4 only)")
    parser.add_argument("--ai-model", default=None)
    parser.add_argument("--ai-denoise-strength", type=float, default=None)
    parser.add_argument("--ai-pre-upscale-lanczos", action="store_true")
    # FIX-R10-M2: --gpu backend override per CLAUDE.md §0
    parser.add_argument("--gpu", choices=["auto", "cuda", "cpu"],
                       default=None,
                       help="Force GPU backend (auto/cuda/cpu)")
    # BUG-054 fix: parallel batch processing
    parser.add_argument("--workers", type=int, default=1,
                       help="Number of parallel workers")
    # ISSUE-P-MAIN-001 fix: manifest workflow support
    parser.add_argument("--manifest", default=None,
                       help="Path to batch manifest YAML/JSONL")
    parser.add_argument("--resume", action="store_true",
                       help="Resume from manifest")

    # FIX-P10-6: Add missing run modes per v7.32 §3.3
    parser.add_argument("--scan", action="store_true",
                       help="Scan directories and generate manifest")
    parser.add_argument("--output-manifest",
                       default="batch_manifest.yaml",
                       help="Output path for generated manifest")
    parser.add_argument("--edit-manifest", action="store_true",
                       help="Interactively edit manifest")
    parser.add_argument("--mode",
                       choices=["batch", "quick", "interactive"],
                       default=None,
                       help="Execution mode")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: scan and execute directly")

    args = parser.parse_args()

    # Validate input arguments
    input_modes = [args.input, args.input_dir, args.manifest,
                   args.scan]
    if sum(bool(m) for m in input_modes) == 0:
        parser.error("Either --input, --input-dir, --manifest, "
                     "or --scan must be specified")
    if sum(bool(m) for m in [args.input, args.input_dir,
                             args.manifest]) > 1:
        parser.error("--input, --input-dir, and --manifest "
                     "are mutually exclusive")

    config = _load_config(args.config)

    # BUG-014 fix: default=True in argparse ensures correct default
    force_traditional = args.force_traditional_first

    # Build common CLI params
    cli_params = {}
    if args.output:
        _out = args.output
        # FIX-2026-10-22：单文件模式下 `--output` 若指向**已存在的目录**，
        # 必须拼上文件名，不能把目录本身当成输出路径。
        #
        # 实测（真实素材）：`--input 003.JPG --output E:\...\one`
        #   → params["output_path"] = "E:\...\one"
        #   → stage6 把 `output.job_test.partial` 重命名到该目录
        #   → `[WinError 5] Access is denied: '...partial' -> '...one'`
        #   → 留下 .partial + output.status.json = RECOVERY_REQUIRED，
        #     而**进程退出码仍是 0**，看起来"成功"。
        # 批量模式不受影响（manifest 里每个 job 都有自己的 output_path，
        # 后写入覆盖）。这里只在目标确实是目录（或以分隔符结尾）时补齐。
        if os.path.isdir(_out) or _out.endswith(("/", "\\")):
            _stem = Path(args.input).stem if args.input else "output"
            _out = str(Path(_out) / f"{_stem}_restored.avif")
        cli_params["output_path"] = _out
    if args.crf is not None:
        cli_params["crf"] = args.crf
    if args.preset is not None:
        cli_params["preset"] = args.preset
    if args.aq_mode is not None:
        cli_params["aq_mode"] = args.aq_mode
    # FIX-P10-13: Map "none" to empty string for face_restore
    if args.face_restore is not None:
        if args.face_restore == "none":
            cli_params["face_restore"] = ""
        else:
            cli_params["face_restore"] = args.face_restore
    if args.ai_enabled:
        cli_params["ai_enabled"] = True
    # Bug 11.2/11.3 fix: unify CLI key names with stage scripts
    if args.ai_upscale_factor is not None:
        cli_params["upscale_scale"] = args.ai_upscale_factor
    if args.ai_model is not None:
        cli_params["upscale_model"] = args.ai_model
    if args.ai_denoise_strength is not None:
        cli_params["ai_denoise_strength"] = args.ai_denoise_strength
    if args.ai_pre_upscale_lanczos:
        cli_params["pre_upscale_lanczos"] = True
    if args.force_allow_banned_upscale:
        cli_params["force_allow_banned_upscale"] = True

    # FIX-P10-11: --keep-intermediates replaces --no-cleanup
    cli_params["cleanup_intermediates"] = not args.keep_intermediates
    cli_params["temp_dir"] = args.temp_dir
    cli_params["log_dir"] = args.log_dir
    cli_params["force_traditional_first"] = force_traditional

    # FIX-R10-M1: Inject --stage into cli_params (consumed in _run_single_job
    # stage loop via cli_params["_stage"]). Only set when explicitly provided.
    if args.stage is not None:
        if not 1 <= args.stage <= 6:
            parser.error("--stage must be in 1..6")
        cli_params["_stage"] = args.stage
    # FIX-R10-M2: Inject --gpu backend override per CLAUDE.md §0
    if args.gpu is not None:
        cli_params["gpu"] = args.gpu
        if args.gpu == "cpu":
            cli_params["gpu_priority"] = ["cpu"]
        elif args.gpu == "cuda":
            cli_params["gpu_priority"] = ["cuda", "cpu"]

    # FIX-P10-2: Propagate --no-wsl-redirect to cli_params per v7.32 §3.10.10
    if args.no_wsl_redirect:
        cli_params["no_wsl_redirect"] = True

    # FIX-P10-6: --scan mode per v7.32 §3.3
    if args.scan:
        scan_dirs = []
        if args.input:
            scan_dirs.append(args.input)
        if args.input_dir:
            scan_dirs.append(args.input_dir)
        if not scan_dirs:
            parser.error("--scan requires --input or --input-dir")
        # FIX-P10-6: Import from shared_modules, not SCRIPT_DIR
        sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))
        try:
            from manifest_generate import generate_manifest
            # FIX-R5-08: 必须传 pipeline_type="photo"！
            # 原来漏传，generate_manifest 默认 "video" → 按视频扩展名
            # （.avi/.mkv/.mp4/...）扫描，照片一个都找不到；即便找到，
            # 输出名也会是 .mkv 而不是 .avif，AI 封禁规则也会走视频分支
            # （HDV_AVCHD/ANALOG_TAPE 而不是 SCAN_PRINT/DIGITAL_PHOTO）。
            # 也就是说照片链的 manifest 批处理**从未真正工作过**——
            # 历史 ai_photo* 验收全是单文件模式（--input <file>）绕开的。
            # cli_output_dir 与视频链一致（FIX-R5-07）：扫描尊重 --output，
            # manifest 记绝对路径。
            generate_manifest(scan_dirs, args.output_manifest, config,
                              "photo",
                              cli_output_dir=getattr(args, 'output', None))
        except ImportError as e:
            print(f"manifest_generate module not available: {e}",
                  file=sys.stderr)
            sys.exit(80)  # Dependency missing
        return 0

    # FIX-P10-6: --edit-manifest mode per v7.32 §3.3
    if args.edit_manifest:
        manifest_path = (args.manifest or args.output_manifest)
        if not manifest_path:
            parser.error("--edit-manifest requires --manifest "
                         "or --output-manifest")
        # FIX-P10-6: Import from shared_modules, not SCRIPT_DIR
        sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))
        try:
            from manifest_editor import edit_manifest
            edit_manifest(manifest_path)
        except ImportError as e:
            print(f"manifest_editor module not available: {e}",
                  file=sys.stderr)
            sys.exit(80)
        return 0

    # FIX-P10-7: Support both --quick flag and --mode quick per v7.32 §3.3
    if args.quick or args.mode == "quick":
        if not args.input_dir:
            parser.error("--quick requires --input-dir")
        input_dir = Path(args.input_dir)
        photo_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff",
                     ".bmp", ".webp", ".avif")
        input_files = sorted(
            f for f in input_dir.iterdir()
            if f.is_file() and f.suffix.lower() in photo_exts
        )
        if not input_files:
            print(f"No photo files found in {args.input_dir}",
                  file=sys.stderr)
            sys.exit(81)

        print(f"Quick mode: {len(input_files)} files")
        sys.path.insert(0, str(SCRIPT_DIR))
        from classify import classify_photo

        failed = 0
        for idx, input_path in enumerate(input_files):
            classification = classify_photo(
                str(input_path), config.get("photo", {}))
            job = {
                "job_id": f"photo_{int(time.time())}_{idx}",
                "input_path": str(input_path),
                "detected_class": classification.get(
                    "detected_class", ""),
                "confidence": classification.get("confidence", 0),
                "ai_upscale_banned": classification.get(
                    "ai_upscale_banned", False),
                "original_width": classification.get(
                    "original_width", 0),
                "original_height": classification.get(
                    "original_height", 0),
                "native_width": classification.get(
                    "native_width", 0),
                "native_height": classification.get(
                    "native_height", 0),
                "native_dar": classification.get("native_dar", ""),
                "dpi": classification.get("dpi"),
                "file_size": classification.get("file_size", 0),
                "file_mtime": classification.get("file_mtime", 0),
                "file_hash_prefix": classification.get(
                    "file_hash_prefix", ""),
                "user_overrides": {},
            }
            rc = _run_single_job(
                job, args.config, config, dict(cli_params))
            if rc != 0:
                failed += 1
                print(f"[{input_path.name}] FAILED (rc={rc})",
                      file=sys.stderr)

        print(f"\nQuick mode complete: "
              f"{len(input_files) - failed}/{len(input_files)} "
              f"succeeded")
        sys.exit(0 if not failed else 86)

    # FIX-P10-10: Interactive mode per v7.32 §3.3 Mode A
    if args.mode == "interactive" and args.input:
        cli_params["_mode"] = "interactive"  # FIX-P11-3: mark for _run_single_job
        sys.path.insert(0, str(SCRIPT_DIR))
        from classify import classify_photo
        classification = classify_photo(
            args.input, config.get("photo", {}))

        print(f"\n{'='*50}")
        print(f"  File: {args.input}")
        print(f"  Class: {classification.get('detected_class', '?')}")
        print(f"  Confidence: "
              f"{classification.get('confidence', 0):.1%}")
        print(f"  AI Upscale Banned: "
              f"{classification.get('ai_upscale_banned', False)}")
        print(f"  Dimensions: "
              f"{classification.get('original_width', '?')}x"
              f"{classification.get('original_height', '?')}")
        print(f"  DPI: {classification.get('dpi', 'unknown')}")
        print(f"{'='*50}")

        # v7.32 §3.10.2: AI ban interactive confirmation
        if (classification.get('ai_upscale_banned')
                and args.force_allow_banned_upscale):
            confirm = input(
                "\n⚠️  AI upscale BANNED for this image. "
                "Force allow? [y/N] ")
            if confirm.lower() != 'y':
                print("Aborted by user.")
                sys.exit(0)
            cli_params["force_allow_banned_upscale"] = True

        # v7.32 §3.3: Allow user to modify params
        proceed = input("\nProceed with restoration? [Y/n] ")
        if proceed.lower() == 'n':
            sys.exit(0)

        # Continue to single-file execution (fall through)

    # Preview mode
    # FIX-R10-M3: --preview-only per v7.32 §3.3 mode C (strictly no execution)
    if args.preview or args.preview_only:
        # FIX-P10-12: Support --input-dir in preview mode
        preview_input = args.input or args.input_dir
        print(f"Preview mode: showing job plan for {preview_input}")
        if args.input:
            sys.path.insert(0, str(SCRIPT_DIR))
            from classify import classify_photo
            classification = classify_photo(args.input, config.get("photo", {}))
            job = {
                "job_id": args.job_id or f"photo_{int(time.time())}",
                "input_path": args.input,
                "detected_class": classification.get(
                    "detected_class", ""),
                "confidence": classification.get("confidence", 0),
                "ai_upscale_banned": classification.get(
                    "ai_upscale_banned", False),
            }
            print(json.dumps(job, indent=2, ensure_ascii=False))

            # Generate actual preview frames
            try:
                sys.path.insert(0, str(
                    SCRIPT_DIR.parent / "shared_modules"))
                from preview import extract_preview_frames
                # FIX-P10-11: Use pathlib.Path per CLAUDE.md §4
                preview_dir = str(Path(args.temp_dir) / "previews")
                Path(preview_dir).mkdir(parents=True, exist_ok=True)
                preview_frames = extract_preview_frames(
                    args.input, preview_dir,
                    num_frames=4, force_compare_frames=False
                )
                if preview_frames:
                    print(f"Preview frames generated: {preview_frames}")
            except Exception as e:
                print(f"Preview generation skipped: {e}",
                      file=sys.stderr)

        # FIX-P10-12: Handle --input-dir for preview mode
        # FIX-P10-11: Use pathlib.Path per CLAUDE.md §4
        if args.input_dir:
            photo_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff",
                         ".bmp", ".webp", ".avif")
            for p in Path(args.input_dir).iterdir():
                if p.suffix.lower() in photo_exts:
                    try:
                        sys.path.insert(0, str(
                            SCRIPT_DIR.parent / "shared_modules"))
                        from preview import extract_preview_frames
                        preview_dir = str(
                            Path(args.temp_dir) / "previews")
                        Path(preview_dir).mkdir(
                            parents=True, exist_ok=True)
                        extract_preview_frames(
                            str(p), preview_dir, num_frames=4,
                            force_compare_frames=False)
                    except Exception as e:
                        print(f"Preview for {p} skipped: {e}",
                              file=sys.stderr)
        # FIX-R10-M3: Both --preview and --preview-only return here without
        # entering the execution branches below. --preview-only is strictly
        # no-execution per v7.32 §3.3 mode C.
        return 0

    # ISSUE-P-MAIN-001 fix: Manifest mode
    if args.manifest:
        sys.path.insert(0, str(SCRIPT_DIR.parent / "shared_modules"))
        from manifest_loader import (
            load_manifest, filter_jobs,
            update_job_status, save_manifest, save_manifest_job_result
        )
        try:
            manifest = load_manifest(args.manifest)
        except Exception as e:
            print(f"Failed to load manifest: {e}", file=sys.stderr)
            sys.exit(81)

        # FIX-P10-7: Only PENDING+FAILED in non-resume mode
        # FIX-P10-11: Include PROCESSING in resume mode per v7.32 §3.4.5
        # Do NOT include SKIP per v7.32 §3.4.5
        if args.resume:
            statuses = ["PENDING", "FAILED", "PROCESSING"]
        else:
            statuses = ["PENDING", "FAILED"]

        jobs = filter_jobs(manifest, statuses=statuses)
        # FIX-P10-5: Sort by priority descending per v7.32 §3.4.5
        jobs.sort(key=lambda j: j.get("priority", 0), reverse=True)
        if not jobs:
            print("No pending jobs in manifest")
            sys.exit(0)

        print(f"Manifest mode: {len(jobs)} jobs to process "
              f"(statuses={statuses})")
        # RV-07: 本 run 先领取归属（generation 单调递增），这样旧 run 的迟到
        # 结果不会覆盖本次结果，也不会出现"末尾整份旧快照回写"。
        from manifest_loader import acquire_run_ownership
        run_owner = f"photo-run-{os.getpid()}"
        _claim_ids = [j.get("job_id") for j in jobs]
        if not _claim_ids:
            # R3-05：本次没有任务 → 零领取，不动别人的 owner/generation
            print("No jobs to process.")
            return
        run_generation = acquire_run_ownership(
            args.manifest, _claim_ids, owner=run_owner)
        if run_generation is None:
            # R3-11：领取失败 = 结果无处可写。不能照常产出并报成功。
            print("ERROR: could not claim this run's jobs in the manifest; "
                  "not producing artifacts that cannot be recorded.",
                  file=sys.stderr)
            sys.exit(82)

        results = []
        failures: List[tuple] = []

        def _record(job: Dict[str, Any], rc: int) -> None:
            """RV-07: 收到一个结果就立刻落盘（锁内单任务更新）。"""
            status = "DONE" if rc == 0 else "FAILED"
            update_job_status(manifest, job["job_id"], status,
                              error_message="" if rc == 0 else f"Exit code {rc}")
            # R2-06：持久化的真实契约包含"抛异常"（磁盘拒绝写入、
            # Path.replace 失败等）。旧代码只判断返回值，异常会直接逃出
            # _record 并打断整个批次。这里收口并保留原始原因。
            try:
                saved = save_manifest_job_result(
                    args.manifest, job["job_id"], status,
                    error_message="" if rc == 0 else f"Exit code {rc}",
                    owner=run_owner, generation=run_generation)
                save_reason = ""
            except Exception as e:
                saved = False
                save_reason = f"{type(e).__name__}: {e}"
                print(f"[{job['job_id']}] result persistence raised "
                      f"{save_reason}", file=sys.stderr)
            results.append((job, rc))
            if rc != 0 or not saved:
                failures.append((job, rc if rc != 0 else 82))
                print(f"[{job['job_id']}] FAILED "
                      f"with exit code {rc}", file=sys.stderr)
            if not saved:
                # 磁盘不可写时**不谎称**已记录失败状态：只把它作为本次运行的
                # 失败原因报出去，诊断信息走 stderr/日志。
                print(f"[{job['job_id']}] result could NOT be persisted to "
                      f"the manifest", file=sys.stderr)

        if args.workers > 1:
            print(f"Parallel processing with {args.workers} workers "
                  f"(spawn context)")
            # FIX-P10-9: Use spawn context per v7.32 §3.10.18
            import multiprocessing as mp
            pool = mp.get_context('spawn').Pool(processes=args.workers)
            try:
                # R2-05：旧代码 `for future, job in futures.items(): future.get()`
                # 是**按提交顺序**阻塞等待 —— 前一个 job 慢，后面已经完成的
                # job 就进不了 _record；此时调度器退出会丢掉它们的状态。
                # 改成用 callback：结果一到就由 pool 的 result handler 线程
                # 调用 _record，天然是"完成顺序"。
                futures = []
                for job in jobs:
                    # 每个 job 各自绑定（不能把 job 捕获在外层定义里）
                    def _done(rc, _job=job):
                        _record(_job, rc)

                    def _failed(exc, _job=job):
                        print(f"[{_job['job_id']}] Worker crashed: {exc}",
                              file=sys.stderr)
                        _record(_job, 84)

                    futures.append(pool.apply_async(
                        _run_single_job,
                        (job, args.config, config, dict(cli_params)),
                        callback=_done, error_callback=_failed))
                # 只负责等待，不再按提交顺序收集结果
                for future in futures:
                    future.wait()
            finally:
                pool.close()
                pool.join()
        else:
            for job in jobs:
                rc = _run_single_job(
                    job, args.config, config, dict(cli_params))
                _record(job, rc)

        # RV-07: 不再执行 save_manifest(manifest, ...) —— 那是开头读入的旧快照，
        # 会把本批已经写好的 DONE 倒盖回 PENDING（复核已复现 B: DONE -> PENDING）。

        success_count = sum(1 for _, rc in results if rc == 0)
        print(f"\nManifest complete: "
              f"{success_count}/{len(jobs)} succeeded")
        if failures:
            print(f"Failed jobs ({len(failures)}):")
            for j, rc in failures:
                print(f"  {j['job_id']} ({j['input_path']}) "
                      f"-> exit {rc}")
        sys.exit(0 if not failures
                 else max(rc for _, rc in failures))

    # Batch mode via --input-dir (Bug 11.8 / P-1 fix)
    # BUG-054 fix: support --workers for parallel batch processing
    # FIX-P10-11: Use pathlib.Path per CLAUDE.md §4
    if args.input_dir:
        photo_exts = (".jpg", ".jpeg", ".png", ".tif", ".tiff",
                     ".bmp", ".webp", ".avif")
        input_dir_path = Path(args.input_dir)
        input_files = sorted(
            str(f) for f in input_dir_path.iterdir()
            if f.is_file() and f.suffix.lower() in photo_exts
        )

        if not input_files:
            print(f"No photo files found in {args.input_dir}",
                  file=sys.stderr)
            sys.exit(81)

        print(f"Batch mode: {len(input_files)} files found "
              f"in {args.input_dir}")
        sys.path.insert(0, str(SCRIPT_DIR))
        from classify import classify_photo

        # Pre-classify all files (lightweight, keep in main process)
        jobs = []
        for idx, input_path in enumerate(input_files):
            classification = classify_photo(input_path, config.get("photo", {}))
            detected_class = classification.get("detected_class", "")
            job_id = args.job_id or f"photo_{int(time.time())}_{idx}"
            job = {
                "job_id": job_id,
                "input_path": input_path,
                "detected_class": detected_class,
                "confidence": classification.get("confidence", 0),
                "ai_upscale_banned": classification.get(
                    "ai_upscale_banned", False),
                "original_width": classification.get(
                    "original_width", 0),
                "original_height": classification.get(
                    "original_height", 0),
                "native_width": classification.get(
                    "native_width", 0),
                "native_height": classification.get(
                    "native_height", 0),
                "native_dar": classification.get("native_dar", ""),
                "dpi": classification.get("dpi"),
                "file_size": classification.get("file_size", 0),
                "file_mtime": classification.get("file_mtime", 0),
                "file_hash_prefix": classification.get(
                    "file_hash_prefix", ""),
                "user_overrides": {},
            }
            jobs.append((input_path, job))

        # FIX-P10-5: Sort by priority (default 0) per v7.32 §3.4.5
        jobs.sort(key=lambda item: item[1].get("priority", 0),
                  reverse=True)

        results = []
        if args.workers > 1:
            print(f"Parallel processing with {args.workers} workers "
                  f"(spawn context)")
            # FIX-P10-9: Use spawn context per v7.32 §3.10.18
            import multiprocessing as mp
            pool = mp.get_context('spawn').Pool(processes=args.workers)
            try:
                futures = {}
                for input_path, job in jobs:
                    future = pool.apply_async(
                        _run_single_job,
                        (job, args.config, config, dict(cli_params)))
                    futures[future] = input_path
                for future, input_path in futures.items():
                    try:
                        rc = future.get()
                    except Exception as e:
                        print(f"[{input_path}] Worker crashed: {e}",
                              file=sys.stderr)
                        rc = 84
                    results.append((input_path, rc))
                    if rc != 0:
                        print(f"[{input_path}] FAILED "
                              f"with exit code {rc}", file=sys.stderr)
            finally:
                pool.close()
                pool.join()
        else:
            # Sequential fallback
            for input_path, job in jobs:
                rc = _run_single_job(
                    job, args.config, config, dict(cli_params))
                results.append((input_path, rc))
                if rc != 0:
                    print(f"[{job['job_id']}] FAILED "
                          f"with exit code {rc}", file=sys.stderr)

        # Summary
        success_count = sum(1 for _, rc in results if rc == 0)
        print(f"\nBatch complete: "
              f"{success_count}/{len(input_files)} succeeded")
        failed = [(p, rc) for p, rc in results if rc != 0]
        if failed:
            print(f"Failed files ({len(failed)}):")
            for p, rc in failed:
                print(f"  {p} -> exit {rc}")
        sys.exit(0 if not failed
                 else max(rc for _, rc in failed))

    # Single-file mode
    sys.path.insert(0, str(SCRIPT_DIR))
    from classify import classify_photo
    classification = classify_photo(args.input, config.get("photo", {}))
    detected_class = classification.get("detected_class", "")

    job = {
        "job_id": args.job_id or f"photo_{int(time.time())}",
        "input_path": args.input,
        "detected_class": detected_class,
        "confidence": classification.get("confidence", 0),
        "ai_upscale_banned": classification.get(
            "ai_upscale_banned", False),
        "original_width": classification.get("original_width", 0),
        "original_height": classification.get("original_height", 0),
        "native_width": classification.get("native_width", 0),
        "native_height": classification.get("native_height", 0),
        "native_dar": classification.get("native_dar", ""),
        "dpi": classification.get("dpi"),
        "file_size": classification.get("file_size", 0),
        "file_mtime": classification.get("file_mtime", 0),
        "file_hash_prefix": classification.get("file_hash_prefix", ""),
        "user_overrides": {},
    }

    rc = _run_single_job(job, args.config, config, cli_params)
    sys.exit(rc)


if __name__ == "__main__":
    main()
