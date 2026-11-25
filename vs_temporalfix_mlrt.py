# vs-mlrt path (tensorrt, directml) of vs_temporalfix 2.4.1 by pifroggi https://github.com/pifroggi/vs_temporalfix (Apache-2.0)

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Union

import vapoursynth as vs
from vs_temporalfix_utils import basic_expr, gen_shifts, get_tiles, interpolate_onnx, split_gridsample, exclude_regions

core = vs.core

MODEL_FILES: Dict[int, str] = {
    0: "temporalfix_s0_v1_op18_fp16.onnx",
    1: "temporalfix_s1_v1.1_op18_fp16.onnx",
    2: "temporalfix_s2_v1_op18_fp16.onnx",
    3: "temporalfix_s3_v1_op18_fp16.onnx",
}


def _plugin_info(namespace: str) -> Mapping[str, Union[bytes, str]]:
    # Version() of vs-mlrt's trt or ort plugin
    try:
        return getattr(core, namespace).Version()
    except Exception as e:
        raise RuntimeError(f"vs_temporalfix_mlrt: vs-mlrt plugin '{namespace}' is not loaded.") from e


def _default_model_folder(namespace: str) -> str:
    # <folder of vstrt/vsort>/models/temporalfix
    plugin_dir = os.path.dirname(os.fsdecode(_plugin_info(namespace)["path"]))
    return os.path.join(plugin_dir, "models", "temporalfix")


def _trtexec(plugin_path: str, trt_version: List[int]) -> Path:
    # trtexec next to vstrt (vsmlrt-cuda) or a system trtexec of the same TensorRT version
    exe_name = "trtexec.exe" if os.name == "nt" else "trtexec"
    bundled = Path(plugin_path) / "vsmlrt-cuda" / exe_name
    if bundled.is_file() and os.access(str(bundled), os.X_OK):
        return bundled
    system = shutil.which("trtexec")
    if system is not None:
        help_output = subprocess.run([system, "--help"], check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="locale", errors="replace")
        found = re.search(r"\[TensorRT v(\d+)\]", f"{help_output.stdout}\n{help_output.stderr}")
        if found is not None:
            version = int(found.group(1))
            if [version // 10000, (version % 10000) // 100, version % 100] == trt_version:
                return Path(system)
    raise FileNotFoundError(f"vs_temporalfix_mlrt: No trtexec for TensorRT {'.'.join(map(str, trt_version))} found (looked for {bundled} and trtexec in PATH).")


def _build_engine(onnx_path: str, engine_path: str, engine_w: int, engine_h: int, gpu_id: int, trt_version: List[int], trtexec_path: Path) -> None:
    # one dimension slightly dynamic avoids large engines from saved constants
    opt_shapes = f"input:1x21x{engine_h}x{engine_w}"
    min_shapes = f"input:1x21x{engine_h}x{engine_w - 8}"
    io_formats = "fp16:chw" if trt_version[0] < 11 else "chw"
    cmd = [
        str(trtexec_path),
        *(["--stronglyTyped"] if trt_version[0] < 11 else []),
        "--skipInference",
        "--memPoolSize=workspace:4096",
        "--avgTiming=8",
        "--builderOptimizationLevel=3",
        f"--inputIOFormats={io_formats}",
        f"--outputIOFormats={io_formats}",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes={min_shapes}",
        f"--optShapes={opt_shapes}",
        f"--maxShapes={opt_shapes}",
        f"--device={gpu_id}",
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="locale", errors="replace")
    except subprocess.CalledProcessError as e:
        msg = f"vs_temporalfix_mlrt: trtexec failed while building the TensorRT engine.\n  Command: {' '.join(cmd)}\n  Return code: {e.returncode}\n"
        if e.stdout:
            msg += f"\n=== trtexec stdout ===\n{e.stdout}"
        if e.stderr:
            msg += f"\n=== trtexec stderr ===\n{e.stderr}"
        raise RuntimeError(msg) from e


def _get_engine(model_dir: str, engine_dir: str, strength: float, engine_w: int, engine_h: int, gpu_id: int, force_rebuild: bool = False) -> str:
    info = _plugin_info("trt")
    strength_lower = math.floor(strength)
    strength_upper = math.ceil(strength)
    name_strength = str(strength).rstrip("0").rstrip(".")

    if strength_lower == strength_upper:
        model_file = MODEL_FILES[strength_lower]
        model_name = os.path.splitext(model_file)[0].split("_op")[0]
        onnx_path: Union[str, List[str]] = os.path.join(model_dir, model_file)
    else:
        name_lower = os.path.splitext(MODEL_FILES[strength_lower])[0].split("temporalfix_")[1].split("_op")[0]
        name_upper = os.path.splitext(MODEL_FILES[strength_upper])[0].split("temporalfix_")[1].split("_op")[0]
        model_name = f"temporalfix_s{name_strength}_[{name_lower}+{name_upper}]"
        onnx_path = [os.path.join(model_dir, MODEL_FILES[strength_lower]), os.path.join(model_dir, MODEL_FILES[strength_upper])]

    os.makedirs(engine_dir, exist_ok=True)
    engine_path = os.path.join(engine_dir, f"{model_name}_h{engine_h}_w{engine_w}_fp16_gpu{gpu_id}.engine")
    if not force_rebuild and os.path.isfile(engine_path) and os.path.getsize(engine_path) >= 512:
        return engine_path

    plugin_path = os.path.dirname(os.fsdecode(info["path"]))
    trt_version = int(os.fsdecode(info["tensorrt_version"]))
    trt_version_list = [trt_version // 10000, (trt_version % 10000) // 100, trt_version % 100]
    trtexec_path = _trtexec(plugin_path, trt_version_list)

    temp_dir = None
    try:
        if isinstance(onnx_path, list):
            temp_dir = tempfile.TemporaryDirectory(prefix=f"{model_name}_", dir=engine_dir)
            temp_path = os.path.join(temp_dir.name, f"{model_name}.onnx")
            interpolate_onnx(onnx_path_lower=onnx_path[0], onnx_path_upper=onnx_path[1], save_path=temp_path, weighting=strength - strength_lower)
            onnx_path = temp_path
        logging.warning("vs_temporalfix_mlrt: Building new TensorRT engine for strength=%s with width=%d and height=%d. This may take a few minutes.", name_strength, engine_w, engine_h)
        _build_engine(onnx_path, engine_path, engine_w, engine_h, gpu_id, trt_version_list, trtexec_path)
        logging.warning("vs_temporalfix_mlrt: Engine building complete.")
        return engine_path
    finally:
        if temp_dir is not None:
            temp_dir.cleanup()


def _tensorrt_inference(input_clips: List[vs.VideoNode], model_dir: str, engine_dir: str, strength: float, clip_w: int, clip_h: int, tiles: int, overlap: int, num_streams: int, gpu_id: int) -> vs.VideoNode:
    tile_w, tile_h, _, _ = get_tiles(clip_w=clip_w, clip_h=clip_h, tiles=tiles, overlap=overlap)
    engine_path = _get_engine(model_dir, engine_dir, strength, tile_w, tile_h, gpu_id)
    model_args = dict(engine_path=engine_path, num_streams=num_streams, device_id=gpu_id, **(dict(tilesize=(tile_w, tile_h), overlap=(overlap, overlap)) if tiles > 1 else {}))
    try:
        return core.trt.Model(input_clips, **model_args)
    except vs.Error as e:
        # engines of another TensorRT or driver version fail to deserialize: rebuild once
        if not any(k in str(e).lower() for k in ("serialize", "serialization", "deserialize", "deserialization")):
            raise
        logging.warning("vs_temporalfix_mlrt: Engine loading failed. This may be due to a TensorRT or driver update. Rebuilding...")
        model_args["engine_path"] = _get_engine(model_dir, engine_dir, strength, tile_w, tile_h, gpu_id, force_rebuild=True)
        return core.trt.Model(input_clips, **model_args)


def _directml_inference(input_clips: List[vs.VideoNode], model_dir: str, strength: float, clip_w: int, clip_h: int, tiles: int, overlap: int, num_streams: int, gpu_id: int) -> vs.VideoNode:
    import onnx

    # DirectML GridSample loses exact addressing once the flattened grid exceeds 2^24 (batch 6, 2 flow planes)
    tile_w, tile_h, _, _ = get_tiles(clip_w=clip_w, clip_h=clip_h, tiles=tiles, overlap=overlap)
    max_pixels = 16777216
    split = 6 * tile_h * tile_w * 2 > max_pixels
    if split and tile_h * tile_w * 2 > max_pixels:
        raise ValueError("vs_temporalfix_mlrt: The input dimensions are too large for the DirectML backend. Increase tiles or choose a different backend.")

    strength_lower = math.floor(strength)
    strength_upper = math.ceil(strength)
    if strength_lower == strength_upper:
        onnx_model = onnx.load(os.path.join(model_dir, MODEL_FILES[strength_lower]))
    else:
        onnx_model = interpolate_onnx(onnx_path_lower=os.path.join(model_dir, MODEL_FILES[strength_lower]),
                                      onnx_path_upper=os.path.join(model_dir, MODEL_FILES[strength_upper]), weighting=strength - strength_lower)
    if split:
        onnx_model = split_gridsample(onnx_model)

    out = core.ort.Model(input_clips, network_path=onnx_model.SerializeToString(), provider="DML", path_is_serialization=True, device_id=gpu_id, num_streams=num_streams,
                         **(dict(tilesize=(tile_w, tile_h), overlap=(overlap, overlap)) if tiles > 1 else {}))
    if num_streams > 1:
        # pre-initializing DirectML makes multi stream inference faster
        out.get_frame(0)
    return out


def model(clip: vs.VideoNode, strength: float = 2.0, exclude: Optional[str] = None, backend: str = "tensorrt", tiles: int = 1, num_streams: int = 1, gpu_id: int = 0,
          model_folder: Optional[str] = None, engine_folder: Optional[str] = None, tools: Optional[Mapping[str, str]] = None) -> vs.VideoNode:
    """TemporalFix model through vs-mlrt.

    clip: RGBH, even dimensions, at least 4 frames.
    strength: 0.0-3.0; fractional values interpolate the two neighbouring models.
    exclude: frame ranges to keep untouched, e.g. "[10 20] [600 900]".
    backend: 'tensorrt' (vstrt) or 'directml' (vsort, Windows only).
    tiles: 1, 2, 4, 6 or 8.
    model_folder: folder with the temporalfix ONNX files; default <vs-mlrt plugin folder>/models/temporalfix.
    engine_folder: TensorRT engine folder; default <model_folder>/engines.
    """
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("vs_temporalfix_mlrt: Clip must be a vapoursynth clip.")
    if clip.format.id == vs.PresetVideoFormat.NONE or clip.width == 0 or clip.height == 0:
        raise TypeError("vs_temporalfix_mlrt: Clip must have constant format and dimensions.")
    if vs.__version__.release_major >= 80 and clip.gpu_resident:
        raise ValueError("vs_temporalfix_mlrt: GPU based input clips are not supported. Download the input clip to CPU first.")
    if clip.width % 2 != 0 or clip.height % 2 != 0:
        raise ValueError("vs_temporalfix_mlrt: Clip dimensions must be even.")
    if clip.num_frames < 4:
        raise ValueError("vs_temporalfix_mlrt: Clip must be at least 4 frames long.")
    if clip.format.id != vs.RGBH:
        raise ValueError("vs_temporalfix_mlrt: Clip must be in RGBH format.")
    if strength < 0 or strength > 3:
        raise ValueError("vs_temporalfix_mlrt: Strength must be in the 0.0-3.0 range.")
    if num_streams < 1:
        raise ValueError("vs_temporalfix_mlrt: num_streams must be at least 1.")
    if not isinstance(gpu_id, int) or isinstance(gpu_id, bool) or gpu_id < 0:
        raise ValueError("vs_temporalfix_mlrt: gpu_id must be a non-negative integer.")
    if not isinstance(backend, str):
        raise TypeError("vs_temporalfix_mlrt: Backend must be a string.")
    backend = backend.lower()
    if backend in ("trt", "tensorrt"):
        namespace = "trt"
    elif backend in ("dml", "directml"):
        if sys.platform != "win32":
            raise RuntimeError("vs_temporalfix_mlrt: The DirectML backend is only available on Windows.")
        namespace = "ort"
    else:
        raise ValueError("vs_temporalfix_mlrt: Backend must be 'tensorrt' or 'directml'.")

    orig_clip = clip
    strength = round(float(strength), 2)
    clip_w = clip.width
    clip_h = clip.height
    overlap = min(64, int(64 * ((clip_w * clip_h) / (1920 * 1080)) ** 0.5))  # smaller inputs get a smaller tile overlap
    model_dir = os.path.abspath(model_folder) if model_folder is not None else _default_model_folder(namespace)
    engine_dir = os.path.abspath(engine_folder) if engine_folder is not None else os.path.join(model_dir, "engines")
    if strength == 0:
        return clip

    clip = basic_expr(clip, expr=["x 0 max 1 min"], tools=tools)
    input_clips = gen_shifts(clip, radius=3)  # [-3 .. +3]
    if namespace == "trt":
        out = _tensorrt_inference(input_clips, model_dir, engine_dir, strength, clip_w, clip_h, tiles, overlap, num_streams, gpu_id)
    else:
        out = _directml_inference(input_clips, model_dir, strength, clip_w, clip_h, tiles, overlap, num_streams, gpu_id)

    out = core.std.CopyFrameProps(out, clip)  # props of frame n, not of the shifted -3 clip
    return exclude_regions(out, orig_clip, exclude=exclude)
