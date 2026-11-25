# TensorRT path of vs_undistort 2.3.1 by pifroggi https://github.com/pifroggi/vs_undistort (GPL-2.0), model from TMT https://github.com/xg416/TMT

from __future__ import annotations

import logging
import math
import os
import re
import shutil
import subprocess
from pathlib import Path
from typing import List, Mapping, Optional, Tuple, Union

import vapoursynth as vs
from helpers import pick_tool

core = vs.core

MODEL_FILES = {
    "bilinear": "UNet3d_TMT_lin_op19_fp16.onnx",
    "bicubic": "UNet3d_TMT_cub_op19_fp16.onnx",
}


def _expr(clips: Union[vs.VideoNode, List[vs.VideoNode]], expr: List[str], tools: Optional[Mapping[str, str]] = None) -> vs.VideoNode:
    return getattr(core, pick_tool(tools, 'expr', ('akarin', 'std')) or 'std').Expr(clips, expr)


def _get_window(clip: vs.VideoNode, temp_window: int) -> List[vs.VideoNode]:
    # pad with black frames to a multiple of temp_window; clip i holds frames i, i+temp_window, ...
    pad = (-clip.num_frames) % temp_window
    if pad:
        clip = core.std.Splice([clip, core.std.BlankClip(clip=clip, length=pad)])
    return [core.std.SelectEvery(clip[i:], cycle=temp_window, offsets=[0]) for i in range(temp_window)]


def _trim_overlaps(clip: vs.VideoNode, full_length: int, temp_window: int, window_overlap: int) -> vs.VideoNode:
    if window_overlap == 0 or clip.num_frames <= temp_window:
        return clip[:full_length]
    remaining = core.std.SelectEvery(clip[temp_window:], cycle=temp_window, offsets=list(range(window_overlap, temp_window)), modify_duration=False)
    return core.std.Splice([clip[:temp_window], remaining])[:full_length]


def _insert_overlaps(clip: vs.VideoNode, temp_window: int, window_overlap: int) -> vs.VideoNode:
    stride = temp_window - window_overlap
    length = clip.num_frames
    if window_overlap == 0 or length <= temp_window:
        return clip

    window_count = 1 + (length - window_overlap - 1) // stride
    if stride == 1:  # SelectEvery needs cycle > 1
        phases = [clip[offset:] for offset in range(temp_window)]
    else:
        phases = []
        for offset in range(temp_window):
            phase = core.std.SelectEvery(clip, cycle=stride, offsets=offset % stride, modify_duration=False)
            skip = offset // stride
            phases.append(phase[skip:] if skip else phase)

    overlapped = core.std.Interleave(phases, modify_duration=False)
    last_start = (window_count - 1) * stride
    return overlapped[:(window_count - 1) * temp_window + min(temp_window, length - last_start)]


def _get_tiles(clip_w: int, clip_h: int, tiles: int, overlap: int = 0) -> Tuple[int, int]:
    # tile size (multiple of 16) of the most square layout
    if tiles not in (1, 2, 4, 6, 8, 12, 16, 24, 32):
        raise ValueError("vs_undistort_mlrt: Tiles must be 1, 2, 4, 6, 8, 12, 16, 24, or 32.")

    layouts = {
        1: [(1, 1)],
        2: [(2, 1), (1, 2)],
        4: [(4, 1), (2, 2), (1, 4)],
        6: [(6, 1), (3, 2), (2, 3), (1, 6)],
        8: [(8, 1), (4, 2), (2, 4), (1, 8)],
        12: [(12, 1), (6, 2), (4, 3), (3, 4), (2, 6), (1, 12)],
        16: [(16, 1), (8, 2), (4, 4), (2, 8), (1, 16)],
        24: [(24, 1), (12, 2), (8, 3), (6, 4), (4, 6), (3, 8), (2, 12), (1, 24)],
        32: [(32, 1), (16, 2), (8, 4), (4, 8), (2, 16), (1, 32)],
    }[tiles]

    def _tile_size(layout: Tuple[int, int]) -> Tuple[int, int]:
        cols, rows = layout
        tile_w = math.ceil(math.ceil((clip_w + 2 * overlap * (cols - 1)) / cols) / 16) * 16
        tile_h = math.ceil(math.ceil((clip_h + 2 * overlap * (rows - 1)) / rows) / 16) * 16
        return tile_w, tile_h

    def _layout_valid(layout: Tuple[int, int]) -> bool:
        # tiles need a positive non overlapped stride
        cols, rows = layout
        tile_w, tile_h = _tile_size(layout)
        return not (cols > 1 and tile_w <= 2 * overlap) and not (rows > 1 and tile_h <= 2 * overlap)

    def _score(layout: Tuple[int, int]) -> Tuple[float, int, int]:
        tile_w, tile_h = _tile_size(layout)
        cols, rows = layout
        return (abs(math.log(tile_w / tile_h)), rows if clip_w >= clip_h else cols, abs(cols - rows))

    valid_layouts = [layout for layout in layouts if _layout_valid(layout)]
    if not valid_layouts:
        raise ValueError("vs_undistort_mlrt: Clip dimensions are too small for current tile amount. Reduce tiles or overlap.")
    tile_w, tile_h = _tile_size(min(valid_layouts, key=_score))
    if tile_w > 2 * tile_h or tile_h > 2 * tile_w:
        raise ValueError("vs_undistort_mlrt: The current tile amount produces tiles that are too elongated. Try a different tile amount.")
    return tile_w, tile_h


def _plugin_info() -> Mapping[str, Union[bytes, str]]:
    try:
        return core.trt.Version()
    except Exception as e:
        raise RuntimeError("vs_undistort_mlrt: vs-mlrt plugin 'trt' is not loaded.") from e


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
    raise FileNotFoundError(f"vs_undistort_mlrt: No trtexec for TensorRT {'.'.join(map(str, trt_version))} found (looked for {bundled} and trtexec in PATH).")


def _build_engine(onnx_path: str, engine_path: str, temp_window: int, engine_w: int, engine_h: int, interpolation: str, trt_version: List[int], trtexec_path: Path) -> None:
    # compute capability >= 12 (RTX 50) with bicubic needs dynamic shapes and no workspace limit
    bic50 = interpolation == "bicubic" and int(core.trt.DeviceProperties(0)["major"]) >= 12
    opt_shapes = f"input:1x{temp_window * 3}x{engine_h}x{engine_w}"
    min_shapes = f"input:1x{temp_window * 3}x{engine_h}x{engine_w - 8 if trt_version >= [11, 2, 0] or bic50 else engine_w}"
    io_formats = "fp16:chw" if trt_version[0] < 11 else "chw"
    cmd = [
        str(trtexec_path),
        *(["--stronglyTyped"] if trt_version[0] < 11 else []),
        *(["--markDebug=grid_sampler,grid_sampler_1"] if interpolation == "bicubic" else []),
        *(["--memPoolSize=workspace:6144"] if not bic50 else []),
        "--skipInference",
        "--avgTiming=8",
        "--builderOptimizationLevel=3",
        f"--inputIOFormats={io_formats}",
        f"--outputIOFormats={io_formats}",
        f"--onnx={onnx_path}",
        f"--saveEngine={engine_path}",
        f"--minShapes={min_shapes}",
        f"--optShapes={opt_shapes}",
        f"--maxShapes={opt_shapes}",
    ]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, encoding="locale", errors="replace")
    except subprocess.CalledProcessError as e:
        msg = f"vs_undistort_mlrt: trtexec failed while building the TensorRT engine.\n  Command: {' '.join(cmd)}\n  Return code: {e.returncode}\n"
        if e.stdout:
            msg += f"\n=== trtexec stdout ===\n{e.stdout}"
        if e.stderr:
            msg += f"\n=== trtexec stderr ===\n{e.stderr}"
        raise RuntimeError(msg) from e


def _get_engine(onnx_path: str, engine_dir: str, temp_window: int, engine_w: int, engine_h: int, interpolation: str, force_rebuild: bool = False) -> str:
    os.makedirs(engine_dir, exist_ok=True)
    engine_path = os.path.join(engine_dir, f"undistort_{interpolation}_t{temp_window}_h{engine_h}_w{engine_w}_fp16.engine")
    info = _plugin_info()
    if not force_rebuild and os.path.isfile(engine_path) and os.path.getsize(engine_path) >= 512:
        return engine_path

    plugin_path = os.path.dirname(os.fsdecode(info["path"]))
    trt_version = int(os.fsdecode(info["tensorrt_version"]))
    trt_version_list = [trt_version // 10000, (trt_version % 10000) // 100, trt_version % 100]
    trtexec_path = _trtexec(plugin_path, trt_version_list)
    logging.warning("vs_undistort_mlrt: Building new TensorRT engine for interpolation='%s' with temp_window=%d, width=%d, and height=%d. This may take a few minutes.", interpolation, temp_window, engine_w, engine_h)
    _build_engine(onnx_path, engine_path, temp_window, engine_w, engine_h, interpolation, trt_version_list, trtexec_path)
    logging.warning("vs_undistort_mlrt: Engine building complete.")
    return engine_path


def _tensorrt_inference(input_clips: List[vs.VideoNode], onnx_path: str, engine_dir: str, temp_window: int, tile_w: int, tile_h: int, overlap: int, tiles: int,
                        interpolation: str, flex_out_prop: str) -> Mapping[str, Union[vs.VideoNode, int]]:
    engine_path = _get_engine(onnx_path, engine_dir, temp_window, tile_w, tile_h, interpolation)
    model_args = dict(engine_path=engine_path, num_streams=1, flexible_output_prop=flex_out_prop, **(dict(tilesize=(tile_w, tile_h), overlap=(overlap, overlap)) if tiles > 1 else {}))
    try:
        return core.trt.Model(input_clips, **model_args)
    except vs.Error as e:
        # engines of another TensorRT or driver version fail to deserialize: rebuild once
        if not any(k in str(e).lower() for k in ("serialize", "serialization", "deserialize", "deserialization")):
            raise
        logging.warning("vs_undistort_mlrt: Engine loading failed. This may be due to a TensorRT or driver update. Rebuilding...")
        model_args["engine_path"] = _get_engine(onnx_path, engine_dir, temp_window, tile_w, tile_h, interpolation, force_rebuild=True)
        return core.trt.Model(input_clips, **model_args)


def undistort(clip: vs.VideoNode, temp_window: int = 10, window_overlap: int = 0, interpolation: str = "bicubic", tiles: int = 1, overlap: int = 8,
              model_folder: Optional[str] = None, engine_folder: Optional[str] = None, tools: Optional[Mapping[str, str]] = None) -> vs.VideoNode:
    """Removes distortions or wobble (TMT model) through vs-mlrt's TensorRT plugin.

    clip: RGBH.
    temp_window: frames processed as one chunk (>= 2); temp_window * tile width * tile height must stay below 2^22.
    window_overlap: overlap between temporal windows, 0..temp_window-1.
    interpolation: 'bicubic' or 'bilinear' warp.
    tiles: 1, 2, 4, 6, 8, 12, 16, 24 or 32; tiles are at least 128x128.
    overlap: overlap between tiles, at most half the tile size.
    model_folder: folder with the undistort ONNX files; default <vs-mlrt plugin folder>/models/undistort.
    engine_folder: TensorRT engine folder; default <model_folder>/engines.
    """
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("vs_undistort_mlrt: Clip must be a vapoursynth clip.")
    if clip.format.id == vs.PresetVideoFormat.NONE or clip.width == 0 or clip.height == 0:
        raise TypeError("vs_undistort_mlrt: Clip must have constant format and dimensions.")
    if vs.__version__.release_major >= 80 and clip.gpu_resident:
        raise ValueError("vs_undistort_mlrt: GPU based input clips are not supported. Download the input clip to CPU first.")
    if clip.format.id != vs.RGBH:
        raise ValueError("vs_undistort_mlrt: Clip must be in RGBH format.")
    if not isinstance(temp_window, int) or isinstance(temp_window, bool) or temp_window < 2:
        raise ValueError("vs_undistort_mlrt: temp_window must be an integer of at least 2.")
    if not isinstance(window_overlap, int) or isinstance(window_overlap, bool) or window_overlap < 0 or window_overlap >= temp_window:
        raise ValueError("vs_undistort_mlrt: window_overlap must be an integer from 0 to temp_window - 1.")
    if not isinstance(tiles, int) or isinstance(tiles, bool):
        raise TypeError("vs_undistort_mlrt: Tiles must be an integer.")
    if not isinstance(overlap, int) or isinstance(overlap, bool) or overlap < 0:
        raise ValueError("vs_undistort_mlrt: Overlap must be a non-negative integer.")
    if not isinstance(interpolation, str) or interpolation.lower() not in MODEL_FILES:
        raise ValueError("vs_undistort_mlrt: Interpolation must be 'bilinear' or 'bicubic'.")
    interpolation = interpolation.lower()

    clip = _expr(clip, ["x 0 max 1 min"], tools=tools)
    tile_w, tile_h = _get_tiles(clip_w=clip.width, clip_h=clip.height, tiles=tiles, overlap=overlap)

    # pad when the tile is larger than the clip
    pad_w = max(0, tile_w - clip.width)
    pad_h = max(0, tile_h - clip.height)
    if pad_w or pad_h:
        clip = core.std.AddBorders(clip, right=pad_w, bottom=pad_h)
    if tile_w < 128 or tile_h < 128:
        raise ValueError("vs_undistort_mlrt: Tile size must be at least 128 x 128. Reduce tiles.")
    if tiles > 1 and overlap > min(tile_w, tile_h) // 2:
        raise ValueError("vs_undistort_mlrt: Overlap can not be larger than half of tile size. Reduce overlap.")

    # larger inputs make TensorRT pick an extremely slow engine
    max_pixels = 4194304
    cur_pixels = temp_window * tile_w * tile_h
    if cur_pixels >= max_pixels:
        raise ValueError(f"vs_undistort_mlrt: temp_window * tile width * tile height must be smaller than {max_pixels} (currently {cur_pixels}). Increase tiles or reduce temp_window.")

    model_dir = os.path.abspath(model_folder) if model_folder is not None else os.path.join(os.path.dirname(os.fsdecode(_plugin_info()["path"])), "models", "undistort")
    engine_dir = os.path.abspath(engine_folder) if engine_folder is not None else os.path.join(model_dir, "engines")
    flex_out_prop = "undistort_output"

    overlap_clip = _insert_overlaps(clip, temp_window, window_overlap)
    stacked = _tensorrt_inference(_get_window(overlap_clip, temp_window), os.path.join(model_dir, MODEL_FILES[interpolation]), engine_dir,
                                  temp_window, tile_w, tile_h, overlap, tiles, interpolation, flex_out_prop)

    # the model returns temp_window RGB frames as planes of one frame
    carrier = stacked["clip"]
    num_planes = stacked["num_planes"]
    planes = [carrier.std.PropToClip(prop=f"{flex_out_prop}{i}") for i in range(num_planes)]
    grouped = [core.std.ShufflePlanes(planes[i:i + 3], [0, 0, 0], vs.RGB) for i in range(0, num_planes, 3)]
    out = core.std.Interleave(grouped)
    if pad_w or pad_h:
        out = core.std.Crop(out, right=pad_w, bottom=pad_h)
    out = _trim_overlaps(out, clip.num_frames, temp_window, window_overlap)
    return core.std.CopyFrameProps(out, clip)
