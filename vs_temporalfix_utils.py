# Script by pifroggi https://github.com/pifroggi/vs_temporalfix
# or tepete and pifroggi on Discord

from __future__ import annotations

import re
import math
from typing import List, Mapping, Optional, Sequence, Tuple, Union

import vapoursynth as vs
from helpers import pick_tool, pick_zsmooth, tool_function
from misc import mvu_namespace, to_cpu

core = vs.core


def temporal_median(clip: vs.VideoNode, radius: int = 1, planes: Optional[Union[int, Sequence[int]]] = None, tools: Optional[Mapping[str, str]] = None) -> vs.VideoNode:
    # fallback plugin because zsmooth does not support non AVX2 CPUs
    return tool_function(tools, 'tmedian', 'TemporalMedian')(clip, radius=radius, planes=planes)


def repair(clip: vs.VideoNode, repairclip: vs.VideoNode, mode: Sequence[int] = (1,), tools: Optional[Mapping[str, str]] = None) -> vs.VideoNode:
    # fallback plugin because zsmooth does not support non AVX2 CPUs
    return tool_function(tools, 'rg', 'Repair')(clip, repairclip, mode=list(mode))


def median(clip: vs.VideoNode, radius: int = 1, planes: Optional[Union[int, Sequence[int]]] = None, tools: Optional[Mapping[str, str]] = None) -> vs.VideoNode:
    # fallback plugin because zsmooth does not support non AVX2 CPUs
    usable = {'zsmooth': hasattr(core, 'zsmooth'), 'std': radius == 1, 'ctmf': True}
    median = pick_zsmooth(tools, 'median', ('zsmooth', 'std', 'ctmf'), lambda name: usable.get(name, False))
    if median in ('zsmooth', 'neo_smo'):
        return getattr(core, median).Median(clip, radius=radius, planes=planes)
    elif median == 'std':
        return core.std.Median(clip, planes=planes)
    else:
        return core.ctmf.CTMF(clip, radius=radius, planes=planes)


def basic_expr(clips: Union[vs.VideoNode, Sequence[vs.VideoNode]], expr: Union[str, Sequence[str]], format: Optional[int] = None, tools: Optional[Mapping[str, str]] = None) -> vs.VideoNode:
    # backend for basic exprs supported by std.Expr
    return getattr(core, pick_tool(tools, 'expr', ('akarin', 'std')) or 'std').Expr(clips, expr, format=format)


def advanced_expr(clips: Union[vs.VideoNode, Sequence[vs.VideoNode]], expr: Union[str, Sequence[str]], format: Optional[int] = None, tools: Optional[Mapping[str, str]] = None) -> vs.VideoNode:
    # backend for advanced exprs not possible with std.Expr; akarin is the only one, other tools['expr'] values warn
    pick_tool(tools, 'expr', ('akarin',), lambda name: True)
    return core.akarin.Expr(clips, expr, format=format)


def box_blur(clip: vs.VideoNode, planes: Optional[Union[int, Sequence[int]]] = None, hradius: int = 1, hpasses: int = 1, vradius: int = 1, vpasses: int = 1, tools: Optional[Mapping[str, str]] = None) -> vs.VideoNode:
    # optional plugin for slight speed boost
    return tool_function(tools, 'boxblur', 'BoxBlur')(clip, planes=planes, hradius=hradius, hpasses=hpasses, vradius=vradius, vpasses=vpasses)


def min_blur(clip: vs.VideoNode, planes: Union[int, Sequence[int]] = (0, 1, 2), tools: Optional[Mapping[str, str]] = None) -> vs.VideoNode:
    # simplified function from G41Fun https://github.com/Vapoursynth-Plugins-Gitify/G41Fun
    # original avisynth function by Didée https://avisynth.nl/index.php/MinBlur

    if clip.format.num_planes == 1:
        planes = [0]
    planes = [planes] if isinstance(planes, int) else list(planes)
    
    mat1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    mat2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    RG11 = core.std.Convolution(clip, matrix=mat1, planes=planes).std.Convolution(matrix=mat2, planes=planes)
    RG4  = median(clip, radius=2, planes=planes, tools=tools)
    expr = "x y - x z - * 0 < x dup y - abs x z - abs < y z ? ?"
    return basic_expr([clip, RG11, RG4], [expr if i in planes else "" for i in range(clip.format.num_planes)], tools=tools)


def average_color_fix(clip: vs.VideoNode, ref: vs.VideoNode, radius: int = 4, passes: int = 4, tools: Optional[Mapping[str, str]] = None) -> vs.VideoNode:
    # simplified from https://github.com/pifroggi/vs_colorfix
    blurred_reference = box_blur(ref, hradius=radius, hpasses=passes, vradius=radius, vpasses=passes, tools=tools)
    blurred_clip = box_blur(clip, hradius=radius, hpasses=passes, vradius=radius, vpasses=passes, tools=tools)
    diff_clip = core.std.MakeDiff(blurred_reference, blurred_clip)
    return core.std.MergeDiff(clip, diff_clip)


def average_color_fix_fast(clip: vs.VideoNode, ref: vs.VideoNode, downscale_factor: int = 8, tools: Optional[Mapping[str, str]] = None) -> vs.VideoNode:
    # faster but faint blocky artifacts
    width  = int(clip.width  / downscale_factor) >> clip.format.subsampling_w << clip.format.subsampling_w
    height = int(clip.height / downscale_factor) >> clip.format.subsampling_h << clip.format.subsampling_h
    downscaled_reference = core.resize.Bilinear(ref,  width=width, height=height)
    downscaled_clip = core.resize.Bilinear(clip, width=width, height=height)
    diff_clip = core.std.MakeDiff(downscaled_reference, downscaled_clip)
    diff_clip = core.resize.Bilinear(diff_clip, width=clip.width, height=clip.height)
    return core.std.MergeDiff(clip, diff_clip)


def frequency_merge(low: vs.VideoNode, high: vs.VideoNode, radius: int = 40, passes: int = 3, tools: Optional[Mapping[str, str]] = None) -> vs.VideoNode:
    # merges low freqs of one clip with high freqs of another clip
    low_remaining  = box_blur(low,  hradius=radius, hpasses=passes, vradius=radius, vpasses=passes, tools=tools)
    high_removed   = box_blur(high, hradius=radius, hpasses=passes, vradius=radius, vpasses=passes, tools=tools)
    high_remaining = core.std.MakeDiff(high, high_removed)
    return core.std.MergeDiff(low_remaining, high_remaining)


def tweak_darks(src: vs.VideoNode, strength: float = 2.5, amp: float = 0.2, tools: Optional[Mapping[str, str]] = None) -> vs.VideoNode:
    # simplified DitherLumaRebuild function that works on full range
    # DitherLumaRebuild function from G41Fun https://github.com/Vapoursynth-Plugins-Gitify/G41Fun
    # originally created by cretindesalpes https://forum.doom9.org/showthread.php?p=1548318
    bd = src.format.bits_per_sample
    scale = 1 << (bd - 8)

    x = "x" if bd == 8 else f"x {scale} /"
    t = f"{x} 255 / 0 max 1 min"
    k = (strength - 1) * amp
    e = f"{k} {1 + amp} {(1 + amp) * amp} {t} {amp} + / - * {t} {1 - k} * + {1 << bd} *"
    expr = [e] + [""] * (src.format.num_planes - 1)
    return basic_expr([src], expr, tools=tools)


def contrasharp(clip: vs.VideoNode, src: vs.VideoNode, rep: int = 24, planes: Union[int, Sequence[int]] = (0, 1, 2), tools: Optional[Mapping[str, str]] = None) -> vs.VideoNode:
    # simplified function from G41Fun https://github.com/Vapoursynth-Plugins-Gitify/G41Fun
    # original avisynth function by Didée at the VERY GRAINY thread https://forum.doom9.org/showthread.php?p=1076491

    if clip.format.num_planes == 1:
        planes = [0]
    planes = [planes] if isinstance(planes, int) else list(planes)

    mat1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    mat2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    bd   = clip.format.bits_per_sample
    mid  = 1 << (bd - 1)
    num  = clip.format.num_planes

    s    = min_blur(clip, planes, tools=tools)  # damp down remaining spots of the denoised clip
    RG11 = core.std.Convolution(s, matrix=mat1, planes=planes).std.Convolution(matrix=mat2, planes=planes)
    ssD  = core.std.MakeDiff(s, RG11, planes)  # the difference of a simple kernel blur
    allD = core.std.MakeDiff(src, clip, planes)  # the difference achieved by the denoising
    ssDD = repair(ssD, allD, [rep if i in planes else 0 for i in range(num)], tools=tools)  # limit the difference to the max of what the denoising removed locally
    expr = "x {} - abs y {} - abs < x y ?".format(mid, mid)  # abs(diff) after limiting may not be bigger than before
    ssDD = basic_expr([ssDD, ssD], [expr if i in planes else "" for i in range(num)], tools=tools)
    return core.std.MergeDiff(clip, ssDD, planes)  # apply the limited difference (sharpening is just inverse blurring)


def exclude_regions(clip: vs.VideoNode, replacement: vs.VideoNode, exclude: Optional[str] = None) -> vs.VideoNode:
    # simplified ReplaceFrames function from fvsfunc https://github.com/Irrational-Encoding-Wizardry/fvsfunc
    # which is a port of ReplaceFramesSimple by James D. Lin http://avisynth.nl/index.php/RemapFrames

    if exclude is None:
        return clip
    if not isinstance(exclude, str):
        raise TypeError('vs_temporalfix: Exclusions are set like this: exclude="[100 300] [600 900] [2000 2500]", where the first number in the brackets is the start frame and the second is the end frame (inclusive).')

    exclude = exclude.replace(",", " ").replace(":", " ")
    frames = re.findall(r"\d+(?!\d*\s*\d*\s*\d*\])", exclude)
    ranges = re.findall(r"\[\s*\d+\s+\d+\s*\]", exclude)
    maps: List[List[int]] = []
    for range_ in ranges:
        maps.append([int(x) for x in range_.strip("[ ]").split()])
    for frame in frames:
        maps.append([int(frame), int(frame)])

    replace_frames: List[int] = []
    for start, end in maps:
        if start > end:
            raise ValueError("vs_temporalfix: Exclusions start frame is bigger than end frame: [{} {}]".format(start, end))
        if start >= clip.num_frames:
            raise ValueError("vs_temporalfix: Exclusions start frame {} is outside the clip. The last valid frame is {}.".format(start, clip.num_frames - 1))
        if end >= clip.num_frames:
            raise ValueError("vs_temporalfix: Exclusions end frame {} is outside the clip. The last valid frame is {}.".format(end, clip.num_frames - 1))
        replace_frames.extend(range(start, end + 1))

    if not replace_frames:
        return clip
    return clip.vszip.RFS(replacement, frames=replace_frames)


def lowfreq_denoise(low: vs.VideoNode, high: vs.VideoNode, motionmask: vs.VideoNode, thsad: int = 200, tr: int = 6, tools: Optional[Mapping[str, str]] = None) -> vs.VideoNode:
    # temporally denoise low frequencies only

    bs  = 8
    pel = 1

    # downscale clips
    downscale_factor = 8
    width  = int(low.width  / downscale_factor) >> low.format.subsampling_w << low.format.subsampling_w
    height = int(low.height / downscale_factor) >> low.format.subsampling_h << low.format.subsampling_h
    low_down   = core.resize.Bicubic(low,      width=width, height=height)
    motionmask = core.resize.Point(motionmask, width=width, height=height)
    motionmask = core.std.Maximum(motionmask)                  # expand mask
    prefilter  = tweak_darks(low_down, strength=2.5, amp=0.2, tools=tools)  # brighten darks

    mvu = mvu_namespace(tools)  # mvutensils or mvgpu (tools['mv'])

    # create superclips
    pref_sup = mvu.Super(prefilter, blksize=bs, overlap=bs // 2, pel=pel, sharp=1, rfilter=2)
    low_sup  = mvu.Super(low_down,  blksize=bs, overlap=bs // 2, pel=pel, sharp=0, onelevel=True)

    # analyse and degrain
    low_vecs = mvu.AnalyseMany(pref_sup, radius=tr, search=2, searchparam=1, mvlambda=0, lsad=400, plevel=0, pnew=0, pzero=0, globalmv=False)
    low_degr = to_cpu(mvu.Degrain(low_down, low_sup, low_vecs, thsad=[thsad, thsad], planes=[0]))

    # merge
    low_degr = core.std.MaskedMerge(low_degr, low_down, motionmask)               # reduce blending/ghosting
    low_degr = core.resize.Bicubic(low_degr, width=low.width, height=low.height)  # resize back to original res
    return frequency_merge(low_degr, high, 10, 3, tools=tools)                                 # merge low freqs with original high freqs


def gen_shifts(clip: vs.VideoNode, radius: int) -> List[vs.VideoNode]:
    # create shifted versions of the input clip with mirror padding
    frames = clip.num_frames
    prefix = clip[1:radius + 1][::-1]
    suffix = clip[-2:-radius - 2:-1]
    padded = prefix + clip + suffix

    shifts = []
    for offset in range(-radius, radius + 1):
        if offset == 0:
            shifts.append(clip)
        else:
            start = radius + offset
            shifts.append(padded[start:start + frames])

    return shifts


def get_spans(length: int, tile_length: int, count: int) -> List[Tuple[int, int, int, int]]:
    if count == 1:
        return [(0, length, 0, length)]

    max_start = length - tile_length
    starts = [round(i * max_start / (count - 1)) for i in range(count)]

    spans = []
    for i, start in enumerate(starts):
        dst0 = 0 if i == 0 else (starts[i - 1] + tile_length + start) // 2
        dst1 = length if i == count - 1 else (start + tile_length + starts[i + 1]) // 2
        spans.append((start, start + tile_length, dst0, dst1))
    return spans


def get_tiles(clip_w: int, clip_h: int, tiles: int, overlap: int = 0) -> Tuple[int, int, int, int]:
    # calculate tile size and choose the most square layout
    if tiles not in (1, 2, 4, 6, 8):
        raise ValueError("vs_temporalfix: Tiles must be 1, 2, 4, 6, or 8.")

    layouts = {
        1: [(1, 1)],
        2: [(2, 1), (1, 2)],
        4: [(4, 1), (2, 2), (1, 4)],
        6: [(6, 1), (3, 2), (2, 3), (1, 6)],
        8: [(8, 1), (4, 2), (2, 4), (1, 8)],
    }[tiles]

    def _tile_size(layout: Tuple[int, int]) -> Tuple[int, int]:
        cols, rows = layout
        tile_w = math.ceil((clip_w + 2 * overlap * (cols - 1)) / cols)
        tile_h = math.ceil((clip_h + 2 * overlap * (rows - 1)) / rows)
        return tile_w, tile_h

    def _layout_valid(layout: Tuple[int, int]) -> bool:
        # tiles must have a positive non overlapped stride
        cols, rows = layout
        tile_w, tile_h = _tile_size(layout)
        if cols > 1 and tile_w <= 2 * overlap:
            return False
        if rows > 1 and tile_h <= 2 * overlap:
            return False
        return True

    def _score(layout: Tuple[int, int]) -> Tuple[float, int, int]:
        tile_w, tile_h = _tile_size(layout)
        cols, rows = layout

        tile_aspect = tile_w / tile_h
        square_error = abs(math.log(tile_aspect))

        orientation_penalty = rows if clip_w >= clip_h else cols
        balance_penalty = abs(cols - rows)

        return (square_error, orientation_penalty, balance_penalty)

    valid_layouts = [layout for layout in layouts if _layout_valid(layout)]
    if not valid_layouts:
        raise ValueError("vs_temporalfix: Clip dimensions are too small for current tile amount. Reduce tiles.")

    cols, rows = min(valid_layouts, key=_score)
    tile_w, tile_h = _tile_size((cols, rows))
    return tile_w, tile_h, cols, rows


def interpolate_onnx(onnx_path_lower: str, onnx_path_upper: str, save_path: str, weighting: float) -> None:
    # interpolate two onnx models
    import onnx
    import numpy as np
    from onnx import TensorProto, numpy_helper

    float_types = {TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE}
    model_lower = onnx.load(onnx_path_lower)
    model_upper = onnx.load(onnx_path_upper)
    init_upper  = {init.name: init for init in model_upper.graph.initializer}

    for i, init_lower in enumerate(model_lower.graph.initializer):
        if init_lower.data_type in float_types:
            array_lower = numpy_helper.to_array(init_lower)
            array_upper = numpy_helper.to_array(init_upper[init_lower.name])
            array_lerp  = (array_lower.astype(np.float32) * (1.0 - weighting) + array_upper.astype(np.float32) * weighting).astype(array_lower.dtype, copy=False)
            model_lower.graph.initializer[i].CopyFrom(numpy_helper.from_array(array_lerp, name=init_lower.name))

    onnx.save_model(model_lower, save_path)
