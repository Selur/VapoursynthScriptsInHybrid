
# Script by pifroggi https://github.com/pifroggi/vs_temporalfix
# or tepete and pifroggi on Discord

import re
import math
import vapoursynth as vs

core = vs.core


def temporal_median(clip, radius=1, planes=None):
    # fallback plugin because zsmooth does not support non AVX2 CPUs
    if hasattr(core, "zsmooth"):
        return core.zsmooth.TemporalMedian(clip, radius=radius, planes=planes)
    else:
        return core.tmedian.TemporalMedian(clip, radius=radius, planes=planes)


def repair(clip, repairclip, mode=[1]):
    # fallback plugin because zsmooth does not support non AVX2 CPUs
    if hasattr(core, "zsmooth"):
        return core.zsmooth.Repair(clip, repairclip, mode=mode)
    else:
        return core.rgvs.Repair(clip, repairclip, mode=mode)


def median(clip, radius=1, planes=None):
    # fallback plugin because zsmooth does not support non AVX2 CPUs
    if hasattr(core, "zsmooth"):
        return core.zsmooth.Median(clip, radius=radius, planes=planes)
    elif radius == 1:
        return core.std.Median(clip, planes=planes)
    else:
        return core.ctmf.CTMF(clip, radius=radius, planes=planes)


def expression(clips, expr, format=None):
    # optional plugin for slight speed boost
    if hasattr(core, "akarin"):
        return core.akarin.Expr(clips, expr, format=format)
    else:
        return core.std.Expr(clips, expr, format=format)


def box_blur(clip, planes=None, hradius=1, hpasses=1, vradius=1, vpasses=1):
    # optional plugin for slight speed boost
    if hasattr(core, "vszip"):
        return core.vszip.BoxBlur(clip, planes=planes, hradius=hradius, hpasses=hpasses, vradius=vradius, vpasses=vpasses)
    else:
        return core.std.BoxBlur(clip, planes=planes, hradius=hradius, hpasses=hpasses, vradius=vradius, vpasses=vpasses)


def min_blur(clip, planes=[0, 1, 2]):
    # simplified function from G41Fun https://github.com/Vapoursynth-Plugins-Gitify/G41Fun
    # original avisynth function by Didée https://avisynth.nl/index.php/MinBlur

    if clip.format.num_planes == 1:
        planes = [0]
    if isinstance(planes, int):
        planes = [planes]
    
    mat1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    mat2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    RG11 = core.std.Convolution(clip, matrix=mat1, planes=planes).std.Convolution(matrix=mat2, planes=planes)
    RG4  = median(clip, radius=2, planes=planes)
    expr = "x y - x z - * 0 < x dup y - abs x z - abs < y z ? ?"
    return expression([clip, RG11, RG4], [expr if i in planes else "" for i in range(clip.format.num_planes)])


def average_color_fix(clip, ref, radius=4, passes=4):
    # simplified from https://github.com/pifroggi/vs_colorfix
    blurred_reference = box_blur(ref, hradius=radius, hpasses=passes, vradius=radius, vpasses=passes)
    blurred_clip = box_blur(clip, hradius=radius, hpasses=passes, vradius=radius, vpasses=passes)
    diff_clip = core.std.MakeDiff(blurred_reference, blurred_clip)
    return core.std.MergeDiff(clip, diff_clip)


def average_color_fix_fast(clip, ref, downscale_factor=8):
    # faster but faint blocky artifacts
    downscaled_reference = core.resize.Bilinear(ref, width=clip.width / downscale_factor, height=clip.height / downscale_factor)
    downscaled_clip = core.resize.Bilinear(clip, width=clip.width / downscale_factor, height=clip.height / downscale_factor)
    diff_clip = core.std.MakeDiff(downscaled_reference, downscaled_clip)
    diff_clip = core.resize.Bilinear(diff_clip, width=clip.width, height=clip.height)
    return core.std.MergeDiff(clip, diff_clip)


def frequency_merge(low, high, radius=40, passes=3):
    # merges low freqs of one clip with high freqs of another clip
    low_remaining  = box_blur(low,  hradius=radius, hpasses=passes, vradius=radius, vpasses=passes)
    high_removed   = box_blur(high, hradius=radius, hpasses=passes, vradius=radius, vpasses=passes)
    high_remaining = core.std.MakeDiff(high, high_removed)
    return core.std.MergeDiff(low_remaining, high_remaining)


def mvsf_analyze(sup, tr, args):
    #function simplified from mvmulti: https://github.com/IFeelBloated/vapoursynth-mvtools-sf/blob/r9/src/mvmulti.py
    vecs = []
    for i in range(1, tr + 1):
        vecs += [
            core.mvsf.Analyze(sup, isb=True,  delta=i, **args),
            core.mvsf.Analyze(sup, isb=False, delta=i, **args),
        ]
    return vecs


def mvsf_degrain(clip, sup, vecs, tr, args):
    #function simplified from mvmulti: https://github.com/IFeelBloated/vapoursynth-mvtools-sf/blob/r9/src/mvmulti.py
    if not 1 <= tr <= 24:
        raise ValueError("vs_temporalfix: Temporal radius (tr) can not be larger than 24 with mvtools-sf r9 or older. Upgrade to r10 pre-release or newer for larger radii.")

    degrain = getattr(core.mvsf, f"Degrain{tr}")
    return degrain(clip, sup, *vecs, **args)


def mv_analyze(sup, tr, args):
    bv1 = core.mv.Analyse(sup, isb=True,  delta=1, **args)
    fv1 = core.mv.Analyse(sup, isb=False, delta=1, **args)
    vecs = [bv1, fv1]

    if tr > 1:
        bv2 = core.mv.Analyse(sup, isb=True,  delta=2, **args)
        fv2 = core.mv.Analyse(sup, isb=False, delta=2, **args)
        vecs += [bv2, fv2]
    if tr > 2:
        bv3 = core.mv.Analyse(sup, isb=True,  delta=3, **args)
        fv3 = core.mv.Analyse(sup, isb=False, delta=3, **args)
        vecs += [bv3, fv3]
    if tr > 3:
        bv4 = core.mv.Analyse(sup, isb=True,  delta=4, **args)
        fv4 = core.mv.Analyse(sup, isb=False, delta=4, **args)
        vecs += [bv4, fv4]
    if tr > 4:
        bv5 = core.mv.Analyse(sup, isb=True,  delta=5, **args)
        fv5 = core.mv.Analyse(sup, isb=False, delta=5, **args)
        vecs += [bv5, fv5]
    if tr > 5:
        bv6 = core.mv.Analyse(sup, isb=True,  delta=6, **args)
        fv6 = core.mv.Analyse(sup, isb=False, delta=6, **args)
        vecs += [bv6, fv6]

    return vecs


def mv_degrain(clip, sup, vecs, tr, args):
    if tr == 6:
        return core.mv.Degrain6(clip, sup, *vecs, **args)
    elif tr == 5:
        return core.mv.Degrain5(clip, sup, *vecs, **args)
    elif tr == 4:
        return core.mv.Degrain4(clip, sup, *vecs, **args)
    elif tr == 3:
        return core.mv.Degrain3(clip, sup, *vecs, **args)
    elif tr == 2:
        return core.mv.Degrain2(clip, sup, *vecs, **args)
    elif tr == 1:
        return core.mv.Degrain1(clip, sup, *vecs, **args)
    raise ValueError("Temporal radius (tr) must be in the range 1-6.")


def tweak_darks(src, strength=2.5, amp=0.2):
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
    return core.std.Expr([src], expr)


def contrasharp(clip, src, rep=24, planes=[0, 1, 2]):
    # simplified function from G41Fun https://github.com/Vapoursynth-Plugins-Gitify/G41Fun
    # original avisynth function by Didée at the VERY GRAINY thread https://forum.doom9.org/showthread.php?p=1076491

    if clip.format.num_planes == 1:
        planes = [0]
    if isinstance(planes, int):
        planes = [planes]

    mat1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    mat2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    bd   = clip.format.bits_per_sample
    mid  = 1 << (bd - 1)
    num  = clip.format.num_planes

    s    = min_blur(clip, planes)  # damp down remaining spots of the denoised clip
    RG11 = core.std.Convolution(s, matrix=mat1, planes=planes).std.Convolution(matrix=mat2, planes=planes)
    ssD  = core.std.MakeDiff(s, RG11, planes)  # the difference of a simple kernel blur
    allD = core.std.MakeDiff(src, clip, planes)  # the difference achieved by the denoising
    ssDD = repair(ssD, allD, [rep if i in planes else 0 for i in range(num)])  # limit the difference to the max of what the denoising removed locally
    expr = "x {} - abs y {} - abs < x y ?".format(mid, mid)  # abs(diff) after limiting may not be bigger than before
    ssDD = expression([ssDD, ssD], [expr if i in planes else "" for i in range(num)])
    return core.std.MergeDiff(clip, ssDD, planes)  # apply the limited difference (sharpening is just inverse blurring)


def exclude_regions(clip, replacement, exclude=None):
    # simplified ReplaceFrames function from fvsfunc https://github.com/Irrational-Encoding-Wizardry/fvsfunc
    # which is a port of ReplaceFramesSimple by James D. Lin http://avisynth.nl/index.php/RemapFrames
    import re

    if exclude is None:
        return clip
    if not isinstance(exclude, str):
        raise TypeError('vs_temporalfix: Exclusions are set like this: exclude="[100 300] [600 900] [2000 2500]", where the first number in the brackets is the start frame and the second is the end frame (inclusive).')

    exclude = exclude.replace(",", " ").replace(":", " ")
    frames = re.findall(r"\d+(?!\d*\s*\d*\s*\d*\])", exclude)
    ranges = re.findall(r"\[\s*\d+\s+\d+\s*\]", exclude)
    maps = []
    for range_ in ranges:
        maps.append([int(x) for x in range_.strip("[ ]").split()])
    for frame in frames:
        maps.append([int(frame), int(frame)])

    for start, end in maps:
        if start > end:
            raise ValueError("vs_temporalfix: Exclusions start frame is bigger than end frame: [{} {}]".format(start, end))
        if start >= clip.num_frames:
            raise ValueError("vs_temporalfix: Exclusions start frame {} is outside the clip, which has only {} frames.".format(start, clip.num_frames))

    out = clip
    for start, end in maps:
        temp = replacement[start : end + 1]
        temp = replacement[start : end + 1]
        if start != 0:
            temp = out[:start] + temp
        if end < out.num_frames - 1:
            temp = temp + out[end + 1 :]
        out = temp
    return out


def lowfreq_denoise(low, high, motionmask, thsad=200, tr=6):
    # temporally denoise low frequencies only

    bs  = 8
    pel = 1
    analyze_args = dict(blksize=bs, overlap=bs // 2, search=4, searchparam=1, truemotion=False)
    degrain_args = dict(thsad=thsad, plane=0)

    # downscale clips
    downscale_factor = 8
    low_down   = core.resize.Bicubic(low,      width=low.width // downscale_factor, height=low.height // downscale_factor)
    motionmask = core.resize.Point(motionmask, width=low.width // downscale_factor, height=low.height // downscale_factor)
    motionmask = core.std.Maximum(motionmask)                  # expand mask
    prefilter  = tweak_darks(low_down, strength=2.5, amp=0.2)  # brighten darks

    # create super clips
    pref_sup = core.mv.Super(prefilter, pel=pel, sharp=1, rfilter=4)
    low_sup  = core.mv.Super(low_down,  pel=pel, sharp=0, rfilter=1, levels=1)

    # analyze and degrain
    low_vecs = mv_analyze(pref_sup, tr, analyze_args)
    low_degr = mv_degrain(low_down, low_sup, low_vecs, tr, degrain_args)

    # merge
    low_degr = core.std.MaskedMerge(low_degr, low_down, motionmask)               # reduce blending/ghosting
    low_degr = core.resize.Bicubic(low_degr, width=low.width, height=low.height)  # resize back to original res
    return frequency_merge(low_degr, high, 10, 3)                                 # merge low freqs with original high freqs


def gen_shifts(clip, radius):
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


def get_spans(length, tile_length, count):
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


def get_tiles(clip_w, clip_h, tiles, overlap=0):
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

    def _tile_size(layout):
        cols, rows = layout
        tile_w = math.ceil((clip_w + 2 * overlap * (cols - 1)) / cols)
        tile_h = math.ceil((clip_h + 2 * overlap * (rows - 1)) / rows)
        return tile_w, tile_h

    def _layout_valid(layout):
        # tiles must have a positive non overlapped stride
        cols, rows = layout
        tile_w, tile_h = _tile_size(layout)
        if cols > 1 and tile_w <= 2 * overlap:
            return False
        if rows > 1 and tile_h <= 2 * overlap:
            return False
        return True

    def _score(layout):
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