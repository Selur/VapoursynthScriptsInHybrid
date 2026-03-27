import vapoursynth as vs
from vapoursynth import core

import math


#####################################################
#                                                   #
# LUTDeRainbow, a derainbowing script by Scintilla  #
# Last updated 2022-10-08                           #
#                                                   #
#####################################################
#
# Requires YUV input, frame-based only.
# Is of reasonable speed (faster than aWarpSharp, slower than DeGrainMedian).
# Suggestions for improvement welcome: scintilla@aquilinestudios.org
#
# Arguments:
#
# cthresh (int, default=10) - This determines how close the chroma values of the
#   pixel in the previous and next frames have to be for the pixel to
#   be hit.  Higher values (within reason) should catch more rainbows,
#   but may introduce unwanted artifacts.  Probably shouldn't be set
#   above 20 or so.
#
# ythresh (int, default=10) - If the y parameter is set true, then this
#   determines how close the luma values of the pixel in the previous
#   and next frames have to be for the pixel to be hit.  Just as with
#   cthresh.
#
# y (bool, default=True) - Determines whether luma difference will be considered
#   in determining which pixels to hit and which to leave alone.
#
# linkUV (bool, default=True) - Determines whether both chroma channels are
#   considered in determining which pixels in each channel to hit.
#   When set true, only pixels that meet the thresholds for both U and
#   V will be hit; when set false, the U and V channels are masked
#   separately (so a pixel could have its U hit but not its V, or vice
#   versa).
#
# mask (bool, default=False) - When set true, the function will return the mask
#   (for combined U/V) instead of the image.  Formerly used to find the
#   best values of cthresh and ythresh.  If linkUV=false, then this
#   mask won't actually be used anyway (because each chroma channel
#   will have its own mask).
#
###################
def LUTDeRainbow(input, cthresh=10, ythresh=10, y=True, linkUV=True, mask=False):
    inputbits = input.format.bits_per_sample
    if not isinstance(input, vs.VideoNode) or input.format.color_family != vs.YUV or inputbits > 16:
        raise vs.Error('LUTDeRainbow: This is not an 8-16 bit YUV clip')

    # Since LUT2 can't handle clips with more than 10 bits, we default to using
    # Expr and MaskedMerge to handle the same logic for higher bit depths.
    useExpr = inputbits > 10                                    

    shift = inputbits - 8
    peak = (1 << inputbits) - 1

    cthresh = scale(cthresh, peak)
    ythresh = scale(ythresh, peak)

    input_minus = input.std.DuplicateFrames(frames=[0])
    input_plus = input.std.Trim(first=1) + input.std.Trim(first=input.num_frames - 1)

    input_u = GetPlane(input, 1)
    input_v = GetPlane(input, 2)
    input_minus_y = GetPlane(input_minus, 0)
    input_minus_u = GetPlane(input_minus, 1)
    input_minus_v = GetPlane(input_minus, 2)
    input_plus_y = GetPlane(input_plus, 0)
    input_plus_u = GetPlane(input_plus, 1)
    input_plus_v = GetPlane(input_plus, 2)
    EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr
    average_y = EXPR([input_minus_y, input_plus_y], expr=[f'x y - abs {ythresh} < {peak} 0 ?']).resize.Bilinear(input_u.width, input_u.height)
    average_u = EXPR([input_minus_u, input_plus_u], expr=[f'x y - abs {cthresh} < x y + 2 / 0 ?'])
    average_v = EXPR([input_minus_v, input_plus_v], expr=[f'x y - abs {cthresh} < x y + 2 / 0 ?'])

    umask = average_u.std.Binarize(threshold=21 << shift)
    vmask = average_v.std.Binarize(threshold=21 << shift)
    if useExpr:
        themask = EXPR([umask, vmask], expr=[f'x y + {peak + 1} < 0 {peak} ?'])
        if y:
            umask = core.std.MaskedMerge(core.std.BlankClip(average_y), average_y, umask)
            vmask = core.std.MaskedMerge(core.std.BlankClip(average_y), average_y, vmask)
            themask = core.std.MaskedMerge(core.std.BlankClip(average_y), average_y, themask)
    else:                    
      themask = core.std.Lut2(umask, vmask, function=lambda x, y: x & y)
      if y:
          umask = core.std.Lut2(umask, average_y, function=lambda x, y: x & y)
          vmask = core.std.Lut2(vmask, average_y, function=lambda x, y: x & y)
          themask = core.std.Lut2(themask, average_y, function=lambda x, y: x & y)

    fixed_u = core.std.Merge(average_u, input_u)
    fixed_v = core.std.Merge(average_v, input_v)

    output_u = core.std.MaskedMerge(input_u, fixed_u, themask if linkUV else umask)
    output_v = core.std.MaskedMerge(input_v, fixed_v, themask if linkUV else vmask)

    output = core.std.ShufflePlanes([input, output_u, output_v], planes=[0, 0, 0], colorfamily=input.format.color_family)

    if mask:
        return themask.resize.Point(input.width, input.height)
    else:
        return output

def SRFComb(clip: vs.VideoNode,
            DotCrawlThSAD: int = 500,
            RainbowThSAD: int = 500,
            SpatialDeDotCraw: bool = True) -> vs.VideoNode:
    """
    SRFComb by real.finder
    (https://github.com/realfinder/AVS-Stuff/blob/master/avs%202.6%20and%20up/SRFComb.avsi)
    Best of many AVS Rainbow & Dot Crawl Removal approaches.
    Version 1.01 — converted to VapourSynth
    Requires: mv, rgvs, std, resize core plugins, 
              color.Tweak https://github.com/Selur/VapoursynthScriptsInHybrid/blob/7e1b4e2f19fa4a2bfbbf44518e987f76ebc2de59/color.py#L9
    Optional: zsmooth, vszip
      - zsmooth: faster Repair
      - vszip:   faster BoxBlur, and proper Checkmate port
                 (note: vszip.Checkmate is 8-bit only)
    """

    import color

    def BoxBlur(clip, hradius=1, hpasses=1, vradius=1, vpasses=1, planes=None):
        kwargs = dict(hradius=hradius, hpasses=hpasses, vradius=vradius, vpasses=vpasses)
        if planes is not None:
            kwargs['planes'] = planes
        if hasattr(core, 'vszip'):
            return core.vszip.BoxBlur(clip, **kwargs)
        return core.std.BoxBlur(clip, **kwargs)

    def Repair(clip, ref, mode):
        if hasattr(core, 'zsmooth'):
            return core.zsmooth.Repair(clip, ref, mode)
        return core.rgvs.Repair(clip, ref, mode)

    def Expr(clips, expr):
        if hasattr(core, 'llvmexpr'):
            return core.llvmexpr.Expr(clips, expr)
        elif hasattr(core, 'akarin'):
            return core.akarin.Expr(clips, expr)
        elif hasattr(core, 'cranexpr'):
            return core.cranexpr.Expr(clips, expr)
        else:
            return core.std.Expr(clips, expr)

    def Checkmate(clip):
        if hasattr(core, 'vszip'):
            bits = clip.format.bits_per_sample
            if bits != 8:
                clip8 = core.resize.Point(clip, format=clip.format.replace(bits_per_sample=8))
                result = core.vszip.Checkmate(clip8)
                return core.resize.Point(result, format=clip.format)
            return core.vszip.Checkmate(clip)
        return clip

    # --- Format checks ---
    fmt = clip.format
    assert fmt is not None,             "SRFComb: clip must have a known format"
    assert fmt.color_family != vs.RGB,  "SRFComb: Planar YUV input only"
    assert fmt.color_family != vs.GRAY, "SRFComb: does not work with Greyscale video"

    fullchr = fmt.subsampling_w == 0 and fmt.subsampling_h == 0  # 4:4:4
    chr420  = fmt.subsampling_w == 1 and fmt.subsampling_h == 1  # 4:2:0
    chr422  = fmt.subsampling_w == 1 and fmt.subsampling_h == 0  # 4:2:2

    peak  = (1 << fmt.bits_per_sample) - 1
    half  = 1 << (fmt.bits_per_sample - 1)
    scale = peak / 255

    thSAD  = DotCrawlThSAD
    thSADC = RainbowThSAD

    # Everything works in field-space until the final Weave
    separated = core.std.SeparateFields(clip, tff=True)  # adjust tff to match your source
    ogs = separated

    # --- LumaSpatialDeDot (field-space) ---
    if SpatialDeDotCraw:
        sep_y = core.std.ShufflePlanes(separated, 0, vs.GRAY)
        sep_u = core.std.ShufflePlanes(separated, 1, vs.GRAY)
        sep_v = core.std.ShufflePlanes(separated, 2, vs.GRAY)
        blurred        = BoxBlur(sep_y, hradius=1, hpasses=3, vradius=0, vpasses=0)
        sharpened      = core.std.Convolution(blurred, [0,-1,0,-1,5,-1,0,-1,0])
        luma_dedc      = core.std.ShufflePlanes([sharpened, sep_u, sep_v], [0,0,0], vs.YUV)
        LumaSpatialDeDot = Repair(luma_dedc, separated, 5)
    else:
        LumaSpatialDeDot = separated

    # --- pre: luma processing chain, separated to field-space ---
    ogy    = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    dedc   = Repair(ogy, BoxBlur(ogy, hradius=1, hpasses=1, vradius=1, vpasses=1), 1)
    tr1    = Repair(dedc, ogy, 3)
    cm     = Checkmate(tr1)
    tr2    = Repair(cm, ogy, 3)
    rep    = Repair(tr2, ogy, 1)
    rep_sf   = core.std.SeparateFields(rep, tff=True)
    ablurred = BoxBlur(rep_sf, hradius=1, hpasses=1, vradius=0, vpasses=0)
    blurred2 = BoxBlur(ablurred, hradius=1, hpasses=3, vradius=0, vpasses=0)
    pre_y    = core.std.Convolution(blurred2, [0,-1,0,-1,5,-1,0,-1,0])
    # Reassemble as YUV with chroma from ogs
    pre = core.std.ShufflePlanes(
        [pre_y,
         core.std.ShufflePlanes(ogs, 1, vs.GRAY),
         core.std.ShufflePlanes(ogs, 2, vs.GRAY)],
        [0, 0, 0], vs.YUV)

    # --- preymask: horizontal edge mask on separated luma (field-space, GRAY) ---
    ogy_sf    = core.std.SeparateFields(ogy, tff=True)
    ogy_vblur = BoxBlur(ogy_sf, vradius=1, vpasses=3, hradius=0, hpasses=0)
    ogy_vshrp = core.std.Convolution(ogy_vblur, [0,-1,0,-1,5,-1,0,-1,0])
    preymask  = core.std.Convolution(ogy_vshrp, [0,0,0,-1,0,1,0,0,0])
    preymask  = Expr(preymask, f"x {4*scale} < 0 x {8*scale} > {peak} x ? ?")

    # --- precmask: horizontal edge mask on pre_y (field-space, GRAY) ---
    pre_y_edge = core.std.Convolution(pre_y, [0,0,0,-1,0,1,0,0,0])
    precmask   = Expr(pre_y_edge, f"x {round(scale)} < 0 x {round(3*scale)} > {peak} x ? ?")

    # --- cuvmask: chroma activity mask (chroma-sized, GRAY) ---
    sat0  = color.Tweak(ogs, sat=0.0,  coring=False)
    sat10 = color.Tweak(ogs, sat=10.0, coring=False)
    diff_u = core.std.MakeDiff(
        core.std.ShufflePlanes(sat0,  1, vs.GRAY),
        core.std.ShufflePlanes(sat10, 1, vs.GRAY)
    )
    diff_v = core.std.MakeDiff(
        core.std.ShufflePlanes(sat0,  2, vs.GRAY),
        core.std.ShufflePlanes(sat10, 2, vs.GRAY)
    )
    lut_expr = f"x {half} = 0 x {half} - abs {peak} * {half} / ?"
    cuvmask  = Expr([Expr(diff_u, lut_expr), Expr(diff_v, lut_expr)], "x y max")

    # Upscale cuvmask to full luma field dimensions
    field_w = ogs.width
    field_h = ogs.height
    shift = 0.25 if (chr422 or chr420) else (None if fullchr else 0.375)
    cuvmaskf = core.resize.Bilinear(cuvmask, field_w, field_h,
                                    src_left=shift if shift is not None else 0)

    # --- premask: intersection of chroma activity and pre edge (field-space, GRAY) ---
    premask = Expr([cuvmaskf, precmask], "x y min")
    premask = core.std.Maximum(premask, coordinates=[0,0,0,1,1,0,0,0])
    premask = core.std.Inflate(premask)

    # --- ycombmask: intersection of chroma activity and luma edge (field-space, GRAY) ---
    ycombmask = Expr([cuvmaskf, preymask], "x y min")
    ycombmask = core.std.Maximum(ycombmask, coordinates=[0,0,0,1,1,0,0,0])
    ycombmask = core.std.Maximum(ycombmask, coordinates=[0,0,0,1,1,0,0,0])
    ycombmask = core.std.Inflate(ycombmask)

    # uvcombmask: downscale premask to chroma dimensions
    cuv_w     = diff_u.width
    cuv_h     = diff_u.height
    neg_shift = -0.25 if (chr422 or chr420) else (None if fullchr else -0.375)
    uvcombmask = core.resize.Bilinear(premask, cuv_w, cuv_h,
                                      src_left=neg_shift if neg_shift is not None else 0)
    # Upscale uvcombmask back to full luma field dimensions for MaskedMerge
    uvcombmask_full = core.resize.Bilinear(uvcombmask, field_w, field_h)

    # --- pre2: blend pre into ogs on luma using premask ---
    pre2 = core.std.MaskedMerge(ogs, pre, premask, planes=[0])
    pre2 = core.std.Merge(pre2, Repair(pre2, ogs, 1))

    # --- MVTools motion analysis ---
    super_search = core.mv.Super(pre2, pel=4, rfilter=4)
    bv1 = core.mv.Analyse(super_search, blksize=8, isb=True,  delta=2, overlap=2, dct=8)
    fv1 = core.mv.Analyse(super_search, blksize=8, isb=False, delta=2, overlap=2, dct=8)

    # --- lastLumaSpatialDeDot ---
    lastLSD = core.std.MaskedMerge(separated, LumaSpatialDeDot, ycombmask, planes=[0])

    # --- MDegrain1 ---
    super_lsd = core.mv.Super(lastLSD, pel=4, levels=1)
    degrained = core.mv.Degrain1(lastLSD, super_lsd, bv1, fv1,
                                 thsad=thSAD, thsadc=thSADC)

    # --- Final merge: ycombmask on luma, uvcombmask on chroma ---
    merged = core.std.MaskedMerge(ogs, degrained, ycombmask,        planes=[0])
    merged = core.std.MaskedMerge(merged, degrained, uvcombmask_full, planes=[1, 2])

    result = core.std.DoubleWeave(merged, tff=True)[::2]
    return result

# Taken from muvsfunc
def GetPlane(clip, plane=None):
    # input clip
    if not isinstance(clip, vs.VideoNode):
        raise type_error('"clip" must be a clip!')

    # Get properties of input clip
    sFormat = clip.format
    sNumPlanes = sFormat.num_planes

    # Parameters
    if plane is None:
        plane = 0
    elif not isinstance(plane, int):
        raise type_error('"plane" must be an int!')
    elif plane < 0 or plane > sNumPlanes:
        raise value_error(f'valid range of "plane" is [0, {sNumPlanes})!')

    # Process
    return core.std.ShufflePlanes(clip, plane, vs.GRAY)
    
def cround(x: float) -> int:
    return math.floor(x + 0.5) if x > 0 else math.ceil(x - 0.5)

def scale(value, peak):
    return cround(value * peak / 255) if peak != 1 else value / 255