from vapoursynth import core
import vapoursynth as vs

import math

from typing import Optional, Union, Sequence




##############################
# FastLineDarken 1.4x MT MOD #
##############################
#
# Written by Vectrangle    (http://forum.doom9.org/showthread.php?t=82125)
# Didée: - Speed Boost, Updated: 11th May 2007
# Dogway - added protection option. 12-May-2011F
#
# Parameters are:
#  strength (integer)   - Line darkening amount, 0-256. Default 48. Represents the _maximum_ amount
#                         that the luma will be reduced by, weaker lines will be reduced by
#                         proportionately less.
#  protection (integer) - Prevents the darkest lines from being darkened. Protection acts as a threshold.
#                         Values range from 0 (no prot) to ~50 (protect everything)
#  luma_cap (integer)   - value from 0 (black) to 255 (white), used to stop the darkening
#                         determination from being 'blinded' by bright pixels, and to stop grey
#                         lines on white backgrounds being darkened. Any pixels brighter than
#                         luma_cap are treated as only being as bright as luma_cap. Lowering
#                         luma_cap tends to reduce line darkening. 255 disables capping. Default 191.
#  threshold (integer)  - any pixels that were going to be darkened by an amount less than
#                         threshold will not be touched. setting this to 0 will disable it, setting
#                         it to 4 (default) is recommended, since often a lot of random pixels are
#                         marked for very slight darkening and a threshold of about 4 should fix
#                         them. Note if you set threshold too high, some lines will not be darkened
#  thinning (integer)   - optional line thinning amount, 0-256. Setting this to 0 will disable it,
#                         which is gives a _big_ speed increase. Note that thinning the lines will
#                         inherently darken the remaining pixels in each line a little. Default 0.
def FastLineDarkenMOD(c, strength=48, protection=5, luma_cap=191, threshold=4, thinning=0):
    if not isinstance(c, vs.VideoNode):
        raise vs.Error('FastLineDarkenMOD: this is not a clip')

    if c.format.color_family == vs.RGB:
        raise vs.Error('FastLineDarkenMOD: RGB format is not supported')

    peak = (1 << c.format.bits_per_sample) - 1 if c.format.sample_type == vs.INTEGER else 1.0

    if c.format.color_family != vs.GRAY:
        c_orig = c
        c = GetPlane(c, 0)
    else:
        c_orig = None

    ## parameters ##
    Str = strength / 128
    lum = scale(luma_cap, peak)
    thr = scale(threshold, peak)
    thn = thinning / 16

    ## filtering ##
    exin = c.std.Maximum(threshold=peak / (protection + 1)).std.Minimum()
    EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else (core.akarin.Expr if hasattr(core, 'akarin') else core.std.Expr)
    thick = EXPR([c, exin], expr=[f'y {lum} < y {lum} ? x {thr} + > x y {lum} < y {lum} ? - 0 ? {Str} * x +'])
    if thinning <= 0:
        last = thick
    else:
        diff = EXPR([c, exin], expr=[f'y {lum} < y {lum} ? x {thr} + > x y {lum} < y {lum} ? - 0 ? {scale(127, peak)} +'])
        linemask = EXPR(diff.std.Minimum(), expr=[f'x {scale(127, peak)} - {thn} * {peak} +']).std.Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])
        thin = EXPR([c.std.Maximum(), diff], expr=[f'x y {scale(127, peak)} - {Str} 1 + * +'])
        last = core.std.MaskedMerge(thin, thick, linemask)

    if c_orig is not None:
        last = core.std.ShufflePlanes([last, c_orig], planes=[0, 1, 2], colorfamily=c_orig.format.color_family)
    return last
    
    
#####################
## Toon v0.82 edit ##
#####################
#
# function created by mf
#   support by Soulhunter ;-)
#   ported to masktools v2 and optimized by Didee (0.82)
#   added parameters and smaller changes by MOmonster (0.82 edited)
#
# toon v0.8 is the newest light-weight build of mf´s nice line darken function mf_toon
#
# Parameters:
#  str (float) - Strength of the line darken. Default is 1.0
#  l_thr (int) - Lower threshold for the linemask. Default is 2
#  u_thr (int) - Upper threshold for the linemask. Default is 12
#  blur (int)  - "blur" parameter of AWarpSharp2. Default is 2
#  depth (int) - "depth" parameter of AWarpSharp2. Default is 32
def Toon(input, str=1.0, l_thr=2, u_thr=12, blur=2, depth=32):
    if not isinstance(input, vs.VideoNode):
        raise vs.Error('Toon: this is not a clip')

    if input.format.color_family == vs.RGB:
        raise vs.Error('Toon: RGB format is not supported')

    neutral = 1 << (input.format.bits_per_sample - 1)
    peak = (1 << input.format.bits_per_sample) - 1
    multiple = peak / 255

    if input.format.color_family != vs.GRAY:
        input_orig = input
        input = GetPlane(input, 0)
    else:
        input_orig = None

    lthr = neutral + scale(l_thr, peak)
    lthr8 = lthr / multiple
    uthr = neutral + scale(u_thr, peak)
    uthr8 = uthr / multiple
    ludiff = u_thr - l_thr

    last = core.std.MakeDiff(input.std.Maximum().std.Minimum(), input)
    EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else (core.akarin.Expr if hasattr(core, 'akarin') else core.std.Expr)
    last = EXPR([last, Padding(last, 6, 6, 6, 6).warp.AWarpSharp2(blur=blur, depth=depth).std.Crop(6, 6, 6, 6)], expr=['x y min'])
    expr = f'y {lthr} <= {neutral} y {uthr} >= x {uthr8} y {multiple} / - 128 * x {multiple} / y {multiple} / {lthr8} - * + {ludiff} / {multiple} * ? {neutral} - {str} * {neutral} + ?'
    last = core.std.MakeDiff(input, EXPR([last, last.std.Maximum()], expr=[expr]))

    if input_orig is not None:
        last = core.std.ShufflePlanes([last, input_orig], planes=[0, 1, 2], colorfamily=input_orig.format.color_family)
    return last

def Padding(clip: vs.VideoNode, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('Padding: this is not a clip')

    if left < 0 or right < 0 or top < 0 or bottom < 0:
        raise vs.Error('Padding: border size to pad must not be negative')

    width = clip.width + left + right
    height = clip.height + top + bottom

    return clip.resize.Point(width, height, src_left=-left, src_top=-top, src_width=width, src_height=height)

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