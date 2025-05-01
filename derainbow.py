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
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
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