import vapoursynth as vs
from vapoursynth import core

import math

from functools import partial


# Taken from havsfunc
########################################################
#                                                      #
# LUTDeCrawl, a dot crawl removal script by Scintilla  #
# Created 10/3/08                                      #
# Last updated 10/3/08                                 #
#                                                      #
########################################################
#
# Requires YUV input, frame-based only.
# Is of average speed (faster than VagueDenoiser, slower than HQDN3D).
# Suggestions for improvement welcome: scintilla@aquilinestudios.org
#
# Arguments:
#
# ythresh (int, default=10) - This determines how close the luma values of the
#   pixel in the previous and next frames have to be for the pixel to
#   be hit.  Higher values (within reason) should catch more dot crawl,
#   but may introduce unwanted artifacts.  Probably shouldn't be set
#   above 20 or so.
#
# cthresh (int, default=10) - This determines how close the chroma values of the
#   pixel in the previous and next frames have to be for the pixel to
#   be hit.  Just as with ythresh.
#
# maxdiff (int, default=50) - This is the maximum difference allowed between the
#   luma values of the pixel in the CURRENT frame and in each of its
#   neighbour frames (so, the upper limit to what fluctuations are
#   considered dot crawl).  Lower values will reduce artifacts but may
#   cause the filter to miss some dot crawl.  Obviously, this should
#   never be lower than ythresh.  Meaningless if usemaxdiff = false.
#
# scnchg (int, default=25) - Scene change detection threshold.  Any frame with
#   total luma difference between it and the previous/next frame greater
#   than this value will not be processed.
#
# usemaxdiff (bool, default=True) - Whether or not to reject luma fluctuations
#   higher than maxdiff.  Setting this to false is not recommended, as
#   it may introduce artifacts; but on the other hand, it produces a
#   30% speed boost.  Test on your particular source.
#
# mask (bool, default=False) - When set true, the function will return the mask
#   instead of the image.  Use to find the best values of cthresh,
#   ythresh, and maxdiff.
#   (The scene change threshold, scnchg, is not reflected in the mask.)
#
###################
def LUTDeCrawl(input, ythresh=10, cthresh=10, maxdiff=50, scnchg=25, usemaxdiff=True, mask=False):
    def YDifferenceFromPrevious(n, f, clips):
        if f.props['_SceneChangePrev']:
            return clips[0]
        else:
            return clips[1]

    def YDifferenceToNext(n, f, clips):
        if f.props['_SceneChangeNext']:
            return clips[0]
        else:
            return clips[1]

    if not isinstance(input, vs.VideoNode) or input.format.color_family != vs.YUV or input.format.bits_per_sample > 10:
        raise vs.Error('LUTDeCrawl: This is not an 8-10 bit YUV clip')

    shift = input.format.bits_per_sample - 8
    peak = (1 << input.format.bits_per_sample) - 1

    ythresh = scale(ythresh, peak)
    cthresh = scale(cthresh, peak)
    maxdiff = scale(maxdiff, peak)

    input_minus = input.std.DuplicateFrames(frames=[0])
    input_plus = input.std.Trim(first=1) + input.std.Trim(first=input.num_frames - 1)

    input_y = GetPlane(input, 0)
    input_minus_y = GetPlane(input_minus, 0)
    input_minus_u = GetPlane(input_minus, 1)
    input_minus_v = GetPlane(input_minus, 2)
    input_plus_y = GetPlane(input_plus, 0)
    input_plus_u = GetPlane(input_plus, 1)
    input_plus_v = GetPlane(input_plus, 2)
    EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr

    average_y = EXPR([input_minus_y, input_plus_y], expr=[f'x y - abs {ythresh} < x y + 2 / 0 ?'])
    average_u = EXPR([input_minus_u, input_plus_u], expr=[f'x y - abs {cthresh} < {peak} 0 ?'])
    average_v = EXPR([input_minus_v, input_plus_v], expr=[f'x y - abs {cthresh} < {peak} 0 ?'])

    ymask = average_y.std.Binarize(threshold=1 << shift)
    if usemaxdiff:
        diffplus_y = EXPR([input_plus_y, input_y], expr=[f'x y - abs {maxdiff} < {peak} 0 ?'])
        diffminus_y = EXPR([input_minus_y, input_y], expr=[f'x y - abs {maxdiff} < {peak} 0 ?'])
        diffs_y = core.std.Lut2(diffplus_y, diffminus_y, function=lambda x, y: x & y)
        ymask = core.std.Lut2(ymask, diffs_y, function=lambda x, y: x & y)
    cmask = core.std.Lut2(average_u.std.Binarize(threshold=129 << shift), average_v.std.Binarize(threshold=129 << shift), function=lambda x, y: x & y)
    cmask = cmask.resize.Point(input.width, input.height)

    themask = core.std.Lut2(ymask, cmask, function=lambda x, y: x & y)

    fixed_y = core.std.Merge(average_y, input_y)

    output = core.std.ShufflePlanes([core.std.MaskedMerge(input_y, fixed_y, themask), input], planes=[0, 1, 2], colorfamily=input.format.color_family)

    input = SCDetect(input, threshold=scnchg / 255)
    output = output.std.FrameEval(eval=partial(YDifferenceFromPrevious, clips=[input, output]), prop_src=input)
    output = output.std.FrameEval(eval=partial(YDifferenceToNext, clips=[input, output]), prop_src=input)

    if mask:
        return themask
    else:
        return output


# Helpers

def cround(x: float) -> int:
    return math.floor(x + 0.5) if x > 0 else math.ceil(x - 0.5)

def scale(value, peak):
    return cround(value * peak / 255) if peak != 1 else value / 255
    
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
    
def SCDetect(clip: vs.VideoNode, threshold: float = 0.1) -> vs.VideoNode:
    def copy_property(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        fout = f[0].copy()
        fout.props['_SceneChangePrev'] = f[1].props['_SceneChangePrev']
        fout.props['_SceneChangeNext'] = f[1].props['_SceneChangeNext']
        return fout

    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('SCDetect: this is not a clip')

    sc = clip
    if clip.format.color_family == vs.RGB:
        sc = clip.resize.Point(format=vs.GRAY8, matrix_s='709')

    sc = sc.misc.SCDetect(threshold=threshold)
    if clip.format.color_family == vs.RGB:
        sc = clip.std.ModifyFrame(clips=[clip, sc], selector=copy_property)

    return sc
