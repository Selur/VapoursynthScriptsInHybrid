#####################################################
#                                                   #
# Hysteria, a line darkening script by Scintilla    #
# Last updated 6/13/17                              #
#                                                   #
#####################################################
#
# Syntax:
# Hysteria(clip, strength= 1.0, usemask=True, lowthresh=6, highthresh=20, luma_cap=191, maxchg=255, minchg=0, planes = [0], luma=True, showmask=False)
#
# Requires YV12 input, frame-based only.  Is reasonably fast.
# Suggestions for improvement welcome: scintilla@aquilinestudios.org
#
#
# Arguments:
#
# strength (default=1.0) - This is a multiplicative factor for the amounts
#    by which the pixels are darkened.  Ordinarily, each pixel is
#    darkened by the difference between its luma value and the average
#    luma value of its brighter spatial neighbours.  So if you want more
#    darkening, increase this value.
#
# usemask (default=True) - Whether or not to apply the mask.  If False,
#    the entire image will have its edges darkened instead of just the
#    edges detected in the mask.  Could be useful on some sources
#    (specifically, it will often make dark lines look thicker), but
#    you will probably want to stick to lower values of strength if you
#    choose to go that route.
#
# lowthresh (default=6) - This is the threshold used for the noisy mask.
#    Increase this value if your mask is picking up too much noise
#    around the edges, or decrease it if the mask is not being grown
#    thick enough to cover all of each edge.
#
# highthresh (default=20) - This is the threshold used for the clean mask.
#    Increase this value if your mask is picking up too many weaker
#    edges, or decrease it if the mask is not picking up enough.
#
# luma_cap (default=191) - An idea swiped from FLD/VMToon.  Any pixels
#    brighter than this value will not be darkened at all, no matter
#    what the rest of the parameters are.  This is useful if you have
#    lighter edges that you do not want darkened.  0 will result in
#    no darkening at all, while 255 will turn off the cap.
#
# maxchg (default=255) - No pixel will be darkened by more than this
#    amount, no matter how high you set the strength parameter.
#    This can be useful if you want to darken your weaker lines more
#    without going overboard on the stronger ones.  0 will result in
#    no darkening at all, while 255 (the default) will turn off the
#    limiting.
#
# minchg (default=0) - Another idea swiped from FLD/VMToon (though in
#    those functions it was called "threshold").  Any pixels that
#    would have been darkened by less than this amount will instead
#    not be darkened at all.  This can be useful if you have noise
#    that is getting darkened slightly.  0 (the default) will turn
#    off the thresholding, while 255 will result in no darkening at all.
# 
# planes (default=0) - Luma plane
#
# luma (default=True) - Use luma plane for masking
#    
# showmask (default=False) - When True, the function will display the
#    current edge mask plus the chroma from the original image.
#    Use this to find the optimal values of lowthresh and highthresh.
#
###################
#
# Changelog:
#
# 9/11/10: Is this thing on?
# 10/4/15: Port to VapourSynth by Overdrive80
# 6/13/17: removed dependency, fixed parameter scaling and masking by BluBb_mADe
#
###################


import vapoursynth as vs


def Hysteria(clip, strength=1.0, usemask=True, lowthresh=6, highthresh=20, luma_cap=191, maxchg=255, minchg=0,
             planes=[0], luma=True, showmask=False):
    core = vs.core
    if not isinstance(clip, vs.VideoNode):
        raise ValueError('This is not a clip')

    max_bitval = (1 << clip.format.bits_per_sample) - 1

    def scale(old_value):
        return int((old_value * max_bitval) / 255)

    # This scales the colordepth dependant parameters
    if clip.format.bits_per_sample != 8:
        lowthresh = scale(lowthresh)
        highthresh = scale(highthresh)

        luma_cap = scale(luma_cap)

        maxchg = scale(maxchg)
        minchg = scale(minchg)

    # Medium value
    mid = (2 ** clip.format.bits_per_sample) // 2

    # imitate mt_edge(mode=cartoon) (stolen from Frechdachs)
    noisymask = core.std.Convolution(clip, matrix=[0, -2, 1, 0, 1, 0, 0, 0, 0], planes=planes, saturate=True)
    noisymask = core.std.Expr(noisymask, ['x {high} >= {maxvalue} x {low} <= 0 x ? ?'
                              .format(low=lowthresh, high=lowthresh, maxvalue=max_bitval)])

    cleanmask = core.std.Convolution(clip, matrix=[0, -2, 1, 0, 1, 0, 0, 0, 0], planes=planes, saturate=True)
    cleanmask = core.std.Expr(cleanmask, ['x {high} >= {maxvalue} x {low} <= 0 x ? ?'
                              .format(low=highthresh, high=highthresh, maxvalue=max_bitval)])

    themask = core.misc.Hysteresis(cleanmask, noisymask)
    themask = core.std.Inflate(themask)

    # blur replacement
    themask = core.std.Convolution(themask, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    themask = core.std.Convolution(themask, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])

    themask = core.std.Deflate(themask)

    clipa = core.std.Inflate(clip, planes=[0])
    diffs = core.std.MakeDiff(clipa, clip)
    diffs = core.std.Expr([diffs], ['x {mid} - {strength} *'.format(strength=strength, mid=mid)])

    darkened = core.std.Expr([clip, diffs], [
        'x x {luma_cap} > 0 y {maxchg} > {maxchg} y {minchg} < 0 y ? ? ? -'.format(luma_cap=luma_cap, maxchg=maxchg,
                                                                                   minchg=minchg)])

    if usemask:
        themask = core.std.ShufflePlanes(themask, planes=[0], colorfamily=vs.GRAY)
        final = core.std.MaskedMerge(clip, darkened, themask, planes=planes, first_plane=luma)
    else:
        final = core.std.ShufflePlanes(clips=[darkened, clip], planes=[0, 1, 2], colorfamily=vs.YUV)

    if not showmask:
        return final

    mascara = core.std.Levels(themask, min_in=0, max_in=max_bitval, min_out=scale(80), max_out=max_bitval)
    mascara = core.std.ShufflePlanes([mascara, clip], planes=[0, 1, 2], colorfamily=vs.YUV)
    return mascara
