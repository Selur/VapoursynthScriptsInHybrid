import vapoursynth as vs
from vapoursynth import core
from typing import List
import masked
import mvsfunc as mvf

def RainbowSmooth(clip, radius=3, lthresh=0, hthresh=220, mask="original"):
    core = vs.core
    
    if isinstance(mask, str):
        if mask == "original":
            mask = core.std.Expr(clips=[clip.std.Maximum(planes=0), clip.std.Minimum(planes=0)], expr=["x y - 90 > 255 x y - 255 90 / * ?", "", ""])
        elif mask == "prewitt":
            mask = core.std.Prewitt(clip=clip, planes=0)
        elif mask == "sobel":
            mask = core.std.Sobel(clip=clip, planes=0)
        elif mask == "tcanny":
            mask = core.tcanny.TCanny(clip)
        elif mask == "fast_sobel":
            mask = masked.fast_sobel(clip)
        elif mask == "kirsch":
            mask = masked.kirsch(clip)
        elif mask == "retinex_edgemask":
            mask = masked.retinex_edgemask(clip)
            mask = mvf.Depth(mask, clip.format.bits_per_sample)

    lderain = clip

    if lthresh > 0:
        lderain = clip.smoothuv.SmoothUV(radius=radius, threshold=lthresh, interlaced=False)

    hderain = clip.smoothuv.SmoothUV(radius=radius, threshold=hthresh, interlaced=False)

    if hthresh > lthresh:
        return core.std.MaskedMerge(clipa=lderain, clipb=hderain, mask=mask, planes=[1, 2], first_plane=True)
    else:
        return lderain

def derainbow(clip: vs.VideoNode) -> vs.VideoNode:
    clip_pre = split(clip[0] + clip[:-1])[1:]
    clip_post = split(clip[1:] + clip[-1])[1:]
    clip_chroma = split(clip)[1:]
    rainbowmask = core.std.Expr(
        [clip_pre[0], clip_chroma[0], clip_post[0], clip_pre[1], clip_chroma[1], clip_post[1]],
        "x y - abs x z - abs + a b - abs a c - abs + + 20 / 5 pow 20 *",
    ).resize.Point(clip.width, clip.height)
    clip_luma = get_y(clip)
    linemask = core.std.Expr(
        clips=[clip_luma.std.Maximum(), clip_luma.std.Minimum()],
        expr="x y - 90 > 255 x y - 255 90 / * ?",
    ).std.Maximum()
    rainbowmask = core.std.MaskedMerge(core.std.BlankClip(rainbowmask), rainbowmask, linemask)
    derainbow = RainbowSmooth(clip, mask=rainbowmask, radius=4, lthresh=0, hthresh=90)
    # Old technique for dealing with dot crawl
    # generate a mask from transposed nnedi3 to get just vertical lines
    # luma_transposed = get_y(derainbow).std.Transpose()
    # nnedi3_v = core.nnedi3cl.NNEDI3CL(luma_transposed, 2, nsize=4, nns=0)
    # nnedi3_v = (
    #     core.std.Expr(
    #         [luma_transposed, nnedi3_v[::2], nnedi3_v[1::2]],
    #         "x y - abs x z - abs < z y ? x - abs 2 / 2 pow 2 *",
    #     )
    #     .std.Transpose()
    #     .std.Maximum()
    #     .std.Inflate()
    # )
    nnedi3 = core.nnedi3cl.NNEDI3CL(derainbow, 2, nsize=4, nns=0, planes=[1, 2])
    # Old technique for dealing with dot crawl - part 2
    # double rate deinterlace, then choose whichever field was furthest from source.
    # this completely removes dots, but also does a sort of weird vertical blur
    # masking with nnedi3_v sort of works, but the mask isnt perfect, and the effect is too strong
    # to not have a near perfect mask
    # nnedi3_l = core.std.Expr(
    #     [get_y(derainbow), get_y(nnedi3[::2]), get_y(nnedi3[1::2])],
    #     "x y - abs x z - abs < z y ?",
    # ).rgvs.Repair(get_y(derainbow), 1)
    # nnedi3_l = core.std.MaskedMerge(get_y(derainbow), nnedi3_l, nnedi3_v)
    nnedi3_c = core.average.Mean([nnedi3[::2], nnedi3[1::2]])
    nnedi3 = core.std.ShufflePlanes([derainbow, nnedi3_c], [0, 1, 2], vs.YUV)
    return nnedi3
    

def plane(clip: vs.VideoNode, planeno: int, /) -> vs.VideoNode:
    """Extracts the plane with the given index from the input clip.

    If given a one-plane clip and ``planeno=0``, returns `clip` (no-op).

    >>> src = vs.core.std.BlankClip(format=vs.YUV420P8)
    >>> V = plane(src, 2)

    :param clip:     The clip to extract the plane from.
    :param planeno:  The index of which plane to extract.

    :return:         A grayscale clip that only contains the given plane.
    """
    if clip.format.num_planes == 1 and planeno == 0:
        return clip
    return core.std.ShufflePlanes(clip, planeno, vs.GRAY)
    
def get_y(clip: vs.VideoNode, /) -> vs.VideoNode:
    """Helper to get the luma plane of a clip.

    If passed a single-plane ``vapoursynth.GRAY`` clip, :func:`plane` will assume it to `be` the luma plane
    itself and returns the `clip` (no-op).

    :param clip: Input clip.

    :return:     Luma plane of the input `clip`. Will return the input `clip` if it is a single-plane grayscale clip.
    """
    if clip.format.color_family not in (vs.YUV, vs.GRAY):
        raise ValueError('The clip must have a luma plane.')
    return plane(clip, 0)


def split(clip: vs.VideoNode, /) -> List[vs.VideoNode]:
    """Returns a list of planes (VideoNodes) from the given input clip.

    >>> src = vs.core.std.BlankClip(format=vs.RGB27)
    >>> R, G, B  = split(src)

    :param clip:  Input clip.

    :return:      List of planes from the input `clip`.
    """
    return [plane(clip, x) for x in range(clip.format.num_planes)]