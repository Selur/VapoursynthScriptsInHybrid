import vapoursynth as vs
from vapoursynth import core
from typing import List

from helpers import GetPlane, NNEDI3
import masked


def RainbowSmooth(clip, radius=3, lthresh=0, hthresh=220, mask="original"):
    if isinstance(mask, str):
        if mask == "original":
            EXPR = core.akarin.Expr if hasattr(core, "akarin") else core.cranexpr.Expr if hasattr(core, "cranexpr") else core.std.Expr
            mask = EXPR(clips=[clip.std.Maximum(planes=0), clip.std.Minimum(planes=0)], expr=["x y - 90 > 255 x y - 255 90 / * ?", "", ""])
        elif mask == "prewitt":
            PREWITT = core.edgemasks.ExPrewitt if hasattr(core,"edgemasks") else core.std.Prewitt
            mask = PREWITT(clip, planes=0)
        elif mask == "sobel":
            SOBEL = core.edgemasks.Sobel if hasattr(core,"edgemasks") else core.std.Sobel
            mask = SOBEL(clip, planes=0)
        elif mask == "tcanny":
            mask = core.tcanny.TCanny(clip)
        elif mask == "fast_sobel":
            mask = masked.fast_sobel(clip)
        elif mask == "kirsch":
            KIRSCH = core.edgemasks.Kirsch if hasattr(core,"edgemasks") else core.std.kirsch
            mask = KIRSCH(clip)
        elif mask == "retinex_edgemask":
            mask = Depth(masked.retinex_edgemask(clip), clip.format.bits_per_sample)

    lderain = clip

    if lthresh > 0:
        lderain = clip.smoothuv.SmoothUV(radius=radius, threshold=lthresh, interlaced=False)

    hderain = clip.smoothuv.SmoothUV(radius=radius, threshold=hthresh, interlaced=False)

    if hthresh > lthresh:
        return core.std.MaskedMerge(clipa=lderain, clipb=hderain, mask=mask, planes=[1, 2], first_plane=True)

    return lderain


def derainbow(clip: vs.VideoNode) -> vs.VideoNode:
    EXPR = core.akarin.Expr if hasattr(core, "akarin") else core.cranexpr.Expr if hasattr(core, "cranexpr") else core.std.Expr

    pre = clip[0] + clip[:-1]
    post = clip[1:] + clip[-1]

    rainbowmask = EXPR(
        [GetPlane(pre, 0), GetPlane(clip, 1), GetPlane(post, 0), GetPlane(pre, 1), GetPlane(clip, 2), GetPlane(post, 1)],
        "x y - abs x z - abs + a b - abs a c - abs + + 20 / 5 pow 20 *"
    ).resize.Point(clip.width, clip.height)

    luma = GetPlane(clip, 0)

    linemask = EXPR(clips=[luma.std.Maximum(), luma.std.Minimum()], expr="x y - 90 > 255 x y - 255 90 / * ?").std.Maximum()

    rainbowmask = core.std.MaskedMerge(core.std.BlankClip(rainbowmask), rainbowmask, linemask)

    derainbow = RainbowSmooth(clip, mask=rainbowmask, radius=4, lthresh=0, hthresh=90)

    nnedi3 = NNEDI3(derainbow, field=2, nsize=4, nns=0, planes=[1, 2])

    nnedi3_c = core.artyfox.Mean([nnedi3[::2], nnedi3[1::2]]) if hasattr(core, "artyfox") else core.average.Mean([nnedi3[::2], nnedi3[1::2]])

    return core.std.ShufflePlanes([derainbow, nnedi3_c], [0, 1, 2], vs.YUV)

