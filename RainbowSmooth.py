import vapoursynth as vs
from vapoursynth import core
from typing import List

from helpers import Depth, GetPlane, NNEDI3, get_expr, pick_tool
import masked


def RainbowSmooth(clip, radius=3, lthresh=0, hthresh=220, mask="original", tools=None):
    if isinstance(mask, str):
        if mask == "original":
            EXPR = get_expr(tools)
            mask = EXPR(clips=[clip.std.Maximum(planes=0), clip.std.Minimum(planes=0)], expr=["x y - 90 > 255 x y - 255 90 / * ?", "", ""])
        elif mask == "prewitt":
            PREWITT = core.edgemasks.ExPrewitt if pick_tool(tools, 'edgemasks', ('edgemasks', 'std')) == 'edgemasks' else core.std.Prewitt
            mask = PREWITT(clip, planes=0)
        elif mask == "sobel":
            SOBEL = core.edgemasks.Sobel if pick_tool(tools, 'edgemasks', ('edgemasks', 'std')) == 'edgemasks' else core.std.Sobel
            mask = SOBEL(clip, planes=0)
        elif mask == "tcanny":
            mask = core.tcanny.TCanny(clip)
        elif mask == "fast_sobel":
            mask = masked.fast_sobel(clip, tools=tools)
        elif mask == "kirsch":
            mask = core.edgemasks.Kirsch(clip) if pick_tool(tools, 'edgemasks', ('edgemasks', 'std')) == 'edgemasks' else masked.kirsch(clip, tools=tools)
        elif mask == "retinex_edgemask":
            mask = Depth(masked.retinex_edgemask(clip, tools=tools), clip.format.bits_per_sample)

    lderain = clip

    if lthresh > 0:
        lderain = clip.smoothuv.SmoothUV(radius=radius, threshold=lthresh, interlaced=False)

    hderain = clip.smoothuv.SmoothUV(radius=radius, threshold=hthresh, interlaced=False)

    if hthresh > lthresh:
        return core.std.MaskedMerge(clipa=lderain, clipb=hderain, mask=mask, planes=[1, 2], first_plane=True)

    return lderain


def derainbow(clip: vs.VideoNode, tools=None) -> vs.VideoNode:
    EXPR = get_expr(tools)

    pre = clip[0] + clip[:-1]
    post = clip[1:] + clip[-1]

    rainbowmask = EXPR(
        [GetPlane(pre, 0), GetPlane(clip, 1), GetPlane(post, 0), GetPlane(pre, 1), GetPlane(clip, 2), GetPlane(post, 1)],
        "x y - abs x z - abs + a b - abs a c - abs + + 20 / 5 pow 20 *"
    ).resize.Point(clip.width, clip.height)

    luma = GetPlane(clip, 0)

    linemask = EXPR(clips=[luma.std.Maximum(), luma.std.Minimum()], expr="x y - 90 > 255 x y - 255 90 / * ?").std.Maximum()

    rainbowmask = core.std.MaskedMerge(core.std.BlankClip(rainbowmask), rainbowmask, linemask)

    derainbow = RainbowSmooth(clip, mask=rainbowmask, radius=4, lthresh=0, hthresh=90, tools=tools)

    nnedi3 = NNEDI3(derainbow, field=2, nsize=4, nns=0, planes=[1, 2], tools=tools)

    nnedi3_c = core.artyfox.Mean([nnedi3[::2], nnedi3[1::2]]) if pick_tool(tools, 'average', ('artyfox', 'average')) == 'artyfox' else core.average.Mean([nnedi3[::2], nnedi3[1::2]])

    return core.std.ShufflePlanes([derainbow, nnedi3_c], [0, 1, 2], vs.YUV)

