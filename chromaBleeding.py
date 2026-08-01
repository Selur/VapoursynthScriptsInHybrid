import vapoursynth as vs
from vapoursynth import core

from typing import Sequence

from helpers import GetPlane

# taken from old havsfunc
def FixChromaBleedingMod(input: vs.VideoNode, cx: int = 4, cy: int = 4, thr: float = 4.0, strength: float = 0.8, blur: bool = False) -> vs.VideoNode:
    from color import Tweak

    if not isinstance(input, vs.VideoNode):
        raise vs.Error('FixChromaBleedingMod: input must be a clip')

    if input.format.color_family != vs.YUV or input.format.sample_type != vs.INTEGER:
        raise vs.Error('FixChromaBleedingMod: only YUV format with integer sample type is supported')

    # Extract V plane and apply initial tweak
    v_plane = GetPlane(input, 2)
    v_tweaked = Tweak(v_plane, sat=thr)
    
    if blur:
        mask_area = v_tweaked.std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    else:
        mask_area = v_tweaked

    EXPR = core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr
    # Apply basic mask processing (ensure dynamic level scaling)
    mask = EXPR(mask_area, ['x 128 - abs 2 *'])
    mask = mask.std.Inflate()

    # Chroma shifting and strength adjustment
    shifted_chroma = Tweak(input.resize.Spline16(src_left=cx, src_top=cy), sat=strength)

    # Masked merging for U and V planes
    u_result = core.std.MaskedMerge(GetPlane(input, 1), GetPlane(shifted_chroma, 1), mask)
    v_result = core.std.MaskedMerge(GetPlane(input, 2), GetPlane(shifted_chroma, 2), mask)

    return core.std.ShufflePlanes([GetPlane(input, 0), u_result, v_result],[0, 0, 0], vs.YUV)