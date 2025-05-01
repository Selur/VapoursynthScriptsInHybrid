import vapoursynth as vs
from vapoursynth import core

from typing import Sequence

# taken from old havsfunc
def FixChromaBleedingMod(input: vs.VideoNode, cx: int = 4, cy: int = 4, thr: float = 4.0, strength: float = 0.8, blur: bool = False) -> vs.VideoNode:
    from color import Tweak

    if not isinstance(input, vs.VideoNode):
        raise vs.Error('FixChromaBleedingMod: input must be a clip')

    if input.format.color_family != vs.YUV or input.format.sample_type != vs.INTEGER:
        raise vs.Error('FixChromaBleedingMod: only YUV format with integer sample type is supported')

    # Extract V plane and apply initial tweak
    v_plane = plane(input, 2)
    v_tweaked = Tweak(v_plane, sat=thr)
    
    if blur:
        mask_area = v_tweaked.std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    else:
        mask_area = v_tweaked

    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    # Apply basic mask processing (ensure dynamic level scaling)
    mask = EXPR(mask_area, ['x 128 - abs 2 *'])
    mask = mask.std.Inflate()

    # Chroma shifting and strength adjustment
    shifted_chroma = Tweak(input.resize.Spline16(src_left=cx, src_top=cy), sat=strength)

    # Masked merging for U and V planes
    u_result = core.std.MaskedMerge(plane(input, 1), plane(shifted_chroma, 1), mask)
    v_result = core.std.MaskedMerge(plane(input, 2), plane(shifted_chroma, 2), mask)

    return join([plane(input, 0), u_result, v_result])


# Helpers

# from vsutil
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

# from vsutil
def join(planes: Sequence[vs.VideoNode], family: vs.ColorFamily = vs.YUV) -> vs.VideoNode:
    """Joins the supplied sequence of planes into a single VideoNode (defaults to YUV).

    >>> planes = [Y, U, V]
    >>> clip_YUV = join(planes)
    >>> plane = core.std.BlankClip(format=vs.GRAY8)
    >>> clip_GRAY = join([plane], family=vs.GRAY)

    :param planes:  Sequence of one-plane ``vapoursynth.GRAY`` clips to merge.
    :param family:  Output color family.

    :return:        Merged clip of the supplied `planes`.
    """
    return planes[0] if len(planes) == 1 and family == vs.GRAY \
        else core.std.ShufflePlanes(planes, [0, 0, 0], family)