import vapoursynth as vs
from vapoursynth import core


from typing import Optional, Union, Sequence

# Taken form old havsfunc
# Vinverse: a small, but effective function against (residual) combing, by Didée
# sstr: strength of contra sharpening
# amnt: change no pixel by more than this (default=255: unrestricted)
# chroma: chroma mode, True=process chroma, False=pass chroma through
# scl: scale factor for vshrpD*vblurD < 0
def Vinverse(clp, sstr=2.7, amnt=255, chroma=True, scl=0.25):
    if not isinstance(clp, vs.VideoNode):
        raise vs.Error('Vinverse: this is not a clip')

    if clp.format.sample_type == vs.INTEGER:
        neutral = 1 << (clp.format.bits_per_sample - 1)
        peak = (1 << clp.format.bits_per_sample) - 1
    else:
        neutral = 0.0
        peak = 1.0

    if not chroma and clp.format.color_family != vs.GRAY:
        clp_orig = clp
        clp = GetPlane(clp, 0)
    else:
        clp_orig = None

    vblur = clp.std.Convolution(matrix=[50, 99, 50], mode='v')
    vblurD = core.std.MakeDiff(clp, vblur)
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    vshrp = EXPR([vblur, vblur.std.Convolution(matrix=[1, 4, 6, 4, 1], mode='v')], expr=[f'x x y - {sstr} * +'])
    vshrpD = core.std.MakeDiff(vshrp, vblur)
    expr = f'x {neutral} - y {neutral} - * 0 < x {neutral} - abs y {neutral} - abs < x y ? {neutral} - {scl} * {neutral} + x {neutral} - abs y {neutral} - abs < x y ? ?'
    vlimD = EXPR([vshrpD, vblurD], expr=[expr])
    last = core.std.MergeDiff(vblur, vlimD)
    if amnt <= 0:
        return clp
    elif amnt < 255:
        last = EXPR([clp, last], expr=['x {AMN} + y < x {AMN} + x {AMN} - y > x {AMN} - y ? ?'.format(AMN=scale(amnt, peak))])

    if clp_orig is not None:
        last = core.std.ShufflePlanes([last, clp_orig], planes=[0, 1, 2], colorfamily=clp_orig.format.color_family)
    return last

# Taken form old havsfunc
def Vinverse2(clp, sstr=2.7, amnt=255, chroma=True, scl=0.25):
    if not isinstance(clp, vs.VideoNode):
        raise vs.Error('Vinverse2: this is not a clip')

    if clp.format.sample_type == vs.INTEGER:
        neutral = 1 << (clp.format.bits_per_sample - 1)
        peak = (1 << clp.format.bits_per_sample) - 1
    else:
        neutral = 0.0
        peak = 1.0

    if not chroma and clp.format.color_family != vs.GRAY:
        clp_orig = clp
        clp = GetPlane(clp, 0)
    else:
        clp_orig = None

    vblur = sbrV(clp)
    vblurD = core.std.MakeDiff(clp, vblur)
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    vshrp = EXPR([vblur, vblur.std.Convolution(matrix=[1, 2, 1], mode='v')], expr=[f'x x y - {sstr} * +'])
    vshrpD = core.std.MakeDiff(vshrp, vblur)
    expr = f'x {neutral} - y {neutral} - * 0 < x {neutral} - abs y {neutral} - abs < x y ? {neutral} - {scl} * {neutral} + x {neutral} - abs y {neutral} - abs < x y ? ?'
    vlimD = EXPR([vshrpD, vblurD], expr=[expr])
    last = core.std.MergeDiff(vblur, vlimD)
    if amnt <= 0:
        return clp
    elif amnt < 255:
        last = EXPR([clp, last], expr=['x {AMN} + y < x {AMN} + x {AMN} - y > x {AMN} - y ? ?'.format(AMN=scale(amnt, peak))])

    if clp_orig is not None:
        last = core.std.ShufflePlanes([last, clp_orig], planes=[0, 1, 2], colorfamily=clp_orig.format.color_family)
    return last

# Taken form old havsfunc
def sbrV(c: vs.VideoNode, r: int = 1, planes: Optional[Union[int, Sequence[int]]] = None) -> vs.VideoNode:
    if not isinstance(c, vs.VideoNode):
        raise vs.Error('sbrV: this is not a clip')

    neutral = 1 << (c.format.bits_per_sample - 1) if c.format.sample_type == vs.INTEGER else 0.0

    plane_range = range(c.format.num_planes)

    if planes is None:
        planes = list(plane_range)
    elif isinstance(planes, int):
        planes = [planes]

    matrix1 = [1, 2, 1]
    matrix2 = [1, 4, 6, 4, 1]

    RG11 = c.std.Convolution(matrix=matrix1, planes=planes, mode='v')
    if r >= 2:
        RG11 = RG11.std.Convolution(matrix=matrix2, planes=planes, mode='v')
    if r >= 3:
        RG11 = RG11.std.Convolution(matrix=matrix2, planes=planes, mode='v')

    RG11D = core.std.MakeDiff(c, RG11, planes=planes)

    RG11DS = RG11D.std.Convolution(matrix=matrix1, planes=planes, mode='v')
    if r >= 2:
        RG11DS = RG11DS.std.Convolution(matrix=matrix2, planes=planes, mode='v')
    if r >= 3:
        RG11DS = RG11DS.std.Convolution(matrix=matrix2, planes=planes, mode='v')
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    RG11DD = EXPR(
        [RG11D, RG11DS],
        expr=[f'x y - x {neutral} - * 0 < {neutral} x y - abs x {neutral} - abs < x y - {neutral} + x ? ?' if i in planes else '' for i in plane_range],
    )
    return core.std.MakeDiff(c, RG11DD, planes=planes)
    
    
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