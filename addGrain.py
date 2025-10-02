import vapoursynth as vs
from vapoursynth import core

from typing import Union, Sequence, Optional
import math

# Taken from old havsfunc
# Parameters:
#  g1str (float)       - [0.0 - ???] strength of grain / for dark areas. Default is 7.0
#  g2str (float)       - [0.0 - ???] strength of grain / for midtone areas. Default is 5.0
#  g3str (float)       - [0.0 - ???] strength of grain / for bright areas. Default is 3.0
#  g1shrp (int)        - [0 - 100] sharpness of grain / for dark areas (NO EFFECT when g1size=1.0 !!). Default is 60
#  g2shrp (int)        - [0 - 100] sharpness of grain / for midtone areas (NO EFFECT when g2size=1.0 !!). Default is 66
#  g3shrp (int)        - [0 - 100] sharpness of grain / for bright areas (NO EFFECT when g3size=1.0 !!). Default is 80
#  g1size (float)      - [0.5 - 4.0] size of grain / for dark areas. Default is 1.5
#  g2size (float)      - [0.5 - 4.0] size of grain / for midtone areas. Default is 1.2
#  g3size (float)      - [0.5 - 4.0] size of grain / for bright areas. Default is 0.9
#  temp_avg (int)      - [0 - 100] percentage of noise's temporal averaging. Default is 0
#  ontop_grain (float) - [0 - ???] additional grain to put on top of prev. generated grain. Default is 0.0
#  seed (int)          - specifies a repeatable grain sequence. Set to at least 0 to use.
#  th1 (int)           - start of dark->midtone mixing zone. Default is 24
#  th2 (int)           - end of dark->midtone mixing zone. Default is 56
#  th3 (int)           - start of midtone->bright mixing zone. Default is 128
#  th4 (int)           - end of midtone->bright mixing zone. Default is 160
def GrainFactory3(clp, g1str=7.0, g2str=5.0, g3str=3.0, g1shrp=60, g2shrp=66, g3shrp=80, g1size=1.5, g2size=1.2, g3size=0.9, temp_avg=0, ontop_grain=0.0, seed=-1, th1=24, th2=56, th3=128, th4=160):
    if not isinstance(clp, vs.VideoNode):
        raise vs.Error('GrainFactory3: this is not a clip')

    if clp.format.color_family == vs.RGB:
        raise vs.Error('GrainFactory3: RGB format is not supported')

    if clp.format.sample_type == vs.INTEGER:
        neutral = 1 << (clp.format.bits_per_sample - 1)
        peak = (1 << clp.format.bits_per_sample) - 1
    else:
        neutral = 0.0
        peak = 1.0

    if clp.format.color_family != vs.GRAY:
        clp_orig = clp
        clp = GetPlane(clp, 0)
    else:
        clp_orig = None

    ox = clp.width
    oy = clp.height
    sx1 = m4(ox / g1size)
    sy1 = m4(oy / g1size)
    sx1a = m4((ox + sx1) / 2)
    sy1a = m4((oy + sy1) / 2)
    sx2 = m4(ox / g2size)
    sy2 = m4(oy / g2size)
    sx2a = m4((ox + sx2) / 2)
    sy2a = m4((oy + sy2) / 2)
    sx3 = m4(ox / g3size)
    sy3 = m4(oy / g3size)
    sx3a = m4((ox + sx3) / 2)
    sy3a = m4((oy + sy3) / 2)

    b1 = g1shrp / -50 + 1
    b2 = g2shrp / -50 + 1
    b3 = g3shrp / -50 + 1
    b1a = b1 / 2
    b2a = b2 / 2
    b3a = b3 / 2
    c1 = (1 - b1) / 2
    c2 = (1 - b2) / 2
    c3 = (1 - b3) / 2
    c1a = (1 - b1a) / 2
    c2a = (1 - b2a) / 2
    c3a = (1 - b3a) / 2
    tmpavg = temp_avg / 100
    th1 = scale(th1, peak)
    th2 = scale(th2, peak)
    th3 = scale(th3, peak)
    th4 = scale(th4, peak)

    grainlayer1 = clp.std.BlankClip(width=sx1, height=sy1, color=[neutral]).grain.Add(var=g1str, seed=seed)
    if g1size != 1 and (sx1 != ox or sy1 != oy):
        if g1size > 1.5:
            grainlayer1 = grainlayer1.resize.Bicubic(sx1a, sy1a, filter_param_a=b1a, filter_param_b=c1a).resize.Bicubic(ox, oy, filter_param_a=b1a, filter_param_b=c1a)
        else:
            grainlayer1 = grainlayer1.resize.Bicubic(ox, oy, filter_param_a=b1, filter_param_b=c1)

    grainlayer2 = clp.std.BlankClip(width=sx2, height=sy2, color=[neutral]).grain.Add(var=g2str, seed=seed)
    if g2size != 1 and (sx2 != ox or sy2 != oy):
        if g2size > 1.5:
            grainlayer2 = grainlayer2.resize.Bicubic(sx2a, sy2a, filter_param_a=b2a, filter_param_b=c2a).resize.Bicubic(ox, oy, filter_param_a=b2a, filter_param_b=c2a)
        else:
            grainlayer2 = grainlayer2.resize.Bicubic(ox, oy, filter_param_a=b2, filter_param_b=c2)

    grainlayer3 = clp.std.BlankClip(width=sx3, height=sy3, color=[neutral]).grain.Add(var=g3str, seed=seed)
    if g3size != 1 and (sx3 != ox or sy3 != oy):
        if g3size > 1.5:
            grainlayer3 = grainlayer3.resize.Bicubic(sx3a, sy3a, filter_param_a=b3a, filter_param_b=c3a).resize.Bicubic(ox, oy, filter_param_a=b3a, filter_param_b=c3a)
        else:
            grainlayer3 = grainlayer3.resize.Bicubic(ox, oy, filter_param_a=b3, filter_param_b=c3)

    expr1 = f'x {th1} < 0 x {th2} > {peak} {peak} {th2 - th1} / x {th1} - * ? ?'
    expr2 = f'x {th3} < 0 x {th4} > {peak} {peak} {th4 - th3} / x {th3} - * ? ?'
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    grainlayer = core.std.MaskedMerge(core.std.MaskedMerge(grainlayer1, grainlayer2, EXPR(clp, expr=[expr1])), grainlayer3, EXPR(clp, expr=[expr2]))
    if temp_avg > 0:
        grainlayer = core.std.Merge(grainlayer, AverageFrames(grainlayer, weights=[1] * 3), weight=[tmpavg])
    if ontop_grain > 0:
        grainlayer = grainlayer.grain.Add(var=ontop_grain, seed=seed)

    result = core.std.MakeDiff(clp, grainlayer)

    if clp_orig is not None:
        result = core.std.ShufflePlanes([result, clp_orig], planes=[0, 1, 2], colorfamily=clp_orig.format.color_family)
    return result
    
    
# HELPERS

def cround(x: float) -> int:
    return math.floor(x + 0.5) if x > 0 else math.ceil(x - 0.5)

def m4(x: Union[float, int]) -> int:
    return 16 if x < 16 else cround(x / 4) * 4

def scale(value, peak):
    return cround(value * peak / 255) if peak != 1 else value / 255

# from vsutil    
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
    
def AverageFrames(
    clip: vs.VideoNode, weights: Union[float, Sequence[float]], scenechange: Optional[float] = None, planes: Optional[Union[int, Sequence[int]]] = None
) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('AverageFrames: this is not a clip')

    if scenechange:
        clip = SCDetect(clip, threshold=scenechange)
    return clip.std.AverageFrames(weights=weights, scenechange=scenechange, planes=planes)
    
    
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