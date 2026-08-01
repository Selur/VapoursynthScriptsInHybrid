import vapoursynth as vs
from vapoursynth import core

from typing import Union, Sequence, Optional
import math
from helpers import GetPlane, m4, scale

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
def GrainFactory3(
    clp: vs.VideoNode,
    g1str: float = 7.0,
    g2str: float = 5.0,
    g3str: float = 3.0,
    g1shrp: int = 60,
    g2shrp: int = 66,
    g3shrp: int = 80,
    g1size: float = 1.5,
    g2size: float = 1.2,
    g3size: float = 0.9,
    temp_avg: int = 0,
    ontop_grain: float = 0.0,
    seed: int = -1,
    th1: int = 24,
    th2: int = 56,
    th3: int = 128,
    th4: int = 160,
) -> vs.VideoNode:
# Validate input.
    if not isinstance(clp, vs.VideoNode):
        raise vs.Error("GrainFactory3: this is not a clip")

    if clp.format is None:
        raise vs.Error("GrainFactory3: clip must have a constant format")

    if clp.format.color_family == vs.RGB:
        raise vs.Error("GrainFactory3: RGB format is not supported")

    # Determine the neutral gray value and maximum sample value for the clip format.
    if clp.format.sample_type == vs.INTEGER:
        neutral: int | float = 1 << (clp.format.bits_per_sample - 1)
        peak: int | float = (1 << (clp.format.bits_per_sample)) - 1
    else:
        neutral = 0.0
        peak = 1.0

    # Grain is generated only on the luma plane and later recombined with chroma.
    if clp.format.color_family != vs.GRAY:
        clp_orig: vs.VideoNode | None = clp
        clp = GetPlane(clp, 0)
    else:
        clp_orig = None

    # Calculate intermediate resolutions for each grain layer.
    # Grain size is controlled by generating noise at a different resolution
    # and scaling it back to the original dimensions.
    ox: int = clp.width
    oy: int = clp.height
    sx1: int = m4(ox / g1size)
    sy1: int = m4(oy / g1size)
    sx1a: int = m4((ox + sx1) / 2)
    sy1a: int = m4((oy + sy1) / 2)
    sx2: int = m4(ox / g2size)
    sy2: int = m4(oy / g2size)
    sx2a: int = m4((ox + sx2) / 2)
    sy2a: int = m4((oy + sy2) / 2)
    sx3: int = m4(ox / g3size)
    sy3: int = m4(oy / g3size)
    sx3a: int = m4((ox + sx3) / 2)
    sy3a: int = m4((oy + sy3) / 2)

    # Convert the user "sharpness" setting into Bicubic filter parameters.
    # Lower values produce softer, coarser grain while higher values preserve
    # finer grain detail during resizing.
    b1: float = g1shrp / -50 + 1
    b2: float = g2shrp / -50 + 1
    b3: float = g3shrp / -50 + 1
    b1a: float = b1 / 2
    b2a: float = b2 / 2
    b3a: float = b3 / 2
    c1: float = (1 - b1) / 2
    c2: float = (1 - b2) / 2
    c3: float = (1 - b3) / 2
    c1a: float = (1 - b1a) / 2
    c2a: float = (1 - b2a) / 2
    c3a: float = (1 - b3a) / 2

    # Convert temporal averaging percentage to Merge weight.
    tmpavg: float = temp_avg / 100

    # Scale luma thresholds to the clip's bit depth.
    th1 = scale(th1, peak)
    th2 = scale(th2, peak)
    th3 = scale(th3, peak)
    th4 = scale(th4, peak)

    # Dark grain layer
    grainlayer1: vs.VideoNode = clp.std.BlankClip(width=sx1, height=sy1, color=[neutral])

    # Use the newer noise plugin if available, otherwise fall back to grain.
    GRAIN = core.noise.Add if hasattr(core, "noise") else core.grain.Add
    grainlayer1 = GRAIN(grainlayer1, var=g1str, seed=seed)

    # Resize the grain back to the original resolution.
    # Large grain sizes (>1.5) use a two-stage resize to reduce scaling artifacts.
    if g1size != 1 and (sx1 != ox or sy1 != oy):
        if g1size > 1.5:
            grainlayer1 = grainlayer1.resize.Bicubic(sx1a, sy1a, filter_param_a=b1a, filter_param_b=c1a).resize.Bicubic(ox, oy, filter_param_a=b1a, filter_param_b=c1a)
        else:
            grainlayer1 = grainlayer1.resize.Bicubic(ox, oy, filter_param_a=b1, filter_param_b=c1)

    # Midtone grain layer
    grainlayer2: vs.VideoNode = clp.std.BlankClip(width=sx2, height=sy2, color=[neutral])
    grainlayer2 = GRAIN(grainlayer2, var=g2str, seed=seed)

    if g2size != 1 and (sx2 != ox or sy2 != oy):
        if g2size > 1.5:
            grainlayer2 = grainlayer2.resize.Bicubic(sx2a, sy2a, filter_param_a=b2a, filter_param_b=c2a).resize.Bicubic(ox, oy, filter_param_a=b2a, filter_param_b=c2a)
        else:
            grainlayer2 = grainlayer2.resize.Bicubic(ox, oy, filter_param_a=b2, filter_param_b=c2)

    # Bright grain layer
    grainlayer3: vs.VideoNode = clp.std.BlankClip(width=sx3, height=sy3, color=[neutral])
    grainlayer3 = GRAIN(grainlayer3, var=g3str, seed=seed)

    if g3size != 1 and (sx3 != ox or sy3 != oy):
        if g3size > 1.5:
            grainlayer3 = grainlayer3.resize.Bicubic(sx3a, sy3a, filter_param_a=b3a, filter_param_b=c3a).resize.Bicubic(ox, oy, filter_param_a=b3a, filter_param_b=c3a)
        else:
            grainlayer3 = grainlayer3.resize.Bicubic(ox, oy, filter_param_a=b3, filter_param_b=c3)

    # Build masks that smoothly blend between the three grain layers based on luma.
    expr1: str = f"x {th1} < 0 x {th2} > {peak} {peak} {th2 - th1} / x {th1} - * ? ?"
    expr2: str = f"x {th3} < 0 x {th4} > {peak} {peak} {th4 - th3} / x {th3} - * ? ?"

    # Prefer akarin.Expr, then cranexpr, then std.Expr.
    EXPR = core.akarin.Expr if hasattr(core, "akarin") else core.cranexpr.Expr if hasattr(core, "cranexpr") else core.std.Expr

    # Blend dark → midtone → bright grain according to the luma masks.
    grainlayer: vs.VideoNode = core.std.MaskedMerge(core.std.MaskedMerge(grainlayer1, grainlayer2, EXPR(clp, expr=[expr1])), grainlayer3, EXPR(clp, expr=[expr2]))

    # Optionally reduce temporal noise by averaging neighbouring frames.
    if temp_avg > 0:
        import misc
        grainlayer = core.std.Merge(grainlayer, misc.AverageFrames(grainlayer, weights=[1] * 3), weight=[tmpavg])

    # Optionally add a final layer of fine grain over the result.
    if ontop_grain > 0:
        grainlayer = GRAIN(grainlayer, var=ontop_grain, seed=seed)

    # Convert the generated grain into a difference clip and apply it.
    result: vs.VideoNode = core.std.MakeDiff(clp, grainlayer)

    # Restore the original chroma planes if the input was not grayscale.
    if clp_orig is not None:
        result = core.std.ShufflePlanes([result, clp_orig], planes=[0, 1, 2], colorfamily=clp_orig.format.color_family)

    return result
    

