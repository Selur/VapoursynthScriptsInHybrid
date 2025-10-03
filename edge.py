import vapoursynth as vs
from vapoursynth import core

import math
from functools import partial

# Taken from old havsfunc
# a.k.a. BalanceBordersMod
def bbmod(c, cTop, cBottom, cLeft, cRight, thresh=128, blur=999):
    if not isinstance(c, vs.VideoNode):
        raise vs.Error('bbmod: this is not a clip')

    if c.format.color_family in [vs.GRAY, vs.RGB]:
        raise vs.Error('bbmod: Gray and RGB formats are not supported')

    if thresh <= 0:
        raise vs.Error('bbmod: thresh must be greater than 0')

    if blur <= 0:
        raise vs.Error('bbmod: blur must be greater than 0')

    neutral = 1 << (c.format.bits_per_sample - 1)
    peak = (1 << c.format.bits_per_sample) - 1

    BicubicResize = partial(core.resize.Bicubic, filter_param_a=1, filter_param_b=0)

    def btb(c, cTop):
        cWidth = c.width
        cHeight = c.height
        cTop = min(cTop, cHeight - 1)
        blurWidth = max(8, math.floor(cWidth / blur))

        c2 = c.resize.Point(cWidth * 2, cHeight * 2)

        last = c2.std.CropAbs(width=cWidth * 2, height=2, top=cTop * 2)
        last = last.resize.Point(cWidth * 2, cTop * 2)
        EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else (core.akarin.Expr if hasattr(core, 'akarin') else core.std.Expr)
        referenceBlurChroma = BicubicResize(BicubicResize(EXPR(last, expr=[f'x {neutral} - abs 2 *', '']), blurWidth * 2, cTop * 2), cWidth * 2, cTop * 2)
        referenceBlur = BicubicResize(BicubicResize(last, blurWidth * 2, cTop * 2), cWidth * 2, cTop * 2)

        original = c2.std.CropAbs(width=cWidth * 2, height=cTop * 2)

        last = BicubicResize(original, blurWidth * 2, cTop * 2)
        originalBlurChroma = BicubicResize(BicubicResize(EXPR(last, expr=[f'x {neutral} - abs 2 *', '']), blurWidth * 2, cTop * 2), cWidth * 2, cTop * 2)
        originalBlur = BicubicResize(BicubicResize(last, blurWidth * 2, cTop * 2), cWidth * 2, cTop * 2)

        balancedChroma = EXPR([original, originalBlurChroma, referenceBlurChroma], expr=['', f'z y / 8 min 0.4 max x {neutral} - * {neutral} +'])
        expr = 'z {i} - y {i} - / 8 min 0.4 max x {i} - * {i} +'.format(i=scale(16, peak))
        balancedLuma = EXPR([balancedChroma, originalBlur, referenceBlur], expr=[expr, 'z y - x +'])

        difference = core.std.MakeDiff(balancedLuma, original)
        difference = EXPR(difference, expr=[f'x {scale(128 + thresh, peak)} min {scale(128 - thresh, peak)} max'])

        last = core.std.MergeDiff(original, difference)
        return core.std.StackVertical([last, c2.std.CropAbs(width=cWidth * 2, height=(cHeight - cTop) * 2, top=cTop * 2)]).resize.Point(cWidth, cHeight)

    if cTop > 0:
        c = btb(c, cTop)
    c = c.std.Transpose().std.FlipHorizontal()
    if cLeft > 0:
        c = btb(c, cLeft)
    c = c.std.Transpose().std.FlipHorizontal()
    if cBottom > 0:
        c = btb(c, cBottom)
    c = c.std.Transpose().std.FlipHorizontal()
    if cRight > 0:
        c = btb(c, cRight)
    return c.std.Transpose().std.FlipHorizontal()


# Helpers


def cround(x: float) -> int:
    return math.floor(x + 0.5) if x > 0 else math.ceil(x - 0.5)

def scale(value, peak):
    return cround(value * peak / 255) if peak != 1 else value / 255