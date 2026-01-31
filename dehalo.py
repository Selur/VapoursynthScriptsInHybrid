import vapoursynth as vs
from vapoursynth import core

import math
from typing import Union, Optional, Sequence, TypeVar

from vsutil import scale_value

def DeHalo_alpha(
    clp: vs.VideoNode,
    rx: float = 2.0,
    ry: float = 2.0,
    darkstr: float = 1.0,
    brightstr: float = 1.0,
    lowsens: float = 50.0,
    highsens: float = 50.0,
    ss: float = 1.5,
) -> vs.VideoNode:
    '''
    Reduce halo artifacts that can occur when sharpening.

    Parameters:
        clp: Clip to process.

        rx, ry: As usual, the radii for halo removal. This function is rather sensitive to the radius settings.
            Set it as low as possible! If radius is set too high, it will start missing small spots.

        darkstr, brightstr: The strength factors for processing dark and bright halos. Default 1.0 both for symmetrical processing.
            On Comic/Anime, darkstr=0.4~0.8 sometimes might be better ... sometimes. In General, the function seems to preserve dark lines rather good.

        lowsens, highsens: Sensitivity settings, not that easy to describe them exactly ...
            In a sense, they define a window between how weak an achieved effect has to be to get fully accepted,
            and how strong an achieved effect has to be to get fully discarded.

        ss: Supersampling factor, to avoid creation of aliasing.
    '''
    if not isinstance(clp, vs.VideoNode):
        raise vs.Error('DeHalo_alpha: this is not a clip')

    if clp.format.color_family == vs.RGB:
        raise vs.Error('DeHalo_alpha: RGB format is not supported')

    bits = clp.format.bits_per_sample

    if clp.format.color_family != vs.GRAY:
        clp_orig = clp
        clp = get_y(clp)
    else:
        clp_orig = None

    ox = clp.width
    oy = clp.height
    EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else (core.akarin.Expr if hasattr(core, 'akarin') else core.std.Expr)
    halos = clp.resize.Bicubic(m4(ox / rx), m4(oy / ry), filter_param_a=1 / 3, filter_param_b=1 / 3).resize.Bicubic(ox, oy, filter_param_a=1, filter_param_b=0)
    are = EXPR([clp.std.Maximum(), clp.std.Minimum()], expr='x y -')
    ugly = EXPR([halos.std.Maximum(), halos.std.Minimum()], expr='x y -')
    so = EXPR(
        [ugly, are],
        expr=f'y x - y 0.000001 + / {scale_value(255, 8, bits)} * {scale_value(lowsens, 8, bits)} - y {scale_value(256, 8, bits)} + {scale_value(512, 8, bits)} / {highsens / 100} + *',
    )
    if clp.format.sample_type == vs.FLOAT:
        so = so.vszip.Limiter() if hasattr(core,'vszip') else so.std.Limiter()
    lets = core.std.MaskedMerge(halos, clp, so)
    if ss <= 1:
      if hasattr(core, 'zsmooth'):
        remove = core.zsmooth.Repair(clp, lets, mode=1)
      else:
        remove = core.rgvs.Repair(clp, lets, mode=1)
    else:
        remove = EXPR(
            [
                EXPR(
                    [
                        clp.resize.Lanczos(m4(ox * ss), m4(oy * ss)),
                        lets.std.Maximum().resize.Bicubic(m4(ox * ss), m4(oy * ss), filter_param_a=1 / 3, filter_param_b=1 / 3),
                    ],
                    expr='x y min',
                ),
                lets.std.Minimum().resize.Bicubic(m4(ox * ss), m4(oy * ss), filter_param_a=1 / 3, filter_param_b=1 / 3),
            ],
            expr='x y max',
        ).resize.Lanczos(ox, oy)
    them = EXPR([clp, remove], expr=f'x y < x x y - {darkstr} * - x x y - {brightstr} * - ?')

    if clp_orig is not None:
        them = core.std.ShufflePlanes([them, clp_orig], planes=[0, 1, 2], colorfamily=clp_orig.format.color_family)
    return them
    
def EdgeCleaner(c: vs.VideoNode, strength: int = 10, rep: bool = True, rmode: int = 17, smode: int = 0, hot: bool = False) -> vs.VideoNode:
    '''
    EdgeCleaner v1.04
    A simple edge cleaning and weak dehaloing function.

    Parameters:
        c: Clip to process.

        strength: Specifies edge denoising strength.

        rep: Activates Repair for the aWarpSharped clip.

        rmode: Specifies the Repair mode.
            1 is very mild and good for halos,
            16 and 18 are good for edge structure preserval on strong settings but keep more halos and edge noise,
            17 is similar to 16 but keeps much less haloing, other modes are not recommended.

        smode: Specifies what method will be used for finding small particles, ie stars. 0 is disabled, 1 uses RemoveGrain.

        hot: Specifies whether removal of hot pixels should take place.
    '''
    if not isinstance(c, vs.VideoNode):
        raise vs.Error('EdgeCleaner: this is not a clip')

    if c.format.color_family == vs.RGB:
        raise vs.Error('EdgeCleaner: RGB format is not supported')

    bits = c.format.bits_per_sample
    peak = (1 << bits) - 1

    if c.format.color_family != vs.GRAY:
        c_orig = c
        c = get_y(c)
    else:
        c_orig = None

    if smode > 0:
        strength += 4

    main = Padding(c, 6, 6, 6, 6).warp.AWarpSharp2(blur=1, depth=cround(strength / 2)).std.Crop(6, 6, 6, 6)
    if rep:
      if hasattr(core, 'zsmooth'):
        main = core.zsmooth.Repair(main, c, mode=rmode)
      else:
        main = core.rgvs.Repair(main, c, mode=rmode)
    EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else (core.akarin.Expr if hasattr(core, 'akarin') else core.std.Expr)
    mask = (
        EXPR(AvsPrewitt(c), expr=f'x {scale_value(4, 8, bits)} < 0 x {scale_value(32, 8, bits)} > {peak} x ? ?')
        .std.InvertMask()
        .std.Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])
    )

    final = core.std.MaskedMerge(c, main, mask)
    if hot:
      if hasattr(core, 'zsmooth'):
        final = core.zsmooth.Repair(final, c, mode=2)
      else:
        final = core.rgvs.Repair(final, c, mode=2)
    if smode > 0:
        RG = core.zsmooth.RemoveGrain if hasattr(core,'zsmooth') else core.rgvs.RemoveGrain
        clean = RG(c, mode=17)
        diff = core.std.MakeDiff(c, clean)
        mask = EXPR(AvsPrewitt(RG(diff.std.Levels(min_in=scale_value(40, 8, bits), max_in=scale_value(168, 8, bits), gamma=0.35), mode=7)),
            expr=f'x {scale_value(4, 8, bits)} < 0 x {scale_value(16, 8, bits)} > {peak} x ? ?'
        )
        final = core.std.MaskedMerge(final, c, mask)

    if c_orig is not None:
        final = core.std.ShufflePlanes([final, c_orig], planes=[0, 1, 2], colorfamily=c_orig.format.color_family)
    return final

def FineDehalo(
    src: vs.VideoNode,
    rx: float = 2.0,
    ry: Optional[float] = None,
    thmi: int = 80,
    thma: int = 128,
    thlimi: int = 50,
    thlima: int = 100,
    darkstr: float = 1.0,
    brightstr: float = 1.0,
    showmask: int = 0,
    contra: float = 0.0,
    excl: bool = True,
    edgeproc: float = 0.0,
    mask: Optional[vs.VideoNode] = None,
) -> vs.VideoNode:
    '''
    Halo removal script that uses DeHalo_alpha with a few masks and optional contra-sharpening to try remove halos without removing important details.

    Parameters:
        src: Clip to process.

        rx, ry: The radii for halo removal in DeHalo_alpha.

        thmi, thma: Minimum and maximum threshold for sharp edges; keep only the sharpest edges (line edges).
            To see the effects of these settings take a look at the strong mask (showmask=4).

        thlimi, thlima: Minimum and maximum limiting threshold; includes more edges than previously, but ignores simple details.

        darkstr, brightstr: The strength factors for processing dark and bright halos in DeHalo_alpha.

        showmask: Shows mask; useful for adjusting settings.
            0 = none
            1 = outside mask
            2 = shrink mask
            3 = edge mask
            4 = strong mask

        contra: Contra-sharpening.

        excl: Activates an additional step (exclusion zones) to make sure that the main edges are really excluded.

        mask: Basic edge mask to apply the threshold instead of applying to the mask created by AvsPrewitt.
    '''
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('FineDehalo: this is not a clip')

    if src.format.color_family == vs.RGB:
        raise vs.Error('FineDehalo: RGB format is not supported')

    if mask is not None:
        if not isinstance(mask, vs.VideoNode):
            raise vs.Error('FineDehalo: mask is not a clip')

        if mask.format.color_family != vs.GRAY:
            raise vs.Error('FineDehalo: mask must be Gray format')

    is_float = src.format.sample_type == vs.FLOAT

    bits = src.format.bits_per_sample

    if src.format.color_family != vs.GRAY:
        src_orig = src
        src = get_y(src)
    else:
        src_orig = None

    ry = fallback(ry, rx)

    rx_i = cround(rx)
    ry_i = cround(ry)

    # Dehaloing #

    dehaloed = DeHalo_alpha(src, rx=rx, ry=ry, darkstr=darkstr, brightstr=brightstr)

    # Contrasharpening
    if contra > 0:
        dehaloed = FineDehalo_contrasharp(dehaloed, src, contra)

    # Main edges #

    # Basic edge detection, thresholding will be applied later
    edges = fallback(mask, AvsPrewitt(src))

    vszip = hasattr(core,'vszip')
    # Keeps only the sharpest edges (line edges)
    EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else (core.akarin.Expr if hasattr(core, 'akarin') else core.std.Expr)
    strong = EXPR(edges, expr=f'x {scale_value(thmi, 8, bits)} - {thma - thmi} / 255 *')
    if is_float:
        strong = strong.vszip.Limiter() if vszip else strong.std.Limiter()

    # Extends them to include the potential halos
    large = mt_expand_multi(strong, sw=rx_i, sh=ry_i)

    # Exclusion zones #

    # When two edges are close from each other (both edges of a single line or multiple parallel color bands),
    # the halo removal oversmoothes them or makes seriously bleed the bands, producing annoying artifacts.
    # Therefore we have to produce a mask to exclude these zones from the halo removal.

    # Includes more edges than previously, but ignores simple details
    light = EXPR(edges, expr=f'x {scale_value(thlimi, 8, bits)} - {thlima - thlimi} / 255 *')
    if is_float:
        light = light.vszip.Limiter() if vszip else light.std.Limiter()

    # To build the exclusion zone, we make grow the edge mask, then shrink it to its original shape.
    # During the growing stage, close adjacent edge masks will join and merge, forming a solid area, which will remain solid even after the shrinking stage.

    # Mask growing
    shrink = mt_expand_multi(light, mode='ellipse', sw=rx_i, sh=ry_i)

    # At this point, because the mask was made of a shades of grey, we may end up with large areas of dark grey after shrinking.
    # To avoid this, we amplify and saturate the mask here (actually we could even binarize it).
    shrink = EXPR(shrink, expr='x 4 *')
    if is_float:
        shrink = shrink.vszip.Limiter() if vszip else shrink.std.Limiter()

    # Mask shrinking
    shrink = mt_inpand_multi(shrink, mode='ellipse', sw=rx_i, sh=ry_i)

    # This mask is almost binary, which will produce distinct discontinuities once applied. Then we have to smooth it.
    shrink = shrink.std.Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1]).std.Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])

    # Final mask building #

    # Previous mask may be a bit weak on the pure edge side, so we ensure that the main edges are really excluded.
    # We do not want them to be smoothed by the halo removal.
    if excl:
        shr_med = EXPR([strong, shrink], expr='x y max')
    else:
        shr_med = strong

    # Subtracts masks and amplifies the difference to be sure we get 255 on the areas to be processed
    outside = EXPR([large, shr_med], expr='x y - 2 *')
    if is_float:
        outside = outside.vszip.Limiter() if vszip else outside.std.Limiter()

    # If edge processing is required, adds the edgemask
    if edgeproc > 0:
        outside = EXPR([outside, strong], expr=f'x y {edgeproc * 0.66} * +')
        if is_float:
            outside = outside.vszip.Limiter() if vszip else outside.std.Limiter()

    # Smooth again and amplify to grow the mask a bit, otherwise the halo parts sticking to the edges could be missed
    outside = EXPR(outside.std.Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1]), expr='x 2 *')
    if is_float:
        outside = outside.vszip.Limiter() if vszip else outside.std.Limiter()

    # Masking #

    if showmask <= 0:
        last = core.std.MaskedMerge(src, dehaloed, outside)

    if src_orig is not None:
        if showmask <= 0:
            return core.std.ShufflePlanes([last, src_orig], planes=[0, 1, 2], colorfamily=src_orig.format.color_family)
        elif showmask == 1:
            return outside.resize.Bicubic(format=src_orig.format)
        elif showmask == 2:
            return shrink.resize.Bicubic(format=src_orig.format)
        elif showmask == 3:
            return edges.resize.Bicubic(format=src_orig.format)
        else:
            return strong.resize.Bicubic(format=src_orig.format)
    else:
        if showmask <= 0:
            return last
        elif showmask == 1:
            return outside
        elif showmask == 2:
            return shrink
        elif showmask == 3:
            return edges
        else:
            return strong

def FineDehalo_contrasharp(dehaloed: vs.VideoNode, src: vs.VideoNode, level: float) -> vs.VideoNode:
    '''level == 1.0 : normal contrasharp'''
    if not (isinstance(dehaloed, vs.VideoNode) and isinstance(src, vs.VideoNode)):
        raise vs.Error('FineDehalo_contrasharp: this is not a clip')

    if dehaloed.format.color_family == vs.RGB:
        raise vs.Error('FineDehalo_contrasharp: RGB format is not supported')

    if dehaloed.format.id != src.format.id:
        raise vs.Error('FineDehalo_contrasharp: clips must have the same format')

    neutral = 1 << (dehaloed.format.bits_per_sample - 1) if dehaloed.format.sample_type == vs.INTEGER else 0.0

    if dehaloed.format.color_family != vs.GRAY:
        dehaloed_orig = dehaloed
        dehaloed = get_y(dehaloed)
        src = get_y(src)
    else:
        dehaloed_orig = None

    bb = dehaloed.std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    if hasattr(core, 'zsmooth'):
      bb2 = core.zsmooth.Repair(bb, core.zsmooth.Repair(bb, bb.ctmf.CTMF(radius=2), mode=1), mode=1)
    else:
      bb2 = core.rgvs.Repair(bb, core.rgvs.Repair(bb, bb.ctmf.CTMF(radius=2), mode=1), mode=1)
    xd = core.std.MakeDiff(bb, bb2)
    EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else (core.akarin.Expr if hasattr(core, 'akarin') else core.std.Expr)
    xd = EXPR(xd, expr=f'x {neutral} - 2.49 * {level} * {neutral} +')
    xdd = EXPR(
        [xd, core.std.MakeDiff(src, dehaloed)], expr=f'x {neutral} - y {neutral} - * 0 < {neutral} x {neutral} - abs y {neutral} - abs < x y ? ?'
    )
    last = core.std.MergeDiff(dehaloed, xdd)

    if dehaloed_orig is not None:
        last = core.std.ShufflePlanes([last, dehaloed_orig], planes=[0, 1, 2], colorfamily=dehaloed_orig.format.color_family)
    return last

def YAHR(clp: vs.VideoNode, blur: int = 2, depth: int = 32) -> vs.VideoNode:
    '''
    Y'et A'nother H'alo R'educing script

    Parameters:
        clp: Clip to process.

        blur: "blur" parameter of AWarpSharp2.

        depth: "depth" parameter of AWarpSharp2.
    '''
    if not isinstance(clp, vs.VideoNode):
        raise vs.Error('YAHR: this is not a clip')

    if clp.format.color_family == vs.RGB:
        raise vs.Error('YAHR: RGB format is not supported')

    if clp.format.color_family != vs.GRAY:
        clp_orig = clp
        clp = get_y(clp)
    else:
        clp_orig = None

    b1 = MinBlur(clp, 2).std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    b1D = core.std.MakeDiff(clp, b1)
    w1 = Padding(clp, 6, 6, 6, 6).warp.AWarpSharp2(blur=blur, depth=depth).std.Crop(6, 6, 6, 6)
    w1b1 = MinBlur(w1, 2).std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    w1b1D = core.std.MakeDiff(w1, w1b1)
    if hasattr(core, 'zsmooth'):
      DD = core.zsmooth.Repair(b1D, w1b1D, mode=13)
    else:
      DD = core.rgvs.Repair(b1D, w1b1D, mode=13)
    DD2 = core.std.MakeDiff(b1D, DD)
    last = core.std.MakeDiff(clp, DD2)

    if clp_orig is not None:
        last = core.std.ShufflePlanes([last, clp_orig], planes=[0, 1, 2], colorfamily=clp_orig.format.color_family)
    return last

def AvsPrewitt(clip: vs.VideoNode, planes: Optional[Union[int, Sequence[int]]] = None) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('AvsPrewitt: this is not a clip')

    plane_range = range(clip.format.num_planes)

    if planes is None:
        planes = list(plane_range)
    elif isinstance(planes, int):
        planes = [planes]
    EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else (core.akarin.Expr if hasattr(core, 'akarin') else core.std.Expr)
    return EXPR(
        [
            clip.std.Convolution(matrix=[1, 1, 0, 1, 0, -1, 0, -1, -1], planes=planes, saturate=False),
            clip.std.Convolution(matrix=[1, 1, 1, 0, 0, 0, -1, -1, -1], planes=planes, saturate=False),
            clip.std.Convolution(matrix=[1, 0, -1, 1, 0, -1, 1, 0, -1], planes=planes, saturate=False),
            clip.std.Convolution(matrix=[0, -1, -1, 1, 0, -1, 1, 1, 0], planes=planes, saturate=False),
        ],
        expr=['x y max z max a max' if i in planes else '' for i in plane_range],
    )

def Padding(clip: vs.VideoNode, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('Padding: this is not a clip')

    if left < 0 or right < 0 or top < 0 or bottom < 0:
        raise vs.Error('Padding: border size to pad must not be negative')

    width = clip.width + left + right
    height = clip.height + top + bottom

    return clip.resize.Point(width, height, src_left=-left, src_top=-top, src_width=width, src_height=height)
    
def mt_expand_multi(src: vs.VideoNode, mode: str = 'rectangle', planes: Optional[Union[int, Sequence[int]]] = None, sw: int = 1, sh: int = 1) -> vs.VideoNode:
    '''
    Calls std.Maximum multiple times in order to grow the mask from the desired width and height.

    Parameters:
        src: Clip to process.

        mode: "rectangle", "ellipse" or "losange". Ellipses are actually combinations of rectangles and losanges and look more like octogons.
            Losanges are truncated (not scaled) when sw and sh are not equal.

        planes: Specifies which planes will be processed. Any unprocessed planes will be simply copied.

        sw: Growing shape width. 0 is allowed.

        sh: Growing shape height. 0 is allowed.
    '''
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('mt_expand_multi: this is not a clip')

    if sw > 0 and sh > 0:
        mode_m = [0, 1, 0, 1, 1, 0, 1, 0] if mode == 'losange' or (mode == 'ellipse' and (sw % 3) != 1) else [1, 1, 1, 1, 1, 1, 1, 1]
    elif sw > 0:
        mode_m = [0, 0, 0, 1, 1, 0, 0, 0]
    elif sh > 0:
        mode_m = [0, 1, 0, 0, 0, 0, 1, 0]
    else:
        mode_m = None

    if mode_m is not None:
        src = mt_expand_multi(src.std.Maximum(planes=planes, coordinates=mode_m), mode=mode, planes=planes, sw=sw - 1, sh=sh - 1)
    return src


def mt_inpand_multi(src: vs.VideoNode, mode: str = 'rectangle', planes: Optional[Union[int, Sequence[int]]] = None, sw: int = 1, sh: int = 1) -> vs.VideoNode:
    '''
    Calls std.Minimum multiple times in order to shrink the mask from the desired width and height.

    Parameters:
        src: Clip to process.

        mode: "rectangle", "ellipse" or "losange". Ellipses are actually combinations of rectangles and losanges and look more like octogons.
            Losanges are truncated (not scaled) when sw and sh are not equal.

        planes: Specifies which planes will be processed. Any unprocessed planes will be simply copied.

        sw: Shrinking shape width. 0 is allowed.

        sh: Shrinking shape height. 0 is allowed.
    '''
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('mt_inpand_multi: this is not a clip')

    if sw > 0 and sh > 0:
        mode_m = [0, 1, 0, 1, 1, 0, 1, 0] if mode == 'losange' or (mode == 'ellipse' and (sw % 3) != 1) else [1, 1, 1, 1, 1, 1, 1, 1]
    elif sw > 0:
        mode_m = [0, 0, 0, 1, 1, 0, 0, 0]
    elif sh > 0:
        mode_m = [0, 1, 0, 0, 0, 0, 1, 0]
    else:
        mode_m = None

    if mode_m is not None:
        src = mt_inpand_multi(src.std.Minimum(planes=planes, coordinates=mode_m), mode=mode, planes=planes, sw=sw - 1, sh=sh - 1)
    return src

# MinBlur   by Didée (http://avisynth.nl/index.php/MinBlur)
# Nifty Gauss/Median combination
def MinBlur(clp: vs.VideoNode, r: int=1, planes: Optional[Union[int, Sequence[int]]] = None) -> vs.VideoNode:
    if not isinstance(clp, vs.VideoNode):
        raise vs.Error('MinBlur: This is not a clip')

    if planes is None:
        planes = list(range(clp.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    matrix1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    matrix2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    has_zsmooth = hasattr(core,'zsmooth')
    if r <= 0:
        RG11 = sbr(clp, planes=planes)
        RG4 = clp.zsmooth.Median(planes=planes) if has_zsmooth else clp.std.Median(planes=planes)
    elif r == 1:
        RG11 = clp.std.Convolution(matrix=matrix1, planes=planes)
        RG4 = clp.zsmooth.Median(planes=planes) if has_zsmooth else clp.std.Median(planes=planes)
    elif r == 2:
        RG11 = clp.std.Convolution(matrix=matrix1, planes=planes).std.Convolution(matrix=matrix2, planes=planes)
        RG4 = clp.ctmf.CTMF(radius=2, planes=planes)
    else:
        RG11 = clp.std.Convolution(matrix=matrix1, planes=planes).std.Convolution(matrix=matrix2, planes=planes).std.Convolution(matrix=matrix2, planes=planes)
        if clp.format.bits_per_sample == 16:
            s16 = clp
            RG4 = clp.fmtc.bitdepth(bits=12, planes=planes, dmode=1).ctmf.CTMF(radius=3, planes=planes).fmtc.bitdepth(bits=16, planes=planes)
            RG4 = LimitFilter(s16, RG4, thr=0.0625, elast=2, planes=planes)
        else:
            RG4 = clp.ctmf.CTMF(radius=3, planes=planes, opt=2)

    expr = 'x y - x z - * 0 < x x y - abs x z - abs < y z ? ?'
    EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else (core.akarin.Expr if hasattr(core, 'akarin') else core.std.Expr)
    return EXPR([clp, RG11, RG4], expr=[expr if i in planes else '' for i in range(clp.format.num_planes)])

# Try to remove 2nd order halos
# Added presets to classic FineDehalo2
def SecondOrderDehalo(
    src: vs.VideoNode,
    preset: str = 'default',
    hconv: list[int] | None = None,
    vconv: list[int] | None = None,
    edgemask: str | None = None,
    growmask: str | None = None,
    showmask: int = 0
) -> vs.VideoNode:
    """
    SecondOrderDehalo - Second-order dehalo removal using directional edge masks and convolution.

    Parameters:
        src (vs.VideoNode):
            Input clip. Must be GRAY or YUV (not RGB).
        preset (str):
            Preset for typical content:
                - 'default'
                - 'strong'
                - 'light'
                - 'anime'
                - 'anime-strong'
                - 'liveaction'
                - 'liveaction-strong'
        hconv (list[int]):
            Optional horizontal convolution kernel override.
        vconv (list[int]):
            Optional vertical convolution kernel override.
        edgemask (str):
            Edge detection method: 'sobel' or 'prewitt'. Defaults based on preset.
        growmask (str):
            Grow mask type: 'default', 'strong', 'tight', or 'fast'. Controls edge expansion aggressiveness.
        showmask (int):
            If 1, returns combined mask for previewing. If 0, returns processed clip.

    Returns:
        vs.VideoNode: Dehaloed clip or visualized mask depending on showmask.
    """

    if not isinstance(src, vs.VideoNode):
        raise TypeError('FineDehalo2: This is not a clip')
    if src.format.color_family == vs.RGB:
        raise vs.Error('FineDehalo2: RGB format is not supported')

    # Extract luma if needed
    if src.format.color_family != vs.GRAY:
        src_orig = src
        src = core.std.ShufflePlanes(src, 0, vs.GRAY)
    else:
        src_orig = None

    # Preset parameters
    presets = {
        'default': {
            'hconv': [-1, -2, 0, 0, 40, 0, 0, -2, -1],
            'vconv': [-2, -1, 0, 0, 40, 0, 0, -1, -2],
            'edgemask': 'sobel',
            'growmask': 'default'
        },
        'strong': {
            'hconv': [-2, -3, 0, 0, 50, 0, 0, -3, -2],
            'vconv': [-3, -2, 0, 0, 50, 0, 0, -2, -3],
            'edgemask': 'prewitt',
            'growmask': 'strong'
        },
        'light': {
            'hconv': [-1, 0, 0, 0, 30, 0, 0, 0, -1],
            'vconv': [0, -1, 0, 0, 30, 0, 0, -1, 0],
            'edgemask': 'sobel',
            'growmask': 'fast'
        },
        'anime': {
            'hconv': [-1, -1, 0, 0, 35, 0, 0, -1, -1],
            'vconv': [-1, -1, 0, 0, 35, 0, 0, -1, -1],
            'edgemask': 'sobel',
            'growmask': 'tight'
        },
        'anime-strong': {
            'hconv': [-2, -2, 0, 0, 45, 0, 0, -2, -2],
            'vconv': [-2, -2, 0, 0, 45, 0, 0, -2, -2],
            'edgemask': 'prewitt',
            'growmask': 'strong'
        },
        'liveaction': {
            'hconv': [-1, -2, 0, 0, 30, 0, 0, -2, -1],
            'vconv': [-2, -1, 0, 0, 30, 0, 0, -1, -2],
            'edgemask': 'sobel',
            'growmask': 'default'
        },
        'liveaction-strong': {
            'hconv': [-2, -3, 0, 0, 40, 0, 0, -3, -2],
            'vconv': [-3, -2, 0, 0, 40, 0, 0, -2, -3],
            'edgemask': 'prewitt',
            'growmask': 'strong'
        }
    }

    # Load preset
    preset_vals = presets.get(preset)
    if not preset_vals:
        raise vs.Error(f'FineDehalo2: Unknown preset "{preset}"')

    hconv = hconv or preset_vals['hconv']
    vconv = vconv or preset_vals['vconv']
    edgemask = edgemask or preset_vals['edgemask']
    growmask = growmask or preset_vals['growmask']

    # --- Edge mask generation ---
    if edgemask == 'sobel':
        mask_h = core.std.Convolution(src, matrix=[1, 2, 1, 0, 0, 0, -1, -2, -1], divisor=4, saturate=False)
        mask_v = core.std.Convolution(src, matrix=[1, 0, -1, 2, 0, -2, 1, 0, -1], divisor=4, saturate=False)
    elif edgemask == 'prewitt':
        mask_h = core.std.Convolution(src, matrix=[1, 1, 1, 0, 0, 0, -1, -1, -1], divisor=3, saturate=False)
        mask_v = core.std.Convolution(src, matrix=[1, 0, -1, 1, 0, -1, 1, 0, -1], divisor=3, saturate=False)
    else:
        raise vs.Error(f'FineDehalo2: Unknown edgemask type "{edgemask}"')

    EXPR = core.akarin.Expr if hasattr(core, 'akarin') else core.std.Expr
    temp_h = EXPR([mask_h, mask_v], ['x 3 * y -'])
    temp_v = EXPR([mask_v, mask_h], ['x 3 * y -'])

    # --- Mask grower ---
    def grow(mask: vs.VideoNode, kind: str, coordinates: list[int]) -> vs.VideoNode:
        mask = core.std.Maximum(mask, coordinates=coordinates).std.Minimum(coordinates=coordinates)
        mask_1 = core.std.Maximum(mask, coordinates=coordinates)
        mask_2 = core.std.Maximum(mask_1, coordinates=coordinates).std.Maximum(coordinates=coordinates)
        diff = EXPR([mask_2, mask_1], ['x y -'])

        if kind == 'tight':
            conv = core.std.Convolution(diff, matrix=[1] * 9)
            return EXPR([conv], ['x 1.2 *'])
        elif kind == 'fast':
            conv = core.std.Convolution(diff, matrix=[0, 1, 0, 1, 4, 1, 0, 1, 0])
            return EXPR([conv], ['x 1.5 *'])
        elif kind == 'strong':
            conv = core.std.Convolution(diff, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
            return EXPR([conv], ['x 2.0 *'])
        else:
            conv = core.std.Convolution(diff, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
            return EXPR([conv], ['x 1.8 *'])

    mask_h = grow(temp_h, growmask, [0, 1, 0, 0, 0, 0, 1, 0])
    mask_v = grow(temp_v, growmask, [0, 0, 0, 1, 1, 0, 0, 0])

    # --- Apply dehalo ---
    fix_h = core.std.Convolution(src, matrix=vconv, mode='v')
    fix_v = core.std.Convolution(src, matrix=hconv, mode='h')

    if showmask:
        last = EXPR([mask_h, mask_v], ['x y max'])
    else:
        last = core.std.MaskedMerge(src, fix_h, mask_h)
        last = core.std.MaskedMerge(last, fix_v, mask_v)

    # Restore chroma
    if src_orig is not None:
        if showmask:
            return core.resize.Bicubic(last, format=src_orig.format.id)
        return core.std.ShufflePlanes([last, src_orig], planes=[0, 1, 2], colorfamily=src_orig.format.color_family)
    return last

def BlindDeHalo3(clp: vs.VideoNode, rx: float = 3.0, ry: float = 3.0, strength: float = 125,
                 lodamp: float = 0, hidamp: float = 0, sharpness: float = 0, tweaker: float = 0,
                 PPmode: int = 0, PPlimit: Optional[int] = None, interlaced: bool = False
                 ) -> vs.VideoNode:
    """Avisynth's BlindDeHalo3() version: 3_MT2

    This script removes the light & dark halos from too strong "Edge Enhancement".

    Author: Didée (https://forum.doom9.org/attachment.php?attachmentid=5599&d=1143030001)

    Only the first plane (luma) will be processed.

    Args:
        clp: Input clip.

        rx, ry: (float) The radii to use for the [quasi-] Gaussian blur, on which the halo removal is based. Default is 3.0.

        strength: (float) The overall strength of the halo removal effect. Default is 125.

        lodamp, hidamp: (float) With these two values, one can reduce the basic effect on areas that would change only little anyway (lodamp),
            and/or on areas that would change very much (hidamp).
            lodamp does a reasonable job in keeping more detail in affected areas.
            hidamp is intended to keep rather small areas that are very bright or very dark from getting processed too strong.
            Works OK on sources that contain only weak haloing - for sources with strong over sharpening,
                it should not be used, mostly. (Usage has zero impact on speed.)
            Range: 0.0 to ??? (try 4.0 as a start)
            Default is 0.0.

        sharpness: (float) By setting this bigger than 0.0, the affected areas will come out with better sharpness.
            However, strength must be chosen somewhat bigger as well, then, to get the same effect than without.
            (This is the same as initial version's "maskblur" option.)
            Range: 0.0 to 1.58.
            Default is 0.

        tweaker: (float) May be used to get a stronger effect, separately from altering "strength".
            (Also in accordance to initial version's working methodology. I had no better idea for naming this parameter.)
            Range: 0.0 - 1.00.
            Default is 0.

        PPmode: (int) When set to "1" or "2", a second cleaning operation after the basic halo removal is done.
            This deals with:
                a) Removing/reducing those corona lines that sometimes are left over by BlindDeHalo
                b) Improving on mosquito noise, if some is present.
            PPmode=1 uses a simple Gaussian blur for post-cleaning. PPmode=2 uses a 3*3 average, with zero weighting of the center pixel.
            Also, PPmode can be "-1" or "-2". In this case, the main dehaloing step is completely discarded, and *only* the PP cleaning is done.
            This has less effect on halos, but can deal for sources containing more mosquito noise than halos.
            Default is 0.

        PPlimit: (int) Can be used to make the PP routine change no pixel by more than [PPlimit].
            I'm not sure if this makes much sense in this context. However the option is there - you never know what it might be good for.
            Default is 0.

        interlaced: (bool) As formerly, this is intended for sources that were originally interlaced, but then made progressive by deinterlacing.
            It aims in particular at clips that made their way through Restore24.
            Default is False.

    """

    funcName = 'BlindDeHalo3'

    if not isinstance(clp, vs.VideoNode):
        raise TypeError(funcName + ': \"clp\" is not a clip!')

    if clp.format.sample_type != vs.INTEGER:
        raise TypeError(funcName + ': Only integer clip is supported!')

    if PPlimit is None:
        PPlimit = 4 if abs(PPmode) == 3 else 0

    bits = clp.format.bits_per_sample
    isGray = clp.format.color_family == vs.GRAY
    neutral = 1 << (bits - 1)

    if not isGray:
        clp_src = clp
        clp = GetPlane(clp)

    sharpness = min(sharpness, 1.58)
    tweaker = min(tweaker, 1.0)
    strength *= 1 + sharpness * 0.25
    RR = (rx + ry) / 2
    ST = strength / 100
    LD = scale_value(lodamp, 8, bits)
    HD = hidamp ** 2
    TWK0 = 'x y - {i} /'.format(i=12 / ST / RR)
    TWK = 'x y - {i} / abs'.format(i=12 / ST / RR)
    TWK_HLIGHT = ('x y - abs {i} < {neutral} {TWK} {neutral} {TWK} - {TWK} {neutral} / * + {TWK0} {TWK} {LD} + / * '
        '{neutral} {TWK} - {j} / dup * {neutral} {TWK} - {j} / dup * {HD} + / * {neutral} + ?'.format(
            i=1 << (bits-8), neutral=neutral, TWK=TWK, TWK0=TWK0, LD=LD, j=scale_value(20, 8, bits), HD=HD))

    i = clp if not interlaced else core.std.SeparateFields(clp, tff=True)
    oxi = i.width
    oyi = i.height
    sm = core.resize.Bicubic(i, m4(oxi/rx), m4(oyi/ry), filter_param_a=1/3, filter_param_b=1/3)
    mm = core.std.Expr([sm.std.Maximum(), sm.std.Minimum()], ['x y - 4 *']).std.Maximum().std.Deflate().std.Convolution([1]*9)
    mm = mm.std.Inflate().resize.Bicubic(oxi, oyi, filter_param_a=1, filter_param_b=0).std.Inflate()
    sm = core.resize.Bicubic(sm, oxi, oyi, filter_param_a=1, filter_param_b=0)
    smd = core.std.Expr([Sharpen(i, tweaker), sm], [TWK_HLIGHT])
    if sharpness != 0.:
        smd = Blur(smd, sharpness)
    clean = core.std.Expr([i, smd], ['x y {neutral} - -'.format(neutral=neutral)])
    clean = core.std.MaskedMerge(i, clean, mm)

    if PPmode != 0:
        LL = scale_value(PPlimit, 8, bits)
        LIM = 'x {LL} + y < x {LL} + x {LL} - y > x {LL} - y ? ?'.format(LL=LL)

        base = i if PPmode < 0 else clean
        small = core.resize.Bicubic(base, m4(oxi / math.sqrt(rx * 1.5)), m4(oyi / math.sqrt(ry * 1.5)), filter_param_a=1/3, filter_param_b=1/3)
        ex1 = Blur(small.std.Maximum(), 0.5)
        in1 = Blur(small.std.Minimum(), 0.5)
        hull = core.std.Expr([ex1.std.Maximum().std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1]), ex1, in1,
            in1.std.Minimum().std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])],
            ['x y - {i} - 5 * z a - {i} - 5 * max'.format(i=1 << (bits-8))]).resize.Bicubic(oxi, oyi, filter_param_a=1, filter_param_b=0)

        if abs(PPmode) == 1:
            postclean = core.std.MaskedMerge(base, small.resize.Bicubic(oxi, oyi, filter_param_a=1, filter_param_b=0), hull)
        elif abs(PPmode) == 2:
            postclean = core.std.MaskedMerge(base, base.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1]), hull)
        elif abs(PPmode) == 3:
            if hasattr(core,'zsmooth'):
              postclean = core.std.MaskedMerge(base, base.zsmooth.Median(), hull)
            else:
              postclean = core.std.MaskedMerge(base, base.std.Median(), hull)
        else:
            raise ValueError(funcName + ': \"PPmode\" must be in [-3 ... 3]!')
    else:
        postclean = clean

    if PPlimit != 0:
        postclean = core.std.Expr([base, postclean], [LIM])

    last = haf_Weave(postclean, tff=True) if interlaced else postclean

    if not isGray:
        last = core.std.ShufflePlanes([last, clp_src], list(range(clp_src.format.num_planes)), clp_src.format.color_family)

    return last
    
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
    
def cround(x: float) -> int:
    return math.floor(x + 0.5) if x > 0 else math.ceil(x - 0.5)

def m4(value, mult=4.0):
    return 16 if value < 16 else int(round(value / mult) * mult)
    
def Blur(clip: vs.VideoNode, amountH: float = 1.0, amountV: Optional[float] = None,
         planes: Optional[Union[int, Sequence[int]]] = None
         ) -> vs.VideoNode:
    """Avisynth's internel filter Blur()

    Simple 3x3-kernel blurring filter.

    In fact Blur(n) is just an alias for Sharpen(-n).

    Args:
        clip: Input clip.

        amountH, amountV: (float) Blur uses the kernel is [(1-1/2^amount)/2, 1/2^amount, (1-1/2^amount)/2].
            A value of 1.0 gets you a (1/4, 1/2, 1/4) for example.
            Negative Blur actually sharpens the image.
            The allowable range for Blur is from -1.0 to +1.58.
            If \"amountV\" is not set manually, it will be set to \"amountH\".
            Default is 1.0.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from the source clip, "clip".

    """

    funcName = 'Blur'

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" is not a clip!')

    if amountH < -1 or amountH > 1.5849625:
        raise ValueError(funcName + ': \'amountH\' have not a correct value! [-1 ~ 1.58]')

    if amountV is None:
        amountV = amountH
    else:
        if amountV < -1 or amountV > 1.5849625:
            raise ValueError(funcName + ': \'amountV\' have not a correct value! [-1 ~ 1.58]')

    return Sharpen(clip, -amountH, -amountV, planes)
    
def Sharpen(clip: vs.VideoNode, amountH: float = 1.0, amountV: Optional[float] = None,
            planes: Optional[Union[int, Sequence[int]]] = None
            ) -> vs.VideoNode:
    """Avisynth's internel filter Sharpen()

    Simple 3x3-kernel sharpening filter.

    Args:
        clip: Input clip.

        amountH, amountV: (float) Sharpen uses the kernel is [(1-2^amount)/2, 2^amount, (1-2^amount)/2].
            A value of 1.0 gets you a (-1/2, 2, -1/2) for example.
            Negative Sharpen actually blurs the image.
            The allowable range for Sharpen is from -1.58 to +1.0.
            If \"amountV\" is not set manually, it will be set to \"amountH\".
            Default is 1.0.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from the source clip, "clip".

    """

    funcName = 'Sharpen'

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" is not a clip!')

    if amountH < -1.5849625 or amountH > 1:
        raise ValueError(funcName + ': \'amountH\' have not a correct value! [-1.58 ~ 1]')

    if amountV is None:
        amountV = amountH
    else:
        if amountV < -1.5849625 or amountV > 1:
            raise ValueError(funcName + ': \'amountV\' have not a correct value! [-1.58 ~ 1]')

    if planes is None:
        planes = list(range(clip.format.num_planes))

    center_weight_v = math.floor(2 ** (amountV - 1) * 1023 + 0.5)
    outer_weight_v = math.floor((0.25 - 2 ** (amountV - 2)) * 1023 + 0.5)
    center_weight_h = math.floor(2 ** (amountH - 1) * 1023 + 0.5)
    outer_weight_h = math.floor((0.25 - 2 ** (amountH - 2)) * 1023 + 0.5)

    conv_mat_v = [outer_weight_v, center_weight_v, outer_weight_v]
    conv_mat_h = [outer_weight_h, center_weight_h, outer_weight_h]

    if math.fabs(amountH) >= 0.00002201361136: # log2(1+1/65536)
        clip = core.std.Convolution(clip, conv_mat_v, planes=planes, mode='v')

    if math.fabs(amountV) >= 0.00002201361136:
        clip = core.std.Convolution(clip, conv_mat_h, planes=planes, mode='h')

    return clip
        
# Taken from sfrom vsutil
T = TypeVar('T')
def fallback(value: Optional[T], fallback_value: T) -> T:
    """Utility function that returns a value or a fallback if the value is ``None``.

    >>> fallback(5, 6)
    5
    >>> fallback(None, 6)
    6

    :param value:           Argument that can be ``None``.
    :param fallback_value:  Fallback value that is returned if `value` is ``None``.

    :return:                The input `value` or `fallback_value` if `value` is ``None``.
    """
    return fallback_value if value is None else value


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
