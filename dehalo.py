import vapoursynth as vs
from vapoursynth import core

import math

from vsutil import get_depth, get_y, scale_value, fallback

from typing import Union, Optional, Sequence

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

    bits = get_depth(clp)

    if clp.format.color_family != vs.GRAY:
        clp_orig = clp
        clp = get_y(clp)
    else:
        clp_orig = None

    ox = clp.width
    oy = clp.height
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
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

    bits = get_depth(c)
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
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
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
        mask = EXPR(AvsPrewitt(diff.std.Levels(min_in=scale_value(40, 8, bits), max_in=scale_value(168, 8, bits), gamma=0.35).RG(mode=7)),
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

    bits = get_depth(src)

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
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
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

    neutral = 1 << (get_depth(dehaloed) - 1) if dehaloed.format.sample_type == vs.INTEGER else 0.0

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
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
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


def cround(x: float) -> int:
    return math.floor(x + 0.5) if x > 0 else math.ceil(x - 0.5)

def m4(x: Union[float, int]) -> int:
    return 16 if x < 16 else cround(x / 4) * 4

def AvsPrewitt(clip: vs.VideoNode, planes: Optional[Union[int, Sequence[int]]] = None) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('AvsPrewitt: this is not a clip')

    plane_range = range(clip.format.num_planes)

    if planes is None:
        planes = list(plane_range)
    elif isinstance(planes, int):
        planes = [planes]
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
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

def MinBlur(clp: vs.VideoNode, r: int = 1, planes: Optional[Union[int, Sequence[int]]] = None) -> vs.VideoNode:
    '''Nifty Gauss/Median combination'''
    from mvsfunc import LimitFilter

    if not isinstance(clp, vs.VideoNode):
        raise vs.Error('MinBlur: this is not a clip')

    plane_range = range(clp.format.num_planes)

    if planes is None:
        planes = list(plane_range)
    elif isinstance(planes, int):
        planes = [planes]

    matrix1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    matrix2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    if r <= 0:
        RG11 = sbr(clp, planes=planes)
        RG4 = clp.std.Median(planes=planes)
    elif r == 1:
        RG11 = clp.std.Convolution(matrix=matrix1, planes=planes)
        RG4 = clp.std.Median(planes=planes)
    elif r == 2:
        RG11 = clp.std.Convolution(matrix=matrix1, planes=planes).std.Convolution(matrix=matrix2, planes=planes)
        RG4 = clp.ctmf.CTMF(radius=2, planes=planes)
    else:
        RG11 = clp.std.Convolution(matrix=matrix1, planes=planes).std.Convolution(matrix=matrix2, planes=planes).std.Convolution(matrix=matrix2, planes=planes)
        if get_depth(clp) == 16:
            s16 = clp
            RG4 = depth(clp, 12, dither_type=Dither.NONE).ctmf.CTMF(radius=3, planes=planes)
            RG4 = LimitFilter(s16, depth(RG4, 16), thr=0.0625, elast=2, planes=planes)
        else:
            RG4 = clp.ctmf.CTMF(radius=3, planes=planes)
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    return EXPR([clp, RG11, RG4], expr=['x y - x z - * 0 < x x y - abs x z - abs < y z ? ?' if i in planes else '' for i in plane_range])
