
import vapoursynth as vs
from vapoursynth import core

import math
from typing import Optional, Union, Sequence, TypeVar

def HQDeringmod(
    input: vs.VideoNode,
    smoothed: Optional[vs.VideoNode] = None,
    ringmask: Optional[vs.VideoNode] = None,
    mrad: int = 1,
    msmooth: int = 1,
    incedge: bool = False,
    mthr: int = 60,
    minp: int = 1,
    nrmode: Optional[int] = None,
    sigma: float = 128.0,
    sigma2: Optional[float] = None,
    sbsize: Optional[int] = None,
    sosize: Optional[int] = None,
    sharp: int = 1,
    drrep: Optional[int] = None,
    thr: float = 12.0,
    elast: float = 2.0,
    darkthr: Optional[float] = None,
    planes: Union[int, Sequence[int]] = 0,
    show: bool = False,
    cuda: bool = False,                   
) -> vs.VideoNode:
    '''
    HQDering mod v1.8
    Applies deringing by using a smart smoother near edges (where ringing occurs) only.

    Parameters:
        input: Clip to process.

        mrad: Expanding of edge mask, higher value means more aggressive processing.

        msmooth: Inflate of edge mask, smooth boundaries of mask.

        incedge: Whether to include edge in ring mask, by default ring mask only include area near edges.

        mthr: Threshold of prewitt edge mask, lower value means more aggressive processing.
            But for strong ringing, lower value will treat some ringing as edge, which protects this ringing from being processed.

        minp: Inpanding of prewitt edge mask, higher value means more aggressive processing.

        nrmode: Kernel of deringing.
            0 = DFTTest
            1 = MinBlur(r=1)
            2 = MinBlur(r=2)
            3 = MinBlur(r=3)

        sigma: Sigma for medium frequecies in DFTTest.

        sigma2: Sigma for low & high frequecies in DFTTest.

        sbsize: Length of the sides of the spatial window in DFTTest.

        sosize: Spatial overlap amount in DFTTest.

        sharp: Whether to use contra-sharpening to resharp deringed clip, 1-3 represents radius, 0 means no sharpening.

        drrep: Use repair for details retention, recommended values are 24/23/13/12/1.

        thr: The same meaning with "thr" in LimitFilter.

        elast: The same meaning with "elast" in LimitFilter.

        darkthr: Threshold for darker area near edges, by default equals to thr/4. Set it lower if you think de-ringing destroys too much lines, etc.
            When "darkthr" is not equal to "thr", "thr" limits darkening while "darkthr" limits brightening.

        planes: Specifies which planes will be processed. Any unprocessed planes will be simply copied.

        show: Whether to output mask clip instead of filtered clip.
        cuda: Whether to enable CUDA functionality (for dfttest2).                                                          
    '''
    if not isinstance(input, vs.VideoNode):
        raise vs.Error('HQDeringmod: this is not a clip')

    if input.format.color_family == vs.RGB:
        raise vs.Error('HQDeringmod: RGB format is not supported')

    if smoothed is not None:
        if not isinstance(smoothed, vs.VideoNode):
            raise vs.Error('HQDeringmod: smoothed is not a clip')

        if smoothed.format.id != input.format.id:
            raise vs.Error("HQDeringmod: smoothed must have the same format as input")

    if ringmask is not None and not isinstance(ringmask, vs.VideoNode):
        raise vs.Error("HQDeringmod: ringmask is not a clip")

    is_gray = input.format.color_family == vs.GRAY

    bits = input.format.bits_per_sample
    neutral = 1 << (bits - 1)
    peak = (1 << bits) - 1

    plane_range = range(input.format.num_planes)

    if isinstance(planes, int):
        planes = [planes]

    HD = input.width > 1024 or input.height > 576

    nrmode = fallback(nrmode, 2 if HD else 1)
    sigma2 = fallback(sigma2, sigma / 16)
    sbsize = fallback(sbsize, 8 if HD else 6)
    sosize = fallback(sosize, 6 if HD else 4)
    drrep = fallback(drrep, 24 if nrmode > 0 else 0)
    darkthr = fallback(darkthr, thr / 4)

    # Kernel: Smoothing
    if smoothed is None:
        if nrmode <= 0:
          useDFTTest2 = false
          # NVRTC is faster than cuFFT but only supports `sbsize == 16`
          if cuda and sbsize == 16:
            useDFTTest2 = hasattr(core,'dfttest2_nvrtc')            
          if useDFTTest2:
              import dfttest2
              smoothed = dfttest2.DFTTest(input, sbsize=sbsize, sosize=sosize, tbsize=1, slocation=[0.0, sigma2, 0.05, sigma, 0.5, sigma, 0.75, sigma2, 1.0, 0.0], planes=planes, backend=dfttest2.Backend.NVRTC)
          else:                      
            smoothed = input.dfttest.DFTTest(sbsize=sbsize, sosize=sosize, tbsize=1, slocation=[0.0, sigma2, 0.05, sigma, 0.5, sigma, 0.75, sigma2, 1.0, 0.0], planes=planes)
        else:
            smoothed = MinBlur(input, nrmode, planes)

    # Post-Process: Contra-Sharpening
    matrix1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    matrix2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    has_zsmooth = hasattr(core,'zsmooth')
    EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr
    if sharp <= 0:
        sclp = smoothed
    else:
        pre = smoothed.zsmooth.Median(planes=planes) if has_zsmooth else smoothed.std.Median(planes=planes)
        if sharp == 1:
            method = pre.std.Convolution(matrix=matrix1, planes=planes)
        elif sharp == 2:
            method = pre.std.Convolution(matrix=matrix1, planes=planes).std.Convolution(matrix=matrix2, planes=planes)
        else:
            method = (
                pre.std.Convolution(matrix=matrix1, planes=planes).std.Convolution(matrix=matrix2, planes=planes).std.Convolution(matrix=matrix2, planes=planes)
            )
        sharpdiff = core.std.MakeDiff(pre, method, planes=planes)
        allD = core.std.MakeDiff(input, smoothed, planes=planes)
        if hasattr(core,'zsmooth'):
          ssDD = core.zsmooth.Repair(sharpdiff, allD, mode=[1 if i in planes else 0 for i in plane_range])
        else:
          ssDD = core.rgvs.Repair(sharpdiff, allD, mode=[1 if i in planes else 0 for i in plane_range])
        ssDD = EXPR(
            [ssDD, sharpdiff], expr=[f'x {neutral} - abs y {neutral} - abs <= x y ?' if i in planes else '' for i in plane_range]
        )
        sclp = core.std.MergeDiff(smoothed, ssDD, planes=planes)

    # Post-Process: Repairing
    if drrep <= 0:
        repclp = sclp
    else:
      if has_zsmooth:
        repclp = core.zsmooth.Repair(input, sclp, mode=[drrep if i in planes else 0 for i in plane_range])
      else:
        repclp = core.rgvs.Repair(input, sclp, mode=[drrep if i in planes else 0 for i in plane_range])

    # Post-Process: Limiting
    if (thr <= 0 and darkthr <= 0) or (thr >= 255 and darkthr >= 255):
        limitclp = repclp
    else:
        limitclp = LimitFilter(repclp, input, thr=thr, elast=elast, brighten_thr=darkthr, planes=planes)
    EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr
    # Post-Process: Ringing Mask Generating
    if ringmask is None:
        expr = f'x {scale(mthr, bits)} < 0 x ?'
        prewittm = EXPR(AvsPrewitt(input, planes=0), expr=expr if is_gray else [expr, ''])
        fmask = core.misc.Hysteresis(prewittm.zsmooth.Median(planes=0), prewittm, planes=0) if has_zsmooth else core.misc.Hysteresis(prewittm.std.Median(planes=0), prewittm, planes=0)
        if mrad > 0:
            omask = mt_expand_multi(fmask, planes=0, sw=mrad, sh=mrad)
        else:
            omask = fmask
        if msmooth > 0:
            omask = mt_inflate_multi(omask, planes=0, radius=msmooth)
        if incedge:
            ringmask = omask
        else:
            if minp > 3:
                imask = fmask.std.Minimum(planes=0).std.Minimum(planes=0)
            elif minp > 2:
                imask = fmask.std.Inflate(planes=0).std.Minimum(planes=0).std.Minimum(planes=0)
            elif minp > 1:
                imask = fmask.std.Minimum(planes=0)
            elif minp > 0:
                imask = fmask.std.Inflate(planes=0).std.Minimum(planes=0)
            else:
                imask = fmask
            expr = f'x {peak} y - * {peak} /'
            ringmask = EXPR([omask, imask], expr=expr if is_gray else [expr, ''])

    # Mask Merging & Output
    if show:
        if is_gray:
            return ringmask
        else:
            return EXPR(ringmask, expr=['', repr(neutral)])
    else:
        return core.std.MaskedMerge(input, limitclp, ringmask, planes=planes, first_plane=True)
        
# Taken from mvsfunc
def mdering(clip: vs.VideoNode, thr: float = 2) -> vs.VideoNode:
    """A simple light and bright DCT ringing remover

    It is a special instance of TMinBlur (r=1 and only filter the bright part) for higher performance.
    Post-processing is needed to reduce degradation of flat and texture areas.

    Args:
        clip: Input clip.

        thr: (float) Threshold in 8 bits scale.
            Default is 2.

    """

    if clip.format.sample_type != vs.INTEGER:
        raise TypeError(funcName + ': \"clip\" must be an integer clip!')

    bits = clip.format.bits_per_sample
    thr = scale(thr, bits)

    rg11_1 = core.std.Convolution(clip, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    rg11_2 = core.std.Convolution(rg11_1, [1]*9)
    rg4_1 = core.std.Median(clip)

    if bits <= 12:
        rg4_2 = core.ctmf.CTMF(clip, radius=2)
    else:
        rg4_2 = core.fmtc.bitdepth(clip, bits=12, dmode=1).ctmf.CTMF(radius=2).fmtc.bitdepth(bits=bits)
        rg4_2 = LimitFilter(clip, rg4_2, thr=0.0625, elast=2)
    EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr
    minblur_1 = EXPR([clip, rg11_1, rg4_1], ['x y - x z - xor x x y - abs x z - abs < y z ? ?'])
    minblur_2 = EXPR([clip, rg11_2, rg4_2], ['x y - x z - xor x x y - abs x z - abs < y z ? ?'])
    dering = EXPR([clip, minblur_1, minblur_2], ['y z - abs {thr} <= y x <= and y x ?'.format(thr=thr)])

    return dering

# Taken from mvsfunc
################################################################################################################################
## Utility function: LimitFilter()
################################################################################################################################
## Similar to the AviSynth function Dither_limit_dif16() and HQDeringmod_limit_dif16().
## It acts as a post-processor, and is very useful to limit the difference of filtering while avoiding artifacts.
## Commonly used cases:
##     de-banding
##     de-ringing
##     de-noising
##     sharpening
##     combining high precision source with low precision filtering: mvf.LimitFilter(src, flt, thr=1.0, elast=2.0)
################################################################################################################################
## There are 2 implementations, default one with std.Expr, the other with std.Lut.
## The Expr version supports all mode, while the Lut version doesn't support float input and ref clip.
## Also the Lut version will truncate the filtering diff if it exceeds half the value range(128 for 8-bit, 32768 for 16-bit).
## The Lut version might be faster than Expr version in some cases, for example 8-bit input and brighten_thr != thr.
################################################################################################################################
## Algorithm for Y/R/G/B plane (for chroma, replace "thr" and "brighten_thr" with "thrc")
##     dif = flt - src
##     dif_ref = flt - ref
##     dif_abs = abs(dif_ref)
##     thr_1 = brighten_thr if (dif > 0) else thr
##     thr_2 = thr_1 * elast
##
##     if dif_abs <= thr_1:
##         final = flt
##     elif dif_abs >= thr_2:
##         final = src
##     else:
##         final = src + dif * (thr_2 - dif_abs) / (thr_2 - thr_1)
################################################################################################################################
## Basic parameters
##     flt {clip}: filtered clip, to compute the filtering diff
##         can be of YUV/RGB/Gray color family, can be of 8-16 bit integer or 16/32 bit float
##     src {clip}: source clip, to apply the filtering diff
##         must be of the same format and dimension as "flt"
##     ref {clip} (optional): reference clip, to compute the weight to be applied on filtering diff
##         must be of the same format and dimension as "flt"
##         default: None (use "src")
##     thr {float}: threshold (8-bit scale) to limit filtering diff
##         default: 1.0
##     elast {float}: elasticity of the soft threshold
##         default: 2.0
##     planes {int[]}: specify which planes to process
##         unprocessed planes will be copied from "flt"
##         default: all planes will be processed, [0,1,2] for YUV/RGB input, [0] for Gray input
################################################################################################################################
## Advanced parameters
##     brighten_thr {float}: threshold (8-bit scale) for filtering diff that brightening the image (Y/R/G/B plane)
##         set a value different from "thr" is useful to limit the overshoot/undershoot/blurring introduced in sharpening/de-ringing
##         default is the same as "thr"
##     thrc {float}: threshold (8-bit scale) for chroma (U/V/Co/Cg plane)
##         default is the same as "thr"
##     force_expr {bool}
##         - True: force to use the std.Expr implementation
##         - False: use the std.Lut implementation if available
##         default: True
################################################################################################################################
def LimitFilter(flt, src, ref=None, thr=None, elast=None, brighten_thr=None, thrc=None, force_expr=None, planes=None):
    # input clip
    if not isinstance(flt, vs.VideoNode):
        raise type_error('"flt" must be a clip!')
    if not isinstance(src, vs.VideoNode):
        raise type_error('"src" must be a clip!')
    if ref is not None and not isinstance(ref, vs.VideoNode):
        raise type_error('"ref" must be a clip!')

    # Get properties of input clip
    sFormat = flt.format
    if sFormat.id != src.format.id:
        raise value_error('"flt" and "src" must be of the same format!')
    if flt.width != src.width or flt.height != src.height:
        raise value_error('"flt" and "src" must be of the same width and height!')

    if ref is not None:
        if sFormat.id != ref.format.id:
            raise value_error('"flt" and "ref" must be of the same format!')
        if flt.width != ref.width or flt.height != ref.height:
            raise value_error('"flt" and "ref" must be of the same width and height!')

    sColorFamily = sFormat.color_family
    CheckColorFamily(sColorFamily)
    sIsYUV = sColorFamily == vs.YUV

    sSType = sFormat.sample_type
    sbitPS = sFormat.bits_per_sample
    sNumPlanes = sFormat.num_planes

    # Parameters
    if thr is None:
        thr = 1.0
    elif isinstance(thr, int) or isinstance(thr, float):
        if thr < 0:
            raise value_error('valid range of "thr" is [0, +inf)')
    else:
        raise type_error('"thr" must be an int or a float!')

    if elast is None:
        elast = 2.0
    elif isinstance(elast, int) or isinstance(elast, float):
        if elast < 1:
            raise value_error('valid range of "elast" is [1, +inf)')
    else:
        raise type_error('"elast" must be an int or a float!')

    if brighten_thr is None:
        brighten_thr = thr
    elif isinstance(brighten_thr, int) or isinstance(brighten_thr, float):
        if brighten_thr < 0:
            raise value_error('valid range of "brighten_thr" is [0, +inf)')
    else:
        raise type_error('"brighten_thr" must be an int or a float!')

    if thrc is None:
        thrc = thr
    elif isinstance(thrc, int) or isinstance(thrc, float):
        if thrc < 0:
            raise value_error('valid range of "thrc" is [0, +inf)')
    else:
        raise type_error('"thrc" must be an int or a float!')

    if force_expr is None:
        force_expr = True
    elif not isinstance(force_expr, int):
        raise type_error('"force_expr" must be a bool!')
    if ref is not None or sSType != vs.INTEGER:
        force_expr = True

    VSMaxPlaneNum = 3
    # planes
    process = [0 for i in range(VSMaxPlaneNum)]

    if planes is None:
        process = [1 for i in range(VSMaxPlaneNum)]
    elif isinstance(planes, int):
        if planes < 0 or planes >= VSMaxPlaneNum:
            raise value_error(f'valid range of "planes" is [0, {VSMaxPlaneNum})!')
        process[planes] = 1
    elif isinstance(planes, Sequence):
        for p in planes:
            if not isinstance(p, int):
                raise type_error('"planes" must be a (sequence of) int!')
            elif p < 0 or p >= VSMaxPlaneNum:
                raise value_error(f'valid range of "planes" is [0, {VSMaxPlaneNum})!')
            process[p] = 1
    else:
        raise type_error('"planes" must be a (sequence of) int!')

    # Process
    if thr <= 0 and brighten_thr <= 0:
        if sIsYUV:
            if thrc <= 0:
                return src
        else:
            return src
    if thr >= 255 and brighten_thr >= 255:
        if sIsYUV:
            if thrc >= 255:
                return flt
        else:
            return flt
    if thr >= 128 or brighten_thr >= 128:
        force_expr = True

    if force_expr: # implementation with std.Expr
        valueRange = (1 << sbitPS) - 1 if sSType == vs.INTEGER else 1
        limitExprY = _limit_filter_expr(ref is not None, thr, elast, brighten_thr, valueRange)
        limitExprC = _limit_filter_expr(ref is not None, thrc, elast, thrc, valueRange)
        expr = []
        EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr
        for i in range(sNumPlanes):
            if process[i]:
                if i > 0 and (sIsYUV):
                    expr.append(limitExprC)
                else:
                    expr.append(limitExprY)
            else:
                expr.append("")

        if ref is None:
            clip = EXPR([flt, src], expr)
        else:
            clip = EXPR([flt, src, ref], expr)
    else: # implementation with std.MakeDiff, std.Lut and std.MergeDiff
        diff = core.std.MakeDiff(flt, src, planes=planes)
        if sIsYUV:
            if process[0]:
                diff = _limit_diff_lut(diff, thr, elast, brighten_thr, [0])
            if process[1] or process[2]:
                _planes = []
                if process[1]:
                    _planes.append(1)
                if process[2]:
                    _planes.append(2)
                diff = _limit_diff_lut(diff, thrc, elast, thrc, _planes)
        else:
            diff = _limit_diff_lut(diff, thr, elast, brighten_thr, planes)
        clip = core.std.MakeDiff(flt, diff, planes=planes)

    # Output
    return clip
################################################################################################################################


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
    EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr
    return EXPR([clp, RG11, RG4], expr=[expr if i in planes else '' for i in range(clp.format.num_planes)])
    
    
    
################################################################################################################################
## Helper function: CheckColorFamily()
################################################################################################################################
def CheckColorFamily(color_family, valid_list=None, invalid_list=None):
    if valid_list is None:
        valid_list = ('RGB', 'YUV', 'GRAY')
    if invalid_list is None:
        invalid_list = ('COMPAT', 'UNDEFINED')
    # check invalid list
    for cf in invalid_list:
        if color_family == getattr(vs, cf, None):
            raise value_error(f'color family *{cf}* is not supported!')
    # check valid list
    if valid_list:
        if color_family not in [getattr(vs, cf, None) for cf in valid_list]:
            raise value_error(f'color family not supported, only {valid_list} are accepted')
################################################################################################################################

################################################################################################################################
## Internal used functions for LimitFilter()
################################################################################################################################
def _limit_filter_expr(defref, thr, elast, largen_thr, value_range):
    flt = " x "
    src = " y "
    ref = " z " if defref else src

    dif = f" {flt} {src} - "
    dif_ref = f" {flt} {ref} - "
    dif_abs = dif_ref + " abs "

    thr = thr * value_range / 255
    largen_thr = largen_thr * value_range / 255

    if thr <= 0 and largen_thr <= 0:
        limitExpr = f" {src} "
    elif thr >= value_range and largen_thr >= value_range:
        limitExpr = ""
    else:
        if thr <= 0:
            limitExpr = f" {src} "
        elif thr >= value_range:
            limitExpr = f" {flt} "
        elif elast <= 1:
            limitExpr = f" {dif_abs} {thr} <= {flt} {src} ? "
        else:
            thr_1 = thr
            thr_2 = thr * elast
            thr_slope = 1 / (thr_2 - thr_1)
            # final = src + dif * (thr_2 - dif_abs) / (thr_2 - thr_1)
            limitExpr = f" {src} {dif} {thr_2} {dif_abs} - * {thr_slope} * + "
            limitExpr = f" {dif_abs} {thr_1} <= {flt} {dif_abs} {thr_2} >= {src} " + limitExpr + " ? ? "

        if largen_thr != thr:
            if largen_thr <= 0:
                limitExprLargen = f" {src} "
            elif largen_thr >= value_range:
                limitExprLargen = f" {flt} "
            elif elast <= 1:
                limitExprLargen = f" {dif_abs} {largen_thr} <= {flt} {src} ? "
            else:
                thr_1 = largen_thr
                thr_2 = largen_thr * elast
                thr_slope = 1 / (thr_2 - thr_1)
                # final = src + dif * (thr_2 - dif_abs) / (thr_2 - thr_1)
                limitExprLargen = f" {src} {dif} {thr_2} {dif_abs} - * {thr_slope} * + "
                limitExprLargen = f" {dif_abs} {thr_1} <= {flt} {dif_abs} {thr_2} >= {src} " + limitExprLargen + " ? ? "
            limitExpr = f" {flt} {ref} > " + limitExprLargen + " " + limitExpr + " ? "

    return limitExpr
    
    

def _expr2(clips: vs.VideoNode | list[vs.VideoNode], expr: str | list[str]) -> vs.VideoNode:
    """Expr — prefers llvmexpr → akarin → cranexpr → std."""
    if hasattr(core, "llvmexpr"):
        return core.llvmexpr.Expr(clips, expr)
    if hasattr(core, "akarin"):
        return core.akarin.Expr(clips, expr)
    if hasattr(core, "cranexpr"):
        return core.cranexpr.Expr(clips, expr)
    raise RuntimeError(
            "AntiRingLR requires akarin, llvmexpr, or cranexpr — "
            "none found. Install akarin, llvmexpr or cranexpr into your VS plugins folder."
    )

def AntiRingLR(
    cl: vs.VideoNode,
    planes: str = "all",
    weightL: float = 0.25,
    offsetL: int = 1,
    weightR: float = 0.25,
    offsetR: int = 1,
    knee: float = 0.7,
    pr2: float = 0.5,
    pr3: float = 0.5,
) -> vs.VideoNode:
    """
    Hans' Ringing Remover — VapourSynth port.
    Output format always matches input format.

    Parameters
    ----------
    cl      : Input clip.
    planes  : "luma", "chroma", or "all".
    weightL : Left-side correction intensity  (0.15 – 0.7; 0.0 = inactive).
    offsetL : Left-side distance from edge    (1 – 3).
    weightR : Right-side correction intensity.
    offsetR : Right-side distance from edge   (1 – 3).
    knee    : Softness of the on/off curve    (0.2 – 1.5).
    pr2     : Protection for radius-2 signal components (0.0 – 1.5).
    pr3     : Protection for radius-3 signal components (0.0 – 1.5).
    """

 

    fmt_in   = cl.format
    is_gray  = fmt_in.color_family == vs.GRAY
    work_fmt = vs.GRAYS if is_gray else vs.YUV444PS

    cl_f = core.resize.Bicubic(cl, format=work_fmt)

    pr2_coeff = pr2 * 0.36
    pr3_coeff = pr3 * 0.3
    scale     = 10.0 / knee
    bias      = 0.5  * knee

    s128 = 128.0 / 255.0
    s64  =  64.0 / 255.0
    s127 = 127.0 / 255.0

    luma_L = (
        "x[2,0] x[1,0] - dup * 0.75 * "
        "x[1,0] x - dup * + "
        "x x[-1,0] - dup * + "
        "x[-1,0] x[-2,0] - dup * 0.75 * + sqrt "
    )

    prot_L = (
        f"x[3,0] x[1,0] - dup * 0.66 * x[2,0] x[0,0] - dup * + "
        f"x[1,0] x[-1,0] - dup * + x[0,0] x[-2,0] - dup * 0.66 * + sqrt {pr2_coeff} * - "
        f"x[3,0] x[0,0] - dup * 0.5 * x[2,0] x[-1,0] - dup * + "
        f"x[1,0] x[-2,0] - dup * + x[0,0] x[-3,0] - dup * 0.66 * + sqrt {pr3_coeff} * - "
    )

    sigmoid_correct_L = (
        f"{scale} / {bias} - 0.0 max "
        f"dup 1.0 - swap 1.0 + / 1.0 + {s128} * 1.0 min "
        f"x[{offsetL},0] x[{2 + offsetL},0] - * {weightL} * -{s64} max {s127} min "
    )

    luma_R = (
        "x[2,0] x[1,0] - dup * 0.75 * "
        "x[1,0] x - dup * + "
        "x x[-1,0] - dup * + "
        "x[-1,0] x[-2,0] - dup * 0.75 * + sqrt "
    )

    prot_R = (
        f"x[-3,0] x[-1,0] - dup * 0.66 * x[-2,0] x[0,0] - dup * + "
        f"x[-1,0] x[1,0] - dup * + x[0,0] x[2,0] - dup * 0.66 * + sqrt {pr2_coeff} * - "
        f"x[-3,0] x[0,0] - dup * 0.5 * x[-2,0] x[1,0] - dup * + "
        f"x[-1,0] x[2,0] - dup * + x[0,0] x[3,0] - dup * 0.66 * + sqrt {pr3_coeff} * - "
    )

    sigmoid_correct_R = (
        f"{scale} / {bias} - 0.0 max "
        f"dup 1.0 - swap 1.0 + / 1.0 + {s128} * 1.0 min "
        f"x[{-offsetR},0] x[{-2 - offsetR},0] - * {weightR} * -{s64} max {s127} min + x + "
    )

    expr = (
        luma_L + prot_L + sigmoid_correct_L
        + luma_R + prot_R + sigmoid_correct_R
    )

    do_luma   = planes in ("luma",   "all")
    do_chroma = planes in ("chroma", "all")

    expr_y = expr if do_luma   else "x"
    expr_u = expr if do_chroma else "x"
    expr_v = expr if do_chroma else "x"
    expr_list = [expr_y] if is_gray else [expr_y, expr_u, expr_v]

    result = _expr2(cl_f, expr_list)

    return core.resize.Bicubic(result, format=fmt_in)
    
    
def AntiRingLRUD(
    cl: vs.VideoNode,
    planes: str = "all",
    weightL: float = 0.25,
    offsetL: int = 1,
    weightR: float = 0.25,
    offsetR: int = 1,
    knee: float = 0.7,
    pr2: float = 0.5,
    pr3: float = 0.5
) -> vs.VideoNode:
    """
    Hans' Ringing Remover — horizontal + vertical pass.
    Applies AntiRingLR twice: once normally (fixes left/right ringing),
    and once on a 90°-rotated clip (fixes up/down ringing).
    """

    # Pass 1 — horizontal ringing
    result = AntiRingLR(cl, planes=planes,
                        weightL=weightL, offsetL=offsetL,
                        weightR=weightR, offsetR=offsetR,
                        knee=knee, pr2=pr2, pr3=pr3)

    # Pass 2 — vertical ringing via 90° rotation
    rotated = core.std.Transpose(result)
    rotated = AntiRingLR(rotated, planes=planes,
                         weightL=weightL, offsetL=offsetL,
                         weightR=weightR, offsetR=offsetR,
                         knee=knee, pr2=pr2, pr3=pr3)
    result = core.std.Transpose(rotated)

    return result



def AntiRingLR2(
    cl: vs.VideoNode,
    planes: str = "luma",
    weightL: float = 0.25,
    offsetL: int = 1,
    weightR: float = 0.25,
    offsetR: int = 1,
    knee: float = 0.7,
    pr2: float = 0.5,
    pr3: float = 0.5,
) -> vs.VideoNode:
    """
    Hans' Ringing Remover v2
    Extends AntiRingLR with a secondary right-side correction pass
    (weightR * 0.6, offset+1) that suppresses a second artifact lobe.

    Parameters
    ----------
    cl      : Input clip.
    planes  : "luma", "chroma", or "all".
    weightL : Left-side correction intensity  (0.15 – 0.7; 0.0 = inactive).
    offsetL : Left-side distance from edge    (1 – 3).
    weightR : Right-side correction intensity.
    offsetR : Right-side distance from edge   (1 – 3).
    knee    : Softness of the on/off curve    (0.2 – 1.5).
    pr2     : Protection for radius-2 signal components (0.0 – 1.5).
    pr3     : Protection for radius-3 signal components (0.0 – 1.5).
    """

    fmt_in   = cl.format
    is_gray  = fmt_in.color_family == vs.GRAY
    work_fmt = vs.GRAYS if is_gray else vs.YUV444PS

    cl_f = core.resize.Bicubic(cl, format=work_fmt)

    pr2_coeff  = pr2 * 0.36
    pr3_coeff  = pr3 * 0.3
    scale      = 10.0 / knee
    bias       = 0.5  * knee
    weightR2   = weightR * 0.6

    s128 = 128.0 / 255.0
    s64  =  64.0 / 255.0
    s127 = 127.0 / 255.0

    # --- Left-side block ---
    luma_L = (
        "x[2,0] x[1,0] - dup * 0.75 * "
        "x[1,0] x - dup * + "
        "x x[-1,0] - dup * + "
        "x[-1,0] x[-2,0] - dup * 0.75 * + sqrt "
    )
    prot_L = (
        f"x[3,0] x[1,0] - dup * 0.66 * x[2,0] x[0,0] - dup * + "
        f"x[1,0] x[-1,0] - dup * + x[0,0] x[-2,0] - dup * 0.66 * + sqrt {pr2_coeff} * - "
        f"x[3,0] x[0,0] - dup * 0.5 * x[2,0] x[-1,0] - dup * + "
        f"x[1,0] x[-2,0] - dup * + x[0,0] x[-3,0] - dup * 0.66 * + sqrt {pr3_coeff} * - "
    )
    sigmoid_correct_L = (
        f"{scale} / {bias} - 0.0 max "
        f"dup 1.0 - swap 1.0 + / 1.0 + {s128} * 1.0 min "
        f"x[{offsetL},0] x[{2 + offsetL},0] - * {weightL} * -{s64} max {s127} min "
    )

    # --- Right-side primary block ---
    luma_R = (
        "x[2,0] x[1,0] - dup * 0.75 * "
        "x[1,0] x - dup * + "
        "x x[-1,0] - dup * + "
        "x[-1,0] x[-2,0] - dup * 0.75 * + sqrt "
    )
    prot_R = (
        f"x[-3,0] x[-1,0] - dup * 0.66 * x[-2,0] x[0,0] - dup * + "
        f"x[-1,0] x[1,0] - dup * + x[0,0] x[2,0] - dup * 0.66 * + sqrt {pr2_coeff} * - "
        f"x[-3,0] x[0,0] - dup * 0.5 * x[-2,0] x[1,0] - dup * + "
        f"x[-1,0] x[2,0] - dup * + x[0,0] x[3,0] - dup * 0.66 * + sqrt {pr3_coeff} * - "
    )
    sigmoid_correct_R = (
        f"{scale} / {bias} - 0.0 max "
        f"dup 1.0 - swap 1.0 + / 1.0 + {s128} * 1.0 min "
        f"x[{-offsetR},0] x[{-2 - offsetR},0] - * {weightR} * -{s64} max {s127} min + "
    )

    # --- Right-side secondary block (offset+1, weight*0.6, subtracted) ---
    # Targets the second ringing lobe one pixel further right.
    # AviSynth original uses a shifted luma window: x[1,0]..x[-3,0]
    luma_R2 = (
        "x[1,0] x[0,0] - dup * 0.75 * "
        "x[0,0] x[-1,0] - dup * + "
        "x[-1,0] x[-2,0] - dup * + "
        "x[-2,0] x[-3,0] - dup * 0.75 * + sqrt "
    )
    prot_R2 = (
        f"x[-4,0] x[-2,0] - dup * 0.66 * x[-3,0] x[-1,0] - dup * + "
        f"x[-2,0] x[0,0] - dup * + x[-1,0] x[1,0] - dup * 0.66 * + sqrt {pr2_coeff} * - "
        f"x[-4,0] x[-1,0] - dup * 0.5 * x[-3,0] x[0,0] - dup * + "
        f"x[-2,0] x[1,0] - dup * + x[-1,0] x[2,0] - dup * 0.66 * + sqrt {pr3_coeff} * - "
    )
    sigmoid_correct_R2 = (
        f"{scale} / {bias} - 0.0 max "
        f"dup 1.0 - swap 1.0 + / 1.0 + {s128} * 1.0 min "
        f"x[{-1 - offsetR},0] x[{-3 - offsetR},0] - * {weightR2} * -{s64} max {s127} min - "
    )

    expr = (
        luma_L + prot_L + sigmoid_correct_L
        + luma_R + prot_R + sigmoid_correct_R
        + luma_R2 + prot_R2 + sigmoid_correct_R2
        + "x + "
    )

    do_luma   = planes in ("luma",   "all")
    do_chroma = planes in ("chroma", "all")

    expr_y = expr if do_luma   else "x"
    expr_u = expr if do_chroma else "x"
    expr_v = expr if do_chroma else "x"
    expr_list = [expr_y] if is_gray else [expr_y, expr_u, expr_v]

    result = _expr2(cl_f, expr_list)

    return core.resize.Bicubic(result, format=fmt_in)
    
    
def AntiRingLR2UD(
    cl: vs.VideoNode,
    planes: list[int] | None = None,
    weightL: float = 0.25,
    offsetL: int = 1,
    weightR: float = 0.25,
    offsetR: int = 1,
    knee: float = 0.7,
    pr2: float = 0.5,
    pr3: float = 0.5,
) -> vs.VideoNode:
    """
    Hans' Ringing Remover v2 — horizontal + vertical pass.
    Applies AntiRingLR2 twice: once normally (fixes left/right ringing),
    and once on a 90°-rotated clip (fixes up/down ringing).

    planes : List of plane indices to process. Default [0] = luma only.
    """

    if planes is None:
        planes = [0]

    result = AntiRingLR2(cl, planes=planes,
                         weightL=weightL, offsetL=offsetL,
                         weightR=weightR, offsetR=offsetR,
                         knee=knee, pr2=pr2, pr3=pr3)

    rotated = core.std.Transpose(result)
    rotated = AntiRingLR2(rotated, planes=planes,
                          weightL=weightL, offsetL=offsetL,
                          weightR=weightR, offsetR=offsetR,
                          knee=knee, pr2=pr2, pr3=pr3)
    return core.std.Transpose(rotated)
    
    
################################################################################################################################

def AvsPrewitt(clip: vs.VideoNode, planes: Optional[Union[int, Sequence[int]]] = None) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('AvsPrewitt: this is not a clip')

    plane_range = range(clip.format.num_planes)

    if planes is None:
        planes = list(plane_range)
    elif isinstance(planes, int):
        planes = [planes]
    EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr
    return EXPR(
        [
            clip.std.Convolution(matrix=[1, 1, 0, 1, 0, -1, 0, -1, -1], planes=planes, saturate=False),
            clip.std.Convolution(matrix=[1, 1, 1, 0, 0, 0, -1, -1, -1], planes=planes, saturate=False),
            clip.std.Convolution(matrix=[1, 0, -1, 1, 0, -1, 1, 0, -1], planes=planes, saturate=False),
            clip.std.Convolution(matrix=[0, -1, -1, 1, 0, -1, 1, 1, 0], planes=planes, saturate=False),
        ],
        expr=['x y max z max a max' if i in planes else '' for i in plane_range],
    )

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
    
def mt_inflate_multi(src: vs.VideoNode, planes: Optional[Union[int, Sequence[int]]] = None, radius: int = 1) -> vs.VideoNode:
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('mt_inflate_multi: this is not a clip')

    for _ in range(radius):
        src = src.std.Inflate(planes=planes)
    return src
    
    
def cround(x: float) -> int:
    return math.floor(x + 0.5) if x > 0 else math.ceil(x - 0.5)
    
def scale(value, peak):
    return cround(value * peak / 255) if peak != 1 else value / 255
        
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
