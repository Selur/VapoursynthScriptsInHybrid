import vapoursynth as vs
from vapoursynth import core
import color
import math
from typing import Union, Optional, Callable, Dict, Any


__version__ = '2'


def nnedi3_resample(input, target_width=None, target_height=None, src_left=None, src_top=None, src_width=None, src_height=None, csp=None, mats=None, matd=None, cplaces=None, cplaced=None, fulls=None, fulld=None, curves=None, curved=None, sigmoid=None, scale_thr=None, nsize=None, nns=None, qual=None, etype=None, pscrn=None, opt=None, int16_prescreener=None, int16_predictor=None, exp=None, kernel=None, invks=False, taps=None, invkstaps=3, a1=None, a2=None, chromak_up=None, chromak_up_taps=None, chromak_up_a1=None, chromak_up_a2=None, chromak_down=None, chromak_down_invks=False, chromak_down_invkstaps=3, chromak_down_taps=None, chromak_down_a1=None, chromak_down_a2=None, mode=None, device=None):
    funcName = 'nnedi3_resample'
    
    # Get property about input clip
    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': This is not a clip!')
    
    sFormat = input.format
    
    sColorFamily = sFormat.color_family
    sIsGRAY = sColorFamily == vs.GRAY
    sIsYUV = sColorFamily == vs.YUV
    sIsRGB = sColorFamily == vs.RGB
    
    sbitPS = sFormat.bits_per_sample
    
    sHSubS = 1 << sFormat.subsampling_w
    sVSubS = 1 << sFormat.subsampling_h
    sIsSubS = sHSubS > 1 or sVSubS > 1
    
    sPlaneNum = sFormat.num_planes
    
    # Get property about output clip
    dFormat = sFormat if csp is None else core.get_video_format(csp)
    
    dColorFamily = dFormat.color_family
    dIsGRAY = dColorFamily == vs.GRAY
    dIsYUV = dColorFamily == vs.YUV
    dIsRGB = dColorFamily == vs.RGB
    
    dbitPS = dFormat.bits_per_sample
    
    dHSubS = 1 << dFormat.subsampling_w
    dVSubS = 1 << dFormat.subsampling_h
    dIsSubS = dHSubS > 1 or dVSubS > 1
    
    dPlaneNum = dFormat.num_planes
    
    # Parameters of format
    SD = input.width <= 1024 and input.height <= 576
    HD = input.width <= 2048 and input.height <= 1536
    
    if mats is None:
        mats = "601" if SD else "709" if HD else "2020"
    else:
        mats = mats.lower()
    if matd is None:
        matd = mats
    else:
        matd = matd.lower()
        # Matrix of output clip makes sense only if dst is not of RGB
        if dIsRGB:
            matd = mats
        # Matrix of input clip makes sense only src is not of GRAY or RGB
        if sIsGRAY or sIsRGB:
            mats = matd
    if cplaces is None:
        if sHSubS == 4:
            cplaces = 'dv'
        else:
            cplaces = 'mpeg2'
    else:
        cplaces = cplaces.lower()
    if cplaced is None:
        if dHSubS == 4:
            cplaced = 'dv'
        else:
            cplaced = cplaces
    else:
        cplaced = cplaced.lower()
    if fulls is None:
        fulls = sColorFamily == vs.RGB
    if fulld is None:
        if dColorFamily == sColorFamily:
            fulld = fulls
        else:
            fulld = dColorFamily == vs.RGB
    if curves is None:
        curves = 'linear'
    else:
        curves = curves.lower()
    if curved is None:
        curved = curves
    else:
        curved = curved.lower()
    if sigmoid is None:
        sigmoid = False
    
    # Parameters of scaling
    if target_width is None:
        target_width = input.width
    if target_height is None:
        target_height = input.height
    if src_left is None:
        src_left = 0
    if src_top is None:
        src_top = 0
    if src_width is None:
        src_width = input.width
    elif src_width <= 0:
        src_width = input.width - src_left + src_width
    if src_height is None:
        src_height = input.height
    elif src_height <= 0:
        src_height = input.height - src_top + src_height
    if scale_thr is None:
        scale_thr = 1.125
    
    src_right = src_width - input.width + src_left
    src_bottom = src_height - input.height + src_top
    
    hScale = target_width / src_width
    vScale = target_height / src_height
    
    # Parameters of nnedi3
    if nsize is None:
        nsize = 0
    if nns is None:
        nns = 3
    if qual is None:
        qual = 2
    
    # Parameters of fmtc.resample
    if kernel is None:
        if not invks:
            kernel = 'spline36'
        else:
            kernel = 'bilinear'
    else:
        kernel = kernel.lower()
    if chromak_up is None:
        chromak_up = 'nnedi3'
    else:
        chromak_up = chromak_up.lower()
    if chromak_up == 'softcubic':
        chromak_up = 'bicubic'
        if chromak_up_a1 is None:
            chromak_up_a1 = 75
        chromak_up_a1 = chromak_up_a1 / 100
        chromak_up_a2 = 1 - chromak_up_a1
    if chromak_down is None:
        chromak_down = 'bicubic'
    else:
        chromak_down = chromak_down.lower()
    if chromak_down == 'softcubic':
        chromak_down = 'bicubic'
        if chromak_down_a1 is None:
            chromak_down_a1 = 75
        chromak_down_a1 = chromak_down_a1 / 100
        chromak_down_a2 = 1 - chromak_down_a1
    
    # Procedure decision
    hIsScale = hScale != 1
    vIsScale = vScale != 1
    isScale = hIsScale or vIsScale
    hResample = hIsScale or int(src_left) != src_left or int(src_right) != src_right
    vResample = vIsScale or int(src_top) != src_top or int(src_bottom) != src_bottom
    resample = hResample or vResample
    hReSubS = dHSubS != sHSubS
    vReSubS = dVSubS != sVSubS
    reSubS = hReSubS or vReSubS
    sigmoid = sigmoid and resample
    sGammaConv = curves != 'linear'
    dGammaConv = curved != 'linear'
    gammaConv = (sGammaConv or dGammaConv or sigmoid) and (resample or curved != curves)
    scaleInGRAY = sIsGRAY or dIsGRAY
    scaleInYUV = not scaleInGRAY and mats == matd and not gammaConv and (reSubS or (sIsYUV and dIsYUV))
    scaleInRGB = not scaleInGRAY and not scaleInYUV
    # If matrix conversion or gamma correction is applied, scaling will be done in RGB. Otherwise, if at least one of input&output clip is RGB and no chroma subsampling is involved, scaling will be done in RGB.
    
    # Chroma placement relative to the frame center in luma scale
    sCLeftAlign = cplaces == 'mpeg2' or cplaces == 'dv'
    sHCPlace = 0 if not sCLeftAlign else 0.5 - sHSubS / 2
    sVCPlace = 0
    dCLeftAlign = cplaced == 'mpeg2' or cplaced == 'dv'
    dHCPlace = 0 if not dCLeftAlign else 0.5 - dHSubS / 2
    dVCPlace = 0
    
    # Convert depth to 16-bit
    last = color.Depth(input, depth=16, fulls=fulls)
    
    # Color space conversion before scaling
    if scaleInGRAY and sIsYUV:
        if mats != matd:
            last = core.fmtc.matrix(last, mats=mats, matd=matd, fulls=fulls, fulld=fulld, col_fam=vs.GRAY, singleout=0)
        last = core.std.ShufflePlanes(last, [0], vs.GRAY)
    elif scaleInGRAY and sIsRGB:
        # Matrix conversion for output clip of GRAY
        last = core.fmtc.matrix(last, mat=matd, fulls=fulls, fulld=fulld, col_fam=vs.GRAY, singleout=0)
        fulls = fulld
    elif scaleInRGB and sIsYUV:
        # Chroma upsampling
        if sIsSubS:
            if chromak_up == 'nnedi3':
                # Separate planes
                Y = core.std.ShufflePlanes(last, [0], vs.GRAY)
                U = core.std.ShufflePlanes(last, [1], vs.GRAY)
                V = core.std.ShufflePlanes(last, [2], vs.GRAY)
                # Chroma up-scaling
                U = nnedi3_resample_kernel(U, Y.width, Y.height, -sHCPlace / sHSubS, -sVCPlace / sVSubS, None, None, 1, nsize, nns, qual, etype, pscrn, opt, int16_prescreener, int16_predictor, exp, kernel, taps, a1, a2, mode=mode, device=device)
                V = nnedi3_resample_kernel(V, Y.width, Y.height, -sHCPlace / sHSubS, -sVCPlace / sVSubS, None, None, 1, nsize, nns, qual, etype, pscrn, opt, int16_prescreener, int16_predictor, exp, kernel, taps, a1, a2, mode=mode, device=device)
                # Merge planes
                last = core.std.ShufflePlanes([Y, U, V], [0, 0, 0], last.format.color_family)
            else:
                last = core.fmtc.resample(last, kernel=chromak_up, taps=chromak_up_taps, a1=chromak_up_a1, a2=chromak_up_a2, css="444", fulls=fulls, cplaces=cplaces)
        # Matrix conversion
        if mats == '2020cl':
            last = core.fmtc.matrix2020cl(last, fulls)
        else:
            last = core.fmtc.matrix(last, mat=mats, fulls=fulls, fulld=True, col_fam=vs.RGB, singleout=-1)
        fulls = True
    elif scaleInYUV and sIsRGB:
        # Matrix conversion
        if matd == '2020cl':
            last = core.fmtc.matrix2020cl(last, fulld)
        else:
            last = core.fmtc.matrix(last, mat=matd, fulls=fulls, fulld=fulld, col_fam=vs.YUV, singleout=-1)
        fulls = fulld
    
    # Scaling
    if scaleInGRAY or scaleInRGB:
        if gammaConv and sGammaConv:
            last = GammaToLinear(last, fulls, fulls, curves, sigmoid=sigmoid)
        elif sigmoid:
            last = SigmoidInverse(last)
        last = nnedi3_resample_kernel(last, target_width, target_height, src_left, src_top, src_width, src_height, scale_thr, nsize, nns, qual, etype, pscrn, opt, int16_prescreener, int16_predictor, exp, kernel, taps, a1, a2, invks, invkstaps, mode, device)
        if gammaConv and dGammaConv:
            last = LinearToGamma(last, fulls, fulls, curved, sigmoid=sigmoid)
        elif sigmoid:
            last = SigmoidDirect(last)
    elif scaleInYUV:
        # Separate planes
        Y = core.std.ShufflePlanes(last, [0], vs.GRAY)
        U = core.std.ShufflePlanes(last, [1], vs.GRAY)
        V = core.std.ShufflePlanes(last, [2], vs.GRAY)
        # Scale Y
        Y = nnedi3_resample_kernel(Y, target_width, target_height, src_left, src_top, src_width, src_height, scale_thr, nsize, nns, qual, etype, pscrn, opt, int16_prescreener, int16_predictor, exp, kernel, taps, a1, a2, mode=mode, device=device)
        # Scale UV
        dCw = target_width // dHSubS
        dCh = target_height // dVSubS
        dCsx = ((src_left - sHCPlace) * hScale + dHCPlace) / hScale / sHSubS
        dCsy = ((src_top - sVCPlace) * vScale + dVCPlace) / vScale / sVSubS
        dCsw = src_width / sHSubS
        dCsh = src_height / sVSubS
        U = nnedi3_resample_kernel(U, dCw, dCh, dCsx, dCsy, dCsw, dCsh, scale_thr, nsize, nns, qual, etype, pscrn, opt, int16_prescreener, int16_predictor, exp, kernel, taps, a1, a2, mode=mode, device=device)
        V = nnedi3_resample_kernel(V, dCw, dCh, dCsx, dCsy, dCsw, dCsh, scale_thr, nsize, nns, qual, etype, pscrn, opt, int16_prescreener, int16_predictor, exp, kernel, taps, a1, a2, mode=mode, device=device)
        # Merge planes
        last = core.std.ShufflePlanes([Y, U, V], [0, 0, 0], last.format.color_family)
    
    # Color space conversion after scaling
    if scaleInGRAY and dIsYUV:
        dCw = target_width // dHSubS
        dCh = target_height // dVSubS
        last = color.Depth(last, depth=dbitPS, fulls=fulls, fulld=fulld)
        blkUV = core.std.BlankClip(last, dCw, dCh, color=[1 << (dbitPS - 1)])
        last = core.std.ShufflePlanes([last, blkUV, blkUV], [0, 0, 0], dColorFamily)
    elif scaleInGRAY and dIsRGB:
        last = color.Depth(last, depth=dbitPS, fulls=fulls, fulld=fulld)
        last = core.std.ShufflePlanes([last, last, last], [0, 0, 0], dColorFamily)
    elif scaleInRGB and dIsYUV:
        # Matrix conversion
        if matd == '2020cl':
            last = core.fmtc.matrix2020cl(last, fulld)
        else:
            last = core.fmtc.matrix(last, mat=matd, fulls=fulls, fulld=fulld, col_fam=dColorFamily, singleout=-1)
        # Chroma subsampling
        if dIsSubS:
            dCSS = '411' if dHSubS == 4 else '420' if dVSubS == 2 else '422'
            last = core.fmtc.resample(last, kernel=chromak_down, taps=chromak_down_taps, a1=chromak_down_a1, a2=chromak_down_a2, css=dCSS, fulls=fulld, cplaced=cplaced, invks=chromak_down_invks, invkstaps=chromak_down_invkstaps, planes=[2,3,3])
        last = color.Depth(last, depth=dbitPS, fulls=fulld)
    elif scaleInYUV and dIsRGB:
        # Matrix conversion
        if mats == '2020cl':
            last = core.fmtc.matrix2020cl(last, fulls)
        else:
            last = core.fmtc.matrix(last, mat=mats, fulls=fulls, fulld=True, col_fam=vs.RGB, singleout=-1)
        last = color.Depth(last, depth=dbitPS, fulls=True, fulld=fulld)
    else:
        last = color.Depth(last, depth=dbitPS, fulls=fulls, fulld=fulld)
    
    # Output
    return last


def nnedi3_resample_kernel(input, target_width=None, target_height=None, src_left=None, src_top=None, src_width=None, src_height=None, scale_thr=None, nsize=None, nns=None, qual=None, etype=None, pscrn=None, opt=None, int16_prescreener=None, int16_predictor=None, exp=None, kernel=None, taps=None, a1=None, a2=None, invks=False, invkstaps=3, mode=None, device=None):

    # Parameters of scaling
    if target_width is None:
        target_width = input.width
    if target_height is None:
        target_height = input.height
    if src_left is None:
        src_left = 0
    if src_top is None:
        src_top = 0
    if src_width is None:
        src_width = input.width
    elif src_width <= 0:
        src_width = input.width - src_left + src_width
    if src_height is None:
        src_height = input.height
    elif src_height <= 0:
        src_height = input.height - src_top + src_height
    if scale_thr is None:
        scale_thr = 1.125
    
    src_right = src_width - input.width + src_left
    src_bottom = src_height - input.height + src_top
    
    hScale = target_width / src_width
    vScale = target_height / src_height
    
    # Parameters of nnedi3
    if nsize is None:
        nsize = 0
    if nns is None:
        nns = 3
    if qual is None:
        qual = 2
    
    # Parameters of fmtc.resample
    if kernel is None:
        kernel = 'spline36'
    else:
        kernel = kernel.lower()
    
    # Procedure decision
    hIsScale = hScale != 1
    vIsScale = vScale != 1
    isScale = hIsScale or vIsScale
    hResample = hIsScale or int(src_left) != src_left or int(src_right) != src_right
    vResample = vIsScale or int(src_top) != src_top or int(src_bottom) != src_bottom
    resample = hResample or vResample
    
    # Scaling
    last = input
    
    if hResample:
        last = core.std.Transpose(last)
        last = nnedi3_resample_kernel_vertical(last, target_width, src_left, src_width, scale_thr, nsize, nns, qual, etype, pscrn, opt, int16_prescreener, int16_predictor, exp, kernel, taps, a1, a2, invks, invkstaps, mode, device)
        last = core.std.Transpose(last)
    if vResample:
        last = nnedi3_resample_kernel_vertical(last, target_height, src_top, src_height, scale_thr, nsize, nns, qual, etype, pscrn, opt, int16_prescreener, int16_predictor, exp, kernel, taps, a1, a2, invks, invkstaps, mode, device)
    
    # Output
    return last


def nnedi3_resample_kernel_vertical(input, target_height=None, src_top=None, src_height=None, scale_thr=None, nsize=None, nns=None, qual=None, etype=None, pscrn=None, opt=None, int16_prescreener=None, int16_predictor=None, exp=None, kernel=None, taps=None, a1=None, a2=None, invks=False, invkstaps=3, mode=None, device=None):
    
    # Parameters of scaling
    if target_height is None:
        target_height = input.height
    if src_top is None:
        src_top = 0
    if src_height is None:
        src_height = input.height
    elif src_height <= 0:
        src_height = input.height - src_top + src_height
    if scale_thr is None:
        scale_thr = 1.125
    
    scale = target_height / src_height # Total scaling ratio
    eTimes = math.ceil(math.log(scale / scale_thr, 2)) if scale > scale_thr else 0 # Iterative times of nnedi3
    eScale = 1 << eTimes # Scaling ratio of nnedi3
    pScale = scale / eScale # Scaling ratio of fmtc.resample
    
    # Parameters of nnedi3
    if nsize is None:
        nsize = 0
    if nns is None:
        nns = 3
    if qual is None:
        qual = 2
    
    # Parameters of fmtc.resample
    if kernel is None:
        kernel = 'spline36'
    else:
        kernel = kernel.lower()
    
    # Skip scaling if not needed
    if scale == 1 and src_top == 0 and src_height == input.height:
        return input
    
    # Scaling with nnedi3
    last = nnedi3_rpow2_vertical(input, eTimes, 1, nsize, nns, qual, etype, pscrn, opt, int16_prescreener, int16_predictor, exp, mode, device)
    
    # Center shift calculation
    vShift = 0.5 if eTimes >= 1 else 0
    
    # Scaling with fmtc.resample as well as correct center shift
    w = last.width
    h = target_height
    sx = 0
    sy = src_top * eScale - vShift
    sw = last.width
    sh = src_height * eScale
    
    if h != last.height or sy != 0 or sh != last.height:
        if h < last.height and invks is True:
            last = core.fmtc.resample(last, w, h, sx, sy, sw, sh, kernel=kernel, taps=taps, a1=a1, a2=a2, invks=True, invkstaps=invkstaps)
        else:
            last = core.fmtc.resample(last, w, h, sx, sy, sw, sh, kernel=kernel, taps=taps, a1=a1, a2=a2)
    
    # Output
    return last


def nnedi3_rpow2_vertical(input, eTimes=1, field=1, nsize=None, nns=None, qual=None, etype=None, pscrn=None, opt=None, int16_prescreener=None, int16_predictor=None, exp=None, mode=None, device=None):
    
    if eTimes >= 1:
        last = nnedi3_dh(input, field, nsize, nns, qual, etype, pscrn, opt, int16_prescreener, int16_predictor, exp, mode, device)
        eTimes = eTimes - 1
        field = 0
    else:
        last = input
    
    if eTimes >= 1:
        return nnedi3_rpow2_vertical(last, eTimes, field, nsize, nns, qual, etype, pscrn, opt, int16_prescreener, int16_predictor, exp, mode, device)
    else:
        return last


def nnedi3_dh(input, field=1, nsize=None, nns=None, qual=None, etype=None, pscrn=None, opt=None, int16_prescreener=None, int16_predictor=None, exp=None, mode=None, device=None):
    nnedi3_args1 = dict(nsize=nsize, nns=nns, qual=qual, etype=etype, pscrn=pscrn)
    nnedi3_args2 = dict(opt=opt, int16_prescreener=int16_prescreener, int16_predictor=int16_predictor, exp=exp)

    # check nnedi3 plugin installation
    has_znedi3 = hasattr(core, 'znedi3')
    has_nnedi3 = hasattr(core, 'nnedi3')
    has_nnedi3cl = hasattr(core, 'nnedi3cl') or hasattr(core, 'sneedif')
    if not (has_znedi3 or has_nnedi3 or has_nnedi3cl):
        raise ValueError(f'nnedi3_dh: znedi3, nnedi3, nnedi3cl or sneedif installation not found.')

    # select default plugin
    if mode is None:
        mode = 'znedi3' if has_znedi3 else 'nnedi3' if has_nnedi3 else 'nnedi3cl' if has_nnedi3cl else None

    # nnedi3 interpolation to double height
    if mode == 'znedi3':
        res = core.znedi3.nnedi3(input, field=field, dh=True, **nnedi3_args1, **nnedi3_args2)
    elif mode == 'nnedi3':
        res = core.nnedi3.nnedi3(input, field=field, dh=True, **nnedi3_args1, **nnedi3_args2)
    elif mode == 'nnedi3cl':
        if hasattr(core, 'sneedif'):
          res = core.sneedif.NNEDI3(input, field=field, dh=True, **nnedi3_args1, device=device)
        else:
          res = core.nnedi3cl.NNEDI3CL(input, field=field, dh=True, **nnedi3_args1, device=device)
    else:
        raise ValueError(f'nnedi3_dh: Unsupported mode={mode}, should be znedi3, nnedi3 or nnedi3cl.')

    return res


## Gamma conversion functions from HAvsFunc-r18
# Convert the luma channel to linear light
def GammaToLinear(src, fulls=True, fulld=True, curve='709', planes=[0, 1, 2], gcor=1., sigmoid=False, thr=0.5, cont=6.5):
    if not isinstance(src, vs.VideoNode) or src.format.bits_per_sample != 16:
        raise ValueError('GammaToLinear: This is not a 16-bit clip')
    
    return LinearAndGamma(src, False, fulls, fulld, curve.lower(), planes, gcor, sigmoid, thr, cont)

# Convert back a clip to gamma-corrected luma
def LinearToGamma(src, fulls=True, fulld=True, curve='709', planes=[0, 1, 2], gcor=1., sigmoid=False, thr=0.5, cont=6.5):
    if not isinstance(src, vs.VideoNode) or src.format.bits_per_sample != 16:
        raise ValueError('LinearToGamma: This is not a 16-bit clip')
    
    return LinearAndGamma(src, True, fulls, fulld, curve.lower(), planes, gcor, sigmoid, thr, cont)

def LinearAndGamma(src, l2g_flag, fulls, fulld, curve, planes, gcor, sigmoid, thr, cont):
    
    if curve == 'srgb':
        c_num = 0
    elif curve in ['709', '601', '170']:
        c_num = 1
    elif curve == '240':
        c_num = 2
    elif curve == '2020':
        c_num = 3
    else:
        raise ValueError('LinearAndGamma: wrong curve value')
    
    if src.format.color_family == vs.GRAY:
        planes = [0]
    
    #                 BT-709/601
    #        sRGB     SMPTE 170M   SMPTE 240M   BT-2020
    k0    = [0.04045, 0.081,       0.0912,      0.08145][c_num]
    phi   = [12.92,   4.5,         4.0,         4.5][c_num]
    alpha = [0.055,   0.099,       0.1115,      0.0993][c_num]
    gamma = [2.4,     2.22222,     2.22222,     2.22222][c_num]
    
    def g2l(x):
        expr = x / 65536 if fulls else (x - 4096) / 56064
        if expr <= k0:
            expr /= phi
        else:
            expr = ((expr + alpha) / (1 + alpha)) ** gamma
        if gcor != 1 and expr >= 0:
            expr **= gcor
        if sigmoid:
            x0 = 1 / (1 + math.exp(cont * thr))
            x1 = 1 / (1 + math.exp(cont * (thr - 1)))
            expr = thr - math.log(max(1 / max(expr * (x1 - x0) + x0, 0.000001) - 1, 0.000001)) / cont
        if fulld:
            return min(max(round(expr * 65536), 0), 65535)
        else:
            return min(max(round(expr * 56064 + 4096), 0), 65535)
    
    # E' = (E <= k0 / phi)   ?   E * phi   :   (E ^ (1 / gamma)) * (alpha + 1) - alpha
    def l2g(x):
        expr = x / 65536 if fulls else (x - 4096) / 56064
        if sigmoid:
            x0 = 1 / (1 + math.exp(cont * thr))
            x1 = 1 / (1 + math.exp(cont * (thr - 1)))
            expr = (1 / (1 + math.exp(cont * (thr - expr))) - x0) / (x1 - x0)
        if gcor != 1 and expr >= 0:
            expr **= gcor
        if expr <= k0 / phi:
            expr *= phi
        else:
            expr = expr ** (1 / gamma) * (alpha + 1) - alpha
        if fulld:
            return min(max(round(expr * 65536), 0), 65535)
        else:
            return min(max(round(expr * 56064 + 4096), 0), 65535)
    
    return core.std.Lut(src, planes=planes, function=l2g if l2g_flag else g2l)

# Apply the inverse sigmoid curve to a clip in linear luminance
def SigmoidInverse(src, thr=0.5, cont=6.5, planes=[0, 1, 2]):
    
    if not isinstance(src, vs.VideoNode) or src.format.bits_per_sample != 16:
        raise ValueError('SigmoidInverse: This is not a 16-bit clip')
    
    if src.format.color_family == vs.GRAY:
        planes = [0]
    
    def get_lut(x):
        x0 = 1 / (1 + math.exp(cont * thr))
        x1 = 1 / (1 + math.exp(cont * (thr - 1)))
        return min(max(round((thr - math.log(max(1 / max(x / 65536 * (x1 - x0) + x0, 0.000001) - 1, 0.000001)) / cont) * 65536), 0), 65535)
    
    return core.std.Lut(src, planes=planes, function=get_lut)

# Convert back a clip to linear luminance
def SigmoidDirect(src, thr=0.5, cont=6.5, planes=[0, 1, 2]):
    
    if not isinstance(src, vs.VideoNode) or src.format.bits_per_sample != 16:
        raise ValueError('SigmoidDirect: This is not a 16-bit clip')
    
    if src.format.color_family == vs.GRAY:
        planes = [0]
    
    def get_lut(x):
        x0 = 1 / (1 + math.exp(cont * thr))
        x1 = 1 / (1 + math.exp(cont * (thr - 1)))
        return min(max(round(((1 / (1 + math.exp(cont * (thr - x / 65536))) - x0) / (x1 - x0)) * 65536), 0), 65535)
    
    return core.std.Lut(src, planes=planes, function=get_lut)
## Gamma conversion functions from HAvsFunc-r18


import functools
from typing import Sequence

def SSIM_downsample(clip: vs.VideoNode, w: int, h: int, smooth: Union[float, Union[vs.Func, Callable[..., vs.VideoNode]]] = 1,
                    kernel: Optional[str] = None, use_fmtc: bool = False, gamma: bool = False,
                    fulls: bool = False, fulld: bool = False, curve: str = '709', sigmoid: bool = False,
                    epsilon: float = 1e-6, depth_args: Optional[Dict[str, Any]] = None,
                    **resample_args: Any) -> vs.VideoNode:
    """SSIM downsampler

    SSIM downsampler is an image downscaling technique that aims to optimize for the perceptual quality of the downscaled results.
    Image downscaling is considered as an optimization problem
    where the difference between the input and output images is measured using famous Structural SIMilarity (SSIM) index.
    The solution is derived in closed-form, which leads to the simple, efficient implementation.
    The downscaled images retain perceptually important features and details,
    resulting in an accurate and spatio-temporally consistent representation of the high resolution input.

    This is an pseudo-implementation of SSIM downsampler with slight modification.
    The pre-downsampling is performed by vszimg/fmtconv, and the behaviour of convolution at the border is uniform.

    All the internal calculations are done at 32-bit float, except gamma correction is done at integer.

    Args:
        clip: The input clip.

        w, h: The size of the output clip.

        smooth: (int, float or function) The method to smooth the image.
            If it's an int, it specifies the "radius" of the internel used boxfilter, i.e. the window has a size of (2*smooth+1)x(2*smooth+1).
            If it's a float, it specifies the "sigma" of core.tcanny.TCanny, i.e. the standard deviation of gaussian blur.
            If it's a function, it acs as a general smoother.
            Default is 1. The 3x3 boxfilter will be performed.

        kernel: (string) Resample kernel of vszimg/fmtconv.
            Default is 'Bicubic'.

        use_fmtc: (bool) Whether to use fmtconv for downsampling. If not, vszimg (core.resize.*) will be used.
            Default is False.

        depth_args: (dict) Additional arguments passed to color.Depth().
            Default is {}.

        gamma: (bool) Default is False.
            Set to true to turn on gamma correction for the y channel.

        fulls: (bool) Default is False.
            Specifies if the luma is limited range (False) or full range (True)

        fulld: (bool) Default is False.
            Same as fulls, but for output.

        curve: (string) Default is '709'.
            Type of gamma mapping.

        sigmoid: (bool) Default is False.
            When True, applies a sigmoidal curve after the power-like curve (or before when converting from linear to gamma-corrected).
            This helps reducing the dark halo artefacts around sharp edges caused by resizing in linear luminance.

        resample_args: (dict) Additional arguments passed to vszimg/fmtconv in the form of keyword arguments.
            Refer to the documentation of downsample() as an example.

            Default is {}.

    Ref:
        [1] Oeztireli, A. C., & Gross, M. (2015). Perceptually based downscaling of images. ACM Transactions on Graphics (TOG), 34(4), 77.

    """

    funcName = 'SSIM_downsample'

    if not isinstance(clip, vs.VideoNode):
        raise TypeError(funcName + ': \"clip\" must be a clip!')

    if depth_args is None:
        depth_args = {}

    if callable(smooth):
        Filter = smooth
    elif isinstance(smooth, int):
        Filter = functools.partial(BoxFilter, radius=smooth+1)
    elif isinstance(smooth, float):
        Filter = functools.partial(core.tcanny.TCanny, sigma=smooth, mode=-1)
    else:
        raise TypeError(funcName + ': \"smooth\" must be a int, float or a function!')

    if kernel is None:
        kernel = 'Bicubic'

    if gamma:
        clip = color.GammaToLinear(color.Depth(clip, 16), fulls=fulls, fulld=fulld, curve=curve, sigmoid=sigmoid, planes=[0])

    clip = color.Depth(clip, depth=32, sample=vs.FLOAT, **depth_args)

    EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else (core.akarin.Expr if hasattr(core, 'akarin') else core.std.Expr)
    kernel = kernel.capitalize()
    if use_fmtc:
        l = core.fmtc.resample(clip, w, h, kernel=kernel, **resample_args)
        l2 = core.fmtc.resample(EXPR([clip], ['x dup *']), w, h, kernel=kernel, **resample_args)
    else: # use vszimg
        l = eval(f'core.resize.{kernel}')(clip, w, h, **resample_args)
        l2 = eval(f'core.resize.{kernel}')(EXPR([clip], ["x dup *"]), w, h, **resample_args)

    m = Filter(l)
    sl_plus_m_square = Filter(core.std.Expr([l], ['x dup *']))
    sh_plus_m_square = Filter(l2)
    m_square = EXPR([m], ['x dup *'])
    r = EXPR([sl_plus_m_square, sh_plus_m_square, m_square], ['x z - {eps} < 0 y z - x z - / sqrt ?'.format(eps=epsilon)])
    t = Filter(EXPR([r, m], ['x y *']))
    m = Filter(m)
    r = Filter(r)
    d = EXPR([m, r, l, t], ['x y z * + a -'])

    if gamma:
        d = LinearToGamma(Depth(d, 16), fulls=fulls, fulld=fulld, curve=curve, sigmoid=sigmoid, planes=[0])

    return d

def Depth(src, bits, dither_type='error_diffusion', range=None, range_in=None):
    src_f = src.format
    src_cf = src_f.color_family
    src_st = src_f.sample_type
    src_bits = src_f.bits_per_sample
    src_sw = src_f.subsampling_w
    src_sh = src_f.subsampling_h
    dst_st = vs.INTEGER if bits < 32 else vs.FLOAT

    if isinstance(range, str):
        range = RANGEDICT[range]

    if isinstance(range_in, str):
        range_in = RANGEDICT[range_in]

    if (src_bits, range_in) == (bits, range):
        return src
    out_f = core.register_format(src_cf, dst_st, bits, src_sw, src_sh)
    return core.resize.Point(src, format=out_f.id, dither_type=dither_type, range=range, range_in=range_in)
    
def BoxFilter(input: vs.VideoNode, radius: int = 16, radius_v: Optional[int] = None, planes: Optional[Union[int, Sequence[int]]] = None,
              fmtc_conv: int = 0, radius_thr: Optional[int] = None,
              resample_args: Optional[Dict[str, Any]] = None, keep_bits: bool = True,
              depth_args: Optional[Dict[str, Any]] = None
              ) -> vs.VideoNode:
    '''Box filter

    Performs a box filtering on the input clip.
    Box filtering consists in averaging all the pixels in a square area whose center is the output pixel.
    You can approximate a large gaussian filtering by cascading a few box filters.

    Args:
        input: Input clip to be filtered.

        radius, radius_v: (int) Size of the averaged square. The size is (radius*2-1) * (radius*2-1).
            If "radius_v" is None, it will be set to "radius".
            Default is 16.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from the source clip, "input".

        fmtc_conv: (0~2) Whether to use fmtc.resample for convolution.
            It's recommended to input clip without chroma subsampling when using fmtc.resample, otherwise the output may be incorrect.
            0: False. 1: True (except both "radius" and "radius_v" is strictly smaller than 4).
                2: Auto, determined by radius_thr (exclusive).
            Default is 0.

        radius_thr: (int) Threshold of wheter to use fmtc.resample when "fmtc_conv" is 2.
            Default is 11 for integer input and 21 for float input.
            Only works when "fmtc_conv" is enabled.

        resample_args: (dict) Additional parameters passed to core.fmtc.resample in the form of dict.
            It's recommended to set "flt" to True for higher precision, like:
                flt = muf.BoxFilter(src, resample_args=dict(flt=True))
            Only works when "fmtc_conv" is enabled.
            Default is {}.

        keep_bits: (bool) Whether to keep the bitdepth of the output the same as input.
            Only works when "fmtc_conv" is enabled and input is integer.

        depth_args: (dict) Additional parameters passed to color.Depth in the form of dict.
            Only works when "fmtc_conv" is enabled, input is integer and "keep_bits" is True.
            Default is {}.

    '''

    funcName = 'BoxFilter'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if planes is None:
        planes = list(range(input.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    if radius_v is None:
        radius_v = radius

    if radius == radius_v == 1:
        return input

    if radius_thr is None:
        radius_thr = 21 if input.format.sample_type == vs.FLOAT else 11 # Values are measured from my experiment

    if resample_args is None:
        resample_args = {}

    if depth_args is None:
        depth_args = {}

    planes2 = [(3 if i in planes else 2) for i in range(input.format.num_planes)]
    width = radius * 2 - 1
    width_v = radius_v * 2 - 1
    kernel = [1 / width] * width
    kernel_v = [1 / width_v] * width_v

    # process
    if input.format.sample_type == vs.FLOAT:
        if core.version_number() < 33:
            raise NotImplementedError(funcName + (': Please update your VapourSynth.'
                'BoxBlur on float sample has not yet been implemented on current version.'))
        elif radius == radius_v == 2 or radius == radius_v == 3:
            return core.std.Convolution(input, [1] * ((radius * 2 - 1) * (radius * 2 - 1)), planes=planes, mode='s')

        else:
            if fmtc_conv == 1 or (fmtc_conv != 0 and radius > radius_thr): # Use fmtc.resample for convolution
                flt = core.fmtc.resample(input, kernel='impulse', impulseh=kernel, impulsev=kernel_v, planes=planes2,
                    cnorm=False, fh=-1, fv=-1, center=False, **resample_args)
                return flt # No bitdepth conversion is required since fmtc.resample outputs the same bitdepth as input

            elif core.version_number() >= 39:
                return core.std.BoxBlur(input, hradius=radius-1, vradius=radius_v-1, planes=planes)

            else: # BoxBlur on float sample has not been implemented
                if radius > 1:
                    input = core.std.Convolution(input, [1] * (radius * 2 - 1), planes=planes, mode='h')
                if radius_v > 1:
                    input = core.std.Convolution(input, [1] * (radius_v * 2 - 1), planes=planes, mode='v')
                return input

    else: # input.format.sample_type == vs.INTEGER
        if radius == radius_v == 2 or radius == radius_v == 3:
            return core.std.Convolution(input, [1] * ((radius * 2 - 1) * (radius * 2 - 1)), planes=planes, mode='s')

        else:
            if fmtc_conv == 1 or (fmtc_conv != 0 and radius > radius_thr): # Use fmtc.resample for convolution
                flt = core.fmtc.resample(input, kernel='impulse', impulseh=kernel, impulsev=kernel_v, planes=planes2,
                    cnorm=False, fh=-1, fv=-1, center=False, **resample_args)
                if keep_bits and input.format.bits_per_sample != flt.format.bits_per_sample:
                    flt = color.Depth(flt, depth=input.format.bits_per_sample, **depth_args)
                return flt

            elif hasattr(core.std, 'BoxBlur'):
                return core.std.BoxBlur(input, hradius=radius-1, vradius=radius_v-1, planes=planes)

            else: # BoxBlur was not found
                if radius > 1:
                    input = core.std.Convolution(input, [1] * (radius * 2 - 1), planes=planes, mode='h')
                if radius_v > 1:
                    input = core.std.Convolution(input, [1] * (radius_v * 2 - 1), planes=planes, mode='v')
                return input