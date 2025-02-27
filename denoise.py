import vapoursynth as vs
from vapoursynth import core

import math

from typing import Optional, Union, Sequence

#from vsutil import Dither, depth, fallback, get_depth, get_y, join, plane, scale_value
from vsutil import get_depth, scale_value



# Taken from havsfunc
def KNLMeansCL(
    clip: vs.VideoNode,
    d: Optional[int] = None,
    a: Optional[int] = None,
    s: Optional[int] = None,
    h: Optional[float] = None,
    wmode: Optional[int] = None,
    wref: Optional[float] = None,
    device_type: Optional[str] = None,
    device_id: Optional[int] = None,
) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('KNLMeansCL: this is not a clip')

    if clip.format.color_family != vs.YUV:
        raise vs.Error('KNLMeansCL: this wrapper is intended to be used only for YUV format')

    use_cuda = hasattr(core, 'nlm_cuda')
    subsampled = clip.format.subsampling_w > 0 or clip.format.subsampling_h > 0

    if use_cuda:
        nlmeans = clip.nlm_cuda.NLMeans
    else:
        nlmeans = clip.knlm.KNLMeansCL

    if subsampled:
        return nlmeans(d=d, a=a, s=s, h=h, wmode=wmode, wref=wref, device_type=device_type, device_id=device_id).knlm.KNLMeansCL(
            d=d, a=a, s=s, h=h, channels='UV', wmode=wmode, wref=wref, device_type=device_type, device_id=device_id
        )
    else:
        return nlmeans(d=d, a=a, s=s, h=h, channels='YUV', wmode=wmode, wref=wref, device_type=device_type, device_id=device_id)
        
        
        
####################################################################################################################################
###                                                                                                                              ###
###                                   Motion-Compensated Temporal Denoise: MCTemporalDenoise()                                   ###
###                                                                                                                              ###
###                                                     v1.4.20 by "LaTo INV."                                                   ###
###                                                                                                                              ###
###                                                           2 July 2010                                                        ###
###                                                                                                                              ###
####################################################################################################################################
###
###
###
### /!\ Needed filters: MVTools, DFTTest, FFT3DFilter, TTempSmooth, RGVS, Deblock, DCTFilter
### -------------------
###
###
###
### USAGE: MCTemporalDenoise(i, radius, pfMode, sigma, twopass, useTTmpSm, limit, limit2, post, chroma, refine,
###                          deblock, useQED, quant1, quant2,
###                          edgeclean, ECrad, ECthr,
###                          stabilize, maxr, TTstr,
###                          bwbh, owoh, blksize, overlap,
###                          bt, ncpu,
###                          thSAD, thSADC, thSAD2, thSADC2, thSCD1, thSCD2,
###                          truemotion, MVglobal, pel, pelsearch, search, searchparam, MVsharp, DCT,
###                          p, settings)
###
###
###
### PARAMETERS:
### -----------
###
### +---------+
### | DENOISE |
### +---------+--------------------------------------------------------------------------------------+
### | radius    : Temporal radius [1...6]                                                            |
### | pfMode    : Pre-filter mode [-1=off,0=FFT3DFilter,1=MinBlur(1),2=MinBlur(2),3=DFTTest]         |
### | sigma     : FFT3D sigma for the pre-filtering clip (if pfMode=0)                               |
### | twopass   : Do the denoising job in 2 stages (stronger but very slow)                          |
### | useTTmpSm : Use MDegrain (faster) or MCompensate+TTempSmooth (stronger)                        |
### | limit     : Limit the effect of the first denoising [-1=auto,0=off,1...255]                    |
### | limit2    : Limit the effect of the second denoising (if twopass=true) [-1=auto,0=off,1...255] |
### | post      : Sigma value for post-denoising with FFT3D [0=off,...]                              |
### | chroma    : Process or not the chroma plane                                                    |
### | refine    : Refine and recalculate motion data of previously estimated motion vectors          |
### +------------------------------------------------------------------------------------------------+
###
###
### +---------+
### | DEBLOCK |
### +---------+-----------------------------------------------------------------------------------+
### | deblock : Enable deblocking before the denoising                                            |
### | useQED  : If true, use Deblock_QED, else use Deblock (faster & stronger)                    |
### | quant1  : Deblock_QED "quant1" parameter (Deblock "quant" parameter is "(quant1+quant2)/2") |
### | quant2  : Deblock_QED "quant2" parameter (Deblock "quant" parameter is "(quant1+quant2)/2") |
### +---------------------------------------------------------------------------------------------+
###
###
### +------------------------------+
### | EDGECLEAN: DERING, DEHALO... |
### +------------------------------+-----------------------------------------------------------------------------------------------------+
### | edgeclean : Enable safe edgeclean process after the denoising (only on edges which are in non-detailed areas, so less detail loss) |
### | ECrad     : Radius for mask (the higher, the greater distance from the edge is filtered)                                           |
### | ECthr     : Threshold for mask (the higher, the less "small edges" are process) [0...255]                                          |
### +------------------------------------------------------------------------------------------------------------------------------------+
###
###
### +-----------+
### | STABILIZE |
### +-----------+------------------------------------------------------------------------------------------------+
### | stabilize : Enable TTempSmooth post processing to stabilize flat areas (background will be less "nervous") |
### | maxr      : Temporal radius (the higher, the more stable image)                                            |
### | TTstr     : Strength (see TTempSmooth docs)                                                                |
### +------------------------------------------------------------------------------------------------------------+
###
###
### +---------------------+
### | BLOCKSIZE / OVERLAP |
### +---------------------+----------------+
### | bwbh    : FFT3D blocksize            |
### | owoh    : FFT3D overlap              |
### |             - for speed:   bwbh/4    |
### |             - for quality: bwbh/2    |
### | blksize : MVTools blocksize          |
### | overlap : MVTools overlap            |
### |             - for speed:   blksize/4 |
### |             - for quality: blksize/2 |
### +--------------------------------------+
###
###
### +-------+
### | FFT3D |
### +-------+--------------------------+
### | bt   : FFT3D block temporal size |
### | ncpu : FFT3DFilter ncpu          |
### +----------------------------------+
###
###
### +---------+
### | MVTOOLS |
### +---------+------------------------------------------------------+
### | thSAD   : MVTools thSAD for the first pass                     |
### | thSADC  : MVTools thSADC for the first pass                    |
### | thSAD2  : MVTools thSAD for the second pass (if twopass=true)  |
### | thSADC2 : MVTools thSADC for the second pass (if twopass=true) |
### | thSCD1  : MVTools thSCD1                                       |
### | thSCD2  : MVTools thSCD2                                       |
### +-----------------------------------+----------------------------+
### | truemotion  : MVTools truemotion  |
### | MVglobal    : MVTools global      |
### | pel         : MVTools pel         |
### | pelsearch   : MVTools pelsearch   |
### | search      : MVTools search      |
### | searchparam : MVTools searchparam |
### | MVsharp     : MVTools sharp       |
### | DCT         : MVTools DCT         |
### +-----------------------------------+
###
###
### +--------+
### | GLOBAL |
### +--------+-----------------------------------------------------+
### | p        : Set an external prefilter clip                    |
### | settings : Global MCTemporalDenoise settings [default="low"] |
### |             - "very low"                                     |
### |             - "low"                                          |
### |             - "medium"                                       |
### |             - "high"                                         |
### |             - "very high"                                    |
### +--------------------------------------------------------------+
###
###
###
### DEFAULTS:
### ---------
###
### +-------------+----------------------+----------------------+----------------------+----------------------+----------------------+
### | SETTINGS    |      VERY LOW        |      LOW             |      MEDIUM          |      HIGH            |      VERY HIGH       |
### |-------------+----------------------+----------------------+----------------------+----------------------+----------------------|
### | radius      |      1               |      2               |      3               |      2               |      3               |
### | pfMode      |      3               |      3               |      3               |      3               |      3               |
### | sigma       |      2               |      4               |      8               |      12              |      16              |
### | twopass     |      false           |      false           |      false           |      true            |      true            |
### | useTTmpSm   |      false           |      false           |      false           |      false           |      false           |
### | limit       |      -1              |      -1              |      -1              |      -1              |      0               |
### | limit2      |      -1              |      -1              |      -1              |      0               |      0               |
### | post        |      0               |      0               |      0               |      0               |      0               |
### | chroma      |      false           |      false           |      true            |      true            |      true            |
### |-------------+----------------------+----------------------+----------------------+----------------------+----------------------|
### | deblock     |      false           |      false           |      false           |      false           |      false           |
### | useQED      |      true            |      true            |      true            |      false           |      false           |
### | quant1      |      10              |      20              |      30              |      30              |      40              |
### | quant2      |      20              |      40              |      60              |      60              |      80              |
### |-------------+----------------------+----------------------+----------------------+----------------------+----------------------|
### | edgeclean   |      false           |      false           |      false           |      false           |      false           |
### | ECrad       |      1               |      2               |      3               |      4               |      5               |
### | ECthr       |      64              |      32              |      32              |      16              |      16              |
### |-------------+----------------------+----------------------+----------------------+----------------------+----------------------|
### | stabilize   |      false           |      false           |      false           |      true            |      true            |
### | maxr        |      1               |      1               |      2               |      2               |      2               |
### | TTstr       |      1               |      1               |      1               |      2               |      2               |
### |-------------+----------------------+----------------------+----------------------+----------------------+----------------------|
### | bwbh        |      HD?16:8         |      HD?16:8         |      HD?16:8         |      HD?16:8         |      HD?16:8         |
### | owoh        |      HD? 8:4         |      HD? 8:4         |      HD? 8:4         |      HD? 8:4         |      HD? 8:4         |
### | blksize     |      HD?16:8         |      HD?16:8         |      HD?16:8         |      HD?16:8         |      HD?16:8         |
### | overlap     |      HD? 8:4         |      HD? 8:4         |      HD? 8:4         |      HD? 8:4         |      HD? 8:4         |
### |-------------+----------------------+----------------------+----------------------+----------------------+----------------------|
### | bt          |      1               |      3               |      3               |      3               |      4               |
### | ncpu        |      1               |      1               |      1               |      1               |      1               |
### |-------------+----------------------+----------------------+----------------------+----------------------+----------------------|
### | thSAD       |      200             |      300             |      400             |      500             |      600             |
### | thSADC      |      thSAD/2         |      thSAD/2         |      thSAD/2         |      thSAD/2         |      thSAD/2         |
### | thSAD2      |      200             |      300             |      400             |      500             |      600             |
### | thSADC2     |      thSAD2/2        |      thSAD2/2        |      thSAD2/2        |      thSAD2/2        |      thSAD2/2        |
### | thSCD1      |      200             |      300             |      400             |      500             |      600             |
### | thSCD2      |      90              |      100             |      100             |      130             |      130             |
### |-------------+----------------------+----------------------+----------------------+----------------------+----------------------|
### | truemotion  |      false           |      false           |      false           |      false           |      false           |
### | MVglobal    |      true            |      true            |      true            |      true            |      true            |
### | pel         |      1               |      2               |      2               |      2               |      2               |
### | pelsearch   |      1               |      2               |      2               |      2               |      2               |
### | search      |      4               |      4               |      4               |      4               |      4               |
### | searchparam |      2               |      2               |      2               |      2               |      2               |
### | MVsharp     |      2               |      2               |      2               |      1               |      0               |
### | DCT         |      0               |      0               |      0               |      0               |      0               |
### +-------------+----------------------+----------------------+----------------------+----------------------+----------------------+
###
####################################################################################################################################
def MCTemporalDenoise(i, radius=None, pfMode=3, sigma=None, twopass=None, useTTmpSm=False, limit=None, limit2=None, post=0, chroma=None, refine=False, deblock=False, useQED=None, quant1=None,
                      quant2=None, edgeclean=False, ECrad=None, ECthr=None, stabilize=None, maxr=None, TTstr=None, bwbh=None, owoh=None, blksize=None, overlap=None, bt=None, ncpu=1, thSAD=None,
                      thSADC=None, thSAD2=None, thSADC2=None, thSCD1=None, thSCD2=None, truemotion=False, MVglobal=True, pel=None, pelsearch=None, search=4, searchparam=2, MVsharp=None, DCT=0, p=None,
                      settings='low', cuda=False):
    if not isinstance(i, vs.VideoNode):
        raise vs.Error('MCTemporalDenoise: this is not a clip')

    if p is not None and (not isinstance(p, vs.VideoNode) or p.format.id != i.format.id):
        raise vs.Error("MCTemporalDenoise: 'p' must be the same format as input")

    isGray = (i.format.color_family == vs.GRAY)

    neutral = 1 << (i.format.bits_per_sample - 1)
    peak = (1 << i.format.bits_per_sample) - 1

    ### DEFAULTS
    try:
        settings_num = ['very low', 'low', 'medium', 'high', 'very high'].index(settings.lower())
    except:
        raise vs.Error('MCTemporalDenoise: these settings do not exist')

    HD = i.width > 1024 or i.height > 576

    if radius is None:
        radius = [1, 2, 3, 2, 3][settings_num]
    if sigma is None:
        sigma = [2, 4, 8, 12, 16][settings_num]
    if twopass is None:
        twopass = [False, False, False, True, True][settings_num]
    if limit is None:
        limit = [-1, -1, -1, -1, 0][settings_num]
    if limit2 is None:
        limit2 = [-1, -1, -1, 0, 0][settings_num]
    if chroma is None:
        chroma = [False, False, True, True, True][settings_num]
    if useQED is None:
        useQED = [True, True, True, False, False][settings_num]
    if quant1 is None:
        quant1 = [10, 20, 30, 30, 40][settings_num]
    if quant2 is None:
        quant2 = [20, 40, 60, 60, 80][settings_num]
    if ECrad is None:
        ECrad = [1, 2, 3, 4, 5][settings_num]
    if ECthr is None:
        ECthr = [64, 32, 32, 16, 16][settings_num]
    if stabilize is None:
        stabilize = [False, False, False, True, True][settings_num]
    if maxr is None:
        maxr = [1, 1, 2, 2, 2][settings_num]
    if TTstr is None:
        TTstr = [1, 1, 1, 2, 2][settings_num]
    if bwbh is None:
        bwbh = 16 if HD else 8
    if owoh is None:
        owoh = 8 if HD else 4
    if blksize is None:
        blksize = 16 if HD else 8
    if overlap is None:
        overlap = 8 if HD else 4
    if bt is None:
        bt = [1, 3, 3, 3, 4][settings_num]
    if thSAD is None:
        thSAD = [200, 300, 400, 500, 600][settings_num]
    if thSADC is None:
        thSADC = thSAD // 2
    if thSAD2 is None:
        thSAD2 = [200, 300, 400, 500, 600][settings_num]
    if thSADC2 is None:
        thSADC2 = thSAD2 // 2
    if thSCD1 is None:
        thSCD1 = [200, 300, 400, 500, 600][settings_num]
    if thSCD2 is None:
        thSCD2 = [90, 100, 100, 130, 130][settings_num]
    if pel is None:
        pel = [1, 2, 2, 2, 2][settings_num]
    if pelsearch is None:
        pelsearch = [1, 2, 2, 2, 2][settings_num]
    if MVsharp is None:
        MVsharp = [2, 2, 2, 1, 0][settings_num]

    sigma *= peak / 255
    limit = scale(limit, peak)
    limit2 = scale(limit2, peak)
    post *= peak / 255
    ECthr = scale(ECthr, peak)
    planes = [0, 1, 2] if chroma and not isGray else [0]

    ### INPUT
    mod = bwbh if bwbh >= blksize else blksize
    xi = i.width
    xf = math.ceil(xi / mod) * mod - xi + mod
    xf = xf + xf%4
    xn = int(xi + xf)
    yi = i.height
    yf = math.ceil(yi / mod) * mod - yi + mod
    yf = yf + yf%4
    yn = int(yi + yf)

    pointresize_args = dict(width=xn, height=yn, src_left=-xf / 2, src_top=-yf / 2, src_width=xn, src_height=yn)
    i = i.resize.Point(**pointresize_args)

    if cuda:
      try:
         dfttest2 = importlib.import_module('dfttest2')
      except ModuleNotFoundError:
        dfttest2 = None
    else:
      dfttest2 = None

    ### PREFILTERING
    fft3d_args = dict(planes=planes, bw=bwbh, bh=bwbh, bt=bt, ow=owoh, oh=owoh, ncpu=ncpu)
    if p is not None:
        p = p.resize.Point(**pointresize_args)
    elif pfMode <= -1:
        p = i
    elif pfMode == 0:
        if hasattr(core, 'neo_fft3d'):
            p = i.neo_fft3d.FFT3D(sigma=sigma * 0.8, sigma2=sigma * 0.6, sigma3=sigma * 0.4, sigma4=sigma * 0.2, **fft3d_args)
        else:                              
            p = i.fft3dfilter.FFT3DFilter(sigma=sigma * 0.8, sigma2=sigma * 0.6, sigma3=sigma * 0.4, sigma4=sigma * 0.2, **fft3d_args)
    elif pfMode >= 3:
        if dfttest2:
          p = dfttest2.DFTTest(i, tbsize=1, slocation=[0.0,4.0, 0.2,9.0, 1.0,15.0], planes=planes, backend=dfttest2.Backend.NVRTC)
        else:
          p = i.dfttest.DFTTest(tbsize=1, slocation=[0.0,4.0, 0.2,9.0, 1.0,15.0], planes=planes)
    else:
        p = MinBlur(i, r=pfMode, planes=planes)

    pD = core.std.MakeDiff(i, p, planes=planes)
    p = DitherLumaRebuild(p, s0=1, chroma=chroma)

    ### DEBLOCKING
    crop_args = dict(left=xf // 2, right=xf // 2, top=yf // 2, bottom=yf // 2)
    
    if not deblock:
        d = i
    elif useQED:
        d = Deblock_QED(i.std.Crop(**crop_args), quant1=quant1, quant2=quant2, uv=3 if chroma else 2).resize.Point(**pointresize_args)
    else:
        d = i.std.Crop(**crop_args).deblock.Deblock(quant=(quant1 + quant2) // 2, planes=planes).resize.Point(**pointresize_args)

    ### PREPARING
    super_args = dict(hpad=0, vpad=0, pel=pel, chroma=chroma, sharp=MVsharp)
    pMVS = p.mv.Super(rfilter=4 if refine else 2, **super_args)
    if refine:
        rMVS = p.mv.Super(levels=1, **super_args)

    analyse_args = dict(blksize=blksize, search=search, searchparam=searchparam, pelsearch=pelsearch, chroma=chroma, truemotion=truemotion, global_=MVglobal, overlap=overlap, dct=DCT)
    recalculate_args = dict(thsad=thSAD // 2, blksize=max(blksize // 2, 4), search=search, chroma=chroma, truemotion=truemotion, overlap=max(overlap // 2, 2), dct=DCT)
    f1v = pMVS.mv.Analyse(isb=False, delta=1, **analyse_args)
    b1v = pMVS.mv.Analyse(isb=True, delta=1, **analyse_args)
    if refine:
        f1v = core.mv.Recalculate(rMVS, f1v, **recalculate_args)
        b1v = core.mv.Recalculate(rMVS, b1v, **recalculate_args)
    if radius > 1:
        f2v = pMVS.mv.Analyse(isb=False, delta=2, **analyse_args)
        b2v = pMVS.mv.Analyse(isb=True, delta=2, **analyse_args)
        if refine:
            f2v = core.mv.Recalculate(rMVS, f2v, **recalculate_args)
            b2v = core.mv.Recalculate(rMVS, b2v, **recalculate_args)
    if radius > 2:
        f3v = pMVS.mv.Analyse(isb=False, delta=3, **analyse_args)
        b3v = pMVS.mv.Analyse(isb=True, delta=3, **analyse_args)
        if refine:
            f3v = core.mv.Recalculate(rMVS, f3v, **recalculate_args)
            b3v = core.mv.Recalculate(rMVS, b3v, **recalculate_args)
    if radius > 3:
        f4v = pMVS.mv.Analyse(isb=False, delta=4, **analyse_args)
        b4v = pMVS.mv.Analyse(isb=True, delta=4, **analyse_args)
        if refine:
            f4v = core.mv.Recalculate(rMVS, f4v, **recalculate_args)
            b4v = core.mv.Recalculate(rMVS, b4v, **recalculate_args)
    if radius > 4:
        f5v = pMVS.mv.Analyse(isb=False, delta=5, **analyse_args)
        b5v = pMVS.mv.Analyse(isb=True, delta=5, **analyse_args)
        if refine:
            f5v = core.mv.Recalculate(rMVS, f5v, **recalculate_args)
            b5v = core.mv.Recalculate(rMVS, b5v, **recalculate_args)
    if radius > 5:
        f6v = pMVS.mv.Analyse(isb=False, delta=6, **analyse_args)
        b6v = pMVS.mv.Analyse(isb=True, delta=6, **analyse_args)
        if refine:
            f6v = core.mv.Recalculate(rMVS, f6v, **recalculate_args)
            b6v = core.mv.Recalculate(rMVS, b6v, **recalculate_args)

    # if useTTmpSm or stabilize:
        # mask_args = dict(ml=thSAD, gamma=0.999, kind=1, ysc=255)
        # SAD_f1m = core.mv.Mask(d, f1v, **mask_args)
        # SAD_b1m = core.mv.Mask(d, b1v, **mask_args)

    def MCTD_MVD(i, iMVS, thSAD, thSADC):
        degrain_args = dict(thsad=thSAD, thsadc=thSADC, plane=4 if chroma else 0, thscd1=thSCD1, thscd2=thSCD2)
        if radius <= 1:
            sm = core.mv.Degrain1(i, iMVS, b1v, f1v, **degrain_args)
        elif radius == 2:
            sm = core.mv.Degrain2(i, iMVS, b1v, f1v, b2v, f2v, **degrain_args)
        elif radius == 3:
            sm = core.mv.Degrain3(i, iMVS, b1v, f1v, b2v, f2v, b3v, f3v, **degrain_args)
        elif radius == 4:
            mv12 = core.mv.Degrain2(i, iMVS, b1v, f1v, b2v, f2v, **degrain_args)
            mv34 = core.mv.Degrain2(i, iMVS, b3v, f3v, b4v, f4v, **degrain_args)
            sm = core.std.Merge(mv12, mv34, weight=[0.4444])
        elif radius == 5:
            mv123 = core.mv.Degrain3(i, iMVS, b1v, f1v, b2v, f2v, b3v, f3v, **degrain_args)
            mv45 = core.mv.Degrain2(i, iMVS, b4v, f4v, b5v, f5v, **degrain_args)
            sm = core.std.Merge(mv123, mv45, weight=[0.4545])
        else:
            mv123 = core.mv.Degrain3(i, iMVS, b1v, f1v, b2v, f2v, b3v, f3v, **degrain_args)
            mv456 = core.mv.Degrain3(i, iMVS, b4v, f4v, b5v, f5v, b6v, f6v, **degrain_args)
            sm = core.std.Merge(mv123, mv456, weight=[0.4615])

        return sm

    def MCTD_TTSM(i, iMVS, thSAD):
        compensate_args = dict(thsad=thSAD, thscd1=thSCD1, thscd2=thSCD2)
        f1c = core.mv.Compensate(i, iMVS, f1v, **compensate_args)
        b1c = core.mv.Compensate(i, iMVS, b1v, **compensate_args)
        if radius > 1:
            f2c = core.mv.Compensate(i, iMVS, f2v, **compensate_args)
            b2c = core.mv.Compensate(i, iMVS, b2v, **compensate_args)
            # SAD_f2m = core.mv.Mask(i, f2v, **mask_args)
            # SAD_b2m = core.mv.Mask(i, b2v, **mask_args)
        if radius > 2:
            f3c = core.mv.Compensate(i, iMVS, f3v, **compensate_args)
            b3c = core.mv.Compensate(i, iMVS, b3v, **compensate_args)
            # SAD_f3m = core.mv.Mask(i, f3v, **mask_args)
            # SAD_b3m = core.mv.Mask(i, b3v, **mask_args)
        if radius > 3:
            f4c = core.mv.Compensate(i, iMVS, f4v, **compensate_args)
            b4c = core.mv.Compensate(i, iMVS, b4v, **compensate_args)
            # SAD_f4m = core.mv.Mask(i, f4v, **mask_args)
            # SAD_b4m = core.mv.Mask(i, b4v, **mask_args)
        if radius > 4:
            f5c = core.mv.Compensate(i, iMVS, f5v, **compensate_args)
            b5c = core.mv.Compensate(i, iMVS, b5v, **compensate_args)
            # SAD_f5m = core.mv.Mask(i, f5v, **mask_args)
            # SAD_b5m = core.mv.Mask(i, b5v, **mask_args)
        if radius > 5:
            f6c = core.mv.Compensate(i, iMVS, f6v, **compensate_args)
            b6c = core.mv.Compensate(i, iMVS, b6v, **compensate_args)
            # SAD_f6m = core.mv.Mask(i, f6v, **mask_args)
            # SAD_b6m = core.mv.Mask(i, b6v, **mask_args)

        # b = i.std.BlankClip(color=[0] if isGray else [0, neutral, neutral])
        if radius <= 1:
            c = core.std.Interleave([f1c, i, b1c])
            # SAD_m = core.std.Interleave([SAD_f1m, b, SAD_b1m])
        elif radius == 2:
            c = core.std.Interleave([f2c, f1c, i, b1c, b2c])
            # SAD_m = core.std.Interleave([SAD_f2m, SAD_f1m, b, SAD_b1m, SAD_b2m])
        elif radius == 3:
            c = core.std.Interleave([f3c, f2c, f1c, i, b1c, b2c, b3c])
            # SAD_m = core.std.Interleave([SAD_f3m, SAD_f2m, SAD_f1m, b, SAD_b1m, SAD_b2m, SAD_b3m])
        elif radius == 4:
            c = core.std.Interleave([f4c, f3c, f2c, f1c, i, b1c, b2c, b3c, b4c])
            # SAD_m = core.std.Interleave([SAD_f4m, SAD_f3m, SAD_f2m, SAD_f1m, b, SAD_b1m, SAD_b2m, SAD_b3m, SAD_b4m])
        elif radius == 5:
            c = core.std.Interleave([f5c, f4c, f3c, f2c, f1c, i, b1c, b2c, b3c, b4c, b5c])
            # SAD_m = core.std.Interleave([SAD_f5m, SAD_f4m, SAD_f3m, SAD_f2m, SAD_f1m, b, SAD_b1m, SAD_b2m, SAD_b3m, SAD_b4m, SAD_b5m])
        else:
            c = core.std.Interleave([f6c, f5c, f4c, f3c, f2c, f1c, i, b1c, b2c, b3c, b4c, b5c, b6c])
            # SAD_m = core.std.Interleave([SAD_f6m, SAD_f5m, SAD_f4m, SAD_f3m, SAD_f2m, SAD_f1m, b, SAD_b1m, SAD_b2m, SAD_b3m, SAD_b4m, SAD_b5m, SAD_b6m])

        # sm = c.ttmpsm.TTempSmooth(maxr=radius, thresh=[255], mdiff=[1], strength=radius + 1, scthresh=99.9, fp=False, pfclip=SAD_m, planes=planes)
        sm = c.ttmpsm.TTempSmooth(maxr=radius, thresh=[255], mdiff=[1], strength=radius + 1, scthresh=99.9, fp=False, planes=planes)
        return sm.std.SelectEvery(cycle=radius * 2 + 1, offsets=[radius])

    ### DENOISING: FIRST PASS
    dMVS = d.mv.Super(levels=1, **super_args)
    sm = MCTD_TTSM(d, dMVS, thSAD) if useTTmpSm else MCTD_MVD(d, dMVS, thSAD, thSADC)

    if limit <= -1:
        smD = core.std.MakeDiff(i, sm, planes=planes)
        expr = f'x {neutral} - abs y {neutral} - abs < x y ?'
        DD = core.std.Expr([pD, smD], expr=[expr] if chroma or isGray else [expr, ''])
        smL = core.std.MakeDiff(i, DD, planes=planes)
    elif limit > 0:
        expr = f'x y - abs {limit} <= x x y - 0 < y {limit} - y {limit} + ? ?'
        smL = core.std.Expr([sm, i], expr=[expr] if chroma or isGray else [expr, ''])
    else:
        smL = sm

    ### DENOISING: SECOND PASS
    if twopass:
        smLMVS = smL.mv.Super(levels=1, **super_args)
        sm = MCTD_TTSM(smL, smLMVS, thSAD2) if useTTmpSm else MCTD_MVD(smL, smLMVS, thSAD2, thSADC2)

        if limit2 <= -1:
            smD = core.std.MakeDiff(i, sm, planes=planes)
            expr = f'x {neutral} - abs y {neutral} - abs < x y ?'
            DD = core.std.Expr([pD, smD], expr=[expr] if chroma or isGray else [expr, ''])
            smL = core.std.MakeDiff(i, DD, planes=planes)
        elif limit2 > 0:
            expr = f'x y - abs {limit2} <= x x y - 0 < y {limit2} - y {limit2} + ? ?'
            smL = core.std.Expr([sm, i], expr=[expr] if chroma or isGray else [expr, ''])
        else:
            smL = sm

    ### POST-DENOISING: FFT3D
    if post <= 0:
        smP = smL
    else:
        if hasattr(core, 'neo_fft3d'):
            smP = smL.neo_fft3d.FFT3D(sigma=post * 0.8, sigma2=post * 0.6, sigma3=post * 0.4, sigma4=post * 0.2, **fft3d_args)
        else:                              
            smP = smL.fft3dfilter.FFT3DFilter(sigma=post * 0.8, sigma2=post * 0.6, sigma3=post * 0.4, sigma4=post * 0.2, **fft3d_args)

    ### EDGECLEANING
    if edgeclean:
        mP = AvsPrewitt(GetPlane(smP, 0))
        mS = mt_expand_multi(mP, sw=ECrad, sh=ECrad).std.Inflate()
        mD = core.std.Expr([mS, mP.std.Inflate()], expr=[f'x y - {ECthr} <= 0 x y - ?']).std.Inflate().std.Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])
        if dfttest2:
          smP = core.std.MaskedMerge(smP, DeHalo_alpha(dfttest2.DFTTest(smP, tbsize=1, planes=planes), darkstr=0), mD, planes=planes, backend=dfttest2.Backend.NVRTC)
        else:
          smP = core.std.MaskedMerge(smP, DeHalo_alpha(smP.dfttest.DFTTest(tbsize=1, planes=planes), darkstr=0), mD, planes=planes)
    ### STABILIZING
    if stabilize:
        # mM = core.std.Merge(GetPlane(SAD_f1m, 0), GetPlane(SAD_b1m, 0)).std.Lut(function=lambda x: min(cround(x ** 1.6), peak))
        mE = AvsPrewitt(GetPlane(smP, 0)).std.Lut(function=lambda x: min(cround(x ** 1.8), peak)).std.Median().std.Inflate()
        # mF = core.std.Expr([mM, mE], expr=['x y max']).std.Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])
        mF = mE.std.Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])
        TTc = smP.ttmpsm.TTempSmooth(maxr=maxr, mdiff=[255], strength=TTstr, planes=planes)
        smP = core.std.MaskedMerge(TTc, smP, mF, planes=planes)

    ### OUTPUT
    return smP.std.Crop(**crop_args)

def cround(x: float) -> int:
    return math.floor(x + 0.5) if x > 0 else math.ceil(x - 0.5)

def scale(value, peak):
    return cround(value * peak / 255) if peak != 1 else value / 255
    
def DitherLumaRebuild(src: vs.VideoNode, s0: float = 2.0, c: float = 0.0625, chroma: bool = True) -> vs.VideoNode:
    '''Converts luma (and chroma) to PC levels, and optionally allows tweaking for pumping up the darks. (for the clip to be fed to motion search only)'''
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('DitherLumaRebuild: this is not a clip')

    if src.format.color_family == vs.RGB:
        raise vs.Error('DitherLumaRebuild: RGB format is not supported')

    is_gray = src.format.color_family == vs.GRAY
    is_integer = src.format.sample_type == vs.INTEGER

    bits = get_depth(src)
    neutral = 1 << (bits - 1)

    k = (s0 - 1) * c
    t = f'x {scale_value(16, 8, bits)} - {scale_value(219, 8, bits)} / 0 max 1 min' if is_integer else 'x 0 max 1 min'
    e = f'{k} {1 + c} {(1 + c) * c} {t} {c} + / - * {t} 1 {k} - * + ' + (f'{scale_value(256, 8, bits)} *' if is_integer else '')
    return src.std.Expr(expr=e if is_gray else [e, f'x {neutral} - 128 * 112 / {neutral} +' if chroma and is_integer else ''])
    
def AvsPrewitt(clip: vs.VideoNode, planes: Optional[Union[int, Sequence[int]]] = None) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('AvsPrewitt: this is not a clip')

    plane_range = range(clip.format.num_planes)

    if planes is None:
        planes = list(plane_range)
    elif isinstance(planes, int):
        planes = [planes]

    return core.std.Expr(
        [
            clip.std.Convolution(matrix=[1, 1, 0, 1, 0, -1, 0, -1, -1], planes=planes, saturate=False),
            clip.std.Convolution(matrix=[1, 1, 1, 0, 0, 0, -1, -1, -1], planes=planes, saturate=False),
            clip.std.Convolution(matrix=[1, 0, -1, 1, 0, -1, 1, 0, -1], planes=planes, saturate=False),
            clip.std.Convolution(matrix=[0, -1, -1, 1, 0, -1, 1, 1, 0], planes=planes, saturate=False),
        ],
        expr=['x y max z max a max' if i in planes else '' for i in plane_range],
    )

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