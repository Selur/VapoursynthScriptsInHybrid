
import vapoursynth as vs
from vapoursynth import core

import math

from typing import Sequence, Union, Optional

from vsutil import scale_value

def STPresso(
    clp: vs.VideoNode,
    limit: int = 3,
    bias: int = 24,
    RGmode: Union[int, vs.VideoNode] = 4,
    tthr: int = 12,
    tlimit: int = 3,
    tbias: int = 49,
    back: int = 1,
    planes: Optional[Union[int, Sequence[int]]] = None,
) -> vs.VideoNode:
    """
    Dampen the grain just a little, to keep the original look.

    Parameters:
        clp: Clip to process.

        limit: The spatial part won't change a pixel more than this.

        bias: The percentage of the spatial filter that will apply.

        RGmode: The spatial filter is RemoveGrain, this is its mode. It also accepts loading your personal prefiltered clip.

        tthr: Temporal threshold for fluxsmooth. Can be set "a good bit bigger" than usually.

        tlimit: The temporal filter won't change a pixel more than this.

        tbias: The percentage of the temporal filter that will apply.

        back: After all changes have been calculated, reduce all pixel changes by this value. (shift "back" towards original value)

        planes: Specifies which planes will be processed. Any unprocessed planes will be simply copied.
    """
    if not isinstance(clp, vs.VideoNode):
        raise vs.Error('STPresso: this is not a clip')

    plane_range = range(clp.format.num_planes)

    if planes is None:
        planes = list(plane_range)
    elif isinstance(planes, int):
        planes = [planes]

    bits = clp.format.bits_per_sample
    limit = scale_value(limit, 8, bits)
    tthr = scale_value(tthr, 8, bits)
    tlimit = scale_value(tlimit, 8, bits)
    back = scale_value(back, 8, bits)

    LIM = cround(limit * 100 / bias - 1) if limit > 0 else cround(scale_value(100 / bias, 8, bits))
    TLIM = cround(tlimit * 100 / tbias - 1) if tlimit > 0 else cround(scale_value(100 / tbias, 8, bits))

    if limit < 0:
        expr = f'x y - abs {LIM} < x x 1 x y - dup abs / * - ?'
    else:
        expr = f'x y - abs {scale_value(1, 8, bits)} < x x {LIM} + y < x {limit} + x {LIM} - y > x {limit} - x {100 - bias} * y {bias} * + 100 / ? ? ?'
    if tlimit < 0:
        texpr = f'x y - abs {TLIM} < x x 1 x y - dup abs / * - ?'
    else:
        texpr = f'x y - abs {scale_value(1, 8, bits)} < x x {TLIM} + y < x {tlimit} + x {TLIM} - y > x {tlimit} - x {100 - tbias} * y {tbias} * + 100 / ? ? ?'

    if isinstance(RGmode, vs.VideoNode):
        bzz = RGmode
    else:
        if RGmode == 4:
            bzz = clp.std.Median(planes=planes)
        elif RGmode in [11, 12]:
            bzz = clp.std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1], planes=planes)
        elif RGmode == 19:
            bzz = clp.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1], planes=planes)
        elif RGmode == 20:
            bzz = clp.std.Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1], planes=planes)
        else:
            RG = core.zsmooth.RemoveGrain if hasattr(core,'zsmooth') else core.rgvs.RemoveGrain
            bzz = RG(clp, mode=RGmode)
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    last = EXPR([clp, bzz], expr=[expr if i in planes else '' for i in plane_range])

    if tthr > 0:
        analyse_args = dict(truemotion=False, delta=1, blksize=16, overlap=8)

        mvSuper = bzz.mv.Super(sharp=1)
        bv1 = mvSuper.mv.Analyse(isb=True, **analyse_args)
        fv1 = mvSuper.mv.Analyse(isb=False, **analyse_args)
        bc1 = core.mv.Compensate(bzz, mvSuper, bv1)
        fc1 = core.mv.Compensate(bzz, mvSuper, fv1)

        interleave = core.std.Interleave([fc1, bzz, bc1])
        #FX = core.zsmooth.FluxSmoothT if hasattr(core,'zsmooth') else core.flux.SmoothT
        FX = core.flux.SmoothT
        smooth = FX(interleave, temporal_threshold=tthr, planes=planes)
        smooth = smooth.std.SelectEvery(cycle=3, offsets=1)

        diff = core.std.MakeDiff(bzz, smooth, planes=planes)
        diff = core.std.MakeDiff(last, diff, planes=planes)
        last = EXPR([last, diff], expr=[texpr if i in planes else '' for i in plane_range])

    if back > 0:
        expr = f'x {back} + y < x {back} + x {back} - y > x {back} - y ? ?'
        last = EXPR([last, clp], expr=[expr if i in planes else '' for i in plane_range])

    return last
    
########################
#
# Ported version of Temporal Degrain from avisynth
#
# Function by Sagekilla, idea + original script created by Didee
# ported by Hinterwaeldlers
#
# Works as a simple temporal degraining function that'll remove MOST grain
# from video sources, including dancing grain, like the grain found on 300.
# Also note, the parameters don't need to be tweaked much.
#
# Required vapoursynth plugins:
# * FFT3DFilter, if no pre calmed clip is given
# * MVTools
# * hqdn3d
#
# Note, the name of the parameters have been slightly changed from the avisynth
# variant, to hopefully clarify their meaning. In addition the GPU parameter is
# dropped, as there is no FFT3DFilter GPU variant.
#
# Usage:
# * inpClip:       The clip to process
# * denoiseClip:   An optional, pre calmed clip. This one should be "dead calm".
#                  If none is given, a clip is generated with FFT3DFilter
# * sigma:         FFT3DFilter filter strength
#                  Not used if a denoised clip is given
# * blockWidth:    FFT3DFilter block width
# * blockHeigth:   FFT3DFilter block width
# * sigma2:        FFT3DFilter sigma2 parameter
#                  Not used if a denoised clip is given
# * sigma3:        FFT3DFilter sigma3 parameter
#                  Not used if a denoised clip is given
# * sigma4:        FFT3DFilter sigma4 parameter
#                  Not used if a denoised clip is given
# * overlapWidth:  FFT3DFilter overlap width parameter
#                  Not used if a denoised clip is given
# * overlapHeight: FFT3DFilter overlap height parameter
#                  Not used if a denoised clip is given
# * blockSize:     MVTools Analyse block size
# * pel:           MVTools Analyse subpixel accurancy
# * overlapValue:  MVTools Analyse overlap parameter
# * degrain:       Amount of MVTools degrain vectors used.
#                  Valid values are: 1, 2 and 3
# * maxPxChange:   Limit the maximum allowed pixel change
# * thrDegrain1:   MVTools degrain SAD threshold in stage 1
# * thrDegrain2:   MVTools degrain SAD threshold in stage 2
# * HQ:            Adjust the filtering level:
#                  * 0: disable any prefiltering (not recommneded)
#                  * 1: extra prefiltering step
#                  * 2: extra pre- and postfiltering step

def TemporalDegrain(          \
      inpClip                 \
    , denoiseClip   = None    \
    , sigma         = 16      \
    , blockWidth    = 16      \
    , blockHeight   = 16      \
    , sigma2        = None    \
    , sigma3        = None    \
    , sigma4        = None    \
    , overlapWidth  = None    \
    , overlapHeight = None    \
    , blockSize     = 16      \
    , pel           = 2       \
    , overlapValue  = None    \
    , degrain       = 2       \
    , maxPxChange   = 255     \
    , thrDegrain1   = 400     \
    , thrDegrain2   = 300     \
    , HQ            = 1       \
) :

    if int(degrain) != degrain or degrain < 1 or degrain > 3:
        raise SyntaxError(\
            "Invalid degrain paramter! Must be 1, 2 or 3 (given as int)")

    neutral = 1 << (inpClip.format.bits_per_sample - 1)

    # Set the default value of not given values
    if sigma2 is None:
        sigma2 = sigma * 0.625
    if sigma3 is None:
        sigma3 = sigma * 0.375
    if sigma4 is None:
        sigma4 = sigma * 0.250
    if overlapWidth is None:
        overlapWidth = blockWidth // 2
    if overlapHeight is None:
        overlapHeight = blockHeight // 2
    if overlapValue is None:
        overlapValue = blockSize // 2

    # Input adjustments
    sigma2 = math.floor(sigma2)
    sigma3 = math.floor(sigma3)
    sigma4 = math.floor(sigma4)
    if overlapValue * 2 > blockSize:
        overlapValue = blockSize // 2

    # Taking care of a missing denoising clip and use of fft3d to determine it
    if denoiseClip is None:
        if hasattr(core, 'neo_fft3d'):
          denoiseClip = inpClip.neo_fft3d.FFT3D(sigma=sigma\
              , sigma2=sigma2, sigma3=sigma3, sigma4=sigma4, bw=blockWidth\
              , bh=blockHeight, ow=overlapWidth, oh=overlapHeight)
        else:                                                    
          denoiseClip = inpClip.fft3dfilter.FFT3DFilter(sigma=sigma\
              , sigma2=sigma2, sigma3=sigma3, sigma4=sigma4, bw=blockWidth\
              , bh=blockHeight, ow=overlapWidth, oh=overlapHeight)

    # If HQ is activated, do an additional denoising
    if HQ > 0:
        filterClip = denoiseClip.hqdn3d.Hqdn3d(4,3,6,3)
    else:
        filterClip = denoiseClip

    # "spat" is a prefiltered clip which is used to limit the effect of the 1st
    # MV-denoise stage. For simplicity, we just use the same FFT3DFilter.
    # There's lots of other possibilities.
    spatD = core.std.MakeDiff(inpClip, filterClip)

    # Motion vector search (With very basic parameters. Add your own parameters
    # as needed.)
    srchSuper = filterClip.mv.Super(pel=pel)

    if degrain == 3:
        bvec3 = srchSuper.mv.Analyse(isb=True, delta=3, blksize=blockSize\
            , overlap=overlapValue)
        fvec3 = srchSuper.mv.Analyse(isb=False, delta=3, blksize=blockSize\
            , overlap=overlapValue)

    if degrain >= 2:
        bvec2 = srchSuper.mv.Analyse(isb=True, delta=2, blksize=blockSize\
            , overlap=overlapValue)
        fvec2 = srchSuper.mv.Analyse(isb=False, delta=2, blksize=blockSize\
            , overlap=overlapValue)

    bvec1 = srchSuper.mv.Analyse(isb=True, delta=1, blksize=blockSize\
        , overlap=overlapValue)
    fvec1 = srchSuper.mv.Analyse(isb=False, delta=1, blksize=blockSize\
        , overlap=overlapValue)

    # First MV-denoising stage. Usually here's some temporal-medianfiltering
    # going on. For simplicity, we just use MVDegrain.
    inpSuper = inpClip.mv.Super(pel=2, levels=1)
    if degrain == 3:
        nr1 = core.mv.Degrain3(inpClip, inpSuper, bvec1, fvec1, bvec2, fvec2\
            , bvec3, fvec3, thsad=thrDegrain1, limit=maxPxChange)
    elif degrain == 2:
        nr1 = core.mv.Degrain2(inpClip, inpSuper, bvec1, fvec1, bvec2, fvec2\
            , thsad=thrDegrain1, limit=maxPxChange)
    else:
        nr1 = core.mv.Degrain1(inpClip, inpSuper, bvec1, fvec1\
            , thsad=thrDegrain1, limit=maxPxChange)
    nr1Diff = core.std.MakeDiff(inpClip, nr1)

    # Limit NR1 to not do more than what "spat" would do.
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    dd = EXPR([spatD, nr1Diff], expr=[f'x {neutral} - abs y {neutral} - abs < x y ?'])
    nr1X = core.std.MakeDiff(inpClip, dd, planes=0)

    # Second MV-denoising stage
    nr1x_super = nr1X.mv.Super(pel=2, levels=1)

    if degrain == 3:
        nr2 = core.mv.Degrain3(nr1X, nr1x_super, bvec1, fvec1, bvec2, fvec2\
            , bvec3, fvec3, thsad=thrDegrain2, limit=maxPxChange)
    elif degrain == 2:
        nr2 = core.mv.Degrain2(nr1X, nr1x_super, bvec1, fvec1, bvec2, fvec2\
            , thsad=thrDegrain2, limit=maxPxChange)
    else:
        nr2 = core.mv.Degrain1(nr1X, nr1x_super, bvec1, fvec1\
            , thsad=thrDegrain2, limit=maxPxChange)

    # Temporal filter to remove the last bits of dancinc pixels, YMMV.
    if HQ >= 2:
        nr2 = nr2.hqdn3d.Hqdn3d(0,0,4,1)

    # Contra-sharpening: sharpen the denoised clip, but don't add more than
    # what was removed previously.
    # Here: A simple area-based version with relaxed restriction. The full
    # version is more complicated.
    return ContraSharpening(nr2, inpClip)

def MLDegrain(clip, scale1=1.5, scale2=2, thSAD=400, tr=3, rec=False, chroma=True, soft=[0]*3):
    """
    Multi-Level MDegrain
    Multi level in the sense of using multiple scalings.
    The observation was that when downscaling the source to a smallish resolution, then a vanilla MDegrain can produce a very stable result. 
    Hence, it's an obvious approach to first make a small-but-stable-denoised clip, and then work the way upwards to the original resolution.
    From: https://forum.doom9.org/showthread.php?p=1512413 by Didée

    Args:
        scale1 (float) - Scaling factor between original and medium scale
        scale2 (float) - Scaling factor between medium and small scale
        tr     (int)   - Strength of the denoising (1-24).
        thSAD  (int)   - Soft threshold of block sum absolute differences.
                         Low value can result in staggered denoising,
                         High value can result in ghosting and artifacts.
        rec    (bool)  - Recalculate the motion vectors to obtain more precision.
        chroma (bool)  - Whether to process chroma.
        soft (float[]) - [small, medium, original] ranges from 0 to 1, 0 means disabled, 1 means 100% strength.
                         Do slight sharpening where motionmatch is good, do slight blurring where motionmatch is bad.
    """

    isFLOAT = clip.format.sample_type == vs.FLOAT

    if not isinstance(clip, vs.VideoNode) or clip.format.color_family not in [vs.GRAY, vs.YUV]:
        raise TypeError('MLDegrain: This is not a GRAY or YUV clip!')
    
    if tr > 3 and not isFLOAT:
        raise TypeError("MLDegrain: DegrainN is only available in float")

    w = clip.width
    h = clip.height
    w1 = m4(w / scale1)
    h1 = m4(h / scale1)
    w2 = m4(w1 / scale2)
    h2 = m4(h1 / scale2)
    sm1 = clip.resize.Bicubic(w1, h1) # medium scale
    sm2 = sm1.resize.Bicubic(w2, h2)  # small scale
    D12 = core.std.MakeDiff(sm2.resize.Bicubic(w1, h1), sm1) # residual of (small)<>(medium)
    D10 = core.std.MakeDiff(sm1.resize.Bicubic(w, h), clip)  # residual of (medium)<>(original)
    lev2 = MLD_helper(sm2, sm2, tr, thSAD, rec, chroma, soft[0]) # Filter on smalle scale
    up1 = lev2.resize.Bicubic(w1, h1)
    up2 = up1.resize.Bicubic(w, h)
    M1 = MLD_helper(D12, up1, tr, thSAD, rec, chroma, soft[1])   # Filter on medium scale
    lev1 = core.std.MakeDiff(up1, M1)
    up3 = lev1.resize.Bicubic(w, h)
    M2 = MLD_helper(D10, up2, tr, thSAD, rec, chroma, soft[2])   # Filter on original scale

    return core.std.MakeDiff(up3, M2)


def MLD_helper(clip, srch, tr, thSAD, rec, chroma, soft):
    """ Helper function used in Multi-Level MDegrain"""

    if not isinstance(clip, vs.VideoNode) or clip.format.color_family not in [vs.GRAY, vs.YUV]:
        raise TypeError('MLD_helper: This is not a GRAY or YUV clip!')
    
    isFLOAT = clip.format.sample_type == vs.FLOAT
    isGRAY = clip.format.color_family == vs.GRAY
    S = core.mvsf.Super if isFLOAT else core.mv.Super
    A = core.mvsf.Analyse if isFLOAT else core.mv.Analyse
    R = core.mvsf.Recalculate if isFLOAT else core.mv.Recalculate
    D1 = core.mvsf.Degrain1 if isFLOAT else core.mv.Degrain1
    D2 = core.mvsf.Degrain2 if isFLOAT else core.mv.Degrain2
    D3 = core.mvsf.Degrain3 if isFLOAT else core.mv.Degrain3
    bs = 32 if clip.width > 2400 else 16 if clip.width > 960 else 8
    pel = 1 if clip.width > 960 else 2
    truemotion = False if clip.width > 960 else True
    chroma = False if isGRAY else chroma
    planes = [0, 1, 2] if chroma else [0]
    plane = 4 if chroma else 0

    analyse_args = dict(blksize=bs, overlap=bs//2, search=5, chroma=chroma, truemotion=truemotion)
    recalculate_args = dict(blksize=bs//2, overlap=bs//4, search=5, chroma=chroma, truemotion=truemotion)
    sup1 = S(DitherLumaRebuild(srch, 1), hpad=bs, vpad=bs, pel=pel, sharp=1, rfilter=4)

    if soft > 0:
        if clip.width > 1280:
            RG = core.std.Convolution(clip, matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1], planes=planes)
        elif clip.width > 640:
            RG = core.std.Convolution(clip, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1], planes=planes)
        else:
            RG = MinBlur(clip, 1, planes)
        RG = core.std.Merge(clip, RG, [soft] if chroma or isGRAY else [soft, 0]) if soft < 1 else RG
        EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
        sup2 = S(EXPR([clip, RG], ['x dup y - +'] if chroma or isGRAY else ['x dup y - +', '']), hpad=bs, vpad=bs, pel=pel, levels=1, rfilter=1)
    else:
        RG = clip
        sup2 = S(clip, hpad=bs, vpad=bs, pel=pel, levels=1, rfilter=1)

    if tr < 4:
        bv1 = A(sup1, isb=True,  delta=1, **analyse_args)
        fv1 = A(sup1, isb=False, delta=1, **analyse_args)
        if tr > 1:
            bv2 = A(sup1, isb=True,  delta=2, **analyse_args)
            fv2 = A(sup1, isb=False, delta=2, **analyse_args)
        if tr > 2:
            bv3 = A(sup1, isb=True,  delta=3, **analyse_args)
            fv3 = A(sup1, isb=False, delta=3, **analyse_args)
    else:
        vec = mvmulti.Analyze(sup1, tr=tr, **analyse_args)
        
    if rec:
        if tr < 4:
            bv1 = R(sup1, bv1, **recalculate_args)
            fv1 = R(sup1, fv1, **recalculate_args)
            if tr > 1:
                bv2 = R(sup1, bv2, **recalculate_args)
                fv2 = R(sup1, fv2, **recalculate_args)
            if tr > 2:
                bv3 = R(sup1, bv3, **recalculate_args)
                fv3 = R(sup1, fv3, **recalculate_args)    
        else:
            vec = mvmulti.Recalculate(sup1, vec, tr=tr, **recalculate_args)
    
    if tr < 4:
        if tr == 1:
            return D1(RG, sup2, bv1, fv1, thsad=thSAD, plane=plane)
        elif tr == 2:
            return D2(RG, sup2, bv1, fv1, bv2, fv2, thsad=thSAD, plane=plane)
        else:
            return D3(RG, sup2, bv1, fv1, bv2, fv2, bv3, fv3, thsad=thSAD, plane=plane)
    else:
        return mvmulti.DegrainN(RG, sup2, vec, tr=tr, thsad=thSAD, plane=plane)

def TemporalDegrain2(clip, degrainTR=1, degrainPlane=4, grainLevel=2, grainLevelSetup=False, meAlg=4, meAlgPar=None, meSubpel=None, meBlksz=None, meTM=False,
    limitSigma=None, limitBlksz=None, fftThreads=None, postFFT=0, postTR=1, postSigma=1, postMix=0, postBlkSize=None, knlDevId=0, ppSAD1=None, ppSAD2=None, 
    ppSCD1=None, thSCD2=128, DCT=0, SubPelInterp=2, SrchClipPP=None, GlobalMotion=True, ChromaMotion=True, rec=False, extraSharp=False, outputStage=2, neo=True):
    """
    Temporal Degrain Updated by ErazorTT                               
                                                                          
    Based on function by Sagekilla, idea + original script created by Didee
    Works as a simple temporal degraining function that'll remove             
    MOST or even ALL grain and noise from video sources,                      
    including dancing grain, like the grain found on 300.                     
    Also note, the parameters don't need to be tweaked much.                  
                                                                           
    Required plugins:                                                         
    FFT3DFilter: https://github.com/myrsloik/VapourSynth-FFT3DFilter   
    MVtools(sf): https://github.com/dubhater/vapoursynth-mvtools (https://github.com/IFeelBloated/vapoursynth-mvtools-sf)                   
                                                                           
    Optional plugins:                                                         
    dfttest: https://github.com/HomeOfVapourSynthEvolution/VapourSynth-DFTTest            
    KNLMeansCL: https://github.com/Khanattila/KNLMeansCL

    recommendations to be followed for each new movie:
      1. start with default settings
      2. if less denoising is needed set grainLevel to 0, if you need more degraining start over reading at next paragraph
      3. if you need even less denoising:
         - EITHER: set outputStage to 1 or even 0 (faster)
         - OR: use the postMix setting and increase the value from 0 to as much as 100 (slower)

    recommendations for strong degraining: 
      1. start with default settings
      2. search the noisiest* patch of the entire movie, enable grainLevelSetup (=true), zoom in as much as you can and prepare yourself for pixel peeping. (*it really MUST be the noisiest region where you want this filter to be effective)
      3. compare the output on this noisy* patch of your movie with different settings of grainLevel (0 to 3) and use the setting where the noise level is lowest (irrespectable of whether you find this to be too much filtering).
         If multiple grainLevel settings yield comparable results while grainLevelSetup=true and observing at maximal zoom be sure to use the lowest setting! If you're unsure leave it at the default (2), your result might no be optimal, but it will still be great.
      4. disable grainLevelSetup (=false), or just remove this argument from the function call. Now revert the zoom and look from a normal distance at different scenes of the movie and decide if you like what you see.
      5. if more denoising is needed try postFFT=1 with postSigma=1, then tune postSigma (obvious blocking and banding of regions in the sky are indications of a value which is at least a factor 2 too high)
      6. if you would need a postSigma of more than 2, try first to increase degrainTR to 2. The goal is to balance the numerical values of postSigma and degrainTR, some prefer more simga and others more TR, it's up to you. However, do not increase degrainTR above 1/8th of the fps (at 24fps up to 3).
      7. if you cranked up postSigma higher than 3 then try postFFT=3 instead. Also if there are any issues with banding then try postFFT=3.
      8. if the luma is clean but you still have visible chroma noise then you can adjust postSigmaC which will separately clean the chroma planes (at a considerable amount of processing speed).

    use only the following knobs (all other settings should already be where they need to be):
      - degrainTR (1), temporal radius of degrain, usefull range: min=default=1, max=fps/8. Higher values do clean the video more, but also increase probability of wrongly identified motion vectors which leads to washed out regions
      - grainLevel (2), if input noise level is relatively low set this to 0, if its unusually high you might need to increase it to 3. The right setting must be found using grainLevelSetup=true while all other settings are at default. Set this setting such that the noise level is lowest.
      - grainLevelSetup (false), only to be used while finding the right setting for grainLevel. This will skip all your other settings!
      - postFFT (0), if you want to remove absolutely all remaining noise suggestion is to use 1 or 2 (ff3dfilter) or for slightly higher quality at the expense of potentially worse speed 3 (dfttest). 4 is KNLMeansCL. 0 is simply RemoveGrain(1)
      - postSigma (1), increase it to remove all the remaining noise you want removed, but do not increase too much since unnecessary high values have severe negative impact on either banding and/or sharpness
      - degrainPlane (4), if you just want to denoise only the chroma use 3 (this helps with compressability while the clip is almost identical to the original)
      - outputStage (2), if the degraining is too strong, you can output earlier stages
      - postMix (0), if the degraining is too strong, increase the value going from 0 to 100
      - fftThreads (1), usefull if you have processor cores to spare, increasing to 2 will probably help a little with speed.
      - rec (false), enables use of Recalculate for refining motion analysis. Enable for higher quality motion estimation but lower performance.
    
    Changelog
    December 21, 2022 - based on v2.6.3 of Avisynth version (Adub/adworacz):
        Feature:
        - Add support for grainLevel and grainLevelSetup, with associated
          autotunning, in alignment with AVS version.
        - Add postMix support, for mixing in grain again, in alignment with AVS
          version.
        - Add outputStage support, which enables outputting a clip at different
          stages of the denoising process, in alignment with AVS version.
        - Update documentation based on AVS version.
        - Add support for postBlkSize, in alignment with AVS version.
        - all ppSAD* and ppSCD* variables are autotuned based on grain level.

        Changes:
        - Set meAlg default to 4 to match the AVS version, should be a nice
          speed improvement for a small drop in quality.
        - Set degrainTR default to 1 to match the AVS version. Better aligns
          with use of new grainLevel arg as well.
        - Tune meAlgPar to match Dogway's latest AVS SMDegrain settings.
        - Adjust MSuper settings to better align with AVS version. More work required.
        - Minor code cleanup.
        - Renamed internal 'i' variable to `bitDepthMultiplier`

        Bug Fixes:
        - Fix hpad/vpad bug, which wasn't taking blocksize into effect before.
        - Fix bug with postFFT 4, KNLMeansCL args.
        - Removed weird bitdepth scaling of sigma values using internal `i`
          variable, which caused crazy sigma values to be used. Sigma is
          bitdepth indepenent, so don't know why it was ever used.

    December 22, 2022 (Adub/adworacz): 
        - Removed internal LSAD and PLevel params, as they simply restated the existing MVtools defaults.
        - Fixed `rec` behavior with properly tuned recalculate args and a dedicated super clip with levels=1.

    TODO:
        - Add support for BM3D (CPU/CUDA), dfttest2 (CPU/CUDA)
        - Investigate usage of rfilter=4. AVS version doesn't have it. MCTemporalDenoise uses 4 if refine else 2, SMDegrain uses 3 or default (2).
    """

    if not isinstance(clip, vs.VideoNode) or clip.format.color_family not in [vs.GRAY, vs.YUV]:
        raise TypeError("TemporalDegrain2: This is not a GRAY or YUV clip!")
    
    w = clip.width
    h = clip.height
    bd = clip.format.bits_per_sample
    isFLOAT = clip.format.sample_type == vs.FLOAT
    isGRAY = clip.format.color_family == vs.GRAY
    # Seems to be used for some kind of bit depth scaling...
    bitDepthMultiplier = 0.00392 if isFLOAT else 1 << (bd - 8)
    mid = 0.5 if isFLOAT else 1 << (bd - 1)
    if hasattr(core, 'mvsf') and isFLOAT:  
      S = core.mvsf.Super
      A = core.mvsf.Analyze
      C = core.mvsf.Compensate
      R = core.mvsf.Recalculate
      D1 = core.mvsf.Degrain1
      D2 = core.mvsf.Degrain2
      D3 = core.mvsf.Degrain3
    else:
      S = core.mv.Super
      A = core.mv.Analyse
      C = core.mv.Compensate
      R = core.mv.Recalculate
      D1 = core.mv.Degrain1
      D2 = core.mv.Degrain2
      D3 = core.mv.Degrain3
    
    if hasattr(core, 'zsmooth'):
      RG = core.zsmooth.RemoveGrain
    elif hasattr(core, 'rgsf') and isFLOAT:  
      RG = core.rgsf.RemoveGrain
    else:
      RG = core.rgvs.RemoveGrain

    if meAlgPar is None:
        # radius/range parameter for the motion estimation algorithms
        # AVS version uses the following, but values seemed to be based on an
        # incorrect understanding of the MVTools motion seach algorithm, mistaking 
        # it for the exact x264 behavior.
        # meAlgPar = [2,2,2,2,16,24,2,2][meAlg] 
        # Using Dogway's SMDegrain options here instead of the TemporalDegrain2 AVSI versions, which seem wrong.
        meAlgPar = 5 if rec and meTM else 2

    longlat = max(w, h)
    shortlat = min(w, h)
    # Scale grainLevel from -2-3 -> 0-5
    grainLevel = grainLevel + 2

    if (longlat<=1050 and shortlat<=576):
        autoTune = 0
    elif (longlat<=1280 and shortlat<=720):
        autoTune = 1
    elif (longlat<=2048 and shortlat<=1152):
        autoTune = 2
    else:
        autoTune = 3

    ChromaNoise = (degrainPlane > 0)
    
    if meSubpel is None:
        meSubpel = [4, 2, 2, 1][autoTune]
    
    if meBlksz is None:
        meBlksz = [8, 8, 16, 32][autoTune]

    limitAT = [-1, -1, 0, 0, 0, 1][grainLevel] + autoTune + 1

    if limitSigma is None:
        limitSigma = [6,8,12,16,32,48][limitAT]
    
    if limitBlksz is None:
        limitBlksz = [12,16,24,32,64,96][limitAT]

    if SrchClipPP is None:
        SrchClipPP = [0,0,0,3,3,3][grainLevel]

    if isGRAY:
        ChromaMotion = False
        ChromaNoise = False
        degrainPlane = 0
    
    if degrainPlane == 0:
        fPlane = [0]
    elif degrainPlane == 1:
        fPlane = [1]
    elif degrainPlane == 2:
        fPlane = [2]
    elif degrainPlane == 3:
        fPlane = [1, 2]
    else:
        fPlane = [0, 1, 2]

    if postFFT <= 0:
        postTR = 0

    if postFFT == 3:
        postTR = min(postTR, 7)

    if postFFT in [1, 2]:
        postTR = min(postTR, 2)

    if postBlkSize is None:
        postBlkSize = [0,48,32,12,0,0][postFFT]

    if grainLevelSetup:
        outputStage = 0
        degrainTR = 3
        
    if degrainTR > 3 or postTR > 3:
      import mvmulti

    rad = 3 if extraSharp else None
    mat = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    hpad = meBlksz
    vpad = meBlksz
    postTD  = postTR * 2 + 1
    maxTR = max(degrainTR, postTR)
    Overlap = meBlksz / 2
    Lambda = (1000 if meTM else 100) * (meBlksz ** 2) // 64
    PNew = 50 if meTM else 25

    ppSAD1 = ppSAD1 if ppSAD1 is not None else [3,5,7,9,11,13][grainLevel]
    ppSAD2 = ppSAD2 if ppSAD2 is not None else [2,4,5,6,7,8][grainLevel]
    ppSCD1 = ppSCD1 if ppSCD1 is not None else [3,3,3,4,5,6][grainLevel]

    if DCT == 5:
        #rescale threshold to match the SAD values when using SATD
        ppSAD1 *= 1.7
        ppSAD2 *= 1.7
        # ppSCD1 - this must not be scaled since scd is always based on SAD independently of the actual dct setting

    #here the per-pixel measure is converted to the per-8x8-Block (8*8 = 64) measure MVTools is using
    thSAD1 = int(ppSAD1 * 64)
    thSAD2 = int(ppSAD2 * 64)
    thSCD1 = int(ppSCD1 * 64)
    CMplanes = [0, 1, 2] if ChromaMotion else [0]
    
    if maxTR > 3 and not isFLOAT:
        raise ValueError("TemporalDegrain2: maxTR > 3 requires input of float sample type")
    
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    
    if SrchClipPP == 1:
        spatialBlur = core.resize.Bilinear(clip, m4(w/2), m4(h/2)).std.Convolution(matrix=mat, planes=CMplanes).resize.Bilinear(w, h)
    elif SrchClipPP > 1:
        spatialBlur = core.tcanny.TCanny(clip, sigma=2, mode=-1, planes=CMplanes)
        spatialBlur = core.std.Merge(spatialBlur, clip, [0.1] if ChromaMotion or isGRAY else [0.1, 0])
    else:
        spatialBlur = clip
    if SrchClipPP < 3:
        srchClip = spatialBlur
    else:
        expr = 'x {a} + y < x {b} + x {a} - y > x {b} - x y + 2 / ? ?'.format(a=7*bitDepthMultiplier, b=2*bitDepthMultiplier)
        srchClip = EXPR([spatialBlur, clip], [expr] if ChromaMotion or isGRAY else [expr, ''])

    super_args = dict(pel=meSubpel, hpad=hpad, vpad=vpad, sharp=SubPelInterp, chroma=ChromaMotion)
    analyse_args = dict(blksize=meBlksz, overlap=Overlap, search=meAlg, searchparam=meAlgPar, pelsearch=meSubpel, truemotion=meTM, lambda_=Lambda, pnew=PNew, global_=GlobalMotion, dct=DCT, chroma=ChromaMotion)
    recalculate_args = dict(thsad=thSAD1 // 2, blksize=max(meBlksz // 2, 4), overlap=max(Overlap // 2, 2), search=meAlg, searchparam=meAlgPar, truemotion=meTM, lambda_=Lambda/4, pnew=PNew, dct=DCT, chroma=ChromaMotion)

    lumaRebuild = DitherLumaRebuild(srchClip, s0=1, chroma=ChromaMotion)

    srchSuper = S(lumaRebuild, rfilter=4, **super_args)
    recSuper = S(lumaRebuild, levels=1, **super_args)
    
    if (maxTR > 0) and (degrainTR < 4 or postTR < 4):
        bVec1 = A(srchSuper, isb=True,  delta=1, **analyse_args)
        fVec1 = A(srchSuper, isb=False, delta=1, **analyse_args)
        if rec:
            bVec1 = R(recSuper, bVec1, **recalculate_args)
            fVec1 = R(recSuper, fVec1, **recalculate_args)
        if maxTR > 1:
            bVec2 = A(srchSuper, isb=True,  delta=2, **analyse_args)
            fVec2 = A(srchSuper, isb=False, delta=2, **analyse_args)
            if rec:
                bVec2 = R(recSuper, bVec2, **recalculate_args)
                fVec2 = R(recSuper, fVec2, **recalculate_args)
        if maxTR > 2:
            bVec3 = A(srchSuper, isb=True,  delta=3, **analyse_args)
            fVec3 = A(srchSuper, isb=False, delta=3, **analyse_args)
            if rec:
                bVec3 = R(recSuper, bVec3, **recalculate_args)
                fVec3 = R(recSuper, fVec3, **recalculate_args)

    if degrainTR > 3:
        vmulti1 = mvmulti.Analyze(srchSuper, tr=degrainTR, **analyse_args)
        if rec:
            vmulti1 = mvmulti.Recalculate(srchSuper, vmulti1, tr=degrainTR, **recalculate_args)

    if postTR > 3:
        vmulti2 = mvmulti.Analyze(srchSuper, tr=postTR, **analyse_args)
        if rec:
            vmulti2 = mvmulti.Recalculate(srchSuper, vmulti2, tr=postTR, **recalculate_args)
    #---------------------------------------
    # Degrain
    # "spat" is a prefiltered clip which is used to limit the effect of the 1st MV-denoise stage.
    if degrainTR > 0:
        s2 = limitSigma * 0.625
        s3 = limitSigma * 0.375
        s4 = limitSigma * 0.250
        ovNum = [4, 4, 4, 3, 2, 2][grainLevel]
        ov = 2 * round(limitBlksz / ovNum * 0.5)

        if neo and hasattr(core, 'neo_fft3d'):
          spat = core.neo_fft3d.FFT3D(clip, planes=fPlane, sigma=limitSigma, sigma2=s2, sigma3=s3, sigma4=s4, bt=3, bw=limitBlksz, bh=limitBlksz, ow=ov, oh=ov, ncpu=fftThreads)
        else:
          spat = core.fft3dfilter.FFT3DFilter(clip, planes=fPlane, sigma=limitSigma, sigma2=s2, sigma3=s3, sigma4=s4, bt=3, bw=limitBlksz, bh=limitBlksz, ow=ov, oh=ov, ncpu=fftThreads)
        spatD  = core.std.MakeDiff(clip, spat)
  
    # Update super args for all other motion analysis
    super_args |= dict(levels=1)

    # First MV-denoising stage. Usually here's some temporal-medianfiltering going on.
    # For simplicity, we just use MDegrain.
    if degrainTR > 0:
        supero = S(clip, **super_args)

        if degrainTR < 2:
            NR1 = D1(clip, supero, bVec1, fVec1, plane=degrainPlane, thsad=thSAD1, thscd1=thSCD1, thscd2=thSCD2)
        elif degrainTR < 3:
            NR1 = D2(clip, supero, bVec1, fVec1, bVec2, fVec2, plane=degrainPlane, thsad=thSAD1, thscd1=thSCD1, thscd2=thSCD2)
        elif degrainTR < 4:
            NR1 = D3(clip, supero, bVec1, fVec1, bVec2, fVec2, bVec3, fVec3, plane=degrainPlane, thsad=thSAD1, thscd1=thSCD1, thscd2=thSCD2)
        else:
            NR1 = mvmulti.DegrainN(clip, supero, vmulti1, tr=degrainTR, plane=degrainPlane, thsad=thSAD1, thscd1=thSCD1, thscd2=thSCD2)

    # Limit NR1 to not do more than what "spat" would do.
    if degrainTR > 0:
        NR1D = core.std.MakeDiff(clip, NR1)
        expr = 'x abs y abs < x y ?' if isFLOAT else f'x {mid} - abs y {mid} - abs < x y ?'
        DD   = EXPR([spatD, NR1D], [expr])
        NR1x = core.std.MakeDiff(clip, DD, [0])
    else:
        NR1x = clip
  
    # Second MV-denoising stage. We use MDegrain.
    if degrainTR > 0:
        NR1x_super = S(NR1x, **super_args)

        if degrainTR < 2:
            NR2 = D1(NR1x, NR1x_super, bVec1, fVec1, plane=degrainPlane, thsad=thSAD2, thscd1=thSCD1, thscd2=thSCD2)
        elif degrainTR < 3:
            NR2 = D2(NR1x, NR1x_super, bVec1, fVec1, bVec2, fVec2, plane=degrainPlane, thsad=thSAD2, thscd1=thSCD1, thscd2=thSCD2)
        elif degrainTR < 4:
            NR2 = D3(NR1x, NR1x_super, bVec1, fVec1, bVec2, fVec2, bVec3, fVec3, plane=degrainPlane, thsad=thSAD2, thscd1=thSCD1, thscd2=thSCD2)
        else:
            NR2 = mvmulti.DegrainN(NR1x, NR1x_super, vmulti1, tr=degrainTR, plane=degrainPlane, thsad=thSAD2, thscd1=thSCD1, thscd2=thSCD2)
    else:
        NR2 = clip
    
    #---------------------------------------
    # post FFT
    if postTR > 0:
        fullSuper = S(NR2, **super_args)

    if postTR > 0:
        if postTR == 1:
            noiseWindow = core.std.Interleave([C(NR2, fullSuper, fVec1, thsad=thSAD2, thscd1=thSCD1, thscd2=thSCD2), NR2,
                                               C(NR2, fullSuper, bVec1, thsad=thSAD2, thscd1=thSCD1, thscd2=thSCD2)])
        elif postTR == 2:
            noiseWindow = core.std.Interleave([C(NR2, fullSuper, fVec2, thsad=thSAD2, thscd1=thSCD1, thscd2=thSCD2),
                                               C(NR2, fullSuper, fVec1, thsad=thSAD2, thscd1=thSCD1, thscd2=thSCD2), NR2,
                                               C(NR2, fullSuper, bVec1, thsad=thSAD2, thscd1=thSCD1, thscd2=thSCD2),
                                               C(NR2, fullSuper, bVec2, thsad=thSAD2, thscd1=thSCD1, thscd2=thSCD2)])
        elif postTR == 3:
            noiseWindow = core.std.Interleave([C(NR2, fullSuper, fVec3, thsad=thSAD2, thscd1=thSCD1, thscd2=thSCD2),
                                               C(NR2, fullSuper, fVec2, thsad=thSAD2, thscd1=thSCD1, thscd2=thSCD2),
                                               C(NR2, fullSuper, fVec1, thsad=thSAD2, thscd1=thSCD1, thscd2=thSCD2), NR2,
                                               C(NR2, fullSuper, bVec1, thsad=thSAD2, thscd1=thSCD1, thscd2=thSCD2),
                                               C(NR2, fullSuper, bVec2, thsad=thSAD2, thscd1=thSCD1, thscd2=thSCD2),
                                               C(NR2, fullSuper, bVec3, thsad=thSAD2, thscd1=thSCD1, thscd2=thSCD2)])
        else:
            noiseWindow = mvmulti.Compensate(NR2, fullSuper, vmulti2, thsad=thSAD2, thscd1=thSCD1, thscd2=thSCD2, tr=postTR)
    else:
        noiseWindow = NR2
    
    if postFFT == 3:
        dnWindow = core.dfttest.DFTTest(noiseWindow, sigma=postSigma*4, tbsize=postTD, planes=fPlane, sbsize=postBlkSize, sosize=postBlkSize*9/12)
    elif postFFT == 4:
      if ChromaNoise:
        dnWindow = KNLMeansCL(noiseWindow, d=postTR, a=2, h=postSigma/2, device_id=knlDevId)
      else:
        use_cuda = hasattr(core, 'nlm_cuda')
        if use_cuda:
          nlmeans = clip.nlm_cuda.NLMeans
        else:
          nlmeans = clip.knlm.KNLMeansCL
        dnWindow = nlmeans(noiseWindow, d=postTR, a=2, h=postSigma/2, device_id=knlDevId)
    elif postFFT > 0:
        if postFFT == 1 and hasattr(core, 'neo_fft3d'):
          dnWindow = core.neo_fft3d.FFT3D(noiseWindow, sigma=postSigma, planes=fPlane, bt=postTD, ncpu=fftThreads, bw=postBlkSize, bh=postBlkSize)
        else:
          dnWindow = core.fft3dfilter.FFT3DFilter(noiseWindow, sigma=postSigma, planes=fPlane, bt=postTD, ncpu=fftThreads, bw=postBlkSize, bh=postBlkSize)
    else:
        dnWindow = RG(noiseWindow, mode=1)
    
    if postTR > 0:
        dnWindow = dnWindow[postTR::postTD]

    sharpened = ContraSharpening(dnWindow, clip, rad)

    if postMix > 0:
        sharpened = EXPR([clip,sharpened],f"x {postMix} * y {100-postMix} * + 100 /")

    return [NR1x, NR2, sharpened][outputStage]



# Helpers
    
def cround(x: float) -> int:
    return math.floor(x + 0.5) if x > 0 else math.ceil(x - 0.5)

def m4(x: Union[float, int]) -> int:
    return 16 if x < 16 else cround(x / 4) * 4
   
    
def DitherLumaRebuild(src: vs.VideoNode, s0: float = 2.0, c: float = 0.0625, chroma: bool = True) -> vs.VideoNode:
    '''Converts luma (and chroma) to PC levels, and optionally allows tweaking for pumping up the darks. (for the clip to be fed to motion search only)'''
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('DitherLumaRebuild: this is not a clip')

    if src.format.color_family == vs.RGB:
        raise vs.Error('DitherLumaRebuild: RGB format is not supported')

    is_gray = src.format.color_family == vs.GRAY
    is_integer = src.format.sample_type == vs.INTEGER

    bits = src.format.bits_per_sample
    neutral = 1 << (bits - 1)

    k = (s0 - 1) * c
    t = f'x {scale_value(16, 8, bits)} - {scale_value(219, 8, bits)} / 0 max 1 min' if is_integer else 'x 0 max 1 min'
    e = f'{k} {1 + c} {(1 + c) * c} {t} {c} + / - * {t} 1 {k} - * + ' + (f'{scale_value(256, 8, bits)} *' if is_integer else '')
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    return EXPR(src, expr=e if is_gray else [e, f'x {neutral} - 128 * 112 / {neutral} +' if chroma and is_integer else ''])
    
########################################
## Didée's functions:

# contra-sharpening: sharpen the denoised clip, but don't add more to any pixel than what was removed previously.
# script function from Didée, at the VERY GRAINY thread (http://forum.doom9.org/showthread.php?p=1076491#post1076491)
#
# Parameters:
#  radius (int)   - Spatial radius for contra-sharpening (1-3). Default is 2 for HD / 1 for SD
#  rep (int)      - Mode of repair to limit the difference. Default is 13
#  planes (int[]) - Whether to process the corresponding plane. The other planes will be passed through unchanged.
def ContraSharpening(denoised, original, radius=None, rep=13, planes=None):
    if not (isinstance(denoised, vs.VideoNode) and isinstance(original, vs.VideoNode)):
        raise vs.Error('ContraSharpening: This is not a clip')

    if denoised.format.id != original.format.id:
        raise vs.Error('ContraSharpening: Clips must be the same format')

    neutral = 1 << (denoised.format.bits_per_sample - 1)

    if planes is None:
        planes = list(range(denoised.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    if radius is None:
        radius = 2 if denoised.width > 1024 or denoised.height > 576 else 1

    s = MinBlur(denoised, r=radius, planes=planes)                                                                   # damp down remaining spots of the denoised clip

    matrix1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    matrix2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    if radius <= 1:
        RG11 = s.std.Convolution(matrix=matrix1, planes=planes)
    elif radius == 2:
        RG11 = s.std.Convolution(matrix=matrix1, planes=planes).std.Convolution(matrix=matrix2, planes=planes)
    else:
        RG11 = s.std.Convolution(matrix=matrix1, planes=planes).std.Convolution(matrix=matrix2, planes=planes).std.Convolution(matrix=matrix2, planes=planes)

    ssD = core.std.MakeDiff(s, RG11, planes=planes)                                                                  # the difference of a simple kernel blur
    allD = core.std.MakeDiff(original, denoised, planes=planes)                                                      # the difference achieved by the denoising
    if hasattr(core, 'zsmooth'):
      ssDD = core.zsmooth.Repair(ssD, allD, mode=[rep if i in planes else 0 for i in range(denoised.format.num_planes)])  # limit the difference to the max of what the denoising removed locally
    else:
      ssDD = core.rgvs.Repair(ssD, allD, mode=[rep if i in planes else 0 for i in range(denoised.format.num_planes)])  # limit the difference to the max of what the denoising removed locally
    expr = f'x {neutral} - abs y {neutral} - abs < x y ?'
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    ssDD = EXPR([ssDD, ssD], expr=[expr if i in planes else '' for i in range(denoised.format.num_planes)]) # abs(diff) after limiting may not be bigger than before
    return core.std.MergeDiff(denoised, ssDD, planes=planes)
    
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
        for i in range(sNumPlanes):
            if process[i]:
                if i > 0 and (sIsYUV):
                    expr.append(limitExprC)
                else:
                    expr.append(limitExprY)
            else:
                expr.append("")
        EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
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
def MinBlur(clp, r=1, planes=None):
    if not isinstance(clp, vs.VideoNode):
        raise vs.Error('MinBlur: This is not a clip')

    if planes is None:
        planes = list(range(clp.format.num_planes))
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
        if clp.format.bits_per_sample == 16:
            s16 = clp
            RG4 = clp.fmtc.bitdepth(bits=12, planes=planes, dmode=1).ctmf.CTMF(radius=3, planes=planes).fmtc.bitdepth(bits=16, planes=planes)
            RG4 = LimitFilter(s16, RG4, thr=0.0625, elast=2, planes=planes)
        else:
            RG4 = clp.ctmf.CTMF(radius=3, planes=planes)

    expr = 'x y - x z - * 0 < x x y - abs x z - abs < y z ? ?'
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    return EXPR([clp, RG11, RG4], expr=[expr if i in planes else '' for i in range(clp.format.num_planes)])


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
        if subsampled:
          clip = nlmeans(d=d, a=a, s=s, h=h, channels='Y', wmode=wmode, wref=wref, device_id=device_id)
          return nlmeans(d=d, a=a, s=s, h=h, channels='UV', wmode=wmode, wref=wref, device_id=device_id)
        else:
          return nlmeans(d=d, a=a, s=s, h=h, channels='YUV', wmode=wmode, wref=wref, device_id=device_id)
    else:
      nlmeans = clip.knlm.KNLMeansCL
      if subsampled:
          clip = nlmeans(d=d, a=a, s=s, h=h, channels='Y', wmode=wmode, wref=wref, device_type=device_type, device_id=device_id)
          return nlmeans(d=d, a=a, s=s, h=h, channels='UV', wmode=wmode, wref=wref, device_type=device_type, device_id=device_id)
      else:
          return nlmeans(d=d, a=a, s=s, h=h, channels='YUV', wmode=wmode, wref=wref, device_type=device_type, device_id=device_id)

""" From https://gist.github.com/4re/b5399b1801072458fc80#file-mcdegrainsharp-py 
"""

def _sharpen(clip, strength, planes):
    core = vs.core
    blur = core.tcanny.TCanny(clip, sigma=strength, mode=-1, planes=planes)
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    return EXPR([clip, blur], "x x + y -")


def mcdegrainsharp(clip, frames=2, bblur=0.3, csharp=0.3, bsrch=True, thsad=400, plane=4):
    """Based on MCDegrain By Didee:
    http://forum.doom9.org/showthread.php?t=161594
    Also based on DiDee observations in this thread:
    http://forum.doom9.org/showthread.php?t=161580

    "Denoise with MDegrainX, do slight sharpening where motionmatch is good,
    do slight blurring where motionmatch is bad"

    In areas where MAnalyse cannot find good matches,
    the blur() will be dominant.
    In areas where good matches are found,
    the sharpen()'ed pixels will overweight the blur()'ed pixels
    when the pixel averaging is performed.

    Args:
        frames (int): Strength of the denoising (1-3).
        bblur (float): Strength of the blurring for bad motion matching areas.
        csharp (float): Strength of the sharpening for god motion match areas.
        bsrch (bool): Blur the clip for the super clip for the motion search.
        thsad (int): Soft threshold of block sum absolute differences.
            Low values can result in staggered denoising,
            large values can result in ghosting and artefacts.
        plane (int): Sets processed color plane:
            0 - luma, 1 - chroma U, 2 - chroma V, 3 - both chromas, 4 - all.
    """
    core = vs.core

    if bblur > 1.58 or bblur < 0.0:
        raise ValueError('"bblur" must be between 0.0 and 1.58')
    if csharp > 1.0 or csharp < 0.0:
        raise ValueError('"csharp" must be between 0.0 and 1.0')

    blksize = 16 if clip.width > 960 else 8
    bblur = ((bblur * 2.83) / 1.58)
    csharp = ((csharp * 2.83) / 1.0)

    if plane == 3:
        planes = [1, 2]
    elif plane == 4:
        planes = [0, 1, 2]
    else:
        planes = plane
    
    c2 = core.tcanny.TCanny(clip, sigma=bblur, mode=-1, planes=planes)

    if bsrch is True:
        super_a = core.mv.Super(c2, pel=2, sharp=1)
    else:
        super_a = core.mv.Super(clip, pel=2, sharp=1)

    super_rend = core.mv.Super(_sharpen(clip, csharp, planes=planes), pel=2, sharp=1, levels=1)

    mvbw3 = core.mv.Analyse(super_a, isb=True, delta=3,
                            overlap=blksize//2, blksize=blksize)
    mvbw2 = core.mv.Analyse(super_a, isb=True, delta=2,
                            overlap=blksize//2, blksize=blksize)
    mvbw1 = core.mv.Analyse(super_a, isb=True, delta=1,
                            overlap=blksize//2, blksize=blksize)
    mvfw1 = core.mv.Analyse(super_a, isb=False, delta=1,
                            overlap=blksize//2, blksize=blksize)
    mvfw2 = core.mv.Analyse(super_a, isb=False, delta=2,
                            overlap=blksize//2, blksize=blksize)
    mvfw3 = core.mv.Analyse(super_a, isb=False, delta=3,
                            overlap=blksize//2, blksize=blksize)

    if frames == 1:
        last = core.mv.Degrain1(clip=c2, super=super_rend,
                                mvbw=mvbw1, mvfw=mvfw1, thsad=thsad,
                                plane=plane)
    elif frames == 2:
        last = core.mv.Degrain2(clip=c2, super=super_rend,
                                mvbw=mvbw1, mvfw=mvfw1,
                                mvbw2=mvbw2, mvfw2=mvfw2,
                                thsad=thsad, plane=plane)
    elif frames == 3:
        last = core.mv.Degrain3(clip=c2, super=super_rend,
                                mvbw=mvbw1, mvfw=mvfw1, mvbw2=mvbw2,
                                mvfw2=mvfw2, mvbw3=mvbw3, mvfw3=mvfw3,
                                thsad=thsad, plane=plane)
    else:
        raise ValueError('"frames" must be 1, 2 or 3.')

    return last
