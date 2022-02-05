import math
from functools import partial

import mvsfunc as mvf
import vapoursynth as vs

# copied from old havsfunc since in newer havsfunc they get removed
# - temporaldegrain
# - aaf

core = vs.core

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
    dd = core.std.Expr([spatD, nr1Diff], expr=[f'x {neutral} - abs y {neutral} - abs < x y ?'])
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
    

########################
# Ported version of aaf by MOmonster from avisynth
# Ported by Hinterwaeldlers
#
# aaf is one of the many aaa() modifications, so this is not my own basic idea
# the difference to aaa() is the repair postprocessing that allows also smaller sampling
# values without producing artefacts
# this makes aaf much faster (with small aas values)
#
# needed filters:
#	- MaskTools v2
#	- SangNom
#	- Repair (RemoveGrain pack)
#
# parameter description:
#	- mode
#			there are two modes you can use to reduce the side effects of sangnom
#			the default mode is "repair", it´s faster then the second mode="edge" and
#			avoid most artefacts also for smaller aas values
#			the mode "edge" filters only on edges and keep details sharper, read also estr/bstr
#			if you set another string than these two, no postprocessing will be done
#	- aas
#			this is the basic quality vs speed factor of aaf	->anti aliasing scaling
#			negative values process the horizontal and vertical direction without resizing
#			the complete source, this is much faster than with the absolut value, but will also
#			create more artefacts if you don´t use a repair mode
#			that higher the absolut value of aas that higher is the scaling factor, that better
#			is the quality, that slower the function and that lower the antialiasing effect
#			with aas=1.0 aaf performs like aa() and aaa()		[-2.0...2.0  -> -0.7]
#	- aay/aax
#			with aay and aax you can set the antialiasing strength in horizontal and vertical direction
#			if you set one of these parameter <=0 this direction won´t be processed
#			this give a nice speedup, but is only seldom useful	[0...64  ->28,aay]
#	- estr/bstr
#			these two parameters regulate the processing strength, they are only used with mode="edge"
#			estr is the strength on hard edges and bstr is the basic strength on flat areas
#			softer edges are calculated between these strength limits
#			estr has to be bigger than bstr		[0...255  ->255,40]

def aaf(                \
      inputClip         \
    , mode = "repair"   \
    , aas  = -0.7       \
    , aar  = None       \
    , aay  = 28         \
    , aax  = None       \
    , estr = 255        \
    , bstr = 40         \
) :
    mode = mode.lower()
    if aas < 0:
        aas = (aas-1)*0.25
    else:
        aas = (aas+1)*0.25
    # Determine the default parameters, which depend on other input
    if aar is None:
        aar = math.fabs(aas)
    if aax is None:
        aax = aay

    sx = inputClip.width
    sy = inputClip.height

    isGray = (inputClip.format.color_family == vs.GRAY)

    neutral = 1 << (inputClip.format.bits_per_sample - 1)
    peak = (1 << inputClip.format.bits_per_sample) - 1

    if aay > 0:
        # Do the upscaling
        if aas < 0:
            aa = inputClip.resize.Lanczos(sx, 4*int(sy*aar))
        elif aar == 0.5:
            aa = inputClip.resize.Point(2*sx, 2*sy)
        else:
            aa = inputClip.resize.Lanczos(4*int(sx*aar), 4*int(sy*aar))

        # y-Edges
        aa = aa.sangnom.SangNom(aa=aay)
    else:
        aa = inputClip

    if aax > 0:
        if aas < 0:
            aa = aa.resize.Lanczos(4*int(sx*aar), sy)
        aa = aa.std.Transpose()
        # x-Edges
        aa = aa.sangnom.SangNom(aa=aax)
        aa = aa.std.Transpose()

    # Restore original scaling
    aa = aa.resize.Lanczos(sx, sy)

    repMode = [18] if isGray else [18, 0]

    if mode == "repair":
        return core.rgvs.Repair(aa, inputClip, mode=repMode)

    if mode != "edge":
        return aa

    # u=1, v=1 is not directly so use the copy
    mask = core.std.MakeDiff(inputClip.std.Maximum(planes=0)\
                             , inputClip.std.Minimum(planes=0)\
                             , planes=0)
    expr = 'x {i} > {estr} x {neutral} - {j} 90 / * {bstr} + ?'.format(i=scale(218, peak), estr=scale(estr, peak), neutral=neutral, j=estr - bstr, bstr=scale(bstr, peak))
    mask = mask.std.Expr(expr=[expr] if isGray else [expr, ''])

    merged = core.std.MaskedMerge(inputClip, aa, mask, planes=0)
    if aas > 0.84:
        return merged
    return core.rgvs.Repair(merged, inputClip, mode=repMode)

######################################## HELPER FUNCTION ########################################    
    
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
    ssDD = core.rgvs.Repair(ssD, allD, mode=[rep if i in planes else 0 for i in range(denoised.format.num_planes)])  # limit the difference to the max of what the denoising removed locally
    expr = f'x {neutral} - abs y {neutral} - abs < x y ?'
    ssDD = core.std.Expr([ssDD, ssD], expr=[expr if i in planes else '' for i in range(denoised.format.num_planes)]) # abs(diff) after limiting may not be bigger than before
    return core.std.MergeDiff(denoised, ssDD, planes=planes)
    
    
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
            RG4 = mvf.LimitFilter(s16, RG4, thr=0.0625, elast=2, planes=planes)
        else:
            RG4 = clp.ctmf.CTMF(radius=3, planes=planes)

    expr = 'x y - x z - * 0 < x x y - abs x z - abs < y z ? ?'
    return core.std.Expr([clp, RG11, RG4], expr=[expr if i in planes else '' for i in range(clp.format.num_planes)])