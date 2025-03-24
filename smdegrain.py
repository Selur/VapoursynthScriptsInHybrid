import vapoursynth as vs
from vapoursynth import core

import math

from typing import Sequence, Union, Optional

import nnedi3_resample
import sharpen

################################################################################################
###                                                                                          ###
###                           Simple MDegrain Mod - SMDegrain()                              ###
###                                                                                          ###
###                       Mod by Dogway - Original idea by Caroliano                         ###
###                                                                                          ###
###          Special Thanks: Sagekilla, Didée, cretindesalpes, Gavino and MVtools people     ###
###                                                                                          ###
###                       v3.1.2d (Dogway's mod) - 21 July 2015                              ###
###                                                                                          ###
################################################################################################
###
### General purpose simple degrain function. Pure temporal denoiser. Basically a wrapper(function)/frontend of mvtools2+mdegrain
### with some added common related options. Goal is accessibility and quality but not targeted to any specific kind of source.
### The reason behind is to keep it simple so aside masktools2 you will only need MVTools2.
###
### Check documentation for deep explanation on settings and defaults.
### VideoHelp thread: (http://forum.videohelp.com/threads/369142)
###
################################################################################################

# Globals
bv6 = bv4 = bv3 = bv2 = bv1 = fv1 = fv2 = fv3 = fv4 = fv6 = None

def SMDegrain(input, tr=2, thSAD=300, thSADC=None, RefineMotion=False, contrasharp=None, CClip=None, interlaced=False, tff=None, plane=4, Globals=0, pel=None, subpixel=2, prefilter=-1, mfilter=None,
              blksize=None, overlap=None, search=4, truemotion=None, MVglobal=None, dct=0, limit=255, limitc=None, thSCD1=400, thSCD2=130, chroma=True, hpad=None, vpad=None, Str=1.0, Amp=0.0625, opencl=False, device=None):
    if not isinstance(input, vs.VideoNode):
        raise vs.Error('SMDegrain: This is not a clip')

    if input.format.color_family == vs.GRAY:
        plane = 0
        chroma = False

    peak = (1 << input.format.bits_per_sample) - 1

    # Defaults & Conditionals
    thSAD2 = thSAD // 2
    if thSADC is None:
        thSADC = thSAD2

    GlobalR = (Globals == 1)
    GlobalO = (Globals >= 3)
    if1 = CClip is not None

    if contrasharp is None:
        contrasharp = not GlobalO and if1

    w = input.width
    h = input.height
    preclip = isinstance(prefilter, vs.VideoNode)
    ifC = isinstance(contrasharp, bool)
    if0 = contrasharp if ifC else contrasharp > 0
    is_large = w > 1024 or h > 576

    if pel is None:
        pel = 1 if is_large else 2
    if pel < 2:
        subpixel = min(subpixel, 2)
    pelclip = pel > 1 and subpixel >= 3

    if blksize is None:
        blksize = 16 if is_large else 8
    blk2 = blksize // 2
    if overlap is None:
        overlap = blk2
    ovl2 = overlap // 2
    if truemotion is None:
        truemotion = not is_large
    if MVglobal is None:
        MVglobal = truemotion

    planes = [0, 1, 2] if chroma else [0]
    plane0 = (plane != 0)

    if hpad is None:
        hpad = blksize
    if vpad is None:
        vpad = blksize
    limit = scale(limit, peak)
    if limitc is None:
        limitc = limit
    else:
        limitc = scale(limitc, peak)

    # Error Report
    if not (ifC or isinstance(contrasharp, int)):
        raise vs.Error("SMDegrain: 'contrasharp' only accepts bool and integer inputs")
    if if1 and (not isinstance(CClip, vs.VideoNode) or CClip.format.id != input.format.id):
        raise vs.Error("SMDegrain: 'CClip' must be the same format as input")
    if interlaced and h & 3:
        raise vs.Error('SMDegrain: Interlaced source requires mod 4 height sizes')
    if interlaced and not isinstance(tff, bool):
        raise vs.Error("SMDegrain: 'tff' must be set if source is interlaced. Setting tff to true means top field first and false means bottom field first")
    if not (isinstance(prefilter, int) or preclip):
        raise vs.Error("SMDegrain: 'prefilter' only accepts integer and clip inputs")
    if preclip and prefilter.format.id != input.format.id:
        raise vs.Error("SMDegrain: 'prefilter' must be the same format as input")
    if mfilter is not None and (not isinstance(mfilter, vs.VideoNode) or mfilter.format.id != input.format.id):
        raise vs.Error("SMDegrain: 'mfilter' must be the same format as input")
    if RefineMotion and blksize < 8:
        raise vs.Error('SMDegrain: For RefineMotion you need a blksize of at least 8')
    if not chroma and plane != 0:
        raise vs.Error('SMDegrain: Denoising chroma with luma only vectors is bugged in mvtools and thus unsupported')

    # RefineMotion Variables
    if RefineMotion:
        halfblksize = blk2                                         # MRecalculate works with half block size
        halfoverlap = overlap if overlap <= 2 else ovl2 + ovl2 % 2 # Halve the overlap to suit the halved block size
        halfthSAD = thSAD2                                         # MRecalculate uses a more strict thSAD, which defaults to 150 (half of function's default of 300)

    # Input preparation for Interlacing
    if not interlaced:
        inputP = input
    else:
        inputP = input.std.SeparateFields(tff=tff)
        h = h/2

    # Prefilter & Motion Filter
    if mfilter is None:
        mfilter = inputP

    if not GlobalR:
        if preclip:
            pref = prefilter
        elif prefilter <= -1:
            pref = inputP
        elif prefilter == 3:
            expr = 'x {i} < {peak} x {j} > 0 {peak} x {i} - {peak} {j} {i} - / * - ? ?'.format(i=scale(16, peak), j=scale(75, peak), peak=peak)
            try:
              import importlib
              dfttest = importlib.import_module('dfttest2')
              DFTTest = dfttest.DFTTest
            except ModuleNotFoundError:
              DFTTest = core.dfttest.DFTTest
            filtered = DFTTest(inputP, tbsize=1, slocation=[0.0,4.0, 0.2,9.0, 1.0,15.0], planes=planes);
            pref = core.std.MaskedMerge(filtered, inputP, GetPlane(inputP, 0).std.Expr(expr=[expr]), planes=planes)
        elif prefilter >= 4:
            pref = KNLMeansCL(inputP, d=1, a=1, h=7)
        else:
            pref = MinBlur(inputP, r=prefilter, planes=planes)
    else:
        pref = inputP

    # Default Auto-Prefilter - Luma expansion TV->PC (up to 16% more values for motion estimation)
    if not GlobalR:
        pref = DitherLumaRebuild(pref, s0=Str, c=Amp, chroma=chroma)

    # Motion vectors search
    global bv6, bv4, bv3, bv2, bv1, fv1, fv2, fv3, fv4, fv6
    super_args = dict(hpad=hpad, vpad=vpad, pel=pel)
    # Subpixel 3
    if pelclip:
      nnediMode = 'nnedi3cl' if opencl else 'znedi3'
      cshift = 0.25 if pel == 2 else 0.375
      pclip = nnedi3_resample.nnedi3_resample(pref, w * pel, h * pel, src_left=cshift, src_top=cshift, nns=4, mode=nnediMode, device=device)
      if not GlobalR:
         pclip2 = nnedi3_resample.nnedi3_resample(inputP, w * pel, h * pel, src_left=cshift, src_top=cshift, nns=4, mode=nnediMode, device=device)
      super_search = pref.mv.Super(chroma=chroma, rfilter=4, pelclip=pclip, **super_args)
    else:
      super_search = pref.mv.Super(chroma=chroma, sharp=subpixel, rfilter=4, **super_args)

    
    analyse_args = dict(blksize=blksize, search=search, chroma=chroma, truemotion=truemotion, global_=MVglobal, overlap=overlap, dct=dct)
    if RefineMotion:
        recalculate_args = dict(thsad=halfthSAD, blksize=halfblksize, search=search, chroma=chroma, truemotion=truemotion, overlap=halfoverlap, dct=dct)


    if not GlobalR:
        if pelclip:
            super_render = inputP.mv.Super(levels=1, chroma=plane0, pelclip=pclip2, **super_args)
            if RefineMotion:
                Recalculate = pref.mv.Super(levels=1, chroma=chroma, pelclip=pclip, **super_args)
        else:
            super_render = inputP.mv.Super(levels=1, chroma=plane0, sharp=subpixel, **super_args)
            if RefineMotion:
                Recalculate = pref.mv.Super(levels=1, chroma=chroma, sharp=subpixel, **super_args)

        if interlaced:
            if tr > 2:
                bv6 = super_search.mv.Analyse(isb=True, delta=6, **analyse_args)
                fv6 = super_search.mv.Analyse(isb=False, delta=6, **analyse_args)
                if RefineMotion:
                    bv6 = core.mv.Recalculate(Recalculate, bv6, **recalculate_args)
                    fv6 = core.mv.Recalculate(Recalculate, fv6, **recalculate_args)
            if tr > 1:
                bv4 = super_search.mv.Analyse(isb=True, delta=4, **analyse_args)
                fv4 = super_search.mv.Analyse(isb=False, delta=4, **analyse_args)
                if RefineMotion:
                    bv4 = core.mv.Recalculate(Recalculate, bv4, **recalculate_args)
                    fv4 = core.mv.Recalculate(Recalculate, fv4, **recalculate_args)
        else:
            if tr > 2:
                bv3 = super_search.mv.Analyse(isb=True, delta=3, **analyse_args)
                fv3 = super_search.mv.Analyse(isb=False, delta=3, **analyse_args)
                if RefineMotion:
                    bv3 = core.mv.Recalculate(Recalculate, bv3, **recalculate_args)
                    fv3 = core.mv.Recalculate(Recalculate, fv3, **recalculate_args)
            bv1 = super_search.mv.Analyse(isb=True, delta=1, **analyse_args)
            fv1 = super_search.mv.Analyse(isb=False, delta=1, **analyse_args)
            if RefineMotion:
                bv1 = core.mv.Recalculate(Recalculate, bv1, **recalculate_args)
                fv1 = core.mv.Recalculate(Recalculate, fv1, **recalculate_args)
        if interlaced or tr > 1:
            bv2 = super_search.mv.Analyse(isb=True, delta=2, **analyse_args)
            fv2 = super_search.mv.Analyse(isb=False, delta=2, **analyse_args)
            if RefineMotion:
                bv2 = core.mv.Recalculate(Recalculate, bv2, **recalculate_args)
                fv2 = core.mv.Recalculate(Recalculate, fv2, **recalculate_args)
    else:
        super_render = super_search

    # Finally, MDegrain
    search_params = dict(blksize=blksize, search=search, chroma=chroma, truemotion=truemotion, global_=MVglobal, overlap=overlap, dct=dct)
    refine_params = dict(thsad=thSAD2, blksize=blksize // 2, search=search, chroma=chroma, truemotion=truemotion, overlap=overlap // 2, dct=dct) if RefineMotion else None
    vectors = get_motion_vectors(super_search, super_render if RefineMotion else None, search_params, refine_params, tr, interlaced)
    degrain_args = dict(thsad=thSAD, thsadc=thSADC, plane=plane, limit=limit, limitc=limitc, thscd1=thSCD1, thscd2=thSCD2)
    if not GlobalO:
      if interlaced:
        if tr >= 3:
            output = core.mv.Degrain3(mfilter, super_render, bv2, fv2, bv4, fv4, bv6, fv6, **degrain_args)
        elif tr == 2:
            output = core.mv.Degrain2(mfilter, super_render, bv2, fv2, bv4, fv4, **degrain_args)
        else:
            output = core.mv.Degrain1(mfilter, super_render, bv2, fv2, **degrain_args)
      else:
        if tr >= 6:
          output = core.mv.Degrain6(inputP, super_render, vectors['bv1'], vectors['fv1'], vectors['bv2'], vectors['fv2'], vectors['bv3'], vectors['fv3'], vectors['bv4'], vectors['fv4'], vectors['bv5'], vectors['fv5'], vectors['bv6'], vectors['fv6'], **degrain_args)
        elif tr == 5:
          output = core.mv.Degrain5(inputP, super_render, vectors['bv1'], vectors['fv1'], vectors['bv2'], vectors['fv2'], vectors['bv3'], vectors['fv3'], vectors['bv4'], vectors['fv4'], vectors['bv5'], vectors['fv5'], **degrain_args)
        elif tr == 4: 
          output = core.mv.Degrain4(inputP, super_render, vectors['bv1'], vectors['fv1'], vectors['bv2'], vectors['fv2'], vectors['bv3'], vectors['fv3'], vectors['bv4'], vectors['fv4'], **degrain_args)
        elif tr == 3:
          output = core.mv.Degrain3(inputP, super_render, vectors['bv1'], vectors['fv1'], vectors['bv2'], vectors['fv2'], vectors['bv3'], vectors['fv3'], **degrain_args)
        elif tr == 2:
          output = core.mv.Degrain2(inputP, super_render, vectors['bv1'], vectors['fv1'], vectors['bv2'], vectors['fv2'], **degrain_args)
        else:
          output = core.mv.Degrain1(inputP, super_render, vectors['bv1'], vectors['fv1'], **degrain_args)

  # Contrasharp (only sharpens luma)
    if not GlobalO and if0:
        if if1:
            if interlaced:
                CClip = CClip.std.SeparateFields(tff=tff)
        else:
            CClip = inputP

    # Output
    if not GlobalO:
        if if0:
            if interlaced:
                if ifC:
                    return Weave(ContraSharpening(output, CClip, planes=planes), tff=tff)
                else:
                    return Weave(sharpen.LSFmod(output, strength=contrasharp, source=CClip, Lmode=0, soothe=False, defaults='slow'), tff=tff)
            elif ifC:
                return ContraSharpening(output, CClip, planes=planes)
            else:
                return sharpen.LSFmod(output, strength=contrasharp, source=CClip, Lmode=0, soothe=False, defaults='slow')
        elif interlaced:
            return Weave(output, tff=tff)
        else:
            return output
    else:
        return input

# Helpers

def Padding(clip: vs.VideoNode, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('Padding: this is not a clip')

    if left < 0 or right < 0 or top < 0 or bottom < 0:
        raise vs.Error('Padding: border size to pad must not be negative')

    width = clip.width + left + right
    height = clip.height + top + bottom

    return clip.resize.Point(width, height, src_left=-left, src_top=-top, src_width=width, src_height=height)

def ContraSharpening(
    denoised: vs.VideoNode, original: vs.VideoNode, radius: int = 1, rep: int = 1, planes: Optional[Union[int, Sequence[int]]] = None
) -> vs.VideoNode:
    '''
    contra-sharpening: sharpen the denoised clip, but don't add more to any pixel than what was removed previously.

    Parameters:
        denoised: Denoised clip to sharpen.

        original: Original clip before denoising.

        radius: Spatial radius for contra-sharpening.

        rep: Mode of repair to limit the difference.

        planes: Specifies which planes will be processed. Any unprocessed planes will be simply copied.
            By default only luma plane will be processed for non-RGB formats.
    '''
    if not (isinstance(denoised, vs.VideoNode) and isinstance(original, vs.VideoNode)):
        raise vs.Error('ContraSharpening: this is not a clip')

    if denoised.format.id != original.format.id:
        raise vs.Error('ContraSharpening: clips must have the same format')

    neutral = (1 << denoised.format.bits_per_sample) - 1

    plane_range = range(denoised.format.num_planes)

    if planes is None:
        planes = [0] if denoised.format.color_family != vs.RGB else [0, 1, 2]
    elif isinstance(planes, int):
        planes = [planes]

    pad = 2 if radius < 3 else 4
    denoised = Padding(denoised, pad, pad, pad, pad)
    original = Padding(original, pad, pad, pad, pad)

    matrix1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    matrix2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    # damp down remaining spots of the denoised clip
    s = MinBlur(denoised, radius, planes)
    # the difference achieved by the denoising
    allD = core.std.MakeDiff(original, denoised, planes=planes)

    RG11 = s.std.Convolution(matrix=matrix1, planes=planes)
    if radius >= 2:
        RG11 = RG11.std.Convolution(matrix=matrix2, planes=planes)
    if radius >= 3:
        RG11 = RG11.std.Convolution(matrix=matrix2, planes=planes)

    # the difference of a simple kernel blur
    ssD = core.std.MakeDiff(s, RG11, planes=planes)
    # limit the difference to the max of what the denoising removed locally
    ssDD = core.rgvs.Repair(ssD, allD, mode=[rep if i in planes else 0 for i in plane_range])
    # abs(diff) after limiting may not be bigger than before
    ssDD = core.std.Expr([ssDD, ssD], expr=[f'x {neutral} - abs y {neutral} - abs < x y ?' if i in planes else '' for i in plane_range])
    # apply the limited difference (sharpening is just inverse blurring)
    last = core.std.MergeDiff(denoised, ssDD, planes=planes)
    return last.std.Crop(pad, pad, pad, pad)

def cround(x: float) -> int:
    return math.floor(x + 0.5) if x > 0 else math.ceil(x - 0.5)

def scale(value, peak):
    return cround(value * peak / 255) if peak != 1 else value / 255
    
    
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
        if clp.format.bits_per_sample == 16:
            s16 = clp
            RG4 = depth(clp, 12, dither_type=Dither.NONE).ctmf.CTMF(radius=3, planes=planes)
            RG4 = LimitFilter(s16, depth(RG4, 16), thr=0.0625, elast=2, planes=planes)
        else:
            RG4 = clp.ctmf.CTMF(radius=3, planes=planes)

    return core.std.Expr([clp, RG11, RG4], expr=['x y - x z - * 0 < x x y - abs x z - abs < y z ? ?' if i in planes else '' for i in plane_range])
    
    
def DitherLumaRebuild(src, s0=2., c=0.0625, chroma=True):
    # Converts luma (and chroma) to PC levels, and optionally allows tweaking for pumping up the darks. (for the clip to be fed to motion search only)
    # By courtesy of cretindesalpes. (https://forum.doom9.org/showthread.php?p=1548318)

    if not isinstance(src, vs.VideoNode):
        raise TypeError("DitherLumaRebuild: This is not a clip!")
    
    bd = src.format.bits_per_sample
    isFLOAT = src.format.sample_type == vs.FLOAT
    i = 0.00390625 if isFLOAT else 1 << (bd - 8)

    x = 'x {} /'.format(i) if bd != 8 else 'x'
    expr = 'x 128 * 112 /' if isFLOAT else '{} 128 - 128 * 112 / 128 + {} *'.format(x, i)
    k = (s0 - 1) * c
    t = '{} 16 - 219 / 0 max 1 min'.format(x)
    c1 = 1 + c
    c2 = c1 * c
    e = '{} {} {} {} {} + / - * {} 1 {} - * + {} *'.format(k, c1, c2, t, c, t, k, 256*i)
    
    return core.std.Expr([src], [e] if src.format.num_planes == 1 else [e, expr if chroma else ''])
    
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
          return nlmeans(d=d, a=a, s=s, h=h, channels='YUV', wmode=wmode, wref=wref, device_type=device_type, device_id=device_id)
    else:
      nlmeans = clip.knlm.KNLMeansCL
      if subsampled:
          clip = nlmeans(d=d, a=a, s=s, h=h, channels='Y', wmode=wmode, wref=wref, device_type=device_type, device_id=device_id)
          return nlmeans(d=d, a=a, s=s, h=h, channels='UV', wmode=wmode, wref=wref, device_type=device_type, device_id=device_id)
      else:
          return nlmeans(d=d, a=a, s=s, h=h, channels='YUV', wmode=wmode, wref=wref, device_type=device_type, device_id=device_id)
        
def sbr(c: vs.VideoNode, r: int = 1, planes: Optional[Union[int, Sequence[int]]] = None) -> vs.VideoNode:
    '''make a highpass on a blur's difference (well, kind of that)'''
    if not isinstance(c, vs.VideoNode):
        raise vs.Error('sbr: this is not a clip')

    neutral = 1 << (c.format.bits_per_sample - 1) if c.format.sample_type == vs.INTEGER else 0.0

    plane_range = range(c.format.num_planes)

    if planes is None:
        planes = list(plane_range)
    elif isinstance(planes, int):
        planes = [planes]

    matrix1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    matrix2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    RG11 = c.std.Convolution(matrix=matrix1, planes=planes)
    if r >= 2:
        RG11 = RG11.std.Convolution(matrix=matrix2, planes=planes)
    if r >= 3:
        RG11 = RG11.std.Convolution(matrix=matrix2, planes=planes)

    RG11D = core.std.MakeDiff(c, RG11, planes=planes)

    RG11DS = RG11D.std.Convolution(matrix=matrix1, planes=planes)
    if r >= 2:
        RG11DS = RG11DS.std.Convolution(matrix=matrix2, planes=planes)
    if r >= 3:
        RG11DS = RG11DS.std.Convolution(matrix=matrix2, planes=planes)

    RG11DD = core.std.Expr(
        [RG11D, RG11DS],
        expr=[f'x y - x {neutral} - * 0 < {neutral} x y - abs x {neutral} - abs < x y - {neutral} + x ? ?' if i in planes else '' for i in plane_range],
    )
    return core.std.MakeDiff(c, RG11DD, planes=planes)
    
def get_motion_vectors(super_search, refine, search_params, refine_params, tr, interlaced):
    vectors = {}
    
    if tr >= 1:
        vectors['bv1'] = super_search.mv.Analyse(isb=True, delta=1, **search_params)
        vectors['fv1'] = super_search.mv.Analyse(isb=False, delta=1, **search_params)
        if refine:
            vectors['bv1'] = core.mv.Recalculate(refine, vectors['bv1'], **refine_params)
            vectors['fv1'] = core.mv.Recalculate(refine, vectors['fv1'], **refine_params)
    
    if interlaced or tr >= 2:
        vectors['bv2'] = super_search.mv.Analyse(isb=True, delta=2, **search_params)
        vectors['fv2'] = super_search.mv.Analyse(isb=False, delta=2, **search_params)
        if refine:
            vectors['bv2'] = core.mv.Recalculate(refine, vectors['bv2'], **refine_params)
            vectors['fv2'] = core.mv.Recalculate(refine, vectors['fv2'], **refine_params)
    
    if tr >= 3:
        for i in range(3, tr + 1):
            vectors[f'bv{i}'] = super_search.mv.Analyse(isb=True, delta=i, **search_params)
            vectors[f'fv{i}'] = super_search.mv.Analyse(isb=False, delta=i, **search_params)
            if refine:
                vectors[f'bv{i}'] = core.mv.Recalculate(refine, vectors[f'bv{i}'], **refine_params)
                vectors[f'fv{i}'] = core.mv.Recalculate(refine, vectors[f'fv{i}'], **refine_params)
    
    return vectors