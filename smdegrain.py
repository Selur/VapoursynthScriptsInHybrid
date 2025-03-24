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



from enum import Enum
class SearchMode(Enum):
    DIA = 0
    HEX = 1
    UMH = 2
    HIERARCHICAL = 4

def SMDegrain(input, tr=2, thSAD=300, thSADC=None, RefineMotion=False, contrasharp=None, CClip=None, interlaced=False, tff=None, plane=4, pel=None, subpixel=2, prefilter=-1, mfilter=None,
              blksize=None, overlap=None, search=SearchMode.HIERARCHICAL.value, truemotion=None, MVglobal=None, dct=0, limit=255, limitc=None, thSCD1=400, thSCD2=130, chroma=True, hpad=None, vpad=None, Str=1.0, Amp=0.0625, opencl=False, device=None):
    
    validate_input(input, CClip, prefilter, mfilter, interlaced, tff)

    peak = (1 << input.format.bits_per_sample) - 1

    thSAD2 = thSAD // 2
    thSADC = thSADC or thSAD2

    if input.format.color_family == vs.GRAY:
        plane, chroma = 0, False

    w, h = input.width, input.height
    is_large = w > 1024 or h > 576

    pel = pel if pel is not None else (1 if is_large else 2)
    subpixel = min(subpixel, 2) if pel < 2 else subpixel
    pelclip = pel > 1 and subpixel >= 3

    blksize = blksize or (16 if is_large else 8)
    overlap = overlap or (blksize // 2)
    
    truemotion = truemotion if truemotion is not None else not is_large
    MVglobal = MVglobal if MVglobal is not None else truemotion

    planes = [0, 1, 2] if chroma else [0]
    
    hpad = hpad or blksize
    vpad = vpad or blksize

    limit = scale(limit, peak)
    limitc = scale(limitc if limitc is not None else limit, peak)

    inputP = input.std.SeparateFields(tff=tff) if interlaced else input

    # Prefilter selection
    if isinstance(prefilter, vs.VideoNode):
        pref = prefilter
    elif prefilter <= -1:
        pref = inputP
    else:
        pref = nnedi3_resample.nnedi3_resample(inputP, w, h, nns=4, device=device)

    super_args = dict(hpad=hpad, vpad=vpad, pel=pel)
    search_params = dict(blksize=blksize, search=search, chroma=chroma, truemotion=truemotion, global_=MVglobal, overlap=overlap, dct=dct)
    
    if pelclip:
        pclip = nnedi3_resample.nnedi3_resample(pref, w * pel, h * pel, nns=4, device=device)
        super_search = pref.mv.Super(chroma=chroma, rfilter=4, pelclip=pclip, **super_args)
    else:
        super_search = pref.mv.Super(chroma=chroma, sharp=subpixel, rfilter=4, **super_args)

    super_render = inputP.mv.Super(levels=1, chroma=plane != 0, pelclip=pclip if pelclip else None, **super_args)

    refine_params = dict(thsad=thSAD2, blksize=blksize // 2, search=search, chroma=chroma, truemotion=truemotion, overlap=overlap // 2, dct=dct) if RefineMotion else None
    vectors = get_motion_vectors(super_search, super_render if RefineMotion else None, search_params, refine_params, tr, interlaced)

    degrain_args = dict(thsad=thSAD, thsadc=thSADC, plane=plane, limit=limit, limitc=limitc, thscd1=thSCD1, thscd2=thSCD2)
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

    return output

# Helpers



def scale(value, peak):
    return value * peak // 255 if peak > 255 else value

def validate_input(clip, cclip, prefilter, mfilter, interlaced, tff):
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('SMDegrain: This is not a clip')
    
    if cclip and (not isinstance(cclip, vs.VideoNode) or cclip.format.id != clip.format.id):
        raise vs.Error("SMDegrain: 'CClip' must be the same format as input")

    if prefilter and (not isinstance(prefilter, int) and prefilter.format.id != clip.format.id):
        raise vs.Error("SMDegrain: 'prefilter' must be the same format as input")

    if mfilter and (not isinstance(mfilter, vs.VideoNode) or mfilter.format.id != clip.format.id):
        raise vs.Error("SMDegrain: 'mfilter' must be the same format as input")

    if interlaced and not isinstance(tff, bool):
        raise vs.Error("SMDegrain: 'tff' must be set if source is interlaced")

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
