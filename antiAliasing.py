import vapoursynth as vs
from vapoursynth import core

import math

from typing import TypeVar, Optional
from functools import partial

T = TypeVar('T')

# Taken from old havsfunc
def daa(
    c: vs.VideoNode,
    nsize: Optional[int] = None,
    nns: Optional[int] = None,
    qual: Optional[int] = None,
    pscrn: Optional[int] = None,
    int16_prescreener: Optional[bool] = None,
    int16_predictor: Optional[bool] = None,
    exp: Optional[int] = None,
    opencl: bool = False,
    device: Optional[int] = None,
) -> vs.VideoNode:
    '''
    Anti-aliasing with contra-sharpening by Didée.

    It averages two independent interpolations, where each interpolation set works between odd-distanced pixels.
    This on its own provides sufficient amount of blurring. Enough blurring that the script uses a contra-sharpening step to counteract the blurring.
    '''
    if not isinstance(c, vs.VideoNode):
        raise vs.Error('daa: this is not a clip')

    if opencl:
        nnedi3 = partial(core.nnedi3cl.NNEDI3CL, nsize=nsize, nns=nns, qual=qual, pscrn=pscrn, device=device)
    else:
        if hasattr(core,'znedi3'):
          nnedi3 = partial(core.znedi3.nnedi3, nsize=nsize, nns=nns, qual=qual, pscrn=pscrn, int16_prescreener=int16_prescreener, int16_predictor=int16_predictor, exp=exp)
        else:
          nnedi3 = partial(core.nnedi3.nnedi3, nsize=nsize, nns=nns, qual=qual, pscrn=pscrn, int16_prescreener=int16_prescreener, int16_predictor=int16_predictor, exp=exp)

    nn = nnedi3(c, field=3)
    dbl = core.std.Merge(nn[::2], nn[1::2])
    dblD = core.std.MakeDiff(c, dbl)
    shrpD = core.std.MakeDiff(dbl, dbl.std.Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1] if c.width > 1100 else [1, 2, 1, 2, 4, 2, 1, 2, 1]))
    if hasattr(core,'zsmooth'):
      DD = core.zsmooth.Repair(shrpD, dblD, mode=13)
    else:
      DD = core.rgvs.Repair(shrpD, dblD, mode=13)
    return core.std.MergeDiff(dbl, DD)

def daamod(c, nsize=None, nns=None, qual=None, pscrn=None, exp=None, opencl=False, device=None, rep=9):
    """Anti-aliasing with contra-sharpening by Didée, modded by GMJCZP"""

    if not isinstance(c, vs.VideoNode):
        raise TypeError("daamod: This is not a clip")

    isFLOAT = c.format.sample_type == vs.FLOAT
    if hasattr(core,'zsmooth'):
      R = core.zsmooth.Repair
      V = core.zsmooth.VerticalCleaner
    else:
      R = core.rgsf.Repair if isFLOAT else core.rgvs.Repair
      V = core.rgsf.VerticalCleaner if isFLOAT else core.rgvs.VerticalCleaner

    if opencl:
        NNEDI3 = core.nnedi3cl.NNEDI3CL
        nnedi3_args = dict(nsize=nsize, nns=nns, qual=qual, pscrn=pscrn, device=device)
    else:
        NNEDI3 = core.znedi3.nnedi3 if hasattr(core, 'znedi3') and not isFLOAT else core.nnedi3.nnedi3
        nnedi3_args = dict(nsize=nsize, nns=nns, qual=qual, pscrn=pscrn, exp=exp)

    nn = NNEDI3(c, field=3, **nnedi3_args)
    dbl = core.std.Merge(nn[::2], nn[1::2])
    dblD = core.std.MakeDiff(c, dbl)
    shrpD = dbl.std.MakeDiff(dbl.std.Convolution(matrix=[1]*9 if c.width > 1000 else [1, 2, 1, 2, 4, 2, 1, 2, 1]))
    shrpD = V(shrpD, mode=2)
    DD = R(shrpD, dblD, [rep])
    return core.std.MergeDiff(dbl, DD)

# Taken from old havsfunc
def santiag(
    c: vs.VideoNode,
    strh: int = 1,
    strv: int = 1,
    type: str = 'nnedi3',
    nsize: Optional[int] = None,
    nns: Optional[int] = None,
    qual: Optional[int] = None,
    pscrn: Optional[int] = None,
    int16_prescreener: Optional[bool] = None,
    int16_predictor: Optional[bool] = None,
    exp: Optional[int] = None,
    aa: Optional[int] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    nrad: Optional[int] = None,
    mdis: Optional[int] = None,
    vcheck: Optional[int] = None,
    fw: Optional[int] = None,
    fh: Optional[int] = None,
    halfres: bool = False,
    typeh: Optional[str] = None,
    typev: Optional[str] = None,
    opencl: bool = False,
    device: Optional[int] = None,
) -> vs.VideoNode:
    '''
    santiag v1.6
    Simple antialiasing

    type = "nnedi3", "eedi2", "eedi3" or "sangnom"
    '''

    def santiag_dir(c: vs.VideoNode, strength: int, type: str, fw: Optional[int] = None, fh: Optional[int] = None) -> vs.VideoNode:
        fw = fallback(fw, c.width)
        fh = fallback(fh, c.height)

        c = santiag_stronger(c, strength, type)

        return c.resize.Spline36(fw, fh, src_top=0 if halfres else 0.5)

    def santiag_stronger(c: vs.VideoNode, strength: int, type: str) -> vs.VideoNode:
        if opencl:
            nnedi3 = partial(core.nnedi3cl.NNEDI3CL, nsize=nsize, nns=nns, qual=qual, pscrn=pscrn, device=device)
            if hasattr(core, 'EEDI3CL'):
              eedi3 = partial(core.eedi3m.EEDI3CL, alpha=alpha, beta=beta, gamma=gamma, nrad=nrad, mdis=mdis, vcheck=vcheck, device=device)
            else:
              eedi3 = partial(core.eedi3m.EEDI3, alpha=alpha, beta=beta, gamma=gamma, nrad=nrad, mdis=mdis, vcheck=vcheck)
        else:
            if hasattr(core, 'znedi3'):
              nnedi3 = partial(core.znedi3.nnedi3, nsize=nsize, nns=nns, qual=qual, pscrn=pscrn, int16_prescreener=int16_prescreener, int16_predictor=int16_predictor, exp=exp)
            else:
              nnedi3 = partial(core.nnedi3.nnedi3, nsize=nsize, nns=nns, qual=qual, pscrn=pscrn, int16_prescreener=int16_prescreener, int16_predictor=int16_predictor, exp=exp)
            
            if hasattr(core, 'EEDI3CL'):
              eedi3 = partial(core.eedi3m.EEDI3CL, alpha=alpha, beta=beta, gamma=gamma, nrad=nrad, mdis=mdis, vcheck=vcheck, device=device)
            else:
              eedi3 = partial(core.eedi3m.EEDI3, alpha=alpha, beta=beta, gamma=gamma, nrad=nrad, mdis=mdis, vcheck=vcheck)

        strength = max(strength, 0)
        field = strength % 2
        dh = strength <= 0 and not halfres

        if strength > 0:
            c = santiag_stronger(c, strength - 1, type)

        w = c.width
        h = c.height

        if type == 'nnedi3':
            return nnedi3(c, field=field, dh=dh)
        elif type == 'eedi2':
            if not dh:
                c = c.resize.Point(w, h // 2, src_top=1 - field)
            return c.eedi2.EEDI2(field=field)
        elif type == 'eedi3':
            sclip = nnedi3(c, field=field, dh=dh)
            return eedi3(c, field=field, dh=dh, sclip=sclip)
        elif type == 'sangnom':
            if dh:
                c = c.resize.Spline36(w, h * 2, src_top=-0.25)
            return c.sangnom.SangNom(order=field + 1, aa=aa)
        else:
            raise vs.Error('santiag: unexpected value for type')

    if not isinstance(c, vs.VideoNode):
        raise vs.Error('santiag: this is not a clip')

    type = type.lower()
    typeh = type if typeh is None else typeh.lower()
    typev = type if typev is None else typev.lower()

    w = c.width
    h = c.height
    fwh = fw if strv < 0 else w
    fhh = fh if strv < 0 else h

    if strh >= 0:
        c = santiag_dir(c, strh, typeh, fwh, fhh)
    if strv >= 0:
        c = santiag_dir(c.std.Transpose(), strv, typev, fh, fw).std.Transpose()

    fw = fallback(fw, w)
    fh = fallback(fh, h)
    if strh < 0 and strv < 0:
        c = c.resize.Spline36(fw, fh)
    return c
    
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
      if hasattr(core,'zsmooth'):
        return core.zsmooth.Repair(aa, inputClip, mode=repMode)
      else:
        return core.rgvs.Repair(aa, inputClip, mode=repMode)

    if mode != "edge":
        return aa

    def cround(x: float) -> int:
       return math.floor(x + 0.5) if x > 0 else math.ceil(x - 0.5)

    def scale(value, peak):
      return cround(value * peak / 255) if peak != 1 else value / 255

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

# Taken from sfrom vsutil
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