import vapoursynth as vs
from vapoursynth import core

from typing import Sequence, Union, Optional


import math
import warnings

from helpers import scale, Padding, DitherLumaRebuild, DFTTest, GetPlane, KNLMeansCL, get_expr
from misc import MV, MinBlur
from sharpen import LSFmod, ContraSharpening
from nnedi3_resample import nnedi3_resample


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


def SMDegrain(input, tr=2, thSAD=300, thSADC=None, RefineMotion=False, contrasharp=None, CClip=None, interlaced=False, tff=None, plane=4, Globals=0, pel=None, subpixel=2, prefilter=-1, mfilter=None,
              blksize=None, overlap=None, search=4, truemotion=None, MVglobal=None, dct=0, limit=255, limitc=None, thSCD1=400, thSCD2=130, chroma=True, hpad=None, vpad=None, Str=1.0, Amp=0.0625, opencl=False, device=None):
    # RefineMotion: False/0 = off, True/1 = one Recalculate pass, N = N passes each halving the block size.
    # limit/limitc: maximum pixel change on the 8-bit scale (255 = off), scaled to the clip's bit depth by MV.
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
    if overlap is None:
        overlap = blksize // 2
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
    if limitc is None:
        limitc = limit

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
    if not (isinstance(RefineMotion, int) and RefineMotion >= 0):
        raise vs.Error("SMDegrain: 'RefineMotion' must be a bool or a non-negative integer")
    if not (isinstance(tr, int) and tr >= 1):
        raise vs.Error("SMDegrain: 'tr' must be a positive integer")
    # not sure whether this is still true, so I disabled it
    #if not chroma and plane != 0:
    #    raise vs.Error('SMDegrain: Denoising chroma with luma only vectors is bugged in mvtools and thus unsupported')

    # Each RefineMotion pass halves the block size; 4x4 is the smallest block size.
    refine_passes = int(RefineMotion)
    max_passes = 0
    while (blksize >> (max_passes + 1)) >= 4:
        max_passes += 1
    if refine_passes > max_passes:
        warnings.warn(f'SMDegrain: RefineMotion={refine_passes} exceeds what blksize={blksize} allows, using {max_passes}', stacklevel=2)
        refine_passes = max_passes

    max_tr = MV.max_degrain_radius(input)
    if max_tr is not None and tr > max_tr:
        warnings.warn(f'SMDegrain: tr={tr} exceeds the largest radius the motion vector plugin supports, using {max_tr}', stacklevel=2)
        tr = max_tr

    # Input preparation for Interlacing
    if not interlaced:
        inputP = input
    else:
        inputP = input.std.SeparateFields(tff=tff)
        h = h/2

    # Prefilter & Motion Filter
    if mfilter is None:
        mfilter = inputP
    EXPR = get_expr()
    if not GlobalR:
        if preclip:
            pref = prefilter
        elif prefilter <= -1:
            pref = inputP
        elif prefilter == 3:
            expr = 'x {i} < {peak} x {j} > 0 {peak} x {i} - {peak} {j} {i} - / * - ? ?'.format(i=scale(16, peak), j=scale(75, peak), peak=peak)
            # Takes the first DFTTest implementation that is loaded, GPU ones first.
            filtered = DFTTest(inputP, tbsize=1, slocation=[0.0,4.0, 0.2,9.0, 1.0,15.0], planes=planes)
            pref = core.std.MaskedMerge(filtered, inputP, EXPR(GetPlane(inputP, 0), expr=[expr]), planes=planes)
        elif prefilter >= 4:
            # Takes the first NLMeans implementation that is loaded: nlm_ispc (CPU),
            # nlm_cuda (CUDA), knlm (KNLMeansCL, OpenCL). Usually exactly one of
            # them is loaded for this filter - nlm_ispc when 'opencl' is off,
            # otherwise whichever GPU port was chosen.
            # device_type/device_id only reach knlm and nlm_cuda; 'device' is -1 or None
            # when no device was pinned, and neither plugin takes that as an index.
            pref = KNLMeansCL(inputP, d=1, a=1, h=7, device_type='gpu' if opencl else None,
                              device_id=device if isinstance(device, int) and device >= 0 else None)
        else:
            pref = MinBlur(inputP, r=prefilter, planes=planes)
    else:
        pref = inputP

    # Default Auto-Prefilter - Luma expansion TV->PC (up to 16% more values for motion estimation)
    if not GlobalR:
        pref = DitherLumaRebuild(pref, s0=Str, c=Amp, chroma=chroma)

    # Motion vectors search
    super_args = dict(hpad=hpad, vpad=vpad, pel=pel, blksize=blksize, overlap=overlap)
    # Subpixel 3
    if pelclip:
      nnediMode = 'nnedi3cl' if opencl else 'znedi3'
      cshift = 0.25 if pel == 2 else 0.375
      pclip = nnedi3_resample(pref, w * pel, h * pel, src_left=cshift, src_top=cshift, nns=4, mode=nnediMode, device=device)
      if not GlobalR:
         pclip2 = nnedi3_resample(inputP, w * pel, h * pel, src_left=cshift, src_top=cshift, nns=4, mode=nnediMode, device=device)
      super_search = MV.Super(pref, chroma=chroma, rfilter=4, pelclip=pclip, **super_args)
    else:
      super_search = MV.Super(pref, chroma=chroma, sharp=subpixel, rfilter=4, **super_args)

    refine_super = None
    if not GlobalR:
        if pelclip:
            super_render = MV.Super(inputP, levels=1, chroma=plane0, pelclip=pclip2, **super_args)
            if refine_passes:
                refine_super = MV.Super(pref, levels=1, chroma=chroma, pelclip=pclip, **super_args)
        else:
            super_render = MV.Super(inputP, levels=1, chroma=plane0, sharp=subpixel, **super_args)
            if refine_passes:
                refine_super = MV.Super(pref, levels=1, chroma=chroma, sharp=subpixel, **super_args)
    else:
        super_render = super_search
        if refine_passes:
            refine_super = super_render

    search_params = dict(blksize=blksize, search=search, chroma=chroma, truemotion=truemotion, global_=MVglobal, overlap=overlap, dct=dct)
    # Overlap of a refine pass: at most half its block size, a multiple of the chroma subsampling.
    overlap_align = (1 << max(input.format.subsampling_w, input.format.subsampling_h)) if chroma else 1
    refine_params = []
    for i in range(1, refine_passes + 1):
        refine_blksize = blksize >> i
        refine_overlap = min(overlap >> i, refine_blksize // 2)
        refine_overlap -= refine_overlap % overlap_align
        refine_params.append(dict(thsad=thSAD2, blksize=refine_blksize, search=search, chroma=chroma, truemotion=truemotion, overlap=refine_overlap, dct=dct))

    vectors = get_motion_vectors(super_search, refine_super, search_params, refine_params, tr, interlaced)

    # Finally, MDegrain
    if not GlobalO:
        degrain_clip = mfilter if interlaced else inputP
        output = MV.Degrain(degrain_clip, super_render, *vectors, thsad=thSAD, thsadc=thSADC, plane=plane, limit=limit, limitc=limitc, thscd1=thSCD1, thscd2=thSCD2)

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
                    return Weave(LSFmod(output, strength=contrasharp, source=CClip, Lmode=0, soothe=False, defaults='slow'), tff=tff)
            elif ifC:
                return ContraSharpening(output, CClip, planes=planes)
            else:
                return LSFmod(output, strength=contrasharp, source=CClip, Lmode=0, soothe=False, defaults='slow')
        elif interlaced:
            return Weave(output, tff=tff)
        else:
            return output
    else:
        return input

# Helpers

def get_motion_vectors(super_search, refine_super, search_params, refine_params, tr, interlaced):
    '''Returns [bv1, fv1, bv2, fv2, ...] up to radius tr; separated fields use every second field (same parity).'''
    vectors = MV.AnalyseMany(super_search, radius=tr, delta=2 if interlaced else 1, **search_params)
    for params in refine_params:
        vectors = MV.Recalculate(refine_super, vectors, **params)
    return vectors

def Weave(clip: vs.VideoNode, tff: Optional[bool] = None) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('Weave: this is not a clip')

    if tff is None:
        with clip.get_frame(0) as f:
            if f.props.get('_Field') not in [1, 2]:
                raise vs.Error('Weave: tff was not specified and field order could not be determined from frame properties')

    return clip.std.DoubleWeave(tff=tff)[::2]
