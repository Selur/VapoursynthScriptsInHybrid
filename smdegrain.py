import vapoursynth as vs
from vapoursynth import core

from typing import Sequence, Union, Optional


import math

from helpers import scale, Padding, DitherLumaRebuild
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
    # not sure whether this is still true, so I disabled it
    #if not chroma and plane != 0:
    #    raise vs.Error('SMDegrain: Denoising chroma with luma only vectors is bugged in mvtools and thus unsupported')

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
    EXPR = core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr
    if not GlobalR:
        if preclip:
            pref = prefilter
        elif prefilter <= -1:
            pref = inputP
        elif prefilter == 3:
            expr = 'x {i} < {peak} x {j} > 0 {peak} x {i} - {peak} {j} {i} - / * - ? ?'.format(i=scale(16, peak), j=scale(75, peak), peak=peak)
            if hasattr(core,'dfttest2_nvrtc'):
              import dfttest2
              DFTTest = dfttest2.DFTTest
              filtered = DFTTest(inputP, tbsize=1, slocation=[0.0,4.0, 0.2,9.0, 1.0,15.0], planes=planes, backend=dfttest2.Backend.NVRTC)
            else:
              DFTTest = core.dfttest.DFTTest
              filtered = DFTTest(inputP, tbsize=1, slocation=[0.0,4.0, 0.2,9.0, 1.0,15.0], planes=planes)
            pref = core.std.MaskedMerge(filtered, inputP, EXPR(GetPlane(inputP, 0), expr=[expr]), planes=planes)            
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

    
    analyse_args = dict(blksize=blksize, search=search, chroma=chroma, truemotion=truemotion, global_=MVglobal, overlap=overlap, dct=dct)
    if RefineMotion:
        recalculate_args = dict(thsad=halfthSAD, blksize=halfblksize, search=search, chroma=chroma, truemotion=truemotion, overlap=halfoverlap, dct=dct)


    if not GlobalR:
        if pelclip:
            super_render = MV.Super(inputP, levels=1, chroma=plane0, pelclip=pclip2, **super_args)
            if RefineMotion:
                Recalculate = MV.Super(pref, levels=1, chroma=chroma, pelclip=pclip, **super_args)
        else:
            super_render = MV.Super(inputP, levels=1, chroma=plane0, sharp=subpixel, **super_args)
            if RefineMotion:
                Recalculate = MV.Super(pref, levels=1, chroma=chroma, sharp=subpixel, **super_args)

        if interlaced:
            if tr > 2:
                bv6 = MV.Analyse(super_search, isb=True, delta=6, **analyse_args)
                fv6 = MV.Analyse(super_search, isb=False, delta=6, **analyse_args)
                if RefineMotion:
                    bv6 = MV.Recalculate(Recalculate, bv6, **recalculate_args)
                    fv6 = MV.Recalculate(Recalculate, fv6, **recalculate_args)
            if tr > 1:
                bv4 = MV.Analyse(super_search, isb=True, delta=4, **analyse_args)
                fv4 = MV.Analyse(super_search, isb=False, delta=4, **analyse_args)
                if RefineMotion:
                    bv4 = MV.Recalculate(Recalculate, bv4, **recalculate_args)
                    fv4 = MV.Recalculate(Recalculate, fv4, **recalculate_args)
        else:
            if tr > 2:
                bv3 = MV.Analyse(super_search, isb=True, delta=3, **analyse_args)
                fv3 = MV.Analyse(super_search, isb=False, delta=3, **analyse_args)
                if RefineMotion:
                    bv3 = MV.Recalculate(Recalculate, bv3, **recalculate_args)
                    fv3 = MV.Recalculate(Recalculate, fv3, **recalculate_args)
            bv1 = MV.Analyse(super_search, isb=True, delta=1, **analyse_args)
            fv1 = MV.Analyse(super_search, isb=False, delta=1, **analyse_args)
            if RefineMotion:
                bv1 = MV.Recalculate(Recalculate, bv1, **recalculate_args)
                fv1 = MV.Recalculate(Recalculate, fv1, **recalculate_args)
        if interlaced or tr > 1:
            bv2 = MV.Analyse(super_search, isb=True, delta=2, **analyse_args)
            fv2 = MV.Analyse(super_search, isb=False, delta=2, **analyse_args)
            if RefineMotion:
                bv2 = MV.Recalculate(Recalculate, bv2, **recalculate_args)
                fv2 = MV.Recalculate(Recalculate, fv2, **recalculate_args)
    else:
        super_render = super_search

    # Finally, MDegrain
    search_params = dict(blksize=blksize, search=search, chroma=chroma, truemotion=truemotion, global_=MVglobal, overlap=overlap, dct=dct)
    refine_params = dict(thsad=thSAD2, blksize=blksize // 2, search=search, chroma=chroma, truemotion=truemotion, overlap=overlap // 2, dct=dct) if RefineMotion else None
        
    refine_super = None
    if RefineMotion:
        # If Recalculate was created earlier in the function it will be in locals()
        refine_super = Recalculate if 'Recalculate' in locals() else super_render
    vectors = get_motion_vectors(super_search, refine_super, search_params, refine_params, tr, interlaced)
    
    degrain_args = dict(thsad=thSAD, thsadc=thSADC, plane=plane, limit=limit, limitc=limitc, thscd1=thSCD1, thscd2=thSCD2)
    if not GlobalO:
      if interlaced:
        if tr >= 3:
            output = MV.Degrain3(mfilter, super_render, bv2, fv2, bv4, fv4, bv6, fv6, **degrain_args)
        elif tr == 2:
            output = MV.Degrain2(mfilter, super_render, bv2, fv2, bv4, fv4, **degrain_args)
        else:
            output = MV.Degrain1(mfilter, super_render, bv2, fv2, **degrain_args)
      else:
        if tr >= 6:
          output = MV.Degrain6(inputP, super_render, vectors['bv1'], vectors['fv1'], vectors['bv2'], vectors['fv2'], vectors['bv3'], vectors['fv3'], vectors['bv4'], vectors['fv4'], vectors['bv5'], vectors['fv5'], vectors['bv6'], vectors['fv6'], **degrain_args)
        elif tr == 5:
          output = MV.Degrain5(inputP, super_render, vectors['bv1'], vectors['fv1'], vectors['bv2'], vectors['fv2'], vectors['bv3'], vectors['fv3'], vectors['bv4'], vectors['fv4'], vectors['bv5'], vectors['fv5'], **degrain_args)
        elif tr == 4: 
          output = MV.Degrain4(inputP, super_render, vectors['bv1'], vectors['fv1'], vectors['bv2'], vectors['fv2'], vectors['bv3'], vectors['fv3'], vectors['bv4'], vectors['fv4'], **degrain_args)
        elif tr == 3:
          output = MV.Degrain3(inputP, super_render, vectors['bv1'], vectors['fv1'], vectors['bv2'], vectors['fv2'], vectors['bv3'], vectors['fv3'], **degrain_args)
        elif tr == 2:
          output = MV.Degrain2(inputP, super_render, vectors['bv1'], vectors['fv1'], vectors['bv2'], vectors['fv2'], **degrain_args)
        else:
          output = MV.Degrain1(inputP, super_render, vectors['bv1'], vectors['fv1'], **degrain_args)

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
     
def get_motion_vectors(super_search, refine, search_params, refine_params, tr, interlaced):
    vectors = {}
    
    if tr >= 1:
        vectors['bv1'] = MV.Analyse(super_search, isb=True, delta=1, **search_params)
        vectors['fv1'] = MV.Analyse(super_search, isb=False, delta=1, **search_params)
        if refine:
            vectors['bv1'] = MV.Recalculate(refine, vectors['bv1'], **refine_params)
            vectors['fv1'] = MV.Recalculate(refine, vectors['fv1'], **refine_params)
    
    if interlaced or tr >= 2:
        vectors['bv2'] = MV.Analyse(super_search, isb=True, delta=2, **search_params)
        vectors['fv2'] = MV.Analyse(super_search, isb=False, delta=2, **search_params)
        if refine:
            vectors['bv2'] = MV.Recalculate(refine, vectors['bv2'], **refine_params)
            vectors['fv2'] = MV.Recalculate(refine, vectors['fv2'], **refine_params)
    
    if tr >= 3:
        for i in range(3, tr + 1):
            vectors[f'bv{i}'] = MV.Analyse(super_search, isb=True, delta=i, **search_params)
            vectors[f'fv{i}'] = MV.Analyse(super_search, isb=False, delta=i, **search_params)
            if refine:
                vectors[f'bv{i}'] = MV.Recalculate(refine, vectors[f'bv{i}'], **refine_params)
                vectors[f'fv{i}'] = MV.Recalculate(refine, vectors[f'fv{i}'], **refine_params)
    
    return vectors
    
def Weave(clip: vs.VideoNode, tff: Optional[bool] = None) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('Weave: this is not a clip')

    if tff is None:
        with clip.get_frame(0) as f:
            if f.props.get('_Field') not in [1, 2]:
                raise vs.Error('Weave: tff was not specified and field order could not be determined from frame properties')

    return clip.std.DoubleWeave(tff=tff)[::2]
