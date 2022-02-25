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

import vapoursynth as vs
core = vs.core

import havsfunc
import mvsfunc as mvf

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
    if4 = w > 1024 or h > 576

    if pel is None:
        pel = 1 if if4 else 2
    if pel < 2:
        subpixel = min(subpixel, 2)
    pelclip = pel > 1 and subpixel >= 3

    if blksize is None:
        blksize = 16 if if4 else 8
    blk2 = blksize // 2
    if overlap is None:
        overlap = blk2
    ovl2 = overlap // 2
    if truemotion is None:
        truemotion = not if4
    if MVglobal is None:
        MVglobal = truemotion

    planes = [0, 1, 2] if chroma else [0]
    plane0 = (plane != 0)

    if hpad is None:
        hpad = blksize
    if vpad is None:
        vpad = blksize
    limit = havsfunc.scale(limit, peak)
    if limitc is None:
        limitc = limit
    else:
        limitc = havsfunc.scale(limitc, peak)

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

    # Prefilter & Motion Filter
    if mfilter is None:
        mfilter = inputP

    if not GlobalR:
        if preclip:
            pref = prefilter
        elif prefilter <= -1:
            pref = inputP
        elif prefilter == 3:
            expr = 'x {i} < {peak} x {j} > 0 {peak} x {i} - {peak} {j} {i} - / * - ? ?'.format(i=havsfunc
            .scale(16, peak), j=havsfunc.scale(75, peak), peak=peak)
            pref = core.std.MaskedMerge(inputP.dfttest.DFTTest(tbsize=1, slocation=[0.0,4.0, 0.2,9.0, 1.0,15.0], planes=planes),
                                        inputP,
                                        mvf.GetPlane(inputP, 0).std.Expr(expr=[expr]),
                                        planes=planes)
        elif prefilter >= 4:
            if chroma:
                pref = KNLMeansCL(inputP, d=1, a=1, h=7)
            else:
                pref = inputP.knlm.KNLMeansCL(d=1, a=1, h=7)
        else:
            pref = havsfunc.MinBlur(inputP, r=prefilter, planes=planes)
    else:
        pref = inputP

    # Default Auto-Prefilter - Luma expansion TV->PC (up to 16% more values for motion estimation)
    if not GlobalR:
        pref = havsfunc.DitherLumaRebuild(pref, s0=Str, c=Amp, chroma=chroma)

    # Subpixel 3
    if pelclip:
        nnediMode = 'znedi3' if opencl else 'znedi'
        cshift = 0.25 if pel == 2 else 0.375
        pclip = nnedi3_resample.nnedi3_resample(pref, w * pel, h * pel, src_left=cshift, src_top=cshift, nns=4, mode=nnediMode, device=device)
        if not GlobalR:
            pclip2 = nnedi3_resample.nnedi3_resample(inputP, w * pel, h * pel, src_left=cshift, src_top=cshift, nns=4, mode=nnediMode, device=device)

    # Motion vectors search
    global bv6, bv4, bv3, bv2, bv1, fv1, fv2, fv3, fv4, fv6
    super_args = dict(hpad=hpad, vpad=vpad, pel=pel)
    analyse_args = dict(blksize=blksize, search=search, chroma=chroma, truemotion=truemotion, global_=MVglobal, overlap=overlap, dct=dct)
    if RefineMotion:
        recalculate_args = dict(thsad=halfthSAD, blksize=halfblksize, search=search, chroma=chroma, truemotion=truemotion, overlap=halfoverlap, dct=dct)

    if pelclip:
        super_search = pref.mv.Super(chroma=chroma, rfilter=4, pelclip=pclip, **super_args)
    else:
        super_search = pref.mv.Super(chroma=chroma, sharp=subpixel, rfilter=4, **super_args)

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
            if tr >= 3:
                output = core.mv.Degrain3(mfilter, super_render, bv1, fv1, bv2, fv2, bv3, fv3, **degrain_args)
            elif tr == 2:
                output = core.mv.Degrain2(mfilter, super_render, bv1, fv1, bv2, fv2, **degrain_args)
            else:
                output = core.mv.Degrain1(mfilter, super_render, bv1, fv1, **degrain_args)

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
                    return havsfunc.Weave(ContraSharpening(output, CClip, planes=planes), tff=tff)
                else:
                    return havsfunc.Weave(LSFmod(output, strength=contrasharp, source=CClip, Lmode=0, soothe=False, defaults='slow'), tff=tff)
            elif ifC:
                return havsfunc.ContraSharpening(output, CClip, planes=planes)
            else:
                return havsfunc.LSFmod(output, strength=contrasharp, source=CClip, Lmode=0, soothe=False, defaults='slow')
        elif interlaced:
            return havsfunc.Weave(output, tff=tff)
        else:
            return output
    else:
        return input
