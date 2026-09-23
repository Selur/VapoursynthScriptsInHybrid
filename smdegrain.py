import vapoursynth as vs
from vapoursynth import core

from typing import Sequence, Union, Optional


import math
import warnings

from helpers import scale, cround, Padding, DitherLumaRebuild, DFTTest, GetPlane, KNLMeansCL, get_expr, is_limited_range, BM3D
from misc import MV, MinBlur
from gaussblur import GaussBlur
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
              blksize=None, overlap=None, search=4, truemotion=None, MVglobal=None, dct=0, limit=255, limitc=None, thSCD1=None, thSCD2=130, chroma=True, hpad=None, vpad=None, Str=1.0, Amp=0.0625, opencl=False, device=None,
              tv_range=None, v4formulas=False, LFR=False, DCTFlicker=False, bm3d_backend=None):
    # RefineMotion: False/0 = off, True/1 = one Recalculate pass, N = N passes each halving the block size.
    # limit/limitc: maximum pixel change on the 8-bit scale (255 = off), scaled to the clip's bit depth by MV.
    # tv_range: range of the input for the luma rebuild of the motion search clip; None = read from the frame properties.
    # v4formulas: thSADC, thSCD1, the refine threshold and the motion search parameters follow Dogway's SMDegrain 4.x.
    # LFR: Dogway's Low Frequency Restore; True = 300 Hz cutoff at 1920 wide, a number = cutoff in Hz (at least 50), False/0 = off.
    # DCTFlicker: with LFR, calms the restored low frequencies with a second, temporal SMDegrain pass on them.
    # bm3d_backend: BM3D implementation for prefilter 5 ('bm3dcuda', 'bm3dhip', 'bm3dcpu', 'bm3d'); None = first one loaded.
    if not isinstance(input, vs.VideoNode):
        raise vs.Error('SMDegrain: This is not a clip')

    if input.format.color_family == vs.GRAY:
        plane = 0
        chroma = False

    peak = (1 << input.format.bits_per_sample) - 1

    # Defaults & Conditionals
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

    is_hd = w > 1099 or h > 599
    is_uhd = w > 2599 or h > 1499
    if v4formulas:
        csad = _scale_csad(plane in (0, 4), chroma, input.format, is_hd)
        if thSADC is None:
            thSADC = cround(thSAD * 0.755 * 0.25 * math.exp(csad * 0.693))
        if thSCD1 is None:
            thSCD1 = cround(0.35 * thSAD + 260)
        thSAD_refine = cround(math.exp(-101 / (thSAD * 0.83)) * 360)
    else:
        if thSADC is None:
            thSADC = thSAD // 2
        if thSCD1 is None:
            thSCD1 = 400
        thSAD_refine = thSAD // 2

    if tv_range is None:
        tv_range = is_limited_range(input)

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
    if not (isinstance(LFR, (bool, int, float)) and LFR >= 0):
        raise vs.Error("SMDegrain: 'LFR' must be a bool or a cutoff frequency in Hz")
    lfr_active = LFR is True or (not isinstance(LFR, bool) and LFR > 0)
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
        elif prefilter == 5:
            pref = _bm3d_prefilter(inputP, chroma, device, bm3d_backend)
        elif prefilter == 6:
            pref = _dgdenoise_prefilter(inputP, chroma, device)
        elif prefilter > 6:
            raise vs.Error("SMDegrain: 'prefilter' must be -1 to 6 or a clip")
        elif prefilter == 4:
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
        pref = DitherLumaRebuild(pref, s0=Str, c=Amp, chroma=chroma, tv_range=tv_range)
        # LFR: low frequency expansion for the motion search, the higher thSAD the more protection.
        if lfr_active:
            pref = _lfr_unsharp(pref, thSAD / 1800.0, planes)

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
    refine_search = {}
    if v4formulas:
        searchparam = (2 if is_uhd else 5) if refine_passes and truemotion else (1 if is_uhd else 2)
        search_params.update(searchparam=searchparam, pelsearch=max(0, searchparam * 2 - 2), pglobal=11, plevel=0)
        refine_search = dict(searchparam=max(0, cround(math.exp(0.69 * searchparam - 1.79) - 0.67)))
    # Overlap of a refine pass: at most half its block size, a multiple of the chroma subsampling.
    overlap_align = (1 << max(input.format.subsampling_w, input.format.subsampling_h)) if chroma else 1
    refine_params = []
    for i in range(1, refine_passes + 1):
        refine_blksize = blksize >> i
        refine_overlap = min(overlap >> i, refine_blksize // 2)
        refine_overlap -= refine_overlap % overlap_align
        refine_params.append(dict(thsad=thSAD_refine, blksize=refine_blksize, search=search, chroma=chroma, truemotion=truemotion, overlap=refine_overlap, dct=dct, **refine_search))

    vectors = get_motion_vectors(super_search, refine_super, search_params, refine_params, tr, interlaced)

    # Finally, MDegrain
    if not GlobalO:
        degrain_clip = mfilter if interlaced else inputP
        output = MV.Degrain(degrain_clip, super_render, *vectors, thsad=thSAD, thsadc=thSADC, plane=plane, limit=limit, limitc=limitc, thscd1=thSCD1, thscd2=thSCD2)

    # Low Frequency Restore (luma only): puts the low frequencies of mfilter back where the vectors are unreliable.
    if lfr_active and not GlobalO:
        sigma = _lfr_sigma(LFR, input.width, input.height)
        luma = GetPlane(output, 0)
        mfilter_lp = GaussBlur(GetPlane(mfilter, 0), sigma)
        if DCTFlicker:
            flicker_pref = mfilter_lp if mfilter.format.color_family == vs.GRAY else core.std.ShufflePlanes([mfilter_lp, mfilter], [0, 1, 2], vs.YUV)
            flicker_pass = SMDegrain(mfilter, tr=math.ceil(tr / 3), thSAD=thSAD // 2, blksize=blksize, prefilter=flicker_pref, pel=1,
                                     Str=1.0, tv_range=False, plane=0, chroma=False, truemotion=False, v4formulas=v4formulas)
            mfilter_lp = core.zsmooth.TemporalRepair(mfilter_lp, GaussBlur(GetPlane(flicker_pass, 0), sigma), mode=4)
        mask = _lfr_mask(vectors, GetPlane(mfilter, 0), 2.222 if DCTFlicker else 0.5, thSCD1, thSCD2)
        luma = EXPR([luma, GaussBlur(luma, sigma), mfilter_lp, mask], expr=[f'x y z - a * {peak} / -'])
        output = luma if output.format.color_family == vs.GRAY else core.std.ShufflePlanes([luma, output], [0, 1, 2], vs.YUV)

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

def _lfr_sigma(LFR, width: int, height: int) -> float:
    '''Gaussian sigma of Dogway's Low Frequency Restore for a cutoff in Hz (-3 dB); True = 3.46 at 1920 wide (300 Hz).'''
    if LFR is True:
        return 3.46 * width / 1920.0
    hz = max(float(LFR), 50.0)
    return max(width, height) * 2.0 / (math.sqrt(math.log(2) / 2) * hz * 2 * math.pi)

def _lfr_unsharp(clip: vs.VideoNode, strength: float, planes) -> vs.VideoNode:
    '''Dogway's ex_unsharp(strength, Fc=width/8, th=0): clip + strength * (clip - blur), blur = 0.8 * box(r) + 0.2 * box(r + 1), vertical then horizontal.'''
    radius = max(math.ceil(2 * max(clip.width, clip.height) / (clip.width / 8.0)) - 1, 1)
    EXPR = get_expr()
    mix = ['x 0.8 * y 0.2 * +' if p in planes else '' for p in range(clip.format.num_planes)]
    blur = EXPR([clip.std.BoxBlur(planes=planes, hradius=0, hpasses=0, vradius=radius, vpasses=1),
                 clip.std.BoxBlur(planes=planes, hradius=0, hpasses=0, vradius=radius + 1, vpasses=1)], expr=mix)
    blur = EXPR([blur.std.BoxBlur(planes=planes, hradius=radius, hpasses=1, vradius=0, vpasses=0),
                 blur.std.BoxBlur(planes=planes, hradius=radius + 1, hpasses=1, vradius=0, vpasses=0)], expr=mix)
    return EXPR([clip, blur], expr=[f'x x y - {strength} * +' if p in planes else '' for p in range(clip.format.num_planes)])

def _lfr_mask(vectors, luma: vs.VideoNode, gamma: float, thscd1, thscd2) -> vs.VideoNode:
    '''Average SAD mask of all vectors (Dogway: ml=50, ysc=255) in the format of luma, full range.'''
    bits = luma.format.bits_per_sample
    masks = []
    for vector in vectors:
        if MV.use_mvu:
            mask = MV.Mask(luma, vector, ml=50, gamma=gamma, kind=1, ysc=255, thscd1=thscd1, thscd2=thscd2)
        else:
            # mvtools' Mask takes 8-bit clips only and measures the SAD at the vectors' bit depth; mvutensils
            # normalizes it to 8 bit, which Dogway's ml=50 is meant for.
            mask = MV.Mask(core.resize.Point(luma, format=vs.GRAY8), vector, ml=50 * (1 << (bits - 8)), gamma=gamma,
                           kind=1, ysc=255, thscd1=thscd1, thscd2=thscd2)
        # A mask is full range whatever it is tagged with: the tag would win over range_in (mvtools' mask inherits
        # the limited tag of its 8-bit input, mvutensils < 9 tagged it limited by mistake), so drop it before converting.
        mask = core.std.RemoveFrameProps(GetPlane(mask, 0), props=['_Range', '_ColorRange'])
        masks.append(core.resize.Point(mask, format=luma.format.id, range_in=1, range=1))
    return _average(masks)

def _average(clips):
    '''Pixel mean of any number of clips; Expr takes at most 26 inputs, so larger lists are split.'''
    count = len(clips)
    if count == 1:
        return clips[0]
    if count > 26:
        half = count // 2
        return get_expr()([_average(clips[:half]), _average(clips[half:])],
                          expr=[f'x {half / count} * y {(count - half) / count} * +'])
    variables = 'xyzabcdefghijklmnopqrstuvw'
    return get_expr()(clips, expr=[' '.join(variables[:count]) + ' +' * (count - 1) + f' {count} /'])

def _bm3d_prefilter(clip: vs.VideoNode, chroma: bool, device, backend=None) -> vs.VideoNode:
    '''BM3D prefilter like Dogway's ex_BM3D preset "normal" (sigma 10, chroma 5, radius 1) on the chosen or first loaded BM3D plugin.'''
    fmt = clip.format
    params = dict(radius=1, block_step=4, bm_range=16, ps_range=5, device_id=device if isinstance(device, int) else None,
                  backend=backend)
    if chroma and fmt.color_family != vs.GRAY:
        # Chroma is denoised together with luma (CBM3D), which needs 4:4:4.
        work = BM3D(core.resize.Bicubic(clip, format=vs.YUV444PS), sigma=[10.0, 5.0, 5.0], chroma=True, **params)
        return core.resize.Bicubic(work, format=fmt.id)
    luma = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    work = BM3D(core.resize.Point(luma, format=vs.GRAYS), sigma=[10.0], **params)
    work = core.resize.Point(work, format=luma.format.id)
    return work if fmt.color_family == vs.GRAY else core.std.ShufflePlanes([work, clip], [0, 1, 2], vs.YUV)

def _dgdenoise_prefilter(clip: vs.VideoNode, chroma: bool, device) -> vs.VideoNode:
    '''DGDenoise prefilter with Dogway's strengths (luma 0.10, chroma 0.05); DGDenoise takes YV12, YUV420P16 and YUV444P16 only.'''
    fmt = clip.format
    gray = fmt.color_family == vs.GRAY
    work = core.std.ShufflePlanes(clip, [0, 0, 0], vs.YUV) if gray else clip
    if gray or (fmt.subsampling_w, fmt.subsampling_h) == (0, 0):
        work_format = vs.YUV444P16
    elif fmt.id == vs.YUV420P8:
        work_format = vs.YUV420P8
    else:
        work_format = vs.YUV420P16
    if work.format.id != work_format:
        work = core.resize.Bicubic(work, format=work_format)
    work = core.dgdenoise.DGDenoise(work, strength=0.10, cstrength=0.05 if chroma and not gray else 0.0,
                                    device=device if isinstance(device, int) and device >= 0 else 255)
    if gray:
        work = core.std.ShufflePlanes(work, 0, vs.GRAY)
    return work if work.format.id == fmt.id else core.resize.Bicubic(work, format=fmt.id)

def _scale_csad(luma: bool, chroma: bool, fmt: vs.VideoFormat, is_hd: bool) -> int:
    '''Dogway's scaleCSAD exponent (MDegrain mode): how much chroma counts in the SAD relative to luma.'''
    if not luma:
        return 2
    if not chroma:
        return -2
    subsampling = (fmt.subsampling_w, fmt.subsampling_h)
    if subsampling == (0, 0):
        return 2 if is_hd else 1
    if subsampling == (2, 0):
        return 0 if is_hd else -1
    return 2 if is_hd else 0

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
