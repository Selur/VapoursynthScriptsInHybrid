
# Script by pifroggi https://github.com/pifroggi/vs_temporalfix
# or tepete and pifroggi on Discord

# Based on plugins and functions from many different people. See function comments and readme requirements for details.

import vapoursynth as vs
from .vs_temporalfix_utils import *

core = vs.core


def _motion_search_prefilter(clip, thsad=250, tr=6):
    # creates a temporally extremely stable reference for better motion vector estimation, but with lots of ghosting
    # based on SpotLess function from G41Fun https://github.com/Vapoursynth-Plugins-Gitify/G41Fun
    # which was modified from lostfunc https://github.com/theChaosCoder/lostfunc/blob/v1/lostfunc.py#L10
    # which was a port of Didée's original avisynth function https://forum.doom9.org/showthread.php?p=1402690

    # first pass with temporal median
    bs  = 128  # large blocksize to reduce warping
    pel = 1
    sup = core.mv.Super(clip, pel=pel, sharp=1, rfilter=4, hpad=bs // 2, vpad=bs // 2)
    analyse_args = dict(blksize=bs, overlap=bs // 2, search=4, searchparam=2, truemotion=False)
    bv1 = core.mv.Analyse(sup, isb=True,  delta=1, **analyse_args)
    fv1 = core.mv.Analyse(sup, isb=False, delta=1, **analyse_args)
    bc1 = core.mv.Compensate(clip, sup, bv1)
    fc1 = core.mv.Compensate(clip, sup, fv1)
    fcb = core.std.Interleave([fc1, clip, bc1])
    clip = temporal_median(fcb, radius=1, planes=0)[1::3]

    # second pass with degrain and a wide radius (improves pans, zooms and similar, reduces warping)
    bs  = 128  # large blocksize to reduce warping
    pel = 1
    sup = core.mv.Super(clip, pel=pel, sharp=1, rfilter=4)
    analyse_args = dict(blksize=bs, overlap=0, search=4, searchparam=1, truemotion=False)
    vecs = mv_analyze(sup, tr, analyse_args)
    return mv_degrain(clip, sup, vecs, tr, dict(thsad=thsad, plane=0))


def classic(clip, strength=500, tr=6, denoise=False, exclude=None, debug=False):
    """Add temporal coherence to single image AI upscaling models. Also known as temporal consistency, line wiggle fix, stabilization, deshimmering. 
    This is the original CPU based version. It can run on any CPU, but may miss some areas, is slow and only works well for 2D animation.

    Args:
        clip: Temporally unstable upscaled clip.
        strength: Suppression strength of temporal inconsistencies. Higher means more aggressive. `400-700` works great in most cases. 
            The best way to finetune is to find a static scene and adjust till lines and details are stable. 
            Reduce if you get blending/ghosting on small movements, especially in dark or hazy scenes.
        tr: Temporal radius sets the number of frames to average over. `
            Higher means more stable, especially on slow pans and zooms, but is slower. `6` works great in most cases. 
            The best way to finetune is to find a slow pan or zoom and adjust till lines and details are stable.
        denoise: Removes grain and low frequency noise/flicker left over by the main processing step. Only enable if these issues actually exist! 
            It risks to remove some details like every denoiser, but is useful if you're planning to denoise anyway and has the benefit of almost
            no performance impact compared to using an additional denoising filter.
        exclude: Optionally exclude scenes with intended temporal inconsistencies, or in case this causes unexpected issues. 
            Example setting 3 scenes: `exclude="[10 20] [600 900] [2000 2500]"`. 
            First number in the brackets is the first frame of the scene, the second number is the last frame (inclusive).
        debug: Shows areas that will be left untouched in pink. This includes areas with high motion, scene changes and previously excluded scenes. 
            May help while tuning parameters to see if the area is even affected.

            Tip: Crop any black borders on the input clip, as those may cause ghosting on bright frames.  
            Tip: There is a big drop in performance for `tr > 6`, due to switching from `mvtools` to `mvtools-sf`, which is slower.
    """
    
    # based on SMDegrain function from G41Fun https://github.com/Vapoursynth-Plugins-Gitify/G41Fun
    # which is a modification of SMDegrain from havsfunc https://github.com/HomeOfVapourSynthEvolution/havsfunc/blob/r31/havsfunc.py#L3186
    # which is a port of SMDegrain from avisynth https://forum.videohelp.com/threads/369142

    #checks
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("vs_temporalfix: Clip must be a vapoursynth clip.")
    if clip.format.id == vs.PresetVideoFormat.NONE or clip.width == 0 or clip.height == 0:
        raise TypeError("vs_temporalfix: Clip must have constant format and dimensions.")
    if strength < 0:
        raise ValueError("vs_temporalfix: Strength can not be negative.")
    if tr < 1:
        raise ValueError("vs_temporalfix: Temporal radius (tr) must be at least 1.")
    if not isinstance(denoise, bool):
        raise TypeError("vs_temporalfix: Denoise must be either True or False.")
    if not isinstance(debug, bool):
        raise TypeError("vs_temporalfix: Debug must be either True or False.")

    # original props
    props       = clip.get_frame(0).props
    orig_format = clip.format.id
    orig_family = clip.format.color_family
    orig_width  = clip.width
    dflt_range  = vs.RANGE_FULL if orig_family == vs.RGB else vs.RANGE_LIMITED  # if not tagged, default to full for rgb, limited for yuv/gray
    orig_range  = props.get('_Range', dflt_range) if vs.__version__.release_major >= 74 else 1 - props.get('_ColorRange', dflt_range)  # use _Range for R74 and up and _ColorRange and invert due to zimg for older versions

    # settings
    bd          = 16
    mvsf        = tr > 6
    peak        = (1 << bd) - 1 if not mvsf else 1.0
    limit       = 255
    limit       = limit * peak / 255
    limitc      = limit
    strengthc   = strength // 2
    chroma      = False if clip.format.color_family == vs.GRAY else True
    plane       = 4  if chroma else 0
    blksize     = 16 if orig_width > 2400 else 8
    overlap     = 8  if orig_width > 2400 else 4
    pel         = 1  if orig_width > 2400 else 2
    subpixel    = 0
    search      = 4
    searchparam = 1
    DCT         = 0
    thSCD1      = 1000
    thSCD2      = 1000
    MVglobal    = None
    truemotion  = False
    extra_pad   = 16
    dark_str    = 2.5
    dark_amp    = 0.2
    if pel < 2:
        subpixel = min(subpixel, 2)
    if mvsf:
        mvsflegacy = not hasattr(core.mvsf, "Degrain")  # true is plugin version r9 or older, false is r10 pre-release or newer

    ##### prepare input clip #####

    # convert to 16 bit
    orig = clip
    if orig_format != vs.YUV444P16 or orig_range != 1:
        if orig_family == vs.RGB:
            clip = core.resize.Point(clip, format=vs.YUV444P16, range=1, matrix_s="709")
        else:
            clip = core.resize.Point(clip, format=vs.YUV444P16, range=1)
    
    # add borders
    clip = core.std.AddBorders(clip, left=extra_pad, right=extra_pad, top=extra_pad, bottom=extra_pad)
    clip = core.fb.FillBorders(clip, left=extra_pad, right=extra_pad, top=extra_pad, bottom=extra_pad, mode="fillmargins", interlaced=0)
    ref  = clip

    ##### motion mask #####

    # compensate next frame for motionmask so that it works on pans and zooms
    mm_pref   = core.resize.Bilinear(ref, format=vs.GRAY8)
    mm_sup    = core.mv.Super(mm_pref, pel=2, sharp=1, rfilter=4, hpad=64, vpad=64)
    mm_vec    = core.mv.Analyse(mm_sup, isb=False, delta=1, blksize=128, overlap=64, search=5, truemotion=True)
    mm_window = core.mv.Compensate(mm_pref, mm_sup, mm_vec, thsad=200000, thscd1=1000, thscd2=1000)
    mm_window = core.std.Interleave([mm_window, mm_pref])

    # create motionmask to protect large motions
    mm_window = core.resize.Bicubic(mm_window, width=(mm_window.width / mm_window.height) * 320, height=320) # downscale so that small spatial changes are not in the mask, only larger changes
    mm_window = core.retinex.MSRCP(mm_window, sigma=[mm_window.width / 57], lower_thr=0.011, upper_thr=0.011, fulls=True, fulld=True, chroma_protect=1.0) #
    mm_window = core.motionmask.MotionMask(mm_window, th1=[40], th2=[40], tht=33, sc_value=255) # mask large changes, also used for rudimentary scene change detection
    mm = core.std.SelectEvery(mm_window, cycle=2, offsets=1)

    # further process motionmask
    mm = core.std.Maximum(mm) # expand mask
    m1 = mm[1:] + mm[-1:] # shift - 1    The motion mask compares previous to current frame and masks what changed
    m2 = mm[2:] + mm[-2:] # shift - 2    on the current frame. The first -1 shift just puts the mask on the previous
    m3 = mm[3:] + mm[-3:] # shift - 3    frame as well, so that it is on both sides of the change. The following two 
    p1 = mm[:1] + mm[:-1] # shift + 1    shifts backwards and forwards are used to fade in/out the mask to hide ghosting.
    p2 = mm[:2] + mm[:-2] # shift + 2
    mm = expression([mm, m1, m2, m3, p1, p2], expr=["x y + z 0.75 * + a 0.5 * + b 0.75 * + c 0.5 * +"])
    mm = median(mm, radius=1) # median mask
    mm = core.resize.Point(mm, width=ref.width, height=ref.height)
    mm = box_blur(mm, hradius=4, vradius=4, hpasses=2, vpasses=2) # feather mask

    ##### prefilter to help with motion vectors #####
    
    # resize clips if needed, convert to low bit depth for faster motion vector search
    if pel > 1:
        pref      = core.resize.Bicubic(clip, width=clip.width * pel, height=clip.height * pel, format=vs.YUV444P8)
        mm_resize = core.resize.Bilinear(mm,  width=clip.width * pel, height=clip.height * pel)
    else:
        pref      = core.resize.Point(clip, format=vs.YUV444P8)
        mm_resize = mm
    
    # prefilter
    pref_ref = pref
    pref = _motion_search_prefilter(pref, strength // 2, min(tr, 6))  # main prefilter step
    pref = average_color_fix_fast(pref, pref_ref, 32)                 # fix low freqs
    pref = core.std.MaskedMerge(pref, pref_ref, mm_resize)            # fix blending/ghosting
    pref = tweak_darks(pref, strength=dark_str, amp=dark_amp)         # brighten darks

    ##### degrain #####

    # resize and convert if needed
    if not mvsf:
        if pel > 1:
            pelclip = pref
            pref    = core.resize.Bicubic(pref, width=clip.width, height=clip.height)
    else:
        if pel > 1:
            pelclip = core.resize.Bicubic(pref, format=vs.YUV444PS, width=clip.width * pel, height=clip.height * pel)
        pref        = core.resize.Bicubic(pref, format=vs.YUV444PS, width=clip.width,       height=clip.height)
        clip        = core.resize.Point(clip,   format=vs.YUV444PS)

    # superclips
    create_super = core.mv.Super if not mvsf else core.mvsf.Super
    if pel > 1:
        pref_sup = create_super(pref, chroma=chroma, rfilter=4, pel=pel, pelclip=pelclip)
    else:
        pref_sup = create_super(pref, chroma=chroma, rfilter=4, pel=pel, sharp=1)
    clip_sup     = create_super(clip, chroma=chroma, rfilter=1, pel=pel, sharp=subpixel, levels=1)

    # analyze and degrain
    analyse_args = dict(blksize=blksize, search=search, chroma=chroma, truemotion=truemotion, global_=MVglobal, overlap=overlap, dct=DCT, searchparam=searchparam, fields=False)
    if not mvsf:  # using mvtools because it is faster
        degrain_args = dict(thsad=strength, thsadc=strengthc, plane=plane, limit=limit, limitc=limitc, thscd1=thSCD1, thscd2=thSCD2)
        vecs = mv_analyze(pref_sup, tr, analyse_args)
        clip = mv_degrain(clip, clip_sup, vecs, tr, degrain_args)
    
    else:         # using mvtoolssf because it has support for higher tr
        degrain_args = dict(thsad=[strength, strengthc, strengthc], plane=plane, limit=[limit, limitc, limitc], thscd1=thSCD1, thscd2=thSCD2)
        if mvsflegacy:
            vecs = mvsf_analyze(pref_sup, tr, analyse_args)
            clip = mvsf_degrain(clip, clip_sup, vecs, tr, degrain_args)
        else:
            vecs = core.mvsf.Analyze(pref_sup, radius=tr, **analyse_args)
            clip = core.mvsf.Degrain(clip, clip_sup, vecs, **degrain_args)
        clip = core.resize.Point(clip, format=vs.YUV444P16)

    ##### recover details #####

    # colorfix to counter denoising sometimes changing local brightness
    clip = average_color_fix(clip, ref, 4, 4)

    # contrasharp to counter slight blur
    clip = contrasharp(clip, ref, rep=24, planes=[0])

    # mask to find areas where temporalfix may have removed some texture
    flatmask_post = core.std.ShufflePlanes(clip, planes=0, colorfamily=vs.GRAY)
    flatmask_post = core.tcanny.TCanny(flatmask_post, op=3, mode=1, sigma=0.1, scale=5.0, t_h=8.0, t_l=1.0, opt=1) # mask textures post temporalfix
    flatmask_post = median(flatmask_post, radius=1, planes=0)
    flatmask_post = core.std.Invert(flatmask_post) # invert for flat areas instead

    # overlay original on top of flat areas as these areas may have had texture before
    # don't do if denoise as this will bring back light grain in those areas
    if not denoise:
        clip = core.std.MaskedMerge(clip, ref, flatmask_post, planes=0)

    # mask flat areas with block matching artifacts/wrong motion and overlay original
    flatmask_pre  = core.std.ShufflePlanes(ref, planes=0, colorfamily=vs.GRAY)
    flatmask_pre  = core.tcanny.TCanny(flatmask_pre, op=3, mode=1, sigma=0.1, scale=5.0, t_h=8.0, t_l=1.0, opt=1) # mask textures pre temporalfix
    flatmask_pre  = median(flatmask_pre, radius=1, planes=0)
    flatmask_diff = expression([flatmask_post, flatmask_pre], "65535 x y + 32767 - 2 * -") # compare masks to check if there is now more texture than before, which suggests artifacts, then only use part of mask were textures increased
    clip          = core.std.MaskedMerge(clip, ref, flatmask_diff, planes=0) # use mask to overlay original

    # overlay original in areas with large motion to fix blending/ghosting/warping
    mm = core.resize.Point(mm, format=vs.GRAY16)
    clip = core.std.MaskedMerge(clip, ref, mm)

    # denoise low frequencies
    if denoise:
        clip = lowfreq_denoise(ref, clip, mm, strength // 2, min(tr, 6))

    if debug:
        ref_debug = core.std.Levels(ref, gamma=2)  # visualize unttouched areas
        clip = core.std.MaskedMerge(clip, ref_debug, mm)

    ##### finalize output clip #####

    # remove border
    clip = core.std.Crop(clip, left=extra_pad, right=extra_pad, top=extra_pad, bottom=extra_pad)
    
    # convert back to original format
    if orig_format != vs.YUV444P16 or orig_range != 1:
        if orig_family == vs.RGB:
            clip = core.resize.Point(clip, format=orig_format, range=orig_range, dither_type="error_diffusion", matrix_in_s="709")
        else:
            clip = core.resize.Point(clip, format=orig_format, range=orig_range, dither_type="error_diffusion")
    
    # exclude regions from temporal fixing
    if exclude is not None:
        if debug:
            orig = core.std.Levels(orig, gamma=2)  # visualize excluded frames
        clip = ExcludeRegions(clip, orig, exclude=exclude)
    
    # return result
    return clip
