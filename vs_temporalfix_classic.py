
# Script by pifroggi https://github.com/pifroggi/vs_temporalfix
# or tepete and pifroggi on Discord

# Based on plugins and functions from many different people. See function comments here and in utils for details.

import vapoursynth as vs
from vs_temporalfix_utils import temporal_median, median, basic_expr, advanced_expr, box_blur, average_color_fix, average_color_fix_fast, tweak_darks, contrasharp, exclude_regions, lowfreq_denoise

core = vs.core


def _motion_search_prefilter(clip, thsad=250, tr=6):
    # creates a temporally extremely stable reference for better motion vector estimation, but with lots of ghosting
    # based on SpotLess function from G41Fun https://github.com/Vapoursynth-Plugins-Gitify/G41Fun
    # which was modified from lostfunc https://github.com/theChaosCoder/lostfunc/blob/v1/lostfunc.py#L10
    # which was a port of Didée's original avisynth function https://forum.doom9.org/showthread.php?p=1402690

    # first pass with temporal median
    bs  = 128  # large blocksize to reduce warping
    pel = 1
    
    sup  = core.mvu.Super(clip, blksize=bs, overlap=bs // 2, pad=bs // 2, pel=pel, sharp=1, rfilter=2)
    bv1, fv1 = core.mvu.AnalyseMany(sup, radius=1, search=2, searchparam=2, mvlambda=0, lsad=400, plevel=0, pnew=0, pzero=0, globalmv=False)
    bc1  = core.mvu.Compensate(clip, sup, bv1, thscd1=400, thscd2=51)
    fc1  = core.mvu.Compensate(clip, sup, fv1, thscd1=400, thscd2=51)
    fcb  = core.std.Interleave([fc1, clip, bc1])
    clip = temporal_median(fcb, radius=1, planes=0)[1::3]

    # second pass with degrain and a wide radius (improves pans, zooms and similar, reduces warping)
    bs  = 128  # large blocksize to reduce warping
    pel = 1
    
    sup  = core.mvu.Super(clip, blksize=bs, overlap=0, pel=pel, sharp=1, rfilter=2)
    vecs = core.mvu.AnalyseMany(sup, radius=tr, search=2, searchparam=1, mvlambda=0, lsad=400, plevel=0, pnew=0, pzero=0, globalmv=False)
    return core.mvu.Degrain(clip, sup, vecs, thsad=[thsad, thsad], planes=[0])


def _non_global_motion_mask(clip, downscale=320):
    # masks pixels that don't follow global motion and adds temporal smoothing

    # convert to faster format
    clip = core.resize.Point(clip, format=vs.GRAY8, range_s="limited")

    # compensate next frame for motionmask so that it works on pans and zooms
    sup = core.mvu.Super(clip, blksize=128, overlap=64, pad=64, pel=2, sharp=1, rfilter=2)
    vec = core.mvu.Analyse(sup, delta=-1, search=3, mvlambda=1000, lsad=1200, plevel=1, pnew=50, pzero=50, globalmv=True)
    window = core.mvu.Compensate(clip, sup, vec, thsad=200000, thscd1=1000, thscd2=100.0)
    window = core.std.Interleave([window, clip])
    
    # make textures and their motion more detectable
    window = core.resize.Bicubic(window, width=(window.width / window.height) * downscale, height=downscale)  # downscale so that small spatial changes are not in the mask, only larger changes
    blur   = box_blur(window, hradius=window.width // 55, vradius=window.width // 55, hpasses=3, vpasses=3)
    window = basic_expr([window, blur], "y 0 <= 1 x y / 1 + ? log 0 max 1 min", format=vs.GRAYS)  # retinex approximation with box blur instead of gauss
    window = core.vszip.PlaneMinMax(window, minthr=0.011, maxthr=0.011, prop="temporalfix_")
    window = advanced_expr(window, "x.temporalfix_Max x.temporalfix_Min - 0 <= x x x.temporalfix_Min - x.temporalfix_Max x.temporalfix_Min - / 255 * 0.5 + 0 255 clamp ?", format=vs.GRAY8)  # retinex like floor/ceil clipping
    
    # create motionmask to protect large motions
    prev   = window[:1] + window[:-1]  # previous frame
    stats  = core.std.PlaneStats(window, prev, prop="temporalfix_")  # compare previous to current frame
    window = advanced_expr([window, prev, stats], ["z.temporalfix_Diff 255 * 33 > 255 x y - abs 40 > 255 0 ? ?"])  # motionmask replacement, mask large changes and rudimentary scene change detection
    mask = core.std.SelectEvery(window, cycle=2, offsets=1)

    # further process motionmask
    mask = core.std.Maximum(mask)  # expand mask
    m1 = mask[1:] + mask[-1:]  # shift - 1    The motion mask compares previous to current frame and masks what changed
    m2 = mask[2:] + mask[-2:]  # shift - 2    on the current frame. The first -1 shift just puts the mask on the previous
    m3 = mask[3:] + mask[-3:]  # shift - 3    frame as well, so that it is on both sides of the change. The following two 
    p1 = mask[:1] + mask[:-1]  # shift + 1    shifts backwards and forwards are used to fade in/out the mask to hide ghosting.
    p2 = mask[:2] + mask[:-2]  # shift + 2
    mask = basic_expr([mask, m1, m2, m3, p1, p2], expr=["x y + z 0.75 * + a 0.5 * + b 0.75 * + c 0.5 * +"])
    mask = median(mask, radius=1)  # median mask
    mask = core.resize.Point(mask, width=clip.width, height=clip.height)
    return box_blur(mask, hradius=4, vradius=4, hpasses=2, vpasses=2)  # feather mask


def classic(clip, strength=500, tr=6, denoise=False, exclude=None, debug=False):
    """Add temporal coherence to single image AI upscaling models. Also known as temporal consistency, line wiggle fix, stabilization, deshimmering. 
    This is the original CPU based version. It can run on any CPU, but is harder to tune, may miss some areas, and only works well for 2D animation. Check the 
    tips at the bottom for important usage information!

    Args:
        clip: Temporally unstable upscaled clip.
        strength: Suppression strength of temporal inconsistencies. Higher means more aggressive. `400-700` works great in most cases. 
            The best way to finetune is to find a static scene and adjust till lines and details are stable. 
            Reduce if you get blending/ghosting on small movements, especially in dark or hazy scenes.
        tr: Temporal radius sets the number of frames to average over. 
            Higher means more stable, especially on slow pans and zooms, but is slower. `6` works great in most cases. 
            The best way to finetune is to find a slow pan or zoom and adjust till lines and details are stable.
        denoise: Removes grain and low frequency noise/flicker left over by the main processing step. Only enable if these issues actually exist! 
            It risks to remove some details like every denoiser, but is useful if you're planning to denoise anyway and has the benefit of almost 
            no performance impact compared to using an additional denoising filter.
        exclude: Optionally exclude scenes with intended temporal inconsistencies. Brackets define excluded frame ranges. 
            Example for two scenes: `exclude="[10 20] [600 900]"`
        debug: Shows areas that will be left untouched in pink. This includes areas with high motion, scene changes and excluded scenes. 
            May help while tuning parameters to see if the area is even affected.

            Tip: Crop any black borders on the input clip, as those may cause ghosting on bright frames.  
            Tip: If this is extremely slow, try increasing your frame cache to at least 10GB! `core.max_cache_size = 10000` Not needed on vapoursynth R78 and up.
    """
    
    # based on SMDegrain function from G41Fun https://github.com/Vapoursynth-Plugins-Gitify/G41Fun
    # which is a modification of SMDegrain from havsfunc https://github.com/HomeOfVapourSynthEvolution/havsfunc/blob/r31/havsfunc.py#L3186
    # which is a port of SMDegrain from avisynth https://forum.videohelp.com/threads/369142

    #checks
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("vs_temporalfix: Clip must be a vapoursynth clip.")
    if clip.format.id == vs.PresetVideoFormat.NONE or clip.width == 0 or clip.height == 0:
        raise TypeError("vs_temporalfix: Clip must have constant format and dimensions.")
    if not isinstance(strength, int):
        raise TypeError("vs_temporalfix: Strength must be an integer.")
    if strength < 0:
        raise ValueError("vs_temporalfix: Strength can not be negative.")
    if not isinstance(tr, int):
        raise TypeError("vs_temporalfix: Temporal radius (tr) must be an integer.")
    if tr < 1 or tr > 25:
        raise ValueError("vs_temporalfix: Temporal radius (tr) must be in the 1-25 range.")
    if not isinstance(denoise, bool):
        raise TypeError("vs_temporalfix: Denoise must be either True or False.")
    if not isinstance(debug, bool):
        raise TypeError("vs_temporalfix: Debug must be either True or False.")

    # original props
    orig_clip   = clip
    props       = clip.get_frame(0).props
    orig_size   = max(clip.width, clip.height)
    orig_format = clip.format
    orig_family = clip.format.color_family
    dflt_range  = vs.RANGE_FULL if orig_family == vs.RGB else vs.RANGE_LIMITED  # if not tagged, default to full for rgb, limited for yuv/gray
    orig_range  = props.get('_Range', dflt_range) if vs.__version__.release_major >= 74 else 1 - props.get('_ColorRange', dflt_range)  # use _Range for R74 and up and _ColorRange and invert due to zimg for older versions

    # settings
    depth       = 16
    peak        = (1 << depth) - 1
    strengthc   = strength // 2
    chroma      = False if clip.format.color_family == vs.GRAY else True
    blksize     = 16 if orig_size > 2400 else 8
    overlap     = 8  if orig_size > 2400 else 4
    pel         = 1  if orig_size > 2400 else 2
    subpixel    = 0
    search      = 2
    searchparam = 1
    thscd1      = 1000
    thscd2      = 100.0
    dark_str    = 2.5
    dark_amp    = 0.2
    
    # formats
    proc_format = core.get_video_format(getattr(vs, f"YUV444P{depth}")) if orig_family == vs.RGB else orig_format.replace(sample_type=vs.INTEGER, bits_per_sample=depth)  # format for the main processing steps
    pref_format = proc_format.replace(sample_type=vs.INTEGER, bits_per_sample=8, **(dict(subsampling_w=1, subsampling_h=1) if proc_format.color_family == vs.YUV else {}))  # format for the prefilter step
    alys_format = proc_format.replace(sample_type=pref_format.sample_type, bits_per_sample=pref_format.bits_per_sample)  # format for motion analysing during the main processing steps


    ##### prepare input #####

    # convert to proc format
    if orig_format != proc_format or orig_range != 1:
        clip = core.resize.Point(clip, format=proc_format, range=1, **({"matrix_s": "709"} if orig_family == vs.RGB else {}))
    ref  = clip


    ##### prefilter #####
    
    # mask things that do not follow global motion
    motionmask = _non_global_motion_mask(ref, downscale=320)
    
    # resize clips if needed, convert to low bit depth for faster motion vector search
    if pel > 1:
        pref   = core.resize.Bicubic(clip,        width=clip.width * pel, height=clip.height * pel, format=pref_format)
        mm_pel = core.resize.Bilinear(motionmask, width=clip.width * pel, height=clip.height * pel)
    else:
        pref   = core.resize.Bilinear(clip, format=pref_format)
        mm_pel = motionmask
    
    # prefilter
    pref_ref = pref
    pref = _motion_search_prefilter(pref, strength // 2, min(tr, 6))  # main prefilter step to help motion vector search
    pref = average_color_fix_fast(pref, pref_ref, 32)                 # fix low freqs
    pref = core.std.MaskedMerge(pref, pref_ref, mm_pel)               # fix blending/ghosting
    pref = tweak_darks(pref, strength=dark_str, amp=dark_amp)         # brighten darks


    ##### degrain #####

    # resize and convert
    if pel > 1:
        pelclip = core.resize.Bilinear(pref, format=alys_format) if pref.format != alys_format else pref
    if pel > 1 or pref.format != alys_format:
        pref = core.resize.Bicubic(pref, format=alys_format, width=clip.width, height=clip.height)
    
    # create superclips
    pref_sup = core.mvu.Super(pref, blksize=blksize, overlap=overlap, rfilter=2, pel=pel, **({'pelclip': pelclip} if pel > 1 else {'sharp': 1}))
    clip_sup = core.mvu.Super(clip, blksize=blksize, overlap=overlap, pel=pel, sharp=subpixel, onelevel=True)
    
    # analyse and degrain
    vecs = core.mvu.AnalyseMany(pref_sup, radius=tr, search=search, searchparam=searchparam, chroma=chroma, mvlambda=0, lsad=400, plevel=0, pnew=0, pzero=0, globalmv=False, fields=False, satd=False)
    clip = core.mvu.Degrain(clip, clip_sup, vecs, thsad=[strength, strengthc], planes=[0, 1, 2] if chroma else [0], thscd1=thscd1, thscd2=thscd2)


    ##### recover details #####

    # colorfix to counter denoising sometimes changing local brightness
    clip = average_color_fix(clip, ref, 4, 4)

    # contrasharp to counter slight blur
    clip = contrasharp(clip, ref, rep=24, planes=[0])

    # mask to find areas where temporalfix may have removed some texture
    flatmask_post = core.std.ShufflePlanes(clip, 0, vs.GRAY) if clip.format.color_family != vs.GRAY else clip
    flatmask_post = core.edgemasks.Scharr(flatmask_post, scale=15.0)  # mask textures post temporalfix
    flatmask_post = median(flatmask_post, radius=1)
    flatmask_post = core.std.Invert(flatmask_post)  # invert for flat areas instead

    # overlay original on top of flat areas as these areas may have had texture before, but don't do if denoise as this will bring back light grain
    if not denoise:
        clip = core.std.MaskedMerge(clip, ref, flatmask_post, planes=0)

    # mask flat areas with block matching artifacts/wrong motion and overlay original
    flatmask_pre  = core.std.ShufflePlanes(ref, 0, vs.GRAY) if clip.format.color_family != vs.GRAY else ref
    flatmask_pre  = core.edgemasks.Scharr(flatmask_pre, scale=15.0)  # mask textures pre temporalfix
    flatmask_pre  = median(flatmask_pre, radius=1)
    flatmask_diff = basic_expr([flatmask_post, flatmask_pre], expr=f"{peak} x y + {peak // 2} - 2 * -")  # compare masks to check if there is now more texture than before, which suggests artifacts, then only use part of mask were textures increased
    clip = core.std.MaskedMerge(clip, ref, flatmask_diff, planes=0)  # use mask to overlay original

    # overlay original in areas with large motion to fix blending/ghosting/warping
    motionmask = core.resize.Point(motionmask, format=motionmask.format.replace(sample_type=clip.format.sample_type, bits_per_sample=clip.format.bits_per_sample).id)
    clip = core.std.MaskedMerge(clip, ref, motionmask)

    # denoise low frequencies
    if denoise:
        clip = lowfreq_denoise(ref, clip, motionmask, strength // 2, min(tr, 6))

    # overlay debug output
    if debug:
        ref_debug = core.std.Levels(ref, gamma=2)  # visualize unfixed areas
        clip = core.std.MaskedMerge(clip, ref_debug, motionmask)
        clip = exclude_regions(clip, ref_debug, exclude=exclude)  # if debug, do exclude here with ref, so that the replacement is still in yuv


    ##### finalize output #####
    
    # convert back to original format
    if orig_format != proc_format or orig_range != 1:
        clip = core.resize.Point(clip, format=orig_format, range=orig_range, dither_type="error_diffusion")
    
    # exclude regions from temporal fixing
    if debug:  # if debug, already excluded in yuv space for better gamma visualization
        return clip
    return exclude_regions(clip, orig_clip, exclude=exclude)

