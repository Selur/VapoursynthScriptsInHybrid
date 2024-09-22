# Based on plugins and functions from many different people. See function comments and readme requirements for details.

# Script by pifroggi https://github.com/pifroggi/vs_temporalfix
# or tepete on the "Enhance Everything!" Discord Server

import vapoursynth as vs

core = vs.core


def AverageColorFix(clip, ref, radius=4, passes=4):
    # modified from https://github.com/pifroggi/vs_colorfix
    blurred_reference = core.std.BoxBlur(ref, hradius=radius, hpasses=passes, vradius=radius, vpasses=passes)
    blurred_clip = core.std.BoxBlur(clip, hradius=radius, hpasses=passes, vradius=radius, vpasses=passes)
    diff_clip = core.std.MakeDiff(blurred_reference, blurred_clip)
    return core.std.MergeDiff(clip, diff_clip)


def AverageColorFixFast(clip, ref, downscale_factor=8):
    # faster but faint blocky artifacts
    downscaled_reference = core.resize.Bilinear(ref, width=clip.width / downscale_factor, height=clip.height / downscale_factor)
    downscaled_clip = core.resize.Bilinear(clip, width=clip.width / downscale_factor, height=clip.height / downscale_factor)
    diff_clip = core.std.MakeDiff(downscaled_reference, downscaled_clip)
    diff_clip = core.resize.Bilinear(diff_clip, width=clip.width, height=clip.height)
    return core.std.MergeDiff(clip, diff_clip)


def TweakDarks(src, s0=2.0, c=0.0625, chroma=True):
    # simplified DitherLumaRebuild function that works on full range
    # DitherLumaRebuild function from G41Fun https://github.com/Vapoursynth-Plugins-Gitify/G41Fun
    # originally created by cretindesalpes https://forum.doom9.org/showthread.php?p=1548318

    bd = src.format.bits_per_sample
    isFLOAT = src.format.sample_type == vs.FLOAT
    i = 0.00390625 if isFLOAT else 1 << (bd - 8)
    x = "x {} /".format(i) if bd != 8 else "x"
    expr = "x" if isFLOAT else "{} {} *".format(x, i)
    k = (s0 - 1) * c
    t = "{} 255 / 0 max 1 min".format(x)
    c1 = 1 + c
    c2 = c1 * c
    e = "{} {} {} {} {} + / - * {} 1 {} - * + {} *".format(k, c1, c2, t, c, t, k, 256 * i)
    return core.std.Expr([src], [e] if src.format.num_planes == 1 else [e, expr if chroma else ""])


def ExcludeRegions(clip, replacement, exclude=None):
    # simplified ReplaceFrames function from fvsfunc https://github.com/Irrational-Encoding-Wizardry/fvsfunc
    # which is a port of ReplaceFramesSimple by James D. Lin http://avisynth.nl/index.php/RemapFrames
    import re

    if not isinstance(exclude, str):
        raise TypeError('Exclusions are set like this: exclude="[100 300] [600 900] [2000 2500]", where the first number in the brackets is the start frame and the second is the end frame (inclusive).')

    exclude = exclude.replace(",", " ").replace(":", " ")
    frames = re.findall(r"\d+(?!\d*\s*\d*\s*\d*\])", exclude)
    ranges = re.findall(r"\[\s*\d+\s+\d+\s*\]", exclude)
    maps = []
    for range_ in ranges:
        maps.append([int(x) for x in range_.strip("[ ]").split()])
    for frame in frames:
        maps.append([int(frame), int(frame)])

    for start, end in maps:
        if start > end:
            raise ValueError("Exclusions start frame is bigger than end frame: [{} {}]".format(start, end))
        if start >= clip.num_frames:
            raise ValueError("Exclusions start frame {} is outside the clip, which has only {} frames.".format(start, clip.num_frames))

    out = clip
    for start, end in maps:
        temp = replacement[start : end + 1]
        temp = replacement[start : end + 1]
        if start != 0:
            temp = out[:start] + temp
        if end < out.num_frames - 1:
            temp = temp + out[end + 1 :]
        out = temp
    return out


def DegrainPrefilter(clip, thsad=250):
    # based on SpotLess function from G41Fun https://github.com/Vapoursynth-Plugins-Gitify/G41Fun
    # which was modified from lostfunc https://github.com/theChaosCoder/lostfunc/blob/v1/lostfunc.py#L10
    # which was a port of Didée's original avisynth function https://forum.doom9.org/showthread.php?p=1402690

    A = core.mv.Analyse
    C = core.mv.Compensate
    S = core.mv.Super

    # first pass with temporal median
    bs = 128  # large blocksize to reduce warping
    pel = 1
    sup = S(clip, pel=pel, sharp=1, rfilter=4, hpad=bs // 2, vpad=bs // 2)
    analyse_args = dict(blksize=bs, overlap=bs // 2, search=4, searchparam=2)
    bv1 = A(sup, isb=True, delta=1, **analyse_args)
    fv1 = A(sup, isb=False, delta=1, **analyse_args)
    bc1 = C(clip, sup, bv1)
    fc1 = C(clip, sup, fv1)
    fcb = core.std.Interleave([fc1, clip, bc1])
    clip = core.tmedian.TemporalMedian(fcb, 1, 0)[1::3]

    # second pass with degrain and a wide radius (improves pans, zooms and similar, reduces warping)
    bs = 128  # large blocksize to reduce warping
    pel = 1
    sup = S(clip, pel=pel, sharp=1, rfilter=4)
    analyse_args = dict(blksize=bs, overlap=0, search=4, searchparam=1)
    bv6 = A(sup, isb=True,  delta=6, **analyse_args)
    bv5 = A(sup, isb=True,  delta=5, **analyse_args)
    bv4 = A(sup, isb=True,  delta=4, **analyse_args)
    bv3 = A(sup, isb=True,  delta=3, **analyse_args)
    bv2 = A(sup, isb=True,  delta=2, **analyse_args)
    bv1 = A(sup, isb=True,  delta=1, **analyse_args)
    fv1 = A(sup, isb=False, delta=1, **analyse_args)
    fv2 = A(sup, isb=False, delta=2, **analyse_args)
    fv3 = A(sup, isb=False, delta=3, **analyse_args)
    fv4 = A(sup, isb=False, delta=4, **analyse_args)
    fv5 = A(sup, isb=False, delta=5, **analyse_args)
    fv6 = A(sup, isb=False, delta=6, **analyse_args)
    return clip.mv.Degrain6(sup, bv1, fv1, bv2, fv2, bv3, fv3, bv4, fv4, bv5, fv5, bv6, fv6, thsad=thsad, plane=0)


def vs_temporalfix(clip, strength=400, tr=6, exclude=None, debug=False):
    # based on SMDegrain function from G41Fun https://github.com/Vapoursynth-Plugins-Gitify/G41Fun
    # which is a modification of SMDegrain from havsfunc https://github.com/HomeOfVapourSynthEvolution/havsfunc/blob/r31/havsfunc.py#L3186
    # which is a port of SMDegrain from avisynth https://forum.videohelp.com/threads/369142

    ##### defaults & conditionals #####

    if not isinstance(clip, vs.VideoNode):
        raise TypeError("This is not a clip.")

    mvtr = tr
    w = clip.width
    original_format = clip.format
    isGRAY = clip.format.color_family == vs.GRAY
    S = core.mvsf.Super if mvtr > 6 else core.mv.Super
    A = core.mv.Analyse
    D1 = core.mv.Degrain1
    D2 = core.mv.Degrain2
    D3 = core.mv.Degrain3
    D4 = core.mv.Degrain4
    D5 = core.mv.Degrain5
    D6 = core.mv.Degrain6
    bd = 16
    peak = 1.0 if mvtr > 6 else (1 << bd) - 1
    limit = 255
    limit = limit * peak / 255
    limitc = limit
    chroma = False if isGRAY else True
    plane = 4 if chroma else 0
    thSAD = strength
    thSADC = thSAD // 2
    blksize = 16 if w > 2400 else 8
    overlap = 8 if w > 2400 else 4
    pel = 1 if w > 2400 else 2
    subpixel = 0
    search = 4
    searchparam = 1
    DCT = 0
    MVglobal = None
    thSCD1 = 1000
    thSCD2 = 1000
    Str = 2.5
    Amp = 0.2
    border_add = 16
    truemotion = False
    if pel < 2:
        subpixel = min(subpixel, 2)
    if mvtr > 6:
        mvsflegacy = not hasattr(core.mvsf, "Degrain")  # true is plugin version r9 or older, false is r10 pre-release or newer

    ##### prepare #####

    if exclude is not None:
        original = clip
    clip = core.std.AddBorders(clip, left=border_add, right=border_add, top=border_add, bottom=border_add)
    if original_format != vs.YUV444P16:
        clip = core.resize.Point(clip, format=vs.YUV444P16)
    clip = core.fb.FillBorders(clip, left=border_add, right=border_add, top=border_add, bottom=border_add, mode="fillmargins", interlaced=0)
    pre_stabilize = clip

    ##### motion mask #####

    # compensate next frame for motionmask so that it works on pans and zooms
    motionmask_pref = core.resize.Bilinear(pre_stabilize, format=vs.GRAY8)
    motionmask_super = core.mv.Super(motionmask_pref, pel=2, sharp=1, rfilter=4, hpad=64, vpad=64)
    motionmask_vectors = core.mv.Analyse(motionmask_super, isb=False, delta=1, blksize=128, overlap=64, search=5)
    motionmask_window = core.mv.Compensate(motionmask_pref, motionmask_super, motionmask_vectors, thsad=200000, thscd1=1000, thscd2=1000)
    motionmask_window = core.std.Interleave([motionmask_window, motionmask_pref])

    # create motionmask to protect large motions
    motionmask_window = core.retinex.MSRCP(motionmask_window, sigma=[motionmask_window.width // 57], lower_thr=0.011, upper_thr=0.011, fulls=True, fulld=True, chroma_protect=1.2)
    motionmask_window = core.resize.Bicubic(motionmask_window, width=(motionmask_window.width / motionmask_window.height) * 350, height=350)
    motionmask_window = core.motionmask.MotionMask(motionmask_window, th1=[45], th2=[40], tht=33, sc_value=255)
    motionmask = core.std.SelectEvery(motionmask_window, cycle=2, offsets=1)

    # further process motionmask
    motionmask = core.std.Maximum(motionmask)
    m1 = motionmask[1:]  + motionmask[-1]  # shift - 1
    m2 = motionmask[2:]  + motionmask[-2]  # shift - 2
    m3 = motionmask[3:]  + motionmask[-3:] # shift - 3
    p1 = motionmask[0]   + motionmask[:-1] # shift + 1
    p2 = motionmask[0:2] + motionmask[:-2] # shift + 2
    motionmask = core.std.Expr([motionmask, m1, m2, m3, p1, p2], expr=["x y + z 0.75 * + a 0.5 * + b 0.75 * + c 0.5 * +"]) # fades the mask in/out over a few frames
    motionmask = core.std.Median(motionmask)
    motionmask = core.resize.Point(motionmask, width=pre_stabilize.width, height=pre_stabilize.height)
    motionmask = core.std.BoxBlur(motionmask, hradius=4, vradius=4, hpasses=2, vpasses=2)

    ##### prefilter #####

    # prefilter to help with motion vectors
    if pel > 1:
        prefilter = core.resize.Bicubic(clip, format=vs.YUV444P8, width=clip.width * 2, height=clip.height * 2)
        pre_stabilize_resized = core.resize.Bilinear(pre_stabilize, format=vs.YUV444P8, width=clip.width * 2, height=clip.height * 2)
        motionmask_resized = core.resize.Bilinear(motionmask, width=clip.width * 2, height=clip.height * 2)
    else:
        prefilter = core.resize.Point(clip, format=vs.YUV444P8)
        pre_stabilize_resized = core.resize.Point(pre_stabilize, format=vs.YUV444P8)
        motionmask_resized = motionmask
    prefilter = DegrainPrefilter(prefilter, thsad=thSAD // 2)
    prefilter = AverageColorFixFast(prefilter, pre_stabilize_resized, 32)
    prefilter = core.std.MaskedMerge(prefilter, pre_stabilize_resized, motionmask_resized)
    prefilter = TweakDarks(prefilter, s0=Str, c=Amp, chroma=chroma)

    ##### degrain #####

    # resize
    if mvtr > 6:
        if pel > 1:
            pelclip = core.resize.Bicubic(prefilter, format=vs.YUV444PS, width=clip.width * 2, height=clip.height * 2)
        prefilter = core.resize.Bicubic(prefilter, format=vs.YUV444PS, width=clip.width, height=clip.height)
        clip = core.resize.Point(clip, format=vs.YUV444PS)
    else:
        if pel > 1:
            pelclip = prefilter
            prefilter = core.resize.Bicubic(prefilter, width=clip.width, height=clip.height)

    # superclip
    if pel > 1:
        super_search = S(prefilter, chroma=chroma, rfilter=4, pelclip=pelclip, pel=pel)
    else:
        super_search = S(prefilter, chroma=chroma, rfilter=4, sharp=1, pel=pel)
    super_render = S(clip, chroma=chroma, rfilter=1, sharp=subpixel, levels=1, pel=pel)

    # analyze
    analyse_args = dict(blksize=blksize, search=search, chroma=chroma, truemotion=truemotion, global_=MVglobal, overlap=overlap, dct=DCT, searchparam=searchparam, fields=False)
    if mvtr < 7:
        if mvtr > 5:
            bv6 = A(super_search, isb=True,  delta=6, **analyse_args)
            fv6 = A(super_search, isb=False, delta=6, **analyse_args)
        if mvtr > 4:
            bv5 = A(super_search, isb=True,  delta=5, **analyse_args)
            fv5 = A(super_search, isb=False, delta=5, **analyse_args)
        if mvtr > 3:
            bv4 = A(super_search, isb=True,  delta=4, **analyse_args)
            fv4 = A(super_search, isb=False, delta=4, **analyse_args)
        if mvtr > 2:
            bv3 = A(super_search, isb=True,  delta=3, **analyse_args)
            fv3 = A(super_search, isb=False, delta=3, **analyse_args)
        if mvtr > 1:
            bv2 = A(super_search, isb=True,  delta=2, **analyse_args)
            fv2 = A(super_search, isb=False, delta=2, **analyse_args)
        bv1 = A(super_search, isb=True,  delta=1, **analyse_args)
        fv1 = A(super_search, isb=False, delta=1, **analyse_args)
    else:
        if mvsflegacy:
            vec = Analyze(super_search, tr=mvtr, **analyse_args)
        else:
            vec = core.mvsf.Analyze(super_search, radius=mvtr, **analyse_args)

    # degrain
    if mvtr < 7:
        degrain_args = dict(thsad=thSAD, thsadc=thSADC, plane=plane, limit=limit, limitc=limitc, thscd1=thSCD1, thscd2=thSCD2)
        if mvtr == 6:
            clip = D6(clip, super_render, bv1, fv1, bv2, fv2, bv3, fv3, bv4, fv4, bv5, fv5, bv6, fv6, **degrain_args)
        elif mvtr == 5:
            clip = D5(clip, super_render, bv1, fv1, bv2, fv2, bv3, fv3, bv4, fv4, bv5, fv5, **degrain_args)
        elif mvtr == 4:
            clip = D4(clip, super_render, bv1, fv1, bv2, fv2, bv3, fv3, bv4, fv4, **degrain_args)
        elif mvtr == 3:
            clip = D3(clip, super_render, bv1, fv1, bv2, fv2, bv3, fv3, **degrain_args)
        elif mvtr == 2:
            clip = D2(clip, super_render, bv1, fv1, bv2, fv2, **degrain_args)
        else:
            clip = D1(clip, super_render, bv1, fv1, **degrain_args)
    else:
        degrain_args = dict(thsad=[thSAD, thSADC, thSADC], plane=plane, limit=[limit, limitc, limitc], thscd1=thSCD1, thscd2=thSCD2)
        if mvsflegacy:
            clip = DegrainN(clip, super_render, vec, tr=mvtr, **degrain_args)
        else:
            clip = core.mvsf.Degrain(clip, super_render, vec, **degrain_args)
        clip = core.resize.Point(clip, format=vs.YUV444P16)

    ##### recover details #####

    # colorfix
    clip = AverageColorFix(clip, pre_stabilize, 4, 4)

    # contrasharp to counter some textures becoming blurry
    clip = ContraSharpening(clip, pre_stabilize, rep=24, planes=[0])

    # mask to find areas where denoising may have removed some texture
    edgemask = core.std.ShufflePlanes(clip, planes=0, colorfamily=vs.GRAY)
    edgemask = core.tcanny.TCanny(edgemask, op=3, mode=1, sigma=0.1, scale=5.0, t_h=8.0, t_l=1.0, opt=1)
    edgemask = core.std.Median(edgemask, planes=0)
    edgemask = core.std.Invert(edgemask)

    # add missing texture back in
    diff_clip = core.std.MakeDiff(pre_stabilize, clip)
    clip_merged = core.std.MergeDiff(clip, diff_clip, planes=0)
    clip = core.std.MaskedMerge(clip, clip_merged, edgemask, planes=0)

    # mask flat areas with block matching artifacts/wrong motion and add original back
    edgemask2 = core.std.ShufflePlanes(pre_stabilize, planes=0, colorfamily=vs.GRAY)
    edgemask2 = core.tcanny.TCanny(edgemask2, op=3, mode=1, sigma=0.1, scale=5.0, t_h=8.0, t_l=1.0, opt=1)
    edgemask2 = core.std.Median(edgemask2, planes=0)
    edgemask2 = core.std.Invert(edgemask2)
    edgemasks_diff = core.std.MakeDiff(edgemask, edgemask2)
    edgemasks_diff = core.std.Levels(edgemasks_diff, max_in=32768 - 1, max_out=65535 - 1)
    edgemasks_diff = core.std.Invert(edgemasks_diff)
    clip = core.std.MaskedMerge(clip, pre_stabilize, edgemasks_diff, planes=0)

    # apply motionmask to protect regions with large motion
    motionmask = core.resize.Point(motionmask, format=vs.GRAY16)
    if debug:
        gam = core.std.Levels(pre_stabilize, gamma=2)  # just for visualization
        clip = core.std.MaskedMerge(clip, gam, motionmask)
    else:
        clip = core.std.MaskedMerge(clip, pre_stabilize, motionmask)

    ##### finish #####

    # remove border
    clip = core.std.Crop(clip, left=border_add, right=border_add, top=border_add, bottom=border_add)
    if original_format != vs.YUV444P16:
        clip = core.resize.Point(clip, format=original_format)
    if exclude is not None:
        if debug:
            original = core.std.Levels(original, gamma=2)
        clip = ExcludeRegions(clip, original, exclude=exclude)
    return clip


# I have consolidated a few functions here to make sure it doesn't break


# fmt: off
def Analyze(super, blksize=None, blksizev=None, levels=None, search=None, searchparam=None, pelsearch=None, lambda_=None, chroma=None, tr=None, truemotion=None, lsad=None, plevel=None, global_=None, pnew=None, pzero=None, pglobal=None, overlap=None, overlapv=None, divide=None, badsad=None, badrange=None, meander=None, trymany=None, fields=False, TFF=None, search_coarse=None, dct=None):
    #function from mvmulti: https://github.com/IFeelBloated/vapoursynth-mvtools-sf/blob/r9/src/mvmulti.py
    def getvecs(isb, delta):
        return core.mvsf.Analyze(super, isb=isb, blksize=blksize, blksizev=blksizev, levels=levels, search=search, searchparam=searchparam, pelsearch=pelsearch, lambda_=lambda_, chroma=chroma, delta=delta, truemotion=truemotion, lsad=lsad, plevel=plevel, global_=global_, pnew=pnew, pzero=pzero, pglobal=pglobal, overlap=overlap, overlapv=overlapv, divide=divide, badsad=badsad, badrange=badrange, meander=meander, trymany=trymany, fields=fields, tff=TFF, search_coarse=search_coarse, dct=dct)
    bv = [getvecs(True,  i) for i in range(tr, 0, -1)]
    fv = [getvecs(False, i) for i in range(1, tr + 1)]
    return core.std.Interleave(bv + fv)

def Recalculate(super, vectors, thsad=200.0, smooth=1, blksize=8, blksizev=None, search=4, searchparam=2, lambda_=None, chroma=True, truemotion=True, pnew=None, overlap=0, overlapv=None, divide=0, meander=True, fields=False, tff=None, dct=0, tr=3):
    #function from mvmulti: https://github.com/IFeelBloated/vapoursynth-mvtools-sf/blob/r9/src/mvmulti.py
    core         = vs.core
    def refine(delta):
        analyzed = core.std.SelectEvery(vectors, 2*tr, delta)
        refined  = core.mvsf.Recalculate(super, analyzed, thsad=thsad, smooth=smooth, blksize=blksize, blksizev=blksizev, search=search, searchparam=searchparam, lambda_=lambda_, chroma=chroma, truemotion=truemotion, pnew=pnew, overlap=overlap, overlapv=overlapv, divide=divide, meander=meander, fields=fields, tff=tff, dct=dct)
        return refined
    vmulti       = [refine(i) for i in range(0, 2*tr)]
    vmulti       = core.std.Interleave(vmulti)
    return vmulti

def DegrainN(clip, super, mvmulti, tr=3, thsad=400.0, plane=4, limit=1.0, thscd1=400.0, thscd2=130.0):
    #function from mvmulti: https://github.com/IFeelBloated/vapoursynth-mvtools-sf/blob/r9/src/mvmulti.py
    core         = vs.core
    def bvn(n):
        bv       = core.std.SelectEvery(mvmulti, tr*2, tr-n)
        return bv
    def fvn(n):
        fv       = core.std.SelectEvery(mvmulti, tr*2, tr+n-1)
        return fv
    if tr == 1:
       dgn       = core.mvsf.Degrain1(clip, super, bvn(1), fvn(1), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 2:
       dgn       = core.mvsf.Degrain2(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 3:
       dgn       = core.mvsf.Degrain3(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 4:
       dgn       = core.mvsf.Degrain4(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 5:
       dgn       = core.mvsf.Degrain5(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 6:
       dgn       = core.mvsf.Degrain6(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 7:
       dgn       = core.mvsf.Degrain7(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 8:
       dgn       = core.mvsf.Degrain8(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 9:
       dgn       = core.mvsf.Degrain9(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 10:
       dgn       = core.mvsf.Degrain10(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 11:
       dgn       = core.mvsf.Degrain11(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 12:
       dgn       = core.mvsf.Degrain12(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), bvn(12), fvn(12), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 13:
       dgn       = core.mvsf.Degrain13(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), bvn(12), fvn(12), bvn(13), fvn(13), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 14:
       dgn       = core.mvsf.Degrain14(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), bvn(12), fvn(12), bvn(13), fvn(13), bvn(14), fvn(14), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 15:
       dgn       = core.mvsf.Degrain15(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), bvn(12), fvn(12), bvn(13), fvn(13), bvn(14), fvn(14), bvn(15), fvn(15), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 16:
       dgn       = core.mvsf.Degrain16(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), bvn(12), fvn(12), bvn(13), fvn(13), bvn(14), fvn(14), bvn(15), fvn(15), bvn(16), fvn(16), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 17:
       dgn       = core.mvsf.Degrain17(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), bvn(12), fvn(12), bvn(13), fvn(13), bvn(14), fvn(14), bvn(15), fvn(15), bvn(16), fvn(16), bvn(17), fvn(17), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 18:
       dgn       = core.mvsf.Degrain18(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), bvn(12), fvn(12), bvn(13), fvn(13), bvn(14), fvn(14), bvn(15), fvn(15), bvn(16), fvn(16), bvn(17), fvn(17), bvn(18), fvn(18), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 19:
       dgn       = core.mvsf.Degrain19(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), bvn(12), fvn(12), bvn(13), fvn(13), bvn(14), fvn(14), bvn(15), fvn(15), bvn(16), fvn(16), bvn(17), fvn(17), bvn(18), fvn(18), bvn(19), fvn(19), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 20:
       dgn       = core.mvsf.Degrain20(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), bvn(12), fvn(12), bvn(13), fvn(13), bvn(14), fvn(14), bvn(15), fvn(15), bvn(16), fvn(16), bvn(17), fvn(17), bvn(18), fvn(18), bvn(19), fvn(19), bvn(20), fvn(20), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 21:
       dgn       = core.mvsf.Degrain21(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), bvn(12), fvn(12), bvn(13), fvn(13), bvn(14), fvn(14), bvn(15), fvn(15), bvn(16), fvn(16), bvn(17), fvn(17), bvn(18), fvn(18), bvn(19), fvn(19), bvn(20), fvn(20), bvn(21), fvn(21), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 22:
       dgn       = core.mvsf.Degrain22(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), bvn(12), fvn(12), bvn(13), fvn(13), bvn(14), fvn(14), bvn(15), fvn(15), bvn(16), fvn(16), bvn(17), fvn(17), bvn(18), fvn(18), bvn(19), fvn(19), bvn(20), fvn(20), bvn(21), fvn(21), bvn(22), fvn(22), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 23:
       dgn       = core.mvsf.Degrain23(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), bvn(12), fvn(12), bvn(13), fvn(13), bvn(14), fvn(14), bvn(15), fvn(15), bvn(16), fvn(16), bvn(17), fvn(17), bvn(18), fvn(18), bvn(19), fvn(19), bvn(20), fvn(20), bvn(21), fvn(21), bvn(22), fvn(22), bvn(23), fvn(23), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    elif tr == 24:
       dgn       = core.mvsf.Degrain24(clip, super, bvn(1), fvn(1), bvn(2), fvn(2), bvn(3), fvn(3), bvn(4), fvn(4), bvn(5), fvn(5), bvn(6), fvn(6), bvn(7), fvn(7), bvn(8), fvn(8), bvn(9), fvn(9), bvn(10), fvn(10), bvn(11), fvn(11), bvn(12), fvn(12), bvn(13), fvn(13), bvn(14), fvn(14), bvn(15), fvn(15), bvn(16), fvn(16), bvn(17), fvn(17), bvn(18), fvn(18), bvn(19), fvn(19), bvn(20), fvn(20), bvn(21), fvn(21), bvn(22), fvn(22), bvn(23), fvn(23), bvn(24), fvn(24), thsad=thsad, plane=plane, limit=limit, thscd1=thscd1, thscd2=thscd2)
    else:
       raise ValueError("Tr must be between 1 and 24. Upgrade mvtools-sf to r10 pre-release or newer for larger radii.")
    return dgn
# fmt: on


def ContraSharpening(clip, src, radius=None, rep=24, planes=[0, 1, 2]):
    # simplified function from G41Fun https://github.com/Vapoursynth-Plugins-Gitify/G41Fun
    # original avisynth function by Didée at the VERY GRAINY thread https://forum.doom9.org/showthread.php?p=1076491

    if radius is None:
        radius = 2 if clip.width > 960 else 1
    if clip.format.num_planes == 1:
        planes = [0]
    if isinstance(planes, int):
        planes = [planes]

    mat1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    mat2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    bd = clip.format.bits_per_sample
    mid = 1 << (bd - 1)
    num = clip.format.num_planes
    R = core.rgvs.Repair

    s = MinBlur(clip, planes)  # damp down remaining spots of the denoised clip
    RG11 = core.std.Convolution(s, matrix=mat1, planes=planes).std.Convolution(matrix=mat2, planes=planes)
    ssD = core.std.MakeDiff(s, RG11, planes)  # the difference of a simple kernel blur
    allD = core.std.MakeDiff(src, clip, planes)  # the difference achieved by the denoising
    ssDD = R(ssD, allD, [rep if i in planes else 0 for i in range(num)])  # limit the difference to the max of what the denoising removed locally
    expr = "x {} - abs y {} - abs < x y ?".format(mid, mid)  # abs(diff) after limiting may not be bigger than before
    ssDD = core.std.Expr([ssDD, ssD], [expr if i in planes else "" for i in range(num)])
    return core.std.MergeDiff(clip, ssDD, planes)  # apply the limited difference (sharpening is just inverse blurring)


def MinBlur(clip, planes=[0, 1, 2]):
    # simplified function from G41Fun https://github.com/Vapoursynth-Plugins-Gitify/G41Fun
    # original avisynth function by Didée https://avisynth.nl/index.php/MinBlur
    # Nifty Gauss/Median combination

    if clip.format.num_planes == 1:
        planes = [0]
    if isinstance(planes, int):
        planes = [planes]
    mat1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    mat2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    RG11 = core.std.Convolution(clip, matrix=mat1, planes=planes).std.Convolution(matrix=mat2, planes=planes)
    RG4 = core.ctmf.CTMF(clip, radius=2, planes=planes)
    expr = "x y - x z - * 0 < x dup y - abs x z - abs < y z ? ?"
    return core.std.Expr([clip, RG11, RG4], [expr if i in planes else "" for i in range(clip.format.num_planes)])
