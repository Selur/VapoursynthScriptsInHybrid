import vapoursynth as vs
core = vs.core

def SpotLess(
    clip: vs.VideoNode,
    radT: int = 1,
    thsad: int = 10000,
    thsad2: int = None,
    pel: int = None,
    chroma: bool = True,
    ablksize: int = None,
    aoverlap: int = None,
    asearch: int = None,
    ssharp: int = None,
    pglobal: bool = True,
    rec: bool = False,
    rblksize: int = None, 
    roverlap: int = None,
    rsearch: int = None,
    truemotion: bool = True,
    rfilter: int = None,
    blur: bool = False,
    smoother: str = 'tmedian',
    ref: vs.VideoNode = None,
    mStart: bool = False,
    mEnd: bool = False,
    iterations: int = 1,
    debugmask: bool = False,
) -> vs.VideoNode:
    """
    SpotLess performs temporal denoising using motion estimation via mvtools.
    It is optimized for strong denoising and includes optional multi-pass smoothing,
    and adaptive thresholds.

    Args:
        clip:             Input clip. Must be constant format and frame rate.
        radT:             Temporal radius. 1 = use 1 backward + 1 forward frame. Range: 1–10.
        thsad:            SAD threshold for first temporal level.
        thsad2:           SAD threshold for further temporal levels (if radT >= 3). If None, defaults to thsad.
        pel:              Subpixel accuracy for motion estimation. Default: 1 (low-res) or 2 (HD).
        chroma:           Whether to use chroma planes for denoising. Auto-disabled for GRAY input.
        ablksize:         Block size for initial motion analysis. Default depends on clip width.
        aoverlap:         Overlap for motion analysis blocks. Default: half of ablksize.
        asearch:          Search type for motion analysis. Default: 5.
        ssharp:           Sharpness for mv.Super. Default: 1.
        pglobal:          Whether to use global motion estimation.
        rec:              Enable recalculation pass to refine motion vectors.
        rblksize:         Block size for recalculation. Default: ablksize.
        roverlap:         Overlap for recalculation blocks. Default: half of rblksize.
        rsearch:          Search type for recalculation. Default: asearch.
        truemotion:       Enables truemotion mode in mvtools for better accuracy on natural footage.
        rfilter:          Supersampling filter strength for mv.Super. Default: 2.
        blur:             Apply slight spatial blurring before vector analysis for stability.
        smoother:         Temporal smoother to apply. Options:
                          'tmedian'  = TemporalMedian,
                          'ttsmooth' = TTempSmooth,
                          'zsmooth'  = zsmooth (if available),
        ref:              Optional reference clip for mv.Super creation.
        mStart:           Mirror frames at the beginning to improve denoising at clip start.
        mEnd:             Mirror frames at the end to improve denoising at clip end.
        iterations:       How many times to apply the temporal denoising chain. >1 enables stronger denoising.
        debugmask:        If True, returns a stack showing [input | denoised | difference].

    Returns:
        A temporally denoised clip with preserved motion and structure.
        If debugmask is True, a 3-row stack is returned for visual inspection.

    Notes:
        - Mirror extension (mStart/mEnd) helps reduce denoising artifacts at the clip edges.
        - If chroma=False, only the luma plane is denoised.
        - Requires mvtools, tmedian, and optionally zsmooth plugins.
    """
    
    if radT < 1 or radT > 10:
        raise ValueError("radT must be between 1 and 10")
    if pel is None:
        pel = 1 if clip.width > 960 else 2
    if pel not in [1, 2, 4]:
        raise ValueError("pel must be 1, 2 or 4")

    isGRAY = clip.format.color_family == vs.GRAY
    chroma = False if isGRAY else chroma
    planes = [0, 1, 2] if chroma else [0]

    fpsnum, fpsden = clip.fps_num, clip.fps_den

    if ablksize is None:
        ablksize = 32 if clip.width > 2400 else 16 if clip.width > 960 else 8
    if aoverlap is None:
        aoverlap = ablksize // 2
    if asearch is None:
        asearch = 5
    if ssharp is None:
        ssharp = 1
    if rfilter is None:
        rfilter = 2

    if rec:
        rblksize = rblksize or ablksize
        rsearch = rsearch or asearch
        roverlap = roverlap or rblksize // 2

    thsad2 = thsad2 or thsad
    thsad2 = (thsad + thsad2) // 2 if radT >= 3 else thsad2

    S = core.mv.Super
    A = core.mv.Analyse
    R = core.mv.Recalculate
    C = core.mv.Compensate

    # Mirror at start/end if requested
    if mStart or mEnd:
        # Only mirror frames 1…radT (exactly radT frames) to avoid off-by-one
        head = core.std.Reverse(core.std.Trim(clip, 1, radT)) if mStart else None
        tail = core.std.Reverse(core.std.Trim(clip, clip.num_frames - radT, clip.num_frames - 1)) if mEnd else None
        if head and tail:
            clip = head + clip + tail
        elif head:
            clip = head + clip
        elif tail:
            clip = clip + tail

    denoised = clip

    for i in range(iterations):
        supclip = ref or (core.std.Convolution(denoised, matrix=[1,2,1,2,4,2,1,2,1]) if blur else denoised)
        sup = S(supclip, hpad=ablksize, vpad=ablksize, pel=pel, sharp=ssharp, rfilter=rfilter)
        sup_render = S(denoised, levels=1, pel=pel, sharp=ssharp, rfilter=rfilter)

        bv, fv = [], []
        for d in range(1, radT+1):
            bv.append(A(sup, isb=True, delta=d, search=asearch, blksize=ablksize, overlap=aoverlap,
                        chroma=chroma, truemotion=truemotion, pglobal=pglobal))
            fv.append(A(sup, isb=False, delta=d, search=asearch, blksize=ablksize, overlap=aoverlap,
                        chroma=chroma, truemotion=truemotion, pglobal=pglobal))

        if rec:
            for d in range(1, radT+1):
                bv[d-1] = R(sup, bv[d-1], blksize=rblksize, overlap=roverlap, search=rsearch, truemotion=truemotion)
                fv[d-1] = R(sup, fv[d-1], blksize=rblksize, overlap=roverlap, search=rsearch, truemotion=truemotion)

        bc, fc = [], []
        for d in range(1, radT+1):
            thresh = thsad if d == 1 else thsad2
            bc.append(C(denoised, sup_render, bv[d-1], thsad=thresh))
            fc.append(C(denoised, sup_render, fv[d-1], thsad=thresh))

        ic = core.std.Interleave(bc + [denoised] + fc)

        if smoother == 'tmedian':
            out = core.tmedian.TemporalMedian(ic, radius=radT)
        elif smoother == 'ttsmooth':
            out = core.ttmpsm.TTempSmooth(ic, maxr=min(7, radT))
        elif smoother == 'zsmooth':
            out = core.zsmooth.TemporalMedian(ic, radius=radT)
        else:
            raise ValueError(f"Unknown smoother '{smoother}'")

        out = core.std.SelectEvery(out, radT * 2 + 1, radT)
        denoised = out

    # Remove mirrored start/end again
    if mStart:
        denoised = core.std.Trim(denoised, radT, denoised.num_frames - 1)
    if mEnd:
        denoised = core.std.Trim(denoised, 0, denoised.num_frames - 1 - radT)

    if debugmask:
        mask = core.std.Expr([clip, denoised], expr=["x y - abs"])
        return core.std.StackVertical([clip, denoised, mask])

    return core.std.AssumeFPS(denoised, fpsnum=fpsnum, fpsden=fpsden)
