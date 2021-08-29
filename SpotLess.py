import vapoursynth as vs
core = vs.core
#analyse_args= dict(blksize=bs, overlap=bs//2, search=5)
#analyse_args= dict(blksize=bs//2, overlap=bs//4, search=5)
def SpotLess(clip, chroma=True, rec=False, radT=1, ablksz=None, aoverlap=None, asearch=5, pel=None, rblksz=None, roverlap=None, rsearch=None ):
    """
    Args:
        chroma (bool) - Whether to process chroma.
        rec    (bool) - Recalculate the motion vectors to obtain more precision.
        radT   (int)  - Temporal radius in frames
        
        ablksz   - mvtools Analyse blocksize, if not set ablksz = 32 if clip.width > 2400 else 16 if clip.width > 960 else 8, otherwise 4/8/16/32/64
        aoverlap - mvtools Analyse overlap, if not set ablksz/2, otherwise 4/8/16/32/64
        asearch  -  mvtools Analyse search, it not set 5, other wise 1-7        
            search = 0 : 'OneTimeSearch'. searchparam is the step between each vectors tried ( if searchparam is superior to 1, step will be progressively refined ).
            search = 1 : 'NStepSearch'. N is set by searchparam. It's the most well known of the MV search algorithm.
            search = 2 : Logarithmic search, also named Diamond Search. searchparam is the initial step search, there again, it is refined progressively.
            search = 3 : Exhaustive search, searchparam is the radius (square side is 2*radius+1). It is slow, but it gives the best results, SAD-wise.
            search = 4 : Hexagon search, searchparam is the range. (similar to x264).
            search = 5 : Uneven Multi Hexagon (UMH) search, searchparam is the range. (similar to x264).
            search = 6 : pure Horizontal exhaustive search, searchparam is the radius (width is 2*radius+1).
            search = 7 : pure Vertical exhaustive search, searchparam is the radius (height is 2*radius+1).
        rblksz/roverlap/rsearch, same as axxx but for the recalculation
    """
    # modified from lostfunc: https://github.com/theChaosCoder/lostfunc/blob/v1/lostfunc.py#L10

    if not isinstance(clip, vs.VideoNode) or clip.format.color_family not in [vs.GRAY, vs.YUV]:
        raise TypeError("SpotLess: This is not a GRAY or YUV clip!")

    isFLOAT = clip.format.sample_type == vs.FLOAT
    isGRAY = clip.format.color_family == vs.GRAY
    chroma = False if isGRAY else chroma
    planes = [0, 1, 2] if chroma else [0]
    A = core.mvsf.Analyse if isFLOAT else core.mv.Analyse
    C = core.mvsf.Compensate if isFLOAT else core.mv.Compensate
    S = core.mvsf.Super if isFLOAT else core.mv.Super
    R = core.mvsf.Recalculate if isFLOAT else core.mv.Recalculate
    if ablksz == None:
      ablksz = 32 if clip.width > 2400 else 16 if clip.width > 960 else 8
      
    if aoverlap is None:
      aoverlap = ablksz//2
      
    if asearch is None:
      asearch = 5    

    if pel is None:
      pel = 1 if clip.width > 960 else 2
      
    if rsearch is None:
      rsearch = asearch
      
    if roverlap is None:
      roverlap = aoverlap
      
    if rsearch is None:
      rsearch = asearch
      
    sup = S(clip, pel=pel, sharp=1, rfilter=4)

    bv1 = A(sup, isb=True,  delta=radT, blksize=ablksz, overlap=aoverlap, search=asearch)
    fv1 = A(sup, isb=False, delta=radT, blksize=ablksz, overlap=aoverlap, search=asearch)

    if rec:
        bv1 = R(sup, bv1, blksize=rblksz, overlap=roverlap, search=rsearch)
        fv1 = R(sup, fv1, blksize=rblksz, overlap=roverlap, search=rsearch)

    bc1 = C(clip, sup, bv1)
    fc1 = C(clip, sup, fv1)
    fcb = core.std.Interleave([fc1, clip, bc1])

    return fcb.tmedian.TemporalMedian(1, planes)[1::3]
