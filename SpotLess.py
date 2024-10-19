import vapoursynth as vs
core = vs.core

# 2024.10.19: fix bug and made adjustments optional
# 2024.10.18: extended the script to also filter first and last frame

def SpotLess(clip: vs.VideoNode,
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
             mEnd: bool = False
            ) -> vs.VideoNode:
            
    """
      Args:
          radT   (int)  - Temporal radius in frames
          thsad -  mvtools ThSAD is SAD threshold for safe (dummy) compensation. (10000)
              If block SAD is above the thSAD, the block is bad, and we use source block instead of the compensated block. Default is 10000 (practically disabled).
          thsad2 - mvtools ThSAD that will be used for all calculations with radT > 1
          pel = mvtools Motion estimation accuracy. Value can only be 1, 2 or 4.
            1 : means precision to the pixel.
            2 : means precision to half a pixel.
            4 : means precision to quarter of a pixel, produced by spatial interpolation (more accurate but slower and not always better due to big level scale step). 
          chroma (bool) - Whether to process chroma.
          ablksize   - mvtools Analyse blocksize, if not set ablksz = 32 if clip.width > 2400 else 16 if clip.width > 960 else 8, otherwise 4/8/16/32/64
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
          pglobal: mkvootls apply relative penalty (scaled to 256) to SAD cost for global predictor vector.
          rec    (bool) - Recalculate the motion vectors to obtain more precision.
          rblksize/roverlap/rsearch, same as axxx but for the recalculation (rec=True)
          ssharp (int - mvtools Super sharp parameter. Sub-pixel interpolation method for when pel == 2 || 4.
            0 : soft interpolation (bilinear).
            1 : bicubic interpolation (4 tap Catmull-Rom)
            2 : sharper Wiener interpolation (6 tap, similar to Lanczos). 
          rfilter (int) - mvtools Super rfilter. Hierarchical levels smoothing and reducing (halving) filter.
            0 : simple 4 pixels averaging like unfiltered SimpleResize (old method)
            1 : triangle (shifted) filter like ReduceBy2 for more smoothing (decreased aliasing)
            2 : triangle filter like BilinearResize for even more smoothing
            3 : quadratic filter for even more smoothing
            4 : cubic filter like BicubicResize(b=1,c=0) for even more smoothing 
          blur (bool): apply additional bluring during mvtools Super
          truemotion: mvtools Analyse and Recalculate truemotion parameter.
          smoother (string): which smoother to use (tmedian, ttsmooth, zsmooth)
          mStart,  Default False. Mirror start to allow filtering first frame
          mEnd,    Default False. Mirror end to allow filtering first frame  
    """     
    fpsnum = clip.fps_num;
    fpsden = clip.fps_den;
    # Init variables and check sanity
    if radT < 1 or radT > 10:
      raise ValueError("Spotless: radT must be between 1 and 10 (inclusive)")
    if pel==None:
      pel = 1 if clip.width > 960 else 2
    if not pel in [1, 2, 4]:
      raise ValueError("Spotless: pel must be 1, 2 or 4")
    
    # Add mirrored frames if specified
    clip = core.std.Reverse(core.std.Trim(clip, 1, radT)) + clip if mStart else clip
    clip = core.std.Reverse(core.std.Trim(clip, clip.num_frames - radT - 1, clip.num_frames - 2)) + clip if mEnd else clip
    
    thsad2 = thsad2 or thsad
    thsad2 = (thsad + thsad2)/2 if radT>=3 else thsad2
    
    if not isinstance(clip, vs.VideoNode) or clip.format.color_family not in [vs.GRAY, vs.YUV]:
      raise ValueError("SpotLess: This is not a GRAY or YUV clip!")
          
    if ablksize is None:
      ablksize = 32 if clip.width > 2400 else 16 if clip.width > 960 else 8
    if aoverlap is None:
      aoverlap = ablksize//2
    aoverlap = min(aoverlap, ablksize//2)
    if asearch is None:
      asearch = 5  
    if asearch < 0 or asearch > 7:
      raise ValueError("Spotless: search must be between 0 and 7 (inclusive)!")      
    if rfilter is None:
      rfilter = 2
    if rfilter < 0 or rfilter > 4:
      raise ValueError("Spotless: rfilter must be between 0 and 4 (inclusive)")
        
    if ssharp is None:
      ssharp = 1
    if not ssharp in range(3):
      raise ValueError("Spotless: ssharp must be between 0 and 2 (inclusive)")

    if rec:
      if rblksize is None:
        rblksize = ablksize
      if rsearch is None:
        rsearch = asearch
      if rsearch < 0 or rsearch > 7:
        raise ValueError("Spotless: rsearch must between 0 and 7 (inclusive)!")  
      if roverlap is None:
        roverlap = aoverlap
      roverlap = min(roverlap, rblksize/2)
      
    # init functions
    isFLOAT = clip.format.sample_type == vs.FLOAT
    isGRAY = clip.format.color_family == vs.GRAY
    chroma = False if isGRAY else chroma
    planes = [0, 1, 2] if chroma else [0]
    
    A = core.mvsf.Analyse if isFLOAT else core.mv.Analyse
    C = core.mvsf.Compensate if isFLOAT else core.mv.Compensate
    S = core.mvsf.Super if isFLOAT else core.mv.Super
    R = core.mvsf.Recalculate if isFLOAT else core.mv.Recalculate
    
    # Super
    pad = max(ablksize, 8)
    sup = ref or (core.std.Convolution(clip, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1]) if blur else clip)
    sup = S(sup, hpad=pad, vpad=pad, pel=pel, sharp=ssharp, rfilter=rfilter)
    sup_rend = S(clip, pel=pel, sharp=ssharp, rfilter=rfilter, levels=1) if ref or blur else sup

    bv=[]
    fv=[]
    def doAnalysis(bv, fv, delta, search, blksize, overlap, chroma, truemotion, pglobal):
       bv.append(A(sup, isb=True, delta=delta, search=asearch, blksize=ablksize, overlap=aoverlap, chroma=chroma, truemotion=truemotion, pglobal=pglobal))
       fv.append(A(sup, isb=False, delta=delta, search=asearch, blksize=ablksize, overlap=aoverlap, chroma=chroma, truemotion=truemotion, pglobal=pglobal))

    # Analyse
    for delta in range(1, radT+1):
      doAnalysis(bv, fv, delta=delta, search=asearch, blksize=ablksize, overlap=aoverlap, chroma=chroma, truemotion=truemotion, pglobal=pglobal)
    
    def doRecalculate(bv, fv, delta, blksize, overlap, search, truemotion):
       bv[delta-1] = R(sup, bv[delta-1], blksize=blksize, overlap=overlap, search=search, truemotion=truemotion)
       fv[delta-1] = R(sup, fv[delta-1], blksize=blksize, overlap=overlap, search=search, truemotion=truemotion)
    
    if rec:
      for delta in range(1, radT+1):
        doRecalculate(bv, fv, delta, rblksize, roverlap, rsearch, truemotion=truemotion)

    bc=[]
    fc=[]
    def doCompensate(bc, fc, bv, fv, delta, thsad, thsad2):
      if delta != 1:
        thsad = thsad2

      bc.append(C(clip, sup_rend, bv[delta-1], thsad=thsad))
      fc.append(C(clip, sup_rend, fv[delta-1], thsad=thsad))
      
    # Compensate
    for delta in range(1, radT+1):
      doCompensate(bc, fc, bv, fv, delta, thsad, thsad2)
    
    ic =  core.std.Interleave(bc + [clip] + fc)
    if smoother == 'tmedian':
       output = core.tmedian.TemporalMedian(ic, radius=radT)
    elif smoother == 'ttsmooth':
       output = core.ttmpsm.TTempSmooth(ic, maxr=min(7, radT))
    elif smoother == 'zsmooth':   
       output = core.zsmooth.TemporalMedian(ic, radius=radT)
    else:
       raise ValueError("Spotless: unknown smoother "+smoother+"!")  
    
    output = core.std.SelectEvery(output, radT*2+1, radT)  # Return middle frame
        
    output = core.std.Trim(output, radT, output.num_frames - 1) if (mStart) else output
    output = core.std.Trim(output, 0, output.num_frames - 1 - radT) if (mEnd) else output
    return core.std.AssumeFPS(clip=output, fpsnum=fpsnum, fpsden=fpsden)
    