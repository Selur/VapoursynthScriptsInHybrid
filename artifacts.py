from vapoursynth import core
import vapoursynth as vs

# VS port of a script by Didée http://forum.doom9.net/showthread.php?p=1402690#post1402690
# In my experience this filter works very good as a prefilter for SMDegrain(). 
# Filtering only luma seems to help to avoid ghost artefacts.
def DeSpot(o):
  osup = o.mv.Super(pel=2, sharp=2)
  bv1  = osup.mv.Analyse(isb=True, delta=1, blksize=8, overlap=4, search=4)
  fv1  = osup.mv.Analyse(isb=False,delta=1, blksize=8, overlap=4, search=4)
  bc1  = o.mv.Compensate(osup, bv1)
  fc1  = o.mv.Compensate(osup, fv1)
  
  clip = core.std.Interleave([fc1, o, bc1])
  
  if hasattr(core,'zsmooth'):
    clip = clip.zsmooth.Clense()
  else:
    clip = clip.rgvs.Clense()
  
  return clip.std.SelectEvery(cycle=3, offsets=1)

# Requires
# zsmooth: https://github.com/adworacz/zsmooth
# RemoveDirt: https://github.com/pinterf/RemoveDirt
def RemoveSpots(clip: vs.VideoNode, grey: bool = False, limit: int = 16) -> vs.VideoNode:
    planes = [0] if grey else [0, 1, 2]
    clensed = core.zsmooth.Clense(clip, planes=planes)
    sbegin  = core.zsmooth.ForwardClense(clip, planes=planes)
    send    = core.zsmooth.BackwardClense(clip, planes=planes)
    if hasattr(core, 'removedirt') and hasattr(core.removedirt, 'SCSelect'):
        scenechange = core.removedirt.SCSelect(clip, sbegin, send, clensed)  # input -> clip
        RESTORE     = core.removedirt.RestoreMotionBlocks
    else:
        scenechange = core.rmd.SCSelect(clip, sbegin, send, clensed)         # input -> clip
        RESTORE     = core.rmd.RestoreMotionBlocks
    rep_mode = [limit if p in planes else 0 for p in range(clip.format.num_planes)]
    alt     = core.zsmooth.Repair(scenechange, clip, mode=rep_mode)          # sc_selected -> scenechange
    restore = core.zsmooth.Repair(clensed,     clip, mode=rep_mode)
    corrected = RESTORE(
        clensed, restore,
        neighbour=clip,
        alternative=alt,
        gmthreshold=70,
        dist=1,
        dmode=2,
        noise=10,
        noisy=12,
        grey=grey,
    )
    return corrected

# Requires
# zsmooth: https://github.com/adworacz/zsmooth
# RemoveDirt: https://github.com/pinterf/RemoveDirt
# mvtools: https://github.com/Mr-Z-2697/vapoursynth-mvtools
def RemoveSpotsMC3X(clip: vs.VideoNode, limit: int = 6, grey: bool = False) -> vs.VideoNode:
    sup   = core.mv.Super(clip, pel=2)
    bvec  = core.mv.Analyse(sup, isb=False, blksize=8, delta=1, truemotion=True)
    fvec  = core.mv.Analyse(sup, isb=True,  blksize=8, delta=1, truemotion=True)

    backw = core.mv.Flow(clip, sup, bvec)
    forw  = core.mv.Flow(clip, sup, fvec)

    clp = core.std.Interleave([backw, clip, forw])

    clp = RemoveSpots(clp, grey=grey, limit=limit)
    clp = RemoveSpots(clp, grey=grey, limit=limit)
    clp = RemoveSpots(clp, grey=grey, limit=limit)

    clp = core.std.SelectEvery(clp, cycle=3, offsets=[1])

    return clp