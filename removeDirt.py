import vapoursynth as vs
from vapoursynth import core

# dependencies:
# RemoveGrain (http://www.vapoursynth.com/doc/plugins/rgvs.html) or zsmooth (https://github.com/adworacz/zsmooth)
# MVTools (https://github.com/dubhater/vapoursynth-mvtools) or SVP dlls when gpu=True is used
# RemoveDirt (https://github.com/pinterf/removedirtvs, https://github.com/Rational-Encoding-Thaumaturgy/vapoursynth-removedirt)
# ChangeFPS (https://github.com/Selur/VapoursynthScriptsInHybrid/blob/master/ChangeFPS.py)
def RemoveDirt(input: vs.VideoNode, repmode: int=16, remgrainmode: int=17, limit: int=10) -> vs.VideoNode:
  
  zsmooth = hasattr(core, 'zsmooth')
  cleansed = core.zsmooth.Clense(input) if zsmooth else core.rgvs.Clense(input)
  sbegin = core.zsmooth.ForwardClense(input) if zsmooth else core.rgvs.ForwardClense(input)
  send =  core.zsmooth.BackwardClense(input) if zsmooth else core.rgvs.BackwardClense(input)
  if hasattr(core, 'removedirt') and hasattr(core.removedirt, 'SCSelect'):
    # Use removedirt.SCSelect if available
    scenechange = core.removedirt.SCSelect(input, sbegin, send, cleansed)
  else:
    # Fallback to rdvs.SCSelect
    scenechange = core.rdvs.SCSelect(input, sbegin, send, cleansed)
  if zsmooth:
    alt = core.zsmooth.Repair(scenechange, input, mode=[repmode,repmode,1])
    restore = core.zsmooth.Repair(cleansed, input, mode=[repmode,repmode,1])
  else:
    alt = core.rgvs.Repair(scenechange, input, mode=[repmode,repmode,1])
    restore = core.rgvs.Repair(cleansed, input, mode=[repmode,repmode,1])
  
  if hasattr(core, 'removedirt') and hasattr(core.removedirt, 'SCSelect'):
    # Use removedirt.RestoreMotionBlocks if available
    corrected = core.removedirt.RestoreMotionBlocks(cleansed, restore, neighbour=input, alternative=alt, gmthreshold=70, dist=1, dmode=2, noise=limit, noisy=12)
  else:
    # Fallback to rdvs.RestoreMotionBlocks
    corrected = core.rdvs.RestoreMotionBlocks(cleansed, restore, neighbour=input, alternative=alt, gmthreshold=70, dist=1, dmode=2, noise=limit, noisy=12)
  if zsmooth:
    return core.zsmooth.RemoveGrain(corrected, mode=[remgrainmode,remgrainmode,1])
  else:
    return core.rgvs.RemoveGrain(corrected, mode=[remgrainmode,remgrainmode,1])
  
def RemoveDirtMC(input: vs.VideoNode, limit: int=6, repmode: int=16, remgrainmode: int=17, block_size: int=8, block_over: int=4, gpu: bool=False) -> vs.VideoNode:
  if hasattr(core, 'zsmooth'):
    quad = core.zsmooth.RemoveGrain(input, mode=[12,0,1])   # blur the luma for searching motion vectors  orig avs: mode=12, modeU=-1
  else:
    quad = core.rgvs.RemoveGrain(input, mode=[12,0,1])   # blur the luma for searching motion vectors  orig avs: mode=12, modeU=-1
  if gpu:
    import ChangeFPS
    block_over = 0 if block_over == 0 else 1 if block_over == 2 else 2 if block_over == 4 else 3 
    Super = core.svp1.Super(quad, "{gpu:1,pel:4}")
    bvec = core.svp1.Analyse(Super['clip'], Super['data'], input, "{ gpu:1, block:{w:"+str(block_size)+", h:"+str(block_size)+",overlap:"+str(block_over)+"} }")
    fvec = core.svp1.Analyse(Super['clip'], Super['data'], input, "{ gpu:1, block:{w:"+str(block_size)+", h:"+str(block_size)+",overlap:"+str(block_over)+",special:{delta: 1}} }")
    backw = core.svp2.SmoothFps(quad,Super['clip'], Super['data'],bvec['clip'],bvec['data'],"{}") # here the frame rate is doubled
    forw = core.svp2.SmoothFps(quad,Super['clip'], Super['data'],fvec['clip'],fvec['data'],"{}")  # here the frame rate is doubled
    # since backw and forw now have twice the frame count I drop half the frames
    backw = ChangeFPS.ChangeFPS(backw,input.fps_num,input.fps_den)
    forw = ChangeFPS.ChangeFPS(forw,input.fps_num,input.fps_den)
  else:
    #block size of MAnalyze, blksize 8 is much better for 720x576 noisy source than blksize=16 
    #block overlapping of MAnalyze 0! 2 or 4 is not good for my noisy b&w 8mm film source
    i = core.mv.Super(quad, pel=2)    #  avs: i=MSuper(clip,pel=2, isse=false)
    bvec = core.mv.Analyse(super=i,isb=True, blksize=block_size,overlap=block_over, delta=1, truemotion=True, chroma=True)  
    fvec = core.mv.Analyse(super=i,isb=False, blksize=block_size,overlap=block_over, delta=1, truemotion=True, chroma=True)
    backw = core.mv.Flow(clip=quad,super=i,vectors=[bvec])
    forw  = core.mv.Flow(clip=quad,super=i,vectors=[fvec])

  clp = core.std.Interleave([backw,quad,forw])
  clp = RemoveDirt(clp, repmode=2, remgrainmode=17, limit=limit)
  clp = core.std.SelectEvery(clp,3,1)
  return clp
