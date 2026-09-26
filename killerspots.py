import vapoursynth as vs
from misc import get_mv
from helpers import tool_function

# dependencies:
# RemoveGrain (http://www.vapoursynth.com/doc/plugins/rgvs.html) or zsmooth (https://github.com/adworacz/zsmooth)
# MVTools (https://github.com/dubhater/vapoursynth-mvtools) or SVP dlls when gpu=True is used
# RemoveDirt (https://github.com/pinterf/removedirtvs)

# based on https://github.com/FranceBB/KillerSpots/blob/main/KillerSpots.avsi 2.0 

# Function KillerSpots 2.0
# For primitive videos
# Function for spot removal
# Original idea by Didée (https://forum.doom9.org/showthread.php?p=1402690#post1402690 <> SpotRemover)
# Adapted by GMJCZP
# Requirements: MVTools, RGTools, RemoveDirt

def KillerSpots(clip: vs.VideoNode, limit: int=10, advanced: bool=False, tools=None):
  core = vs.core
  MV = get_mv(tools)
  # advanced: Use 'False' for best speed and original KillerSpots. Use 'True' to specify a 'limit'. Default True;
  # limit: default 10, spot removal limit (for advanced=true only)
  osup = MV.Super(clip=clip, pel=2, sharp=2, blksize=8, overlap=4)
  bv1  = MV.Analyse(super=osup, isb=True, delta=1, blksize=8, overlap=4, search=4)
  fv1  = MV.Analyse(super=osup, isb=False, delta=1, blksize=8, overlap=4, search=4)
  bc1  = MV.Compensate(clip, osup, bv1)
  fc1  = MV.Compensate(clip, osup, fv1)
  clip = core.std.Interleave([fc1, clip, bc1])
  if advanced:
    clip = RemoveDirtMod(clip, limit, tools=tools)
  else:
    clip = tool_function(tools, 'rg', 'Clense')(clip)
  clip = core.std.SelectEvery(clip=clip, cycle=3, offsets=1)
  return clip;

# From function RemoveDirt, original adaptation thanks to johnmeyer
def RemoveDirtMod(clip: vs.VideoNode, limit: int =10, tools=None):
  core = vs.core
  clensed = tool_function(tools, 'rg', 'Clense')(clip)
  alt = tool_function(tools, 'rg', 'RemoveGrain')(clip,mode=1)
  restore = tool_function(tools, 'removedirt', 'RestoreMotionBlocks', ('removedirt', 'rdvs'))
  clip = restore(clensed, clip, alternative=alt, pthreshold=4, cthreshold=6, gmthreshold=40, dist=3, dmode=2, noise=limit, noisy=12)
  return clip
