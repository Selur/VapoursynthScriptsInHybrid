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