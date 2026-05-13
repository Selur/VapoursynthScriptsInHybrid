from vapoursynth import core
import vapoursynth as vs

from typing import Optional, Union, Sequence
import misc

# Stab function that uses DePanEstimate from "Depan" plugin which supports 
# the range parameter. DePanEstimate from MvTools is missing that paramater.
# Function copied from https://github.com/HomeOfVapourSynthEvolution/havsfunc
##############################################################################
# Original script by g-force converted into a stand alone script by McCauley #
# latest version from December 10, 2008                                      #
##############################################################################
def Stab(clp, range=1, dxmax=4, dymax=4, mirror=0):
  if not isinstance(clp, vs.VideoNode):
    raise TypeError('Stab: This is not a clip')

  temp = misc.AverageFrames(clp, weights=[1] * 15, scenechange=25 / 255)
  if hasattr(core,'zsmooth'):
    inter = core.std.Interleave([core.zsmooth.Repair(temp, misc.AverageFrames(clp, weights=[1] * 3, scenechange=25 / 255), 1), clp])
  else:
    inter = core.std.Interleave([core.rgvs.Repair(temp, misc.AverageFrames(clp, weights=[1] * 3, scenechange=25 / 255), 1), clp])
  mdata = core.depan.DePanEstimate(inter, range=range, trust=0, dxmax=dxmax, dymax=dymax)
  last = core.depan.DePan(inter, data=mdata, offset=-1, mirror=mirror)
  return last[::2]
  
  

    