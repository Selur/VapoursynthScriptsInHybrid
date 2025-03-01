from vapoursynth import core
import vapoursynth as vs

from typing import Optional, Union, Sequence

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

	temp = AverageFrames(clp, weights=[1] * 15, scenechange=25 / 255)
	inter = core.std.Interleave([core.rgvs.Repair(temp, AverageFrames(clp, weights=[1] * 3, scenechange=25 / 255), 1), clp])
	mdata = core.depan.DePanEstimate(inter, range=range, trust=0, dxmax=dxmax, dymax=dymax)
	last = core.depan.DePan(inter, data=mdata, offset=-1, mirror=mirror)
	return last[::2]
  
  
def AverageFrames(
    clip: vs.VideoNode, weights: Union[float, Sequence[float]], scenechange: Optional[float] = None, planes: Optional[Union[int, Sequence[int]]] = None
) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('AverageFrames: this is not a clip')

    if scenechange:
        clip = SCDetect(clip, threshold=scenechange)
    return clip.std.AverageFrames(weights=weights, scenechange=scenechange, planes=planes)
    
    
def SCDetect(clip: vs.VideoNode, threshold: float = 0.1) -> vs.VideoNode:
    def copy_property(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        fout = f[0].copy()
        fout.props['_SceneChangePrev'] = f[1].props['_SceneChangePrev']
        fout.props['_SceneChangeNext'] = f[1].props['_SceneChangeNext']
        return fout

    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('SCDetect: this is not a clip')

    sc = clip
    if clip.format.color_family == vs.RGB:
        sc = clip.resize.Point(format=vs.GRAY8, matrix_s='709')

    sc = sc.misc.SCDetect(threshold=threshold)
    if clip.format.color_family == vs.RGB:
        sc = clip.std.ModifyFrame(clips=[clip, sc], selector=copy_property)

    return sc