from vapoursynth import core
import vapoursynth as vs

import misc
from helpers import pick_tool, tool_function

def Stab(clp, range=5, dxmax=8, dymax=8, mirror=5, tools=None):
    if not isinstance(clp, vs.VideoNode):
        raise TypeError('Stab: This is not a clip')

    clp = misc.SCDetect(clip=clp, threshold=0.25, tools=tools)

    if pick_tool(tools, 'temporalsoften', ('zsmooth',)) == 'zsmooth':
        temp = core.zsmooth.TemporalSoften(clp, radius=7, threshold=[255], scenechange=-1, scalep=True)
        temp2 = core.zsmooth.TemporalSoften(clp, radius=1, threshold=[255], scenechange=-1, scalep=True)
    else:
        temp = misc.AverageFrames(clp, weights=[1] * 15, tools=tools)
        temp2 = misc.AverageFrames(clp, weights=[1] * 3, tools=tools)
    inter = core.std.Interleave([tool_function(tools, 'rg', 'Repair')(temp, temp2, 1), clp])

    # mvutensils (https://github.com/myrsloik/mvutensils) carries the Depan family over from
    # mvtools largely unchanged: DePanEstimate -> DepanEstimate, DePan -> DepanCompensate, same
    # parameter meanings. Neither this mvtools build nor mvutensils takes `range`: in depan that only widened the neighbourhood of
    # frames estimated (and cached) per call, while frame n's own dx/dy still came solely from
    # the n-1 <-> n correlation, so dropping it does not change the estimate.
    if pick_tool(tools, 'mv', ('mvutensils', 'mv')) == 'mvutensils':
        mdata = core.mvu.DepanEstimate(inter, trust=0, dxmax=dxmax, dymax=dymax)
        return core.mvu.DepanCompensate(inter, data=mdata, offset=-1, mirror=mirror)[::2]

    mdata = core.mv.DepanEstimate(inter, trust=0, dxmax=dxmax, dymax=dymax)
    return core.mv.DepanCompensate(inter, data=mdata, offset=-1, mirror=mirror)[::2]

    