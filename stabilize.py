from vapoursynth import core
import vapoursynth as vs

import misc

def Stab(clp, range=5, dxmax=8, dymax=8, mirror=5):
    if not isinstance(clp, vs.VideoNode):
        raise TypeError('Stab: This is not a clip')

    clp = misc.SCDetect(clip=clp, threshold=0.25)

    if hasattr(core, 'zsmooth'):
        temp = core.zsmooth.TemporalSoften(clp, radius=7, threshold=[255], scenechange=-1, scalep=True)
        temp2 = core.zsmooth.TemporalSoften(clp, radius=1, threshold=[255], scenechange=-1, scalep=True)
        inter = core.std.Interleave([core.zsmooth.Repair(temp, temp2, 1), clp])
    else:
        temp = misc.AverageFrames(clp, weights=[1] * 15)
        temp2 = misc.AverageFrames(clp, weights=[1] * 3)
        inter = core.std.Interleave([core.rgvs.Repair(temp, temp2, 1), clp])

    # mvutensils (https://github.com/myrsloik/mvutensils) carries the Depan family over from
    # mvtools largely unchanged: DePanEstimate -> DepanEstimate, DePan -> DepanCompensate, same
    # parameter meanings. It has no `range`: in depan that only widened the neighbourhood of
    # frames estimated (and cached) per call, while frame n's own dx/dy still came solely from
    # the n-1 <-> n correlation, so dropping it does not change the estimate.
    if hasattr(core, 'mvu'):
        mdata = core.mvu.DepanEstimate(inter, trust=0, dxmax=dxmax, dymax=dymax)
        return core.mvu.DepanCompensate(inter, data=mdata, offset=-1, mirror=mirror)[::2]

    mdata = core.mv.DePanEstimate(inter, range=range, trust=0, dxmax=dxmax, dymax=dymax)
    return core.mv.DePan(inter, data=mdata, offset=-1, mirror=mirror)[::2]

    