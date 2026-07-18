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

    mdata = core.depan.DePanEstimate(inter, range=range, trust=0, dxmax=dxmax, dymax=dymax)
    return core.depan.DePan(inter, data=mdata, offset=-1, mirror=mirror)[::2]

    