import vapoursynth as vs
from vapoursynth import core

def ediaaCuda(a):
    """
    Suggested by Mystery Keeper in "Denoise of tv-anime" thread
    Read the document of Avisynth version for more details.
    requirement: https://github.com/AmusementClub/VapourSynth-EEDI2CUDA/releases
    """

    funcName = 'ediaa_cuda'

    if not isinstance(a, vs.VideoNode):
        raise TypeError(funcName + ': \"a\" must be a clip!')

    last = core.eedi2cuda.EEDI2(a, field=1).std.Transpose()
    last = core.eedi2cuda.EEDI2(last, field=1).std.Transpose()
    last = core.resize.Spline36(last, a.width, a.height, src_left=-0.5, src_top=-0.5)

    return last