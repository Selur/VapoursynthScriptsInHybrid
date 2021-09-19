import vapoursynth as vs

def nnedi3aa(a, opencl=False, device=None):
    """Using nnedi3 (Emulgator):
    Read the document of Avisynth version for more details.
    """

    if not isinstance(a, vs.VideoNode):
        raise TypeError(funcName + ': \"a\" must be a clip!')

    if opencl:
      myNNEDI3 = vs.core.nnedi3cl.NNEDI3CL
      last = myNNEDI3(a, field=1, dh=True, device=device).std.Transpose()
      last = myNNEDI3(last, field=1, dh=True, device=device).std.Transpose()
    else:
      myNNEDI3 = vs.core.znedi3.nnedi3 if hasattr(vs.core, 'znedi3') else vs.core.nnedi3.nnedi3
      last = myNNEDI3(a, field=1, dh=True).std.Transpose()
      last = myNNEDI3(last, field=1, dh=True).std.Transpose()
      
    last = vs.core.resize.Spline36(last, a.width, a.height, src_left=-0.5, src_top=-0.5)
    return last;