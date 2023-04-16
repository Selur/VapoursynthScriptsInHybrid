import vapoursynth as vs
core = vs.core

# port of AviSynth TFMBobN(..) by jagabo, see: https://forum.videohelp.com/threads/409273-QTGMC-shimmer-when-deinterlacing-a-duplicate-frame#post2687062
def TFMBobN(clip: vs.VideoNode,
            pp: int = 6,
            cthresh: int = 9,
            MI: int = 80,
            openCL: bool = False):
  field = clip.get_frame(0).props['_FieldBased']
  if openCL:
    n = core.nnedi3cl.NNEDI3CL(clip = clip, field=field,nns=4)
  else if hasattr(core, 'znedi3'):
    n = core.znedi3.nnedi3(clip=clip, field=field, nns=4)
  else:
    n = core.nnedi3.nnedi3(clip=clip, field=field, nns=4)
  if field == 1: #BFF
    switchedField = 2
  else:  #TFF
    switchedField = 1
  odd = core.tivtc.TFM(clip=clip, field=switchedField, clip2=n[::2], PP=pp, cthresh=cthresh, MI=MI)
  even = core.tivtc.TFM(clip=clip, field=switchedField, clip2=n[1::2], PP=pp, cthresh=cthresh, MI=MI)
  return core.std.Interleave([odd,even])

def TFMBobQ(clip: vs.VideoNode,
            pp: int = 6,
            cthresh: int = 9,
            MI: int = 80,
            openCL: bool = False):
  field = clip.get_frame(0).props['_FieldBased']
  q = havsfunc.QTGMC(Input=clip, Preset="Fast", TFF=(field == 2), opencl=openCL) 
  if field == 1: #BFF
    switchedField = 2
  else:  #TFF
    switchedField = 1
  odd = core.tivtc.TFM(clip=clip, field=switchedField, clip2=q[::2], PP=pp, cthresh=cthresh, MI=MI)
  even = core.tivtc.TFM(clip=clip, field=switchedField, clip2=q[1::2], PP=pp, cthresh=cthresh, MI=MI)
  return core.std.Interleave([odd,even])


