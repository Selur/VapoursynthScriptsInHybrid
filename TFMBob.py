import vapoursynth as vs
core = vs.core

from helpers import NNEDI3

# port of AviSynth TFMBobN(..) by jagabo, see: https://forum.videohelp.com/threads/409273-QTGMC-shimmer-when-deinterlacing-a-duplicate-frame#post2687062

def TFMBobN(clip: vs.VideoNode,
            pp: int = 6,
            cthresh: int = 9,
            MI: int = 80,
            chroma: bool=False,
            openCL: bool = False):
  field = clip.get_frame(0).props['_FieldBased']
  # openCL only reorders the search - NNEDI3 takes whichever implementation is loaded.
  n = NNEDI3(clip, field=field + 1, nns=4, gpu=openCL)
  if field == 1:
    field = 0
  else:
   field = 1
  odd = core.tivtc.TFM(clip=clip, field=field, clip2=n[::2], PP=pp, cthresh=cthresh, MI=MI)
  even = core.tivtc.TFM(clip=clip, field=field, clip2=n[1::2], PP=pp, cthresh=cthresh, MI=MI)
  return core.std.Interleave([odd,even])

def TFMBobQ(clip: vs.VideoNode,
            pp: int = 6,
            cthresh: int = 9,
            MI: int = 80,
            chroma: bool=False,
            openCL: bool = False):
  import qtgmc
  field = clip.get_frame(0).props['_FieldBased']
  q = qtgmc.QTGMC(Input=clip, Preset="Fast", TFF=(field == 2), opencl=openCL) 
  if field == 1:
    field = 0
  else:
   field = 1
  odd = core.tivtc.TFM(clip=clip, field=field, clip2=q[::2], PP=pp, cthresh=cthresh, MI=MI, chroma=chroma)
  even = core.tivtc.TFM(clip=clip, field=field, clip2=q[1::2], PP=pp, cthresh=cthresh, MI=MI, chroma=chroma)
  return core.std.Interleave([odd,even])


