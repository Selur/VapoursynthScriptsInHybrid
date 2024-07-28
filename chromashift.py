import vapoursynth as vs
from vapoursynth import core

def ChromaShift(clip, shift_left=0, shift_right=0, shift_top=0, shift_bottom=0):
  
  core = vs.core
  chroma_u = core.std.ShufflePlanes(clip, planes=[1], colorfamily=vs.GRAY)
  ov = chroma_u;
  chroma_u = core.std.CropRel(chroma_u, left=shift_left)
  chroma_u = core.std.AddBorders(chroma_u, right=shift_left)
  chroma_u = core.std.CropRel(chroma_u, right=shift_right)
  chroma_u = core.std.AddBorders(chroma_u, left=shift_right)
  chroma_u = core.std.CropRel(chroma_u, top=shift_top)
  chroma_u = core.std.AddBorders(chroma_u, top=shift_top)
  chroma_u = core.std.CropRel(chroma_u, top=shift_bottom)
  chroma_u = core.std.AddBorders(chroma_u, top=shift_bottom)
  

  chroma_v = core.std.ShufflePlanes(clip, planes=[2], colorfamily=vs.GRAY)
  ov = chroma_v;
  chroma_v = core.std.CropRel(chroma_v, left=shift_left)
  chroma_v = core.std.AddBorders(chroma_v, right=shift_left)
  chroma_v = core.std.CropRel(chroma_v, right=shift_right)
  chroma_v = core.std.AddBorders(chroma_v, left=shift_right)
  chroma_v = core.std.CropRel(chroma_v, top=shift_top)
  chroma_v = core.std.AddBorders(chroma_v, top=shift_top)
  chroma_v = core.std.CropRel(chroma_v, top=shift_bottom)
  chroma_v = core.std.AddBorders(chroma_v, top=shift_bottom)
  
  
  clip = core.std.ShufflePlanes(clips=[clip, chroma_u, chroma_v], planes=[0, 0, 0], colorfamily=vs.YUV)  
  return clip
  
# added ChromaShiftSP from https://forum.doom9.org/showthread.php?p=1951117#post1951117
def ChromaShiftSP (clip, X=0.0, Y=0.0, shiftU=True, shiftV=True, jeh=True):
	#Vapoursynth version of Avisynth ChromaShiftSP
	#Original AVS ChromaShift_SP: Shift chroma with subpixel accuracy, basic function by IanB, made standalone by McCauley
	#X: positive values shift the chroma to left, negative values to right
  #Y: positive values shift the chroma upwards, negative values downwards
  #shifU: shift U plane
  #shifV: shift V plane
  Yplane = core.std.ShufflePlanes(clip, planes=0, colorfamily=vs.GRAY)
  
  #clp.MergeChroma(clp.Spline16Resize(w, h, X, Y, w+X, h+Y)) } # original
  #clp.MergeChroma(clp.Spline16Resize(w, h, X, Y, w, h)) } # JEH fix
  if jeh:
    shift = core.resize.Spline16(clip, width=clip.width, height=clip.height, src_left=X, src_top=Y, src_width=clip.width, src_height=clip.height)
  else:
    shift = core.resize.Spline16(clip, width=clip.width, height=clip.height, src_left=X, src_top=Y, src_width=clip.width + X, src_height=clip.height + Y)
  
  if not shiftU and  not shiftV:
    raise vs.Error('ChromaShiftSP: at least U or V needs to be shifted!')
  if shiftU and shiftV:
	  merge = core.std.ShufflePlanes(clips=[Yplane, shift], planes=[0, 1, 2], colorfamily=vs.YUV)
  elif shiftU:
    Vplane = core.std.ShufflePlanes(clip, planes=2, colorfamily=vs.GRAY)
    merge = core.std.ShufflePlanes(clips=[Yplane, shift, Vplane], planes=[0, 1, 0], colorfamily=vs.YUV)
  elif shiftV:
    Uplane = core.std.ShufflePlanes(clip, planes=1, colorfamily=vs.GRAY)
    merge = core.std.ShufflePlanes(clips=[Yplane, Uplane, shift], planes=[0, 0, 2], colorfamily=vs.YUV)
  return merge




