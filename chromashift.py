import vapoursynth as vs

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