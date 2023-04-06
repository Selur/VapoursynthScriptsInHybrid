import vapoursynth as vs
core = vs.core

def fillWithMVTools(clip):
  super = core.mv.Super(clip, pel=2)
  vfe = core.mv.Analyse(super, truemotion=True, isb=False, delta=1)
  vbe = core.mv.Analyse(super, truemotion=True, isb=True, delta=1)
  return core.mv.FlowInter(clip, super, mvbw=vbe, mvfw=vfe, time=50)
    
def fillWithRIFE(clip, firstframe=None, rifeModel=1, rifeTTA=False, rifeUHD=False, rifeThresh=0.15):
  clip1 = core.std.AssumeFPS(clip, fpsnum=1, fpsden=1)
  start = core.std.Trim(clip1, first=firstframe-1, length=1)
  end = core.std.Trim(clip1, first=firstframe+1, length=1)
  startend = start + end
  if clip.format != vs.RGBS:
    r = core.resize.Point(startend, format=vs.RGBS, matrix_in_s="709")
  if rifeThresh != 0:
    r = core.misc.SCDetect(clip=r,threshold=rifeThresh)
  r = core.rife.RIFE(r, model=rifeModel, tta=rifeTTA,uhd=rifeUHD)
  if clip.format != vs.RGBS:
    r = core.resize.Point(r, format=clip.format, matrix_s="709")

  r = core.std.Trim(r, first=1, last=1) 
  r = core.std.AssumeFPS(r, fpsnum=1, fpsden=1)
  a = core.std.Trim(clip1, first=0, last=firstframe-1) 
  b = core.std.Trim(clip1, first=firstframe+1)
  join = a + r + b
  return core.std.AssumeFPS(join, src=clip)


def fillWithGMFSSUnion(clip, firstframe=None, gmfsuModel=0, gmfsuThresh=0.15):
  from vsgmfss_union import gmfss_union
  clip1 = core.std.AssumeFPS(clip, fpsnum=1, fpsden=1)
  start = core.std.Trim(clip1, first=firstframe-1, length=1)
  end = core.std.Trim(clip1, first=firstframe+1, length=1)
  startend = start + end
  if clip.format != vs.RGBH:
    r = core.resize.Point(startend, format=vs.RGBH, matrix_in_s="709")
  r = gmfss_union(r, model=gmfsuModel, sc_threshold=gmfsuThresh)
  if clip.format != vs.RGBH:
    r = core.resize.Point(r, format=clip.format, matrix_s="709")
  r = core.std.Trim(r, first=1, last=1) 
  r = core.std.AssumeFPS(r, fpsnum=1, fpsden=1)
  a = core.std.Trim(clip1, first=0, last=firstframe-1) 
  b = core.std.Trim(clip1, first=firstframe+1)
  join = a + r + b
  return core.std.AssumeFPS(join, src=clip)



def fillWithSVP(clip, firstframe=None, gpu=False):  # Here I go wrong since I select the wrong frames 
  clip1 = core.std.AssumeFPS(clip, fpsnum=1, fpsden=1)
  start = core.std.Trim(clip1, first=firstframe-1, length=1)
  end = core.std.Trim(clip1, first=firstframe+1, length=1)
  startend = start + end
  
  if gpu:
    super  = core.svp1.Super(startend,"{gpu:1}")
  else:
    super  = core.svp1.Super(startend,"{gpu:0}")
  vectors= core.svp1.Analyse(super["clip"],super["data"],startend,"{}")
  r = core.svp2.SmoothFps(startend,super["clip"],super["data"],vectors["clip"],vectors["data"],"{}")
  
  r = core.std.Trim(r, first=1, last=1) 
  r = core.std.AssumeFPS(r, fpsnum=1, fpsden=1)
  a = core.std.Trim(clip1, first=0, last=firstframe-1) 
  b = core.std.Trim(clip1, first=firstframe+1)
  join = a + r + b
  return core.std.AssumeFPS(join, src=clip)
    
def FillSingleDrops(clip, thresh=0.3, method="mv", rifeModel=0, rifeTTA=False, rifeUHD=False, rifeThresh=0.15, gmfsuModel=0, gmfsuThresh=0.15, debug=False):
  core = vs.core
  if not isinstance(clip, vs.VideoNode):
      raise ValueError('This is not a clip')
      
  def selectFunc(n, f):
    if f.props['PlaneStatsDiff'] > thresh or n == 0:
      if debug:
        return core.text.Text(clip=clip,text="Org, diff: "+str(f.props['PlaneStatsDiff']),alignment=8)
      return clip
    else:
      if method == "mv":
        filldrops=fillWithMVTools(clip)
      elif method == "svp":
        filldrops=fillWithSVP(clip,n)
      elif method == "svp_gpu":
        filldrops=fillWithSVP(clip,n,gpu=True)
      elif method == "rife":
        filldrops = fillWithRIFE(clip,n,rifeModel,rifeTTA,rifeUHD,rifeThresh)
      elif method == "gmfssunion":
        filldrops = fillWithGMFSSUnion(clip,n,gmfsuModel,gmfsuThresh)
      else:
        raise vs.Error('FillDrops: Unknown method '+method)   
    if debug:
       return core.text.Text(clip=filldrops,text=method+", diff: "+str(f.props['PlaneStatsDiff']),alignment=8)
    return filldrops
          
  diffclip = core.std.PlaneStats(clip, clip[0] + clip)
  fixed = core.std.FrameEval(clip, selectFunc, prop_src=diffclip)
  return fixed