import vapoursynth as vs
core = vs.core
from misc import MV, SCDetect

def fillWithMVTools(clip):
    super = MV.Super(clip, pel=2, blksize=8, overlap=0)
    vfe = MV.Analyse(super, truemotion=True, isb=False, delta=1)
    vbe = MV.Analyse(super, truemotion=True, isb=True, delta=1)
    return MV.FlowInter(clip, super, mvbw=vbe, mvfw=vfe, time=50)
    
def fillWithRIFE(clip, firstframe=None, rifeModel=22, rifeTTA=False, rifeUHD=False, sceneThresh=0.15):
  clip1 = core.std.AssumeFPS(clip, fpsnum=1, fpsden=1)
  start = core.std.Trim(clip1, first=firstframe-1, length=1)
  end = core.std.Trim(clip1, first=firstframe+1, length=1)
  startend = start + end
  startend = core.resize.Bicubic(startend, format=vs.RGBS, matrix_in_s="709")
  r = core.rife.RIFE(startend, model=rifeModel, tta=rifeTTA, uhd=rifeUHD, sc=sceneThresh>0)
  r = core.resize.Bicubic(r, format=clip.format, matrix_s="709")
  r = core.std.Trim(r, first=1, last=1)
  r = core.std.AssumeFPS(r, fpsnum=1, fpsden=1)
  a = core.std.Trim(clip1, first=0, last=firstframe-1)
  b = core.std.Trim(clip1, first=firstframe+1)
  join = a + r + b
  return core.std.AssumeFPS(join, src=clip)

def fillWithSVP(clip, firstframe=None, gpu=False):
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


def fillWithGMFSSUnion(clip, firstframe=None, gmfssModel=0, sceneThresh=0.15):
  from vsgmfss_fortuna import gmfss_fortuna
  clip1 = core.std.AssumeFPS(clip, fpsnum=1, fpsden=1)
  start = core.std.Trim(clip1, first=firstframe-1, length=1)
  end = core.std.Trim(clip1, first=firstframe+1, length=1)
  startend = start + end
  r = core.resize.Bicubic(startend, format=vs.RGBH, matrix_in_s="709")
  r = gmfss_fortuna(r, model=gmfssModel, sc_threshold=sceneThresh)
  r = core.resize.Bicubic(r, format=clip.format, matrix_s="709")
  r = core.std.Trim(r, first=1, last=1) 
  r = core.std.AssumeFPS(r, fpsnum=1, fpsden=1)
  a = core.std.Trim(clip1, first=0, last=firstframe-1) 
  b = core.std.Trim(clip1, first=firstframe+1)
  join = a + r + b
  return core.std.AssumeFPS(join, src=clip)



def fillWithSVP(clip, firstframe=None, gpu=False):
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
    
def FillSingleDrops(clip, thresh=0.3, method="mv", rifeModel=22, rifeTTA=False, rifeUHD=False, sceneThresh=0.15, gmfssModel=0, debug=False):
  core = vs.core
  if not isinstance(clip, vs.VideoNode):
    raise ValueError('This is not a clip')

  if clip.format.color_family != vs.YUV:
    raise ValueError('FillSingleDrops requires YUV input')
    
  if sceneThresh != 0 and (method == "rife" or method == "gmfssfortuna"):
    clip = SCDetect(clip=clip, threshold=sceneThresh)
    
  def selectFunc(n, f):
    if f.props['PlaneStatsDiff'] > thresh or n == 0 or n >= clip.num_frames -1 :
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
        filldrops = fillWithRIFE(clip,n,rifeModel,rifeTTA,rifeUHD, sceneThresh)
      elif method == "gmfssfortuna":
        filldrops = fillWithGMFSSUnion(clip,n,gmfssModel,sceneThresh)
      else:
        raise vs.Error('FillDrops: Unknown method '+method)   
    if debug:
       return core.text.Text(clip=filldrops,text=method+", diff: "+str(f.props['PlaneStatsDiff']),alignment=8)
    return filldrops
          
  diffclip = core.std.PlaneStats(clip, clip[0] + clip)
  fixed = core.std.FrameEval(clip, selectFunc, prop_src=diffclip)
  return fixed

def InsertSingle(clip, afterEveryX=2, method="mv", rifeModel=0, rifeTTA=False, rifeUHD=False, sceneThresh=0.15, gmfssModel=0, debug=False):
  core = vs.core
  if not isinstance(clip, vs.VideoNode):
    raise ValueError('This is not a clip')
    
  if clip.format.color_family != vs.YUV:
    raise ValueError('InsertSingle requires YUV input')  
  
  if sceneThresh > 0 and (method == "rife" or method == "gmfssfortuna"):
    clip = SCDetect(clip=clip, threshold=sceneThresh)
       
  def selectFunc(n):
    if n == 0 or n%afterEveryX != 0:
      return clip
    else:
      
      if method == "mv":
        insertFrame=fillWithMVTools(clip)
      elif method == "svp":
        insertFrame=fillWithSVP(clip,n)
      elif method == "svp_gpu":
        insertFrame=fillWithSVP(clip,n,gpu=True)
      elif method == "rife":
        insertFrame = fillWithRIFE(clip,n,rifeModel,rifeTTA,rifeUHD,sceneThresh)
      elif method == "gmfssfortuna":
        insertFrame = fillWithGMFSSUnion(clip,n,gmfssModel,sceneThresh)
      else:
        raise vs.Error('InsertSingle: Unknown method '+method)   
    return insertFrame.text.Text("Interpolated")

  return core.std.FrameEval(clip, selectFunc)
  
  
def ReplaceSingle(clip, frameList, method="mv", rifeModel=0, rifeTTA=False, rifeUHD=False, sceneThresh=0.15, gmfssModel=0, debug=False):
    core = vs.core
    if not isinstance(clip, vs.VideoNode):
        raise ValueError('This is not a clip')
    if clip.format.color_family != vs.YUV:
       raise ValueError('ReplaceSingle requires YUV input')
    if sceneThresh != 0 and (method == "rife" or method == "gmfssfortuna"):
       clip = SCDetect(clip=clip, threshold=sceneThresh)

    def selectFunc(n):
      if n in frameList:
        if method == "mv":
            insertFrame = fillWithMVTools(clip)
        elif method == "svp":
            insertFrame = fillWithSVP(clip, n)
        elif method == "svp_gpu":
            insertFrame = fillWithSVP(clip, n, gpu=True)
        elif method == "rife":
            insertFrame = fillWithRIFE(clip, n, rifeModel, rifeTTA, rifeUHD, sceneThresh)
        elif method == "gmfssfortuna":
            insertFrame = fillWithGMFSSUnion(clip, n, gmfssModel, sceneThresh)
        else:
            raise vs.Error('replaceSingle: Unknown method ' + method)
        if debug:
           return core.text.Text(clip=insertFrame,text=method+", replaced frame: "+str(n),alignment=8)
        return insertFrame;
      if debug:
        return core.text.Text(clip=clip,text="kept frame: "+str(n),alignment=8)
      return clip
    return core.std.FrameEval(clip, selectFunc)


def fillWithMVToolsM(clip, start, end):
    pair = clip[start - 1:start] + clip[end + 1:end + 2]
    super = MV.Super(pair, pel=2, blksize=8, overlap=0)
    vfe = MV.Analyse(super, truemotion=True, isb=False, delta=1)
    vbe = MV.Analyse(super, truemotion=True, isb=True, delta=1)

    count = end - start + 1
    clips = []

    for i in range(count):
        time = 100.0 * (i + 1) / (count + 1)
        clips.append(MV.FlowInter(pair, super, mvbw=vbe, mvfw=vfe, time=time)[:1])

    return core.std.Splice(clips)


def fillWithRIFEM(clip, start, end, rifeModel=22, rifeTTA=False, rifeUHD=False, sceneThresh=0.15):
    clip1 = core.std.AssumeFPS(clip, fpsnum=1, fpsden=1)
    left = core.std.Trim(clip1, first=start - 1, length=1)
    right = core.std.Trim(clip1, first=end + 1, length=1)
    pair = core.resize.Bicubic(left + right, format=vs.RGBS, matrix_in_s="709")

    count = end - start + 1
    factor = count + 1

    r = core.rife.RIFE(pair, model=rifeModel, factor_num=factor, factor_den=1, tta=rifeTTA, uhd=rifeUHD, sc=sceneThresh > 0)
    r = core.resize.Bicubic(r, format=clip.format, matrix_s="709")

    return r[1:count + 1]


def ReplaceFlagged(clip, method="mv", debug=False):
    core = vs.core
    num_frames = clip.num_frames

    range_cache = {}
    replacements = {}

    def find_range(n, f):
        cached = range_cache.get(n)
        if cached is not None:
            return cached

        if not bool(f.props.get("_UseInterp", 0)):
            return None

        start = n
        while start > 0:
            if not bool(clip.get_frame(start - 1).props.get("_UseInterp", 0)):
                break
            start -= 1

        end = n
        while end < num_frames - 1:
            if not bool(clip.get_frame(end + 1).props.get("_UseInterp", 0)):
                break
            end += 1

        if start == 0 or end == num_frames - 1:
            return None

        result = (start, end)

        for i in range(start, end + 1):
            range_cache[i] = result

        return result

    def build_range(start, end):
        if method == "mv":
            interp = fillWithMVToolsM(clip, start, end)
        elif method == "rife":
            interp = fillWithRIFEM(clip, start, end)
        else:
            raise vs.Error("ReplaceFlagged: unsupported method " + method)

        count = end - start + 1

        for i in range(count):
            replacements[start + i] = interp[i:i + 1]

    def selectFunc(n, f):
        if not bool(f.props.get("_UseInterp", 0)):
            return clip

        if n not in replacements:
            range_info = find_range(n, f)

            if range_info is None:
                return clip

            start, end = range_info
            build_range(start, end)

        result = replacements[n]

        if debug:
            result = core.text.Text(result, text="INTERPOLATED n=" + str(n), alignment=8)

        return result

    return core.std.FrameEval(clip, selectFunc, prop_src=clip, clip_src=[clip])