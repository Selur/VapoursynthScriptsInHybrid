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


def fillWithGMFSSUnion(clip, firstframe=None, gmfssModel=0, gmfssThresh=0.15):
  from vsgmfss_fortuna import gmfss_fortuna
  clip1 = core.std.AssumeFPS(clip, fpsnum=1, fpsden=1)
  start = core.std.Trim(clip1, first=firstframe-1, length=1)
  end = core.std.Trim(clip1, first=firstframe+1, length=1)
  startend = start + end
  if clip.format != vs.RGBH:
    r = core.resize.Point(startend, format=vs.RGBH, matrix_in_s="709")
  r = gmfss_fortuna(r, model=gmfssModel, sc_threshold=gmfssThresh)
  if clip.format != vs.RGBH:
    r = core.resize.Point(r, format=clip.format, matrix_s="709")
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
    
def FillSingleDrops(clip, thresh=0.3, method="mv", rifeModel=0, rifeTTA=False, rifeUHD=False, rifeThresh=0.15, gmfssModel=0, gmfssThresh=0.15, debug=False):
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
      elif method == "gmfssfortuna":
        filldrops = fillWithGMFSSUnion(clip,n,gmfssModel,gmfssThresh)
      else:
        raise vs.Error('FillDrops: Unknown method '+method)   
    if debug:
       return core.text.Text(clip=filldrops,text=method+", diff: "+str(f.props['PlaneStatsDiff']),alignment=8)
    return filldrops
          
  diffclip = core.std.PlaneStats(clip, clip[0] + clip)
  fixed = core.std.FrameEval(clip, selectFunc, prop_src=diffclip)
  return fixed


# replace a single black frame with last frame that was not black
def ReplaceBlackFrames(clip, thresh=0.1, debug=False):
  core = vs.core
  if not isinstance(clip, vs.VideoNode):
      raise ValueError('This is not a clip')
      
  def selectFunc(n, f):
    if f.props['PlaneStatsAverage'] > thresh or n == 0:
      if debug:
        return core.text.Text(clip=clip,text="Org, avg: "+str(f.props['PlaneStatsAverage']),alignment=8)
      return clip
    toReplace = n
    for i in range(n)[::-1]:
      if clip.get_frame(i).props['PlaneStatsAverage'] <= thresh:
        continue
      
      # remove current replace with frame before    
      start = core.std.Trim(clip, first=0, last=n-1) 
      r = core.std.Trim(clip, first=i, last=i)
      end = core.std.Trim(clip, first=n+1) 
      replaced = start + r + end
      break
    
    if debug:
       return core.text.Text(clip=replaced,text="Replaced, avg: "+str(f.props['PlaneStatsAverage']),alignment=8)
    return replaced
          
  clip = core.std.PlaneStats(clip)
  fixed = core.std.FrameEval(clip, selectFunc, prop_src=clip)
  return fixed
 
# replace the frames 'firstFrametoReplace' to 'firstGoodFrame-1' with interpolations between 'firstFrametoReplace - 1' and 'firstGoodFrame'
def fillMultipleWithSVP(clip, firstFrametoReplace=None, firstGoodFrame=None, gpu=False):
  clip1 = core.std.AssumeFPS(clip, fpsnum=1, fpsden=1)
  start = core.std.Trim(clip1, first=firstFrametoReplace-1, length=1) # last good frame before black frames
  end = core.std.Trim(clip1, first=firstGoodFrame, length=1) # first good frame after black frames
  startend = start + end
  if gpu:
    super  = core.svp1.Super(startend,"{gpu:1}")
  else:
    super  = core.svp1.Super(startend,"{gpu:0}")
  vectors= core.svp1.Analyse(super["clip"],super["data"],startend,"{}")

  # interpolate
  r = core.svp2.SmoothFps(startend,super["clip"],super["data"],vectors["clip"],vectors["data"],"{rate:{num:"+str(firstGoodFrame-firstFrametoReplace+1)+",den:1,abs:false}}")
  a = core.std.Trim(clip1, first=0, last=firstFrametoReplace-2) # last good frame before is part of r
  b = core.std.Trim(clip1, first=firstGoodFrame+1) # first good frame, ist part of r
  # join
  join = a + r + b
  if (join.num_frames != clip.num_frames): # did I messup with the join
    raise vs.Error(f"fillMultipleWithSVP: frame count issue join '{join.num_frames}' vs clip '{clip.num_frames}'")
  return core.std.AssumeFPS(join, src=clip)
  

# replace a black frames with interpolations of the surrounding frames.
# threshold average frame threshold
def FillBlackFrames(clip, thresh=0.1, debug=False):
    if not isinstance(clip, vs.VideoNode):
        raise ValueError('This is not a clip')
    
    def selectFunc(n, f):
        nonlocal clip
        firstGoodFrame = f.props['FirstGoodFrame']
        if n == 0:
          clip = core.std.SetFrameProp(clip, prop="FirstGoodFrame", intval=firstGoodFrame)
          return clip
        if firstGoodFrame >= n: # this frame, we already dealt with
            return clip

        if f.props['PlaneStatsAverage'] > thresh or n == 0:
            if debug:
                return core.text.Text(clip=clip, text="Org, avg: " + str(f.props['PlaneStatsAverage']), alignment=8)
            clip = core.std.SetFrameProp(clip, prop="FirstGoodFrame", intval=n)
            return clip
        firstFrametoReplace = n
        firstGoodFrame = n + 1
        for i in range(n, clip.num_frames-1):
            if clip.get_frame(i).props['PlaneStatsAverage'] <= thresh:
                continue
            firstGoodFrame = i
            break
        clip = core.std.SetFrameProp(clip, prop="FirstGoodFrame", intval=firstGoodFrame)
        replaced = fillMultipleWithSVP(clip, firstFrametoReplace, firstGoodFrame, gpu=True)
        if debug:
            return core.text.Text(clip=replaced, text="Replaced from " + str(firstFrametoReplace) + " to " + str(firstGoodFrame-1))
        return replaced
    clip = core.std.SetFrameProp(clip, prop="FirstGoodFrame", intval=0)
    clip = core.std.PlaneStats(clip)
    fixed = core.std.FrameEval(clip, selectFunc, prop_src=clip)
    return fixed