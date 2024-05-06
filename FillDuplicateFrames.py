import vapoursynth as vs
from vapoursynth import core

'''
call using:

from FillDuplicateFrames import FillDuplicateFrames
fdf = FillDuplicateFrames(clip, debug=True, thresh=0.001, method='SVP')
//fdf = FillDuplicateFrames(clip, debug=True, thresh=0.001, method='MV')
//fdf = FillDuplicateFrames(clip, debug=True, thresh=0.001, method='RIFE')
clip = fdf.out

Replaces duplicate frames with interpolations.
v0.0.3
0.0.4 removed and added back RGBH support or RIFE
0.0.3 allow to set device_index for RIFE and support RGBH input for RIFE
0.0.4 removed RGBH since RIFE ncnn does not support it
0.0.5 add general sceneThr
'''

class FillDuplicateFrames:
  # constructor
  def __init__(self, clip: vs.VideoNode, thresh: float=0.001, method: str='SVP', debug: bool=False, sceneThr: float=0.15, device_index: int=0):
      # calculte stats
      self.thresh = thresh
      self.debug = debug
      self.method = method
      self.smooth = None
      self.sceneThr = sceneThr
      self.device_index = device_index
      self.clip = core.std.PlaneStats(clip, clip[0]+clip)
      if sceneThr > 0 and method == 'RIFE':
        clip = core.misc.SCDetect(clip=clip,threshold=sceneThr)
          
  def interpolate(self, n, f):
    out = self.get_current_or_interpolate(n)
    if self.debug:
      return out.text.Text(text="avg: "+str(f.props['PlaneStatsDiff']),alignment=8)            
    return out

  def interpolateWithRIFE(self, clip, n, start, end, rifeModel=22, rifeTTA=False, rifeUHD=False):
    if clip.format.id != vs.RGBS:
      raise ValueError(f'FillDuplicateFrames: "clip" needs to be RGBS when using \'{self.method}\'!')     
        
    num = end - start
    
    self.smooth = core.rife.RIFE(clip, model=rifeModel, factor_num=num, tta=rifeTTA,uhd=rifeUHD,gpu_id=self.device_index)
    self.smooth_start = start
    self.smooth_end   = end
    return self.smooth[n-start]
   
  def interpolateWithMV(self, clip, n, start, end):   
    num = end - start
    sup = core.mv.Super(clip, pel=2, hpad=0, vpad=0)
    bvec = core.mv.Analyse(sup, blksize=16, isb=True, chroma=True, search=3, searchparam=1)
    fvec = core.mv.Analyse(sup, blksize=16, isb=False, chroma=True, search=3, searchparam=1)
    self.smooth = core.mv.FlowFPS(clip, sup, bvec, fvec, num=num, den=1, mask=2)
    self.smooth_start = start
    self.smooth_end   = end
    out = self.smooth[n-start]
    if self.debug:
      return out.text.Text(text="MV",alignment=9)
    return out

  def interpolateWithSVP(self, clip, n, start, end):   
    if clip.format.id != vs.YUV420P8:
      raise ValueError(f'FillDuplicateFrames: "clip" needs to be YUV420P8 when using \'{self.method}\'!')
    if self.method == 'SVP':
      super = core.svp1.Super(clip,"{gpu:1}")
    else: # self.method == 'SVPCPU':
      super = core.svp1.Super(clip,"{gpu:0}")
    vectors = core.svp1.Analyse(super["clip"],super["data"],clip,"{}")
    num = end - start
    self.smooth = core.svp2.SmoothFps(clip,super["clip"],super["data"],vectors["clip"],vectors["data"],f"{{rate:{{num:{num},den:1,abs:true}}}}")
    self.smooth_start = start
    self.smooth_end   = end
    out = self.smooth[n-start]
    if self.debug:
      return out.text.Text(text="SVP",alignment=9)
    return out
  
  def get_current_or_interpolate(self, n):
    if self.is_not_duplicate(n):
    
      if self.potential_scene_change(n):
        if self.debug:
          return self.clip[n].text.Text(text="Input (scene change - 1)", alignment=9)
      elif self.debug:
        #current non dublicate selected
        return self.clip[n].text.Text(text="Input (1)", alignment=9)
      return self.clip[n]

    #dublicate frame, frame is interpolated
    for start in reversed(range(n+1)):
      if self.is_not_duplicate(start):
        break
    else: #there are all black frames preceding n, return current n frame // will be executed then for-look does not end with a break
      if self.debug:
        return self.clip[n].text.Text(text="Input (2)", alignment=9)
      return self.clip[n]
  
    for end in range(n, len(self.clip)):
      if self.potential_scene_change(end):
        #there are all duplicate frames to the end, return current n frame
        if self.debug:
          return self.clip[n].text.Text(text="Input(before scene change)", alignment=9)
        return self.clip[n]
      if self.is_not_duplicate(end):
        break
    else:
      #there are all duplicate frames to the end, return current n frame
      if self.debug:
        return self.clip[n].text.Text(text="Input(3)", alignment=9)
      return self.clip[n]

    #does interpolated smooth clip exist for requested n frame? Use n frame from it.
    if self.smooth is not None and start >= self.smooth_start and end <= self.smooth_end:
      if self.debug:
        return self.smooth[n-start].text.Text(text=self.method, alignment=9)
      return self.smooth[n-start]

    #interpolating two frame clip  into end-start+1 fps
    clip = self.clip[start] + self.clip[end]
    clip = clip.std.AssumeFPS(fpsnum=1, fpsden=1)
    if self.method == 'SVP' or self.method == 'SVPcpu':  
      return self.interpolateWithSVP(clip, n, start, end)
    elif self.method == 'RIFE':
      return self.interpolateWithRIFE(clip, n, start, end)
    elif self.method == 'MV':
      return self.interpolateWithMV(clip, n, start, end)
    else:
      raise ValueError(f'FillDuplicateFrames: "method" \'{self.method}\' is not supported atm.')

  def is_not_duplicate(self, n):
    return self.clip.get_frame(n).props['PlaneStatsDiff'] > self.thresh
  
  def potential_scene_change(self, n):
    return self.clip.get_frame(n).props['PlaneStatsDiff'] > self.sceneThr
  
  @property
  def out(self):
    return core.std.FrameEval(self.clip, self.interpolate, prop_src=self.clip)
    
    


