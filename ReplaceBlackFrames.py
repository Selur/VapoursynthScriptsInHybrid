import vapoursynth as vs
from vapoursynth import core

'''
call using:

from ReplaceBlackFrames import ReplaceBlackFrames
rbf = ReplaceBlackFrames(clip, debug=True, thresh=0.1, method='previous')
clip = rbf.out

debug: whether to display the average luma
method: 
 'previous': replace black frames with the last non-black frame
 'interpolateSVP': replace black frames whit interpolatied frames using SVP (GPU)
 'interpolateSVPCPU': replace black frames whit interpolatied frames using SVP (CPU)


v0.0.1 base version, special thanks to _AI_
'''

class ReplaceBlackFrames:
  # constructor
  def __init__(self, clip: vs.VideoNode, thresh: float=0.1, debug: bool=False, method: str='previous'):
      self.clip = core.std.PlaneStats(clip)
      self.thresh = thresh
      self.debug = debug
      self.method = method
      self.smooth = None

  def previous(self, n, f):
    out = self.get_current_or_previous(n)
    if self.debug:
      return out.text.Text(text="Org, avg: "+str(f.props['PlaneStatsAverage']),alignment=8)            
    return out
  
  def interpolate(self, n, f):
    out = self.get_current_or_interpolate(n)
    if self.debug:
      return out.text.Text(text="avg: "+str(f.props['PlaneStatsAverage']),alignment=8)            
    return out

  def get_current_or_previous(self, n):
    for i in reversed(range(n+1)):
      if self.is_not_black(i):
        return self.clip[i]
    else:
      #all previous are black, return current n frame
      return self.clip[n]

  def get_current_or_interpolate(self, n):
    if self.is_not_black(n):
      #current non black selected
      return self.clip[n]

    #black frame, frame is interpolated
    for start in reversed(range(n+1)):
      if self.is_not_black(start):
        break
    else: #there are all black frames preceding n, return current n frame // will be executed then for-look does not end with a break
      return self.clip[n]
  
    for end in range(n, len(self.clip)):
      if self.is_not_black(end):
        break
    else:
      #there are all black frames to the end, return current n frame
      return self.clip[n]

    #does interpolated smooth clip exist for requested n frame? Use n frame from it.
    if self.smooth is not None and start >= self.smooth_start and end <= self.smooth_end:
      return self.smooth[n-start]

    #interpolating two frame clip  into end-start+1 fps
    clip = self.clip[start] + self.clip[end]
    clip = clip.std.AssumeFPS(fpsnum=1, fpsden=1)
    if self.method == 'interpolateSVP':
      super = core.svp1.Super(clip,"{gpu:1}")
    else:
      super = core.svp1.Super(clip,"{gpu:0}")
    vectors = core.svp1.Analyse(super["clip"],super["data"],clip,"{}")
    num = end - start + 1
    self.smooth = core.svp2.SmoothFps(clip,super["clip"],super["data"],vectors["clip"],vectors["data"],f"{{rate:{{num:{num},den:1,abs:true}}}}")
    self.smooth_start = start
    self.smooth_end   = end
    return self.smooth[n-start]

  def is_not_black(self, n):
    return self.clip.get_frame(n).props['PlaneStatsAverage'] > self.thresh
  
  @property
  def out(self):
    if self.method == 'previous':
      return core.std.FrameEval(self.clip, self.previous, prop_src=self.clip)
    else:
      return core.std.FrameEval(self.clip, self.interpolate, prop_src=self.clip)