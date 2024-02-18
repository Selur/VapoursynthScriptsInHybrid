import vapoursynth as vs
import sys
from vapoursynth import core
from typing import List

try:
  from vsgan import ESRGAN
except:
  pass

'''
DuplicateAwareResizing:
The idea of DuplicateAwareResizing is that if you have a clip with duplicate frames in it,
instead of applying the frame based resizer (without a temporal component) to each frame of a row of duplicates,
you just apply it to the first frame and than duplicate that frame to replace the duplicates.

call using:
from DuplicateAwareResizing import DAResizer
sr = DAResizer(clip, thresh=0.001, method='XXX')
clip = sr.out

'''
# wrapper for easier usage by _AI_
def daResizer(clip:         vs.VideoNode,
              tWidth:       int,
              tHeight:      int,
              thresh:       float=0.001,
              method:       str='Bicubic',
              vsgan_models: List[str]=None,
              debug:        bool=False,
              device_index: int=0):

  sr = DAResizer(clip, tWidth, tHeight, thresh, method, vsgan_models, debug, device_index)
  return sr.out

# tWidth, tHeight: target resolution
# thresh: threshold for duplicate detection
# method: resizing method to use
# model:  model parameter for vsgan
# debug:  add debug info to frame
# device_index: set device to be used for gpu based resizing
class DAResizer:
  # constructor
  def __init__(self, clip: vs.VideoNode, tWidth: int, tHeight: int, thresh: float=0.001, method: str='Bicubic', vsgan_models: List[str]=None, debug: bool=False, device_index: int=0):
    if clip.format.id == vs.RGBH:
      clip = core.resize.Bicubic(clip=clip, format=vs.RGBS)
      clip = core.std.PlaneStats(clip, clip[0]+clip)
      clip = core.resize.Bicubic(clip=clip, format=vs.RGBH)
      self.clip = clip
    else:
      self.clip = core.std.PlaneStats(clip, clip[0]+clip)
    self.thresh = thresh
    self.debug = debug
    self.method = method
    self.models = vsgan_models
    self.device_index = device_index
    self.tWidth = tWidth
    self.tHeight = tHeight
    if self.method == 'VSGAN' and vsgan_models == None:
      raise ValueError(f'DAResizer: "method" \'{self.method}\' called while not setting a model!')      
      
  def daResize(self, n, f):
    out = self.resize(n)
    if self.debug:
      if out.format.id == vs.RGBH:
        out = core.resize.Bicubic(clip=out, format=vs.RGBS)   
        out = core.text.Text(clip=out, text="avg: "+str(f.props['PlaneStatsDiff']),alignment=8)            
        out = core.resize.Bicubic(clip=out, format=vs.RGBH)   
      else:
        out = core.text.Text(clip=out, text="avg: "+str(f.props['PlaneStatsDiff']),alignment=8)            
    return out

  def resize(self, n):
    if self.is_duplicate(n):
      return self.previous
    
    if self.method == 'VSGAN':
      inFormat = self.clip[n].format.id
      self.vsgan = ESRGAN(clip=self.clip[n],device="cuda")
      for model in self.models:
        self.vsgan.load(model)
        self.vsgan.apply()
        resized = self.vsgan.clip
        if inFormat == vs.RGBH:
          resized = core.resize.Bicubic(clip=resized, format=vs.RGBS)  
        resized = core.fmtc.resample(clip=resized, w=self.tWidth, h=self.tHeight, kernel="spline64", interlaced=False, interlacedd=False)
        resized = core.resize.Bicubic(clip=resized, format=inFormat, range_s="full")
      self.previous = resized
      return resized
      
    # apply generig resizer by method 
    resize_method = getattr(self.clip[n].resize, self.method, None)
    if resize_method is None:
      raise ValueError(f'DAResizer: Unknown "method" \'{self.method}\' called!')   
    resized = resize_method(width=self.tWidth, height=self.tHeight)
    self.previous = resized
    return resized
  
  def is_duplicate(self, n):
    # first frame can't be a duplicate, after that check agains the threshold
    return n != 0 and self.clip.get_frame(n).props['PlaneStatsDiff'] <= self.thresh
  
  @property
  def out(self):
    # this only works on YUV atm. needs to be adjusted for VSGAN&co
    return core.std.FrameEval(self.clip.std.BlankClip(width=self.tWidth, height=self.tHeight), self.daResize, prop_src=self.clip)
    
    


