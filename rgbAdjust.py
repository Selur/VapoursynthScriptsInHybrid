# Imports
import vapoursynth as vs

# getting Vapoursynth core
core = vs.core

# based on: https://forum.videohelp.com/threads/396285-Converting-Blu-Ray-YUV-to-RGB-and-back-to-YUV#post2576719 by  _Al_
def Adjust(rgb: vs.VideoNode, r: float=1.0, g: float=1.0, b: float=1.0, a: float=1.0, rb: float=0.0, gb: float=0.0, bb: float=0.0, ab: float=0.0, rg: float=1.0, gg: float=1.0, bg: float=1.0, ag: float=1.0):
  funcName = 'RGBAdjust'
  if rgb.format.color_family != vs.RGB:
    raise ValueError(funcName + ': input clip needs to be RGB!')
  type = rgb.format.sample_type
  size = 2**rgb.format.bits_per_sample
  #adjusting bias values rb,gb,bb for any RGB bit depth
  limited = rgb.get_frame(0).props['_ColorRange'] == 1
  if limited:
    if rb > 235 or rb < -235: raise ValueError(funcName + ': source is flagged as "limited" but rb is out of range [-235,235]!')  
    if gb > 235 or gb < -235: raise ValueError(funcName + ': source is flagged as "limited" but gb is out of range [-235,235]!')
    if bb > 235 or bb < -235: raise ValueError(funcName + ': source is flagged as "limited" but bb is out of range [-235,235]!')
  else:
    if rb > 255 or rb < -255: raise ValueError(funcName + ': source is flagged as "full" but rb is out of range [-255,255]!') 
    if gb > 255 or gb < -255: raise ValueError(funcName + ': source is flagged as "limited" but gb is out of range [-235,235]!')
    if bb > 255 or bb < -255: raise ValueError(funcName + ': source is flagged as "limited" but bb is out of range [-235,235]!')

  if rg < 0: raise ValueError(funcName + ': rg needs to be >= 0!')
  if gg < 0: raise ValueError(funcName + ': gg needs to be >= 0!')
  if bg < 0: raise ValueError(funcName + ': bg needs to be >= 0!')
      
  if limited:
    if type==vs.INTEGER:
      maxVal = 235
    else:
      maxVal = 235.0
  else:
    if type==vs.INTEGER:
      maxVal = 255
    else:
      maxVal = 255.0
  rb,gb,bb = map(lambda b: b if size==maxVal else size/maxVal*b if type==vs.INTEGER else b/maxVal, [rb,gb,bb])

  #x*r + rb , x*g + gb , x*b + bb
  rgb_adjusted = core.std.Expr(rgb, [f"x {r} * {rb} +", f"x {g} * {gb} +", f"x {b} * {bb} +"])

  #gamma per channel
  planes = [core.std.ShufflePlanes(rgb_adjusted, planes=p,  colorfamily=vs.GRAY)  for p in [0,1,2]]
  planes = [core.std.Levels(planes[p], gamma=g) if not g==1 else planes[p] for p, g in enumerate([rg, gg, bg])]
  rgb_adjusted = core.std.ShufflePlanes(planes, planes=[0,0,0], colorfamily = vs.RGB)
  return rgb_adjusted

