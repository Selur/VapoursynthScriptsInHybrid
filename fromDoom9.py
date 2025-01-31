from vapoursynth import core
import vapoursynth as vs
import math

# DeStripe works on YUVXXXPY
# "low frequency" stripes/bands removal filter
# requires https://github.com/AkarinVS/vapoursynth-plugin/releases
#
# int rad: search radius (default: 1, range: 1-5)
# int thr: blur threshold, wil be scaled by bit depth (default: 256, range: 1-256)
# boolean vertical: transposes the source for the filtering, to handle vertical lines instead of horizontal ones. (default: False)
# str hvmode: whether to use vertival or hoizontal convolution
def DeStripe(clip: vs.VideoNode, rad: int=2, offset: int=0, thr: int=256, vertical=False, hvmode: str='v') -> vs.VideoNode:
 
  if (rad < 1) or (rad > 5):
    raise vs.Error('rad not valid (range: 1-5)')
  if (offset < 0) or (offset > (rad-1)):
    raise vs.Error('rad not valid (range: 0-(rad-1)')
  if (hvmode != 'v' and hvmode != 'h'):
    raise vs.Error("mode kein either be 'h' or 'v'")
    
  thr = thr << (clip.format.bits_per_sample - 8) # scale thr by bit depth
  if vertical: 
    clip = core.std.Transpose(clip)
  
  MAP = {
      1: ([1,1,1],),
      2: ([1,1,1,1,1], [1,0,1,0,1]),
      3: ([1,1,1,1,1,1,1], [1,1,0,1,0,1,1], [1,0,0,1,0,0,1]),
      4: ([1,1,1,1,1,1,1,1,1 ], [1,1,1,0,1,0,1,1,1], [1,1,0,0,1,0,0,1,1], [1,0,0,0,1,0,0,0,1]),
      5: ([1,1,1,1,1,1,1,1,1,1,1], [1,1,1,1,0,1,0,1,1,1,1], [1,1,1,0,0,1,0,0,1,1,1], [1,1,0,0,0,1,0,0,0,1,1], [1,0,0,0,0,1,0,0,0,0,1])
      }    
  blurred = clip.std.Convolution(matrix=MAP[rad][offset], mode=hvmode, planes=[0]) 
  diff = core.std.MakeDiff(clip, blurred)

  thr_s=str(thr)
  partial_expr = lambda M, N: f" x x[{M},{N}] - x x[{M},{N}] - x x[{M},{N}] - abs 1 + * x x[{M},{N}] - abs 1 + {thr_s} 1 >= {thr_s} 0.5 pow {thr_s} ? + / - 128 + "

  matrix_length = len(MAP[rad][offset])
  start = offset*2 + 1
  pattern = [(0,0), (0,1), (0,-1), (0,2), (0,-2), (0,3), (0,-3), (0,4), (0,-4), (0,5), (0,-5)]
  pattern = pattern[0:matrix_length]
  pattern = [(0,0)] + pattern[start:]
  expr = ''
  for pair in pattern:
      if hvmode == 'v':
          pair = tuple(reversed(pair))
      expr += partial_expr(*pair)
  expr = expr + f'sort{len(pattern)} ' + 'drop '*int(len(pattern)/2) + 'swap ' + 'drop '*int(len(pattern)/2)
  
  medianDiff = core.akarin.Expr(diff, [expr, ''])
  reconstructedMedian = core.std.MakeDiff(diff, medianDiff)   
  blurred = core.std.MergeDiff(blurred, reconstructedMedian)
  
  if vertical: 
    blurred = core.std.Transpose(blurred)
  
  return blurred

###
# requirements:
# RemoveGrain 
# MVTools2
#
# author: VS_Fan, see: https://forum.doom9.org/showthread.php?p=1769570#post1769570
##
def StabilizeIT(clip: vs.VideoNode, div: float=2.0, initZoom: float=1.0, zoomMax: float=1.0, rotMax: float=10.0, pixelAspect: float=1.0, thSCD1: int=800, thSCD2: int=150, stabMethod: int=1, cutOff: float=0.33, anaError: float=30.0, rgMode: int=4):
  zsmooth = hasattr(core,'zsmooth'):
  pf = core.rgvs.RemoveGrain(clip=clip, mode=rgMode) if zsmooth else core.rgvs.RemoveGrain(clip=clip, mode=rgMode)
  pf = core.resize.Bilinear(clip=pf, width=int(pf.width/div), height=int(pf.height/div))
  pf = core.rgvs.RemoveGrain(clip=clip, mode=rgMode) if zsmooth else core.rgvs.RemoveGrain(clip=clip, mode=rgMode)
  pf = core.resize.Bilinear(clip=pf, width=pf.width*div, height=pf.height*div) 
  super = core.mv.Super(clip=pf) 
  vectors = core.mv.Analyse(super=super, isb=False)
  globalmotion = core.mv.DepanAnalyse(clip=pf, vectors=vectors, pixaspect=pixelAspect, error=anaError, thscd1=thSCD1, thscd2=thSCD2)
  clip = core.mv.DepanStabilise(clip=clip, data=globalmotion, cutoff=cutOff, initzoom=initZoom, zoommax=zoomMax, rotmax=rotMax, pixaspect=pixelAspect, method=stabMethod)
  return clip
  
##
# supports: GrayS, RGBS and YUV444PS
# requires libmvtools_sf_em64t (https://github.com/IFeelBloated/vapoursynth-mvtools-sf)
# mvmulti.py (https://github.com/Selur/VapoursynthScriptsInHybrid/blob/master/mvmulti.py)
# author: takla, Avisynth see: https://forum.doom9.org/showthread.php?t=183192
##
def EZdenoise(clip: vs.VideoNode, thSAD: int=150, thSADC: int=-1, tr: int=3, blksize: int=8, overlap: int=4, pel: int=1, chroma: bool=False, falloff: float=0.9):
  import mvmulti
  
  if thSADC == -1:
    thSADC = thSAD
  Super = core.mvsf.Super(clip=clip, pel=pel, chroma=chroma)
  Multi_Vector = mvmulti.Analyze(super=Super, tr=tr, blksize=blksize, overlap=overlap, chroma=chroma)

  return mvmulti.DegrainN(clip=clip, super=Super, mvmulti=Multi_Vector, tr=tr, thsad=thSAD, thscd1=thSADC, thscd2=int(thSADC*falloff))

# Vapoursynth port by Selur from https://forum.doom9.org/showthread.php?p=1812060#post1812060
#
# Small Deflicker 2.4
# Based in idea by Didée, kurish (http://forum.doom9.org/showthread.php?p=1601335#post1601335)
# Adapted by GMJCZP
# Requirements:
#  Masktools: https://github.com/dubhater/vapoursynth-mvtools
#  RemoveGrain: https://github.com/vapoursynth/vs-removegrain
#  Cnr2: https://github.com/dubhater/vapoursynth-cnr2
#  MSmooth: https://github.com/dubhater/vapoursynth-msmoosh
#  TemporalSoften2: https://github.com/dubhater/vapoursynth-temporalsoften2
#  SceneChange: https://github.com/vapoursynth/vs-miscfilters-obsolete

# USAGE:
# Small_Deflicker(clip, width_clip=width(clip)/4, height_clip=height(clip)/4, preset=2, cnr=False, rep=Dalse) (default values)

# PRESETS:
# preset = 1 Soft deflickering (GMJCZP values)
# preset = 2 Medium deflickering (kurish values, preset by default)
# preset = 3 Hard deflickering (Didée values)

# REPAIR MODE:
# - Chroma noise reduction it is an experimental attempt to mitigate the side effects of the script
# By default it is disabled (only for presets 2 and 3)
# - Repair is an option for certain sources or anime/cartoon content, where ghosting may be evident
# By default it is disabled (maybe for preset = 1 it is not necessary to activate it)
def Small_Deflicker(clip: vs.VideoNode, width: int=0, height: int=0, preset: int=2, cnr: bool=False,rep: bool=True):
  clip = core.misc.SCDetect(clip)
  if width == 0:
    width = toMod(clip.width/4,16)
  if height == 0:
    height = toMod(clip.height/4,16)

  if width%16 != 0:
    raise vs.Error('width: need to be mod 16')
  if height%16 != 0:
    raise vs.Error('height: need to be mod 16')
  if (preset < 0) or (preset > 3):
    raise vs.Error('preset not valid (1, 2 or 3)')

  small = core.resize.Bicubic(clip, width,height) # can be altered, but ~25% of original resolution seems reasonable
  
  if preset == 1:
    smallModified = deflickerPreset1(small)
  elif preset == 2:
    smallModified = deflickerPreset2(small, cnr)
  else :
    smallModified = deflickerPreset3(small, cnr)
   
  clip2 = core.std.MakeDiff(small,smallModified,planes=[0, 1, 2])
  clip2 = core.resize.Bicubic(clip2, clip.width,clip.height)
  clip2 = core.std.MakeDiff(clip, clip2, planes=[0, 1, 2])
  if rep:
    return core.rgvs.Repair(clip2, clip, mode=[10, 10, 10])
  return clip2

# Helper
def toMod(value: int, factor: int=16):
  adjust = value - (value % factor)
  return adjust

# Deflicker Presets
def deflickerPreset1(sm: vs.VideoNode):
  smm = core.focus2.TemporalSoften2(clip=sm,radius=1,luma_threshold=6,chroma_threshold=9,scenechange=10,mode=2)
  smm = core.msmoosh.MSmooth(clip=smm,threshold=0.8,strength=25.0,planes=[1,2])
  smm = core.std.Merge(smm, sm, 0.25)
  smm = core.std.Merge(smm, sm, 0.25)
  
  smm = core.focus2.TemporalSoften2(clip=sm,radius=2,luma_threshold=3,chroma_threshold=5,scenechange=6,mode=2)
  smm = core.msmoosh.MSmooth(clip=smm,threshold=2.0,strength=1.0,planes=[1,2])
  return smm

def deflickerPreset2(sm: vs.VideoNode, chroma: bool):
  smm = core.focus2.TemporalSoften2(clip=sm,radius=1,luma_threshold=12,chroma_threshold=255,scenechange=24,mode=2)
  smm = core.msmoosh.MSmooth(clip=smm,threshold=0.8,strength=25.0,planes=[1,2])
  smm = core.std.Merge(smm, sm, 0.25)
  smm = core.std.Merge(smm, sm, 0.25)
  
  smm = core.focus2.TemporalSoften2(clip=sm,radius=2,luma_threshold=7,chroma_threshold=255,scenechange=20,mode=2)
  smm = core.msmoosh.MSmooth(clip=smm,threshold=2.0,strength=1.0,planes=[1,2])

  if chroma:
    return core.cnr2.Cnr2(smm, mode="ooo", ln=5, un=40, vn=40, scdthr=2.0)
  return smm

def deflickerPreset3(sm: vs.VideoNode, chroma: bool):
  smm = core.focus2.TemporalSoften2(clip=sm,radius=1,luma_threshold=32,chroma_threshold=255,scenechange=24,mode=2)
  smm = core.msmoosh.MSmooth(clip=smm,threshold=0.8,strength=25.0,planes=[1,2])
  smm = core.std.Merge(smm, sm, 0.25)
  smm = core.std.Merge(smm, sm, 0.25)
  
  smm = core.focus2.TemporalSoften2(clip=sm,radius=2,luma_threshold=12,chroma_threshold=255,scenechange=20,mode=2)
  smm = core.msmoosh.MSmooth(clip=smm,threshold=2.0,strength=1.0,planes=[1,2])

  if chroma:
    return core.cnr2.Cnr2(smm, mode="ooo", ln=10, lm=255, un=35, vn=35, scdthr=2.0)
  return smm


## Change Temperature by _Al_ https://forum.doom9.org/showthread.php?p=1993851#post1993851
def change_temperature(clip: vs.VideoNode, temp: int=6500):

    if clip.format.color_family is not vs.RGB or clip.format.sample_type is not vs.FLOAT:
       raise vs.Error('change_temperature: clip must be RGBS')

    rgb = get_rgb(temp)
    r, g, b = [value/255.0 for value in rgb]
    return core.std.Expr([clip], expr=[f"x {r} *", f"x {g} *", f"x {b} *"])
    
def get_rgb(temp: int=6500):
    temp = temp / 100
    if temp <= 66:
        r = 255
    else:
        r = temp - 60
        r = 329.698727466 * math.pow(r, -0.1332047592)
        r = min(max(0, r), 255)

    if temp <= 66:
        g = temp
        g = 99.4708025861 * math.log(g) - 161.1195681661
    else:
        g = temp - 60
        g = 288.1221695283 * math.pow(g, -0.0755148492)
    g = min(max(0, g), 255)

    if temp >= 66:
        b = 255
    else:
        if temp <= 19:
            b = 0
        else:
            b = temp - 10
            b = 138.5177312231 * math.log(b) - 305.0447927307
            b = min(max(0, b), 255)

    return round(r), round(g), round(b)


# ChannelMixer port from _AI_
# https://forum.doom9.org/showthread.php?p=1962889#post1962889
# ChannelMixer (clip, float "RR", float "RG", float "RB", float "GR", float "GG", float "GB", float "BR", float "BG", float "BB")
def channel_mixer(rgb, RR=100.0, RG=0.0,   RB=0.0,
                       GR=0.0,   GG=100.0, GB=0.0,
                       BR=0.0,   BG=0.0,   BB=100.0):
    if not rgb.format.color_family == vs.RGB:
        raise ValueError('channel_mixer: input clip must be RGB color_family')
    return core.std.Expr(rgb, expr = [f'0.01 {RR} * x * 0.01 {RG} * x * + 0.01 {RB} * x * +',
                                      f'0.01 {GR} * x * 0.01 {GG} * x * + 0.01 {GB} * x * +',
                                      f'0.01 {BR} * x * 0.01 {BG} * x * + 0.01 {BB} * x * +'])  

# Port from AVILs Avisynth version: https://forum.doom9.org/showthread.php?t=185261
# Required masktools2, mvtools2
# Should work on YUVXXXP8 

def blur(clip: vs.VideoNode, blur_radius: float=0.5) -> vs.VideoNode:
  # Define the blur radius
  kernel_size = 3  # Use 3 for kernel size 9, or 5 for kernel size 25
  sigma = blur_radius / 3.0
  blur_kernel = [
      round((1 / (2 * 3.14159 * sigma**2)) * 2.71828**(-((x - kernel_size//2)**2 + (y - kernel_size//2)**2) / (2 * sigma**2)), 4)
      for y in range(kernel_size)
      for x in range(kernel_size)
  ]
  sum_kernel = sum(blur_kernel)
  blur_kernel = [val / sum_kernel for val in blur_kernel]
  return vs.core.std.Convolution(clip=clip, matrix=blur_kernel)

def VHSClean(clip: vs.VideoNode, ths: int=100, blur_sharp=True) -> vs.VideoNode:
 
  lambda_      = 40000
  pel         =  2
  tm          = False
  srhp        = 2
  srch        = 4
  badsad      = 2000
  chroma =True
  thsc = ths * 2
  ths1=  ths * 4
  thsc1= ths1* 2
  bs = 8
  bblur  = 0.6
  
  
  sx = core.mv.Super(clip=clip, pel=pel, sharp=1)  
  
  #phase 1. Soft denoising
  f1x = core.mv.Analyse(super_=sx,delta=1,isb=False,truemotion=tm,blksize=16,blksizev=8,overlap=8,overlapv=4,search=srch,searchparam=srhp,badsad=badsad,dct=1,chroma=chroma,lambda_=lambda_)
  b1x = core.mv.Analyse(super_=sx,delta=1,isb=True,truemotion=tm,blksize=16,blksizev=8,overlap=8,overlapv=4,search=srch,searchparam=srhp,badsad=badsad,dct=1,chroma=chroma,lambda_=lambda_)
  f2x = core.mv.Analyse(super_=sx,delta=2,isb=False,truemotion=tm,blksize=16,blksizev=8,overlap=8,overlapv=4,search=srch,searchparam=srhp,badsad=badsad,dct=1,chroma=chroma,lambda_=lambda_)
  b2x = core.mv.Analyse(super_=sx,delta=2,isb=True,truemotion=tm,blksize=16,blksizev=8,overlap=8,overlapv=4,search=srch,searchparam=srhp,badsad=badsad,dct=1,chroma=chroma,lambda_=lambda_)
  x2 = core.mv.Degrain2(clip,sx,b1x,f1x,b2x,f2x,thsad=ths,thsadc=thsc)

  #phase 2. Reinject denoised over original (like a sharpening using blurred version)
  x3=core.std.Expr([clip,x2],expr="x 2 * y -")

  #phase 3. Strong denoising. Same style as MCDegrainSharp (By Didée and Stainless)
  if (blur_sharp):
    sharpen_kernel = [-0.1, -0.1, -0.1, -0.1, 2.0, -0.1, -0.1, -0.1, -0.1] # csharp = 0.6
    x0 = core.std.Convolution(clip=x3, matrix=sharpen_kernel)
    x1 = blur(clip=x3, blur_radius=bblur)
  else:
    x0 = x3
    x1 = x3

  sx0 = core.mv.Super(clip=x0,pel=pel,sharp=1,levels=1)   # Only 1 Level required for sharpened Super (not MAnalyse-ing)
  sx1 = core.mv.Super(clip=x1,pel=pel,sharp=1)

  f1x1 = core.mv.Analyse(super_=sx1,delta=1,isb=False,truemotion=tm,blksize=64,blksizev=64,overlap=32,overlapv=32,search=srch,searchparam=srhp,badsad=badsad,dct=0,chroma=chroma,lambda_=lambda_)
  b1x1 = core.mv.Analyse(super_=sx1,delta=1,isb=True,truemotion=tm,blksize=64,blksizev=64,overlap=32,overlapv=32,search=srch,searchparam=srhp,badsad=badsad,dct=0,chroma=chroma,lambda_=lambda_)
  f2x1 = core.mv.Analyse(super_=sx1,delta=2,isb=False,truemotion=tm,blksize=64,blksizev=64,overlap=32,overlapv=32,search=srch,searchparam=srhp,badsad=badsad,dct=0,chroma=chroma,lambda_=lambda_)
  b2x1 = core.mv.Analyse(super_=sx1,delta=2,isb=True,truemotion=tm,blksize=64,blksizev=64,overlap=32,overlapv=32,search=srch,searchparam=srhp,badsad=badsad,dct=0,chroma=chroma,lambda_=lambda_)
  f3x1 = core.mv.Analyse(super_=sx1,delta=3,isb=False,truemotion=tm,blksize=64,blksizev=64,overlap=32,overlapv=32,search=srch,searchparam=srhp,badsad=badsad,dct=0,chroma=chroma,lambda_=lambda_)
  b3x1 = core.mv.Analyse(super_=sx1,delta=3,isb=True,truemotion=tm,blksize=64,blksizev=64,overlap=32,overlapv=32,search=srch,searchparam=srhp,badsad=badsad,dct=0,chroma=chroma,lambda_=lambda_)
  f4x1 = core.mv.Analyse(super_=sx1,delta=4,isb=False,truemotion=tm,blksize=64,blksizev=64,overlap=32,overlapv=32,search=srch,searchparam=srhp,badsad=badsad,dct=0,chroma=chroma,lambda_=lambda_)
  b4x1 = core.mv.Analyse(super_=sx1,delta=4,isb=True,truemotion=tm,blksize=64,blksizev=64,overlap=32,overlapv=32,search=srch,searchparam=srhp,badsad=badsad,dct=0,chroma=chroma,lambda_=lambda_)
  f5x1 = core.mv.Analyse(super_=sx1,delta=5,isb=False,truemotion=tm,blksize=64,blksizev=64,overlap=32,overlapv=32,search=srch,searchparam=srhp,badsad=badsad,dct=0,chroma=chroma,lambda_=lambda_)
  b5x1 = core.mv.Analyse(super_=sx1,delta=5,isb=True,truemotion=tm,blksize=64,blksizev=64,overlap=32,overlapv=32,search=srch,searchparam=srhp,badsad=badsad,dct=0,chroma=chroma,lambda_=lambda_)
  f6x1 = core.mv.Analyse(super_=sx1,delta=6,isb=False,truemotion=tm,blksize=64,blksizev=64,overlap=32,overlapv=32,search=srch,searchparam=srhp,badsad=badsad,dct=0,chroma=chroma,lambda_=lambda_)
  b6x1 = core.mv.Analyse(super_=sx1,delta=6,isb=True,truemotion=tm,blksize=64,blksizev=64,overlap=32,overlapv=32,search=srch,searchparam=srhp,badsad=badsad,dct=0,chroma=chroma,lambda_=lambda_)

  mv123 = core.mv.Degrain3(x1, sx0, b1x1,f1x1,b2x1,f2x1,b3x1,f3x1,thsad=ths1,thsadc=thsc1)
  mv456 = core.mv.Degrain3(x1, sx0, b4x1,f4x1,b5x1,f5x1,b6x1,f6x1,thsad=ths1,thsadc=thsc1)
  x4 = core.std.Merge(mv123, mv456, weight=[0.3])
 

  #phase 4. Recover quick flying objects and water drops
  mx=core.std.Expr([blur(clip=x4, blur_radius=1.5),blur(clip=x3, blur_radius=1.5)],expr="y x - abs 12 >  255 0 ?")
  if hasattr(vs.core, 'vszip'):
    return core.std.MaskedMerge(clipa=x4,clipb=x3,mask=core.vszip.BoxBlur(mx,2),planes=[0, 1, 2])
  else:
    return core.std.MaskedMerge(clipa=x4,clipb=x3,mask=core.std.BoxBlur(mx,2),planes=[0, 1, 2])
    
    
# masked CAS port of MCAS by Atak_Snajpera https://forum.doom9.org/showthread.php?p=2003218#post2003218
# requires:
#  CAS: https://github.com/HomeOfVapourSynthEvolution/VapourSynth-CAS
#  Overlay from hasfunc: https://github.com/Selur/VapoursynthScriptsInHybrid/blob/master/havsfunc.py
#
# clip: Clip to process. Any planar format with either integer sample type of 8-16 bit depth or float sample type of 32 bit depth is supported.
# sharpness: Sharpening strength.
def maskedCAS(clip: vs.VideoNode, strength: float=0.2):
  if (strength < 0) or (strength > 1):
    raise vs.Error('strength not valid (range: [0-1]')
  import havsfunc
  iMask = core.std.Levels(clip=clip, min_in=0, gamma=2, max_in=2 << clip.format.bits_per_sample -1)
  if hasattr(vs.core, 'vszip'):
    eMask = core.std.Sobel(clip=iMask, planes=[0]).std.InvertMask().std.Levels(min_in=0, gamma=2, max_in=2 << clip.format.bits_per_sample -1).vszip.BoxBlur()  
  else:
    eMask = core.std.Sobel(clip=iMask, planes=[0]).std.InvertMask().std.Levels(min_in=0, gamma=2, max_in=2 << clip.format.bits_per_sample -1).std.BoxBlur()
  sharp = core.cas.CAS(clip=clip, sharpness=1)
  return havsfunc.Overlay(base=clip, overlay=sharp, mask=eMask, opacity=strength)
  
  
# Vapoursynth port of ContrastMask from javlak
# https://forum.doom9.org/showthread.php?p=1514814#post1514814
# requires masktools and TCanny
# the default 'enhance' seems to be too high for normal usage, best start with 1 and increase it slowly
def ContrastMask(clip, gblur=20.0, enhance=10.0):
    enhance = max(0.0, min(enhance, 10.0)) * 0.1

    # Convert to grayscale and invert
    v2 = core.std.ShufflePlanes(clip, planes=0, colorfamily=vs.GRAY)
    v2 = core.std.Invert(v2)

    # Apply Gaussian blur
    v2 = core.tcanny.TCanny(v2, sigma=50, sigma_v=50+gblur, mode=-1)

    # Get the bit depth and scaling factors
    bit_depth = clip.format.bits_per_sample
    color_range = clip.get_frame(0).props.get('_ColorRange', vs.RANGE_FULL)

    if color_range == vs.RANGE_LIMITED:
        max_val = 235 << (bit_depth - 8)
    else:  # full range
        max_val = (1 << bit_depth) - 1
    
    half_max_val = max_val / 2.0

    # Apply the contrast mask effect using Expr
    expr = f"x {half_max_val} > y {max_val} x - {half_max_val} / * x {max_val} x - - + y x {half_max_val} / * ?"
    photoshop_overlay = core.std.Expr([clip.std.ShufflePlanes(planes=0, colorfamily=vs.GRAY), v2], [expr])

    # Merge the original and overlay clips
    photoshop_overlay = core.std.ShufflePlanes([photoshop_overlay, clip], planes=[0, 1, 2], colorfamily=vs.YUV)
    merged = core.std.Merge(clip, photoshop_overlay, weight=enhance)

    return merged


def HaloBuster(input: vs.VideoNode, a: int = 32, h: float = 6.4, thr: float = 1.0, elast: float = 1.5) -> vs.VideoNode:
    # Convert to grayscale format if not already
    gray_format = vs.GRAY16 if input.format.bits_per_sample > 8 else vs.GRAY8
    gray = input.resize.Point(format=gray_format)
    
    # Add borders for padding
    gray = core.std.AddBorders(gray, left=a, top=a, right=a, bottom=a)

    # Apply KNLMeansCL denoising
    clean = core.knlm.KNLMeansCL(gray, d=0, a=a, s=0, h=h)
    
    # Apply TCanny for edge detection
    mask = core.tcanny.TCanny(clean, sigma=1.5, mode=1)
    max_pixel_value = (1 << gray.format.bits_per_sample) - 1
    mask = core.std.Lut(mask, function=lambda x: int(min(max((x / max_pixel_value - 0.24) * 3.2, 0.0), 1.0) * max_pixel_value))
    mask = core.std.Maximum(mask)
    mask = core.std.Inflate(mask)

    # Merge the clean image back using the mask
    merge = core.std.MaskedMerge(gray, clean, mask)
    
    # Limit the merge differences
    if hasattr(core,'zsmooth'):
      limit = core.zsmooth.RemoveGrain(merge, mode=[20])
    else:
      limit = core.rgvs.RemoveGrain(merge, mode=[20])
    
    # Crop the added borders
    crop = core.std.CropRel(limit, left=a, top=a, right=a, bottom=a)

    # Combine the final planes to get the final output
    final = core.std.ShufflePlanes([crop, input], planes=[0, 1, 2], colorfamily=vs.YUV)

    return final


# Usage example:
# input = core.ffms2.Source("input_file")
# result = HaloBuster(input, a=16, h=3.5, thr=1.2, elast=1.8)
# result.set_output()
