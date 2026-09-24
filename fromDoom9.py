from vapoursynth import core
import vapoursynth as vs
import math

try:
    from gaussblur import GaussBlur
except ImportError:
    GaussBlur = None

from misc import get_mv, SCDetect
from color import LimitFilter
from helpers import NLMeans, get_expr, pick_tool, tool_function

def _boxblur_fn(tools=None):
    """Pick the BoxBlur: tools['boxblur'], else vszip, else std."""
    return tool_function(tools, 'boxblur', 'BoxBlur')

def _nlmeans_gray(clip: vs.VideoNode, d: int, a: int, s: int, h: float, tools=None) -> vs.VideoNode:
    """NLMeans on a GRAY clip, using whichever NLMeans plugin is available (tools['nlmeans'] picks it)."""
    return NLMeans(clip, d=d, a=a, s=s, h=h, tools=tools)

# DeStripe works on YUVXXXPY
# "low frequency" stripes/bands removal filter
# optional: https://github.com/AkarinVS/vapoursynth-plugin/releases
#
# int rad: search radius (default: 1, range: 1-5)
# int thr: blur threshold, wil be scaled by bit depth (default: 256, range: 1-256)
# boolean vertical: transposes the source for the filtering, to handle vertical lines instead of horizontal ones. (default: False)
# str hvmode: whether to use vertival or hoizontal convolution
def DeStripe(clip: vs.VideoNode, rad: int=2, offset: int=0, thr: int=256, vertical=False, hvmode: str='v', tools=None) -> vs.VideoNode:
    if (rad < 1) or (rad > 5):
        raise vs.Error('rad not valid (range: 1-5)')
    if (offset < 0) or (offset > (rad-1)):
        raise vs.Error('offset not valid (range: 0-(rad-1))')
    if (hvmode != 'v' and hvmode != 'h'):
        raise vs.Error("hvmode must be either 'h' or 'v'")

    # Scale threshold and neutral level to current bit depth
    shift = clip.format.bits_per_sample - 8
    thr = thr << shift
    mid = 128 << shift

    if vertical:
        clip = core.std.Transpose(clip)

    MAP = {
        1: ([1,1,1],),
        2: ([1,1,1,1,1], [1,0,1,0,1]),
        3: ([1,1,1,1,1,1,1], [1,1,0,1,0,1,1], [1,0,0,1,0,0,1]),
        4: ([1,1,1,1,1,1,1,1,1], [1,1,1,0,1,0,1,1,1], [1,1,0,0,1,0,0,1,1], [1,0,0,0,1,0,0,0,1]),
        5: ([1,1,1,1,1,1,1,1,1,1,1], [1,1,1,1,0,1,0,1,1,1,1], [1,1,1,0,0,1,0,0,1,1,1], [1,1,0,0,0,1,0,0,0,1,1], [1,0,0,0,0,1,0,0,0,0,1])
    }

    blurred = clip.std.Convolution(matrix=MAP[rad][offset], mode=hvmode, planes=[0])
    diff = core.std.MakeDiff(clip, blurred)

    thr_s = str(thr)
    mid_s = str(mid)

    partial_expr = lambda M, N: (
        f" x x[{M},{N}] - x x[{M},{N}] - x x[{M},{N}] - abs 1 + * "
        f"x x[{M},{N}] - abs 1 + {thr_s} 1 >= {thr_s} 0.5 pow {thr_s} ? + / - "
        f"{mid_s} + "   # <--- scaled neutral
    )

    matrix_length = len(MAP[rad][offset])
    start = offset * 2 + 1
    pattern = [(0,0), (0,1), (0,-1), (0,2), (0,-2), (0,3), (0,-3), (0,4), (0,-4), (0,5), (0,-5)]
    pattern = pattern[0:matrix_length]
    pattern = [(0,0)] + pattern[start:]

    expr = ''
    for pair in pattern:
        if hvmode == 'v':
            pair = tuple(reversed(pair))
        expr += partial_expr(*pair)

    expr += f'sort{len(pattern)} ' + 'drop ' * (len(pattern)//2) + 'swap ' + 'drop ' * (len(pattern)//2)

    EXPR = get_expr(tools)

    medianDiff = EXPR(diff, [expr, ''])
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
def StabilizeIT(clip: vs.VideoNode, div: float=2.0, initZoom: float=1.0, zoomMax: float=1.0, rotMax: float=10.0, pixelAspect: float=1.0, thSCD1: int=800, thSCD2: int=150, stabMethod: int=1, cutOff: float=0.33, anaError: float=30.0, rgMode: int=4, tools=None):
  RG = tool_function(tools, 'rg', 'RemoveGrain')
  MV = get_mv(tools)
  pf = RG(clip, mode=rgMode)
  pf = core.resize.Bilinear(clip=pf, width=int(pf.width/div), height=int(pf.height/div))
  pf = RG(pf, mode=rgMode)
  pf = core.resize.Bilinear(clip=pf, width=pf.width*div, height=pf.height*div) 
  _super = MV.Super(clip=pf, blksize=8, overlap=0) 
  vectors = MV.Analyse(super=_super, isb=False)
  globalmotion = MV.DepanAnalyse(clip=pf, vectors=vectors, pixaspect=pixelAspect, error=anaError, thscd1=thSCD1, thscd2=thSCD2)
  clip = MV.DepanStabilise(clip=clip, data=globalmotion, cutoff=cutOff, initzoom=initZoom, zoommax=zoomMax, rotmax=rotMax, pixaspect=pixelAspect, method=stabMethod)
  return clip

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
def Small_Deflicker(clip: vs.VideoNode, width: int=0, height: int=0, preset: int=2, cnr: bool=False,rep: bool=True, tools=None):
  
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
  zsmooth = pick_tool(tools, 'temporalsoften', ('zsmooth', 'focus2'), lambda name: name == 'focus2' or hasattr(core, 'zsmooth')) == 'zsmooth'
  if preset == 1:
    smallModified = deflickerPreset1(small, zsmooth=zsmooth, tools=tools)
  elif preset == 2:
    smallModified = deflickerPreset2(small, cnr, zsmooth=zsmooth, tools=tools)
  else :
    smallModified = deflickerPreset3(small, cnr, zsmooth=zsmooth, tools=tools)
   
  clip2 = core.std.MakeDiff(small,smallModified,planes=[0, 1, 2])
  clip2 = core.resize.Bicubic(clip2, clip.width,clip.height)
  clip2 = core.std.MakeDiff(clip, clip2, planes=[0, 1, 2])
  if rep:
    return tool_function(tools, 'rg', 'Repair')(clip2, clip, mode=[10, 10, 10])
  return clip2

# Helper
def toMod(value: int, factor: int=16):
  adjust = value - (value % factor)
  return adjust

# Deflicker Presets
def deflickerPreset1(sm: vs.VideoNode, zsmooth: bool=False, tools=None):
  if pick_tool(tools, 'msmooth', ('msmoosh', 'std')) == 'msmoosh':
    SMOOTH = core.msmoosh.MSmooth
  else:
    import msmoosh
    from functools import partial
    SMOOTH = partial(msmoosh.MSmooth, tools=tools) 
  
  if zsmooth:
    sm = SCDetect(sm, threshold=0.1, tools=tools)
    smm = core.zsmooth.TemporalSoften(clip=sm,radius=1, threshold=[6,9,9],scenechange=-1,scalep=True)
  else:
    smm = core.focus2.TemporalSoften2(clip=sm,radius=1,luma_threshold=6,chroma_threshold=9,scenechange=10,mode=2)
  smm = SMOOTH(clip=smm,threshold=0.8,strength=25.0,planes=[1,2])
  smm = core.std.Merge(smm, sm, 0.25)
  smm = core.std.Merge(smm, sm, 0.25)
  if zsmooth:
    sm = SCDetect(sm, threshold=0.06, tools=tools)
    smm = core.zsmooth.TemporalSoften(clip=sm,radius=2, threshold=[3,5,5],scenechange=-1,scalep=True)
  else:
    smm = core.focus2.TemporalSoften2(clip=sm,radius=2,luma_threshold=3,chroma_threshold=5,scenechange=6,mode=2)
  smm = SMOOTH(clip=smm,threshold=2.0,strength=1.0,planes=[1,2])
  return smm

def deflickerPreset2(sm: vs.VideoNode, chroma: bool, zsmooth: bool=False, tools=None):
  if pick_tool(tools, 'msmooth', ('msmoosh', 'std')) == 'msmoosh':
    SMOOTH = core.msmoosh.MSmooth
  else:
    import msmoosh
    from functools import partial
    SMOOTH = partial(msmoosh.MSmooth, tools=tools) 
    
  if zsmooth:
    sm = SCDetect(sm, threshold=0.24, tools=tools)
    smm = core.zsmooth.TemporalSoften(clip=sm,radius=1, threshold=[12,255,255],scenechange=-1,scalep=True)
  else:
    smm = core.focus2.TemporalSoften2(clip=sm,radius=1,luma_threshold=12,chroma_threshold=255,scenechange=24,mode=2)
  smm = SMOOTH(clip=smm,threshold=0.8,strength=25.0,planes=[1,2])
  smm = core.std.Merge(smm, sm, 0.25)
  smm = core.std.Merge(smm, sm, 0.25)
  if zsmooth:
    sm = SCDetect(sm, threshold=0.2, tools=tools)
    smm = core.zsmooth.TemporalSoften(clip=sm,radius=2, threshold=[7,255,255],scenechange=-1,scalep=True)
  else:
    smm = core.focus2.TemporalSoften2(clip=sm,radius=2,luma_threshold=7,chroma_threshold=255,scenechange=20,mode=2)
  smm = SMOOTH(clip=smm,threshold=2.0,strength=1.0,planes=[1,2])

  if chroma:
    if pick_tool(tools, 'cnr', ('zsmooth', 'cnr2'), lambda name: name == 'cnr2' or hasattr(core, 'zsmooth')) == 'zsmooth':
      import misc
      smm = misc.SCDetect(smm,threshold=0.02, tools=tools)
      return core.zsmooth.Cnr4(smm, mode="ooo", tmode=0, radius=2, sense=[5, 40, 40])
    return core.cnr2.Cnr2(smm, mode="ooo", ln=5, un=40, vn=40, scdthr=2.0)
  return smm

def deflickerPreset3(sm: vs.VideoNode, chroma: bool, zsmooth: bool=False, tools=None):
  if pick_tool(tools, 'msmooth', ('msmoosh', 'std')) == 'msmoosh':
    SMOOTH = core.msmoosh.MSmooth
  else:
    import msmoosh
    from functools import partial
    SMOOTH = partial(msmoosh.MSmooth, tools=tools) 
    
  if zsmooth:
    sm = SCDetect(sm, threshold=0.24, tools=tools)
    smm = core.zsmooth.TemporalSoften(clip=sm,radius=1, threshold=[32,255,255],scenechange=-1,scalep=True)
  else:
    smm = core.focus2.TemporalSoften2(clip=sm,radius=1,luma_threshold=32,chroma_threshold=255,scenechange=24,mode=2)
  smm = SMOOTH(clip=smm,threshold=0.8,strength=25.0,planes=[1,2])
  smm = core.std.Merge(smm, sm, 0.25)
  smm = core.std.Merge(smm, sm, 0.25)
  if zsmooth:
    sm = SCDetect(sm, threshold=0.2, tools=tools)
    smm = core.zsmooth.TemporalSoften(clip=sm,radius=1, threshold=[12,255,255],scenechange=-1,scalep=True)
  else:
    smm = core.focus2.TemporalSoften2(clip=sm,radius=2,luma_threshold=12,chroma_threshold=255,scenechange=20,mode=2)
  smm = SMOOTH(clip=smm,threshold=2.0,strength=1.0,planes=[1,2])

  if chroma:
    if pick_tool(tools, 'cnr', ('zsmooth', 'cnr2'), lambda name: name == 'cnr2' or hasattr(core, 'zsmooth')) == 'zsmooth':
      import misc
      smm = misc.SCDetect(smm,threshold=0.02, tools=tools)
      return core.zsmooth.Cnr4(smm, mode="ooo", tmode=0, radius=2, sense=[10, 35, 35], str=[255,255,255])
    return core.cnr2.Cnr2(smm, mode="ooo", ln=10, lm=255, un=35, vn=35, scdthr=2.0)
  return smm


## Change Temperature by _Al_ https://forum.doom9.org/showthread.php?p=1993851#post1993851
def change_temperature(clip: vs.VideoNode, temp: int=6500, tools=None):

    if clip.format.color_family is not vs.RGB or clip.format.sample_type is not vs.FLOAT:
       raise vs.Error('change_temperature: clip must be RGBS')

    rgb = get_rgb(temp)
    r, g, b = [value/255.0 for value in rgb]
    EXPR = get_expr(tools)
    return EXPR([clip], expr=[f"x {r} *", f"x {g} *", f"x {b} *"])
    
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
                       BR=0.0,   BG=0.0,   BB=100.0, tools=None):
    if not rgb.format.color_family == vs.RGB:
        raise ValueError('channel_mixer: input clip must be RGB color_family')
    EXPR = get_expr(tools)
    return EXPR(rgb, expr = [f'0.01 {RR} * x * 0.01 {RG} * x * + 0.01 {RB} * x * +',
                                      f'0.01 {GR} * x * 0.01 {GG} * x * + 0.01 {GB} * x * +',
                                      f'0.01 {BR} * x * 0.01 {BG} * x * + 0.01 {BB} * x * +'])  

# Port from AVILs Avisynth version: https://forum.doom9.org/showthread.php?t=185261
# Required masktools2, mvtools2
# Should work on YUVXXXP8 

def _adjust_focus(clip: vs.VideoNode, amount: float) -> vs.VideoNode:
  '''AviSynth's AdjustFocus: kernel [(1-c)/2, c, (1-c)/2] with c = 2^amount, horizontal and vertical.'''
  c = 2.0 ** amount
  kernel = [(1 - c) / 2, c, (1 - c) / 2]
  if clip.format.sample_type == vs.INTEGER:
    # std.Convolution rounds the coefficients to integers for integer clips (at most 1023) and divides by their sum.
    top = max(abs(v) for v in kernel)
    kernel = [round(v / top * 1023) for v in kernel]
  return core.std.Convolution(clip=clip, matrix=kernel, mode='hv')

def blur(clip: vs.VideoNode, amount: float=0.5) -> vs.VideoNode:
  '''AviSynth's Blur(amount), amount in [-1.0, 1.58].'''
  if amount < -1.0 or amount > 1.5849625:
    raise vs.Error('blur: amount must be in the range -1.0 to 1.58')
  return _adjust_focus(clip, -amount)

def sharpen(clip: vs.VideoNode, amount: float=0.5) -> vs.VideoNode:
  '''AviSynth's Sharpen(amount), amount in [-1.58, 1.0].'''
  if amount < -1.5849625 or amount > 1.0:
    raise vs.Error('sharpen: amount must be in the range -1.58 to 1.0')
  return _adjust_focus(clip, amount)

def VHSClean(clip: vs.VideoNode, ths: int=100, blur_sharp=True, tools=None) -> vs.VideoNode:
  MV = get_mv(tools)
 
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
  csharp = 0.6

  
  sx = MV.Super(clip=clip, pel=pel, sharp=1,blksize=16,blksizev=8,overlap=8,overlapv=4)

  #phase 1. Soft denoising
  if MV.use_mvu:
      b1x, f1x, b2x, f2x = MV.AnalyseMany(sx, radius=2, delta=1, truemotion=tm, blksize=16, blksizev=8, overlap=8, overlapv=4, search=srch, searchparam=srhp, badsad=badsad, dct=1, chroma=chroma, lambda_=lambda_)
  else:
      f1x = MV.Analyse(sx,delta=1, isb=False, truemotion=tm, blksize=16, blksizev=8, overlap=8, overlapv=4, search=srch, searchparam=srhp, badsad=badsad, dct=1, chroma=chroma, lambda_=lambda_)
      b1x = MV.Analyse(sx,delta=1, isb=True,  truemotion=tm, blksize=16, blksizev=8, overlap=8, overlapv=4, search=srch, searchparam=srhp, badsad=badsad, dct=1, chroma=chroma, lambda_=lambda_)
      f2x = MV.Analyse(sx,delta=2, isb=False, truemotion=tm, blksize=16, blksizev=8, overlap=8, overlapv=4, search=srch, searchparam=srhp, badsad=badsad, dct=1, chroma=chroma, lambda_=lambda_)
      b2x = MV.Analyse(sx,delta=2, isb=True,  truemotion=tm, blksize=16, blksizev=8, overlap=8, overlapv=4, search=srch, searchparam=srhp, badsad=badsad, dct=1, chroma=chroma, lambda_=lambda_)
  x2 = MV.Degrain2(clip,sx,b1x,f1x,b2x,f2x,thsad=ths,thsadc=thsc)

  #phase 2. Reinject denoised over original (like a sharpening using blurred version)
  EXPR = get_expr(tools)
  x3=EXPR([clip,x2],expr="x 2 * y -")

  #phase 3. Strong denoising. Same style as MCDegrainSharp (By Didée and Stainless)
  if (blur_sharp):
    x0 = sharpen(x3, csharp)
    x1 = blur(x3, bblur)
  else:
    x0 = x3
    x1 = x3

  sx0 = MV.Super(clip=x0,pel=pel,sharp=1,levels=1,blksize=64,overlap=32)   # Only 1 Level required for sharpened Super (not MAnalyse-ing); block geometry must match the vectors analysed from sx1
  sx1 = MV.Super(clip=x1,pel=pel,sharp=1,blksize=64,overlap=32)

  if MV.use_mvu:
      b1x1, f1x1, b2x1, f2x1, b3x1, f3x1, b4x1, f4x1, b5x1, f5x1, b6x1, f6x1 = MV.AnalyseMany(sx1, radius=6, delta=1, truemotion=tm, blksize=64, overlap=32, search=srch, searchparam=srhp, badsad=badsad, dct=0, chroma=chroma, lambda_=lambda_)
  else:
      f1x1 = MV.Analyse(sx1,delta=1, isb=False, truemotion=tm, blksize=64, blksizev=64, overlap=32, overlapv=32, search=srch, searchparam=srhp, badsad=badsad, dct=0, chroma=chroma, lambda_=lambda_)
      b1x1 = MV.Analyse(sx1,delta=1, isb=True,  truemotion=tm, blksize=64, blksizev=64, overlap=32, overlapv=32, search=srch, searchparam=srhp, badsad=badsad, dct=0, chroma=chroma, lambda_=lambda_)
      f2x1 = MV.Analyse(sx1,delta=2, isb=False, truemotion=tm, blksize=64, blksizev=64, overlap=32, overlapv=32, search=srch, searchparam=srhp, badsad=badsad, dct=0, chroma=chroma, lambda_=lambda_)
      b2x1 = MV.Analyse(sx1,delta=2, isb=True,  truemotion=tm, blksize=64, blksizev=64, overlap=32, overlapv=32, search=srch, searchparam=srhp, badsad=badsad, dct=0, chroma=chroma, lambda_=lambda_)
      f3x1 = MV.Analyse(sx1,delta=3, isb=False, truemotion=tm, blksize=64, blksizev=64, overlap=32, overlapv=32, search=srch, searchparam=srhp, badsad=badsad, dct=0, chroma=chroma, lambda_=lambda_)
      b3x1 = MV.Analyse(sx1,delta=3, isb=True,  truemotion=tm, blksize=64, blksizev=64, overlap=32, overlapv=32, search=srch, searchparam=srhp, badsad=badsad, dct=0, chroma=chroma, lambda_=lambda_)
      f4x1 = MV.Analyse(sx1,delta=4, isb=False, truemotion=tm, blksize=64, blksizev=64, overlap=32, overlapv=32, search=srch, searchparam=srhp, badsad=badsad, dct=0, chroma=chroma, lambda_=lambda_)
      b4x1 = MV.Analyse(sx1,delta=4, isb=True,  truemotion=tm, blksize=64, blksizev=64, overlap=32, overlapv=32, search=srch, searchparam=srhp, badsad=badsad, dct=0, chroma=chroma, lambda_=lambda_)
      f5x1 = MV.Analyse(sx1,delta=5, isb=False, truemotion=tm, blksize=64, blksizev=64, overlap=32, overlapv=32, search=srch, searchparam=srhp, badsad=badsad, dct=0, chroma=chroma, lambda_=lambda_)
      b5x1 = MV.Analyse(sx1,delta=5, isb=True,  truemotion=tm, blksize=64, blksizev=64, overlap=32, overlapv=32, search=srch, searchparam=srhp, badsad=badsad, dct=0, chroma=chroma, lambda_=lambda_)
      f6x1 = MV.Analyse(sx1,delta=6, isb=False, truemotion=tm, blksize=64, blksizev=64, overlap=32, overlapv=32, search=srch, searchparam=srhp, badsad=badsad, dct=0, chroma=chroma, lambda_=lambda_)
      b6x1 = MV.Analyse(sx1,delta=6, isb=True,  truemotion=tm, blksize=64, blksizev=64, overlap=32, overlapv=32, search=srch, searchparam=srhp, badsad=badsad, dct=0, chroma=chroma, lambda_=lambda_)

  x4 = MV.Degrain6(x1, sx0, b1x1,f1x1,b2x1,f2x1,b3x1,f3x1,b4x1,f4x1,b5x1,f5x1,b6x1,f6x1,thsad=ths1,thsadc=thsc1,centre_from_clip=blur_sharp)
 

  #phase 4. Recover quick flying objects and water drops
  EXPR = get_expr(tools)
  # threshold 12 and mask maximum 255 are 8-bit values
  if clip.format.sample_type == vs.INTEGER:
    th, peak = 12 << (clip.format.bits_per_sample - 8), (1 << clip.format.bits_per_sample) - 1
  else:
    th, peak = 12 / 255, 1.0
  mx=EXPR([blur(x4, 1.5),blur(x3, 1.5)],expr=f"y x - abs {th} > {peak} 0 ?")
  return core.std.MaskedMerge(clipa=x4,clipb=x3,mask=_boxblur_fn(tools)(mx,hradius=2,vradius=2),planes=[0, 1, 2])
    
    
# masked CAS port of MCAS by Atak_Snajpera https://forum.doom9.org/showthread.php?p=2003218#post2003218
# requires:
#  CAS: https://github.com/HomeOfVapourSynthEvolution/VapourSynth-CAS
#
# clip: Clip to process. Any planar format with either integer sample type of 8-16 bit depth or float sample type of 32 bit depth is supported.
# sharpness: Sharpening strength.
def maskedCAS(clip: vs.VideoNode, strength: float=0.2, tools=None):
  if (strength < 0) or (strength > 1):
    raise vs.Error('strength not valid (range: [0-1]')
  import misc
  iMask = core.std.Levels(clip=clip, min_in=0, gamma=2, max_in=2 << clip.format.bits_per_sample -1)
  eMask = core.std.Sobel(clip=iMask, planes=[0]).std.InvertMask().std.Levels(min_in=0, gamma=2, max_in=2 << clip.format.bits_per_sample -1)
  eMask = _boxblur_fn(tools)(eMask)
  sharp = core.cas.CAS(clip=clip, sharpness=1)
  return misc.Overlay(base=clip, overlay=sharp, mask=eMask, opacity=strength, tools=tools)
  
  
# Vapoursynth port of ContrastMask from javlak
# https://forum.doom9.org/showthread.php?p=1514814#post1514814
# blur: TCanny if loaded, else gaussblur.GaussBlur, else BoxBlur
# the default 'enhance' seems to be too high for normal usage, best start with 1 and increase it slowly
# gblur has no visible effect, as in the original: there it is the chroma variance of the already desaturated copy
def ContrastMask(clip, gblur=20.0, enhance=10.0, tools=None):
    enhance = max(0.0, min(enhance, 10.0)) * 0.1
    EXPR = get_expr(tools)

    # Like the original, work on the coded luma values at full scale, whatever the range.
    luma = core.std.ShufflePlanes(clip, planes=0, colorfamily=vs.GRAY)
    to_normalised = None
    if clip.format.sample_type == vs.FLOAT:
        peak = 1.0
        prop_name = '_Range' if core.core_version.release_major >= 74 else '_ColorRange'
        if clip.get_frame(0).props.get(prop_name, vs.RANGE_FULL) == vs.RANGE_LIMITED:
            # float limited luma is normalised (16 -> 0.0, 235 -> 1.0); map it to the coded scale and back afterwards
            luma = EXPR(luma, f"x {219 / 255} * {16 / 255} +")
            to_normalised = f"x {16 / 255} - {255 / 219} *"
    else:
        peak = (1 << clip.format.bits_per_sample) - 1
    half = peak / 2.0

    # Invert
    v2 = EXPR(luma, f"{peak} x -")

    # Gaussian blur like VariableBlur's GaussianBlur(50.0, ...): luma variance 50
    sigma = math.sqrt(50.0)
    if pick_tool(tools, 'tcanny', ('tcanny', 'std')) == 'tcanny':
      v2 = core.tcanny.TCanny(v2, sigma=sigma, mode=-1)
    elif GaussBlur is not None:
      v2 = GaussBlur(v2, sigma=sigma, tools=tools)
    else:
      # three box passes of radius r have the variance r * (r + 1)
      radius = max(1, round(sigma))
      v2 = _boxblur_fn(tools)(v2, hradius=radius, hpasses=3, vradius=radius, vpasses=3)

    # Photoshop overlay
    expr = f"x {half} > y {peak} x - {half} / * x {peak} x - - + y x {half} / * ?"
    photoshop_overlay = EXPR([luma, v2], [expr])
    if to_normalised:
        photoshop_overlay = EXPR(photoshop_overlay, to_normalised)

    # Merge the original and overlay clips
    photoshop_overlay = core.std.ShufflePlanes([photoshop_overlay, clip], planes=[0, 1, 2], colorfamily=vs.YUV)
    merged = core.std.Merge(clip, photoshop_overlay, weight=enhance)

    return merged


def HaloBuster(input: vs.VideoNode, a: int = 32, h: float = 6.4, thr: float = 1.0, elast: float = 1.5, tools=None) -> vs.VideoNode:
    # Convert to grayscale format if not already
    gray_format = vs.GRAY16 if input.format.bits_per_sample > 8 else vs.GRAY8
    gray = input.resize.Point(format=gray_format)
    
    # Add borders for padding
    gray = core.std.AddBorders(gray, left=a, top=a, right=a, bottom=a)

    # Apply NLMeans denoising
    clean = _nlmeans_gray(gray, d=0, a=a, s=0, h=h, tools=tools)
    
    # Apply TCanny for edge detection
    if pick_tool(tools, 'tcanny', ('tcanny', 'std')) == 'tcanny':
        mask = core.tcanny.TCanny(clean, sigma=1.5, mode=1)
    else:
        blurred = _boxblur_fn(tools)(clean, hradius=2, hpasses=3, vradius=2, vpasses=3)
        if pick_tool(tools, 'edgemasks', ('edgemasks', 'std')) == 'edgemasks':
            mask = core.edgemasks.Sobel(blurred)
        else:
            mask = core.std.Sobel(blurred, planes=0)

    
    max_pixel_value = (1 << gray.format.bits_per_sample) - 1
    mask = core.std.Lut(mask, function=lambda x: int(min(max((x / max_pixel_value - 0.24) * 3.2, 0.0), 1.0) * max_pixel_value))
    mask = core.std.Maximum(mask)
    mask = core.std.Inflate(mask)

    # Merge the clean image back using the mask
    merge = core.std.MaskedMerge(gray, clean, mask)
    
    # Limit the difference against the unfiltered clip, like slimit_dif in the
    # AviSynth original; this is what thr and elast steer
    limit = LimitFilter(merge, gray, thr=thr, elast=elast, tools=tools)
    
    # Crop the added borders
    crop = core.std.CropRel(limit, left=a, top=a, right=a, bottom=a)

    # Combine the final planes to get the final output
    final = core.std.ShufflePlanes([crop, input], planes=[0, 1, 2], colorfamily=vs.YUV)

    return final
