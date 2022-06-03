from vapoursynth import core
import vapoursynth as vs

###
# requirements:
# RemoveGrain 
# MVTools2
#
# author: VS_Fan, see: https://forum.doom9.org/showthread.php?p=1769570#post1769570
##
def StabilizeIT(clip: vs.VideoNode, div: float=2.0, initZoom: float=1.0, zoomMax: float=1.0, rotMax: float=10.0, pixelAspect: float=1.0, thSCD1: int=800, thSCD2: int=150, stabMethod: int=1, cutOff: float=0.33, anaError: float=30.0, rgMode: int=4):
  pf = core.rgvs.RemoveGrain(clip=clip, mode=rgMode)
  pf = core.resize.Bilinear(clip=pf, width=int(pf.width/div), height=int(pf.height/div))
  pf = core.rgvs.RemoveGrain(clip=pf, mode=rgMode)
  pf = core.resize.Bilinear(clip=pf, width=pf.width*div, height=pf.height*div) 
  super = core.mv.Super(clip=pf) 
  vectors = core.mv.Analyse(super=super, isb=False)
  globalmotion = core.mv.DepanAnalyse(clip=pf, vectors=vectors, pixaspect=pixelAspect, error=anaError, thscd1=thSCD1, thscd2=thSCD2)
  clip = core.mv.DepanStabilise(clip=clip, data=globalmotion, cutoff=cutOff, initzoom=initZoom, zoommax=zoomMax, rotmax=rotMax, pixaspect=pixelAspect, method=stabMethod)
  return clip
  
##
# supports: GrayS, RGBS and YUV4xxPS
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
#  SceneChange: https://forum.doom9.org/showthread.php?t=166769

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
  clip = core.scd.Detect(clip)
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
