from vapoursynth import core
import vapoursynth as vs

###
# requirements:
# RemoveGrain 
# MVTools2
#
# author: VS_Fan, see: https://forum.doom9.org/showthread.php?p=1769570#post1769570
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
  
  
  
def ChubbyRain(clip, th = 10, radius = 3 , show= False, interlaced = False, tff = True):
  if interlaced is True:
    res = core.std.SeparateFields(clip=clip, tff=tff)
  else:
    res = clip

  # remove rainbow
  y, u, v = core.std.SplitPlanes(res)
  uc = core.std.Convolution(u, [1,-2,1], mode = "v")
  vc = core.std.Convolution(v, [1,-2,1], mode = "v")
  ucc = core.std.Convolution(u, [1,2,1], planes=0, mode = "v")
  vcc = core.std.Convolution(v, [1,2,1], planes=0, mode = "v")
  cc = core.std.ShufflePlanes([y,ucc,vcc], planes=[0, 0, 0], colorfamily=vs.YUV)
  cc = core.focus2.TemporalSoften2(cc, radius=radius,luma_threshold=0,chroma_threshold=255,scenechange=2,mode=2)

  # create mask
  rainbow = core.std.Expr([uc,vc],"x y + " + str(th) + " > 256 0 ?")
  rainbow = core.resize.Point(rainbow, res.width, res.height)
  #rainbow = core.std.Maximum(rainbow, planes=[0], threshold=3)
  #rainbow = core.std.Maximum(rainbow, planes=[1], threshold=128)
  #rainbow = core.std.Maximum(rainbow, planes=[2], threshold=128)
  rainbow = core.std.Maximum(rainbow)

  resfinal = core.std.MaskedMerge(res, cc, rainbow)
  	
  if show is True:
    output = rainbow
  else:
    if interlaced is True:
      output = core.std.DoubleWeave(resfinal,tff=tff)
      output = core.std.SelectEvery(output, 2, 0)
    else:
      output = resfinal

  return output