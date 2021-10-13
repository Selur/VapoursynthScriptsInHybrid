'''
added nnedi3cl_rpow2
'''
def nnedi3_rpow2(clip,rfactor,correct_shift="fmtconv",nsize=0,nns=3,qual=None,etype=None,pscrn=None,opt=None,int16_prescreener=None,int16_predictor=None,exp=None):
  import vapoursynth as vs
  core = vs.core

  def edi(clip,field,dh):
    if hasattr(core, 'znedi3'):  
      return core.znedi3.nnedi3(clip=clip,field=field,dh=dh,nsize=nsize,nns=nns,qual=qual,etype=etype,pscrn=pscrn,opt=opt,int16_prescreener=int16_prescreener,int16_predictor=int16_predictor,exp=exp)
    else:
      return core.nnedi3.nnedi3(clip=clip,field=field,dh=dh,nsize=nsize,nns=nns,qual=qual,etype=etype,pscrn=pscrn,opt=opt,int16_prescreener=int16_prescreener,int16_predictor=int16_predictor,exp=exp)

  return edi_rpow2(clip=clip,rfactor=rfactor,correct_shift=correct_shift,edi=edi)

def nnedi3cl_rpow2(clip,rfactor,correct_shift="fmtconv",nsize=0,nns=3,qual=None,etype=None,pscrn=None):
  import vapoursynth as vs
  core = vs.core
  
  def edi(clip,field,dh):
    return core.nnedi3cl.NNEDI3CL(clip=clip,field=field,dh=dh,nsize=nsize,nns=nns,qual=qual,etype=etype,pscrn=pscrn)

  return edi_rpow2(clip=clip,rfactor=rfactor,correct_shift=correct_shift,edi=edi)
  
def eedi3_rpow2(clip,rfactor,correct_shift="fmtconv",alpha=None,beta=None,gamma=None,nrad=None,mdis=None,hp=None,ucubic=None,cost3=None,vcheck=None,vthresh0=None,vthresh1=None,vthresh2=None,sclip=None):
  import vapoursynth as vs
  core = vs.core

  def edi(clip,field,dh):
    if hasattr(core, 'eedi3m'):
      return core.eedi3m.EEDI3(clip=clip,field=field,dh=dh,alpha=alpha,beta=beta,gamma=gamma,nrad=nrad,mdis=mdis,hp=hp,ucubic=ucubic,cost3=cost3,vcheck=vcheck,vthresh0=vthresh0,vthresh1=vthresh1,vthresh2=vthresh2,sclip=sclip)
    else:
      return core.eedi3.eedi3(clip=clip,field=field,dh=dh,alpha=alpha,beta=beta,gamma=gamma,nrad=nrad,mdis=mdis,hp=hp,ucubic=ucubic,cost3=cost3,vcheck=vcheck,vthresh0=vthresh0,vthresh1=vthresh1,vthresh2=vthresh2,sclip=sclip)

  return edi_rpow2(clip=clip,rfactor=rfactor,correct_shift=correct_shift,edi=edi)

def eedi2_rpow2(clip,rfactor,correct_shift="fmtconv",mthresh=None,lthresh=None,vthresh=None,estr=None,dstr=None,maxd=None,map=None,nt=None,pp=None):
	import vapoursynth as vs
	core = vs.core
	
	def edi(clip,field,dh):
		return core.eedi2.EEDI2(clip=clip,field=field,mthresh=mthresh,lthresh=lthresh,vthresh=vthresh,estr=estr,dstr=dstr,maxd=maxd,map=map,nt=nt,pp=pp)

	return edi_rpow2(clip=clip,rfactor=rfactor,correct_shift=correct_shift,edi=edi)

def edi_rpow2(clip,rfactor,correct_shift,edi):
	import vapoursynth as vs
	import math
	core = vs.core
	
	steps=math.log(rfactor)/math.log(2) # 2^steps=rfactor
	
	if (steps).is_integer:
		steps=int(steps)
	else :
		raise ValueError('nnedi3_rpow2 : rfactor must be a power of two')
	
	if correct_shift not in [None,"fmtconv","zimg"]:
		raise ValueError('correct_shift must be None, "fmtconv" or "zimg"')
	
	clip=core.std.Transpose(clip)
	for i in range(0,steps):
		clip=edi(clip,field=1,dh=1)
	clip=core.std.Transpose(clip)
	clip=edi(clip,field=1,dh=1)
	for i in range(0,steps-1):
		clip=edi(clip,field=0,dh=1)
	
	if correct_shift=="fmtconv" or correct_shift=="zimg":
		clip=correct_edi_shift(clip,rfactor=rfactor,plugin=correct_shift)
		
	return clip

def correct_edi_shift(clip,rfactor,plugin):
	import vapoursynth as vs
	core = vs.core
	
	if clip.format.subsampling_w==1:
		hshift=-rfactor/2+0.5 # hshift(steps+1)=2*hshift(steps)-0.5
	else :
		hshift=-0.5
	
	if plugin=="fmtconv":
		bits=clip.format.bits_per_sample
		if clip.format.subsampling_h==0:
			clip=core.fmtc.resample(clip=clip,sx=hshift,sy=-0.5)
		else :
			clip=core.fmtc.resample(clip=clip,sx=hshift,sy=-0.5,planes=[3,2,2])
			clip=core.fmtc.resample(clip=clip,sx=hshift,sy=-1,planes=[2,3,3])
		if bits!=16:
			clip=core.fmtc.bitdepth(clip=clip,bits=bits)
	
	if plugin=="zimg":
		if clip.format.subsampling_h==0:
			clip=core.z.Subresize(clip=clip,resample_filter="spline36",width=clip.width,height=clip.height,shift_w=hshift,shift_h=-0.5)
		else :
			Y=core.std.ShufflePlanes(clips=clip, planes=0, colorfamily=vs.GRAY)
			U=core.std.ShufflePlanes(clips=clip, planes=1, colorfamily=vs.GRAY)
			V=core.std.ShufflePlanes(clips=clip, planes=2, colorfamily=vs.GRAY)
			Y=core.z.Subresize(clip=Y,resample_filter="spline36",width=clip.width,height=clip.height,shift_w=hshift,shift_h=-0.5)
			U=core.z.Subresize(clip=U,resample_filter="spline36",width=clip.width,height=clip.height,shift_w=hshift/2,shift_h=-0.5)
			V=core.z.Subresize(clip=V,resample_filter="spline36",width=clip.width,height=clip.height,shift_w=hshift/2,shift_h=-0.5)
			clip=core.std.ShufflePlanes(clips=[Y,U,V], planes=[0,0,0], colorfamily=vs.YUV)
	
	return clip
