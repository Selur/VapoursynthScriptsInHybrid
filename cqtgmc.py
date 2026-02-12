import vapoursynth as vs
from functools import partial
from typing import Optional, Union, Sequence

core = vs.core

def CQTGMC(clip: vs.VideoNode, Sharpness: float=0.25, thSAD1: int=192, thSAD2: int=320, thSAD3: int=128, thSAD4: int=320, tff: bool=True, openCL: bool=False, boxed: bool=False) -> vs.VideoNode:
  
    # pad to mod64
    pHeight = (clip.height + 63) & ~63
    pWidth = (clip.width + 63) & ~63
    padded = Padding(clip, right=pWidth - clip.width, bottom=pHeight - clip.height)
    
    # spatial deinterlace
    X = 3 if tff else 2
    if openCL:
        if hasattr(core, 'sneedif'):
          spatial = core.sneedif.NNEDI3(clip=padded, field=X, qual=2)
        else:
          spatial = core.nnedi3cl.NNEDI3CL(clip=padded, field=X, qual=2)
    else:
        spatial = core.znedi3.nnedi3(clip=padded, field=X, qual=2)

    # temporal deint
    bobbed = core.bwdif.Bwdif(clip=padded, field=X, edeint=spatial)

    RG = core.zsmooth.RemoveGrain if hasattr(core,'zsmooth') else core.rgvs.RemoveGrain
    # denoise
    denoised = RG(clip=bobbed, mode=12)
    if boxed:
        if hasattr(core,'vszip'):
          denoised = core.vzip.BoxBlur(clip=denoised, planes=[0, 1, 2])
        else:
          denoised = core.std.BoxBlur(clip=denoised, planes=[0, 1, 2])
    else:
        denoised = core.fmtc.resample(clip=denoised, kernel="gauss", w=denoised.width, h=denoised.height+0.0001, scaleh=denoised.width+0.0001, interlaced=False)
        denoised = core.resize.Bicubic(clip=denoised, format=bobbed.format, dither_type="error_diffusion")
    denoised = core.std.Merge(clipa=denoised, clipb=bobbed, weight=0.25)
    
    srchClip = denoised
    csuper = core.mv.Super(clip=denoised)
    bvec = core.mv.Analyse(csuper, isb=True, blksize=64, overlap=32)
    fvec = core.mv.Analyse(csuper, isb=False, blksize=64, overlap=32)
    Comp1 = core.mv.Compensate(clip=denoised, super=csuper, vectors=bvec, thsad=thSAD2)
    Comp2 = core.mv.Compensate(clip=denoised, super=csuper, vectors=fvec, thsad=thSAD2)
    denoised = core.std.Interleave([Comp1, denoised, Comp2])
    csuper = core.mv.Super(clip=denoised)
    bvec = core.mv.Analyse(csuper, isb=True, blksize=64, overlap=32)
    fvec = core.mv.Analyse(csuper, isb=False, blksize=64, overlap=32)
    Inter = core.mv.FlowInter(denoised, csuper, bvec, fvec, blend=False)
    
    a = core.std.SelectEvery(clip=Inter, cycle=3, offsets=0)
    b = core.std.SelectEvery(clip=Inter, cycle=3, offsets=1)
    Y = core.std.ShufflePlanes([srchClip], [0], vs.GRAY)
    diffclip = core.std.PlaneStats(Y, Y[0]+Y[0:-1])
    Y_trimmed = Y[1:]+Y[-1:]
    diffclip_trimmed = core.std.PlaneStats(Y_trimmed, Y_trimmed[0]+Y_trimmed[0:-1])
    
    def selectQTGMC(n, f, a, b):
        P = f[0].props['PlaneStatsDiff']
        N = f[1].props['PlaneStatsDiff']
        if N < P:
            return a
        else:
            return b
    
    srchClip = core.std.FrameEval(clip=srchClip, eval=partial(selectQTGMC, a=a, b=b), prop_src=[diffclip, diffclip_trimmed])
    
    csuper = core.mv.Super(clip=srchClip)
    bVec1 = core.mv.Analyse(super=csuper, isb=True, overlap=4, delta=1)
    fVec1 = core.mv.Analyse(super=csuper, isb=False, overlap=4, delta=1)
    csuper = core.mv.Super(clip=bobbed, levels=1)
    bComp1 = core.mv.Compensate(clip=bobbed, super=csuper, vectors=bVec1, thsad=thSAD3)
    fComp1 = core.mv.Compensate(clip=bobbed, super=csuper, vectors=fVec1, thsad=thSAD3)
    
    Inter = core.std.Interleave([
        bobbed.std.SeparateFields(tff=tff).std.SelectEvery(4, 0),
        fComp1.std.SeparateFields(tff=tff).std.SelectEvery(4, 1),
        bComp1.std.SeparateFields(tff=tff).std.SelectEvery(4, 2),
        bobbed.std.SeparateFields(tff=tff).std.SelectEvery(4, 3)
    ]) 
    
    weaved = core.std.DoubleWeave(clip=Inter)[::2]
    csuper = core.mv.Super(clip=weaved, levels=1)
    
    bComp1 = core.mv.Compensate(clip=weaved, super=csuper, vectors=bVec1, thsad=thSAD4)
    fComp1 = core.mv.Compensate(clip=weaved, super=csuper, vectors=fVec1, thsad=thSAD4)
    EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr else core.std.Expr
    tMax = EXPR(clips=[weaved, bComp1], expr=['x y max'])
    tMax = EXPR(clips=[tMax, fComp1], expr=['x y max'])
    tMin = EXPR(clips=[weaved, bComp1], expr=['x y min'])
    tMin = EXPR(clips=[tMin, fComp1], expr=['x y min'])
    
    degrained = core.mv.Degrain1(clip=weaved, super=csuper, mvbw=bVec1, mvfw=fVec1, thsad=thSAD1)
    sharpen = core.std.MergeDiff(degrained, core.std.MakeDiff(degrained, RG(degrained, mode=20)))
    sharpen = mt_clamp(sharpen, tMax, tMin, Sharpness, Sharpness, [0])
    
    csuper = core.mv.Super(sharpen, levels=1)
    degrained = core.mv.Degrain1(clip=degrained, super=csuper, mvbw=bVec1, mvfw=fVec1, thsad=thSAD1)
    
    # Crop back to the original dimensions
    degrained = core.std.Crop(degrained, left=0, top=0, right=pWidth - clip.width, bottom=pHeight - clip.height)
    
    return degrained

def mt_clamp(
    clip: vs.VideoNode,
    bright_limit: vs.VideoNode,
    dark_limit: vs.VideoNode,
    overshoot: int = 0,
    undershoot: int = 0,
    planes: Optional[Union[int, Sequence[int]]] = None,
) -> vs.VideoNode:
    if not (isinstance(clip, vs.VideoNode) and isinstance(bright_limit, vs.VideoNode) and isinstance(dark_limit, vs.VideoNode)):
        raise vs.Error('mt_clamp: this is not a clip')

    if bright_limit.format.id != clip.format.id or dark_limit.format.id != clip.format.id:
        raise vs.Error('mt_clamp: clips must have the same format')

    plane_range = range(clip.format.num_planes)

    if planes is None:
        planes = list(plane_range)
    elif isinstance(planes, int):
        planes = [planes]

    expr = [f'x y {overshoot} + < y {overshoot} + x ? z {undershoot} - > z {undershoot} - x ?' if i in planes else '' for i in plane_range]
    EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr else core.std.Expr
    return EXPR(clips=[clip, bright_limit, dark_limit], expr=expr)


def Padding(input: vs.VideoNode, top: int=0, bottom: int=0, left: int=0, right: int=0, color: Optional[Sequence[int]] = None) -> vs.VideoNode:
    return input.std.AddBorders(left=left, right=right, top=top, bottom=bottom, color=color or [0] * input.format.num_planes)