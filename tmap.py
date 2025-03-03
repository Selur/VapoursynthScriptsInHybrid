import vapoursynth as vs

def tm(clip="",source_peak="",desat=50,lin=True,show_satmask=False,show_clipped=False ) :
    core = vs.core
    c=clip
    o=c
    a=c

    source_peak=source_peak 
    LDR_nits=100     
    exposure_bias=source_peak/LDR_nits

    if (desat <  0) :
        desat=0
    if (desat >  100) :
       desat=100
    desat=desat/100

    ldr_value=(1/exposure_bias)# for example in linear light compressed (0-1 range ) hdr 1000 nits, 100nits becomes 0.1 
    tm=((1*(0.15*1+0.10*0.50)+0.20*0.02) / (1*(0.15*1+0.50)+0.20*0.30)) - 0.02/0.30
    w=((exposure_bias*(0.15*exposure_bias+0.10*0.50)+0.20*0.02)/(exposure_bias*(0.15*exposure_bias+0.50)+0.20*0.30))-0.02/0.30
    tm_ldr_value=tm * (1 / w)#value of 100 nits after the tone mapping
    ldr_value_mult=tm_ldr_value/(1/exposure_bias)#0.1 (100nits) * ldr_value_mult=tm_ldr_value

    tm = core.std.Expr(c, expr="x  {exposure_bias} * 0.15 x  {exposure_bias} * * 0.05 + * 0.004 + x  {exposure_bias} * 0.15 x  {exposure_bias} * * 0.50 + * 0.06 + / 0.02 0.30 / -  ".format(exposure_bias=exposure_bias),format=vs.RGBS)
    w=((exposure_bias*(0.15*exposure_bias+0.10*0.50)+0.20*0.02)/(exposure_bias*(0.15*exposure_bias+0.50)+0.20*0.30))-0.02/0.30
    tm = core.std.Expr(clips=[tm,c], expr="x  1 {w}  / * ".format(exposure_bias=exposure_bias,w=w),format=vs.RGBS)
    vszip = hasattr(core,'vszip')
    tm = core.vszip.Limiter(tm, [0,0,0], [1,1,1]) if vszip else core.std.Limiter(tm, 0, 1)

    if lin == True :
        #linearize the tonemapper curve under 100nits
        tm=core.std.Expr(clips=[tm,o], expr="x {tm_ldr_value} < y {ldr_value_mult} * x ? ".format(tm_ldr_value=tm_ldr_value,ldr_value_mult=ldr_value_mult))
    

    r=core.std.ShufflePlanes(clips=[a], planes=[0], colorfamily=vs.GRAY)
    g=core.std.ShufflePlanes(clips=[a], planes=[1], colorfamily=vs.GRAY)
    b=core.std.ShufflePlanes(clips=[a], planes=[2], colorfamily=vs.GRAY)
    #luminance
    l=core.std.Expr(clips=[r,g,b], expr="x 0.2627 *  y 0.678 * + z 0.0593 * +    ",format=vs.GRAY)


    #value under 100nits after the tone mapping(tm_ldr_value) becomes 0, even tm_ldr_value becomes 0 and then scale all in the 0-1 range fo the mask 
    mask=core.std.Expr(clips=[l], expr="x {ldr_value_mult} * {tm_ldr_value}  - {ldr_value_mult} /".format(ldr_value_mult=ldr_value_mult,tm_ldr_value=tm_ldr_value))
    mask = core.vszip.Limiter(mask, 0, 1) if vszip else core.std.Limiter(mask, 0, 1)


    #reduce the saturation blending with grayscale
    lrgb=core.std.ShufflePlanes(clips=[l,l,l], planes=[0,0,0], colorfamily=vs.RGB)
    asat=core.std.Expr(clips=[a,lrgb], expr=" y {desat} * x 1 {desat} - * + ".format(tm_ldr_value=tm_ldr_value,desat=desat))
    a=core.std.MaskedMerge(a, asat, mask)


    r=core.std.ShufflePlanes(clips=[a], planes=[0], colorfamily=vs.GRAY)
    g=core.std.ShufflePlanes(clips=[a], planes=[1], colorfamily=vs.GRAY)
    b=core.std.ShufflePlanes(clips=[a], planes=[2], colorfamily=vs.GRAY)

    rl=core.std.ShufflePlanes(clips=[tm], planes=[0], colorfamily=vs.GRAY)
    gl=core.std.ShufflePlanes(clips=[tm], planes=[1], colorfamily=vs.GRAY)
    bl=core.std.ShufflePlanes(clips=[tm], planes=[2], colorfamily=vs.GRAY)
    l2=core.std.Expr(clips=[rl,gl,bl], expr="x 0.2627 *  y 0.678 * + z 0.0593 * +    ",format=vs.GRAY)
    nl= l2
    scale=core.std.Expr(clips=[nl,l], expr="x y / ")
    r1=core.std.Expr(clips=[r,scale], expr="x  y *")
    g1=core.std.Expr(clips=[g,scale], expr="x  y *")
    b1=core.std.Expr(clips=[b,scale], expr="x  y *")

    c=core.std.ShufflePlanes(clips=[r1,g1,b1], planes=[0,0,0], colorfamily=vs.RGB)
    c = core.vszip.Limiter(c, [0,0,0], [1,1,1]) if vszip else core.std.Limiter(c, 0, 1)

    if show_satmask == True :
        #show mask
        c=core.std.ShufflePlanes(clips=[mask,mask,mask], planes=[0,0,0], colorfamily=vs.RGB)

    if show_clipped == True :
        #show clipped
        c=core.std.Expr(clips=[o], expr="x {ldr_value_mult} * ".format(ldr_value_mult=ldr_value_mult))
 
    return c 

def hablehdr10tosdr(clip, source_peak=1000, desat=50, tFormat=vs.YUV420P8, tMatrix="709", tRange="limited", color_loc="center", f_a=0.0, f_b=0.75, show_satmask=False,lin=True,show_clipped=False) :
  core = vs.core
  clip=core.resize.Bicubic(clip=clip, format=vs.RGBS, filter_param_a=f_a, filter_param_b=f_b, range_in_s="limited", matrix_in_s="2020ncl", primaries_in_s="2020", primaries_s="2020", transfer_in_s="st2084", transfer_s="linear",dither_type="none", nominal_luminance=1000)
  clip=tm(clip=clip,source_peak=source_peak,desat=desat,show_satmask=show_satmask,lin=lin,show_clipped=show_clipped) 
  if tFormat != vs.RGBS:
    clip=core.resize.Bicubic(clip=clip, format=tFormat, filter_param_a=f_a, filter_param_b=f_b, matrix_s=tMatrix, primaries_in_s="2020", primaries_s=tMatrix, transfer_in_s="linear", transfer_s=tMatrix, dither_type="ordered")
    
  return clip