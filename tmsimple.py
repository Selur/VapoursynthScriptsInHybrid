
import vapoursynth as vs


def tm(clip="",source_peak="" ) :
    core = vs.get_core()
    c=clip
    o=c
    a=c
    s=0.5
    source_peak=source_peak 
    LDR_nits=100     
    exposure_bias=source_peak/LDR_nits

    if (s <  0) :
        s=0

    #tm=((x*exposure_bias*(0.15*x*exposure_bias+0.10*0.50)+0.20*0.02) / (x*exposure_bias*(0.15*x*exposure_bias+0.50)+0.20*0.30)) - 0.02/0.30
    #w=((exposure_bias*(0.15*exposure_bias+0.10*0.50)+0.20*0.02)/(exposure_bias*(0.15*exposure_bias+0.50)+0.20*0.30))-0.02/0.30
    #tm=tm * (1 / w)

    tm = core.std.Expr(c, expr="x  {exposure_bias} * 0.15 x  {exposure_bias} * * 0.05 + * 0.004 + x  {exposure_bias} * 0.15 x  {exposure_bias} * * 0.50 + * 0.06 + / 0.02 0.30 / -  ".format(exposure_bias=exposure_bias),format=vs.RGBS)
    w=((exposure_bias*(0.15*exposure_bias+0.10*0.50)+0.20*0.02)/(exposure_bias*(0.15*exposure_bias+0.50)+0.20*0.30))-0.02/0.30
    tm = core.std.Expr(clips=[tm,c], expr="x  1 {w}  / * ".format(exposure_bias=exposure_bias,w=w),format=vs.RGBS)
    tm = core.std.Limiter(tm, 0, 1)

    r=core.std.ShufflePlanes(clips=[a], planes=[0], colorfamily=vs.GRAY)
    g=core.std.ShufflePlanes(clips=[a], planes=[1], colorfamily=vs.GRAY)
    b=core.std.ShufflePlanes(clips=[a], planes=[2], colorfamily=vs.GRAY)
    l=core.std.Expr(clips=[r,g,b], expr="x 0.2627 *  y 0.678 * + z 0.0593 * +    ",format=vs.GRAY)
    #l=0.2627 * R + 0.678 * G + 0.0593 * B (formula for rec2020 luminance)
    l=core.std.Expr(clips=[l], expr="x 0 =  0.00001 x  ? ",format=vs.GRAY)#avoid dividing by 0

    rl=core.std.ShufflePlanes(clips=[tm], planes=[0], colorfamily=vs.GRAY)
    gl=core.std.ShufflePlanes(clips=[tm], planes=[1], colorfamily=vs.GRAY)
    bl=core.std.ShufflePlanes(clips=[tm], planes=[2], colorfamily=vs.GRAY)
    l2=core.std.Expr(clips=[rl,gl,bl], expr="x 0.2627 *  y 0.678 * + z 0.0593 * +    ",format=vs.GRAY)

    #r1=((r/l)^s)*l2
    #g1=((g/l)^s)*l2
    #b1=((b/l)^s)*l2
    r1=core.std.Expr(clips=[r,l,l2], expr="x  y /  {s} pow  z * ".format(s=s))
    g1=core.std.Expr(clips=[g,l,l2], expr="x  y /  {s} pow  z * ".format(s=s))
    b1=core.std.Expr(clips=[b,l,l2], expr="x  y /  {s} pow z *  ".format(s=s))

    r2=core.std.Expr(clips=[r,l,l2], expr="x  y /   z * ")
    g2=core.std.Expr(clips=[g,l,l2], expr="x  y /   z * ")
    b2=core.std.Expr(clips=[b,l,l2], expr="x  y /  z *  ")
    mask=l2
    mask = core.std.Limiter(mask, 0, 1)

    csat=core.std.ShufflePlanes(clips=[r1,g1,b1], planes=[0,0,0], colorfamily=vs.RGB)

    c=core.std.ShufflePlanes(clips=[r2,g2,b2], planes=[0,0,0], colorfamily=vs.RGB)
    c2=core.std.MaskedMerge(c, csat, mask)
    c=core.std.Merge(c, c2, 0.5)
    c = core.std.Limiter(c, 0, 1)

    return c 


   
def simplehdr10tosdr(clip, source_peak=1000, tFormat=vs.YUV420P8, tMatrix="709", tRange="limited", color_loc="center", f_a=0.0, f_b=0.75) :
  core = vs.get_core()
  clip=core.resize.Bicubic(clip=clip, format=vs.RGBS,filter_param_a=f_a,filter_param_b=f_b, range_in_s="limited", matrix_in_s="2020ncl", primaries_in_s="2020", primaries_s="2020", transfer_in_s="st2084", transfer_s="linear",dither_type="none", nominal_luminance=1000)
  clip=tm(clip=clip,source_peak=source_peak)
  if tFormat != vs.RGBS:
    clip=core.resize.Bicubic(clip=clip, format=tFormat, filter_param_a=f_a, filter_param_b=f_b, matrix_s=tMatrix, primaries_in_s="2020", primaries_s=tMatrix, transfer_in_s="linear", transfer_s=tMatrix, dither_type="ordered")
  return clip