from vapoursynth import core
import vapoursynth as vs
import muvsfunc as muf
import mvsfunc as mvf

"""
adjusted to use zsmooth for TemporalMedian
"""

def mClean(clip, thSAD=400, chroma=True, sharp=10, rn=14, deband=0, depth=0, strength=20, outbits=None, icalc=True, rgmode=18):
    """
    From: https://forum.doom9.org/showthread.php?t=174804 by burfadel
    mClean spatio/temporal denoiser

    +++ Description +++
    Typical spatial filters work by removing large variations in the image on a small scale, reducing noise but also making the image less
    sharp or temporally stable. mClean removes noise whilst retaining as much detail as possible, as well as provide optional image enhancement.

    mClean works primarily in the temporal domain, although there is some spatial limiting.
    Chroma is processed a little differently to luma for optimal results.
    Chroma processing can be disabled with chroma = False.

    +++ Artifacts +++
    Spatial picture artifacts may remain as removing them is a fine balance between removing the unwanted artifact whilst not removing detail.
    Additional dering/dehalo/deblock filters may be required, but should ONLY be uses if required due the detail loss/artifact removal balance.

    +++ Sharpening +++
    Applies a modified unsharp mask to edges and major detected detail. Range of normal sharpening is 0-20. There are 4 additional settings,
    21-24 that provide 'overboost' sharpening. Overboost sharpening is only suitable typically for high definition, high quality sources.
    Actual sharpening calculation is scaled based on resolution.

    +++ ReNoise +++
    ReNoise adds back some of the removed luma noise. Re-adding original noise would be counterproductive, therefore ReNoise modifies this noise
    both spatially and temporally. The result of this modification is the noise becomes much nicer and it's impact on compressibility is greatly
    reduced. It is not applied on areas where the sharpening occurs as that would be counterproductive. Settings range from 0 to 20.
    The strength of renoise is affected by the the amount of original noise removed and how this noise varies between frames.
    It's main purpose is to reduce the 'flatness' that occurs with any form of effective denoising.

    +++ Deband +++
    This will perceptibly improve the quality of the image by reducing banding effect and adding a small amount of temporally stabilised grain
    to both luma and chroma. The settings are not adjustable as the default settings are suitable for most cases without having a large effect
    on compressibility. 0 = disabled, 1 = deband only, 2 = deband and veed

    +++ Depth +++
    This applies a modified warp sharpening on the image that may be useful for certain things, and can improve the perception of image depth.
    Settings range up from 0 to 5. This function will distort the image, for animation a setting of 1 or 2 can be beneficial to improve lines.

    +++ Strength +++
    The strength of the denoising effect can be adjusted using this parameter. It ranges from 20 percent denoising effect with strength 0, up to the
    100 percent of the denoising with strength 20. This function works by blending a scaled percentage of the original image with the processed image.

    +++ Outbits +++
    Specifies the bits per component (bpc) for the output for processing by additional filters. It will also be the bpc that mClean will process.
    If you output at a higher bpc keep in mind that there may be limitations to what subsequent filters and the encoder may support.
    """
    # New parameter icalc, set to True to enable pure integer processing for faster speed. (Ignored if input is of float sample type)
    
    defH = max(clip.height, clip.width // 4 * 3) # Resolution calculation for auto blksize settings
    sharp = min(max(sharp, 0), 24) # Sharp multiplier
    rn = min(max(rn, 0), 20) # Luma ReNoise strength
    deband = min(max(deband, 0), 5)  # Apply deband/veed
    depth = min(max(depth, 0), 5) # Depth enhancement
    strength = min(max(strength, 0), 20) # Strength of denoising
    bd = clip.format.bits_per_sample
    isFLOAT = clip.format.sample_type == vs.FLOAT
    icalc = False if isFLOAT else icalc
    if hasattr(core, 'mvsf'):  
      S = core.mv.Super if icalc else core.mvsf.Super
      A = core.mv.Analyse if icalc else core.mvsf.Analyse
      R = core.mv.Recalculate if icalc else core.mvsf.Recalculate
    else:
     S = core.mv.Super
     A = core.mv.Analyse
     R = core.mv.Recalculate

    if not isinstance(clip, vs.VideoNode) or clip.format.color_family != vs.YUV:
        raise TypeError("mClean: This is not a YUV clip!")

    if outbits is None: # Output bits, default input depth
        outbits = bd

    if deband or depth:
        outbits = min(outbits, 16)

    RE = core.rgsf.Repair if outbits == 32 else core.rgvs.Repair
    RG = core.rgsf.RemoveGrain if outbits == 32 else core.rgvs.RemoveGrain
    sc = 8 if defH > 2880 else 4 if defH > 1440 else 2 if defH > 720 else 1
    i = 0.00392 if outbits == 32 else 1 << (outbits - 8)
    peak = 1.0 if outbits == 32 else (1 << outbits) - 1
    bs = 16 if defH / sc > 360 else 8
    ov = 6 if bs > 12 else 2
    pel = 1 if defH > 720 else 2
    truemotion = False if defH > 720 else True
    lampa = 777 * (bs ** 2) // 64
    depth2 = -depth*3
    depth = depth*2

    if sharp > 20:
        sharp += 30
    elif defH <= 2500:
        sharp = 15 + defH * sharp * 0.0007
    else:
        sharp = 50

    # Denoise preparation
    c = core.vcm.Median(clip, plane=[0, 1, 1]) if chroma else clip

    # Temporal luma noise filter
    if not (isFLOAT or icalc):
        c = c.fmtc.bitdepth(flt=1)
    cy = core.std.ShufflePlanes(c, [0], vs.GRAY)

    super1 = S(c if chroma else cy, hpad=bs, vpad=bs, pel=pel, rfilter=4, sharp=1)
    super2 = S(c if chroma else cy, hpad=bs, vpad=bs, pel=pel, rfilter=1, levels=1)
    analyse_args = dict(blksize=bs, overlap=ov, search=5, truemotion=truemotion)
    recalculate_args = dict(blksize=bs, overlap=ov, search=5, truemotion=truemotion, thsad=180, lambda_=lampa)

    # Analysis
    bvec4 = R(super1, A(super1, isb=True,  delta=4, **analyse_args), **recalculate_args) if not icalc else None
    bvec3 = R(super1, A(super1, isb=True,  delta=3, **analyse_args), **recalculate_args)
    bvec2 = R(super1, A(super1, isb=True,  delta=2, badsad=1100, lsad=1120, **analyse_args), **recalculate_args)
    bvec1 = R(super1, A(super1, isb=True,  delta=1, badsad=1500, lsad=980, badrange=27, **analyse_args), **recalculate_args)
    fvec1 = R(super1, A(super1, isb=False, delta=1, badsad=1500, lsad=980, badrange=27, **analyse_args), **recalculate_args)
    fvec2 = R(super1, A(super1, isb=False, delta=2, badsad=1100, lsad=1120, **analyse_args), **recalculate_args)
    fvec3 = R(super1, A(super1, isb=False, delta=3, **analyse_args), **recalculate_args)
    fvec4 = R(super1, A(super1, isb=False, delta=4, **analyse_args), **recalculate_args) if not icalc else None

    # Applying cleaning
    if not icalc:
        clean = core.mvsf.Degrain4(c if chroma else cy, super2, bvec1, fvec1, bvec2, fvec2, bvec3, fvec3, bvec4, fvec4, thsad=thSAD)
    else:
        clean = core.mv.Degrain3(c if chroma else cy, super2, bvec1, fvec1, bvec2, fvec2, bvec3, fvec3, thsad=thSAD)

    if c.format.bits_per_sample != outbits:
        c = c.fmtc.bitdepth(bits=outbits, dmode=1)
        cy = cy.fmtc.bitdepth(bits=outbits, dmode=1)
        clean = clean.fmtc.bitdepth(bits=outbits, dmode=1)

    uv = core.std.MergeDiff(clean, core.zsmooth.TemporalMedian(core.std.MakeDiff(c, clean, [1, 2]), 1, [1, 2]), [1, 2]) if chroma else c
    clean = core.std.ShufflePlanes(clean, [0], vs.GRAY) if clean.format.num_planes != 1 else clean

    # Post clean, pre-process deband
    filt = core.std.ShufflePlanes([clean, uv], [0, 1, 2], vs.YUV)

    if deband:
        filt = filt.f3kdb.Deband(range=16, preset="high" if chroma else "luma", grainy=defH/15, grainc=defH/16 if chroma else 0, output_depth=outbits)
        clean = core.std.ShufflePlanes(filt, [0], vs.GRAY)
        filt = core.vcm.Veed(filt) if deband == 2 else filt

    # Spatial luma denoising
    clean2 = RG(clean, rgmode)

    # Unsharp filter for spatial detail enhancement
    if sharp:
        if sharp <= 50:
            clsharp = core.std.MakeDiff(clean, muf.Blur(clean2, amountH=0.08+0.03*sharp))
        else:
            clsharp = core.std.MakeDiff(clean, clean2.tcanny.TCanny(sigma=(sharp-46)/4, mode=-1))
        clsharp = core.std.MergeDiff(clean2, RE(clsharp.zsmooth.TemporalMedian(), clsharp, 12))

    # If selected, combining ReNoise
    noise_diff = core.std.MakeDiff(clean2, cy)
    if rn:
        expr = "x {a} < 0 x {b} > {p} 0 x {c} - {p} {a} {d} - / * - ? ?".format(a=32*i, b=45*i, c=35*i, d=65*i, p=peak)
        clean1 = core.std.Merge(clean2, core.std.MergeDiff(clean2, Tweak(noise_diff.zsmooth.TemporalMedian(), cont=1.008+0.00016*rn)), 0.3+rn*0.035)
        clean2 = core.std.MaskedMerge(clean2, clean1, core.std.Expr([core.std.Expr([clean, clean.std.Invert()], 'x y min')], [expr]))

    # Combining spatial detail enhancement with spatial noise reduction using prepared mask
    noise_diff = noise_diff.std.Binarize().std.Invert()
    clean2 = core.std.MaskedMerge(clean2, clsharp if sharp else clean, core.std.Expr([noise_diff, clean.std.Sobel()], 'x y max'))

    # Combining result of luma and chroma cleaning
    output = core.std.ShufflePlanes([clean2, filt], [0, 1, 2], vs.YUV)
    output = core.std.Merge(c, output, 0.2+0.04*strength) if strength < 20 else output
    return core.std.MergeDiff(output, core.std.MakeDiff(output.warp.AWarpSharp2(128, 3, 1, depth2, 1), output.warp.AWarpSharp2(128, 2, 1, depth, 1))) if depth else output
    
# helper used in mClean
def Tweak(clip, hue=None, sat=None, bright=None, cont=None, coring=True):

    bd = clip.format.bits_per_sample
    isFLOAT = clip.format.sample_type == vs.FLOAT
    isGRAY = clip.format.color_family == vs.GRAY
    mid = 0 if isFLOAT else 1 << (bd - 1)

    if clip.format.color_family in [vs.RGB]:
        raise TypeError("Tweak: RGB color family is not supported!")
        
    if not (hue is None and sat is None or isGRAY):
        hue = 0.0 if hue is None else hue
        sat = 1.0 if sat is None else sat
        hue = hue * math.pi / 180
        sinh = math.sin(hue)
        cosh = math.cos(hue)
        cmin = -0.5 if isFLOAT else 16 << (bd - 8) if coring else 0
        cmax = 0.5 if isFLOAT else 240 << (bd - 8) if coring else (1 << bd) - 1
        expr_u = "x {} * y {} * + -0.5 max 0.5 min".format(cosh * sat, sinh * sat) if isFLOAT else "x {} - {} * y {} - {} * + {} + {} max {} min".format(mid, cosh * sat, mid, sinh * sat, mid, cmin, cmax)
        expr_v = "y {} * x {} * - -0.5 max 0.5 min".format(cosh * sat, sinh * sat) if isFLOAT else "y {} - {} * x {} - {} * - {} + {} max {} min".format(mid, cosh * sat, mid, sinh * sat, mid, cmin, cmax)
        src_u = core.std.ShufflePlanes(clip, [1], vs.GRAY)
        src_v = core.std.ShufflePlanes(clip, [2], vs.GRAY)
        dst_u = core.std.Expr([src_u, src_v], expr_u)
        dst_v = core.std.Expr([src_u, src_v], expr_v)
        clip = core.std.ShufflePlanes([clip, dst_u, dst_v], [0, 0, 0], clip.format.color_family)

    if not (bright is None and cont is None):
        bright = 0.0 if bright is None else bright
        cont = 1.0 if cont is None else cont

        if isFLOAT:
            expr = "x {} * {} + 0.0 max 1.0 min".format(cont, bright)
            clip =  core.std.Expr([clip], [expr] if isGRAY else [expr, ''])
        else:
            luma_lut = []
            luma_min = 16  << (bd - 8) if coring else 0
            luma_max = 235 << (bd - 8) if coring else (1 << bd) - 1

            for i in range(1 << bd):
                val = int((i - luma_min) * cont + bright + luma_min + 0.5)
                luma_lut.append(min(max(val, luma_min), luma_max))

            clip = core.std.Lut(clip, [0], luma_lut)

    return clip
