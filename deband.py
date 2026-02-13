import vapoursynth as vs
from vapoursynth import core


from typing import Sequence, Union, Optional

# Taken from fvsfunc
"""
VapourSynth port of Gebbi's GradFun3mod

Based on Muonium's GradFun3 port:
https://github.com/WolframRhodium/muvsfunc

If you don't use any of the newly added arguments
it will behave just like unmodified GradFun3.

Differences:

 - added smode=5 that uses a bilateral filter on the GPU (CUDA)
   output should be very similar to smode=2
 - fixed the strength of the bilateral filter when using 
   smode=2 to match the AviSynth version
 - changed argument lsb to bits (default is input bitdepth)
 - case of the resizer doesn't matter anymore
 - every resizer supported by fmtconv.resample can be specified
 - yuv444 can now be used with any output resolution
 - removed fh and fv arguments for all resizers

Requirements:
 - Bilateral  https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Bilateral
 - BilateralGPU (optional, needs OpenCV 3.2 with CUDA module)  https://github.com/WolframRhodium/VapourSynth-BilateralGPU
 - fmtconv  https://github.com/EleonoreMizo/fmtconv
 - Descale (optional)  https://github.com/Frechdachs/vapoursynth-descale
 - dfttest  https://github.com/HomeOfVapourSynthEvolution/VapourSynth-DFTTest
 - nnedi3  https://github.com/dubhater/vapoursynth-nnedi3
 - nnedi3_rpow2  https://gist.github.com/4re/342624c9e1a144a696c6

Original header:

##################################################################################################################
#
#   High bitdepth tools for Avisynth - GradFun3mod r6
#       based on Dither v1.27.2
#   Author: Firesledge, slightly modified by Gebbi
#
#  What?
#       - This is a slightly modified version of the original GradFun3.
#       - It combines the usual color banding removal stuff with resizers during the process
#         for sexier results (less detail loss, especially for downscales of cartoons).
#       - This is a starter script, not everything is covered through parameters. Modify it to your needs.
#
#   Requirements (in addition to the Dither requirements):
#       - AviSynth 2.6.x
#       - Debilinear, Debicubic, DebilinearM
#       - NNEDI3 + nnedi3_resize16
#
#  Changes from the original GradFun3:
#       - yuv444 = true
#         (4:2:0 -> 4:4:4 colorspace conversion, needs 1920x1080 input)
#       - resizer = [ "none", "Debilinear", "DebilinearM", "Debicubic", "DebicubicM", "Spline16",
#         "Spline36", "Spline64", "lineart_rpow2", "lineart_rpow2_bicubic" ] 
#         (use it only for downscales)
#           NOTE: As of r2 Debicubic doesn't have 16-bit precision, so a Y (luma) plane fix by torch is used here,
#                 more info: https://mechaweaponsvidya.wordpress.com/2015/07/07/a-precise-debicubic/
#                 Without yuv444=true Dither_resize16 is used with an inverse bicubic kernel.
#       - w = 1280, h = 720
#         (output width & height for the resizers; or production resolution for resizer="lineart_rpow2")
#       - smode = 4
#         (the old GradFun3mod behaviour for legacy reasons; based on smode = 1 (dfttest);
#         not useful anymore in most cases, use smode = 2 instead (less detail loss))
#       - deb = true
#         (legacy parameter; same as resizer = "DebilinearM")
#
#  Usage examples:
#       - Source is bilinear 720p->1080p upscale (BD) with 1080p credits overlayed,
#         revert the upscale without fucking up the credits:
#               lwlibavvideosource("lol.m2ts")
#               GradFun3mod(smode=1, yuv444=true, resizer="DebilinearM")
#
#       - same as above, but bicubic Catmull-Rom upscale (outlines are kind of "blocky" and oversharped):
#               GradFun3mod(smode=1, yuv444=true, resizer="DebicubicM", b=0, c=1)
#               (you may try any value between 0 and 0.2 for b, and between 0.7 and 1 for c)
#
#       - You just want to get rid off the banding without changing the resolution:
#               GradFun3(smode=2)
#
#       - Source is 1080p production (BD), downscale to 720p:
#               GradFun3mod(smode=2, yuv444=true, resizer="Spline36")
#
#       - Source is a HDTV transportstream (or CR or whatever), downscale to 720p:
#               GradFun3mod(smode=2, resizer="Spline36")
#
#       - Source is anime, 720p->1080p upscale, keep the resolution
#         but with smoother lineart instead of bilinear upscaled shit:
#               GradFun3mod(smode=2, resizer="lineart_rpow2")
#         This won't actually resize the video but instead mask the lineart and re-upscale it using
#         nnedi3_rpow2 which often results in much better looking lineart (script mostly by Daiz).
#
#       Note: Those examples don't include parameters like thr, radius, elast, mode, ampo, ampn, staticnoise.
#             You probably don't want to use the default values.
#             For 16-bit output use:
#              GradFun3mod(lsb=true).Dither_out()
#
#  What's the production resolution of my korean cartoon?
#       - Use your eyes combined with Debilinear(1280,720) - if it looks like oversharped shit,
#         it was probably produced in a higher resolution.
#       - Use Debilinear(1280,720).BilinearResize(1920,1080) for detail loss search.
#       - Alternatively you can lookup the (estimated) production resolution at
#         http://anibin.blogspot.com  (but don't blindly trust those results)
#
#   This program is free software. It comes without any warranty, to
#   the extent permitted by applicable law. You can redistribute it
#   and/or modify it under the terms of the Do What The Fuck You Want
#   To Public License, Version 2, as published by Sam Hocevar. See
#   http://sam.zoy.org/wtfpl/COPYING for more details.
#
##################################################################################################################

"""
def GradFun3(src, thr=None, radius=None, elast=None, mask=None, mode=None, ampo=None,
                ampn=None, pat=None, dyn=None, staticnoise=None, smode=None, thr_det=None,
                debug=None, thrc=None, radiusc=None, elastc=None, planes=None, ref=None,
                yuv444=None, w=None, h=None, resizer=None, b=None, c=None, bits=None):

    def smooth_mod(src_16, ref_16, smode, radius, thr, elast, planes):
        if smode == 0:
            return _GF3_smoothgrad_multistage(src_16, ref_16, radius, thr, elast, planes)
        elif smode == 1:
            return _GF3_dfttest(src_16, ref_16, radius, thr, elast, planes)
        elif smode == 2:
            return bilateral(src_16, ref_16, radius, thr, elast, planes)
        elif smode == 3:
            return _GF3_smoothgrad_multistage_3(src_16, radius, thr, elast, planes)
        elif smode == 4:
            return dfttest_mod(src_16, ref_16, radius, thr, elast, planes)
        elif smode == 5:
            return bilateral_gpu(src_16, ref_16, radius, thr, elast, planes)
        else:
            raise ValueError(funcname + ': wrong smode value!')

    def dfttest_mod(src, ref, radius, thr, elast, planes):
        hrad = max(radius * 3 // 4, 1)
        last = core.dfttest.DFTTest(src, sigma=thr * 12, sbsize=hrad * 4,
                                    sosize=hrad * 3, tbsize=1, planes=planes)
        last = LimitFilter(last, ref, thr=thr, elast=elast, planes=planes)
        return last

    def bilateral(src, ref, radius, thr, elast, planes):
        thr_1 = max(thr * 4.5, 1.25)
        thr_2 = max(thr * 9, 5.0)
        r4 = max(radius * 4 / 3, 4.0)
        r2 = max(radius * 2 / 3, 3.0)
        r1 = max(radius * 1 / 3, 2.0)
        last = src
        if hasattr(core,'vszip'):
          last = core.vszip.Bilateral(last, ref=ref, sigmaS=r4 / 2, sigmaR=thr_1 / 255, planes=planes, algorithm=0)
        else:
          last = core.bilateral.Bilateral(last, ref=ref, sigmaS=r4 / 2, sigmaR=thr_1 / 255, planes=planes, algorithm=0)
        # NOTE: I get much better results if I just call Bilateral once
        #last = core.bilateral.Bilateral(last, ref=ref, sigmaS=r2 / 2, sigmaR=thr_2 / 255,
        #                                planes=planes, algorithm=0)
        #last = core.bilateral.Bilateral(last, ref=ref, sigmaS=r1 / 2, sigmaR=thr_2 / 255,
        #                                planes=planes, algorithm=0)
        last = LimitFilter(last, src, thr=thr, elast=elast, planes=planes)
        return last

    def bilateral_gpu(src, ref, radius, thr, elast, planes):
        t = max(thr * 4.5, 1.25)
        r = max(radius * 4 / 3, 4.0)
        last = core.bilateralgpu.Bilateral(src, sigma_spatial=r / 2, sigma_color=t)
        last = LimitFilter(last, ref, thr=thr, elast=elast, planes=planes)
        return last

    funcname = 'GradFun3'

    # Type checking
    kwargsdict = {'src': [src, (vs.VideoNode,)], 'thr': [thr, (int, float)], 'radius': [radius, (int,)],
                  'elast': [elast, (int, float)], 'mask': [mask, (int,)], 'mode': [mode, (int,)],
                  'ampo': [ampo, (int, float)], 'ampn': [ampn, (int, float)], 'pat': [pat, (int,)],
                  'dyn': [dyn, (bool,)], 'staticnoise': [staticnoise, (bool,)], 'smode': [smode, (int,)],
                  'thr_det': [thr_det, (int, float)], 'debug': [debug, (bool, int)], 'thrc': [thrc, (int, float)],
                  'radiusc': [radiusc, (int,)], 'elastc': [elastc, (int, float)], 'planes': [planes, (int, list)],
                  'ref': [ref, (vs.VideoNode,)], 'yuv444': [yuv444, (bool,)], 'w': [w, (int,)], 'h': [h, (int,)],
                  'resizer': [resizer, (str,)], 'b': [b, (int, float)], 'c': [c, (int, float)], 'bits': [bits, (int,)]}

    for k, v in kwargsdict.items():
        if v[0] is not None and not isinstance(v[0], v[1]):
            raise TypeError('{funcname}: "{variable}" must be {types}!'
                            .format(funcname=funcname, variable=k, types=' or '.join([TYPEDICT[t] for t in v[1]])))

    # Set defaults
    if smode is None:
        smode = 2
    if thr is None:
        thr = 0.35
    if radius is None:
        radius = 12 if smode not in [0, 3] else 9
    if elast is None:
        elast = 3.0
    if mask is None:
        mask = 2
    if thr_det is None:
        thr_det = 2 + round(max(thr - 0.35, 0) / 0.3)
    if debug is None:
        debug = False
    if thrc is None:
        thrc = thr
    if radiusc is None:
        radiusc = radius
    if elastc is None:
        elastc = elast
    if planes is None:
        planes = list(range(src.format.num_planes))
    if ref is None:
        ref = src
    if yuv444 is None:
        yuv444 = False
    if w is None:
        w = 1280
    if h is None:
        h = 720
    if resizer is None:
        resizer = ''
    if yuv444 and not resizer:
        resizer = 'spline36'
    if b is None:
        b = 1/3
    if c is None:
        c = 1/3
    if bits is None:
        bits = src.format.bits_per_sample

    # Value checking
    if src.format.color_family not in [vs.YUV, vs.GRAY]:
        raise TypeError(funcname + ': "src" must be YUV or GRAY color family!')
    if ref.format.color_family not in [vs.YUV, vs.GRAY]:
        raise TypeError(funcname + ': "ref" must be YUV or GRAY color family!')
    if thr < 0.1 or thr > 10.0:
        raise ValueError(funcname + ': "thr" must be in [0.1, 10.0]!')
    if thrc < 0.1 or thrc > 10.0:
        raise ValueError(funcname + ': "thrc" must be in [0.1, 10.0]!')
    if radius <= 0:
        raise ValueError(funcname + ': "radius" must be positive.')
    if radiusc <= 0:
        raise ValueError(funcname + ': "radiusc" must be positive.')
    if elast < 1:
        raise ValueError(funcname + ': Valid range of "elast" is [1, +inf)!')
    if elastc < 1:
        raise ValueError(funcname + ': Valid range of "elastc" is [1, +inf)!')
    if smode not in [0, 1, 2, 3, 4, 5]:
        raise ValueError(funcname + ': "smode" must be in [0, 1, 2, 3, 4, 5]!')
    if smode in [0, 3]:
        if radius not in list(range(2, 10)):
            raise ValueError(funcname + ': "radius" must be in 2-9 for smode=0 or 3 !')
        if radiusc not in list(range(2, 10)):
            raise ValueError(funcname + ': "radiusc" must be in 2-9 for smode=0 or 3 !')
    elif smode in [1, 4]:
        if radius not in list(range(1, 129)):
            raise ValueError(funcname + ': "radius" must be in 1-128 for smode=1 or smode=4 !')
        if radiusc not in list(range(1, 129)):
            raise ValueError(funcname + ': "radiusc" must be in 1-128 for smode=1 or smode=4 !')
    if thr_det <= 0.0:
        raise ValueError(funcname + ': "thr_det" must be positive!')

    ow = src.width
    oh = src.height

    src_16 = core.fmtc.bitdepth(src, bits=16, planes=planes) if src.format.bits_per_sample < 16 else src
    src_8 = core.fmtc.bitdepth(src, bits=8, dmode=1, planes=[0]) if src.format.bits_per_sample != 8 else src
    ref_16 = core.fmtc.bitdepth(ref, bits=16, planes=planes) if ref.format.bits_per_sample < 16 else ref

    # Do lineart smoothing first for sharper results
    if resizer.lower() == 'lineart_rpow2':
        src_16 = ProtectedDebiXAA(src_16, w, h, bicubic=False)
    elif resizer.lower() == 'lineart_rpow2_bicubic':
        src_16 = ProtectedDebiXAA(src_16, w, h, bicubic=True, b=b, c=c)

    # Main debanding
    chroma_flag = (thrc != thr or radiusc != radius or
                   elastc != elast) and 0 in planes and (1 in planes or 2 in planes)

    if chroma_flag:
        planes2 = [0] if 0 in planes else []
    else:
        planes2 = planes

    if not planes2:
        raise ValueError(funcname + ': no plane is processed')

    flt_y = smooth_mod(src_16, ref_16, smode, radius, thr, elast, planes2)
    if chroma_flag:
        flt_c = smooth_mod(src_16, ref_16, smode, radiusc, thrc, elastc, [x for x in planes if x != 0])
        flt = core.std.ShufflePlanes([flt_y,flt_c], [0,1,2], src.format.color_family)
    else:
        flt = flt_y

    # Edge/detail mask
    td_lo = max(thr_det * 0.75, 1.0)
    td_hi = max(thr_det, 1.0)
    mexpr = 'x {tl} - {th} {tl} - / 255 *'.format(tl=td_lo - 0.0001, th=td_hi + 0.0001)

    if mask > 0:
        dmask = GetPlane(src_8, 0)
        dmask = _Build_gf3_range_mask(dmask, mask)
        EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr
        dmask = EXPR([dmask], [mexpr])
        if hasattr(core,'zsmooth'):
          dmask = core.zsmooth.RemoveGrain(dmask, [22])
        else:
          dmask = core.rgvs.RemoveGrain(dmask, [22])
        if mask > 1:
            dmask = core.std.Convolution(dmask, matrix=[1,2,1,2,4,2,1,2,1])
            if mask > 2:
                dmask = core.std.Convolution(dmask, matrix=[1,1,1,1,1,1,1,1,1])
        dmask = core.fmtc.bitdepth(dmask, bits=16)
        res_16 = core.std.MaskedMerge(flt, src_16, dmask, planes=planes, first_plane=True)
    else:
        res_16 = flt

    # Resizing / colorspace conversion (GradFun3mod)
    res_16_y = core.std.ShufflePlanes(res_16, planes=0, colorfamily=vs.GRAY)
    if resizer.lower() == 'debilinear':
        rkernel = Resize(res_16_y if yuv444 else res_16, w, h, kernel='bilinear', invks=True)
    elif resizer.lower() == 'debicubic':
        rkernel = Resize(res_16_y if yuv444 else res_16, w, h, kernel='bicubic', a1=b, a2=c, invks=True)
    elif resizer.lower() == 'debilinearm':
        rkernel = DebilinearM(res_16_y if yuv444 else res_16, w, h, chroma=not yuv444)
    elif resizer.lower() == 'debicubicm':
        rkernel = DebicubicM(res_16_y if yuv444 else res_16, w, h, b=b, c=c, chroma=not yuv444)
    elif resizer.lower() in ('lineart_rpow2', 'lineart_rpow2_bicubic'):
        if yuv444:
            rkernel = Resize(res_16_y, w, h, kernel='spline36')
        else:
            rkernel = res_16
    elif not resizer:
        rkernel = res_16
    else:
       rkernel = Resize(res_16_y if yuv444 else res_16, w, h, kernel=resizer.lower())

    if yuv444:
        ly = rkernel
        lu = core.std.ShufflePlanes(res_16, planes=1, colorfamily=vs.GRAY)
        lv = core.std.ShufflePlanes(res_16, planes=2, colorfamily=vs.GRAY)
        lu = Resize(lu, w, h, kernel='spline16', sx=0.25)
        lv = Resize(lv, w, h, kernel='spline16', sx=0.25)
        rkernel = core.std.ShufflePlanes([ly,lu,lv], planes=[0,0,0], colorfamily=vs.YUV)
    res_16 = rkernel

    # Dithering
    result = res_16 if bits == 16 else core.fmtc.bitdepth(res_16, bits=bits, planes=planes, dmode=mode, ampo=ampo,
                                                          ampn=ampn, dyn=dyn, staticnoise=staticnoise, patsize=pat)

    if debug:
        last = dmask
        if bits != 16:
            last = core.fmtc.bitdepth(last, bits=bits)
    else:
        last = result
    return last
    
    
    
# Helper


# Taken from mvsfunc
################################################################################################################################
## Utility function: LimitFilter()
################################################################################################################################
## Similar to the AviSynth function Dither_limit_dif16() and HQDeringmod_limit_dif16().
## It acts as a post-processor, and is very useful to limit the difference of filtering while avoiding artifacts.
## Commonly used cases:
##     de-banding
##     de-ringing
##     de-noising
##     sharpening
##     combining high precision source with low precision filtering: LimitFilter(src, flt, thr=1.0, elast=2.0)
################################################################################################################################
## There are 2 implementations, default one with std.Expr, the other with std.Lut.
## The Expr version supports all mode, while the Lut version doesn't support float input and ref clip.
## Also the Lut version will truncate the filtering diff if it exceeds half the value range(128 for 8-bit, 32768 for 16-bit).
## The Lut version might be faster than Expr version in some cases, for example 8-bit input and brighten_thr != thr.
################################################################################################################################
## Algorithm for Y/R/G/B plane (for chroma, replace "thr" and "brighten_thr" with "thrc")
##     dif = flt - src
##     dif_ref = flt - ref
##     dif_abs = abs(dif_ref)
##     thr_1 = brighten_thr if (dif > 0) else thr
##     thr_2 = thr_1 * elast
##
##     if dif_abs <= thr_1:
##         final = flt
##     elif dif_abs >= thr_2:
##         final = src
##     else:
##         final = src + dif * (thr_2 - dif_abs) / (thr_2 - thr_1)
################################################################################################################################
## Basic parameters
##     flt {clip}: filtered clip, to compute the filtering diff
##         can be of YUV/RGB/Gray color family, can be of 8-16 bit integer or 16/32 bit float
##     src {clip}: source clip, to apply the filtering diff
##         must be of the same format and dimension as "flt"
##     ref {clip} (optional): reference clip, to compute the weight to be applied on filtering diff
##         must be of the same format and dimension as "flt"
##         default: None (use "src")
##     thr {float}: threshold (8-bit scale) to limit filtering diff
##         default: 1.0
##     elast {float}: elasticity of the soft threshold
##         default: 2.0
##     planes {int[]}: specify which planes to process
##         unprocessed planes will be copied from "flt"
##         default: all planes will be processed, [0,1,2] for YUV/RGB input, [0] for Gray input
################################################################################################################################
## Advanced parameters
##     brighten_thr {float}: threshold (8-bit scale) for filtering diff that brightening the image (Y/R/G/B plane)
##         set a value different from "thr" is useful to limit the overshoot/undershoot/blurring introduced in sharpening/de-ringing
##         default is the same as "thr"
##     thrc {float}: threshold (8-bit scale) for chroma (U/V/Co/Cg plane)
##         default is the same as "thr"
##     force_expr {bool}
##         - True: force to use the std.Expr implementation
##         - False: use the std.Lut implementation if available
##         default: True
################################################################################################################################
def LimitFilter(flt, src, ref=None, thr=None, elast=None, brighten_thr=None, thrc=None, force_expr=None, planes=None):
    # input clip
    if not isinstance(flt, vs.VideoNode):
        raise type_error('"flt" must be a clip!')
    if not isinstance(src, vs.VideoNode):
        raise type_error('"src" must be a clip!')
    if ref is not None and not isinstance(ref, vs.VideoNode):
        raise type_error('"ref" must be a clip!')

    # Get properties of input clip
    sFormat = flt.format
    if sFormat.id != src.format.id:
        raise value_error('"flt" and "src" must be of the same format!')
    if flt.width != src.width or flt.height != src.height:
        raise value_error('"flt" and "src" must be of the same width and height!')

    if ref is not None:
        if sFormat.id != ref.format.id:
            raise value_error('"flt" and "ref" must be of the same format!')
        if flt.width != ref.width or flt.height != ref.height:
            raise value_error('"flt" and "ref" must be of the same width and height!')

    sColorFamily = sFormat.color_family
    CheckColorFamily(sColorFamily)
    sIsYUV = sColorFamily == vs.YUV

    sSType = sFormat.sample_type
    sbitPS = sFormat.bits_per_sample
    sNumPlanes = sFormat.num_planes

    # Parameters
    if thr is None:
        thr = 1.0
    elif isinstance(thr, int) or isinstance(thr, float):
        if thr < 0:
            raise value_error('valid range of "thr" is [0, +inf)')
    else:
        raise type_error('"thr" must be an int or a float!')

    if elast is None:
        elast = 2.0
    elif isinstance(elast, int) or isinstance(elast, float):
        if elast < 1:
            raise value_error('valid range of "elast" is [1, +inf)')
    else:
        raise type_error('"elast" must be an int or a float!')

    if brighten_thr is None:
        brighten_thr = thr
    elif isinstance(brighten_thr, int) or isinstance(brighten_thr, float):
        if brighten_thr < 0:
            raise value_error('valid range of "brighten_thr" is [0, +inf)')
    else:
        raise type_error('"brighten_thr" must be an int or a float!')

    if thrc is None:
        thrc = thr
    elif isinstance(thrc, int) or isinstance(thrc, float):
        if thrc < 0:
            raise value_error('valid range of "thrc" is [0, +inf)')
    else:
        raise type_error('"thrc" must be an int or a float!')

    if force_expr is None:
        force_expr = True
    elif not isinstance(force_expr, int):
        raise type_error('"force_expr" must be a bool!')
    if ref is not None or sSType != vs.INTEGER:
        force_expr = True

    VSMaxPlaneNum = 3
    # planes
    process = [0 for i in range(VSMaxPlaneNum)]

    if planes is None:
        process = [1 for i in range(VSMaxPlaneNum)]
    elif isinstance(planes, int):
        if planes < 0 or planes >= VSMaxPlaneNum:
            raise value_error(f'valid range of "planes" is [0, {VSMaxPlaneNum})!')
        process[planes] = 1
    elif isinstance(planes, Sequence):
        for p in planes:
            if not isinstance(p, int):
                raise type_error('"planes" must be a (sequence of) int!')
            elif p < 0 or p >= VSMaxPlaneNum:
                raise value_error(f'valid range of "planes" is [0, {VSMaxPlaneNum})!')
            process[p] = 1
    else:
        raise type_error('"planes" must be a (sequence of) int!')

    # Process
    if thr <= 0 and brighten_thr <= 0:
        if sIsYUV:
            if thrc <= 0:
                return src
        else:
            return src
    if thr >= 255 and brighten_thr >= 255:
        if sIsYUV:
            if thrc >= 255:
                return flt
        else:
            return flt
    if thr >= 128 or brighten_thr >= 128:
        force_expr = True

    if force_expr: # implementation with std.Expr
        valueRange = (1 << sbitPS) - 1 if sSType == vs.INTEGER else 1
        limitExprY = _limit_filter_expr(ref is not None, thr, elast, brighten_thr, valueRange)
        limitExprC = _limit_filter_expr(ref is not None, thrc, elast, thrc, valueRange)
        expr = []
        for i in range(sNumPlanes):
            if process[i]:
                if i > 0 and (sIsYUV):
                    expr.append(limitExprC)
                else:
                    expr.append(limitExprY)
            else:
                expr.append("")
        EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr
        if ref is None:
            clip = EXPR([flt, src], expr)
        else:
            clip = EXPR([flt, src, ref], expr)
    else: # implementation with std.MakeDiff, std.Lut and std.MergeDiff
        diff = core.std.MakeDiff(flt, src, planes=planes)
        if sIsYUV:
            if process[0]:
                diff = _limit_diff_lut(diff, thr, elast, brighten_thr, [0])
            if process[1] or process[2]:
                _planes = []
                if process[1]:
                    _planes.append(1)
                if process[2]:
                    _planes.append(2)
                diff = _limit_diff_lut(diff, thrc, elast, thrc, _planes)
        else:
            diff = _limit_diff_lut(diff, thr, elast, brighten_thr, planes)
        clip = core.std.MakeDiff(flt, diff, planes=planes)

    # Output
    return clip
################################################################################################################################


# Taken from muvsfunc
def GetPlane(clip, plane=None):
    # input clip
    if not isinstance(clip, vs.VideoNode):
        raise type_error('"clip" must be a clip!')

    # Get properties of input clip
    sFormat = clip.format
    sNumPlanes = sFormat.num_planes

    # Parameters
    if plane is None:
        plane = 0
    elif not isinstance(plane, int):
        raise type_error('"plane" must be an int!')
    elif plane < 0 or plane > sNumPlanes:
        raise value_error(f'valid range of "plane" is [0, {sNumPlanes})!')

    # Process
    return core.std.ShufflePlanes(clip, plane, vs.GRAY)
    
################################################################################################################################
## Helper function: CheckColorFamily()
################################################################################################################################
def CheckColorFamily(color_family, valid_list=None, invalid_list=None):
    if valid_list is None:
        valid_list = ('RGB', 'YUV', 'GRAY')
    if invalid_list is None:
        invalid_list = ('COMPAT', 'UNDEFINED')
    # check invalid list
    for cf in invalid_list:
        if color_family == getattr(vs, cf, None):
            raise value_error(f'color family *{cf}* is not supported!')
    # check valid list
    if valid_list:
        if color_family not in [getattr(vs, cf, None) for cf in valid_list]:
            raise value_error(f'color family not supported, only {valid_list} are accepted')
################################################################################################################################
################################################################################################################################
## Internal used functions for LimitFilter()
################################################################################################################################
def _limit_filter_expr(defref, thr, elast, largen_thr, value_range):
    flt = " x "
    src = " y "
    ref = " z " if defref else src

    dif = f" {flt} {src} - "
    dif_ref = f" {flt} {ref} - "
    dif_abs = dif_ref + " abs "

    thr = thr * value_range / 255
    largen_thr = largen_thr * value_range / 255

    if thr <= 0 and largen_thr <= 0:
        limitExpr = f" {src} "
    elif thr >= value_range and largen_thr >= value_range:
        limitExpr = ""
    else:
        if thr <= 0:
            limitExpr = f" {src} "
        elif thr >= value_range:
            limitExpr = f" {flt} "
        elif elast <= 1:
            limitExpr = f" {dif_abs} {thr} <= {flt} {src} ? "
        else:
            thr_1 = thr
            thr_2 = thr * elast
            thr_slope = 1 / (thr_2 - thr_1)
            # final = src + dif * (thr_2 - dif_abs) / (thr_2 - thr_1)
            limitExpr = f" {src} {dif} {thr_2} {dif_abs} - * {thr_slope} * + "
            limitExpr = f" {dif_abs} {thr_1} <= {flt} {dif_abs} {thr_2} >= {src} " + limitExpr + " ? ? "

        if largen_thr != thr:
            if largen_thr <= 0:
                limitExprLargen = f" {src} "
            elif largen_thr >= value_range:
                limitExprLargen = f" {flt} "
            elif elast <= 1:
                limitExprLargen = f" {dif_abs} {largen_thr} <= {flt} {src} ? "
            else:
                thr_1 = largen_thr
                thr_2 = largen_thr * elast
                thr_slope = 1 / (thr_2 - thr_1)
                # final = src + dif * (thr_2 - dif_abs) / (thr_2 - thr_1)
                limitExprLargen = f" {src} {dif} {thr_2} {dif_abs} - * {thr_slope} * + "
                limitExprLargen = f" {dif_abs} {thr_1} <= {flt} {dif_abs} {thr_2} >= {src} " + limitExprLargen + " ? ? "
            limitExpr = f" {flt} {ref} > " + limitExprLargen + " " + limitExpr + " ? "

    return limitExpr
################################################################################################################################


################################################################################################################################
## Internal used functions for LimitFilter()
################################################################################################################################
def _limit_diff_lut(diff, thr, elast, largen_thr, planes):
    # input clip
    if not isinstance(diff, vs.VideoNode):
        raise type_error('"diff" must be a clip!', num_stacks=2)

    # Get properties of input clip
    sFormat = diff.format

    sSType = sFormat.sample_type
    sbitPS = sFormat.bits_per_sample

    if sSType == vs.INTEGER:
        neutral = 1 << (sbitPS - 1)
        value_range = (1 << sbitPS) - 1
    else:
        neutral = 0
        value_range = 1
        raise value_error('"diff" must be an int!', num_stacks=2)

    # Process
    thr = thr * value_range / 255
    largen_thr = largen_thr * value_range / 255
    '''
    # for std.MergeDiff(src, limitedDiff)
    if thr <= 0 and largen_thr <= 0:
        def limitLut(x):
            return neutral
        return core.std.Lut(diff, planes=planes, function=limitLut)
    elif thr >= value_range / 2 and largen_thr >= value_range / 2:
        return diff
    elif elast <= 1:
        def limitLut(x):
            dif = x - neutral
            dif_abs = abs(dif)
            thr_1 = largen_thr if dif > 0 else thr
            return x if dif_abs <= thr_1 else neutral
        return core.std.Lut(diff, planes=planes, function=limitLut)
    else:
        def limitLut(x):
            dif = x - neutral
            dif_abs = abs(dif)
            thr_1 = largen_thr if dif > 0 else thr
            thr_2 = thr_1 * elast
            thr_slope = 1 / (thr_2 - thr_1)

            if dif_abs <= thr_1:
                return x
            elif dif_abs >= thr_2:
                return neutral
            else:
                # final = src - dif * ((dif_abs - thr_1) / (thr_2 - thr_1) - 1)
                return round(dif * (thr_2 - dif_abs) * thr_slope + neutral)
    '''
    # for std.MakeDiff(flt, limitedDiff)
    if thr <= 0 and largen_thr <= 0:
        return diff
    elif thr >= value_range / 2 and largen_thr >= value_range / 2:
        def limitLut(x):
            return neutral
        return core.std.Lut(diff, planes=planes, function=limitLut)
    elif elast <= 1:
        def limitLut(x):
            dif = x - neutral
            dif_abs = abs(dif)
            thr_1 = largen_thr if dif > 0 else thr
            return neutral if dif_abs <= thr_1 else x
        return core.std.Lut(diff, planes=planes, function=limitLut)
    else:
        def limitLut(x):
            dif = x - neutral
            dif_abs = abs(dif)
            thr_1 = largen_thr if dif > 0 else thr
            thr_2 = thr_1 * elast

            if dif_abs <= thr_1:
                return neutral
            elif dif_abs >= thr_2:
                return x
            else:
                # final = flt - dif * (dif_abs - thr_1) / (thr_2 - thr_1)
                thr_slope = 1 / (thr_2 - thr_1)
                return round(dif * (dif_abs - thr_1) * thr_slope + neutral)
        return core.std.Lut(diff, planes=planes, function=limitLut)
################################################################################################################################

def _GF3_smoothgrad_multistage(src: vs.VideoNode, ref: vs.VideoNode, radius: int,
                               thr: float, elast: float, planes: Optional[Union[int, Sequence[int]]]
                               ) -> vs.VideoNode:
    ela_2 = max(elast * 0.83, 1.0)
    ela_3 = max(elast * 0.67, 1.0)
    r2 = radius * 2 // 3
    r3 = radius * 3 // 3
    r4 = radius * 4 // 4
    last = src
    last = SmoothGrad(last, radius=r2, thr=thr, elast=elast, ref=ref, planes=planes) if r2 >= 1 else last
    last = SmoothGrad(last, radius=r3, thr=thr * 0.7, elast=ela_2, ref=ref, planes=planes) if r3 >= 1 else last
    last = SmoothGrad(last, radius=r4, thr=thr * 0.46, elast=ela_3, ref=ref, planes=planes) if r4 >= 1 else last
    return last


def _GF3_smoothgrad_multistage_3(src: vs.VideoNode, radius: int, thr: float,
                                 elast: float, planes: Optional[Union[int, Sequence[int]]]
                                 ) -> vs.VideoNode:
    ref = SmoothGrad(src, radius=radius // 3, thr=thr * 0.8, elast=elast)
    last = BoxFilter(src, radius=radius, planes=planes)
    last = BoxFilter(last, radius=radius, planes=planes)
    last = LimitFilter(last, src, thr=thr * 0.6, elast=elast, ref=ref, planes=planes)
    return last


def _GF3_dfttest(src: vs.VideoNode, ref: vs.VideoNode, radius: int,
                 thr: float, elast: float, planes: Optional[Union[int, Sequence[int]]]
                 ) -> vs.VideoNode:
    hrad = max(radius * 3 // 4, 1)
    last = core.dfttest.DFTTest(src, sigma=hrad * thr * thr * 32, sbsize=hrad * 4,
                                sosize=hrad * 3, tbsize=1, planes=planes)
    last = LimitFilter(last, ref, thr=thr, elast=elast, planes=planes)

    return last


def _GF3_bilateral_multistage(src: vs.VideoNode, ref: vs.VideoNode, radius: int,
                              thr: float, elast: float, planes: Optional[Union[int, Sequence[int]]]
                              ) -> vs.VideoNode:
    last = core.bilateral.Bilateral(src, ref=ref, sigmaS=radius / 2, sigmaR=thr / 255, planes=planes, algorithm=0)

    last = LimitFilter(last, src, thr=thr, elast=elast, planes=planes)

    return last


def _Build_gf3_range_mask(src: vs.VideoNode, radius: int = 1) -> vs.VideoNode:
    last = src
    EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr
    if radius > 1:
        ma = mt_expand_multi(last, mode='ellipse', planes=[0], sw=radius, sh=radius)
        mi = mt_inpand_multi(last, mode='ellipse', planes=[0], sw=radius, sh=radius)
        last = EXPR([ma, mi], ['x y -'])
    else:
        bits = src.format.bits_per_sample
        black = 0
        white = (1 << bits) - 1
        maxi = core.std.Maximum(last, [0])
        mini = core.std.Minimum(last, [0])
        exp = "x y -"
        exp2 = "x {thY1} < {black} x ? {thY2} > {white} x ?".format(thY1=0, thY2=255, black=black, white=white)
        last = EXPR([maxi,mini], [exp])
        last = EXPR([last], [exp2])

    return last

# Taken from havsfunc
def mt_expand_multi(src: vs.VideoNode, mode: str = 'rectangle', planes: Optional[Union[int, Sequence[int]]] = None, sw: int = 1, sh: int = 1) -> vs.VideoNode:
    '''
    Calls std.Maximum multiple times in order to grow the mask from the desired width and height.

    Parameters:
        src: Clip to process.

        mode: "rectangle", "ellipse" or "losange". Ellipses are actually combinations of rectangles and losanges and look more like octogons.
            Losanges are truncated (not scaled) when sw and sh are not equal.

        planes: Specifies which planes will be processed. Any unprocessed planes will be simply copied.

        sw: Growing shape width. 0 is allowed.

        sh: Growing shape height. 0 is allowed.
    '''
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('mt_expand_multi: this is not a clip')

    if sw > 0 and sh > 0:
        mode_m = [0, 1, 0, 1, 1, 0, 1, 0] if mode == 'losange' or (mode == 'ellipse' and (sw % 3) != 1) else [1, 1, 1, 1, 1, 1, 1, 1]
    elif sw > 0:
        mode_m = [0, 0, 0, 1, 1, 0, 0, 0]
    elif sh > 0:
        mode_m = [0, 1, 0, 0, 0, 0, 1, 0]
    else:
        mode_m = None

    if mode_m is not None:
        src = mt_expand_multi(src.std.Maximum(planes=planes, coordinates=mode_m), mode=mode, planes=planes, sw=sw - 1, sh=sh - 1)
    return src

# Taken from havsfunc
def mt_inpand_multi(src: vs.VideoNode, mode: str = 'rectangle', planes: Optional[Union[int, Sequence[int]]] = None, sw: int = 1, sh: int = 1) -> vs.VideoNode:
    '''
    Calls std.Minimum multiple times in order to shrink the mask from the desired width and height.

    Parameters:
        src: Clip to process.

        mode: "rectangle", "ellipse" or "losange". Ellipses are actually combinations of rectangles and losanges and look more like octogons.
            Losanges are truncated (not scaled) when sw and sh are not equal.

        planes: Specifies which planes will be processed. Any unprocessed planes will be simply copied.

        sw: Shrinking shape width. 0 is allowed.

        sh: Shrinking shape height. 0 is allowed.
    '''
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('mt_inpand_multi: this is not a clip')

    if sw > 0 and sh > 0:
        mode_m = [0, 1, 0, 1, 1, 0, 1, 0] if mode == 'losange' or (mode == 'ellipse' and (sw % 3) != 1) else [1, 1, 1, 1, 1, 1, 1, 1]
    elif sw > 0:
        mode_m = [0, 0, 0, 1, 1, 0, 0, 0]
    elif sh > 0:
        mode_m = [0, 1, 0, 0, 0, 0, 1, 0]
    else:
        mode_m = None

    if mode_m is not None:
        src = mt_inpand_multi(src.std.Minimum(planes=planes, coordinates=mode_m), mode=mode, planes=planes, sw=sw - 1, sh=sh - 1)
    return src
