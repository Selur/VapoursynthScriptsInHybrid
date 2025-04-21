from vapoursynth import core
import vapoursynth as vs

import math
import importlib
from functools import partial
from typing import Optional, Union, Sequence

################################################################################################
###                                                                                          ###
###                       LimitedSharpenFaster MOD : function LSFmod()                       ###
###                                                                                          ###
###                                Modded Version by LaTo INV.                               ###
###                                                                                          ###
###                                  v1.9 - 05 October 2009                                  ###
###                                                                                          ###
################################################################################################
###
### +--------------+
### | DEPENDENCIES |
### +--------------+
###
### -> RGVS
### -> CAS
###
###
###
### +---------+
### | GENERAL |
### +---------+
###
### strength [int]
### --------------
### Strength of the sharpening
###
### Smode [int: 1,2,3]
### ----------------------
### Sharpen mode:
###    =1 : Range sharpening
###    =2 : Nonlinear sharpening (corrected version)
###    =3 : Contrast Adaptive Sharpening
###
### Smethod [int: 1,2,3]
### --------------------
### Sharpen method: (only used in Smode=1,2)
###    =1 : 3x3 kernel
###    =2 : Min/Max
###    =3 : Min/Max + 3x3 kernel
###
### kernel [int: 11,12,19,20]
### -------------------------
### Kernel used in Smethod=1&3
### In strength order: + 19 > 12 >> 20 > 11 -
###
###
###
### +---------+
### | SPECIAL |
### +---------+
###
### preblur [int: 0,1,2,3]
### --------------------------------
### Mode to avoid noise sharpening & ringing:
###    =-1 : No preblur
###    = 0 : MinBlur(0)
###    = 1 : MinBlur(1)
###    = 2 : MinBlur(2)
###    = 3 : DFTTest
###
### secure [bool]
### -------------
### Mode to avoid banding & oil painting (or face wax) effect of sharpening
###
### source [clip]
### -------------
### If source is defined, LSFmod doesn't sharp more a denoised clip than this source clip
### In this mode, you can safely set Lmode=0 & PP=off
###    Usage:   denoised.LSFmod(source=source)
###    Example: last.FFT3DFilter().LSFmod(source=last,Lmode=0,soft=0)
###
###
###
### +----------------------+
### | NONLINEAR SHARPENING |
### +----------------------+
###
### Szrp [int]
### ----------
### Zero Point:
###    - differences below Szrp are amplified (overdrive sharpening)
###    - differences above Szrp are reduced   (reduced sharpening)
###
### Spwr [int]
### ----------
### Power: exponent for sharpener
###
### SdmpLo [int]
### ------------
### Damp Low: reduce sharpening for small changes [0:disable]
###
### SdmpHi [int]
### ------------
### Damp High: reduce sharpening for big changes [0:disable]
###
###
###
### +----------+
### | LIMITING |
### +----------+
###
### Lmode [int: ...,0,1,2,3,4]
### --------------------------
### Limit mode:
###    <0 : Limit with repair (ex: Lmode=-1 --> repair(1), Lmode=-5 --> repair(5)...)
###    =0 : No limit
###    =1 : Limit to over/undershoot
###    =2 : Limit to over/undershoot on edges and no limit on not-edges
###    =3 : Limit to zero on edges and to over/undershoot on not-edges
###    =4 : Limit to over/undershoot on edges and to over/undershoot2 on not-edges
###
### overshoot [int]
### ---------------
### Limit for pixels that get brighter during sharpening
###
### undershoot [int]
### ----------------
### Limit for pixels that get darker during sharpening
###
### overshoot2 [int]
### ----------------
### Same as overshoot, only for Lmode=4
###
### undershoot2 [int]
### -----------------
### Same as undershoot, only for Lmode=4
###
###
###
### +-----------------+
### | POST-PROCESSING |
### +-----------------+
###
### soft [int: -2,-1,0...100]
### -------------------------
### Soft the sharpening effect (-1 = old autocalculate, -2 = new autocalculate)
###
### soothe [bool]
### -------------
###    =True  : Enable soothe temporal stabilization
###    =False : Disable soothe temporal stabilization
###
### keep [int: 0...100]
### -------------------
### Minimum percent of the original sharpening to keep (only with soothe=True)
###
###
###
### +-------+
### | EDGES |
### +-------+
###
### edgemode [int: -1,0,1,2]
### ------------------------
###    =-1 : Show edgemask
###    = 0 : Sharpening all
###    = 1 : Sharpening only edges
###    = 2 : Sharpening only not-edges
###
### edgemaskHQ [bool]
### -----------------
###    =True  : Original edgemask
###    =False : Faster edgemask
###
###
###
### +------------+
### | UPSAMPLING |
### +------------+
###
### ss_x ; ss_y [float]
### -------------------
### Supersampling factor (reduce aliasing on edges)
###
### dest_x ; dest_y [int]
### ---------------------
### Output resolution after sharpening (avoid a resizing step)
###
###
###
### +----------+
### | SETTINGS |
### +----------+
###
### defaults [string: "old" or "slow" or "fast"]
### --------------------------------------------
###    = "old"  : Reset settings to original version (output will be THE SAME AS LSF)
###    = "slow" : Enable SLOW modded version settings
###    = "fast" : Enable FAST modded version settings
###  --> /!\ [default:"fast"]
###
###
### defaults="old" :  - strength    = 100
### ----------------  - Smode       = 1
###                   - Smethod     = Smode==1?2:1
###                   - kernel      = 11
###
###                   - preblur     = -1
###                   - secure      = false
###                   - source      = undefined
###
###                   - Szrp        = 16
###                   - Spwr        = 2
###                   - SdmpLo      = strength/25
###                   - SdmpHi      = 0
###
###                   - Lmode       = 1
###                   - overshoot   = 1
###                   - undershoot  = overshoot
###                   - overshoot2  = overshoot*2
###                   - undershoot2 = overshoot2
###
###                   - soft        = 0
###                   - soothe      = false
###                   - keep        = 25
###
###                   - edgemode    = 0
###                   - edgemaskHQ  = true
###
###                   - ss_x        = Smode==1?1.50:1.25
###                   - ss_y        = ss_x
###                   - dest_x      = ox
###                   - dest_y      = oy
###
###
### defaults="slow" : - strength    = 100
### ----------------- - Smode       = 2
###                   - Smethod     = 3
###                   - kernel      = 11
###
###                   - preblur     = -1
###                   - secure      = true
###                   - source      = undefined
###
###                   - Szrp        = 16
###                   - Spwr        = 4
###                   - SdmpLo      = 4
###                   - SdmpHi      = 48
###
###                   - Lmode       = 4
###                   - overshoot   = strength/100
###                   - undershoot  = overshoot
###                   - overshoot2  = overshoot*2
###                   - undershoot2 = overshoot2
###
###                   - soft        = -2
###                   - soothe      = true
###                   - keep        = 20
###
###                   - edgemode    = 0
###                   - edgemaskHQ  = true
###
###                   - ss_x        = Smode==3?1.00:1.50
###                   - ss_y        = ss_x
###                   - dest_x      = ox
###                   - dest_y      = oy
###
###
### defaults="fast" : - strength    = 80
### ----------------- - Smode       = 3
###                   - Smethod     = 2
###                   - kernel      = 11
###
###                   - preblur     = 0
###                   - secure      = true
###                   - source      = undefined
###
###                   - Szrp        = 16
###                   - Spwr        = 4
###                   - SdmpLo      = 4
###                   - SdmpHi      = 48
###
###                   - Lmode       = 0
###                   - overshoot   = strength/100
###                   - undershoot  = overshoot
###                   - overshoot2  = overshoot*2
###                   - undershoot2 = overshoot2
###
###                   - soft        = 0
###                   - soothe      = false
###                   - keep        = 20
###
###                   - edgemode    = 0
###                   - edgemaskHQ  = false
###
###                   - ss_x        = Smode==3?1.00:1.25
###                   - ss_y        = ss_x
###                   - dest_x      = ox
###                   - dest_y      = oy
###
################################################################################################
def LSFmod(input, strength=None, Smode=None, Smethod=None, kernel=11, preblur=None, secure=None, source=None, Szrp=16, Spwr=None, SdmpLo=None, SdmpHi=None, Lmode=None, overshoot=None, undershoot=None,
           overshoot2=None, undershoot2=None, soft=None, soothe=None, keep=None, edgemode=0, edgemaskHQ=None, ss_x=None, ss_y=None, dest_x=None, dest_y=None, defaults='fast', cuda=False):
    if not isinstance(input, vs.VideoNode):
        raise vs.Error('LSFmod: this is not a clip')

    if input.format.color_family == vs.RGB:
        raise vs.Error('LSFmod: RGB format is not supported')

    if source is not None and (not isinstance(source, vs.VideoNode) or source.format.id != input.format.id):
        raise vs.Error("LSFmod: 'source' must be the same format as input")

    isGray = (input.format.color_family == vs.GRAY)
    isInteger = (input.format.sample_type == vs.INTEGER)

    if isInteger:
        neutral = 1 << (input.format.bits_per_sample - 1)
        peak = (1 << input.format.bits_per_sample) - 1
        factor = 1 << (input.format.bits_per_sample - 8)
    else:
        neutral = 0.0
        peak = 1.0
        factor = 255.0

    ### DEFAULTS
    try:
        num = ['old', 'slow', 'fast'].index(defaults.lower())
    except:
        raise vs.Error('LSFmod: defaults must be "old" or "slow" or "fast"')

    ox = input.width
    oy = input.height

    if strength is None:
        strength = [100, 100, 80][num]
    if Smode is None:
        Smode = [1, 2, 3][num]
    if Smethod is None:
        Smethod = [2 if Smode == 1 else 1, 3, 2][num]
    if preblur is None:
        preblur = [-1, -1, 0][num]
    if secure is None:
        secure = [False, True, True][num]
    if Spwr is None:
        Spwr = [2, 4, 4][num]
    if SdmpLo is None:
        SdmpLo = [strength // 25, 4, 4][num]
    if SdmpHi is None:
        SdmpHi = [0, 48, 48][num]
    if Lmode is None:
        Lmode = [1, 4, 0][num]
    if overshoot is None:
        overshoot = [1, strength // 100, strength // 100][num]
    if undershoot is None:
        undershoot = overshoot
    if overshoot2 is None:
        overshoot2 = overshoot * 2
    if undershoot2 is None:
        undershoot2 = overshoot2
    if soft is None:
        soft = [0, -2, 0][num]
    if soothe is None:
        soothe = [False, True, False][num]
    if keep is None:
        keep = [25, 20, 20][num]
    if edgemaskHQ is None:
        edgemaskHQ = [True, True, False][num]
    if ss_x is None:
        ss_x = [1.5 if Smode == 1 else 1.25, 1.0 if Smode == 3 else 1.5, 1.0 if Smode == 3 else 1.25][num]
    if ss_y is None:
        ss_y = ss_x
    if dest_x is None:
        dest_x = ox
    if dest_y is None:
        dest_y = oy

    if kernel == 4:
        RemoveGrain = partial(core.std.Median)
    elif kernel in [11, 12]:
        RemoveGrain = partial(core.std.Convolution, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    elif kernel == 19:
        RemoveGrain = partial(core.std.Convolution, matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1])
    elif kernel == 20:
        RemoveGrain = partial(core.std.Convolution, matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])
    else:
        RG = core.zsmooth.RemoveGrain if hasattr(core,'zsmooth') else core.rgvs.RemoveGrain
        RemoveGrain = partial(RG, mode=[kernel])

    if soft == -1:
        soft = math.sqrt(((ss_x + ss_y) / 2 - 1) * 100) * 10
    elif soft <= -2:
        soft = int((1 + 2 / (ss_x + ss_y)) * math.sqrt(strength))
    soft = min(soft, 100)

    xxs = cround(ox * ss_x / 8) * 8
    yys = cround(oy * ss_y / 8) * 8

    Str = strength / 100

    ### SHARP
    if ss_x > 1 or ss_y > 1:
        tmp = input.resize.Spline36(xxs, yys)
    else:
        tmp = input

    if not isGray:
        tmp_orig = tmp
        tmp = GetPlane(tmp, 0)

    if preblur <= -1:
        pre = tmp
    elif preblur >= 3:
        expr = 'x {i} < {peak} x {j} > 0 {peak} x {i} - {peak} {j} {i} - / * - ? ?'.format(i=scale(16, peak), j=scale(75, peak), peak=peak)
        
        if cuda:
          if hasattr(core,'dfttest2_nvrtc'):
            import dfttest2
            pre = core.std.MaskedMerge(dfttest2.DFTTest(tmp, tbsize=1, slocation=[0.0,4.0, 0.2,9.0, 1.0,15.0], backend=dfttest2.Backend.NVRTC), tmp, tmp.std.Expr(expr=[expr]))
          else:
            pre = core.std.MaskedMerge(tmp.dfttest.DFTTest(tbsize=1, slocation=[0.0,4.0, 0.2,9.0, 1.0,15.0]), tmp, tmp.std.Expr(expr=[expr]))
        else:
          pre = core.std.MaskedMerge(tmp.dfttest.DFTTest(tbsize=1, slocation=[0.0,4.0, 0.2,9.0, 1.0,15.0]), tmp, tmp.std.Expr(expr=[expr]))
    else:
        pre = MinBlur(tmp, r=preblur)

    dark_limit = pre.std.Minimum()
    bright_limit = pre.std.Maximum()

    if Smode < 3:
        if Smethod <= 1:
            method = RemoveGrain(pre)
        elif Smethod == 2:
            method = core.std.Merge(dark_limit, bright_limit)
        else:
            method = RemoveGrain(core.std.Merge(dark_limit, bright_limit))

        if secure:
            method = core.std.Expr([method, pre], expr=['x y < x {i} + x y > x {i} - x ? ?'.format(i=scale(1, peak))])

        if preblur > -1:
            method = core.std.MakeDiff(tmp, core.std.MakeDiff(pre, method))

        if Smode <= 1:
            normsharp = core.std.Expr([tmp, method], expr=[f'x x y - {Str} * +'])
        else:
            tmpScaled = tmp.std.Expr(expr=[f'x {1 / factor if isInteger else factor} *'], format=tmp.format.replace(sample_type=vs.FLOAT, bits_per_sample=32))
            methodScaled = method.std.Expr(expr=[f'x {1 / factor if isInteger else factor} *'], format=method.format.replace(sample_type=vs.FLOAT, bits_per_sample=32))
            expr = f'x y = x x x y - abs {Szrp} / {1 / Spwr} pow {Szrp} * {Str} * x y - dup abs / * x y - dup * {Szrp * Szrp} {SdmpLo} + * x y - dup * {SdmpLo} + {Szrp * Szrp} * / * 1 {SdmpHi} 0 = 0 {(Szrp / SdmpHi) ** 4} ? + 1 {SdmpHi} 0 = 0 x y - abs {SdmpHi} / 4 pow ? + / * + ? {factor if isInteger else 1 / factor} *'
            normsharp = core.std.Expr([tmpScaled, methodScaled], expr=[expr], format=tmp.format)
    else:
        normsharp = pre.cas.CAS(sharpness=min(Str, 1))

        if secure:
            normsharp = core.std.Expr([normsharp, pre], expr=['x y < x {i} + x y > x {i} - x ? ?'.format(i=scale(1, peak))])

        if preblur > -1:
            normsharp = core.std.MakeDiff(tmp, core.std.MakeDiff(pre, normsharp))

    ### LIMIT
    normal = mt_clamp(normsharp, bright_limit, dark_limit, scale(overshoot, peak), scale(undershoot, peak))
    second = mt_clamp(normsharp, bright_limit, dark_limit, scale(overshoot2, peak), scale(undershoot2, peak))
    zero = mt_clamp(normsharp, bright_limit, dark_limit, 0, 0)

    if edgemaskHQ:
        edge = tmp.std.Sobel(scale=2)
    else:
        edge = core.std.Expr([tmp.std.Maximum(), tmp.std.Minimum()], expr=['x y -'])
    edge = edge.std.Expr(expr=[f'x {1 / factor if isInteger else factor} * {128 if edgemaskHQ else 32} / 0.86 pow 255 * {factor if isInteger else 1 / factor} *'])

    if Lmode < 0:
      if hasattr(core,'zsmooth'):
        limit1 = core.zsmooth.Repair(normsharp, tmp, mode=[abs(Lmode)])
      else:
        limit1 = core.rgvs.Repair(normsharp, tmp, mode=[abs(Lmode)])
    elif Lmode == 0:
        limit1 = normsharp
    elif Lmode == 1:
        limit1 = normal
    elif Lmode == 2:
        limit1 = core.std.MaskedMerge(normsharp, normal, edge.std.Inflate())
    elif Lmode == 3:
        limit1 = core.std.MaskedMerge(normal, zero, edge.std.Inflate())
    else:
        limit1 = core.std.MaskedMerge(second, normal, edge.std.Inflate())

    if edgemode <= 0:
        limit2 = limit1
    elif edgemode == 1:
        limit2 = core.std.MaskedMerge(tmp, limit1, edge.std.Inflate().std.Inflate().std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1]))
    else:
        limit2 = core.std.MaskedMerge(limit1, tmp, edge.std.Inflate().std.Inflate().std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1]))

    ### SOFT
    if soft == 0:
        PP1 = limit2
    else:
        sharpdiff = core.std.MakeDiff(tmp, limit2)
        sharpdiff = core.std.Expr([sharpdiff, sharpdiff.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1])], expr=[f'x {neutral} - abs y {neutral} - abs > y {soft} * x {100 - soft} * + 100 / x ?'])
        PP1 = core.std.MakeDiff(tmp, sharpdiff)

    ### SOOTHE
    if soothe:
        diff = core.std.MakeDiff(tmp, PP1)
        diff = core.std.Expr([diff, AverageFrames(diff, weights=[1] * 3, scenechange=32 / 255)],
                             expr=[f'x {neutral} - y {neutral} - * 0 < x {neutral} - 100 / {keep} * {neutral} + x {neutral} - abs y {neutral} - abs > x {keep} * y {100 - keep} * + 100 / x ? ?'])
        PP2 = core.std.MakeDiff(tmp, diff)
    else:
        PP2 = PP1

    ### OUTPUT
    if dest_x != ox or dest_y != oy:
        if not isGray:
            PP2 = core.std.ShufflePlanes([PP2, tmp_orig], planes=[0, 1, 2], colorfamily=input.format.color_family)
        out = PP2.resize.Spline36(dest_x, dest_y)
    elif ss_x > 1 or ss_y > 1:
        out = PP2.resize.Spline36(dest_x, dest_y)
        if not isGray:
            out = core.std.ShufflePlanes([out, input], planes=[0, 1, 2], colorfamily=input.format.color_family)
    elif not isGray:
        out = core.std.ShufflePlanes([PP2, input], planes=[0, 1, 2], colorfamily=input.format.color_family)
    else:
        out = PP2

    if edgemode <= -1:
        return edge.resize.Spline36(dest_x, dest_y, format=input.format)
    elif source is not None:
        if dest_x != ox or dest_y != oy:
            src = source.resize.Spline36(dest_x, dest_y)
            In = input.resize.Spline36(dest_x, dest_y)
        else:
            src = source
            In = input

        shrpD = core.std.MakeDiff(In, out, planes=[0])
        expr = f'x {neutral} - abs y {neutral} - abs < x y ?'
        if hasattr(core,'zsmooth'):
          shrpL = core.std.Expr([core.zsmooth.Repair(shrpD, core.std.MakeDiff(In, src, planes=[0]), mode=[1] if isGray else [1, 0]), shrpD], expr=[expr] if isGray else [expr, ''])
        else:
          shrpL = core.std.Expr([core.rgvs.Repair(shrpD, core.std.MakeDiff(In, src, planes=[0]), mode=[1] if isGray else [1, 0]), shrpD], expr=[expr] if isGray else [expr, ''])
        return core.std.MakeDiff(In, shrpL, planes=[0])
    else:
        return out


def FineSharp(clip, mode=1, sstr=2.5, cstr=None, xstr=0, lstr=1.5, pstr=1.28, ldmp=None, hdmp=0.01, rep=12):
    """
    Original author: Didée (https://forum.doom9.org/showthread.php?t=166082)
    Small and relatively fast realtime-sharpening function, for 1080p,
    or after scaling 720p → 1080p during playback.
    (to make 720p look more like being 1080p)
    It's a generic sharpener. Only for good quality sources!
    (If the source is crap, FineSharp will happily sharpen the crap) :)
    Noise/grain will be enhanced, too. The method is GENERIC.

    Modus operandi: A basic nonlinear sharpening method is performed,
    then the *blurred* sharp-difference gets subtracted again.
    
    Args:
        mode  (int)  - 1 to 3, weakest to strongest. When negative -1 to -3,
                       a broader kernel for equalisation is used.
        sstr (float) - strength of sharpening.
        cstr (float) - strength of equalisation (recommended 0.5 to 1.25)
        xstr (float) - strength of XSharpen-style final sharpening, 0.0 to 1.0.
                       (but, better don't go beyond 0.25...)
        lstr (float) - modifier for non-linear sharpening.
        pstr (float) - exponent for non-linear sharpening.
        ldmp (float) - "low damp", to not over-enhance very small differences.
                       (noise coming out of flat areas)
        hdmp (float) - "high damp", this damping term has a larger effect than ldmp
                        when the sharp-difference is larger than 1, vice versa.
        rep   (int)  - repair mode used in final sharpening, recommended modes are 1/12/13.
    """

    color = clip.format.color_family
    bd = clip.format.bits_per_sample
    isFLOAT = clip.format.sample_type == vs.FLOAT
    mid = 0 if isFLOAT else 1 << (bd - 1)
    i = 0.00392 if isFLOAT else 1 << (bd - 8)
    xy = 'x y - {} /'.format(i) if bd != 8 else 'x y -'
    if hasattr(core,'zsmooth'):
      R = core.zsmooth.Repair
    else:
      R = core.rgsf.Repair if isFLOAT else core.rgvs.Repair
    mat1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    mat2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("FineSharp: This is not a clip!")

    if cstr is None:
        cstr = spline(sstr, {0: 0, 0.5: 0.1, 1: 0.6, 2: 0.9, 2.5: 1, 3: 1.1, 3.5: 1.15, 4: 1.2, 8: 1.25, 255: 1.5})
        cstr **= 0.8 if mode > 0 else cstr

    if ldmp is None:
        ldmp = sstr

    sstr = max(sstr, 0)
    cstr = max(cstr, 0)
    xstr = min(max(xstr, 0), 1)
    ldmp = max(ldmp, 0)
    hdmp = max(hdmp, 0)

    if sstr < 0.01 and cstr < 0.01 and xstr < 0.01:
        return clip

    tmp = core.std.ShufflePlanes(clip, [0], vs.GRAY) if color in [vs.YUV] else clip

    if abs(mode) == 1:
        c2 = core.std.Convolution(tmp, matrix=mat1).std.Median()
    else:
        c2 = core.std.Median(tmp).std.Convolution(matrix=mat1)
    if abs(mode) == 3:
        c2 = c2.std.Median()
    
    if sstr >= 0.01:
        expr = 'x y = x dup {} dup dup dup abs {} / {} pow swap3 abs {} + / swap dup * dup {} + / * * {} * + ?'
        shrp = core.std.Expr([tmp, c2], [expr.format(xy, lstr, 1/pstr, hdmp, ldmp, sstr*i)])

        if cstr >= 0.01:
            diff = core.std.MakeDiff(shrp, tmp)
            if cstr != 1:
                expr = 'x {} *'.format(cstr) if isFLOAT else 'x {} - {} * {} +'.format(mid, cstr, mid)
                diff = core.std.Expr([diff], [expr])
            diff = core.std.Convolution(diff, matrix=mat1) if mode > 0 else core.std.Convolution(diff, matrix=mat2)
            shrp = core.std.MakeDiff(shrp, diff)

    if xstr >= 0.01:
        xyshrp = core.std.Expr([shrp, core.std.Convolution(shrp, matrix=mat2)], ['x dup y - 9.69 * +'])
        rpshrp = R(xyshrp, shrp, [rep])
        shrp = core.std.Merge(shrp, rpshrp, [xstr])

    return core.std.ShufflePlanes([shrp, clip], [0, 1, 2], color) if color in [vs.YUV] else shrp

def DetailSharpen(clip, z=4, sstr=1.5, power=4, ldmp=1, mode=1, med=False):
    """
    From: https://forum.doom9.org/showthread.php?t=163598
    Didée: Wanna some sharpening that causes no haloing, without any edge masking?
    
    Args:
        z     (float) - zero point.
        sstr  (float) - strength of non-linear sharpening.
        power (float) - exponent of non-linear sharpening.
        ldmp  (float) - "low damp", to not over-enhance very small differences.
        mode   (int)  - 0: gaussian kernel 1: box kernel
        med   (bool)  - When True, median is used to achieve stronger sharpening.
        
    Examples:
        DetailSharpen() # Original DetailSharpen by Didée.
        DetailSharpen(power=1.5, mode=0, med=True) # Mini-SeeSaw...just without See, and without Saw.
    """
    
    ldmp = max(ldmp, 0)
    bd = clip.format.bits_per_sample
    isFLOAT = clip.format.sample_type == vs.FLOAT
    i = 0.00392 if isFLOAT else 1 << (bd - 8)
    xy = 'x y - {} /'.format(i) if bd != 8 else 'x y -'
    color = clip.format.color_family
    
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("DetailSharpen: This is not a clip!")

    tmp = core.std.ShufflePlanes(clip, [0], vs.GRAY) if color in [vs.YUV] else clip

    if mode == 1:
        blur = core.std.Convolution(tmp, matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])
    else:
        blur = core.std.Convolution(tmp, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    if med:
        blur = blur.std.Median()

    expr = 'x y = x dup {} dup dup abs {} / {} pow swap2 abs {} + / * {} * + ?'
    tmp = core.std.Expr([tmp, blur], [expr.format(xy, z, 1/power, ldmp, sstr*z*i)])

    return core.std.ShufflePlanes([tmp, clip], [0, 1, 2], color) if color in [vs.YUV] else tmp

def psharpen(clip, strength=25, threshold=75, ss_x=1.0, ss_y=1.0, dest_x=None, dest_y=None):
    """From http://forum.doom9.org/showpost.php?p=683344&postcount=28

    Sharpening function similar to LimitedSharpenFaster.

    Args:
        clip (clip): Input clip.
        strength (int): Strength of the sharpening.
        threshold (int): Controls "how much" to be sharpened.
        ss_x (float): Supersampling factor (reduce aliasing on edges).
        ss_y (float): Supersampling factor (reduce aliasing on edges).
        dest_x (int): Output resolution after sharpening.
        dest_y (int): Output resolution after sharpening.
    """

    src = clip

    if dest_x is None:
        dest_x = src.width
    if dest_y is None:
        dest_y = src.height

    strength = clamp(0, strength, 100) / 100.0
    threshold = clamp(0, threshold, 100) / 100.0

    if ss_x < 1.0:
        ss_x = 1.0
    if ss_y < 1.0:
        ss_y = 1.0

    if ss_x != 1.0 or ss_y != 1.0:
        clip = core.resize.Lanczos(clip, width=m4(src.width * ss_x), height=m4(src.height * ss_y))

    resz = clip

    if src.format.num_planes != 1:
        clip = core.std.ShufflePlanes(clips=clip, planes=[0], colorfamily=vs.GRAY)

    max_ = core.std.Maximum(clip)
    min_ = core.std.Minimum(clip)

    nmax = core.std.Expr([max_, min_], ["x y -"])
    nval = core.std.Expr([clip, min_], ["x y -"])

    expr0 = threshold * (1.0 - strength) / (1.0 - (1.0 - threshold) * (1.0 - strength))
    epsilon = 0.000000000000001
    scl = (1 << clip.format.bits_per_sample) // 256
    x = f"x {scl} /" if scl != 1 else "x"
    y = f"y {scl} /" if scl != 1 else "y"

    expr = (
        f"{x} {y} {epsilon} + / 2 * 1 - abs {expr0} < {strength} 1 = {x} {y} 2 / = 0 {y} 2 / ? "
        f"{x} {y} {epsilon} + / 2 * 1 - abs 1 {strength} - / ? {x} {y} {epsilon} + / 2 * 1 - abs 1 {threshold} - "
        f"* {threshold} + ? {x} {y} 2 / > 1 -1 ? * 1 + {y} * 2 / {scl} *"
    )

    nval = core.std.Expr([nval, nmax], [expr])

    clip = core.std.Expr([nval, min_], ["x y +"])

    if src.format.num_planes != 1:
        clip = core.std.ShufflePlanes(
            clips=[clip, resz], planes=[0, 1, 2], colorfamily=src.format.color_family
        )

    if ss_x != 1.0 or ss_y != 1.0 or dest_x != src.width or dest_y != src.height:
        clip = core.resize.Lanczos(clip, width=dest_x, height=dest_y)

    return clip



########################### HELPER


def clamp(minimum, value, maximum):
    return int(max(minimum, min(round(value), maximum)))


def m4(value, mult=4.0):
    return 16 if value < 16 else int(round(value / mult) * mult)



def cround(x: float) -> int:
    return math.floor(x + 0.5) if x > 0 else math.ceil(x - 0.5)
    
def scale(value, peak):
    return cround(value * peak / 255) if peak != 1 else value / 255
    
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
    
# MinBlur   by Didée (http://avisynth.nl/index.php/MinBlur)
# Nifty Gauss/Median combination
def MinBlur(clp, r=1, planes=None):
    if not isinstance(clp, vs.VideoNode):
        raise vs.Error('MinBlur: This is not a clip')

    if planes is None:
        planes = list(range(clp.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    matrix1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    matrix2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    if r <= 0:
        RG11 = sbr(clp, planes=planes)
        RG4 = clp.std.Median(planes=planes)
    elif r == 1:
        RG11 = clp.std.Convolution(matrix=matrix1, planes=planes)
        RG4 = clp.std.Median(planes=planes)
    elif r == 2:
        RG11 = clp.std.Convolution(matrix=matrix1, planes=planes).std.Convolution(matrix=matrix2, planes=planes)
        RG4 = clp.ctmf.CTMF(radius=2, planes=planes)
    else:
        RG11 = clp.std.Convolution(matrix=matrix1, planes=planes).std.Convolution(matrix=matrix2, planes=planes).std.Convolution(matrix=matrix2, planes=planes)
        if clp.format.bits_per_sample == 16:
            s16 = clp
            RG4 = clp.fmtc.bitdepth(bits=12, planes=planes, dmode=1).ctmf.CTMF(radius=3, planes=planes).fmtc.bitdepth(bits=16, planes=planes)
            RG4 = mvf.LimitFilter(s16, RG4, thr=0.0625, elast=2, planes=planes)
        else:
            RG4 = clp.ctmf.CTMF(radius=3, planes=planes)

    expr = 'x y - x z - * 0 < x x y - abs x z - abs < y z ? ?'
    return core.std.Expr([clp, RG11, RG4], expr=[expr if i in planes else '' for i in range(clp.format.num_planes)])
    
    
def sbr(c: vs.VideoNode, r: int = 1, planes: Optional[Union[int, Sequence[int]]] = None) -> vs.VideoNode:
    '''make a highpass on a blur's difference (well, kind of that)'''
    if not isinstance(c, vs.VideoNode):
        raise vs.Error('sbr: this is not a clip')

    neutral = 1 << (c.format.bits_per_sample - 1) if c.format.sample_type == vs.INTEGER else 0.0

    plane_range = range(c.format.num_planes)

    if planes is None:
        planes = list(plane_range)
    elif isinstance(planes, int):
        planes = [planes]

    matrix1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    matrix2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    RG11 = c.std.Convolution(matrix=matrix1, planes=planes)
    if r >= 2:
        RG11 = RG11.std.Convolution(matrix=matrix2, planes=planes)
    if r >= 3:
        RG11 = RG11.std.Convolution(matrix=matrix2, planes=planes)

    RG11D = core.std.MakeDiff(c, RG11, planes=planes)

    RG11DS = RG11D.std.Convolution(matrix=matrix1, planes=planes)
    if r >= 2:
        RG11DS = RG11DS.std.Convolution(matrix=matrix2, planes=planes)
    if r >= 3:
        RG11DS = RG11DS.std.Convolution(matrix=matrix2, planes=planes)

    RG11DD = core.std.Expr(
        [RG11D, RG11DS],
        expr=[f'x y - x {neutral} - * 0 < {neutral} x y - abs x {neutral} - abs < x y - {neutral} + x ? ?' if i in planes else '' for i in plane_range],
    )
    return core.std.MakeDiff(c, RG11DD, planes=planes)
    
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

    return core.std.Expr([clip, bright_limit, dark_limit], expr=[f'x y {overshoot} + min z {undershoot} - max' if i in planes else '' for i in plane_range])

def spline(x, coordinates):
    def get_matrix(px, py, l):
        matrix = []
        matrix.append([(i == 0) * 1.0 for i in range(l + 1)])
        for i in range(1, l - 1):
            p = [0 for t in range(l + 1)]
            p[i - 1] = px[i] - px[i - 1]
            p[i] = 2 * (px[i + 1] - px[i - 1])
            p[i + 1] = px[i + 1] - px[i]
            p[l] = 6 * (((py[i + 1] - py[i]) / p[i + 1]) - (py[i] - py[i - 1]) / p[i - 1])
            matrix.append(p)
        matrix.append([(i == l - 1) * 1.0 for i in range(l + 1)])
        return matrix
    def equation(matrix, dim):
        for i in range(dim):
            num = matrix[i][i]
            for j in range(dim + 1):
                matrix[i][j] /= num
            for j in range(dim):
                if i != j:
                    a = matrix[j][i]
                    for k in range(i, dim + 1):
                        matrix[j][k] -= a * matrix[i][k]
    if not isinstance(coordinates, dict):
        raise TypeError("coordinates must be a dict")
    length = len(coordinates)
    if length < 3:
        raise ValueError("coordinates require at least three pairs")
    px = [key for key in coordinates.keys()]
    py = [val for val in coordinates.values()]
    matrix = get_matrix(px, py, length)
    equation(matrix, length)
    for i in range(length + 1):
        if x >= px[i] and x <= px[i + 1]:
            break
    j = i + 1
    h = px[j] - px[i]
    s = matrix[j][length] * (x - px[i]) ** 3
    s -= matrix[i][length] * (x - px[j]) ** 3
    s /= 6 * h
    s += (py[j] / h - h * matrix[j][length] / 6) * (x - px[i])
    s -= (py[i] / h - h * matrix[i][length] / 6) * (x - px[j])
    
    return s
    
def ContraSharpening(
    denoised: vs.VideoNode, original: vs.VideoNode, radius: int = 1, rep: int = 1, planes: Optional[Union[int, Sequence[int]]] = None
) -> vs.VideoNode:
    '''
    contra-sharpening: sharpen the denoised clip, but don't add more to any pixel than what was removed previously.

    Parameters:
        denoised: Denoised clip to sharpen.

        original: Original clip before denoising.

        radius: Spatial radius for contra-sharpening.

        rep: Mode of repair to limit the difference.

        planes: Specifies which planes will be processed. Any unprocessed planes will be simply copied.
            By default only luma plane will be processed for non-RGB formats.
    '''
    if not (isinstance(denoised, vs.VideoNode) and isinstance(original, vs.VideoNode)):
        raise vs.Error('ContraSharpening: this is not a clip')

    if denoised.format.id != original.format.id:
        raise vs.Error('ContraSharpening: clips must have the same format')

    neutral = 1 << (denoised.format.bits_per_sample - 1)

    plane_range = range(denoised.format.num_planes)

    if planes is None:
        planes = [0] if denoised.format.color_family != vs.RGB else [0, 1, 2]
    elif isinstance(planes, int):
        planes = [planes]

    pad = 2 if radius < 3 else 4
    denoised = Padding(denoised, pad, pad, pad, pad)
    original = Padding(original, pad, pad, pad, pad)

    matrix1 = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    matrix2 = [1, 1, 1, 1, 1, 1, 1, 1, 1]

    # damp down remaining spots of the denoised clip
    s = MinBlur(denoised, radius, planes)
    # the difference achieved by the denoising
    allD = core.std.MakeDiff(original, denoised, planes=planes)

    RG11 = s.std.Convolution(matrix=matrix1, planes=planes)
    if radius >= 2:
        RG11 = RG11.std.Convolution(matrix=matrix2, planes=planes)
    if radius >= 3:
        RG11 = RG11.std.Convolution(matrix=matrix2, planes=planes)

    # the difference of a simple kernel blur
    ssD = core.std.MakeDiff(s, RG11, planes=planes)
    # limit the difference to the max of what the denoising removed locally
    if hasattr(core,'zsmooth'):
      ssDD = core.zsmooth.Repair(ssD, allD, mode=[rep if i in planes else 0 for i in plane_range])
    else:
      ssDD = core.rgvs.Repair(ssD, allD, mode=[rep if i in planes else 0 for i in plane_range])
    # abs(diff) after limiting may not be bigger than before
    ssDD = core.std.Expr([ssDD, ssD], expr=[f'x {neutral} - abs y {neutral} - abs < x y ?' if i in planes else '' for i in plane_range])
    # apply the limited difference (sharpening is just inverse blurring)
    last = core.std.MergeDiff(denoised, ssDD, planes=planes)
    return last.std.Crop(pad, pad, pad, pad)
    
def Padding(clip: vs.VideoNode, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('Padding: this is not a clip')

    if left < 0 or right < 0 or top < 0 or bottom < 0:
        raise vs.Error('Padding: border size to pad must not be negative')

    width = clip.width + left + right
    height = clip.height + top + bottom

    return clip.resize.Point(width, height, src_left=-left, src_top=-top, src_width=width, src_height=height)
    
    
def UnsharpMask(clip: vs.VideoNode, strength: int = 64, radius: int = 3, threshold: int = 8) -> vs.VideoNode:
    """
    Ported by Myrsloik
    
    Applies an unsharp mask filter to a given clip, which sharpens the image by enhancing edges.
    
    The function works by subtracting a blurred version of the clip from the original, then amplifying 
    the difference based on the given strength. The threshold parameter determines the minimum difference
    required for sharpening to take place, and the radius controls the size of the blur applied to the image.
    
    Args:
        clip (vs.VideoNode): The input video clip to which the unsharp mask will be applied.
        strength (Optional[int], default=64): The strength of the sharpening effect (higher values mean stronger sharpening).
        radius (Optional[int], default=3): The radius of the blur filter to use for the unsharp mask.
        threshold (Optional[int], default=8): The threshold for the difference between the original and blurred clip
                                               that will be sharpened. Values between 0 and 255, with 8 being a typical starting point.

    Returns:
        vs.VideoNode: The sharpened video clip after applying the unsharp mask.
    """
    
    # Calculate the maximum pixel value based on the bit depth of the clip
    maxvalue = (1 << clip.format.bits_per_sample) - 1
    
    # Adjust the threshold to match the clip's color depth
    threshold = threshold * maxvalue // 255
    
    # Apply convolution to blur the image horizontally and vertically
    blurclip = clip.std.Convolution(matrix=[1] * (radius * 2 + 1), planes=0, mode='v')
    blurclip = blurclip.std.Convolution(matrix=[1] * (radius * 2 + 1), planes=0, mode='h')
    
    # Create the expression to calculate the sharpened result
    expressions = [
        'x y - abs {} > x y - {} * x + x ?'.format(threshold, strength / 128)
    ]
    
    # If the clip is not grayscale, append an empty expression (to handle color clips)
    if clip.format.color_family != vs.GRAY:
        expressions.append("")
    
    # Apply the expression to blend the original clip and the blurred clip
    return core.std.Expr(clips=[clip, blurclip], expr=expressions)
