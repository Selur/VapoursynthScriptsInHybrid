import vapoursynth as vs
from vapoursynth import core

import math
from functools import partial

# taken from adjust
def Tweak(clip, hue=None, sat=None, bright=None, cont=None, coring=True):
    if clip.format is None:
        raise vs.Error("Tweak: only clips with constant format are accepted.")

    if clip.format.color_family == vs.RGB:
        raise vs.Error("Tweak: RGB clips are not accepted.")
        
    if (hue is not None or sat is not None) and clip.format.color_family != vs.GRAY:
        hue = 0.0 if hue is None else hue
        sat = 1.0 if sat is None else sat

        hue = hue * math.pi / 180.0
        hue_sin = math.sin(hue)
        hue_cos = math.cos(hue)

        gray = 128 << (clip.format.bits_per_sample - 8)

        chroma_min = 0
        chroma_max = (2 ** clip.format.bits_per_sample) - 1
        if coring:
            chroma_min = 16 << (clip.format.bits_per_sample - 8)
            chroma_max = 240 << (clip.format.bits_per_sample - 8)

        expr_u = "x {} - {} * y {} - {} * + {} + {} max {} min".format(gray, hue_cos * sat, gray, hue_sin * sat, gray, chroma_min, chroma_max)
        expr_v = "y {} - {} * x {} - {} * - {} + {} max {} min".format(gray, hue_cos * sat, gray, hue_sin * sat, gray, chroma_min, chroma_max)

        if clip.format.sample_type == vs.FLOAT:
            expr_u = "x {} * y {} * + -0.5 max 0.5 min".format(hue_cos * sat, hue_sin * sat)
            expr_v = "y {} * x {} * - -0.5 max 0.5 min".format(hue_cos * sat, hue_sin * sat)

        src_u = clip.std.ShufflePlanes(planes=1, colorfamily=vs.GRAY)
        src_v = clip.std.ShufflePlanes(planes=2, colorfamily=vs.GRAY)

        dst_u = core.std.Expr(clips=[src_u, src_v], expr=expr_u)
        dst_v = core.std.Expr(clips=[src_u, src_v], expr=expr_v)

        clip = core.std.ShufflePlanes(clips=[clip, dst_u, dst_v], planes=[0, 0, 0], colorfamily=clip.format.color_family)

    if bright is not None or cont is not None:
        bright = 0.0 if bright is None else bright
        cont = 1.0 if cont is None else cont

        if clip.format.sample_type == vs.INTEGER:
            luma_lut = []

            luma_min = 0
            luma_max = (2 ** clip.format.bits_per_sample) - 1
            if coring:
                luma_min = 16 << (clip.format.bits_per_sample - 8)
                luma_max = 235 << (clip.format.bits_per_sample - 8)

            for i in range(2 ** clip.format.bits_per_sample):
                val = int((i - luma_min) * cont + bright + luma_min + 0.5)
                luma_lut.append(min(max(val, luma_min), luma_max))

            clip = clip.std.Lut(planes=0, lut=luma_lut)
        else:
            expression = "x {} * {} + 0.0 max 1.0 min".format(cont, bright)

            clip = clip.std.Expr(expr=[expression, "", ""])

    return clip


# Taken from havsfunc
#########################################################################################
###                                                                                   ###
###                      function Smooth Levels : SmoothLevels()                      ###
###                                                                                   ###
###                                v1.02 by "LaTo INV."                               ###
###                                                                                   ###
###                                  28 January 2009                                  ###
###                                                                                   ###
#########################################################################################
###
###
### /!\ Needed filters : RGVS, f3kdb
### --------------------
###
###
###
### +---------+
### | GENERAL |
### +---------+
###
### Levels options:
### ---------------
### input_low, gamma, input_high, output_low, output_high [default: 0, 1.0, maximum value of input format, 0, maximum value of input format]
### /!\ The value is not internally normalized on an 8-bit scale, and must be scaled to the bit depth of input format manually by users
###
### chroma [default: 50]
### ---------------------
### 0   = no chroma processing     (similar as Ylevels)
### xx  = intermediary
### 100 = normal chroma processing (similar as Levels)
###
### limiter [default: 0]
### --------------------
### 0 = no limiter             (similar as Ylevels)
### 1 = input limiter
### 2 = output limiter         (similar as Levels: coring=false)
### 3 = input & output limiter (similar as Levels: coring=true)
###
###
###
### +----------+
### | LIMITING |
### +----------+
###
### Lmode [default: 0]
### ------------------
### 0 = no limit
### 1 = limit conversion on dark & bright areas (apply conversion @0%   at luma=0 & @100% at luma=Ecenter & @0% at luma=255)
### 2 = limit conversion on dark areas          (apply conversion @0%   at luma=0 & @100% at luma=255)
### 3 = limit conversion on bright areas        (apply conversion @100% at luma=0 & @0%   at luma=255)
###
### DarkSTR [default: 100]
### ----------------------
### Strength for limiting: the higher, the more conversion are reduced on dark areas (for Lmode=1&2)
###
### BrightSTR [default: 100]
### ------------------------
### Strength for limiting: the higher, the more conversion are reduced on bright areas (for Lmode=1&3)
###
### Ecenter [default: median value of input format]
### ----------------------
### Center of expression for Lmode=1
### /!\ The value is not internally normalized on an 8-bit scale, and must be scaled to the bit depth of input format manually by users
###
### protect [default: -1]
### ---------------------
### -1  = protect off
### >=0 = pure black protection
###       ---> don't apply conversion on pixels egal or below this value
###            (ex: with 16, the black areas like borders and generic are untouched so they don't look washed out)
### /!\ The value is not internally normalized on an 8-bit scale, and must be scaled to the bit depth of input format manually by users
###
### Ecurve [default: 0]
### -------------------
### Curve used for limit & protect:
### 0 = use sine curve
### 1 = use linear curve
###
###
###
### +-----------+
### | SMOOTHING |
### +-----------+
###
### Smode [default: -2]
### -------------------
### 2  = smooth on, maxdiff must be < to "255/Mfactor"
### 1  = smooth on, maxdiff must be < to "128/Mfactor"
### 0  = smooth off
### -1 = smooth on if maxdiff < "128/Mfactor", else off
### -2 = smooth on if maxdiff < "255/Mfactor", else off
###
### Mfactor [default: 2]
### --------------------
### The higher, the more precise but the less maxdiff allowed:
### maxdiff=128/Mfactor for Smode1&-1 and maxdiff=255/Mfactor for Smode2&-2
###
### RGmode [default: 12]
### --------------------
### In strength order: + 19 > 12 >> 20 > 11 -
###
### useDB [default: false]
### ---------------------
### Use f3kdb on top of removegrain: prevent posterize when doing levels conversion
###
###
#########################################################################################
def SmoothLevels(input, input_low=0, gamma=1.0, input_high=None, output_low=0, output_high=None, chroma=50, limiter=0, Lmode=0, DarkSTR=100, BrightSTR=100, Ecenter=None, protect=-1, Ecurve=0,
                 Smode=-2, Mfactor=2, RGmode=12, useDB=False):
    if not isinstance(input, vs.VideoNode):
        raise vs.Error('SmoothLevels: this is not a clip')

    if input.format.color_family == vs.RGB:
        raise vs.Error('SmoothLevels: RGB format is not supported')

    isGray = (input.format.color_family == vs.GRAY)

    if input.format.sample_type == vs.INTEGER:
        neutral = [1 << (input.format.bits_per_sample - 1)] * 2
        peak = (1 << input.format.bits_per_sample) - 1
    else:
        neutral = [0.5, 0.0]
        peak = 1.0

    if chroma <= 0 and not isGray:
        input_orig = input
        input = mvf.GetPlane(input, 0)
    else:
        input_orig = None

    if input_high is None:
        input_high = peak

    if output_high is None:
        output_high = peak

    if Ecenter is None:
        Ecenter = neutral[0]

    if gamma <= 0:
        raise vs.Error('SmoothLevels: gamma must be greater than 0.0')

    if Ecenter <= 0 or Ecenter >= peak:
        raise vs.Error('SmoothLevels: Ecenter must be greater than 0 and less than maximum value of input format')

    if Mfactor <= 0:
        raise vs.Error('SmoothLevels: Mfactor must be greater than 0')

    if RGmode == 4:
        RemoveGrain = partial(core.std.Median)
    elif RGmode in [11, 12]:
        RemoveGrain = partial(core.std.Convolution, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    elif RGmode == 19:
        RemoveGrain = partial(core.std.Convolution, matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1])
    elif RGmode == 20:
        RemoveGrain = partial(core.std.Convolution, matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])
    else:
        RG = core.zsmooth.RemoveGrain if hasattr(core,'zsmooth') else core.rgvs.RemoveGrain
        RemoveGrain = partial(RG, mode=[RGmode])

    ### EXPRESSION
    exprY = f'x {input_low} - {input_high - input_low + (input_high == input_low)} / {1 / gamma} pow {output_high - output_low} * {output_low} +'

    if chroma > 0 and not isGray:
        scaleC = ((output_high - output_low) / (input_high - input_low + (input_high == input_low)) + 100 / chroma - 1) / (100 / chroma)
        exprC = f'x {neutral[1]} - {scaleC} * {neutral[1]} +'

    Dstr = DarkSTR / 100
    Bstr = BrightSTR / 100

    if Lmode <= 0:
        exprL = '1'
    elif Ecurve <= 0:
        if Lmode == 1:
            var_d = f'x {Ecenter} /'
            var_b = f'{peak} x - {peak} {Ecenter} - /'
            exprL = f'x {Ecenter} < ' + sine_expr(var_d) + f' {Dstr} pow x {Ecenter} > ' + sine_expr(var_b) + f' {Bstr} pow 1 ? ?'
        elif Lmode == 2:
            var_d = f'x {peak} /'
            exprL = sine_expr(var_d) + f' {Dstr} pow'
        else:
            var_b = f'{peak} x - {peak} /'
            exprL = sine_expr(var_b) + f' {Bstr} pow'
    else:
        if Lmode == 1:
            exprL = f'x {Ecenter} < x {Ecenter} / abs {Dstr} pow x {Ecenter} > 1 x {Ecenter} - {peak - Ecenter} / abs - {Bstr} pow 1 ? ?'
        elif Lmode == 2:
            exprL = f'1 x {peak} - {peak} / abs - {Dstr} pow'
        else:
            exprL = f'x {peak} - {peak} / abs {Bstr} pow'

    if protect <= -1:
        exprP = '1'
    elif Ecurve <= 0:
        var_p = f'x {protect} - {scale(16, peak)} /'
        exprP = f'x {protect} <= 0 x {protect + scale(16, peak)} >= 1 ' + sine_expr(var_p) + f' ? ?'
    else:
        exprP = f'x {protect} <= 0 x {protect + scale(16, peak)} >= 1 x {protect} - {scale(16, peak)} / abs ? ?'

    ### PROCESS
    if limiter == 1 or limiter >= 3:
        limitI = input.std.Expr(expr=[f'x {input_low} max {input_high} min'])
    else:
        limitI = input

    expr = exprL + ' ' + exprP + ' * ' + exprY + ' x - * x +'
    level = limitI.std.Expr(expr=[expr] if chroma <= 0 or isGray else [expr, exprC])
    diff = core.std.Expr([limitI, level], expr=[f'x y - {Mfactor} * {neutral[1]} +'])
    process = RemoveGrain(diff)
    if useDB:
        if hasattr(core, 'neo_f3kdb'):
          process = process.std.Expr(expr=[f'x {neutral[1]} - {Mfactor} / {neutral[1]} +']).neo_f3kdb.Deband(grainy=0, grainc=0, output_depth=input.format.bits_per_sample)
        else:
          process = process.std.Expr(expr=[f'x {neutral[1]} - {Mfactor} / {neutral[1]} +']).f3kdb.Deband(grainy=0, grainc=0, output_depth=input.format.bits_per_sample)
        smth = core.std.MakeDiff(limitI, process)
    else:
        smth = core.std.Expr([limitI, process], expr=[f'x y {neutral[1]} - {Mfactor} / -'])

    level2 = core.std.Expr([limitI, diff], expr=[f'x y {neutral[1]} - {Mfactor} / -'])
    diff2 = core.std.Expr([level2, level], expr=[f'x y - {Mfactor} * {neutral[1]} +'])
    process2 = RemoveGrain(diff2)
    if useDB:
        if hasattr(core, 'neo_f3kdb'):
          process2 = process2.std.Expr(expr=[f'x {neutral[1]} - {Mfactor} / {neutral[1]} +']).neo_f3kdb.Deband(grainy=0, grainc=0, output_depth=input.format.bits_per_sample)
        else:
          process2 = process2.std.Expr(expr=[f'x {neutral[1]} - {Mfactor} / {neutral[1]} +']).f3kdb.Deband(grainy=0, grainc=0, output_depth=input.format.bits_per_sample) 
        smth2 = core.std.MakeDiff(smth, process2)
    else:
        smth2 = core.std.Expr([smth, process2], expr=[f'x y {neutral[1]} - {Mfactor} / -'])

    mask1 = core.std.Expr([limitI, level], expr=[f'x y - abs {neutral[0] / Mfactor} >= {peak} 0 ?'])
    mask2 = core.std.Expr([limitI, level], expr=[f'x y - abs {peak / Mfactor} >= {peak} 0 ?'])

    if Smode >= 2:
        Slevel = smth2
    elif Smode == 1:
        Slevel = smth
    elif Smode == -1:
        Slevel = core.std.MaskedMerge(smth, level, mask1)
    elif Smode <= -2:
        Slevel = core.std.MaskedMerge(core.std.MaskedMerge(smth, smth2, mask1), level, mask2)
    else:
        Slevel = level

    if limiter >= 2:
        limitO = Slevel.std.Expr(expr=[f'x {output_low} max {output_high} min'])
    else:
        limitO = Slevel

    if input_orig is not None:
        limitO = core.std.ShufflePlanes([limitO, input_orig], planes=[0, 1, 2], colorfamily=input_orig.format.color_family)
    return limitO

####### HighBitDepthHistogram
#
# This function wraps VapourSynth's core.hist.<Method> filters to support high bit-depth clips.
# The histogram filters only support 8-bit input, so this function:
# 1. Clones the original clip.
# 2. Converts the clone to 8-bit while preserving the original chroma subsampling.
# 3. Applies the specified histogram method (Classic, Levels, Color, Color2, or Luma).
# 4. Crops the histogram overlay from the extended image.
# 5. Converts the histogram overlay back to the original format.
# 6. Stacks the original clip with the histogram (vertically for Classic, horizontally otherwise).
def HighBitDepthHistogram(clip: vs.VideoNode, method: str = "Classic") -> vs.VideoNode:
    method = method.capitalize()

    if method not in {"Classic", "Levels", "Color", "Color2", "Luma"}:
        raise ValueError(f"Unsupported histogram method: {method}")

    original_format = clip.format

    # Determine 8-bit format matching the input
    target_format_map = {
        (vs.GRAY, 0): vs.GRAY8,
        (vs.YUV, 444): vs.YUV444P8,
        (vs.YUV, 422): vs.YUV422P8,
        (vs.YUV, 420): vs.YUV420P8,
        (vs.RGB, 0): vs.RGB24
    }

    subsampling_w = clip.format.subsampling_w
    subsampling_h = clip.format.subsampling_h
    chroma = 444 if (subsampling_w == 0 and subsampling_h == 0) else 422 if subsampling_w == 1 else 420

    target_key = (clip.format.color_family, chroma if clip.format.color_family == vs.YUV else 0)
    if target_key not in target_format_map:
        raise ValueError(f"No 8-bit format mapping found for {target_key}")

    target_format = target_format_map[target_key]

    # Step 1: Clone clip
    histClip = clip

    # Step 2: Convert histClip to 8-bit
    histClip = core.resize.Bicubic(histClip, format=target_format)

    # Step 3: Apply histogram
    try:
        histClip = getattr(core.hist, method)(clip=histClip)
    except AttributeError:
        raise ValueError(f"Histogram method '{method}' is not available in core.hist.")

    # Step 4: Crop to get just the histogram overlay
    if method == "Classic":
        histClip = core.std.Crop(histClip, bottom=clip.height)
    else:
        histClip = core.std.Crop(histClip, left=clip.width)

    # Step 5: Convert histogram back to original format
    histClip = core.resize.Bicubic(histClip, format=original_format.id)

    # Step 6: Stack original + histogram
    if method == "Classic":
        stacked = core.std.StackVertical([histClip, clip])
    else:
        stacked = core.std.StackHorizontal([clip, histClip])

    return stacked


# based on: https://forum.videohelp.com/threads/396285-Converting-Blu-Ray-YUV-to-RGB-and-back-to-YUV#post2576719 by  _Al_
def RGBAdjust(rgb: vs.VideoNode, r: float=1.0, g: float=1.0, b: float=1.0, a: float=1.0, rb: float=0.0, gb: float=0.0, bb: float=0.0, ab: float=0.0, rg: float=1.0, gg: float=1.0, bg: float=1.0, ag: float=1.0):
  funcName = 'RGBAdjust'
  if rgb.format.color_family != vs.RGB:
    raise ValueError(funcName + ': input clip needs to be RGB!')
  type = rgb.format.sample_type
  size = 2**rgb.format.bits_per_sample
  #adjusting bias values rb,gb,bb for any RGB bit depth
  limited = rgb.get_frame(0).props['_ColorRange'] == 1
  if limited:
    if rb > 235 or rb < -235: raise ValueError(funcName + ': source is flagged as "limited" but rb is out of range [-235,235]!')  
    if gb > 235 or gb < -235: raise ValueError(funcName + ': source is flagged as "limited" but gb is out of range [-235,235]!')
    if bb > 235 or bb < -235: raise ValueError(funcName + ': source is flagged as "limited" but bb is out of range [-235,235]!')
  else:
    if rb > 255 or rb < -255: raise ValueError(funcName + ': source is flagged as "full" but rb is out of range [-255,255]!') 
    if gb > 255 or gb < -255: raise ValueError(funcName + ': source is flagged as "limited" but gb is out of range [-235,235]!')
    if bb > 255 or bb < -255: raise ValueError(funcName + ': source is flagged as "limited" but bb is out of range [-235,235]!')

  if rg < 0: raise ValueError(funcName + ': rg needs to be >= 0!')
  if gg < 0: raise ValueError(funcName + ': gg needs to be >= 0!')
  if bg < 0: raise ValueError(funcName + ': bg needs to be >= 0!')
      
  if limited:
    if type==vs.INTEGER:
      maxVal = 235
    else:
      maxVal = 235.0
  else:
    if type==vs.INTEGER:
      maxVal = 255
    else:
      maxVal = 255.0
  rb,gb,bb = map(lambda b: b if size==maxVal else size/maxVal*b if type==vs.INTEGER else b/maxVal, [rb,gb,bb])

  #x*r + rb , x*g + gb , x*b + bb
  rgb_adjusted = core.std.Expr(rgb, [f"x {r} * {rb} +", f"x {g} * {gb} +", f"x {b} * {bb} +"])

  #gamma per channel
  planes = [core.std.ShufflePlanes(rgb_adjusted, planes=p,  colorfamily=vs.GRAY)  for p in [0,1,2]]
  planes = [core.std.Levels(planes[p], gamma=g) if not g==1 else planes[p] for p, g in enumerate([rg, gg, bg])]
  rgb_adjusted = core.std.ShufflePlanes(planes, planes=[0,0,0], colorfamily = vs.RGB)
  return rgb_adjusted
  
  
def AutoGain(clip: vs.VideoNode, gain_limit: float = 1.0, strength: float = 0.5, darken: bool = False) -> vs.VideoNode:
    """
    Dynamically adjusts the brightness and contrast of a clip based on its luminance range.
    
    Parameters:
    clip (vs.VideoNode): Input clip. Must be YUV format (integer or float samples).
    gain_limit (float): Limits maximum scaling percentage (0 = no gain/stretching, 100 = full gain). Default is 1.0.
    strength (float): Strength of the adjustment. 0.0 = no adjustment, 1.0 = full adjustment. Default is 0.5.
    darken (bool): If True, compresses dynamic range. If False, expands it. Default is False.

    Returns:
    vs.VideoNode: Processed clip with adjusted dynamic range, in original format.
    """
    import vapoursynth as vs
    core = vs.core

    if not clip.format:
        raise ValueError("AutoGain: Variable format clips are not supported.")

    fmt = clip.format

    # Check supported formats
    if fmt.color_family != vs.YUV:
        raise ValueError("AutoGain: Only YUV color family is supported.")
    if fmt.sample_type not in (vs.INTEGER, vs.FLOAT):
        raise ValueError("AutoGain: Only integer and float sample types are supported.")

    # Extract Y plane
    Y = core.std.ShufflePlanes(clip, planes=0, colorfamily=vs.GRAY)

    is_float = fmt.sample_type == vs.FLOAT
    bits = fmt.bits_per_sample
    peak = 1.0 if is_float else (1 << bits) - 1

    props = core.std.PlaneStats(Y)

    def apply_gain(n: int, f: vs.VideoFrame) -> vs.VideoNode:
        avg = f.props.PlaneStatsAverage
        min_ = f.props.PlaneStatsMin
        max_ = f.props.PlaneStatsMax

        y_range = max_ - min_

        # Avoid division by zero if the range is zero
        if y_range == 0:
            scale = 1.0
            offset = 0.0
        else:
            if darken:
                # Compression: reduce contrast by moving toward average
                scale = 1.0 - (gain_limit * strength)
                offset = (1.0 - scale) * avg  # Shift toward the average to compress
            else:
                # Expansion: stretch contrast to use full available range
                ideal_scale = peak / y_range
                scale = 1.0 + (ideal_scale - 1.0) * (gain_limit * strength)
                offset = -scale * min_  # Adjust the offset to use the full available range

        # Calculate final expression (now strength=0 means no adjustment)
        weight = max(min(strength, 1.0), 0.0)
        expr = f"x {offset:.8f} + {scale:.8f} * {weight:.8f} * x {1.0-weight:.8f} * +"
        return core.std.Expr([Y], expr=[expr])

    # Adjusted luma (frame by frame)
    Y_adj = core.std.FrameEval(Y, eval=apply_gain, prop_src=props)

    if fmt.num_planes == 1:
        result = Y_adj
    else:
        # Handle U and V planes separately and merge
        U = core.std.ShufflePlanes(clip, planes=1, colorfamily=vs.GRAY)
        V = core.std.ShufflePlanes(clip, planes=2, colorfamily=vs.GRAY)
        result = core.std.ShufflePlanes([Y_adj, U, V], planes=[0, 0, 0], colorfamily=vs.YUV)

    return result
