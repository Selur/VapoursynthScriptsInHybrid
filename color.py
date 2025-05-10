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
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
        
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
        dst_u = EXPR(clips=[src_u, src_v], expr=expr_u)
        dst_v = EXPR(clips=[src_u, src_v], expr=expr_v)

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

            clip = EXPR(expr=[expression, "", ""])

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
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    if limiter == 1 or limiter >= 3:
        limitI = EXPR(input, expr=[f'x {input_low} max {input_high} min'])
    else:
        limitI = input

    expr = exprL + ' ' + exprP + ' * ' + exprY + ' x - * x +'
    level = EXPR(limitI, expr=[expr] if chroma <= 0 or isGray else [expr, exprC])
    diff = EXPR([limitI, level], expr=[f'x y - {Mfactor} * {neutral[1]} +'])
    process = RemoveGrain(diff)
    if useDB:
        if hasattr(core, 'neo_f3kdb'):
          process = EXPR(process, expr=[f'x {neutral[1]} - {Mfactor} / {neutral[1]} +']).neo_f3kdb.Deband(grainy=0, grainc=0, output_depth=input.format.bits_per_sample)
        else:
          process = EXPR(process, expr=[f'x {neutral[1]} - {Mfactor} / {neutral[1]} +']).f3kdb.Deband(grainy=0, grainc=0, output_depth=input.format.bits_per_sample)
        smth = core.std.MakeDiff(limitI, process)
    else:
        smth = EXPR([limitI, process], expr=[f'x y {neutral[1]} - {Mfactor} / -'])

    level2 = EXPR([limitI, diff], expr=[f'x y {neutral[1]} - {Mfactor} / -'])
    diff2 = EXPR([level2, level], expr=[f'x y - {Mfactor} * {neutral[1]} +'])
    process2 = RemoveGrain(diff2)
    if useDB:
        if hasattr(core, 'neo_f3kdb'):
          process2 = EXPR(process2, expr=[f'x {neutral[1]} - {Mfactor} / {neutral[1]} +']).neo_f3kdb.Deband(grainy=0, grainc=0, output_depth=input.format.bits_per_sample)
        else:
          process2 = EXPR(process2, expr=[f'x {neutral[1]} - {Mfactor} / {neutral[1]} +']).f3kdb.Deband(grainy=0, grainc=0, output_depth=input.format.bits_per_sample) 
        smth2 = core.std.MakeDiff(smth, process2)
    else:
        smth2 = EXPR([smth, process2], expr=[f'x y {neutral[1]} - {Mfactor} / -'])

    mask1 = EXPR([limitI, level], expr=[f'x y - abs {neutral[0] / Mfactor} >= {peak} 0 ?'])
    mask2 = EXPR([limitI, level], expr=[f'x y - abs {peak / Mfactor} >= {peak} 0 ?'])

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
        limitO = EXPR(Slevel, expr=[f'x {output_low} max {output_high} min'])
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

  EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
  #x*r + rb , x*g + gb , x*b + bb
  rgb_adjusted = EXPR(rgb, [f"x {r} * {rb} +", f"x {g} * {gb} +", f"x {b} * {bb} +"])

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
        
        EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
        return EXPR([Y], expr=[expr])

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

# auto white from http://www.vapoursynth.com/doc/functions/frameeval.html
def AutoWhiteAdjust(n, f, clip, core):
   small_number = 0.000000001
   red   = f[0].props['PlaneStatsAverage']
   green = f[1].props['PlaneStatsAverage']
   blue  = f[2].props['PlaneStatsAverage']
   max_rgb = max(red, green, blue)
   red_corr   = max_rgb/max(red, small_number)
   green_corr = max_rgb/max(green, small_number)
   blue_corr  = max_rgb/max(blue, small_number)
   norm = max(blue, math.sqrt(red_corr*red_corr + green_corr*green_corr + blue_corr*blue_corr) / math.sqrt(3), small_number)
   r_gain = red_corr/norm
   g_gain = green_corr/norm
   b_gain = blue_corr/norm
   EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
   return EXPR(clip, expr=['x ' + repr(r_gain) + ' *', 'x ' + repr(g_gain) + ' *', 'x ' + repr(b_gain) + ' *'])

###
# AutoWhite is a function that takes a video clip as an input and calculates the average color values for each of the three color planes (red, green, blue).
# The AutoWhiteAdjust function is then used to adjust the white balance of the input clip based on the color balance of the individual frames.
# This function calculates the correction gain for each color plane (red, green, blue) based on the average color values of each plane, and applies the correction gain to each pixel in the input clip.
# The output is a video clip with corrected white balance.
###
def AutoWhite(clip):
   rgb_clip = clip
   r_avg = core.std.PlaneStats(rgb_clip, plane=0)
   g_avg = core.std.PlaneStats(rgb_clip, plane=1)
   b_avg = core.std.PlaneStats(rgb_clip, plane=2)
   return core.std.FrameEval(rgb_clip, partial(AutoWhiteAdjust, clip=rgb_clip, core=core), prop_src=[r_avg, g_avg, b_avg])

# ToneMapping Simple
def tm(clip="",source_peak="",desat=50,lin=True,show_satmask=False,show_clipped=False ) :
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
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    tm = EXPR(c, expr="x  {exposure_bias} * 0.15 x  {exposure_bias} * * 0.05 + * 0.004 + x  {exposure_bias} * 0.15 x  {exposure_bias} * * 0.50 + * 0.06 + / 0.02 0.30 / -  ".format(exposure_bias=exposure_bias),format=vs.RGBS)
    w=((exposure_bias*(0.15*exposure_bias+0.10*0.50)+0.20*0.02)/(exposure_bias*(0.15*exposure_bias+0.50)+0.20*0.30))-0.02/0.30
    tm = EXPR(clips=[tm,c], expr="x  1 {w}  / * ".format(exposure_bias=exposure_bias,w=w),format=vs.RGBS)
    vszip = hasattr(core,'vszip')
    tm = core.vszip.Limiter(tm, [0,0,0], [1,1,1]) if vszip else core.std.Limiter(tm, 0, 1)

    if lin == True :
        #linearize the tonemapper curve under 100nits
        tm=EXPR(clips=[tm,o], expr="x {tm_ldr_value} < y {ldr_value_mult} * x ? ".format(tm_ldr_value=tm_ldr_value,ldr_value_mult=ldr_value_mult))
    

    r=core.std.ShufflePlanes(clips=[a], planes=[0], colorfamily=vs.GRAY)
    g=core.std.ShufflePlanes(clips=[a], planes=[1], colorfamily=vs.GRAY)
    b=core.std.ShufflePlanes(clips=[a], planes=[2], colorfamily=vs.GRAY)
    #luminance
    l=EXPR(clips=[r,g,b], expr="x 0.2627 *  y 0.678 * + z 0.0593 * +    ",format=vs.GRAY)


    #value under 100nits after the tone mapping(tm_ldr_value) becomes 0, even tm_ldr_value becomes 0 and then scale all in the 0-1 range fo the mask 
    mask=EXPR(clips=[l], expr="x {ldr_value_mult} * {tm_ldr_value}  - {ldr_value_mult} /".format(ldr_value_mult=ldr_value_mult,tm_ldr_value=tm_ldr_value))
    mask = core.vszip.Limiter(mask, 0, 1) if vszip else core.std.Limiter(mask, 0, 1)


    #reduce the saturation blending with grayscale
    lrgb=core.std.ShufflePlanes(clips=[l,l,l], planes=[0,0,0], colorfamily=vs.RGB)
    asat=EXPR(clips=[a,lrgb], expr=" y {desat} * x 1 {desat} - * + ".format(tm_ldr_value=tm_ldr_value,desat=desat))
    a=core.std.MaskedMerge(a, asat, mask)


    r=core.std.ShufflePlanes(clips=[a], planes=[0], colorfamily=vs.GRAY)
    g=core.std.ShufflePlanes(clips=[a], planes=[1], colorfamily=vs.GRAY)
    b=core.std.ShufflePlanes(clips=[a], planes=[2], colorfamily=vs.GRAY)

    rl=core.std.ShufflePlanes(clips=[tm], planes=[0], colorfamily=vs.GRAY)
    gl=core.std.ShufflePlanes(clips=[tm], planes=[1], colorfamily=vs.GRAY)
    bl=core.std.ShufflePlanes(clips=[tm], planes=[2], colorfamily=vs.GRAY)
    l2=EXPR(clips=[rl,gl,bl], expr="x 0.2627 *  y 0.678 * + z 0.0593 * +    ",format=vs.GRAY)
    nl= l2
    scale=EXPR(clips=[nl,l], expr="x y / ")
    r1=EXPR(clips=[r,scale], expr="x  y *")
    g1=EXPR(clips=[g,scale], expr="x  y *")
    b1=EXPR(clips=[b,scale], expr="x  y *")

    c=core.std.ShufflePlanes(clips=[r1,g1,b1], planes=[0,0,0], colorfamily=vs.RGB)
    c = core.vszip.Limiter(c, [0,0,0], [1,1,1]) if vszip else core.std.Limiter(c, 0, 1)

    if show_satmask == True :
        #show mask
        c=core.std.ShufflePlanes(clips=[mask,mask,mask], planes=[0,0,0], colorfamily=vs.RGB)

    if show_clipped == True :
        #show clipped
        c=EXPR(clips=[o], expr="x {ldr_value_mult} * ".format(ldr_value_mult=ldr_value_mult))
 
    return c 

def hablehdr10tosdr(clip, source_peak=1000, desat=50, tFormat=vs.YUV420P8, tMatrix="709", tRange="limited", color_loc="center", f_a=0.0, f_b=0.75, show_satmask=False,lin=True,show_clipped=False) :
  core = vs.core
  clip=core.resize.Bicubic(clip=clip, format=vs.RGBS, filter_param_a=f_a, filter_param_b=f_b, range_in_s="limited", matrix_in_s="2020ncl", primaries_in_s="2020", primaries_s="2020", transfer_in_s="st2084", transfer_s="linear",dither_type="none", nominal_luminance=1000)
  clip=tm(clip=clip,source_peak=source_peak,desat=desat,show_satmask=show_satmask,lin=lin,show_clipped=show_clipped) 
  if tFormat != vs.RGBS:
    clip=core.resize.Bicubic(clip=clip, format=tFormat, filter_param_a=f_a, filter_param_b=f_b, matrix_s=tMatrix, primaries_in_s="2020", primaries_s=tMatrix, transfer_in_s="linear", transfer_s=tMatrix, dither_type="ordered")
    
  return clip
  
def tm_simple(clip="",source_peak="" ) :
    core = vs.core
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
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    tm = EXPR(c, expr="x  {exposure_bias} * 0.15 x  {exposure_bias} * * 0.05 + * 0.004 + x  {exposure_bias} * 0.15 x  {exposure_bias} * * 0.50 + * 0.06 + / 0.02 0.30 / -  ".format(exposure_bias=exposure_bias),format=vs.RGBS)
    w=((exposure_bias*(0.15*exposure_bias+0.10*0.50)+0.20*0.02)/(exposure_bias*(0.15*exposure_bias+0.50)+0.20*0.30))-0.02/0.30
    tm = EXPR(clips=[tm,c], expr="x  1 {w}  / * ".format(exposure_bias=exposure_bias,w=w),format=vs.RGBS)
    vszip = hasattr(core,'vszip')
    tm = core.vszip.Limiter(tm, [0,0,0], [1,1,1]) if vszip else core.std.Limiter(tm, 0, 1)

    r=core.std.ShufflePlanes(clips=[a], planes=[0], colorfamily=vs.GRAY)
    g=core.std.ShufflePlanes(clips=[a], planes=[1], colorfamily=vs.GRAY)
    b=core.std.ShufflePlanes(clips=[a], planes=[2], colorfamily=vs.GRAY)
    l=EXPR(clips=[r,g,b], expr="x 0.2627 *  y 0.678 * + z 0.0593 * +    ",format=vs.GRAY)
    #l=0.2627 * R + 0.678 * G + 0.0593 * B (formula for rec2020 luminance)
    l=EXPR(clips=[l], expr="x 0 =  0.00001 x  ? ",format=vs.GRAY)#avoid dividing by 0

    rl=core.std.ShufflePlanes(clips=[tm], planes=[0], colorfamily=vs.GRAY)
    gl=core.std.ShufflePlanes(clips=[tm], planes=[1], colorfamily=vs.GRAY)
    bl=core.std.ShufflePlanes(clips=[tm], planes=[2], colorfamily=vs.GRAY)
    l2=EXPR(clips=[rl,gl,bl], expr="x 0.2627 *  y 0.678 * + z 0.0593 * +    ",format=vs.GRAY)

    #r1=((r/l)^s)*l2
    #g1=((g/l)^s)*l2
    #b1=((b/l)^s)*l2
    r1=EXPR(clips=[r,l,l2], expr="x  y /  {s} pow  z * ".format(s=s))
    g1=EXPR(clips=[g,l,l2], expr="x  y /  {s} pow  z * ".format(s=s))
    b1=EXPR(clips=[b,l,l2], expr="x  y /  {s} pow z *  ".format(s=s))

    r2=EXPR(clips=[r,l,l2], expr="x  y /   z * ")
    g2=EXPR(clips=[g,l,l2], expr="x  y /   z * ")
    b2=EXPR(clips=[b,l,l2], expr="x  y /  z *  ")
    mask=l2
    mask = core.vszip.Limiter(mask, 0, 1) if vszip else core.std.Limiter(mask, 0, 1)

    csat=core.std.ShufflePlanes(clips=[r1,g1,b1], planes=[0,0,0], colorfamily=vs.RGB)

    c=core.std.ShufflePlanes(clips=[r2,g2,b2], planes=[0,0,0], colorfamily=vs.RGB)
    c2=core.std.MaskedMerge(c, csat, mask)
    c=core.std.Merge(c, c2, 0.5)
    c = core.vszip.Limiter(c, [0,0,0], [1,1,1]) if vszip else core.std.Limiter(c, 0, 1)

    return c 


   
def simplehdr10tosdr(clip, source_peak=1000, tFormat=vs.YUV420P8, tMatrix="709", tRange="limited", color_loc="center", f_a=0.0, f_b=0.75) :
  core = vs.core
  clip=core.resize.Bicubic(clip=clip, format=vs.RGBS,filter_param_a=f_a,filter_param_b=f_b, range_in_s="limited", matrix_in_s="2020ncl", primaries_in_s="2020", primaries_s="2020", transfer_in_s="st2084", transfer_s="linear",dither_type="none", nominal_luminance=1000)
  clip=tm_simple(clip=clip,source_peak=source_peak)
  if tFormat != vs.RGBS:
    clip=core.resize.Bicubic(clip=clip, format=tFormat, filter_param_a=f_a, filter_param_b=f_b, matrix_s=tMatrix, primaries_in_s="2020", primaries_s=tMatrix, transfer_in_s="linear", transfer_s=tMatrix, dither_type="ordered")
  return clip
  
  
 
def ClipRGB(clip: vs.VideoNode, min8: int = 16, max8: int = 235) -> vs.VideoNode:
    """
    Hard-clips all RGB channels of a clip to a specified limited-range interval,
    scaled to the clip's bit depth. Useful for enforcing broadcast-safe RGB levels.

    Args:
        clip (vs.VideoNode): Input RGB clip.
        min8 (int): Lower bound of the range (in 8-bit scale). Default is 16.
        max8 (int): Upper bound of the range (in 8-bit scale). Default is 235.

    Returns:
        vs.VideoNode: RGB clip with channels clipped to the specified range.
    """
    if not clip.format.color_family == vs.RGB:
        raise ValueError("ClipRGB: input must be RGB")

    bits = clip.format.bits_per_sample
    peak = (1 << bits) - 1

    # Scale user-defined 8-bit min/max to current bit depth
    lo = int(round(min8 * peak / 255))
    hi = int(round(max8 * peak / 255))

    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    expr = f'x {lo} max {hi} min'
    return EXPR(clip, [expr] * 3)
