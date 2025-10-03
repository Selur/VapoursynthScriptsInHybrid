import vapoursynth as vs
from vapoursynth import core

import math
from functools import partial
from typing import Optional, Union, Sequence, Any, Dict

# taken from adjust
def Tweak(clip, hue=None, sat=None, bright=None, cont=None, coring=True):
    if clip.format is None:
        raise vs.Error("Tweak: only clips with constant format are accepted.")

    if clip.format.color_family == vs.RGB:
        raise vs.Error("Tweak: RGB clips are not accepted.")
    EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else (core.akarin.Expr if hasattr(core, 'akarin') else core.std.Expr)
        
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

            clip = EXPR(clip, expr=[expression, "", ""])

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

def sine_expr(expr: str) -> str:
    return f'{expr} pi * 0.5 * sin'

def scale(val: int, peak: Union[int, float]) -> Union[int, float]:
    return val * peak / 255 if peak != 1.0 else val / 255
    
def SmoothLevels(
    input: vs.VideoNode,
    input_low: Union[int, float] = 0,
    gamma: float = 1.0,
    input_high: Optional[Union[int, float]] = None,
    output_low: Union[int, float] = 0,
    output_high: Optional[Union[int, float]] = None,
    chroma: int = 50,
    limiter: int = 0,
    Lmode: int = 0,
    DarkSTR: int = 100,
    BrightSTR: int = 100,
    Ecenter: Optional[Union[int, float]] = None,
    protect: Union[int, float] = -1,
    Ecurve: int = 0,
    Smode: int = -2,
    Mfactor: Union[int, float] = 2,
    RGmode: int = 12,
    useDB: bool = False
) -> vs.VideoNode:
    """Optimized SmoothLevels function with performance improvements."""
    
    # Initial validation and format setup
    if not isinstance(input, vs.VideoNode):
        raise vs.Error('SmoothLevels: this is not a clip')
    
    if input.format.color_family == vs.RGB:
        raise vs.Error('SmoothLevels: RGB format is not supported')

    core = vs.core
    EXPR = core.akarin.Expr if hasattr(core, 'akarin') else core.std.Expr
    
    # Precompute format-dependent values
    bits = input.format.bits_per_sample
    is_float = input.format.sample_type == vs.FLOAT
    is_gray = input.format.color_family == vs.GRAY
    neutral = [0.5, 0.0] if is_float else [1 << (bits - 1)] * 2
    peak = 1.0 if is_float else (1 << bits) - 1
    
    # Set default values
    input_high = peak if input_high is None else input_high
    output_high = peak if output_high is None else output_high
    Ecenter = neutral[0] if Ecenter is None else Ecenter
    
    # Parameter validation
    if gamma <= 0:
        raise vs.Error('SmoothLevels: gamma must be greater than 0.0')
    if Ecenter <= 0 or Ecenter >= peak:
        raise vs.Error('SmoothLevels: Ecenter must be within valid range')
    if Mfactor <= 0:
        raise vs.Error('SmoothLevels: Mfactor must be greater than 0')

    # Handle chroma processing
    if chroma <= 0 and not is_gray:
        input_orig = input
        input = core.std.ShufflePlanes(input, planes=0, colorfamily=vs.GRAY)
    else:
        input_orig = None

    # RemoveGrain mode selection
    RG_MAP = {
        4: core.zsmooth.Median if hasattr(core, 'zsmooth') else core.std.Median,
        11: partial(core.std.Convolution, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1]),
        12: partial(core.std.Convolution, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1]),
        19: partial(core.std.Convolution, matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1]),
        20: partial(core.std.Convolution, matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])
    }
    
    RemoveGrain = RG_MAP.get(RGmode)
    if RemoveGrain is None:
        RG = core.zsmooth.RemoveGrain if hasattr(core, 'zsmooth') else core.rgvs.RemoveGrain
        RemoveGrain = partial(RG, mode=[RGmode])

    # Build expressions
    denom = input_high - input_low + (1 if input_high == input_low else 0)
    exprY = f'x {input_low} - {denom} / {1/gamma} pow {output_high - output_low} * {output_low} +'
    
    if chroma > 0 and not is_gray:
        scaleC = ((output_high - output_low) / denom + 100/chroma - 1) / (100/chroma)
        exprC = f'x {neutral[1]} - {scaleC} * {neutral[1]} +'

    # Strength calculations
    Dstr = DarkSTR / 100
    Bstr = BrightSTR / 100

    # Luma expression
    if Lmode <= 0:
        exprL = '1'
    elif Ecurve <= 0:
        if Lmode == 1:
            var_d = f'x {Ecenter} /'
            var_b = f'{peak} x - {peak} {Ecenter} - /'
            exprL = f'x {Ecenter} < ' + sine_expr(var_d) + f' {Dstr} pow x {Ecenter} > ' + sine_expr(var_b) + f' {Bstr} pow 1 ? ?'
        elif Lmode == 2:
            exprL = sine_expr(f'x {peak} /') + f' {Dstr} pow'
        else:
            exprL = sine_expr(f'{peak} x - {peak} /') + f' {Bstr} pow'
    else:
        if Lmode == 1:
            exprL = f'x {Ecenter} < x {Ecenter} / abs {Dstr} pow x {Ecenter} > 1 x {Ecenter} - {peak-Ecenter} / abs - {Bstr} pow 1 ? ?'
        elif Lmode == 2:
            exprL = f'1 x {peak} - {peak} / abs - {Dstr} pow'
        else:
            exprL = f'x {peak} - {peak} / abs {Bstr} pow'

    # Protect expression
    if protect <= -1:
        exprP = '1'
    elif Ecurve <= 0:
        scale_val = scale(16, peak)
        var_p = f'x {protect} - {scale_val} /'
        exprP = f'x {protect} <= 0 x {protect+scale_val} >= 1 ' + sine_expr(var_p) + f' ? ?'
    else:
        scale_val = scale(16, peak)
        exprP = f'x {protect} <= 0 x {protect+scale_val} >= 1 x {protect} - {scale_val} / abs ? ?'

    # Processing pipeline
    limitI = EXPR(input, expr=[f'x {input_low} max {input_high} min']) if limiter in (1, 3) else input
    
    full_expr = exprL + ' ' + exprP + ' * ' + exprY + ' x - * x +'
    level = EXPR(limitI, expr=[full_expr] if chroma <= 0 or is_gray else [full_expr, exprC])
    
    diff_expr = f'x y - {Mfactor} * {neutral[1]} +'
    merge_expr = f'x y {neutral[1]} - {Mfactor} / -'
    
    diff = EXPR([limitI, level], expr=[diff_expr])
    process = RemoveGrain(diff)
    
    if useDB:
        deband_func = core.neo_f3kdb.Deband if hasattr(core, 'neo_f3kdb') else core.f3kdb.Deband
        deband_expr = f'x {neutral[1]} - {Mfactor} / {neutral[1]} +'
        process = deband_func(EXPR(process, expr=[deband_expr]), grainy=0, grainc=0, output_depth=bits)
        smth = core.std.MakeDiff(limitI, process)
    else:
        smth = EXPR([limitI, process], expr=[merge_expr])

    level2 = EXPR([limitI, diff], expr=[merge_expr])
    diff2 = EXPR([level2, level], expr=[diff_expr])
    process2 = RemoveGrain(diff2)
    
    if useDB:
        process2 = deband_func(EXPR(process2, expr=[deband_expr]), grainy=0, grainc=0, output_depth=bits)
        smth2 = core.std.MakeDiff(smth, process2)
    else:
        smth2 = EXPR([smth, process2], expr=[merge_expr])

    # Mask creation
    mask1 = EXPR([limitI, level], expr=[f'x y - abs {neutral[0]/Mfactor} >= {peak} 0 ?'])
    mask2 = EXPR([limitI, level], expr=[f'x y - abs {peak/Mfactor} >= {peak} 0 ?'])

    # Final merge based on Smode
    smode_handlers = {
        -2: lambda: core.std.MaskedMerge(core.std.MaskedMerge(smth, smth2, mask1), level, mask2),
        -1: lambda: core.std.MaskedMerge(smth, level, mask1),
        1: lambda: smth,
        2: lambda: smth2
    }
    Slevel = smode_handlers.get(Smode, lambda: level)()

    # Output limiting
    limitO = EXPR(Slevel, expr=[f'x {output_low} max {output_high} min']) if limiter >= 2 else Slevel

    # Restore chroma if needed
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
    elif method != "Luma":
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

  EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else (core.akarin.Expr if hasattr(core, 'akarin') else core.std.Expr)
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
        
        EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else (core.akarin.Expr if hasattr(core, 'akarin') else core.std.Expr)
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
   EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else (core.akarin.Expr if hasattr(core, 'akarin') else core.std.Expr)
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
    EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else (core.akarin.Expr if hasattr(core, 'akarin') else core.std.Expr)
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
    EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else (core.akarin.Expr if hasattr(core, 'akarin') else core.std.Expr)
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
  
  
  
# Taken from muvsfunc
# Type aliases
def SmoothGrad(input: vs.VideoNode, radius: int = 9, thr: float = 0.25,
               ref: Optional[vs.VideoNode] = None, elast: float = 3.0,
               planes: Optional[Union[int, Sequence[int]]] = None, **limit_filter_args: Any) -> vs.VideoNode:
    '''Avisynth's SmoothGrad

    SmoothGrad smooths the low gradients or flat areas of a 16-bit clip.
    It proceeds by applying a huge blur filter and comparing the result with the input data for each pixel.
    If the difference is below the specified threshold, the filtered version is taken into account,
        otherwise the input pixel remains unchanged.

    Args:
        input: Input clip to be filtered.

        radius: (int) Size of the averaged square. Its width is radius*2-1. Range is 2-9.

        thr: (float) Threshold between reference data and filtered data, on an 8-bit scale.

        ref: Reference clip for the filter output comparison. Specify here the input clip when you cascade several SmoothGrad calls.
            When undefined, the input clip is taken as reference.

        elast: (float) To avoid artifacts, the threshold has some kind of elasticity.
            Value differences falling over this threshold are gradually attenuated, up to thr * elast > 1.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from the source clip, "input".

        limit_filter_args: (dict) Additional arguments passed to LimitFilter in the form of keyword arguments.

    '''
    if planes is None:
        planes = list(range(input.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    # process
    smooth = BoxFilter(input, radius, planes=planes)

    return LimitFilter(smooth, input, ref, thr, elast, planes=planes, **limit_filter_args)
 
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

    EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else (core.akarin.Expr if hasattr(core, 'akarin') else core.std.Expr)
    expr = f'x {lo} max {hi} min'
    return EXPR(clip, [expr] * 3)

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
        EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else (core.akarin.Expr if hasattr(core, 'akarin') else core.std.Expr)
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


def BoxFilter(input: vs.VideoNode, radius: int = 16, radius_v: Optional[int] = None, planes: Optional[Union[int, Sequence[int]]] = None,
              fmtc_conv: int = 0, radius_thr: Optional[int] = None,
              resample_args: Optional[Dict[str, Any]] = None, keep_bits: bool = True,
              depth_args: Optional[Dict[str, Any]] = None
              ) -> vs.VideoNode:
    '''Box filter

    Performs a box filtering on the input clip.
    Box filtering consists in averaging all the pixels in a square area whose center is the output pixel.
    You can approximate a large gaussian filtering by cascading a few box filters.

    Args:
        input: Input clip to be filtered.

        radius, radius_v: (int) Size of the averaged square. The size is (radius*2-1) * (radius*2-1).
            If "radius_v" is None, it will be set to "radius".
            Default is 16.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from the source clip, "input".

        fmtc_conv: (0~2) Whether to use fmtc.resample for convolution.
            It's recommended to input clip without chroma subsampling when using fmtc.resample, otherwise the output may be incorrect.
            0: False. 1: True (except both "radius" and "radius_v" is strictly smaller than 4).
                2: Auto, determined by radius_thr (exclusive).
            Default is 0.

        radius_thr: (int) Threshold of wheter to use fmtc.resample when "fmtc_conv" is 2.
            Default is 11 for integer input and 21 for float input.
            Only works when "fmtc_conv" is enabled.

        resample_args: (dict) Additional parameters passed to core.fmtc.resample in the form of dict.
            It's recommended to set "flt" to True for higher precision, like:
                flt = muf.BoxFilter(src, resample_args=dict(flt=True))
            Only works when "fmtc_conv" is enabled.
            Default is {}.

        keep_bits: (bool) Whether to keep the bitdepth of the output the same as input.
            Only works when "fmtc_conv" is enabled and input is integer.

        depth_args: (dict) Additional parameters passed to mvf.Depth in the form of dict.
            Only works when "fmtc_conv" is enabled, input is integer and "keep_bits" is True.
            Default is {}.

    '''

    funcName = 'BoxFilter'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if planes is None:
        planes = list(range(input.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    if radius_v is None:
        radius_v = radius

    if radius == radius_v == 1:
        return input

    if radius_thr is None:
        radius_thr = 21 if input.format.sample_type == vs.FLOAT else 11 # Values are measured from my experiment

    if resample_args is None:
        resample_args = {}

    if depth_args is None:
        depth_args = {}

    planes2 = [(3 if i in planes else 2) for i in range(input.format.num_planes)]
    width = radius * 2 - 1
    width_v = radius_v * 2 - 1
    kernel = [1 / width] * width
    kernel_v = [1 / width_v] * width_v

    # process
    if input.format.sample_type == vs.FLOAT:
        if core.version_number() < 33:
            raise NotImplementedError(funcName + (': Please update your VapourSynth.'
                'BoxBlur on float sample has not yet been implemented on current version.'))
        elif radius == radius_v == 2 or radius == radius_v == 3:
            return core.std.Convolution(input, [1] * ((radius * 2 - 1) * (radius * 2 - 1)), planes=planes, mode='s')

        else:
            if fmtc_conv == 1 or (fmtc_conv != 0 and radius > radius_thr): # Use fmtc.resample for convolution
                flt = core.fmtc.resample(input, kernel='impulse', impulseh=kernel, impulsev=kernel_v, planes=planes2,
                    cnorm=False, fh=-1, fv=-1, center=False, **resample_args)
                return flt # No bitdepth conversion is required since fmtc.resample outputs the same bitdepth as input

            elif hasattr(core, 'vszip'):
                return core.vszip.BoxBlur(input, hradius=radius-1, vradius=radius_v-1, planes=planes)

            elif core.version_number() >= 39:
                return core.std.BoxBlur(input, hradius=radius-1, vradius=radius_v-1, planes=planes)

            else: # BoxBlur on float sample has not been implemented
                if radius > 1:
                    input = core.std.Convolution(input, [1] * (radius * 2 - 1), planes=planes, mode='h')
                if radius_v > 1:
                    input = core.std.Convolution(input, [1] * (radius_v * 2 - 1), planes=planes, mode='v')
                return input

    else: # input.format.sample_type == vs.INTEGER
        if radius == radius_v == 2 or radius == radius_v == 3:
            return core.std.Convolution(input, [1] * ((radius * 2 - 1) * (radius * 2 - 1)), planes=planes, mode='s')

        else:
            if fmtc_conv == 1 or (fmtc_conv != 0 and radius > radius_thr): # Use fmtc.resample for convolution
                flt = core.fmtc.resample(input, kernel='impulse', impulseh=kernel, impulsev=kernel_v, planes=planes2,
                    cnorm=False, fh=-1, fv=-1, center=False, **resample_args)
                if keep_bits and input.format.bits_per_sample != flt.format.bits_per_sample:
                    flt = mvf.Depth(flt, depth=input.format.bits_per_sample, **depth_args)
                return flt

            elif hasattr(core, 'vszip'):
                return core.vszip.BoxBlur(input, hradius=radius-1, vradius=radius_v-1, planes=planes)

            elif hasattr(core.std, 'BoxBlur'):
                return core.std.BoxBlur(input, hradius=radius-1, vradius=radius_v-1, planes=planes)

            else: # BoxBlur was not found
                if radius > 1:
                    input = core.std.Convolution(input, [1] * (radius * 2 - 1), planes=planes, mode='h')
                if radius_v > 1:
                    input = core.std.Convolution(input, [1] * (radius_v * 2 - 1), planes=planes, mode='v')
                return input
                
                
################################################################################################################################
## Main function: Depth()
################################################################################################################################
## Bit depth conversion with dithering (if needed).
## It's a wrapper for fmtc.bitdepth and zDepth (core.resize/zimg).
## Only constant format is supported, frame properties of the input clip is mostly ignored (only available with zDepth).
################################################################################################################################
## Basic parameters
##     input {clip}: clip to be converted
##         can be of YUV/RGB/Gray color family, can be of 8~16 bit integer or 16/32 bit float
##     depth {int}: output bit depth, can be 1~16 bit integer or 16/32 bit float
##         note that 1~7 bit content is still stored as 8 bit integer format
##         default is the same as that of the input clip
##     sample {int}: output sample type, can be 0 (vs.INTEGER) or 1 (vs.FLOAT)
##         default is the same as that of the input clip
##     fulls {bool}: define if input clip is of full range
##         default: None, assume True for RGB/YCgCo input, assume False for Gray/YUV input
##     fulld {bool}: define if output clip is of full range
##         default is the same as "fulls"
################################################################################################################################
## Advanced parameters
##     dither {int|str}: dithering algorithm applied for depth conversion
##         - {int}: same as "dmode" in fmtc.bitdepth, will be automatically converted if using zDepth
##         - {str}: same as "dither_type" in zDepth, will be automatically converted if using fmtc.bitdepth
##         - default:
##             - output depth is 32, and conversions without quantization error: 1 | "none"
##             - otherwise: 3 | "error_diffusion"
##     useZ {bool}: prefer zDepth or fmtc.bitdepth for depth conversion
##         When 11,13~15 bit integer or 16 bit float is involved, zDepth is always used.
##         - False: prefer fmtc.bitdepth
##         - True: prefer zDepth
##         default: False
################################################################################################################################
## Parameters of fmtc.bitdepth
##     ampo, ampn, dyn, staticnoise, cpuopt, patsize, tpdfo, tpdfn, corplane:
##         same as those in fmtc.bitdepth, ignored when useZ=True
##         *NOTE* no positional arguments, only keyword arguments are accepted
################################################################################################################################
def Depth(input, depth=None, sample=None, fulls=None, fulld=None,
    dither=None, useZ=None, **kwargs):
    # input clip
    clip = input

    if not isinstance(input, vs.VideoNode):
        raise type_error('"input" must be a clip!')

    ## Default values for kwargs
    if 'ampn' not in kwargs:
        kwargs['ampn'] = None
    if 'ampo' not in kwargs:
        kwargs['ampo'] = None

    # Get properties of input clip
    sFormat = input.format

    sColorFamily = sFormat.color_family
    CheckColorFamily(sColorFamily)
    sIsYUV = sColorFamily == vs.YUV
    sIsGRAY = sColorFamily == vs.GRAY

    sbitPS = sFormat.bits_per_sample
    sSType = sFormat.sample_type

    if fulls is None:
        # If not set, assume limited range for YUV and Gray input
        fulls = False if sIsYUV or sIsGRAY else True
    elif not isinstance(fulls, int):
        raise type_error('"fulls" must be a bool!')

    # Get properties of output clip
    lowDepth = False

    if depth is None:
        dbitPS = sbitPS
    elif not isinstance(depth, int):
        raise type_error('"depth" must be an int!')
    else:
        if depth < 8:
            dbitPS = 8
            lowDepth = True
        else:
            dbitPS = depth
    if sample is None:
        if depth is None:
            dSType = sSType
            depth = dbitPS
        else:
            dSType = vs.FLOAT if dbitPS >= 32 else vs.INTEGER
    elif not isinstance(sample, int):
        raise type_error('"sample" must be an int!')
    elif sample != vs.INTEGER and sample != vs.FLOAT:
        raise value_error('"sample" must be either 0 (vs.INTEGER) or 1 (vs.FLOAT)!')
    else:
        dSType = sample
    if depth is None and sSType != vs.FLOAT and sample == vs.FLOAT:
        dbitPS = 32
    elif depth is None and sSType != vs.INTEGER and sample == vs.INTEGER:
        dbitPS = 16
    if dSType == vs.INTEGER and (dbitPS < 1 or dbitPS > 16):
        raise value_error(f'{dbitPS}-bit integer output is not supported!')
    if dSType == vs.FLOAT and (dbitPS != 16 and dbitPS != 32):
        raise value_error(f'{dbitPS}-bit float output is not supported!')

    if fulld is None:
        fulld = fulls
    elif not isinstance(fulld, int):
        raise type_error('"fulld" must be a bool!')

    # Low-depth support
    if lowDepth:
        if dither == "none" or dither == 1:
            clip = _quantization_conversion(clip, sbitPS, depth, vs.INTEGER, fulls, fulld, False, False, 8, 0)
            clip = _quantization_conversion(clip, depth, 8, vs.INTEGER, fulld, fulld, False, False, 8, 0)
            return clip
        else:
            full = fulld
            clip = _quantization_conversion(clip, sbitPS, depth, vs.INTEGER, fulls, full, False, False, 16, 1)
            sSType = vs.INTEGER
            sbitPS = 16
            fulls = False
            fulld = False

    # Whether to use zDepth or fmtc.bitdepth for conversion
    # When 11,13~15 bit integer or 16 bit float is involved, force using zDepth
    if useZ is None:
        useZ = False
    elif not isinstance(useZ, int):
        raise type_error('"useZ" must be a bool!')
    if sSType == vs.INTEGER and (sbitPS == 13 or sbitPS == 15):
        useZ = True
    if dSType == vs.INTEGER and (dbitPS == 11 or 13 <= dbitPS <= 15):
        useZ = True
    if (sSType == vs.FLOAT and sbitPS < 32) or (dSType == vs.FLOAT and dbitPS < 32):
        useZ = True

    # Dithering type
    if kwargs['ampn'] is not None and not isinstance(kwargs['ampn'], (int, float)):
        raise type_error('"ampn" must be an int or a float!')

    if dither is None:
        if dbitPS == 32 or (dbitPS >= sbitPS and fulld == fulls and fulld == False):
            dither = "none" if useZ else 1
        else:
            dither = "error_diffusion" if useZ else 3
    elif not isinstance(dither, (int, str)):
        raise type_error('"dither" must be an int or a str!')
    else:
        if isinstance(dither, str):
            dither = dither.lower()
            if dither != "none" and dither != "ordered" and dither != "random" and dither != "error_diffusion":
                raise value_error('Unsupported "dither" specified!')
        else:
            if dither < 0 or dither > 9:
                raise value_error('Unsupported "dither" specified!')
        if useZ and isinstance(dither, int):
            if dither == 0:
                dither = "ordered"
            elif dither == 1 or dither == 2:
                if kwargs['ampn'] is not None and kwargs['ampn'] > 0:
                    dither = "random"
                else:
                    dither = "none"
            else:
                dither = "error_diffusion"
        elif not useZ and isinstance(dither, str):
            if dither == "none":
                dither = 1
            elif dither == "ordered":
                dither = 0
            elif dither == "random":
                if kwargs['ampn'] is None:
                    dither = 1
                    kwargs['ampn'] = 1
                elif kwargs['ampn'] > 0:
                    dither = 1
                else:
                    dither = 3
            else:
                dither = 3

    if not useZ:
        if kwargs['ampo'] is None:
            kwargs['ampo'] = 1.5 if dither == 0 else 1
        elif not isinstance(kwargs['ampo'], (int, float)):
            raise type_error('"ampo" must be an int or a float!')

    # Skip processing if not needed
    if dSType == sSType and dbitPS == sbitPS and (sSType == vs.FLOAT or fulld == fulls) and not lowDepth:
        return clip

    # Apply conversion
    if useZ:
        clip = zDepth(clip, sample=dSType, depth=dbitPS, range=fulld, range_in=fulls, dither_type=dither)
    else:
        clip = core.fmtc.bitdepth(clip, bits=dbitPS, flt=dSType, fulls=fulls, fulld=fulld, dmode=dither, **kwargs)
        clip = SetColorSpace(clip, ColorRange=0 if fulld else 1)

    # Low-depth support
    if lowDepth:
        clip = _quantization_conversion(clip, depth, 8, vs.INTEGER, full, full, False, False, 8, 0)

    # Output
    return clip
################################################################################################################################

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
## Frame property function: SetColorSpace()
################################################################################################################################
## Modify the color space related frame properties in the given clip.
## Detailed descriptions of these properties: http://www.vapoursynth.com/doc/apireference.html
################################################################################################################################
## Parameters
##     %Any%: for the property named "_%Any%"
##         - None: do nothing
##         - True: do nothing
##         - False: delete corresponding frame properties if exist
##         - {int}: set to this value
################################################################################################################################
def SetColorSpace(clip, ChromaLocation=None, ColorRange=None, Primaries=None, Matrix=None, Transfer=None):
    # input clip
    if not isinstance(clip, vs.VideoNode):
        raise type_error('"clip" must be a clip!')

    # Modify frame properties
    if ChromaLocation is None:
        pass
    elif isinstance(ChromaLocation, bool):
        if ChromaLocation is False:
            clip = RemoveFrameProp(clip, '_ChromaLocation')
    elif isinstance(ChromaLocation, int):
        if ChromaLocation >= 0 and ChromaLocation <=5:
            clip = core.std.SetFrameProp(clip, prop='_ChromaLocation', intval=ChromaLocation)
        else:
            raise value_error('valid range of "ChromaLocation" is [0, 5]!')
    else:
        raise type_error('"ChromaLocation" must be an int or a bool!')

    if ColorRange is None:
        pass
    elif isinstance(ColorRange, bool):
        if ColorRange is False:
            clip = RemoveFrameProp(clip, '_ColorRange')
    elif isinstance(ColorRange, int):
        if ColorRange >= 0 and ColorRange <=1:
            clip = core.std.SetFrameProp(clip, prop='_ColorRange', intval=ColorRange)
        else:
            raise value_error('valid range of "ColorRange" is [0, 1]!')
    else:
        raise type_error('"ColorRange" must be an int or a bool!')

    if Primaries is None:
        pass
    elif isinstance(Primaries, bool):
        if Primaries is False:
            clip = RemoveFrameProp(clip, '_Primaries')
    elif isinstance(Primaries, int):
        clip = core.std.SetFrameProp(clip, prop='_Primaries', intval=Primaries)
    else:
        raise type_error('"Primaries" must be an int or a bool!')

    if Matrix is None:
        pass
    elif isinstance(Matrix, bool):
        if Matrix is False:
            clip = RemoveFrameProp(clip, '_Matrix')
    elif isinstance(Matrix, int):
        clip = core.std.SetFrameProp(clip, prop='_Matrix', intval=Matrix)
    else:
        raise type_error('"Matrix" must be an int or a bool!')

    if Transfer is None:
        pass
    elif isinstance(Transfer, bool):
        if Transfer is False:
            clip = RemoveFrameProp(clip, '_Transfer')
    elif isinstance(Transfer, int):
        clip = core.std.SetFrameProp(clip, prop='_Transfer', intval=Transfer)
    else:
        raise type_error('"Transfer" must be an int or a bool!')

    # Output
    return clip
################################################################################################################################