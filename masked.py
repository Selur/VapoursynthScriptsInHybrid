import vapoursynth as vs
from math import sqrt

from typing import Union
import misc
from helpers import GetPlane, Depth, BoxFilter

core = vs.core

# collection of Mask filters.

# Use retinex to greatly improve the accuracy of the edge detection in dark scenes.
# draft=True is a lot faster, albeit less accurate
# from https://blog.kageru.moe/legacy/edgemasks.html
def retinex_edgemask(src: vs.VideoNode, sigma: int = 1, draft: bool = False) -> vs.VideoNode:
    """
    Use retinex to greatly improve edge detection in dark scenes.
    sigma is the sigma of tcanny.
    """

    luma = GetPlane(src, 0)
    EXPR = core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr
    max_value = 1.0 if src.format.sample_type == vs.FLOAT else (1 << src.format.bits_per_sample) - 1
    if draft:
        ret = EXPR(luma, 'x 65535 / sqrt 65535 *')
    else:
        ret = core.retinex.MSRCP(luma, sigma=[50, 200, 350], upper_thr=0.005)
    k = kirsch(luma)
    if hasattr(core, "tcanny"):
        tc = ret.tcanny.TCanny(mode=1, sigma=sigma).std.Minimum(coordinates=[1, 0, 1, 0, 0, 1, 0, 1])
        return EXPR([k, tc], f"x y + {max_value} min")
    return EXPR([k, ret], f"x y + {max_value} min")

# Kirsch edge detection. This uses 8 directions, so it's slower but better than Sobel (4 directions).
# more information: https://ddl.kageru.moe/konOJ.pdf
# from https://blog.kageru.moe/legacy/edgemasks.html
def kirsch(src: vs.VideoNode) -> vs.VideoNode:
    w = [5]*3 + [-3]*5
    weights = [w[-i:] + w[:-i] for i in range(4)]
    c = [src.std.Convolution((w[:4]+[0]+w[4:]), saturate=False) for w in weights]
    EXPR = core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr
    return EXPR(c, 'x y max z max a max')


# should behave similar to std.Sobel() but faster since it has no additional high-/lowpass or gain.
# the internal filter is also a little brighter
# from https://blog.kageru.moe/legacy/edgemasks.html
def fast_sobel(src: vs.VideoNode) -> vs.VideoNode:
    sx = src.std.Convolution([-1, -2, -1, 0, 0, 0, 1, 2, 1], saturate=False)
    sy = src.std.Convolution([-1, 0, 1, -2, 0, 2, -1, 0, 1], saturate=False)
    EXPR = core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr
    return EXPR([sx, sy], 'x y max')


# a weird kind of edgemask that draws around the edges. probably needs more tweaking/testing
# maybe useful for edge cleaning?
# from https://blog.kageru.moe/legacy/edgemasks.html
def bloated_edgemask(src: vs.VideoNode) -> vs.VideoNode:
    return src.std.Convolution(matrix=[1,  2,  4,  2, 1,
                                       2, -3, -6, -3, 2,
                                       4, -6,  0, -6, 4,
                                       2, -3, -6, -3, 2,
                                       1,  2,  4,  2, 1], saturate=False)

# https://github.com/DeadNews/dnfunc/blob/f5d22057e424fb3b8bd80d1aadd0c2ed2b7e71d5/dnfunc.py#L1212                                                                              
def kirsch2(clip_y: vs.VideoNode) -> vs.VideoNode:
    n = core.std.Convolution(clip_y, [5, 5, 5, -3, 0, -3, -3, -3, -3], divisor=3, saturate=False)
    nw = core.std.Convolution(clip_y, [5, 5, -3, 5, 0, -3, -3, -3, -3], divisor=3, saturate=False)
    w = core.std.Convolution(clip_y, [5, -3, -3, 5, 0, -3, 5, -3, -3], divisor=3, saturate=False)
    sw = core.std.Convolution(clip_y, [-3, -3, -3, 5, 0, -3, 5, 5, -3], divisor=3, saturate=False)
    s = core.std.Convolution(clip_y, [-3, -3, -3, -3, 0, -3, 5, 5, 5], divisor=3, saturate=False)
    se = core.std.Convolution(clip_y, [-3, -3, -3, -3, 0, 5, -3, 5, 5], divisor=3, saturate=False)
    e = core.std.Convolution(clip_y, [-3, -3, 5, -3, 0, 5, -3, -3, 5], divisor=3, saturate=False)
    ne = core.std.Convolution(clip_y, [-3, 5, 5, -3, 0, 5, -3, -3, -3], divisor=3, saturate=False)
    EXPR = core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr
    return EXPR(
        [n, nw, w, sw, s, se, e, ne],
        ["x y max z max a max b max c max d max e max"],
    )
# from https://github.com/theChaosCoder/lostfunc/blob/master/lostfunc.py -> mfToon2/MfTurd
def scale8(x, newmax):
        return x * newmax // 0xFF


def CartoonEdges(clip, low=0, high=255):
    """Should behave like mt_edge(mode="cartoon")"""
    valuerange = (1 << clip.format.bits_per_sample)
    maxvalue = valuerange - 1
    
    low = scale8(low, maxvalue)
    high = scale8(high, maxvalue)
    edges = core.std.Convolution(clip, matrix=[0,-2,1,0,1,0,0,0,0], saturate=True)
    EXPR = core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr
    return EXPR(edges, ['x {high} >= {maxvalue} x {low} <= 0 x ? ?'
                                 .format(low=low, high=high, maxvalue=maxvalue), ''])

def RobertsEdges(clip, low=0, high=255):
    """Should behave like mt_edge(mode="roberts")"""
    valuerange = (1 << clip.format.bits_per_sample)
    maxvalue = valuerange - 1
    
    low = scale8(low, maxvalue)
    high = scale8(high, maxvalue)
    edges = core.std.Convolution(clip, matrix=[0,0,0,0,2,-1,0,-1,0], divisor=2, saturate=False)
    EXPR = core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr
    return EXPR(edges, ['x {high} >= {maxvalue} x {low} <= 0 x ? ?'
                                 .format(low=low, high=high, maxvalue=maxvalue), ''])

# from https://github.com/dnjulek/jvsfunc/blob/main/jvsfunc/mask.py -> Tcanny
def dehalo_mask(src: vs.VideoNode, expand: float = 0.5, iterations: int = 2, brz: int = 255, shift: int = 8) -> vs.VideoNode:
    if not 0 <= brz <= 255:
        raise ValueError("dehalo_mask: brz must be between 0 and 255.")

    src8 = Depth(src, 8)
    luma = GetPlane(src8, 0)

    EXPR = core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr
    edge = EXPR([luma, luma.std.Maximum().std.Maximum()], [f"y x - {shift} - 128 *"])

    if hasattr(core, "tcanny"):
        mask1 = EXPR(edge.tcanny.TCanny(sigma=sqrt(expand * 2), mode=-1), ["x 16 *"])
    else:
        radius = max(1, round(sqrt(expand * 2) * 1.5))
        mask1 = EXPR(BoxFilter(edge, radius, 3), ["x 16 *"])

    mask2 = edge
    for _ in range(iterations):
        mask2 = core.std.Maximum(mask2)
    for _ in range(iterations):
        mask2 = core.std.Minimum(mask2)

    mask2 = mask2.std.Invert().std.Binarize(80)

    mask3 = mask2.std.Inflate().std.Inflate().std.Binarize(brz)
    mask4 = mask3 if brz < 255 else mask2
    mask4 = mask4.std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])

    return Depth(EXPR([mask1, mask4], ["x y min"]), src.format.bits_per_sample, range=1)


def hue_mask(clip: vs.VideoNode, min_hue: Union[float, int], max_hue: Union[float, int]) -> vs.VideoNode:
    """
    Creates a mask based on a given hue range, supporting all RGB-based color spaces.
    
    Parameters:
        clip (vs.VideoNode): Input clip in any RGB-based format.
        min_hue (float | int): Minimum hue value (normalized, 0.0 to 1.0).
        max_hue (float | int): Maximum hue value (normalized, 0.0 to 1.0).
        
    Returns:
        vs.VideoNode: Mask clip with white (255) for pixels within the hue range, black (0) otherwise.
    """
    if clip.format.color_family != vs.RGB:
        raise ValueError("Input clip must be in an RGB-based format.")
    
    # Ensure input is in a supported RGB format with consistent bit depth
    if clip.format.bits_per_sample != 8:
        clip = core.resize.Bicubic(clip, format=vs.RGB24)
    
    # Convert to HSL
    hsl_clip = core.resize.Bicubic(clip, format=vs.YUV444P8, matrix_in_s="709")
    hue = core.std.ShufflePlanes(hsl_clip, planes=0, colorfamily=vs.GRAY)
    
    EXPR = core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr
    # Build the mask
    mask = EXPR(
        [hue],
        expr=f"x {min_hue} >= x {max_hue} <= and 255 0 ?"
    )
    
    # Ensure mask is 8-bit grayscale
    return core.resize.Bicubic(mask, format=vs.GRAY8)


def FinegrainMask(clip: vs.VideoNode, mode: str="RemoveGrain") -> vs.VideoNode:
    """
    Create a fine detail mask using RemoveGrain for smoothing and difference detection.

    Parameters:
    clip      : Input clip (must be YUV or GRAY, int or float).
    mode      : Smoothing mode to use.

    Returns:
    A binary mask highlighting fine detail areas.
    """

    # Check format and get bit depth
    isFLOAT = clip.format.sample_type == vs.FLOAT
    if isFLOAT:
        peak = 1.0
    else:
        peak = (1 << clip.format.bits_per_sample) - 1

    # Extract luma if needed
    luma = core.std.ShufflePlanes(clip, planes=0, colorfamily=vs.GRAY)

    if mode == "RemoveGrain":
      rgMode = 22
      # Smooth with RemoveGrain
      if hasattr(core, 'zsmooth'):
        smoothed = core.zsmooth.RemoveGrain(clip=luma, mode=rgMode)
      elif hasattr(core, 'rgsf') and isFLOAT:  
        smoothed =  core.rgsf.RemoveGrain(clip=luma, mode=rgMode)
      else:
        smoothed =  core.rgvs.RemoveGrain(clip=luma, mode=rgMode)
    elif mode == "Bilinear":
      scale = 0.1
      smoothed = bilinear_denoise(clip=luma, scale=scale, rg=True)
    elif mode == "mClean":
      import denoise

      def processWithMClean(clip: vs.VideoNode) -> tuple[vs.VideoNode, int]:
          original_format = clip.format
          is_rgb = original_format.color_family == vs.RGB

          # Bitdepth und Sample-Typ ermitteln
          bits = original_format.bits_per_sample
          sample_type = original_format.sample_type

          # GRAY-Format-Tabelle
          gray_format_map = {
              (vs.INTEGER, 8): vs.GRAY8,
              (vs.INTEGER, 10): vs.GRAY10,
              (vs.INTEGER, 12): vs.GRAY12,
              (vs.INTEGER, 14): vs.GRAY14,
              (vs.INTEGER, 16): vs.GRAY16,
              (vs.FLOAT, 16): vs.GRAYH,
              (vs.FLOAT, 32): vs.GRAYS,
          }

          # YUV444-Format-Tabelle
          yuv444_format_map = {
              (vs.INTEGER, 8): vs.YUV444P8,
              (vs.INTEGER, 10): vs.YUV444P10,
              (vs.INTEGER, 12): vs.YUV444P12,
              (vs.INTEGER, 14): vs.YUV444P14,
              (vs.INTEGER, 16): vs.YUV444P16,
              (vs.FLOAT, 16): vs.YUV444PH,
              (vs.FLOAT, 32): vs.YUV444PS,
          }

          # Ziel-GRAY und YUV-Format bestimmen
          target_gray_format = gray_format_map.get((sample_type, bits))
          target_yuv444_format = yuv444_format_map.get((sample_type, bits))
          if target_gray_format is None or target_yuv444_format is None:
              raise ValueError(f"Unsupported format: sample_type={sample_type}, bits={bits}")

          # Matrix wählen für RGB → YUV
          matrix = 1 if clip.width < 1280 else 2

          if is_rgb:
              clip = core.resize.Bicubic(clip, format=target_yuv444_format, matrix_s=matrix)
          else:
              # Nur konvertieren, wenn Format nicht schon YUV444 mit korrekter Bitdepth
              if not (clip.format.color_family == vs.YUV and clip.format.subsampling_w == 0 and clip.format.subsampling_h == 0 and
                      clip.format.bits_per_sample == bits and clip.format.sample_type == sample_type):
                  clip = core.resize.Bicubic(clip, format=target_yuv444_format)

          # mClean anwenden
          thresh = 400
          strength = 20
          clip = denoise.mClean(clip=clip, thSAD=thresh, rn=0, strength=strength)

          if is_rgb:
              clip = core.resize.Bicubic(clip, format=original_format.id, matrix_in=matrix)

          return clip, target_gray_format

      smoothed, target_gray_format = processWithMClean(clip)
      smoothed = core.std.ShufflePlanes(smoothed, planes=0, colorfamily=vs.GRAY)
      smoothed = core.resize.Bicubic(smoothed, format=target_gray_format)

      luma = core.resize.Bicubic(luma, format=target_gray_format)
      return core.std.MakeDiff(luma, smoothed)

    else:
      raise ValueError(f"FinedetailMask: mode, unknown mode '{mode}'.")

    # Make difference
    diff = core.std.MakeDiff(luma, smoothed)

    # Use Expr to compute absolute diff from mid-gray
    expr = f"x {peak/2} - abs" if isinstance(peak, float) else f"x {int(peak)//2} - abs"
    EXPR = core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr
    mask = EXPR([diff], expr=expr)
    
    return mask
   
# experimental
def make_color_mask(clip: vs.VideoNode,
                    target_color: tuple,
                    tolerance: int = 30) -> vs.VideoNode:
    """
    Generate a binary mask where pixels close to a given RGB color are white (255), others black.
    Works consistently across float and integer RGB formats.
    """
    if clip.format.color_family != vs.RGB:
        clip = core.resize.Bicubic(clip, format=vs.RGBS)  # Always use float RGB internally

    # Extract float RGB channels (0–1.0)
    r, g, b = [core.std.ShufflePlanes(clip, i, vs.GRAY) for i in range(3)]

    # Convert 8-bit target color to float [0–1]
    target_f = [c / 255 for c in target_color]
    tol = tolerance

    # Compute squared color distance in float (but scaled as if in 8-bit space)
    EXPR = core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr
    dr = EXPR([r], f"x {target_f[0]} - 255 * dup *")  # scale diff to 8-bit range
    dg = EXPR([g], f"x {target_f[1]} - 255 * dup *")
    db = EXPR([b], f"x {target_f[2]} - 255 * dup *")

    dist_sq = EXPR([dr, dg, db], "x y + z +")

    # Threshold in 8-bit distance²
    thresh_sq = tol ** 2
    mask = EXPR([dist_sq], f"x {thresh_sq} < 255 0 ?")

    # Output GRAY8
    return core.resize.Bicubic(mask, format=vs.GRAY8)

def MotionMask(clip: vs.VideoNode, planes=None, th1=None, th2=None, tht=10, sc_value=0):
  
    if hasattr(core,'motionmask'):
      return core.core.motionmask.MotionMask(clip=clip, planes=planes, th1=th1, th2=th2, tht=tht, sc_value=sc_value)
  
    fmt = clip.format
    if fmt.color_family == vs.RGB:
        raise vs.Error("MotionMask: RGB not supported")
    if fmt.sample_type != vs.INTEGER or fmt.bits_per_sample not in range(8, 17):
        raise vs.Error("MotionMask: clip must be 8..16 bit integer")

    num_planes = fmt.num_planes
    max_val = (1 << fmt.bits_per_sample) - 1

    if planes is None:
        planes = list(range(num_planes))
    elif isinstance(planes, int):
        planes = [planes]
    else:
        planes = list(planes)

    def _expand(th, default):
        if th is None:
            th = [default] * num_planes
        elif isinstance(th, int):
            th = [th] * num_planes
        else:
            th = list(th)
        while len(th) < num_planes:
            th.append(th[-1])
        return th

    th1 = _expand(th1, 10)
    th2 = _expand(th2, 10)

    prev = clip[0] + clip[:-1]

    EXPR = core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr

    out_planes = []
    for p in range(num_planes):
        cur_p = core.std.ShufflePlanes(clip, p, vs.GRAY)
        if p not in planes:
            out_planes.append(cur_p)
            continue
        prev_p = core.std.ShufflePlanes(prev, p, vs.GRAY)
        # d <= th1 ? 0 : (d > th2 ? max_val : d)
        expr = (
            f"x y - abs {th1[p]} <= "
            f"0 "
            f"x y - abs {th2[p]} > {max_val} x y - abs ? "
            f"?"
        )
        out_planes.append(EXPR([cur_p, prev_p], expr))

    diff_clip = (
        out_planes[0] if num_planes == 1
        else core.std.ShufflePlanes(out_planes, [0] * num_planes, fmt.color_family)
    )
    diff_clip = core.std.CopyFrameProps(diff_clip, clip)

    sc_clip = misc.SCDetect(clip, threshold=tht / 255)
    sc_scaled = sc_value * max_val // 255
    blank = core.std.BlankClip(clip, color=[sc_scaled] * num_planes)

    def _select(n, f):
        return blank if f.props.get("_SceneChangePrev", 0) == 1 else diff_clip

    return core.std.FrameEval(diff_clip, _select, prop_src=sc_clip)
   
def bilinear_denoise(clip: vs.VideoNode, scale: float = 0.5, rg: bool=False) -> vs.VideoNode:
    """
    Perform simple bilinear denoising by downscaling and upscaling.
    
    Parameters:
        clip   (vs.VideoNode): Input clip.
        scale         (float): Downscale factor (0 < scale < 1).
        
    Returns:
        vs.VideoNode: Denoised clip.
    """
    if not (0 < scale < 1):
        raise ValueError("Scale must be between 0 and 1 (non-inclusive)")

    fmt = clip.format
    if fmt is None:
        raise ValueError("Clip must have a defined format")

    is_rgb = fmt.color_family == vs.RGB

    # Determine modulo requirements
    mod_w = 1 if is_rgb else 2 ** fmt.subsampling_w
    mod_h = 1 if is_rgb else 2 ** fmt.subsampling_h

    # Original resolution
    w, h = clip.width, clip.height

    def safe_mod(value, mod):
        return max(mod, (value // mod) * mod)

    # Downscaled resolution with safety
    low_w = safe_mod(int(w * scale), mod_w)
    low_h = safe_mod(int(h * scale), mod_h)
    low_w = max(mod_w, low_w)
    low_h = max(mod_h, low_h)

    # Downscale and upscale
    down = core.resize.Bilinear(clip, low_w, low_h)
    up = core.resize.Bilinear(down, w, h)
    
    if rg:
      rgMode = 17
      if hasattr(core, 'zsmooth'):
        smoothed = core.zsmooth.RemoveGrain(clip=up, mode=rgMode)
      elif hasattr(core, 'rgsf') and isFLOAT:  
        smoothed = core.rgsf.RemoveGrain(clip=up, mode=rgMode)
      else:
        smoothed = core.rgvs.RemoveGrain(clip=up, mode=rgMode)
      

    return up

