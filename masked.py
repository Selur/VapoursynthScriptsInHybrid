import vapoursynth as vs
from typing import Union, List, Optional, Tuple
core = vs.core

# collection of Mask filters.

# Use retinex to greatly improve the accuracy of the edge detection in dark scenes.
# draft=True is a lot faster, albeit less accurate
# from https://blog.kageru.moe/legacy/edgemasks.html
def retinex_edgemask(src: vs.VideoNode, sigma: int=1, draft: bool=False) -> vs.VideoNode:
    src = Depth(src, 16)
    luma = GetPlane(src, 0)
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    if draft:
        ret = EXPR(luma, 'x 65535 / sqrt 65535 *')
    else:
        ret = core.retinex.MSRCP(luma, sigma=[50, 200, 350], upper_thr=0.005)
    mask = EXPR([kirsch(luma), ret.tcanny.TCanny(mode=1, sigma=sigma).std.Minimum(coordinates=[1, 0, 1, 0, 0, 1, 0, 1])], 'x y +')
    return mask


# Kirsch edge detection. This uses 8 directions, so it's slower but better than Sobel (4 directions).
# more information: https://ddl.kageru.moe/konOJ.pdf
# from https://blog.kageru.moe/legacy/edgemasks.html
def kirsch(src: vs.VideoNode) -> vs.VideoNode:
    w = [5]*3 + [-3]*5
    weights = [w[-i:] + w[:-i] for i in range(4)]
    c = [src.std.Convolution((w[:4]+[0]+w[4:]), saturate=False) for w in weights]
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    return EXPR(c, 'x y max z max a max')


# should behave similar to std.Sobel() but faster since it has no additional high-/lowpass or gain.
# the internal filter is also a little brighter
# from https://blog.kageru.moe/legacy/edgemasks.html
def fast_sobel(src: vs.VideoNode) -> vs.VideoNode:
    sx = src.std.Convolution([-1, -2, -1, 0, 0, 0, 1, 2, 1], saturate=False)
    sy = src.std.Convolution([-1, 0, 1, -2, 0, 2, -1, 0, 1], saturate=False)
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
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
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
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
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    return EXPR(edges, ['x {high} >= {maxvalue} x {low} <= 0 x ? ?'
                                 .format(low=low, high=high, maxvalue=maxvalue), ''])

def RobertsEdges(clip, low=0, high=255):
    """Should behave like mt_edge(mode="roberts")"""
    valuerange = (1 << clip.format.bits_per_sample)
    maxvalue = valuerange - 1
    
    low = scale8(low, maxvalue)
    high = scale8(high, maxvalue)
    edges = core.std.Convolution(clip, matrix=[0,0,0,0,2,-1,0,-1,0], divisor=2, saturate=False)
    return EXPR(edges, ['x {high} >= {maxvalue} x {low} <= 0 x ? ?'
                                 .format(low=low, high=high, maxvalue=maxvalue), ''])

# from https://github.com/dnjulek/jvsfunc/blob/main/jvsfunc/mask.py -> Tcanny
def dehalo_mask(src: vs.VideoNode, expand: float = 0.5, iterations: int = 2, brz: int = 255, shift: int = 8) -> vs.VideoNode:
    from vsutil import depth, iterate, get_depth, get_y
    from math import sqrt
    """
    Based on muvsfunc.YAHRmask(), stand-alone version with some tweaks.

    :param src: Input clip. I suggest to descale (if possible) and nnedi3_rpow2 first, for a cleaner mask.
    :param expand: Expansion of edge mask.
    :param iterations: Protects parallel lines and corners that are usually damaged by YAHR.
    :param brz: Adjusts the internal line thickness.
    :param shift: Corrective shift for fine-tuning iterations
    """
    if brz > 255 or brz < 0:
        raise ValueError('dehalo_mask: brz must be between 0 and 255.')

    src_b = depth(src, 8)
    luma = get_y(src_b)
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    vEdge = EXPR([luma, luma.std.Maximum().std.Maximum()], [f'y x - {shift} - 128 *'])
    mask1 = EXPR(vEdge.tcanny.TCanny(sigma=sqrt(expand*2), mode=-1), ['x 16 *'])
    mask2 = iterate(vEdge, core.std.Maximum, iterations)
    mask2 = iterate(mask2, core.std.Minimum, iterations)
    mask2 = mask2.std.Invert().std.Binarize(80)
    mask3 = mask2.std.Inflate().std.Inflate().std.Binarize(brz)
    mask4 = mask3 if brz < 255 else mask2
    mask4 = mask4.std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    mask = EXPR([mask1, mask4], ['x y min'])
    return depth(mask, get_depth(src), range=1)


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
    
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
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
      def processWithMClean(clip: vs.VideoNode) -> vs.VideoNode:
        original_format = clip.format
        is_rgb = original_format.color_family == vs.RGB

        # For YUV clips, decide color matrix based on resolution
        matrix = 1 if clip.width < 1280 else 2  # 1=BT.601, 2=BT.709

        if is_rgb:
            # RGB to YUV: use matrix_s
            clip = core.resize.Bicubic(clip, format=vs.YUV444P16, matrix_s=matrix)
        else:
            # Already YUV — convert to 444P16 only
            if clip.format.id != vs.YUV444P16:
                clip = core.resize.Bicubic(clip, format=vs.YUV444P16)

        # Apply mClean
        thresh = 400
        strength = 20
        clip = denoise.mClean(clip=clip, thSAD=thresh, rn=0, strength=strength)

        if is_rgb:
            # YUV back to RGB: use matrix_in
            clip = core.resize.Bicubic(clip, format=original_format.id, matrix_in=matrix)

        return clip
      smoothed = processWithMClean(clip)
      smoothed = core.std.ShufflePlanes(smoothed, planes=0, colorfamily=vs.GRAY)
      return core.std.MakeDiff(luma, smoothed)
    else:
      raise ValueError(f"FinedetailMask: mode, unknown mode '{mode}'.")

    # Make difference
    diff = core.std.MakeDiff(luma, smoothed)

    # Use Expr to compute absolute diff from mid-gray
    expr = f"x {peak/2} - abs" if isinstance(peak, float) else f"x {int(peak)//2} - abs"
    if hasattr(core, 'akarin'):
      mask = core.akarin.Expr([diff], expr=expr)
    else:
      mask = core.std.Expr([diff], expr=expr)
    
    return mask
    
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
    
    
def Depth(src, bits, dither_type='error_diffusion', range=None, range_in=None):
    src_f = src.format
    src_cf = src_f.color_family
    src_st = src_f.sample_type
    src_bits = src_f.bits_per_sample
    src_sw = src_f.subsampling_w
    src_sh = src_f.subsampling_h
    dst_st = vs.INTEGER if bits < 32 else vs.FLOAT

    if isinstance(range, str):
        range = RANGEDICT[range]

    if isinstance(range_in, str):
        range_in = RANGEDICT[range_in]

    if (src_bits, range_in) == (bits, range):
        return src
    out_f = core.register_format(src_cf, dst_st, bits, src_sw, src_sh)
    return core.resize.Point(src, format=out_f.id, dither_type=dither_type, range=range, range_in=range_in)


RANGEDICT = {'limited': 0, 'full': 1}
 