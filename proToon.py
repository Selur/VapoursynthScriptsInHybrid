# Port of proToon 0.7.5
# Author: TheProfileth
# see: http://avisynth.nl/index.php/ProToon
# ported by: Selur

##
# The proToon function
# Variables:
# clip   =
#   Input clip. 
# int  strength = 48
#   Line darkening amount, 0-255. 
# int  luma_cap = 191
#   Bright limit for line detection, 0-255 (255 = no limit). 
# int  threshold = 4
#   Threshold to disable slight darkening (of noise) 0-255. 
# int  thinning = 24
#   Line thinning amount, 0-255. 
# bool  sharpen = true
#   Sharpening on/off. 
# bool  mask = true
#   Masking on/off. 
# int  ssw = 4
#   Supersample factor horizontally, 0-inf. 
# int  ssh = 4
#   Supersample factor vertically, 0-inf. 
# int  xstren = 255
#   xsharpening strength, 0-255. 
# int  xstresh = 255
#   xsharpening threshold, 0-255. 
# Dependencies:
#  RemoveGrain: https://github.com/vapoursynth/vs-removegrain
#  or zsmooth (https://github.com/adworacz/zsmooth)
#
##

import vapoursynth as vs
from typing import Optional

core = vs.core

# Define necessary helper functions
def mf_min(a: str, b: str) -> str:
    return f"{a} {b} min"

def mf_max(a: str, b: str) -> str:
    return f"{a} {b} max"

def mf_str_level(x: str, in_low: int, in_high: int, out_low: int, out_high: int, bits: int) -> str:
    scale = (1 << bits) - 1
    in_low = in_low * scale // 255
    in_high = in_high * scale // 255
    out_low = out_low * scale // 255
    out_high = out_high * scale // 255
    return mf_max(mf_min(f"{x} {scale} * {in_low} - {in_high - in_low} / {out_high - out_low} * {out_low} +", f"{235 * scale // 255}"), f"{16 * scale // 255}")

# Xsharpen function based on WarpSharpSupport
def Xsharpen(clip: vs.VideoNode, strength: int = 128, threshold: int = 8) -> vs.VideoNode:
    bits = clip.format.bits_per_sample
    expr = f"y x - x z - min {threshold} < x z - y x - < z y ? {strength / 256} * x {(256 - strength) / 256} * + x ?"
    return core.std.Expr([clip, clip.std.Maximum(planes=0), clip.std.Minimum(planes=0)], [expr, ""])

def merge_chroma(luma: vs.VideoNode, chroma: vs.VideoNode) -> vs.VideoNode:
    return core.std.ShufflePlanes([luma, chroma], planes=[0, 1, 2], colorfamily=vs.YUV)

def proToon(input: vs.VideoNode, 
            strength: int = 48, luma_cap: int = 191, threshold: int = 4, thinning: int = 0, 
            sharpen: bool = True, mask: bool = True, 
            ssw: int = 4, ssh: int = 4, 
            xstren: int = 255, xthresh: int = 255, pcScale: bool = False) -> vs.VideoNode:

    bits = input.format.bits_per_sample
    scale = (1 << bits) - 1

    str_value = strength / 128.0  # line darkening amount, 0-255
    lum = luma_cap * scale // 255  # bright limit for line detection, 0-255 (255 = no limit)
    thr = threshold * scale // 255  # threshold to disable slight darkening (of noise) 0-255
    thn = thinning / 16.0  # line thinning amount, 0-255

    # Create the edgemask
    if hasattr(core, 'zsmooth'):
      edgemask = core.std.Expr(
          [input, core.zsmooth.RemoveGrain(input, 12)],
          expr=[mf_str_level("x y - abs 128 +", 132, 145, 0, 255, bits)]
      ).zsmooth.RemoveGrain(12).std.Expr(expr=[mf_str_level("x", 0, 64, 0, 255, bits)])
    else:
      edgemask = core.std.Expr(
          [input, core.rgvs.RemoveGrain(input, 12)],
          expr=[mf_str_level("x y - abs 128 +", 132, 145, 0, 255, bits)]
      ).rgvs.RemoveGrain(12).std.Expr(expr=[mf_str_level("x", 0, 64, 0, 255, bits)])

    exin = core.std.Maximum(input).std.Minimum()
    diff = core.std.Expr(
        [input, exin], 
        expr=[f"y {lum} < y {lum} ? x {thr} + > x y {lum} < y {lum} ? - 0 ? {127 * scale // 255} +"]
    )
    thick = core.std.Expr(
        [input, exin], 
        expr=[f"y {lum} < y {lum} ? x {thr} + > x y {lum} < y {lum} ? - 0 ? {str_value} * x +"]
    )

    darkened = (thinning == 0) and thick or core.std.MaskedMerge(
        core.std.Expr(
            [core.std.Maximum(input), diff], 
            expr=[f"x y {127 * scale // 255} - {str_value} 1 + * +"]
        ), 
        thick, 
        core.std.Expr(
            [core.std.Minimum(diff)], 
            expr=[f"x {127 * scale // 255} - {thn} * {scale} +"]
        ).std.Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])
    )

    masked = mask and core.std.MaskedMerge(input, darkened, edgemask) or merge_chroma(darkened, input)

    if sharpen:
        upscaled = core.resize.Lanczos(masked, width=input.width * ssw, height=input.height * ssh)
        sharpened = Xsharpen(upscaled, xstren, xthresh)
        sharpened = core.resize.Lanczos(sharpened, width=input.width, height=input.height)
    else:
        sharpened = masked

    output = sharpened

    # Apply scaling limit based on pcScale
    if not pcScale:
        minV = 16 * scale // 255
        maxV = max=235 * scale // 255 
        if (hasattr(core,'vszip')):
          output = core.vszip.Limiter(output, min=[minV,minV,minV], max=[maxV,maxV,maxV])
        else:
          output = core.std.Limiter(output, min=minV, max=maxV)
    return output


