"""
msmooth.py
Drop-in compatible replacements for MSmooth and MSharpen
from https://github.com/dubhatervapoursynth/vapoursynth-msmoosh

Identical call signatures to the original plugin:

    MSmooth(clip, threshold=6.0, strength=3,    mask=False, planes=None)
    MSharpen(clip, threshold=6.0, strength=39.0, mask=False, planes=None)
results will slightly differ, but should be fine.

Plugin dependencies
───────────────────
Required:
    core.edgemasks   – edge detection (Sobel operator)
                       https://github.com/HolyWu/VapourSynth-EdgeMasks
    core.vszip       – vszip.Bilateral (edge-preserving blur) for MSmooth
                       https://github.com/dnjulek/vapoursynth-zip
    core.zsmooth     – zsmooth.RemoveGrain (3×3 weighted blur) for MSharpen
                       https://github.com/adworacz/zsmooth
    core.std         – MaskedMerge, ShufflePlanes (always present)

Expr backend (first available wins):
    core.llvmexpr  → core.akarin  → core.cranexpr  → core.std  (fallback)

Algorithm overview
──────────────────
MSmooth
  1. Build Sobel edge mask via edgemasks.Sobel, binarised at threshold.
  2. Smooth with vszip.Bilateral – edge-preserving, so it naturally avoids
     blurring across detected edges; strength maps to bilateral sigmaS.
  3. std.MaskedMerge(smoothed, original, edge_mask) – hard-restores edge
     pixels that bilateral may still have softened.

MSharpen
  1. Build the same Sobel edge mask.
  2. Blur with zsmooth.RemoveGrain(mode=11) – weighted 3×3 (centre ×4, cross ×2, diagonal ×1).
  3. Apply the sharpening formula via Expr:
         tmp       = clamp(4·src − 3·blur, 0, peak)
         sharpened = (strength_i·tmp + invstrength_i·src) >> bits
  4. std.MaskedMerge(original, sharpened, edge_mask) – apply sharpening
     only at detected edges.
"""

import vapoursynth as vs

core = vs.core

# ── Expr backend selection ───────────────────────────────────────────────────
EXPR = (
    core.llvmexpr.Expr  if hasattr(core, 'llvmexpr')  else
    core.akarin.Expr    if hasattr(core, 'akarin')     else
    core.cranexpr.Expr  if hasattr(core, 'cranexpr')   else
    core.std.Expr
)

# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _fmt(clip: vs.VideoNode) -> vs.VideoFormat:
    if clip.format is None:
        raise vs.Error("msmoosh_plugins: clip must have a constant format")
    return clip.format


def _peak(bits: int) -> int:
    return (1 << bits) - 1


def _plane_list(num_planes: int, planes) -> list:
    if planes is None:
        return list(range(num_planes))
    return sorted(set(planes))


def _is_rgb(clip: vs.VideoNode) -> bool:
    return _fmt(clip).color_family == vs.RGB


def _get_format_id(color_family, sample_type, bits, subsampling_w, subsampling_h) -> int:
    """Return a format ID compatible with both R55+ (query_video_format) and older VS."""
    try:
        fmt = core.query_video_format(color_family, sample_type, bits, subsampling_w, subsampling_h)
    except AttributeError:
        fmt = core.register_format(color_family, sample_type, bits, subsampling_w, subsampling_h)
    return fmt.id


def _depth(clip: vs.VideoNode, bits: int) -> vs.VideoNode:
    """
    Linearly scale an integer clip to a different bit depth.

    Downscale includes a +0.5 rounding term before truncation so that the
    result is rounded rather than floored (e.g. 32-bit int -> 16-bit int).
    Uses the module-level EXPR backend for consistency.
    """
    fmt = _fmt(clip)
    if fmt.bits_per_sample == bits:
        return clip
    if fmt.sample_type != vs.INTEGER:
        raise vs.Error("_depth: clip must be integer format")
    peak_src = _peak(fmt.bits_per_sample)
    peak_dst = _peak(bits)
    fmt_id = _get_format_id(fmt.color_family, fmt.sample_type, bits,
                            fmt.subsampling_w, fmt.subsampling_h)
    if bits < fmt.bits_per_sample:
        # Downscale: multiply first to preserve precision, then round
        expr = f"x {peak_dst} * {peak_src} / 0.5 + trunc"  
    else:
        # Upscale: simple multiply
        expr = f"x {peak_dst} * {peak_src} /"
    return EXPR([clip], expr, format=fmt_id)


# ─────────────────────────────────────────────────────────────────────────────
# Edge mask  (shared by both MSmooth and MSharpen)
# ─────────────────────────────────────────────────────────────────────────────

def _make_edge_mask(clip: vs.VideoNode,
                    threshold_pct: float,
                    proc: list) -> vs.VideoNode:
    """
    Returns a hard binary edge mask in the same format as `clip`.
    Non-zero pixels = edge detected.

    Uses edgemasks.Sobel (HolyWu/VapourSynth-EdgeMasks) for gradient
    detection.

    The raw Sobel magnitude is thresholded at `threshold_pct` % of the
    bit-depth peak to produce a hard 0/peak binary mask.

    For RGB clips all three plane masks are OR-combined so that an edge on
    any channel triggers the mask.
    For YUV/GRAY the luma mask is downsampled to chroma resolution.
    """
    fmt    = _fmt(clip)
    bits   = fmt.bits_per_sample
    peak   = _peak(bits)
    th     = max(0, min(peak, int(threshold_pct * peak / 100)))
    is_rgb = fmt.color_family == vs.RGB

    def _plane_edge(pl: int) -> vs.VideoNode:
        """
        Run edgemasks.Sobel on a single plane and binarise the result.

        edgemasks.Sobel outputs the gradient magnitude scaled to [0, peak]
        (full range, tagged with _Range=1).  We threshold it directly:
        pixels at or above `th` become `peak`, others become 0.
        """
        gray  = clip.std.ShufflePlanes(planes=pl, colorfamily=vs.GRAY)
        # edgemasks.Sobel: planes=[0] processes the single GRAY plane;
        # scale=1.0 keeps the native magnitude range (no amplification).
        sobel = core.edgemasks.Sobel(gray, planes=[0], scale=1.0)
        # Binarise: 0 or peak
        return EXPR([sobel], f"x {th} >= {peak} 0 ?")

    if is_rgb and len(proc) == 3:
        # OR across all three planes so any channel edge triggers the mask
        m0 = _plane_edge(0)
        m1 = _plane_edge(1)
        m2 = _plane_edge(2)
        combined = EXPR([m0, m1, m2], "x y OR z OR")
        return core.std.ShufflePlanes(
            [combined, combined, combined],
            planes=[0, 0, 0],
            colorfamily=vs.RGB
        )
    else:
        # YUV / GRAY: build mask on luma, downsample to chroma resolution
        luma_mask = _plane_edge(0)
        if fmt.num_planes == 1:
            return luma_mask
        chroma_mask = luma_mask.resize.Point(
            width  = clip.width  >> fmt.subsampling_w,
            height = clip.height >> fmt.subsampling_h
        )
        return core.std.ShufflePlanes(
            [luma_mask, chroma_mask, chroma_mask],
            planes=[0, 0, 0],
            colorfamily=fmt.color_family
        )


def _mask_for_merge(clip: vs.VideoNode,
                    edge_mask: vs.VideoNode,
                    proc: list) -> vs.VideoNode:
    """
    Return the edge mask formatted for std.MaskedMerge on `clip`.
    For GRAY clips, strips to a single plane.
    """
    fmt = _fmt(clip)
    if fmt.num_planes == 1:
        return edge_mask.std.ShufflePlanes(0, vs.GRAY)
    return edge_mask


# ─────────────────────────────────────────────────────────────────────────────
# MSmooth
# ─────────────────────────────────────────────────────────────────────────────

def MSmooth(clip: vs.VideoNode,
            threshold: float = 6.0,
            strength: int = 3,
            mask: bool = False,
            planes=None) -> vs.VideoNode:
    """
    Drop-in replacement for msmoosh.MSmooth.

    Uses vszip.Bilateral as the edge-preserving smoother and std.MaskedMerge
    to hard-restore edge pixels.

    Parameters
    ----------
    clip      : input clip (integer 8–32-bit; float clips are passed through)
    threshold : 0–100 %   edge sensitivity          (default 6)
    strength  : 1–25      smoothing strength         (default 3)
    mask      : bool      return edge mask instead   (default False)
    planes    : list[int] planes to process; None = all
    """
    fmt = _fmt(clip)
    if fmt.sample_type == vs.FLOAT:
        return clip   # pass-through: float clips

    # >16-bit integer: process at 16-bit and scale back
    if fmt.bytes_per_sample > 2:
        clip_16 = _depth(clip, 16)
        result_16 = MSmooth(clip_16, threshold=threshold, strength=strength,
                            mask=mask, planes=planes)
        return _depth(result_16, fmt.bits_per_sample)

    if not (0.0 <= threshold <= 100.0):
        raise vs.Error("MSmooth: threshold must be between 0 and 100 %")
    strength = int(strength + 0.5)
    if not (1 <= strength <= 25):
        raise vs.Error("MSmooth: strength must be between 1 and 25")

    bits       = fmt.bits_per_sample
    num_planes = fmt.num_planes
    proc       = _plane_list(num_planes, planes)

    # ── 1. Edge mask ─────────────────────────────────────────────────────────
    edge_mask = _make_edge_mask(clip, threshold, proc)

    if mask:
        return edge_mask

    # ── 2. Bilateral smooth (vszip.Bilateral) ────────────────────────────────
    # Map strength (1–25) → sigmaS (2.2 … 18.3).
    # sigmaR tracks threshold so the filter respects the same edge sensitivity.
    # vszip.Bilateral is a faster drop-in for VapourSynth-Bilateral, with an
    # identical parameter interface.
    sigmaS = 1.5 + (strength - 1) * 0.7          # ~2.2 at s=1, ~18.3 at s=25
    sigmaR = max(0.02, threshold / 100.0 * 0.4)  # colour-range sigma

    smoothed = core.vszip.Bilateral(
        clip,
        sigmaS=sigmaS,
        sigmaR=sigmaR,
        planes=proc,
        algorithm=0   # 0 = auto-select O(1) vs O(sigmaS²) based on params
    )

    # ── 3. Masked merge: keep original at hard edges ─────────────────────────
    # MaskedMerge(base, overlay, mask):
    #   result = base*(1 − mask/peak) + overlay*(mask/peak)
    # mask=peak (edge) → take `overlay` = original clip
    # mask=0    (flat) → take `base`    = smoothed clip
    return core.std.MaskedMerge(
        smoothed, clip,
        _mask_for_merge(clip, edge_mask, proc),
        planes=proc
    )


# ─────────────────────────────────────────────────────────────────────────────
# MSharpen
# ─────────────────────────────────────────────────────────────────────────────

def MSharpen(clip: vs.VideoNode,
             threshold: float = 6.0,
             strength: float = 39.0,
             mask: bool = False,
             planes=None) -> vs.VideoNode:
    """
    Drop-in replacement for msmoosh.MSharpen.

    Uses zsmooth.RemoveGrain(mode=11) as the blur kernel and Expr for the
    sharpening formula, with std.MaskedMerge to apply the effect only at edges.

    Parameters
    ----------
    clip      : input clip (integer 8–32-bit; float clips are passed through)
    threshold : 0–100 %   edge sensitivity          (default 6)
    strength  : 0–100 %   sharpening amount          (default 39)
    mask      : bool      return edge mask instead   (default False)
    planes    : list[int] planes to process; None = all
    """
    fmt = _fmt(clip)
    if fmt.sample_type == vs.FLOAT:
        return clip   # pass-through: float clips

    # >16-bit integer: process at 16-bit and scale back
    if fmt.bytes_per_sample > 2:
        clip_16 = _depth(clip, 16)
        result_16 = MSharpen(clip_16, threshold=threshold, strength=strength,
                             mask=mask, planes=planes)
        return _depth(result_16, fmt.bits_per_sample)

    if not (0.0 <= threshold <= 100.0):
        raise vs.Error("MSharpen: threshold must be between 0 and 100 %")
    if not (0.0 <= strength <= 100.0):
        raise vs.Error("MSharpen: strength must be between 0 and 100 %")

    bits       = fmt.bits_per_sample
    peak       = _peak(bits)
    num_planes = fmt.num_planes
    proc       = _plane_list(num_planes, planes)

    # ── 1. Edge mask ─────────────────────────────────────────────────────────
    edge_mask = _make_edge_mask(clip, threshold, proc)

    if mask:
        return edge_mask

    # ── 2. Blur kernel: zsmooth.RemoveGrain mode 11 ──────────────────────────
    # Mode 11: weighted 3×3 average (centre ×4, cross ×2, diagonal ×1).
    # This is the closest native equivalent to the C++ blur3x3 used in the
    # original before edge detection.  zsmooth also properly handles edge
    # pixels via mirror padding, unlike the original rgvs which copies them.
    # Planes not in proc use mode 0 (copy-through).
    rg_modes = [11 if p in proc else 0 for p in range(num_planes)]
    blurred  = core.zsmooth.RemoveGrain(clip, mode=rg_modes)

    # ── 3. Sharpening formula (exact C++ port) via EXPR ──────────────────────
    # x = src sample,  y = blurred sample
    #   tmp        = clamp(4·x − 3·y, 0, peak)
    #   strength_i = int(strength_pct * peak / 100)
    #   inv_str    = peak − strength_i
    #   sharpened  = (strength_i·tmp + inv_str·x) >> bits
    str_i   = max(0, min(peak, int(strength * peak / 100)))
    inv_str = peak - str_i

    sharpen_expr = (
        f"x 4 * y 3 * - 0 {peak} clamp "   # tmp = clamp(4·src − 3·blur, 0, peak)
        f"{str_i} * "                       # strength_i · tmp
        f"x {inv_str} * + "                 # + inv_str · src
        f"{1 << bits} / trunc "             # ÷ 2^bits  (was: >> bits)
        f"0 {peak} clamp"                    # safety clamp
    )
    # Empty string on unprocessed planes = copy-through in all Expr backends
    plane_exprs = [sharpen_expr if p in proc else "" for p in range(num_planes)]
    sharpened   = EXPR([clip, blurred], plane_exprs)

    # ── 4. Masked merge: apply sharpening only at edges ──────────────────────
    # mask=peak (edge) → take `overlay` = sharpened
    # mask=0    (flat) → take `base`    = original clip
    return core.std.MaskedMerge(
        clip, sharpened,
        _mask_for_merge(clip, edge_mask, proc),
        planes=proc
    )
