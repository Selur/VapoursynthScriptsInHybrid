import vapoursynth as vs
core = vs.core

"""
SpotDelta (Spotless + Delta Restore) – VapourSynth Port
======================================================
Original AviSynth script by chmars (17/06/2021)
VapourSynth port based on the AviSynth source and the VS SpotLess implementation.

Avisynth SpotLessDelta: https://forum.doom9.org/showthread.php?p=1946031
Avisynth SpotLess: https://forum.doom9.org/showthread.php?t=181777

Requirements
------------
  mvtools      (core.mv)
  tmedian      (core.tmedian)  – or zsmooth / ttmpsm
  misc         (core.misc)     – or hysteresis
  akarin / llvmexpr / cranexpr / std.Expr – for mask Expr operations

Usage
-----
  import vapoursynth as vs
  from SpotLess import SpotDelta

  core = vs.core
  src  = core.lsmas.LWLibavSource("myclip.mkv")

  # Option A – cleaned clip (default):
  SpotDelta(src).set_output()

  # Option B – with grain restoration:
  SpotDelta(src, thsad=5000, rgr=True, rgr1=5, rgr2=10).set_output()

  # Option C – side-by-side comparison (source | SpotLess | SpotDelta):
  SpotDelta(src, output='stacked').set_output()

  # Option D – interleaved comparison:
  SpotDelta(src, output='interleaved').set_output()

  # Option E – SpotLess vs SpotDelta + exaggerated-chroma diff:
  SpotDelta(src, output='versus_stacked').set_output()

  # Option F – all four panels including exaggerated-chroma diff:
  SpotDelta(src, output='full_stacked').set_output()
"""

from typing import Optional


# ---------------------------------------------------------------------------
# Plugin selectors
# ---------------------------------------------------------------------------

def _expr_fn():
    """Pick the best available Expr plugin."""
    if hasattr(core, 'akarin'):    return core.akarin.Expr
    if hasattr(core, 'llvmexpr'): return core.llvmexpr.Expr
    if hasattr(core, 'cranexpr'): return core.cranexpr.Expr
    return core.std.Expr

def _boxblur_fn():
    """Pick the best available BoxBlur."""
    if hasattr(core, 'vszip'): return core.vszip.BoxBlur
    return core.std.BoxBlur

def _hysteresis_fn():
    """Pick the best available Hysteresis."""
    if hasattr(core, 'hysteresis'): return core.hysteresis.Hysteresis
    return core.misc.Hysteresis


# ---------------------------------------------------------------------------
# Thin Expr wrappers  (single-plane / GRAY only)
# ---------------------------------------------------------------------------

def _expr1(clip: vs.VideoNode, expr: str) -> vs.VideoNode:
    """Single-input Expr on a GRAY clip."""
    assert clip.format.num_planes == 1, "_expr1 requires a GRAY clip"
    return _expr_fn()([clip], [expr])

def _expr2(a: vs.VideoNode, b: vs.VideoNode, expr: str) -> vs.VideoNode:
    """Two-input Expr on two GRAY clips with identical format."""
    assert a.format.num_planes == 1 and b.format.num_planes == 1, \
        "_expr2 requires GRAY inputs"
    return _expr_fn()([a, b], [expr])


# ---------------------------------------------------------------------------
# Plane helpers
# ---------------------------------------------------------------------------

def _gray(clip: vs.VideoNode) -> vs.VideoNode:
    """Extract luma plane as a GRAY clip."""
    return core.std.ShufflePlanes(clip, 0, vs.GRAY)

def _plane(clip: vs.VideoNode, p: int) -> vs.VideoNode:
    """Extract any single plane as a GRAY clip."""
    return core.std.ShufflePlanes(clip, p, vs.GRAY)

def _blank_gray(ref: vs.VideoNode) -> vs.VideoNode:
    """Black GRAY clip matching the format/size of ref (which may be GRAY or YUV)."""
    gray_fmt = core.query_video_format(
        vs.GRAY,
        ref.format.sample_type,
        ref.format.bits_per_sample
    )
    return core.std.BlankClip(ref, format=gray_fmt.id)


# ---------------------------------------------------------------------------
# Diff helpers
# ---------------------------------------------------------------------------

def _makediff(a: vs.VideoNode, b: vs.VideoNode) -> vs.VideoNode:
    """
    core.std.MakeDiff on two GRAY clips.
    Stores (a - b + mid) with proper clamping for any bit depth.
    """
    assert a.format.num_planes == 1 and b.format.num_planes == 1, \
        "_makediff requires GRAY inputs"
    return core.std.MakeDiff(a, b)

def _adddiff(base: vs.VideoNode, diff: vs.VideoNode) -> vs.VideoNode:
    """
    Inverse of MakeDiff: computes (base + diff - mid) with proper clamping.
    core.std.AddDiff does not exist in VapourSynth, so we use Expr.
    """
    assert base.format.num_planes == 1 and diff.format.num_planes == 1, \
        "_adddiff requires GRAY inputs"
    bits = base.format.bits_per_sample
    mid  = 1 << (bits - 1)
    peak = (1 << bits) - 1
    return _expr2(base, diff, f'x y + {mid} - 0 {peak} clamp')


# ---------------------------------------------------------------------------
# Exaggerated-chroma diff
# ---------------------------------------------------------------------------

def _exaggerated_chroma_diff(source: vs.VideoNode,
                              filtered: vs.VideoNode) -> vs.VideoNode:
    """
    Diff clip where luma is a plain MakeDiff and chroma differences are
    amplified 3x, making colour artefacts clearly visible.
    """
    bits = source.format.bits_per_sample
    mid  = 1 << (bits - 1)
    peak = (1 << bits) - 1

    diff1 = core.std.MakeDiff(source, filtered)

    def _boost_chroma(base: vs.VideoNode, addend: vs.VideoNode) -> vs.VideoNode:
        """Add addend chroma onto base chroma (+mid offset); luma untouched."""
        exprs = [
            '',                                        # luma: unchanged
            f'x y + {mid} - 0 {peak} clamp',          # U: accumulate
            f'x y + {mid} - 0 {peak} clamp',          # V: accumulate
        ]
        return _expr_fn()([base, addend], exprs)

    diff2 = _boost_chroma(diff1, diff1)   # chroma x2
    diff3 = _boost_chroma(diff2, diff1)   # chroma x3

    # Recombine: luma from diff1 (plain), chroma from diff3 (3x amplified)
    return core.std.ShufflePlanes(
        [diff1, diff3, diff3], [0, 1, 2], source.format.color_family
    )


# ---------------------------------------------------------------------------
# Basic mask operations  (all GRAY in, GRAY out)
# ---------------------------------------------------------------------------

def _binarize(clip: vs.VideoNode, threshold: int) -> vs.VideoNode:
    bits = clip.format.bits_per_sample
    peak = (1 << bits) - 1
    return _expr1(clip, f'x {threshold} > {peak} 0 ?')

def _logic_or(a: vs.VideoNode, b: vs.VideoNode) -> vs.VideoNode:
    return _expr2(a, b, 'x y max')

def _logic_and(a: vs.VideoNode, b: vs.VideoNode) -> vs.VideoNode:
    return _expr2(a, b, 'x y min')

def _suppress_with(mask: vs.VideoNode, suppression: vs.VideoNode) -> vs.VideoNode:
    """mask = max(mask − suppression, 0)"""
    return _expr2(mask, suppression, 'x y - 0 max')

def _add_to(mask: vs.VideoNode, other: vs.VideoNode) -> vs.VideoNode:
    return _logic_or(mask, other)

def _must_not_overlap(mask: vs.VideoNode, other: vs.VideoNode) -> vs.VideoNode:
    expanded = _hysteresis_fn()(other, mask)
    return _expr2(mask, expanded, 'x y - 0 max')


# ---------------------------------------------------------------------------
# Morphological helpers  (use native Maximum / Minimum)
# ---------------------------------------------------------------------------

def _remove_small_spots(mask: vs.VideoNode, spot_size: int) -> vs.VideoNode:
    """Erode spot_size times, then hysteresis-restore to original extent."""
    eroded = mask
    for _ in range(spot_size):
        eroded = core.std.Minimum(eroded)
    return _hysteresis_fn()(eroded, mask)

def _expand_mask(mask: vs.VideoNode, amount: int, blur: float = 0.0) -> vs.VideoNode:
    """Dilate amount times, then optionally blur."""
    for _ in range(amount):
        mask = core.std.Maximum(mask)
    if blur > 0:
        r = max(1, int(blur * 2))
        mask = _boxblur_fn()(mask, hradius=r, vradius=r)
    return mask

def _must_be_near(mask: vs.VideoNode, other: vs.VideoNode,
                  max_distance: int, expanded: bool = False) -> vs.VideoNode:
    ea         = max_distance // 2
    eb         = max_distance - ea
    mask_exp   = _expand_mask(mask,  ea)
    other_exp  = _expand_mask(other, eb)
    common     = _logic_or(mask_exp, other_exp)
    if expanded:
        return common
    other_near = _hysteresis_fn()(other, common)
    return _logic_and(mask, other_near)

def _z_padding(clip: vs.VideoNode,
               left: int, top: int, right: int, bottom: int) -> vs.VideoNode:
    """Crop edges then restore with neutral (black) border."""
    cropped = core.std.Crop(clip, left, right, top, bottom)
    return core.std.AddBorders(cropped, left, right, top, bottom)


# ---------------------------------------------------------------------------
# Edge detection  (Scharr, GRAY output)
# ---------------------------------------------------------------------------

def _scharr(clip: vs.VideoNode) -> vs.VideoNode:
    """Scharr edge magnitude – operates on luma, returns GRAY."""
    g  = _gray(clip)
    # Convolution for the horizontal and vertical passes
    sx = core.std.Convolution(g, matrix=[3, 0, -3, 10, 0, -10, 3, 0, -3])
    sy = core.std.Convolution(g, matrix=[3, 10, 3, 0, 0, 0, -3, -10, -3])
    return _expr2(sx, sy, 'x x * y y * + sqrt')


# ---------------------------------------------------------------------------
# Luma delta mask  →  GRAY
# ---------------------------------------------------------------------------

def _luma_delta_mask(source: vs.VideoNode,
                     filtered: vs.VideoNode,
                     direction: str,
                     brightness: float = 0.9,
                     limit_brightness: Optional[float] = None,
                     limit_abs_brightness: float = 0.0,
                     spot_size: int = 2) -> vs.VideoNode:
    """
    Returns a GRAY mask.
      direction '>' : pixels where source luma > filtered luma (by brightness ratio)
      direction '<' : pixels where source luma < filtered luma
    brightness / limit_brightness are ratios relative to the mid-grey value.
    """
    if limit_brightness is None:
        limit_brightness = brightness

    bits = source.format.bits_per_sample
    mid  = 1 << (bits - 1)
    peak = (1 << bits) - 1
    thr  = int(mid * brightness)
    op   = '>' if direction == '>' else '<'

    diff      = _makediff(_gray(source), _gray(filtered))
    mask      = _expr1(diff, f'x {thr} {op} {peak} 0 ?')
    mask_orig = mask

    if limit_brightness != brightness:
        lim  = int(mid * limit_brightness)
        sup  = _expr1(diff, f'x {lim} {op} {peak} 0 ?')
        mask = _suppress_with(mask, sup)

    if limit_abs_brightness > 0.0:
        abs_thr = int(peak * limit_abs_brightness)
        luma    = _gray(source)
        mask    = _expr2(mask, luma, f'y {abs_thr} < 0 x ?')

    mask = _remove_small_spots(mask, spot_size)

    if limit_brightness != brightness or limit_abs_brightness > 0.0:
        mask = _hysteresis_fn()(mask, mask_orig)

    return mask


# ---------------------------------------------------------------------------
# Chroma delta mask  →  GRAY
# ---------------------------------------------------------------------------

def _chroma_delta_mask(source: vs.VideoNode,
                       filtered: vs.VideoNode,
                       delta: int = 3,
                       spot_size: int = 2) -> vs.VideoNode:
    """
    Returns a GRAY mask selecting pixels with large chroma difference
    (Euclidean distance sqrt(Δu² + Δv²)).
    """
    u_diff = _makediff(_plane(source, 1), _plane(filtered, 1))
    v_diff = _makediff(_plane(source, 2), _plane(filtered, 2))

    bits    = source.format.bits_per_sample
    mid     = 1 << (bits - 1)

    u_delta = _expr1(u_diff, f'x {mid} - abs')
    v_delta = _expr1(v_diff, f'x {mid} - abs')

    W, H = source.width, source.height
    if u_delta.width != W or u_delta.height != H:
        u_delta = core.resize.Bilinear(u_delta, W, H)
        v_delta = core.resize.Bilinear(v_delta, W, H)

    uv_delta = _expr2(u_delta, v_delta, 'x x * y y * + sqrt')
    mask     = _binarize(uv_delta, delta)
    mask     = _remove_small_spots(mask, spot_size)

    return mask   # GRAY


# ---------------------------------------------------------------------------
# DeltaRestore  –  applies GRAY masks via MaskedMerge(first_plane=True)
# ---------------------------------------------------------------------------

def _delta_restore(filtered: vs.VideoNode,
                   source: vs.VideoNode,
                   luma:             Optional[vs.VideoNode] = None,
                   chroma:           Optional[vs.VideoNode] = None,
                   chroma_override:  Optional[vs.VideoNode] = None,
                   edges:            Optional[vs.VideoNode] = None,
                   dark:             Optional[vs.VideoNode] = None) -> vs.VideoNode:
    assert any(m is not None for m in [luma, chroma, chroma_override, dark]), \
        "DeltaRestore: need at least one mask"

    mask = luma if luma is not None else _blank_gray(filtered)

    if chroma          is not None: mask = _add_to(mask, chroma)
    if chroma_override is not None: mask = _add_to(mask, chroma_override)
    if edges           is not None: mask = _must_not_overlap(mask, edges)
    if dark            is not None: mask = _add_to(mask, dark)

    # first_plane=True: GRAY mask is broadcast to all planes of the YUV clips
    return core.std.MaskedMerge(filtered, source, mask, first_plane=True)


def _dark_delta_mask(mask: vs.VideoNode,
                     secondary: Optional[vs.VideoNode] = None,
                     edge_mask: Optional[vs.VideoNode] = None) -> vs.VideoNode:
    if secondary is not None: mask = _add_to(mask, secondary)
    if edge_mask is not None: mask = _must_not_overlap(mask, edge_mask)
    return mask


# ---------------------------------------------------------------------------
# UnsharpMask  –  luma sharpened via MakeDiff + Expr, chroma untouched
# ---------------------------------------------------------------------------

def _unsharp_mask(clip: vs.VideoNode,
                  strength: int = 80,
                  radius: int = 5,
                  threshold: int = 1) -> vs.VideoNode:
    """
    Unsharp mask applied to luma only.
    Uses native BoxBlur and MakeDiff; inverse diff applied via Expr.
    """
    luma    = _gray(clip)
    blurred = _boxblur_fn()(luma, hradius=radius, vradius=radius)

    bits = clip.format.bits_per_sample
    peak = (1 << bits) - 1
    s    = strength / 100.0

    diff           = _makediff(luma, blurred)
    sharpened_diff = _expr1(diff,
        f'x {1 << (bits-1)} - dup abs {threshold} > dup {s} * 0 ? + '
        f'{1 << (bits-1)} + 0 {peak} clamp')
    # Add sharpening back: base + diff - mid  (core.std.AddDiff does not exist in VS)
    result_luma    = _adddiff(luma, sharpened_diff)

    if clip.format.num_planes == 1:
        return result_luma

    return core.std.ShufflePlanes(
        [result_luma, clip, clip], [0, 1, 2], clip.format.color_family
    )


# ---------------------------------------------------------------------------
# RestoreGrain  –  uses MakeDiff + Expr (approximates mRD_RestoreGrain)
# ---------------------------------------------------------------------------

def _restore_grain(filtered: vs.VideoNode,
                   source: vs.VideoNode,
                   val1: int = 10,
                   val2: int = 20) -> vs.VideoNode:
    """
    Blends original film grain from source back into the cleaned filtered clip.

    The algorithm:
      diff = MakeDiff(source, filtered)          # per-pixel difference
      abs_diff = abs(diff - mid)
      # soft-limit: restore grain proportionally, fade where diff is large
      clamped = min(abs_diff, str1) - max(0, str2 - abs_diff), clamped >= 0
      weight  = sign(diff - mid)  i.e. (diff-mid) / max(abs(diff-mid), 1)
      grain_diff = clamped * weight + mid
      result = AddDiff(filtered_luma, grain_diff)

    val1 (str1): upper threshold – grain stronger than this is fully restored.
                 Range 1–50, default 10.
    val2 (str2): lower threshold – grain weaker than this is suppressed.
                 Range 1–50, default 20. Must be >= val1 for sensible results.
    """
    bits = source.format.bits_per_sample
    mid  = 1 << (bits - 1)
    peak = (1 << bits) - 1

    # Scale str1/str2 from 8-bit units to the actual bit depth
    scale = peak / 255.0
    str1  = val1 * scale
    str2  = val2 * scale

    src_luma = _gray(source)
    flt_luma = _gray(filtered)

    # diff = source - filtered, stored with mid offset (MakeDiff convention)
    diff = _makediff(src_luma, flt_luma)

    expr = (
        f'x {mid} - '                              # d = diff - mid
        f'dup abs '                                # abs_d  (stack: d abs_d)
        f'dup {str1} min '                         # min(abs_d, str1)
        f'swap {str2} swap - 0 max - '             # - max(str2 - abs_d, 0)
        f'0 max '                                  # clamp clamped >= 0
        f'swap dup abs 1 max / * '                 # * weight = d / max(abs_d,1)
        f'{mid} + 0 {peak} clamp'                  # + mid, clamp to valid range
    )
    grain_diff  = _expr1(diff, expr)
    result_luma = _adddiff(flt_luma, grain_diff)

    if filtered.format.num_planes == 1:
        return result_luma

    # Luma restored, chroma copied from filtered
    return core.std.ShufflePlanes(
        [result_luma, filtered, filtered], [0, 1, 2], filtered.format.color_family
    )


# ---------------------------------------------------------------------------
# SpotLess
# ---------------------------------------------------------------------------

def SpotLess(
    clip: vs.VideoNode,
    radT: int = 1,
    thsad: int = 10000,
    thsad2: int = None,
    pel: int = None,
    chroma: bool = True,
    ablksize: int = None,
    aoverlap: int = None,
    asearch: int = None,
    ssharp: int = None,
    pglobal: bool = True,
    rec: bool = False,
    rblksize: int = None,
    roverlap: int = None,
    rsearch: int = None,
    truemotion: bool = True,
    rfilter: int = None,
    blur: bool = False,
    smoother: str = 'tmedian',
    ref: vs.VideoNode = None,
    mStart: bool = False,
    mEnd: bool = False,
    iterations: int = 1,
    debugmask: bool = False,
) -> vs.VideoNode:
    """
    SpotLess – temporal denoising via motion-compensated median.

    Args:
        clip:       Input clip. Must be constant format and frame rate.
        radT:       Temporal radius (1–10).
        thsad:      SAD threshold at radius 1.
        thsad2:     SAD threshold at radius > 1 (defaults to thsad).
        pel:        Sub-pixel accuracy (1/2/4). Auto if None.
        chroma:     Use chroma in block matching.
        ablksize:   Block size for mv.Analyse. Auto if None.
        aoverlap:   Overlap. Default ablksize//2.
        asearch:    Search type. Default 5.
        ssharp:     mv.Super sharpness. Default 1.
        pglobal:    Global motion estimation.
        rec:        Enable recalculation pass.
        rblksize:   Block size for recalculation.
        roverlap:   Overlap for recalculation.
        rsearch:    Search type for recalculation.
        truemotion: mv truemotion flag.
        rfilter:    mv.Super rfilter. Default 2.
        blur:       Slight blur before vector analysis.
        smoother:   'tmedian' | 'ttsmooth' | 'zsmooth'.
        ref:        Optional reference clip for mv.Super.
        mStart:     Mirror-pad beginning of clip.
        mEnd:       Mirror-pad end of clip.
        iterations: Repeat the denoising chain N times.
        debugmask:  Return [input | denoised | diff] vertical stack.
    """
    if radT < 1 or radT > 10:
        raise ValueError("radT must be 1–10")
    if pel is None:
        pel = 1 if clip.width > 960 else 2
    if pel not in [1, 2, 4]:
        raise ValueError("pel must be 1, 2 or 4")

    isGRAY   = clip.format.color_family == vs.GRAY
    chroma   = False if isGRAY else chroma
    fpsnum, fpsden = clip.fps_num, clip.fps_den

    ablksize = ablksize or (32 if clip.width > 2400 else 16 if clip.width > 960 else 8)
    aoverlap = aoverlap or ablksize // 2
    asearch  = asearch  or 5
    ssharp   = ssharp   or 1
    rfilter  = rfilter  or 2

    if rec:
        rblksize = rblksize or ablksize
        rsearch  = rsearch  or asearch
        roverlap = roverlap or rblksize // 2

    thsad2 = thsad2 or thsad
    if radT >= 3:
        thsad2 = (thsad + thsad2) // 2

    S, A, R, C = core.mv.Super, core.mv.Analyse, core.mv.Recalculate, core.mv.Compensate

    if mStart or mEnd:
        head = core.std.Reverse(core.std.Trim(clip, 1, radT)) if mStart else None
        tail = core.std.Reverse(
                   core.std.Trim(clip, clip.num_frames - radT,
                                 clip.num_frames - 1)) if mEnd else None
        if   head and tail: clip = head + clip + tail
        elif head:          clip = head + clip
        elif tail:          clip = clip + tail

    denoised = clip
    for _ in range(iterations):
        supclip    = ref or (core.std.Convolution(denoised, [1,2,1,2,4,2,1,2,1])
                             if blur else denoised)
        sup        = S(supclip,  hpad=ablksize, vpad=ablksize,
                       pel=pel, sharp=ssharp, rfilter=rfilter)
        sup_render = S(denoised, levels=1,
                       pel=pel, sharp=ssharp, rfilter=rfilter)

        bv, fv = [], []
        kw = dict(search=asearch, blksize=ablksize, overlap=aoverlap,
                  chroma=chroma, truemotion=truemotion, pglobal=pglobal)
        for d in range(1, radT + 1):
            bv.append(A(sup, isb=True,  delta=d, **kw))
            fv.append(A(sup, isb=False, delta=d, **kw))

        if rec:
            kw2 = dict(blksize=rblksize, overlap=roverlap,
                       search=rsearch, truemotion=truemotion)
            for d in range(1, radT + 1):
                bv[d-1] = R(sup, bv[d-1], **kw2)
                fv[d-1] = R(sup, fv[d-1], **kw2)

        bc, fc = [], []
        for d in range(1, radT + 1):
            thresh = thsad if d == 1 else thsad2
            bc.append(C(denoised, sup_render, bv[d-1], thsad=thresh))
            fc.append(C(denoised, sup_render, fv[d-1], thsad=thresh))

        ic = core.std.Interleave(bc + [denoised] + fc)

        if   smoother == 'tmedian':  out = core.tmedian.TemporalMedian(ic, radius=radT)
        elif smoother == 'ttsmooth': out = core.ttmpsm.TTempSmooth(ic, maxr=min(7, radT))
        elif smoother == 'zsmooth':  out = core.zsmooth.TemporalMedian(ic, radius=radT)
        else: raise ValueError(f"Unknown smoother '{smoother}'")

        denoised = core.std.SelectEvery(out, radT * 2 + 1, radT)

    if mStart: denoised = core.std.Trim(denoised, radT, denoised.num_frames - 1)
    if mEnd:   denoised = core.std.Trim(denoised, 0,    denoised.num_frames - 1 - radT)

    if debugmask:
        diff = _expr_fn()([clip, denoised], ['x y - abs'])
        return core.std.StackVertical([clip, denoised, diff])

    return core.std.AssumeFPS(denoised, fpsnum=fpsnum, fpsden=fpsden)


# ---------------------------------------------------------------------------
# SpotDelta
# ---------------------------------------------------------------------------

def SpotDelta(
    clip: vs.VideoNode,

    # SpotLess
    radT: int = 1,
    thsad: int = 10000,
    pel: Optional[int] = None,
    chroma: bool = True,
    ablksize: Optional[int] = None,
    aoverlap: Optional[int] = None,
    asearch: Optional[int] = None,
    truemotion: bool = False,
    blur: bool = False,
    smoother: str = 'tmedian',

    # SpotLess – advanced motion analysis
    thsad2: Optional[int] = None,      # SAD threshold at radius > 1 (None = same as thsad)
    ssharp: Optional[int] = None,      # mv.Super sharpness (None = auto → 1)
    pglobal: bool = True,              # global motion estimation
    rec: bool = False,                 # recalculation pass for refined vectors
    rblksize: Optional[int] = None,    # block size for recalculation (None = ablksize)
    roverlap: Optional[int] = None,    # overlap for recalculation (None = rblksize//2)
    rsearch: Optional[int] = None,     # search type for recalculation (None = asearch)
    rfilter: Optional[int] = None,     # mv.Super rfilter (None = auto → 2)
    ref: Optional[vs.VideoNode] = None,# reference clip for mv.Super
    mStart: bool = False,              # mirror-pad clip start to reduce border artefacts
    mEnd: bool = False,                # mirror-pad clip end
    iterations: int = 1,              # repeat denoising chain N times

    # Sharpening
    sharpen_it: bool = True,
    usharp_strength: int = 80,
    usharp_radius: int = 5,
    usharp_th: int = 1,

    # DeltaRestore tuning
    dark2_brt: float = 0.85,
    dark2_brt_limit: float = 0.86,

    # S-Log / washed-out footage
    slog: bool = False,

    # Grain restoration
    rgr: bool = False,
    rgr1: int = 10,
    rgr2: int = 20,

    # DeltaRestore bright-object tuning (e.g. fast white ball)
    luma_brt: float = 1.1,        # brightness ratio for main luma restore mask
    luma_expand: int = 3,         # dilation amount for luma mask
    light_brt: float = 1.04,      # brightness ratio for light half of dark/light pair

    # Output mode
    output: str = 'restored',
) -> vs.VideoNode:
    """
    SpotDelta – Spotless + DeltaRestore for VapourSynth
    ====================================================
    Removes dirt/dust spots from digitised film via SpotLess (temporal
    motion-compensated median), then uses DeltaRestore to bring back any
    legitimate detail (moving objects, fast hands, balloons…) that SpotLess
    may have incorrectly removed.

    Parameters
    ----------
    clip            Input clip (YUV).
    radT            Temporal radius for SpotLess. Default 1.
    thsad           SAD threshold (default 10000 ≈ nearly off).
    pel             Sub-pixel precision (1/2/4). Auto if None.
    chroma          Use chroma in block matching.
    ablksize        MV block size. Auto-scaled to resolution if None.
    aoverlap        MV overlap. Default ablksize//2.
    asearch         MV search param. Default aoverlap//2.
    truemotion      MV truemotion flag. Default False.
    blur            Blur before vector analysis. Default False.
    smoother        'tmedian' | 'ttsmooth' | 'zsmooth'. Default 'tmedian'.
    thsad2          SAD threshold at temporal radius > 1. Default = thsad.
    ssharp          mv.Super sharpness (1–3). Default 1.
    pglobal         Use global motion estimation. Default True.
    rec             Enable recalculation pass for refined vectors. Default False.
    rblksize        Block size for recalculation. Default = ablksize.
    roverlap        Overlap for recalculation. Default = rblksize//2.
    rsearch         Search type for recalculation. Default = asearch.
    rfilter         mv.Super rfilter strength. Default 2.
    ref             Optional external reference clip for mv.Super.
    mStart          Mirror-pad clip start to reduce border artefacts. Default False.
    mEnd            Mirror-pad clip end. Default False.
    iterations      Repeat the SpotLess denoising chain N times. Default 1.
    sharpen_it      Sharpen input before SpotLess (recommended). Default True.
                    When False, the original clip is passed to SpotLess and
                    DeltaRestore unmodified – no sharpening is applied anywhere.
    usharp_strength Unsharp strength (0–200). Default 80.
    usharp_radius   Unsharp radius in pixels (1–20). Default 5.
    usharp_th       Unsharp threshold (0–255). Default 1.
    dark2_brt       Secondary dark-mask brightness ratio (0.5–1.0). Default 0.97.
    dark2_brt_limit Secondary dark-mask brightness limit (0.3–dark2_brt). Default 0.86.
    slog            Auto-level for mask generation (S-Log / washed-out).
    rgr             Restore film grain after cleaning. Default False.
    rgr1            Grain restoration blur radius for source (1–50). Default 10.
    rgr2            Grain restoration blur radius for filtered (1–50). Default 20.
    luma_brt        Brightness ratio for the main luma restore mask (1.01–1.5).
                    Default 1.1. Lower (e.g. 1.02) to restore bright fast-moving
                    objects like a white tennis ball that SpotLess removes.
    luma_expand     Dilation amount for the luma mask (0–20). Default 3.
                    Increase (e.g. 5) to cover a larger bright object.
    light_brt       Brightness ratio for the light half of the dark/light pair
                    (1.01–1.5). Default 1.04. Lower alongside luma_brt.
    output          What to return:
                      'restored'       – the cleaned clip (default)
                      'stacked'        – StackHorizontal([source, SpotLess, SpotDelta])
                      'interleaved'    – Interleave([source, SpotLess, SpotDelta])
                      'versus_stacked' – StackHorizontal([SpotLess, SpotDelta,
                                         Diff(chroma 3x)])
                      'full_stacked'   – StackHorizontal([source, SpotLess,
                                         SpotDelta, Diff(chroma 3x)])

    Example
    -------
    >>> SpotDelta(src, thsad=10000, rgr=True, rgr1=5, rgr2=10).set_output()
    >>> SpotDelta(src, output='stacked').set_output()
    """

    source = clip

    # ── S-Log: auto-level luma for mask generation only ──────────────────
    if slog:
        bits       = source.format.bits_per_sample
        peak       = (1 << bits) - 1
        source_lvl = core.std.Levels(source,
                                     min_in  = int(peak * 0.063),
                                     max_in  = int(peak * 0.922),
                                     min_out = 0,
                                     max_out = peak,
                                     planes  = [0])
    else:
        source_lvl = source

    # ── Sharpen (luma only, via MakeDiff + Expr) ──────────────────────────
    # When sharpen_it=False the original clips pass through unchanged.
    # source_shp      → fed into SpotLess and used as the DeltaRestore source
    # source_lvl_shp  → used only for mask generation (levelled version)
    if sharpen_it:
        source_shp     = _unsharp_mask(source,     usharp_strength, usharp_radius, usharp_th)
        source_lvl_shp = _unsharp_mask(source_lvl, usharp_strength, usharp_radius, usharp_th)
    else:
        source_shp     = source
        source_lvl_shp = source_lvl

    # ── SpotLess ──────────────────────────────────────────────────────────
    _blksz    = ablksize or (32 if clip.width > 1920 else 16 if clip.width > 960 else 8)
    _olap     = aoverlap or (_blksz // 2)
    _schparam = asearch  or (_olap  // 2)

    sl_kw = dict(
        radT=radT, thsad=thsad, thsad2=thsad2, pel=pel, chroma=chroma,
        ablksize=_blksz, aoverlap=_olap, asearch=_schparam,
        ssharp=ssharp, pglobal=pglobal,
        rec=rec, rblksize=rblksize, roverlap=roverlap, rsearch=rsearch,
        truemotion=truemotion, rfilter=rfilter,
        blur=blur, smoother=smoother,
        ref=ref, mStart=mStart, mEnd=mEnd, iterations=iterations,
    )

    source_shp_spt = SpotLess(source_shp, **sl_kw)

    if slog:
        source_lvl_shp_spt = SpotLess(source_lvl_shp, **sl_kw)
    else:
        source_lvl_shp_spt = source_shp_spt

    # ── Masks (all GRAY throughout) ───────────────────────────────────────

    # Luma mask: where source is brighter than filtered (bright spots)
    # luma_brt / luma_expand are exposed as parameters so fast bright objects
    # (e.g. a white tennis ball) can be tuned without touching the source.
    luma_mask = _luma_delta_mask(source_lvl_shp, source_lvl_shp_spt,
                                 '>', brightness=luma_brt, spot_size=3)
    luma_mask = _expand_mask(luma_mask, luma_expand, blur=4)

    # Chroma mask: large colour difference
    chroma_mask = _chroma_delta_mask(source_lvl_shp, source_lvl_shp_spt,
                                     delta=3, spot_size=3)

    # Suppression mask: where source is darker than filtered
    suppression_mask = _luma_delta_mask(source_lvl_shp, source_lvl_shp_spt,
                                        '<', brightness=0.9, spot_size=2)

    # Remove chroma mask areas that are darker than filtered (likely dirt)
    suppressed_chroma_mask = _suppress_with(chroma_mask, suppression_mask)

    # Edge mask: Scharr detector – sharp edges are likely dirt, not objects
    edges     = _scharr(source_lvl_shp)
    edges     = _z_padding(edges, 60, 14, 14, 10)
    edge_mask = _binarize(edges, 52)

    # Chroma override: larger delta / spot size, not suppressed
    chroma_override = _chroma_delta_mask(source_lvl_shp, source_lvl_shp_spt,
                                         delta=5, spot_size=6)
    chroma_override = _expand_mask(chroma_override, 3, blur=2)

    # Dark / light masks: restore dark blobs near light blobs (e.g. shadows)
    dark  = _luma_delta_mask(source_lvl_shp, source_lvl_shp_spt,
                             '<', brightness=dark2_brt,
                             limit_brightness=0.66, spot_size=5)
    light = _luma_delta_mask(source_lvl_shp, source_lvl_shp_spt,
                             '>', brightness=light_brt, spot_size=5)
    light = _z_padding(light, 60, 14, 14, 10)

    dark_near_light = _must_be_near(dark, light, 28)
    dark_near_light = _expand_mask(dark_near_light, 3, blur=4)

    # Secondary dark mask: stricter thresholds, no proximity requirement
    dark2 = _luma_delta_mask(source_lvl_shp, source_lvl_shp_spt,
                             '<',
                             brightness           = dark2_brt,
                             limit_brightness     = dark2_brt_limit,
                             limit_abs_brightness = 0.12,
                             spot_size            = 5)
    dark2 = _expand_mask(dark2, 2, blur=2)

    dark_edge_mask = _binarize(edges, 54)
    dark_final     = _dark_delta_mask(dark_near_light,
                                      secondary = dark2,
                                      edge_mask = dark_edge_mask)

    # ── DeltaRestore ──────────────────────────────────────────────────────
    delta_restored = _delta_restore(
        filtered        = source_shp_spt,
        source          = source_shp,
        luma            = luma_mask,
        chroma          = suppressed_chroma_mask,
        chroma_override = chroma_override,
        edges           = edge_mask,
        dark            = dark_final,
    )

    # ── Optional grain restoration ────────────────────────────────────────
    if rgr:
        delta_restored = _restore_grain(delta_restored, source, rgr1, rgr2)

    # ── Output ────────────────────────────────────────────────────────────
    if output == 'restored':
        return delta_restored

    # Label clips for comparison modes
    src_labeled      = core.text.Text(source,         'Source',    alignment=8)
    spotless_labeled = core.text.Text(source_shp_spt, 'SpotLess',  alignment=8)
    spotdel_labeled  = core.text.Text(delta_restored, 'SpotDelta', alignment=8)

    if output == 'stacked':
        return core.std.StackHorizontal([src_labeled, spotless_labeled, spotdel_labeled])
    elif output == 'interleaved':
        return core.std.Interleave([src_labeled, spotless_labeled, spotdel_labeled])
    elif output in ('versus_stacked', 'full_stacked'):
        # Exaggerated-chroma diff: luma = plain diff, chroma = 3x amplified
        exc_diff     = _exaggerated_chroma_diff(source_shp_spt, delta_restored)
        diff_labeled = core.text.Text(exc_diff, 'Diff (chroma 3x)', alignment=8)
        if output == 'versus_stacked':
            return core.std.StackHorizontal([spotless_labeled, spotdel_labeled,
                                             diff_labeled])
        else:  # full_stacked
            return core.std.StackHorizontal([src_labeled, spotless_labeled,
                                             spotdel_labeled, diff_labeled])
    else:
        raise ValueError(f"SpotDelta: unknown output mode '{output}'. "
                         f"Use 'restored', 'stacked', 'interleaved', "
                         f"'versus_stacked', or 'full_stacked'.")