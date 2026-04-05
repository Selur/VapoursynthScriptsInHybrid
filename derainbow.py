"""
dot-crawl and rainbow artefact reduction.

Requirements
------------
  mv          https://github.com/Mr-Z-2697/vapoursynth-mvtools
  zsmooth     https://github.com/adworacz/zsmooth
  warp        https://github.com/dubhater/vapoursynth-awarpsharp2
  neo_fft3d   https://github.com/HomeOfAviSynthPlusEvolution/neo_FFT3D
  fft3dfilter https://github.com/myrsloik/VapourSynth-FFT3DFilter  (fallback for neo_fft3d)
  bilateralgpu https://github.com/WolframRhodium/VapourSynth-BilateralGPU
  vszip       https://github.com/dnjulek/vapoursynth-zip
  bilateral   https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Bilateral (fallback)
  llvmexpr    https://github.com/Sunflower-Dolls/Vapoursynth-llvmexpr
  akarin      https://github.com/AkarinVS                           (fallback)
  cranexpr    https://github.com/sgt0/cranexpr                      (fallback)
  rgvs        (fallback Repair for SRFComb)

Optional Python modules
-----------------------
  color.py    https://github.com/Selur/VapoursynthScriptsInHybrid/blob/master/color.py
              Provides Tweak() for an accurate chroma-activity mask.
              If not importable, a simpler approximation is used as fallback.
  sharpen.py  https://github.com/Selur/VapoursynthScriptsInHybrid/blob/master/sharpen.py
              Provides ContraSharpening() for post-degrain detail recovery.
              If not importable, contra-sharpening is silently skipped.
"""

from __future__ import annotations
import math
import vapoursynth as vs

core = vs.core

try:
    from color import Tweak as _color_tweak  # type: ignore
except ImportError:
    _color_tweak = None

try:
    from sharpen import ContraSharpening as _contra_sharpening  # type: ignore
except ImportError:
    _contra_sharpening = None


# ---------------------------------------------------------------------------
# Plugin wrappers — each wrapper tries the fastest available backend first
# ---------------------------------------------------------------------------

def _expr(clips: vs.VideoNode | list[vs.VideoNode], expr: str | list[str]) -> vs.VideoNode:
    """Expr — prefers llvmexpr → akarin → cranexpr → std."""
    if hasattr(core, "llvmexpr"):
        return core.llvmexpr.Expr(clips, expr)
    if hasattr(core, "akarin"):
        return core.akarin.Expr(clips, expr)
    if hasattr(core, "cranexpr"):
        return core.cranexpr.Expr(clips, expr)
    return core.std.Expr(clips, expr)


def _box_blur(
    clip: vs.VideoNode,
    hradius: int = 1,
    hpasses: int = 1,
    vradius: int = 1,
    vpasses: int = 1,
    planes: list[int] | None = None,
) -> vs.VideoNode:
    """BoxBlur — prefers vszip, falls back to std."""
    kwargs: dict = dict(hradius=hradius, hpasses=hpasses, vradius=vradius, vpasses=vpasses)
    if planes is not None:
        kwargs["planes"] = planes
    if hasattr(core, "vszip"):
        return core.vszip.BoxBlur(clip, **kwargs)
    return core.std.BoxBlur(clip, **kwargs)


def _repair(clip: vs.VideoNode, ref: vs.VideoNode, mode: int) -> vs.VideoNode:
    """Repair — prefers zsmooth, falls back to rgvs."""
    if hasattr(core, "zsmooth"):
        return core.zsmooth.Repair(clip, ref, mode)
    return core.rgvs.Repair(clip, ref, mode)


def _checkmate(clip: vs.VideoNode) -> vs.VideoNode:
    """Checkmate deinterlacing-artefact removal via vszip (no-op if unavailable)."""
    if not hasattr(core, "vszip"):
        return clip
    bits = clip.format.bits_per_sample
    if bits != 8:
        fmt8 = clip.format.replace(bits_per_sample=8)
        clip8 = core.resize.Point(clip, format=fmt8)
        result = core.vszip.Checkmate(clip8)
        return core.resize.Point(result, format=clip.format)
    return core.vszip.Checkmate(clip)


def _fft3d(clip: vs.VideoNode, **kwargs) -> vs.VideoNode:
    """FFT3D — prefers neo_fft3d, falls back to fft3dfilter."""
    if hasattr(core, "neo_fft3d"):
        return core.neo_fft3d.FFT3D(clip, **kwargs)
    return core.fft3dfilter.FFT3DFilter(clip, **kwargs)


def _bilateral(clip: vs.VideoNode, sigmaS: float = 3.0, sigmaR: float = 0.02, **kwargs) -> vs.VideoNode:
    """Bilateral filter — prefers bilateralgpu, then vszip, then bilateral."""
    if hasattr(core, "bilateralgpu"):
        return core.bilateralgpu.Bilateral(clip, sigmaS=sigmaS, sigmaR=sigmaR, **kwargs)
    if hasattr(core, "vszip"):
        return core.vszip.Bilateral(clip, sigmaS=sigmaS, sigmaR=sigmaR, **kwargs)
    if hasattr(core, "bilateral"):
        return core.bilateral.Bilateral(clip, sigmaS=sigmaS, sigmaR=sigmaR, **kwargs)
    raise RuntimeError(
        "srfcomb: a bilateral filter plugin is required (bilateralgpu, vszip, or bilateral) — "
        "install one from https://github.com/dnjulek/vapoursynth-zip"
    )


def _temporal_soften(
    clip: vs.VideoNode,
    radius: int,
    luma_threshold: int,
    chroma_threshold: int,
    scenechange: int,
    mode: int,
) -> vs.VideoNode:
    """TemporalSoften — focus2 or zsmooth as fallback."""
    if hasattr(core, "focus2"):
        return core.focus2.TemporalSoften2(
            clip, radius, luma_threshold, chroma_threshold, scenechange, mode
        )
    return core.zsmooth.TemporalSoften(
        clip, radius, [luma_threshold, chroma_threshold], scenechange
    )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _luma(clip: vs.VideoNode) -> vs.VideoNode:
    return core.std.ShufflePlanes(clip, 0, vs.GRAY)


def _get_plane(clip: vs.VideoNode, plane: int) -> vs.VideoNode:
    """Extract a single plane as a GRAY clip."""
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('"clip" must be a clip!')
    sNumPlanes = clip.format.num_planes
    if plane < 0 or plane >= sNumPlanes:
        raise ValueError(f'valid range of "plane" is [0, {sNumPlanes})')
    return core.std.ShufflePlanes(clip, plane, vs.GRAY)


def _weave_fields(clip: vs.VideoNode) -> vs.VideoNode:
    """DoubleWeave then drop odd frames — equivalent to AviSynth Weave()."""
    return core.std.DoubleWeave(clip)[::2]


def _mt_logic(a: vs.VideoNode, b: vs.VideoNode, mode: str = "min") -> vs.VideoNode:
    op = "x y min" if mode == "min" else "x y max"
    return _expr([a, b], op)


def _mt_binarize(clip: vs.VideoNode, threshold: int) -> vs.VideoNode:
    peak = (1 << clip.format.bits_per_sample) - 1
    return _expr([clip], f"x {threshold} > {peak} 0 ?")


def _mt_expand(clip: vs.VideoNode, n: int = 1) -> vs.VideoNode:
    for _ in range(n):
        clip = core.std.Maximum(clip)
    return clip


def _mt_expand_horizontal(clip: vs.VideoNode, n: int = 1) -> vs.VideoNode:
    for _ in range(n):
        clip = core.std.Maximum(clip, coordinates=[0, 0, 0, 1, 1, 0, 0, 0])
    return clip


def _mt_inflate(clip: vs.VideoNode, n: int = 1) -> vs.VideoNode:
    for _ in range(n):
        clip = core.std.Inflate(clip)
    return clip


def _scale_chroma_mask(
    mask: vs.VideoNode,
    target_width: int,
    target_height: int,
    is_420_or_422: bool,
) -> vs.VideoNode:
    """Resize a chroma mask with the correct sub-sampling offset."""
    src_left = -0.25 if is_420_or_422 else 0.0
    return core.resize.Bilinear(mask, target_width, target_height, src_left=src_left)


def _cround(x: float) -> int:
    return math.floor(x + 0.5) if x > 0 else math.ceil(x - 0.5)


def _scale_value(value: int | float, peak: int) -> int:
    """Scale a value from 8-bit range to the target bit depth."""
    return _cround(value * peak / 255) if peak != 1 else value / 255


# ---------------------------------------------------------------------------
# Public function: LUTDeRainbow
# ---------------------------------------------------------------------------

def LUTDeRainbow(
    input: vs.VideoNode,
    cthresh: int = 10,
    ythresh: int = 10,
    y: bool = True,
    linkUV: bool = True,
    mask: bool = False,
) -> vs.VideoNode:
    """
    LUTDeRainbow — frame-based derainbowing by Scintilla.
    Last updated 2022-10-08.

    Requires YUV input, frame-based only.

    Parameters
    ----------
    cthresh : int
        How close chroma values in prev/next frames must be to trigger
        correction. Higher values catch more rainbows but may introduce
        artifacts. Keep below ~20.
    ythresh : int
        Same as cthresh but for luma (only used when y=True).
    y : bool
        Whether luma difference is considered when selecting pixels to fix.
    linkUV : bool
        When True, a pixel is only fixed if both U and V meet the threshold.
        When False, U and V are masked independently.
    mask : bool
        When True, return the combined UV mask instead of the processed clip.
    """
    inputbits = input.format.bits_per_sample
    if not isinstance(input, vs.VideoNode) or input.format.color_family != vs.YUV or inputbits > 16:
        raise vs.Error('LUTDeRainbow: This is not an 8-16 bit YUV clip')

    # LUT2 cannot handle clips with more than 10 bits; use Expr + MaskedMerge instead.
    useExpr = inputbits > 10

    shift = inputbits - 8
    peak  = (1 << inputbits) - 1

    cthresh_scaled = _scale_value(cthresh, peak)
    ythresh_scaled = _scale_value(ythresh, peak)

    input_minus = input.std.DuplicateFrames(frames=[0])
    input_plus  = input.std.Trim(first=1) + input.std.Trim(first=input.num_frames - 1)

    input_u       = _get_plane(input, 1)
    input_v       = _get_plane(input, 2)
    input_minus_y = _get_plane(input_minus, 0)
    input_minus_u = _get_plane(input_minus, 1)
    input_minus_v = _get_plane(input_minus, 2)
    input_plus_y  = _get_plane(input_plus, 0)
    input_plus_u  = _get_plane(input_plus, 1)
    input_plus_v  = _get_plane(input_plus, 2)

    average_y = _expr(
        [input_minus_y, input_plus_y],
        f'x y - abs {ythresh_scaled} < {peak} 0 ?',
    ).resize.Bilinear(input_u.width, input_u.height)

    average_u = _expr(
        [input_minus_u, input_plus_u],
        f'x y - abs {cthresh_scaled} < x y + 2 / 0 ?',
    )
    average_v = _expr(
        [input_minus_v, input_plus_v],
        f'x y - abs {cthresh_scaled} < x y + 2 / 0 ?',
    )

    umask = average_u.std.Binarize(threshold=21 << shift)
    vmask = average_v.std.Binarize(threshold=21 << shift)

    if useExpr:
        themask = _expr([umask, vmask], f'x y + {peak + 1} < 0 {peak} ?')
        if y:
            umask   = core.std.MaskedMerge(core.std.BlankClip(average_y), average_y, umask)
            vmask   = core.std.MaskedMerge(core.std.BlankClip(average_y), average_y, vmask)
            themask = core.std.MaskedMerge(core.std.BlankClip(average_y), average_y, themask)
    else:
        themask = core.std.Lut2(umask, vmask, function=lambda x, y: x & y)
        if y:
            umask   = core.std.Lut2(umask,   average_y, function=lambda x, y: x & y)
            vmask   = core.std.Lut2(vmask,   average_y, function=lambda x, y: x & y)
            themask = core.std.Lut2(themask, average_y, function=lambda x, y: x & y)

    fixed_u = core.std.Merge(average_u, input_u)
    fixed_v = core.std.Merge(average_v, input_v)

    output_u = core.std.MaskedMerge(input_u, fixed_u, themask if linkUV else umask)
    output_v = core.std.MaskedMerge(input_v, fixed_v, themask if linkUV else vmask)

    output = core.std.ShufflePlanes(
        [input, output_u, output_v], planes=[0, 0, 0],
        colorfamily=input.format.color_family,
    )

    if mask:
        return themask.resize.Point(input.width, input.height)
    return output


# ---------------------------------------------------------------------------
# Public function: SRFComb (field-space version)
# ---------------------------------------------------------------------------

def SRFComb(
    clip: vs.VideoNode,
    DotCrawlThSAD: int = 500,
    RainbowThSAD: int = 500,
    SpatialDeDotCraw: bool = True,
) -> vs.VideoNode:
    """
    SRFComb by real.finder — field-space version.
    https://github.com/realfinder/AVS-Stuff/blob/master/avs%202.6%20and%20up/SRFComb.avsi

    Best of many AVS Rainbow & Dot Crawl Removal approaches.
    Version 1.01 — converted to VapourSynth.

    Requires: mv, rgvs/zsmooth, std, resize core plugins,
              color.Tweak (optional, falls back to approximation if unavailable)
    Optional: zsmooth (faster Repair), vszip (faster BoxBlur + Checkmate)

    Parameters
    ----------
    DotCrawlThSAD : int
        SAD threshold for dot-crawl motion compensation.
    RainbowThSAD : int
        SAD threshold for rainbow motion compensation.
    SpatialDeDotCraw : bool
        Enable spatial luma de-dot-crawl pre-processing step.
    """
    fmt = clip.format
    assert fmt is not None,             "SRFComb: clip must have a known format"
    assert fmt.color_family != vs.RGB,  "SRFComb: Planar YUV input only"
    assert fmt.color_family != vs.GRAY, "SRFComb: does not work with Greyscale video"

    fullchr = fmt.subsampling_w == 0 and fmt.subsampling_h == 0  # 4:4:4
    chr420  = fmt.subsampling_w == 1 and fmt.subsampling_h == 1  # 4:2:0
    chr422  = fmt.subsampling_w == 1 and fmt.subsampling_h == 0  # 4:2:2

    peak  = (1 << fmt.bits_per_sample) - 1
    half  = 1 << (fmt.bits_per_sample - 1)
    scale = peak / 255

    thSAD  = DotCrawlThSAD
    thSADC = RainbowThSAD

    # Everything works in field-space until the final Weave
    separated = core.std.SeparateFields(clip, tff=True)
    ogs = separated

    # --- LumaSpatialDeDot (field-space) ---
    if SpatialDeDotCraw:
        sep_y = core.std.ShufflePlanes(separated, 0, vs.GRAY)
        sep_u = core.std.ShufflePlanes(separated, 1, vs.GRAY)
        sep_v = core.std.ShufflePlanes(separated, 2, vs.GRAY)
        blurred          = _box_blur(sep_y, hradius=1, hpasses=3, vradius=0, vpasses=0)
        sharpened        = core.std.Convolution(blurred, [0, -1, 0, -1, 5, -1, 0, -1, 0])
        luma_dedc        = core.std.ShufflePlanes([sharpened, sep_u, sep_v], [0, 0, 0], vs.YUV)
        LumaSpatialDeDot = _repair(luma_dedc, separated, 5)
    else:
        LumaSpatialDeDot = separated

    # --- pre: luma processing chain ---
    ogy    = core.std.ShufflePlanes(clip, 0, vs.GRAY)
    dedc   = _repair(ogy, _box_blur(ogy, hradius=1, hpasses=1, vradius=1, vpasses=1), 1)
    tr1    = _repair(dedc, ogy, 3)
    cm     = _checkmate(tr1)
    tr2    = _repair(cm, ogy, 3)
    rep    = _repair(tr2, ogy, 1)
    rep_sf   = core.std.SeparateFields(rep, tff=True)
    ablurred = _box_blur(rep_sf,   hradius=1, hpasses=1, vradius=0, vpasses=0)
    blurred2 = _box_blur(ablurred, hradius=1, hpasses=3, vradius=0, vpasses=0)
    pre_y    = core.std.Convolution(blurred2, [0, -1, 0, -1, 5, -1, 0, -1, 0])

    pre = core.std.ShufflePlanes(
        [pre_y,
         core.std.ShufflePlanes(ogs, 1, vs.GRAY),
         core.std.ShufflePlanes(ogs, 2, vs.GRAY)],
        [0, 0, 0], vs.YUV,
    )

    # --- preymask: horizontal edge mask on separated luma ---
    ogy_sf    = core.std.SeparateFields(ogy, tff=True)
    ogy_vblur = _box_blur(ogy_sf, vradius=1, vpasses=3, hradius=0, hpasses=0)
    ogy_vshrp = core.std.Convolution(ogy_vblur, [0, -1, 0, -1, 5, -1, 0, -1, 0])
    preymask  = core.std.Convolution(ogy_vshrp, [0, 0, 0, -1, 0, 1, 0, 0, 0])
    preymask  = _expr(preymask, f"x {4*scale} < 0 x {8*scale} > {peak} x ? ?")

    # --- precmask: horizontal edge mask on pre_y ---
    pre_y_edge = core.std.Convolution(pre_y, [0, 0, 0, -1, 0, 1, 0, 0, 0])
    precmask   = _expr(pre_y_edge, f"x {round(scale)} < 0 x {round(3*scale)} > {peak} x ? ?")

    # --- cuvmask: chroma activity mask ---
    if _color_tweak is not None:
        sat0  = _color_tweak(ogs, sat=0.0,  coring=False)
        sat10 = _color_tweak(ogs, sat=10.0, coring=False)
    else:
        # Fallback approximation when color.py is unavailable
        sat0  = core.std.ShufflePlanes(
            [core.std.BlankClip(ogs, color=[0, half, half]),
             core.std.ShufflePlanes(ogs, 1, vs.GRAY),
             core.std.ShufflePlanes(ogs, 2, vs.GRAY)],
            [0, 0, 0], vs.YUV,
        )
        sat10 = ogs  # simplified; true sat=10 boost not available without color.py

    diff_u = core.std.MakeDiff(
        core.std.ShufflePlanes(sat0,  1, vs.GRAY),
        core.std.ShufflePlanes(sat10, 1, vs.GRAY),
    )
    diff_v = core.std.MakeDiff(
        core.std.ShufflePlanes(sat0,  2, vs.GRAY),
        core.std.ShufflePlanes(sat10, 2, vs.GRAY),
    )
    lut_expr = f"x {half} = 0 x {half} - abs {peak} * {half} / ?"
    cuvmask  = _expr([_expr(diff_u, lut_expr), _expr(diff_v, lut_expr)], "x y max")

    # Upscale cuvmask to full luma field dimensions
    field_w = ogs.width
    field_h = ogs.height
    shift_val = 0.25 if (chr422 or chr420) else (None if fullchr else 0.375)
    cuvmaskf  = core.resize.Bilinear(
        cuvmask, field_w, field_h,
        src_left=shift_val if shift_val is not None else 0,
    )

    # --- premask: intersection of chroma activity and pre edge ---
    premask = _expr([cuvmaskf, precmask], "x y min")
    premask = core.std.Maximum(premask, coordinates=[0, 0, 0, 1, 1, 0, 0, 0])
    premask = core.std.Inflate(premask)

    # --- ycombmask: intersection of chroma activity and luma edge ---
    ycombmask = _expr([cuvmaskf, preymask], "x y min")
    ycombmask = core.std.Maximum(ycombmask, coordinates=[0, 0, 0, 1, 1, 0, 0, 0])
    ycombmask = core.std.Maximum(ycombmask, coordinates=[0, 0, 0, 1, 1, 0, 0, 0])
    ycombmask = core.std.Inflate(ycombmask)

    # uvcombmask: downscale premask to chroma dimensions, then back up for MaskedMerge
    cuv_w     = diff_u.width
    cuv_h     = diff_u.height
    neg_shift = -0.25 if (chr422 or chr420) else (None if fullchr else -0.375)
    uvcombmask = core.resize.Bilinear(
        premask, cuv_w, cuv_h,
        src_left=neg_shift if neg_shift is not None else 0,
    )
    uvcombmask_full = core.resize.Bilinear(uvcombmask, field_w, field_h)

    # --- pre2: blend pre into ogs on luma using premask ---
    pre2 = core.std.MaskedMerge(ogs, pre, premask, planes=[0])
    pre2 = core.std.Merge(pre2, _repair(pre2, ogs, 1))

    # --- MVTools motion analysis ---
    super_search = core.mv.Super(pre2, pel=4, rfilter=4)
    bv1 = core.mv.Analyse(super_search, blksize=8, isb=True,  delta=2, overlap=2, dct=8)
    fv1 = core.mv.Analyse(super_search, blksize=8, isb=False, delta=2, overlap=2, dct=8)

    # --- lastLumaSpatialDeDot ---
    lastLSD   = core.std.MaskedMerge(separated, LumaSpatialDeDot, ycombmask, planes=[0])
    super_lsd = core.mv.Super(lastLSD, pel=4, levels=1)
    degrained = core.mv.Degrain1(lastLSD, super_lsd, bv1, fv1, thsad=thSAD, thsadc=thSADC)

    # --- Final merge: ycombmask on luma, uvcombmask on chroma ---
    merged = core.std.MaskedMerge(ogs,    degrained, ycombmask,        planes=[0])
    merged = core.std.MaskedMerge(merged, degrained, uvcombmask_full,  planes=[1, 2])

    return _weave_fields(merged)


# ---------------------------------------------------------------------------
# Public function: SRFComb2
# ---------------------------------------------------------------------------

def SRFComb2(
    clip: vs.VideoNode,
    *,
    dedot: bool | None = None,
    derainbow: bool | None = None,
    thsad: int | None = None,
    thsadc: int | None = None,
    pal: bool | None = None,
    progressive: bool | None = None,
    contrasharp: bool = True,
) -> vs.VideoNode:
    """
    SRFComb2 — spatial + temporal dot-crawl and rainbow artefact reduction.

    Ported from the AviSynth function by real.finder (Doom9), v1.01.

    Parameters
    ----------
    clip : vs.VideoNode
        YUV planar input. Both interlaced and progressive are supported.
        For interlaced material the clip should *not* be pre-separated;
        fields are separated internally.
    dedot : bool, optional
        Enable luma (dot-crawl) reduction. Defaults to *progressive*.
    derainbow : bool, optional
        Enable chroma (rainbow) reduction. Defaults to *progressive*.
    thsad : int, optional
        SAD threshold for dot-crawl motion compensation.
        Defaults to 700 (progressive) or 500 (interlaced).
    thsadc : int, optional
        SAD threshold for rainbow motion compensation.
        Defaults to 700 (progressive) or 500 (interlaced).
    pal : bool, optional
        Hint that the source is PAL (25 fps). Affects mask expansion
        direction. Auto-detected from frame-rate when not given.
    progressive : bool, optional
        Force progressive (True) or interlaced (False) handling.
        When omitted, auto-detected from the _FieldBased frame property.
    contrasharp : bool
        Apply contra-sharpening after motion compensation to recover
        detail lost during degrain (default True).
    """
    if clip.format.color_family != vs.YUV:
        raise ValueError("srfcomb2: YUV planar input required")
    if clip.format.num_planes == 1:
        raise ValueError("srfcomb2: greyscale input is not supported")

    # ------------------------------------------------------------------
    # Resolve defaults
    # ------------------------------------------------------------------
    if progressive is None:
        field_based = clip.get_frame(0).props.get('_FieldBased', 0)
        progressive = field_based == 0
    else:
        field_based = 0 if progressive else 2

    dedot     = progressive if dedot     is None else dedot
    derainbow = progressive if derainbow is None else derainbow
    thsad     = (700 if progressive else 500) if thsad  is None else thsad
    thsadc    = (700 if progressive else 500) if thsadc is None else thsadc

    if pal is None:
        fps = clip.fps_num / clip.fps_den if clip.fps_den else 0
        pal = abs(fps - 25.0) < 0.1

    ss     = clip.format.subsampling_w
    is_sub = ss > 0  # True for 4:2:0 / 4:2:2

    # ------------------------------------------------------------------
    # Field separation (interlaced path)
    # ------------------------------------------------------------------
    fields = clip if progressive else core.std.SeparateFields(clip, tff=field_based == 2)
    oY = _luma(fields)

    # ------------------------------------------------------------------
    # Luma edge / comb mask
    # ------------------------------------------------------------------
    e_h = core.std.Convolution(oY, [0, 0, 0, -16, 0, 16, 0, 0, 0], saturate=False)
    e_v = core.std.Convolution(oY, [0,  0,  0,   0, 2, -2, 0, 0, 0], saturate=False)

    luma_edge = _mt_logic(e_h, e_v, "min")
    luma_mask = _mt_logic(luma_edge, core.std.Invert(oY), "min")

    if pal:
        luma_mask = _mt_expand(luma_mask, 2)
    else:
        luma_mask = _mt_expand_horizontal(luma_mask, 2)
    luma_mask = _mt_inflate(luma_mask)

    # ------------------------------------------------------------------
    # UV difference mask (ouvm) — chroma activity map
    # ------------------------------------------------------------------
    mid = 1 << (clip.format.bits_per_sample - 1)

    if _color_tweak is not None:
        tweak_zero  = _color_tweak(fields, sat=0)
        tweak_boost = _color_tweak(fields, sat=20 if pal else 10)
        chroma_diff = core.std.MakeDiff(tweak_zero, tweak_boost)
        u_diff_raw  = core.std.ShufflePlanes(chroma_diff, 1, vs.GRAY)
        v_diff_raw  = core.std.ShufflePlanes(chroma_diff, 2, vs.GRAY)
        uv_diff     = core.std.Interleave([u_diff_raw, v_diff_raw])
        uv_mapped   = _expr([uv_diff], f"x {mid} = 0 {mid} x - abs 2.28 * ?")
        ouvm_u      = core.std.SelectEvery(uv_mapped, 2, [0])
        ouvm_v      = core.std.SelectEvery(uv_mapped, 2, [1])
        ouvm_sub    = _mt_logic(ouvm_u, ouvm_v, "max")
    else:
        u_plane  = core.std.ShufflePlanes(fields, 1, vs.GRAY)
        v_plane  = core.std.ShufflePlanes(fields, 2, vs.GRAY)
        u_dev    = _expr([u_plane], f"x {mid} - abs 2.28 *")
        v_dev    = _expr([v_plane], f"x {mid} - abs 2.28 *")
        ouvm_sub = _mt_logic(u_dev, v_dev, "max")

    ouvm      = _scale_chroma_mask(ouvm_sub, oY.width, oY.height, is_sub)
    luma_mask = _mt_logic(luma_mask, _mt_binarize(ouvm, 30), "min")

    # ------------------------------------------------------------------
    # Luma denoise (FFT3D)
    # ------------------------------------------------------------------
    cle_y = _fft3d(
        oY,
        sigma =15 if progressive else 33,
        sigma2=15 if progressive else 33,
        sigma3=10 if progressive else 22,
        sigma4=0,
        bt=1,
    )

    if progressive:
        blur   = _box_blur(oY, hradius=1, vradius=1)
        detail = core.std.MakeDiff(oY, blur)
        cle_y  = core.std.MergeDiff(cle_y, _expr([detail], "x 0.25 *"))
    else:
        cle_y = core.std.Merge(cle_y, core.warp.ABlur(oY, blur=2))

    cle_y_woven = cle_y if progressive else _weave_fields(cle_y)
    cle_y_woven = _checkmate(cle_y_woven)
    cle_y_woven = core.zsmooth.TemporalRepair(cle_y_woven, cle_y_woven, mode=3)

    if not progressive:
        cle_y_sep   = core.std.SeparateFields(cle_y_woven)
        cle_y_woven = core.zsmooth.Repair(cle_y_sep, cle_y, mode=3)
        cle_y_woven = _weave_fields(cle_y_woven)
    else:
        cle_y_woven = core.zsmooth.Repair(cle_y_woven, oY, mode=3)

    luma_cleaned_yuv = core.std.ShufflePlanes(
        [cle_y_woven, fields, fields], [0, 1, 2], vs.YUV
    )
    merged = core.std.MaskedMerge(fields, luma_cleaned_yuv, luma_mask)

    # ------------------------------------------------------------------
    # Chroma comb mask
    # ------------------------------------------------------------------
    chm_h = core.std.Convolution(oY, [0, 0, 0, -1, 0, 1, 0, 0, 0], saturate=False)
    chm_v = core.std.Convolution(oY, [0,  0,  0,  0, 2, -2, 0, 0, 0], saturate=False)

    chm = _mt_logic(chm_h, chm_v, "min")
    chm = _mt_logic(chm, _mt_binarize(core.std.Invert(oY), 50), "min")
    chm = _mt_expand_horizontal(chm)
    chm = _mt_inflate(chm)

    # ------------------------------------------------------------------
    # Chroma denoise (FFT3D)
    # ------------------------------------------------------------------
    chroma_clean = _fft3d(
        merged,
        sigma =0,
        sigma2=11 if progressive else 22,
        sigma3= 5 if progressive else 11,
        sigma4=22 if progressive else 44,
        bt=1,
    )
    spati_comb_c = core.std.MaskedMerge(merged, chroma_clean, chm)

    # ------------------------------------------------------------------
    # UV bilateral smoothing
    # ------------------------------------------------------------------
    u_in = core.std.ShufflePlanes(spati_comb_c, 1, vs.GRAY)
    v_in = core.std.ShufflePlanes(spati_comb_c, 2, vs.GRAY)
    uv_interleaved = core.std.Interleave([u_in, v_in])
    uv_filtered    = _bilateral(uv_interleaved, sigmaS=1.4, sigmaR=0.028)
    u_filtered = core.std.SelectEvery(uv_filtered, 2, [0])
    v_filtered = core.std.SelectEvery(uv_filtered, 2, [1])
    spati_comb_c = core.std.ShufflePlanes(
        [spati_comb_c, u_filtered, v_filtered], [0, 0, 0], vs.YUV
    )

    # ------------------------------------------------------------------
    # aWarpSharp2 chroma sharpening
    # ------------------------------------------------------------------
    spati_comb_c = core.warp.AWarpSharp2(spati_comb_c, depth=[16, 8, 8], chroma=1, planes=[1, 2])

    # Combined luma+chroma combmask for the final merge
    chm_uv_resized = _scale_chroma_mask(chm, ouvm_sub.width, ouvm_sub.height, is_sub)
    full_comb_mask = core.std.ShufflePlanes(
        [luma_mask, chm_uv_resized, chm_uv_resized], [0, 0, 0], vs.YUV
    )

    # ------------------------------------------------------------------
    # Motion compensation
    # ------------------------------------------------------------------
    if progressive:
        super_search  = core.mv.Super(spati_comb_c, rfilter=4)
        super_degrain = core.mv.Super(fields, levels=1)

        bv = core.mv.Analyse(super_search, isb=True,  delta=1, overlap=2, dct=8)
        fv = core.mv.Analyse(super_search, isb=False, delta=1, overlap=2, dct=8)

        degrain = core.mv.Degrain1(fields, super_degrain, bv, fv,
                                   thsad=thsad, thsadc=thsadc)
    else:
        even_sc = core.std.SelectEvery(spati_comb_c, 2, [0])
        odd_sc  = core.std.SelectEvery(spati_comb_c, 2, [1])
        even_o  = core.std.SelectEvery(fields, 2, [0])
        odd_o   = core.std.SelectEvery(fields, 2, [1])

        sup_se = core.mv.Super(even_sc, rfilter=4)
        sup_so = core.mv.Super(odd_sc,  rfilter=4)
        sup_de = core.mv.Super(even_o,  levels=1)
        sup_do = core.mv.Super(odd_o,   levels=1)

        bv_e = core.mv.Analyse(sup_se, isb=True,  delta=1, overlap=2, dct=8)
        fv_e = core.mv.Analyse(sup_se, isb=False, delta=1, overlap=2, dct=8)
        bv_o = core.mv.Analyse(sup_so, isb=True,  delta=1, overlap=2, dct=8)
        fv_o = core.mv.Analyse(sup_so, isb=False, delta=1, overlap=2, dct=8)

        even_d  = core.mv.Degrain1(even_o, sup_de, bv_e, fv_e, thsad=thsad, thsadc=thsadc)
        odd_d   = core.mv.Degrain1(odd_o,  sup_do, bv_o, fv_o, thsad=thsad, thsadc=thsadc)
        degrain = core.std.Interleave([even_d, odd_d])

    # ------------------------------------------------------------------
    # Motion mask (only when dedot or derainbow is active)
    # ------------------------------------------------------------------
    if dedot or derainbow:
        if progressive:
            mmask_bv = core.mv.Mask(fields, bv, kind=0, ml=1, ysc=255)
            mmask_fv = core.mv.Mask(fields, fv, kind=0, ml=1, ysc=255)
        else:
            mmask_bv_e = core.mv.Mask(even_o, bv_e, kind=0, ml=1, ysc=255)
            mmask_fv_e = core.mv.Mask(even_o, fv_e, kind=0, ml=1, ysc=255)
            mmask_bv_o = core.mv.Mask(odd_o,  bv_o, kind=0, ml=1, ysc=255)
            mmask_fv_o = core.mv.Mask(odd_o,  fv_o, kind=0, ml=1, ysc=255)
            mmask_e    = _mt_logic(mmask_bv_e, mmask_fv_e, "max")
            mmask_o    = _mt_logic(mmask_bv_o, mmask_fv_o, "max")
            mmask_bv   = core.std.Interleave([mmask_e, mmask_o])
            mmask_fv   = mmask_bv

        motion_mask = _mt_logic(mmask_bv, mmask_fv, "max") if progressive else mmask_bv

        ts          = _temporal_soften(motion_mask, 2, 255, 255, 0, 2)
        motion_mask = _mt_logic(motion_mask, _mt_expand(ts, 1), "max")
        motion_mask = _mt_expand(motion_mask, 1)
        motion_mask = _mt_expand_horizontal(motion_mask)
        motion_mask = _mt_inflate(motion_mask)

        if contrasharp and _contra_sharpening is not None:
            sharp = _contra_sharpening(spati_comb_c, degrain)
        else:
            sharp = spati_comb_c

        merged = core.std.MaskedMerge(
            degrain, sharp, motion_mask,
            planes=(
                [0]      if dedot     and not derainbow else
                [1, 2]   if derainbow and not dedot     else
                [0, 1, 2]
            ),
        )
    else:
        merged = degrain

    # ------------------------------------------------------------------
    # Final merge: blend result back over original via full combmask
    # ------------------------------------------------------------------
    final = core.std.MaskedMerge(fields, merged, full_comb_mask)

    return final if progressive else _weave_fields(final)
