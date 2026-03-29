"""
SeeSaw — "Denoiser-and-Sharpener-are-riding-the-SeeSaw"

Original AviSynth function by Didée (https://avisynth.nl/images/SeeSaw.avs).
VapourSynth port based on muvsfunc by WolframRhodium.

Implements the "crystality sharpen" principle: source, a denoised copy, and a
modified sharpener are intermixed in a seesaw-like fashion.  The goal is to
lift weak detail without oversharpening strong edges, remain temporally stable
(no shimmer), and stay within reasonable bitrate impact.

The pipeline in brief:
  1. Produce a gently clamped "tame" clip — a blend of source and denoised
     source whose blend ratio is governed by `bias`.
  2. Sharpen `tame` using a modified power-curve sharpener (Sharpen2).
     Optionally supersample before sharpening and repair artefacts against a
     1:1 reference pass.
  3. Stabilise the sharpened result temporally and/or spatially via SootheSS.
  4. Apply the resulting sharpening difference back onto a noise-limited copy
     of the original luma, subject to a final absolute or relative clamp.
  5. Recombine with the original chroma for YUV input.
"""

import vapoursynth as vs

core = vs.core

def _Expr(clips, expr):
    if hasattr(core, 'llvmexpr'):
        return core.llvmexpr.Expr(clips, expr)
    elif hasattr(core, 'akarin'):
        return core.akarin.Expr(clips, expr)
    elif hasattr(core, 'cranexpr'):
        return core.cranexpr.Expr(clips, expr)
    else:
        return core.std.Expr(clips, expr)

def _RemoveGrain(clip: "vs.VideoNode", mode: int) -> "vs.VideoNode":
    if hasattr(core, "zsmooth"):
        return core.zsmooth.RemoveGrain(clip, mode)
    if clip.format.sample_type == vs.FLOAT:
        return core.rgsf.RemoveGrain(clip, mode)
    return core.rgvs.RemoveGrain(clip, mode)

def _Repair(clip: "vs.VideoNode", reference: "vs.VideoNode", mode: int) -> "vs.VideoNode":
    if hasattr(core, "zsmooth"):
        return core.zsmooth.Repair(clip, reference, mode)
    if clip.format.sample_type == vs.FLOAT and hasattr(core, 'rgsf'):
        return core.rgsf.Repair(clip, reference, mode)
    return core.rgvs.Repair(clip, reference, mode)

def _TemporalSoften(clip: "vs.VideoNode", radius: int, threshold: int) -> "vs.VideoNode":
    """Temporal soften with scene-change awareness.

    zsmooth.TemporalSoften uses a flat threshold array and scenechange=-1 to
    honour existing _SceneChangePrev/_SceneChangeNext properties set upstream.
    focus2.TemporalSoften2 (the legacy fallback) uses separate luma/chroma
    threshold arguments and its own scenechange parameter.
    """
    if hasattr(core, "zsmooth"):
        # threshold is specified in 8-bit units; zsmooth expects the native
        # bit-depth scale, so pass scalep=True for automatic conversion.
        # scenechange=-1 reuses any _SceneChangePrev/Next props already on the clip.
        return core.zsmooth.TemporalSoften(clip, radius=radius, threshold=[threshold], scenechange=-1, scalep=True)
    # Legacy fallback: luma_threshold=threshold, chroma disabled (0), scenechange=32.
    return core.focus2.TemporalSoften2(clip, radius, threshold, 0, 32, 2)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _m4(x: float) -> int:
    """Round *x* up to the nearest multiple of 4, with a minimum of 16.

    Used to keep supersampled dimensions on a MOD4 boundary, which is required
    by most chroma subsampling formats and avoids half-pixel alignment errors
    when resizing back to the original resolution.
    """
    return max(16, int(round(x / 4.0) * 4))


def _scale(value: float, peak: float) -> float:
    """Map an 8-bit-scale value to the bit-depth-appropriate scale.

    All threshold constants in the original AviSynth script are expressed in
    8-bit units (0-255).  For higher bit depths or float clips the same
    perceptual threshold must be rescaled to the actual sample range.
    """
    return value * peak / 255.0


# ---------------------------------------------------------------------------
# Modified sharpener
# ---------------------------------------------------------------------------

def Sharpen2(
    clip: vs.VideoNode,
    strength: float,
    power: float,
    zp: float,
    lodmp: float,
    hidmp: float,
    rgmode: int,
    nsmode: int = 1,
    diff: bool = False,
) -> vs.VideoNode:
    """Power-curve unsharp-mask sharpener used internally by SeeSaw.

    Rather than a plain unsharp mask (difference * constant), the sharpening
    difference is passed through a power curve centred on *zp*:

        sharpened_diff = sign(d) * (|d| / zp)^(1/power) * zp * strength

    This has two useful properties:
      - Below *zp* the curve is convex, so small differences (fine texture) get
        relatively more boost — recovering detail lost by denoising.
      - Above *zp* the curve is concave, so large differences (strong edges)
        get relatively less boost — suppressing haloing and aliasing.

    *lodmp* further damps the overdrive region for tiny differences, avoiding
    amplification of residual noise right at the zero point.  *hidmp* rolls off
    the gain for very large differences, acting as a soft ceiling.

    Args:
        clip:      Input clip (luma plane only in SeeSaw's usage).
        strength:  Overall sharpening gain multiplier.
        power:     Exponent that controls the curve shape; higher = more
                   aggressive recovery of fine detail relative to coarse edges.
        zp:        Zero-point in sample units; the pivot of the power curve.
        lodmp:     Low-end damping; softens the curve for differences smaller
                   than this value to avoid noise amplification.
        hidmp:     High-end damping; rolls off gain for large differences.
                   Set to 0 to disable.
        rgmode:    RemoveGrain mode used to build the blurred reference.
                   Negative values select a MinBlur radius instead.
        nsmode:    1 = original nonlinear sharpening (default).
                   0 = corrected variant.
        diff:      When True, return the sharpening difference clip rather
                   than the sharpened clip, so the caller can apply its own
                   limiting strategy.
    """
    fmt = clip.format
    if fmt is None:
        raise ValueError("Variable-format clips are not supported.")
    power = max(int(power), 1)

    PWR = str(1.0 / float(power))
    ZRP = str(zp)
    DMP = str(lodmp)
    STR = str(strength)
    HDMP = "1" if hidmp == 0 else f"1 x y - abs {hidmp} / 4 pow +"

    if rgmode > 0:
        blur = _RemoveGrain(clip, rgmode)
    else:
        blur = clip

    if nsmode == 1:
        expr = (
            f"x y = x "
            f"x x y - abs {ZRP} / {PWR} pow {ZRP} * {STR} * "
            f"x y - 2 pow x y - 2 pow {DMP} + / * "
            f"x y - x y - abs / * {HDMP} / + ?"
        )
    else:
        expr = (
            f"x y = x "
            f"x x y - abs {ZRP} / {PWR} pow {ZRP} * {STR} * "
            f"x y - x y - abs / * {HDMP} / + ?"
        )

    # Force core.std.Expr to avoid backend-specific quirks
    sharpened = core.std.Expr([clip, blur], [expr])

    if diff:
        return core.std.MakeDiff(clip, sharpened)
    return sharpened


# ---------------------------------------------------------------------------
# Temporal / spatial soothing
# ---------------------------------------------------------------------------

def SootheSS(
    sharp: vs.VideoNode,
    orig: vs.VideoNode,
    sootheT: int = 25,
    sootheS: int = 0,
) -> vs.VideoNode:
    """Reduce temporal and spatial instability introduced by sharpening.

    Sharpeners amplify frame-to-frame differences in fine texture, causing
    visible shimmer.  SootheSS counteracts this in two passes:

    Spatial pass (sootheS > 0):
        Compares the sharpening difference against a spatially smoothed version
        of itself.  Where the two agree in sign the difference is kept; where
        they disagree (isolated single-pixel spikes, which are usually noise)
        it is attenuated toward the smoothed value.

    Temporal pass (sootheT != 0):
        Compares the sharpening difference against a temporally averaged version
        (radius-1 TemporalSoften).  Same logic: agreement keeps, disagreement
        blends toward the temporal average.  A negative *sootheT* chains two
        temporal passes for stronger stabilisation.

    The blend weight in both passes is (100 - abs(soothe)) / 100, so 100
    replaces the sharp difference entirely with the smoothed version (maximum
    calming) and 0 leaves it untouched.

    Args:
        sharp:    Sharpened clip (luma only in SeeSaw's usage).
        orig:     Pre-sharpening reference clip used to form the difference.
        sootheT:  Temporal soothing strength, -100..100.
        sootheS:  Spatial soothing strength, 0..100.
    """
    sootheT = max(-100, min(100, sootheT))
    sootheS = max(0,    min(100, sootheS))
    ST   = str(100 - abs(sootheT))
    SSPT = str(100 - abs(sootheS))

    isFLOAT = sharp.format.sample_type == vs.FLOAT
    mid = 0 if isFLOAT else 1 << (sharp.format.bits_per_sample - 1)
    m   = str(mid)

    # Work in difference space: values above mid mean the sharpener lifted the
    # pixel; values below mid mean it pulled it down.
    diff = core.std.MakeDiff(orig, sharp)

    if sootheS > 0:
        # Spatially blur the difference with a 3x3 average (RemoveGrain mode 20)
        # then blend toward it wherever the original and blurred signs agree.
        sdiff = _RemoveGrain(diff, 20)
        expr = (
            f"x {m} - y {m} - * 0 < "           # signs disagree: pull toward neutral
            f"x {m} - 100 / {SSPT} * {m} + "
            f"x {m} - abs y {m} - abs > "         # |orig_diff| > |smooth_diff|: blend
            f"x {SSPT} * y 100 {SSPT} - * + 100 / "
            f"x ? ? "
        )
        diff = _Expr([diff, sdiff], [expr])

    def _temporal_pass(d: vs.VideoNode) -> vs.VideoNode:
        """Single temporal soothing pass using a radius-1 temporal average."""
        tdiff = _TemporalSoften(d, radius=1, threshold=255)
        expr = (
            f"x {m} - y {m} - * 0 < "
            f"x {m} - 100 / {ST} * {m} + "
            f"x {m} - abs y {m} - abs > "
            f"x {ST} * y 100 {ST} - * + 100 / "
            f"x ? ? "
        )
        return _Expr([d, tdiff], [expr])

    if sootheT != 0:
        diff = _temporal_pass(diff)
    if sootheT < 0:
        # A second pass for stronger stabilisation when sootheT is negative.
        diff = _temporal_pass(diff)

    # Convert the soothed difference back to a clip in the original domain.
    return core.std.MakeDiff(orig, diff)


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def SeeSaw(
    clip: vs.VideoNode,
    denoised: vs.VideoNode | None = None,
    NRlimit: int = 2,
    NRlimit2: int | None = None,
    sstr: float = 1.5,
    Slimit: int | None = None,
    Spower: float = 4,
    SdampLo: float | None = None,
    SdampHi: float = 24,
    Szp: float = 18,
    bias: float = 49,
    Smode: int | None = None,
    NSmode: int = 1,
    sootheT: int = 50,
    sootheS: int = 0,
    ssx: float = 1,
    ssy: float | None = None,
    diff: bool = False,
) -> vs.VideoNode:
    """Denoiser-and-Sharpener-are-riding-the-SeeSaw.

    Enhances fine detail by intermixing a noisy source, a denoised copy, and a
    nonlinear sharpener in proportions that depend on local contrast.  Strong
    edges are left largely untouched while weak texture is boosted, with
    temporal soothing preventing the result from shimmering.

    Args:
        clip:      Noisy source clip.  YUV or GRAY, integer or float.
        denoised:  Pre-denoised version of *clip*.  Must match format and
                   dimensions exactly.  If omitted, a simple spatial median
                   clamp is used — adequate but not ideal; supplying your own
                   denoised clip gives better results.
        NRlimit:   Hard ceiling (8-bit units) on how much any pixel may move
                   due to denoising.  Prevents the denoiser from erasing strong
                   detail even when `denoised` is aggressive.
        NRlimit2:  Looser intermediate ceiling used when forming the tame blend.
                   Defaults to NRlimit + 1.
        sstr:      Sharpening strength multiplier fed to Sharpen2.  The default
                   of 1.5 is conservative; values above 2.5 risk haloing.
        Slimit:    Final clamp on the sharpening difference (8-bit units).
                   Positive: hard clip — no pixel moves more than this.
                   Negative: soft power-curve reduction — less aliasing risk but
                   also less peak sharpening.  Defaults to NRlimit + 2.
        Spower:    Exponent for Sharpen2's power curve.  Higher values give a
                   steeper curve: finer detail gets more gain relative to coarse
                   edges.  Must be >= 1.
        SdampLo:   Damping applied near the zero-point of Sharpen2's curve to
                   avoid amplifying residual noise.  Scaled internally by sstr
                   and the supersampling factor.  Defaults to Spower + 1.
        SdampHi:   High-end roll-off in Sharpen2.  Attenuates gain for large
                   differences (strong edges).  0 disables it.  Try 15-30.
        Szp:       Zero-point of Sharpen2's power curve in 8-bit units.
                   Below Szp the curve overdrive-sharpens; above it sharpening
                   is progressively reduced.  Scaled internally by sstr and
                   the supersampling factor.
        bias:      0-100.  Controls how much the tame blend leans toward the
                   source (> 50, detail-biased) or toward the denoised clip
                   (< 50, calm-biased).  Default 49 is very slightly calm-biased,
                   matching the AviSynth original.
        Smode:     RemoveGrain mode used inside Sharpen2 when building the blur
                   reference.  Negative values select a MinBlur radius.  Chosen
                   automatically from the supersampling factor if omitted.
        NSmode:    Selects Sharpen2's internal expression:
                   1 = original nonlinear form (default).
                   0 = simplified corrected form.
        sootheT:   Temporal soothing strength, -100..100.  Higher values blend
                   the sharpening difference more strongly toward the temporal
                   average, reducing shimmer.  Negative chains two passes.
        sootheS:   Spatial soothing strength, 0..100.  Blends the sharpening
                   difference toward a spatially smoothed version, reducing
                   isolated pixel spikes.
        ssx, ssy:  Supersampling factors.  Sharpening is performed at
                   ssx*ssy times the input resolution then downscaled, which
                   reduces aliasing on diagonal edges.  Values around 1.25 are
                   a useful compromise; SeeSaw does not urgently require
                   supersampling.
        diff:      When True and Smode > 0, limit the sharpening difference
                   rather than the sharpened clip.  Safer against aliasing but
                   produces a milder result.
    """
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("SeeSaw: clip must be a VideoNode.")

    # --- Fill in defaults that depend on other parameters -------------------

    if NRlimit2 is None:
        NRlimit2 = NRlimit + 1
    if Slimit is None:
        Slimit = NRlimit + 2
    if SdampLo is None:
        SdampLo = float(Spower + 1)
    if ssy is None:
        ssy = ssx
    if Smode is None:
        # Higher supersampling means more aliasing protection is needed, so
        # select a RemoveGrain mode that more aggressively smooths the reference.
        if ssx <= 1.25:
            Smode = 11
        elif ssx <= 1.6:
            Smode = 20
        else:
            Smode = 19

    # --- Format constants ---------------------------------------------------

    color   = clip.format.color_family
    isFLOAT = clip.format.sample_type == vs.FLOAT
    bd      = clip.format.bits_per_sample
    i       = 0.00392 if isFLOAT else float(1 << (bd - 8))   # 1 lsb in normalised units
    mid     = 0       if isFLOAT else 1 << (bd - 1)           # neutral difference value
    peak    = 1.0     if isFLOAT else float((1 << bd) - 1)    # maximum sample value

    Spower  = max(int(Spower), 1)
    SdampHi = max(SdampHi, 0.0)
    ssx     = max(ssx, 1.0)
    ssy     = max(ssy, 1.0)
    ow, oh  = clip.width, clip.height
    xss     = _m4(ow * ssx)
    yss     = _m4(oh * ssy)

    # --- Scale Szp and SdampLo for strength and supersampling ---------------
    #
    # At higher sharpening strengths or with supersampling, effective difference
    # magnitudes grow.  Szp and SdampLo are shifted upward in proportion so the
    # power curve's shape stays perceptually consistent across settings.
    Szp     = Szp     / (sstr ** 0.25) / (((ssx + ssy) / 2.0) ** 0.5)
    SdampLo = max(SdampLo / (sstr ** 0.25) / (((ssx + ssy) / 2.0) ** 0.5), 0.0)

    # --- Threshold constants (8-bit units scaled to working bit depth) ------

    # bias_pct stays in 0-100 for the NRLL formula; bias_f is the 0-1 form
    # used directly in Expr arithmetic.
    bias_pct = float(max(min(bias, 100.0), 1.0))
    bias_f   = bias_pct / 100.0

    NRL  = _scale(NRlimit,  peak)
    NRL2 = _scale(NRlimit2, peak)

    # NRLL is the outer threshold in the tame blend expression.
    # When |src - den| > NRLL the tame clip is clamped to +-NRL2 regardless of
    # bias; within NRLL the blend is proportional.  The (100/bias - 1) factor
    # widens the outer window for calm bias and narrows it for detail bias.
    NRLL = _scale(NRlimit2 * (100.0 / bias_pct) - 1.0, peak)

    SLIM = _scale(float(Slimit), peak) if Slimit >= 0 else 1.0 / abs(Slimit)

    # --- Build the denoised clip if not supplied ----------------------------

    if denoised is None:
        # Simple spatial clamp: accept the local median only when it differs
        # from the source by less than NRL.  Noisier than a temporal denoiser
        # but requires no extra clip.
        dnexpr = f"x {NRL} + y < x {NRL} + x {NRL} - y > x {NRL} - y ? ?"
        if color == vs.YUV:
            denoised = _Expr(
                [clip, core.std.Median(clip, [0])], [dnexpr, ""]
            )
        else:
            denoised = _Expr([clip, core.std.Median(clip)], [dnexpr])
    else:
        if not isinstance(denoised, vs.VideoNode):
            raise TypeError("SeeSaw: denoised must be a VideoNode.")
        if denoised.format.id != clip.format.id:
            raise TypeError("SeeSaw: denoised must have the same format as clip.")
        if denoised.width != clip.width or denoised.height != clip.height:
            raise TypeError("SeeSaw: denoised must be the same size as clip.")

    # --- Isolate luma for all sharpening work -------------------------------
    #
    # Sharpening is applied to luma only.  Chroma is carried through unchanged
    # from the original clip and recombined at the very end.

    if color == vs.YUV:
        tmp  = core.std.ShufflePlanes(clip,     [0], vs.GRAY)
        tmp2 = core.std.ShufflePlanes(denoised, [0], vs.GRAY) if clip is not denoised else tmp
    else:
        tmp  = clip
        tmp2 = denoised

    # --- Form the "tame" blend ----------------------------------------------
    #
    # tame is a pixel-wise weighted blend of source and denoised, with hard
    # clamps at the extremes:
    #   - If |src - den| > NRLL: clamp the change to +-NRL2 (prevent large
    #     excursions caused by an aggressive denoiser).
    #   - Otherwise: mix src*bias_f + den*(1-bias_f).
    # The result lies between source and denoised, preserving detail the
    # denoiser might have suppressed while still benefiting from noise reduction.

    tameexpr = (
        f"x {NRLL} + y < x {NRL2} + "
        f"x {NRLL} - y > x {NRL2} - "
        f"x {bias_f} * y {1.0 - bias_f} * + ? ?"
    )
    tame = _Expr([tmp, tmp2], [tameexpr])

    # --- Sharpening ---------------------------------------------------------
    #
    # When supersampling (ssx/ssy > 1): tame is upscaled, sharpened, then
    # downscaled.  This allows the sharpener to resolve sub-pixel detail and
    # reduces aliasing on diagonal edges.
    #
    # A Repair() call clamps the supersampled result against a 1:1 pass (head),
    # preventing supersampling from introducing ringing that the 1:1 pass would
    # not have produced.

    if Smode > 0:
        # 1:1 reference pass used as the Repair target when supersampling.
        head = Sharpen2(tame, sstr, Spower, Szp, SdampLo, SdampHi, 4, NSmode, diff)
        # Mask strong edges so the sharpener does not overshoot them; the
        # Sobel+expand+blur mask covers the full edge neighbourhood.
        head = core.std.MaskedMerge(
            head, tame,
            tame.std.Sobel().std.Maximum().std.Convolution(matrix=[1] * 9)
        )

    if ssx == 1.0 and ssy == 1.0:
        if Smode <= 0:
            sharp = Sharpen2(tame, sstr, Spower, Szp, SdampLo, SdampHi, Smode, NSmode, diff)
        else:
            sharp = _Repair(
                Sharpen2(tame, sstr, Spower, Szp, SdampLo, SdampHi, Smode, NSmode, diff),
                head, 1
            )
    else:
        if Smode <= 0:
            sharp = (
                Sharpen2(tame.resize.Spline36(xss, yss), sstr, Spower, Szp, SdampLo, SdampHi, Smode, NSmode, diff)
                .resize.Spline36(ow, oh)
            )
        else:
            sharp = _Repair(
                Sharpen2(tame.resize.Spline36(xss, yss), sstr, Spower, Szp, SdampLo, SdampHi, Smode, NSmode, diff),
                head.resize.Spline36(xss, yss), 1
            ).resize.Spline36(ow, oh)

    if diff and Smode > 0:
        # Sharpen2 returned a difference clip; merge it back onto tame now.
        sharp = core.std.MergeDiff(tame, sharp)

    # --- Temporal/spatial stabilisation ------------------------------------

    soothed   = SootheSS(sharp, tame, sootheT, sootheS)
    sharpdiff = core.std.MakeDiff(tame, soothed)

    # --- Re-apply denoising with NRlimit clamp ------------------------------
    #
    # Apply the original denoising difference back onto the source, clamping
    # each pixel's change to +-NRL.  This ensures the denoiser cannot move any
    # single pixel by more than NRlimit steps regardless of how aggressive the
    # supplied denoised clip is.

    if NRlimit == 0 or clip == denoised:
        # No denoising requested, or denoised is identical to source.
        calm = tmp
    else:
        NRdiff = core.std.MakeDiff(tmp, tmp2)
        if isFLOAT:
            expr = f"y {NRL} > x {NRL} - y -{NRL} < x {NRL} + x y - ? ?"
        else:
            expr = f"y {mid} {NRL} + > x {NRL} - y {mid} {NRL} - < x {NRL} + x y {mid} - - ? ?"
        calm = _Expr([tmp, NRdiff], [expr])

    # --- Apply sharpening difference with Slimit clamp ----------------------
    #
    # The soothed sharpening difference is added to the noise-limited calm clip,
    # with a final hard or soft clamp.
    #
    # Positive Slimit: hard clip — no pixel changes by more than SLIM.
    # Negative Slimit: soft power curve — diff_out = sign(d) * |d|^(1/|Slimit|),
    #   yielding progressively less sharpening for larger differences without
    #   a sharp cutoff, at the cost of slightly lower peak sharpening.

    if Slimit >= 0:
        if isFLOAT:
            limitexpr = f"y {SLIM} > x {SLIM} - y -{SLIM} < x {SLIM} + x y - ? ?"
        else:
            limitexpr = f"y {mid} {SLIM} + > x {SLIM} - y {mid} {SLIM} - < x {SLIM} + x y {mid} - - ? ?"
        limited = _Expr([calm, sharpdiff], [limitexpr])
    else:
        if isFLOAT:
            limitexpr = f"y 0 = x dup y abs {i} / {SLIM} pow {i} * y 0 > 1 -1 ? * - ?"
        else:
            limitexpr = f"y {mid} = x dup y {mid} - abs {i} / {SLIM} pow {i} * y {mid} > 1 -1 ? * - ?"
        limited = _Expr([calm, sharpdiff], [limitexpr])

    # --- Recombine chroma and return ----------------------------------------

    if color == vs.YUV:
        return core.std.ShufflePlanes([limited, clip], [0, 1, 2], color)
    return limited