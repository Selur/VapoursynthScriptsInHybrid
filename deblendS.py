"""
Stateful blend/telecine frame restorer for VapourSynth
======================================================================
Port of AviSynth/havsfunc sRestore, all output modes.

Handles mixed progressive-blend and telecined sources.  Uses the same
detection signals and decision logic as srestore (bclp/dclp blend
metrics, 7-frame rolling diff/motion/blend history, cadence offset
tracking).

Output modes
------------
omode 6 (the default) is the only mode that changes the frame RATE:
it resolves the cadence and decimates src_fps down to true_fps.  All
other modes leave the frame count untouched and only replace or
reorder frames.

  1 - 5   Cadence-neutral deblending.  frfac is forced to 1, so no
          decimation happens and the motion metric is not measured
          (srestore uses a constant 1 there).  The modes differ in
          which neighbour a detected blend is replaced with:
            1  always take the previous frame
            2  take the next frame
            3  take whichever of the two neighbours has the smaller
               diff to the blend
            4  decide by the blend/clear ratio of the neighbours
            5  full double-blend removal -- can shift by two frames
               in either direction
  6       Default.  Cadence detection plus decimation to true_fps.
  pp0 -   Blend REMOVAL instead of blend replacement.  A frame
  pp3     detected as a blend is not swapped for a neighbour but
          rebuilt from its neighbours ("unblended").  Also cadence-
          neutral.  The four variants differ in how the two unblend
          candidates are combined:
            pp0  fixed 4-tap arithmetic combination, no mask
            pp1  masked merge driven by the diff mask
            pp2  masked merge driven by the blend mask
            pp3  masked merge driven by the combined mask, plus a
                 chroma blur
          Building the reconstruction clip costs noticeably more than
          the numeric modes, which only ever pick an existing frame.

Design
------
The detection state is a Markov chain over frames: the decision for
frame n depends on the whole history before it.  It is therefore
advanced strictly in order and cached, so a frame is decided exactly
once no matter how often VapourSynth asks for it.

The work happens on demand, on whichever thread requests the frame --
there is no background worker.  See Engine for why that matters: a
separate thread deadlocks against the VapourSynth thread pool.

NOTE: There is NO scene-cut based state reset during normal sequential
playback.  AviSynth's srestore has no such mechanism, and resetting on
scene cuts (especially after long black/static sections) causes the
Markov state to diverge badly from the AviSynth reference.

NOTE: Seeking backwards is free (decisions are cached), seeking far
forward is not -- every intervening frame's detection statistics still
have to be computed, because the state depends on them.

Usage
-----
deblendS(clip, src_fps=60000/1001, true_fps=24000/1001, omode=6)

deblendS6(clip, ...) is kept as a thin wrapper that calls deblendS
with omode=6, so older scripts keep working unchanged.

Parameters
----------
clip        VideoNode   Input clip. Must be YUV.
src_fps     float       Source fps (default: read from clip).
true_fps    float       Target fps (default: nearest lower common rate).
                        ONLY has an effect with omode=6; every other
                        mode is cadence-neutral and keeps the source
                        rate.
omode       int|str     Output mode, see "Output modes" above.
                        1-6 or "pp0"-"pp3".  Default 6.
optical_flow          bool   When True, detected blend frames are reconstructed
                             using motion-compensated interpolation instead of
                             the default simple 50/50 neighbour merge.
                             Default: False.
                             WHEN TO USE: live-action or CG with smooth motion.
                             WHEN NOT TO USE: anime/cel animation — vectors are
                             unreliable on hard-cut discrete motion; the simple
                             merge is better.
optical_flow_engine   str    Which engine to use when optical_flow=True.
                             Both MVTools values fall back through the same
                             three plugins, they only differ in priority:
                               "mvtools" (default)
                                 core.mvu  -> core.mvsf -> core.mv
                               "mvtools-sf"
                                 core.mvsf -> core.mvu  -> core.mv
                             i.e. "mvtools-sf" takes MVTools-sf when it is
                             installed and otherwise behaves exactly like
                             "mvtools".
                               core.mvu  = MVUtensils
                               core.mvsf = MVTools-sf (32 bit float)
                               core.mv   = MVTools2 (8 bit)
                             Both take of_pel and of_blksize.
                             "svp" — uses SVP (core.svp1 + core.svp2) with
                               adaptive block sizes and multi-level refinement.
                               Generally better quality on live-action but
                               requires the SVP plugin to be installed.
                               of_pel / of_blksize are ignored for SVP.
of_pel         int      Sub-pixel precision for optical flow vector search.
                        1=pixel, 2=half-pixel (default), 4=quarter-pixel.
                        Higher values are slower but more accurate.
of_blksize     int      Block size for MVTools vector search (default 16).
                        Smaller blocks (8) capture finer motion at the cost
                        of speed.  Larger blocks (32) are faster.
show_debug     bool     Attach DeblendTarget/DeblendUseMec/DeblendIsBlend/
                        DeblendThr frame props AND burn a text overlay onto
                        the video showing those values.  Useful for tuning.

dclip       VideoNode   Alternate clip used purely for blend detection.
                        Output frames always come from 'clip'.  Pass a
                        denoised or pre-filtered version here when the
                        source has noise, blocking, or ringing artefacts
                        that would otherwise interfere with the diff/motion
                        metrics.  Safe to omit for clean sources.
                        If both dclip and denoise are given, denoise is
                        applied on top of the supplied dclip.
denoise     str|None    Purely spatial pre-denoise applied to dclip before
                        building the detection thumbnails.  Only affects
                        detection, never the output frames.
                        Temporal denoisers must NOT be used here because
                        they smooth exactly the inter-frame diff signal
                        that blend detection relies on.
                        None (default) — no denoising.
                        "RemoveGrain" — two-pass spatial filter:
                            pass 1: RG mode 2  (clip to neighbourhood
                                    min/max, kills blocking/spike pixels)
                            pass 2: RG mode 12 (3x3 weighted blur,
                                    smooths fine grain)
                            Fast, no extra plugins required.
                            Good for: compressed sources with blocking,
                            moderate grain, ringing artefacts.
                        "NLMeans" — Non-local means spatial denoise
                            (d=0, purely single-frame, no temporal radius).
                            Requires the nlm_cuda or nlm_ispc plugin.
                            h=7 on luma, chroma untouched (detection is
                            luma-only after the thumbnail downscale).
                            Good for: heavy film grain, analog noise where
                            RemoveGrain is not strong enough.
thresh      int         Detection threshold, as srestore's 'thresh'.
                        Default 16 (the AviSynth default); internally used
                        as abs(thresh) + 0.01, so the sign is irrelevant.

                        LOWER  -> more sensitive, more frames treated as
                                  blends/motion.
                        HIGHER -> more conservative.

                        Worth lowering for animation.  The threshold is
                        compared against three metrics, and two of them
                        (blend/clear from bclp, diff from dclp) are built
                        from PlaneStatsMin/Max, so a small moving region
                        still registers.  The motion metric is not: it is
                        PlaneStatsDiff, a MEAN over the whole frame.  On
                        cel animation most of the frame is a static flat
                        area, which drags that mean down, so gates like
                        'mb > thr' rarely fire even when the blending is
                        obvious.  Live action with full-frame motion does
                        not have this bias.

                        Never derive this from the clip automatically:
                        black or static frames at the start corrupt any
                        such estimate and produce a threshold far below
                        16, which flags nearly everything as a blend.
bsize       int         Edge length of the square detection thumbnails
                        that bclp and dclp are scaled down to.  Default
                        32, matching AviSynth srestore.

                        This is a CALIBRATION value, not a quality dial.
                        It scales b_v, c_v and d_v (which are min/max
                        over the thumbnail) but NOT the motion metric
                        m_v, which is measured on the intermediate clip.
                        Raising it therefore shifts the balance between
                        the metrics, and since the decisions mix absolute
                        comparisons against thresh with ratio terms
                        between the metrics, the net effect is NOT
                        monotonic.  Measured across several clips, blend
                        detection rises with bsize on some segments,
                        peaks at 48 on others, and falls on others.
                        Higher is not "more sensitive".

                        There IS a structural argument for touching it:
                        the intermediate detection clip scales with the
                        source resolution while this thumbnail does not,
                        so a 1080p source averages roughly 5x more pixels
                        into each thumbnail pixel than NTSC DVD does, and
                        2160p about 19x, while thresh stays absolute.  At
                        the metric level, 1080p with bsize=48 does line up
                        with 480p at bsize=32.

                        That does NOT translate into a usable rule of
                        thumb.  Measured over 28 clips of mixed blended
                        material, going 32 -> 48 -> 64 lowered the number
                        of detected blends on 11 clips, raised it on 4,
                        and was non-monotonic on 13 -- including the one
                        1080p clip, which fell (32 -> 22 -> 19).  So do
                        not raise this "because the source is HD".

                        Treat it as an experimental knob: 32 is the
                        srestore reference and the safe default, and 48
                        or 64 are worth A/B-ing on your own material when
                        32 gives an unsatisfying result.  There is no
                        substitute for looking at the output.
nlmeans_h   float       Strength (h) for NLMeans denoising. Default 7.0.
                        Increase for heavier grain (10-16), decrease for
                        light noise (3-5).  Only used when denoise="NLMeans".
mode        int         Controls duplicate/merge detection and mec usage.
                        Default is 2.  Options:

                         0  Basic frame selection only.  No duplicate or
                            merge detection, no mec clip.
                         1  Identical to 0 -- duplicate detection needs
                            abs(mode) >= 2, so it is still disabled here.
                         2  Default.  Mec clip built, duplicate detection
                            active, merge detection off.
                         3  Mec clip built, merge detection active, but
                            duplicate detection disabled.
                         4  Most aggressive.  Mec clip + duplicate
                            detection + merge detection.  Try this if
                            mode 2 misses blends.
                        A NEGATIVE mode enables chroma-aware detection:
                        the U and V planes are stacked next to Y in the
                        detection clip, so chroma contributes to the
                        blend metrics.  Everything else behaves exactly
                        like the matching positive mode.  Useful on
                        sources where the blend is far more visible in
                        chroma than in luma.
                        -1  chroma-aware variant of  1
                        -2  chroma-aware variant of  2
                        -3  chroma-aware variant of  3
                        -4  chroma-aware variant of  4

                        Summary of what each flag enables:
                          abs(mode) >= 2  ->  mec clip is constructed
                          abs(mode) >= 2 and abs(mode) != 3
                                          ->  duplicate detection active
                                              (mode 0 and 1 are identical:
                                               plain frame selection only)
                          abs(mode) >  2  ->  merge (mec) detection active
                          mode < 0        ->  chroma included in detection
"""

from __future__ import annotations

import math
import threading
from dataclasses import dataclass
from typing import Optional, Union

from helpers import GetPlane, cround, scale_value

import vapoursynth as vs

core = vs.core

# Default detection threshold, matching AviSynth srestore's thresh=16.
# Overridable per call via the 'thresh' parameter, exactly as in srestore.
#
# It must NEVER be auto-estimated from the clip: black or static frames at
# the start corrupt any bootstrap measurement and yield thr << 16, which
# then flags almost everything as a blend.  A value the caller supplies is
# a different thing entirely and carries none of that risk.
_THR = 16.01

_MAGIC_OFFSET = 1.0 / 64.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _nearest_lower_fps(src: float) -> float:
    candidates = [24000/1001, 24.0, 25.0, 30000/1001, 30.0,
                  48000/1001, 48.0, 50.0, 60000/1001, 60.0]
    below = [c for c in candidates if c < src - 0.01]
    return max(below) if below else src

# ---------------------------------------------------------------------------
# Detection clip construction  (identical to srestore)
# ---------------------------------------------------------------------------

def _build_detection_clips(
    dclip: vs.VideoNode,
    bsize: int   = 32,
    srad:  float = 12.0,
    mode:  int   = 2,
    bom:   bool  = False,
) -> tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode]:
    """
    Returns (bclp, dclp, det).

    det -- detection clip with Trim(first=2) applied.
           Used for bclp / dclp AND for dc_motion, matching AviSynth
           where det = det.pointresize(...).trim(2,0) and all further
           operations use that trimmed det.

           For mode >= 0 this is the luma plane alone.  For mode < 0 it
           is the AviSynth chroma-aware layout: U and V side by side on
           top, Y underneath, so chroma contributes to the blend metrics.

    bom -- True for the pp0-pp3 modes.  srestore builds bclp from a
           different pair of clips there, and halves the contrast of
           the detection clip first.  det is returned unscaled because
           the motion metric it feeds is not measured in those modes.
    """
    if dclip.format.id != vs.YUV420P8:
        dclip = dclip.resize.Bicubic(format=vs.YUV420P8)

    new_w = int(dclip.width  / 2 / srad + 4) * 4
    new_h = int(dclip.height / 2 / srad + 4) * 4
    small = dclip.resize.Point(new_w, new_h).std.Trim(first=2)

    if mode < 0:
        # AviSynth srestore:
        #   det = stackvertical(stackhorizontal(det.utoy(), det.vtoy()), det)
        # The clip is YUV420P8 here, so U and V are new_w/2 x new_h/2 --
        # side by side they are exactly new_w wide and stack cleanly on
        # top of the new_w x new_h luma.  new_w/new_h are multiples of 4,
        # so both halves are integral.
        det = core.std.StackVertical([
            core.std.StackHorizontal([GetPlane(small, 1), GetPlane(small, 2)]),
            GetPlane(small, 0),
        ])
    else:
        det = GetPlane(small, 0)

    EXPR =  core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr

    # AviSynth srestore: bom ? det.mt_lut("x 2 / 64 +") : det
    # Halves the contrast before the differences are taken, because the
    # pp metric below squares them and would otherwise clip.
    src_det = EXPR(det, expr=['x 0.5 * 64 +']) if bom else det

    diff = core.std.MakeDiff(src_det, src_det.std.Trim(first=1))

    if not bom:
        expr_b = ('x 128 - y 128 - * 0 > '
                  'x 128 - abs y 128 - abs < '
                  'x 128 - 128 x - * y 128 - 128 y - * ? '
                  'x y + 256 - dup * ? 0.25 * 128 +')
        bclp = EXPR([diff, diff.std.Trim(first=1)], expr=[expr_b])
    else:
        # srestore expr2, fed from diff(n+1) and diff(n) + diff(n+2).
        expr_b = 'x y - dup * 3 * x y + 256 - dup * - 128 +'
        bclp = EXPR([diff.std.Trim(first=1),
                     core.std.MergeDiff(diff, diff.std.Trim(first=2))], expr=[expr_b])
    bclp = bclp.resize.Bilinear(bsize, bsize)

    dclp = (diff.std.Trim(first=1)
               .std.Lut(function=lambda x: max(cround(abs(x - 128) ** 1.1 - 1), 0))
               .resize.Bilinear(bsize, bsize))

    return bclp, dclp, det


# ---------------------------------------------------------------------------
# Raw per-frame stats
# ---------------------------------------------------------------------------

@dataclass
class RawStats:
    b_min:   float
    b_max:   float
    d_max:   float
    dc_diff: float


# ---------------------------------------------------------------------------
# Decision state
# ---------------------------------------------------------------------------

def _initial_state() -> dict:
    """
    All history slots start at _MAGIC_OFFSET.
    On frame 0 jmp=False so every slot is immediately overwritten with
    the real frame-0 values -- matching AviSynth where the globals are
    first written on frame 0 via the jmp=false branch.
    """
    v = _MAGIC_OFFSET
    return dict(
        lfr=-100, offs=0.0, ldet=-100, lpos=0,
        d32=v, d21=v, d10=v, d01=v, d12=v, d23=v, d34=v,
        m42=v, m31=v, m20=v, m11=v, m02=v, m13=v, m24=v,
        bp2=v, bp1=v, bn0=v, bn1=v, bn2=v, bn3=v,
        cp2=v, cp1=v, cn0=v, cn1=v, cn2=v, cn3=v,
    )


# ---------------------------------------------------------------------------
# _compute_decision -- faithful port of the srestore decision logic
# ---------------------------------------------------------------------------

def _compute_decision(
    n:     int,
    s:     RawStats,
    state: dict,
    frfac: float,
    numr:  int,
    denm:  int,
    thr:   float,
    mode:  int,
    omode: Union[int, str] = 6,
    bom:   bool            = False,
) -> tuple[int, bool, bool, bool]:
    """
    Mutates state in-place.

    Returns (target_frame_index, use_mec, is_blend, use_pp).

    use_pp is srestore's dup==4 case: the frame cannot be resolved by
    picking a neighbour and has to be rebuilt instead.  It only ever
    becomes True for the pp0-pp3 modes; target is then the frame's own
    index, because the reconstruction clip is aligned with the source.
    """

    jmp  = (state['lfr'] + 1 == n)
    cfo  = ((n % denm) * numr * 2 + denm + numr) % (2 * denm) - denm
    bfo  = (-numr < cfo <= numr)
    state['lfr'] = n

    if bfo:
        if state['offs'] <= -4 * numr:
            state['offs'] += 2 * denm
        elif state['offs'] >= 4 * numr:
            state['offs'] -= 2 * denm

    pos = (0 if frfac == 1
           else -cround((cfo + state['offs']) / (2 * numr)) if bfo
           else state['lpos'])
    cof = cfo + state['offs'] + 2 * numr * pos

    state['ldet'] = -1 if n + pos == state['ldet'] else n + pos

    ## diff value shifting ##
    d_v = s.d_max + _MAGIC_OFFSET
    if jmp:
        d43          = state['d32']
        state['d32'] = state['d21']
        state['d21'] = state['d10']
        state['d10'] = state['d01']
        state['d01'] = state['d12']
        state['d12'] = state['d23']
        state['d23'] = state['d34']
    else:
        d43 = state['d32'] = state['d21'] = state['d10'] = \
              state['d01'] = state['d12'] = state['d23'] = d_v
    state['d34'] = d_v

    ## motion value shifting ##
    # dc_diff already scaled to [0,255] in _fetch (PlaneStatsDiff * 255),
    # matching AviSynth's lumadifference() range.
    #
    # AviSynth srestore:
    #   m_v = !bom && abs(omode)>5 ? lumadifference(...)+0.015625 : 1
    # Every mode below 6, and every pp mode, runs on a constant motion
    # metric.  Engine skips the frame fetch entirely in that case, so
    # this is a second line of defence, not the only one.
    m_v = s.dc_diff if (not bom and abs(omode) > 5) else 1.0
    if jmp:
        m53          = state['m42']
        state['m42'] = state['m31']
        state['m31'] = state['m20']
        state['m20'] = state['m11']
        state['m11'] = state['m02']
        state['m02'] = state['m13']
        state['m13'] = state['m24']
    else:
        m53 = state['m42'] = state['m31'] = state['m20'] = \
              state['m11'] = state['m02'] = state['m13'] = m_v
    state['m24'] = m_v

    ## get blend and clear values ##
    b_v = 128.0 - s.b_min   # high when bclp minimum is low
    if b_v < 1.0:
        b_v = 0.125
    c_v = s.b_max - 128.0   # high when bclp maximum is high
    if c_v < 1.0:
        c_v = 0.125

    ## blend value shifting ##
    if jmp:
        bp3          = state['bp2']
        state['bp2'] = state['bp1']
        state['bp1'] = state['bn0']
        state['bn0'] = state['bn1']
        state['bn1'] = state['bn2']
        state['bn2'] = state['bn3']
    else:
        bp3 = state['bp2'] = state['bp1'] = state['bn0'] = \
              state['bn1'] = state['bn2'] = (b_v - c_v if bom else b_v)
    state['bn3'] = b_v - c_v if bom else b_v

    ## clear value shifting ##
    if jmp:
        cp3          = state['cp2']
        state['cp2'] = state['cp1']
        state['cp1'] = state['cn0']
        state['cn0'] = state['cn1']
        state['cn1'] = state['cn2']
        state['cn2'] = state['cn3']
    else:
        cp3 = state['cp2'] = state['cp1'] = state['cn0'] = \
              state['cn1'] = state['cn2'] = c_v
    state['cn3'] = c_v

    ## used detection values ##
    i = pos + 2

    bb  = [bp3,          state['bp2'], state['bp1'], state['bn0'], state['bn1']][i]
    bc  = [state['bp2'], state['bp1'], state['bn0'], state['bn1'], state['bn2']][i]
    bn  = [state['bp1'], state['bn0'], state['bn1'], state['bn2'], state['bn3']][i]
    cb  = [cp3,          state['cp2'], state['cp1'], state['cn0'], state['cn1']][i]
    cc  = [state['cp2'], state['cp1'], state['cn0'], state['cn1'], state['cn2']][i]
    cn  = [state['cp1'], state['cn0'], state['cn1'], state['cn2'], state['cn3']][i]
    dbb = [d43,          state['d32'], state['d21'], state['d10'], state['d01']][i]
    dbc = [state['d32'], state['d21'], state['d10'], state['d01'], state['d12']][i]
    dcn = [state['d21'], state['d10'], state['d01'], state['d12'], state['d23']][i]
    dnn = [state['d10'], state['d01'], state['d12'], state['d23'], state['d34']][i]
    dn2 = [state['d01'], state['d12'], state['d23'], state['d34'], state['d34']][i]
    mb1 = [m53,          state['m42'], state['m31'], state['m20'], state['m11']][i]
    mb  = [state['m42'], state['m31'], state['m20'], state['m11'], state['m02']][i]
    mc  = [state['m31'], state['m20'], state['m11'], state['m02'], state['m13']][i]
    mn  = [state['m20'], state['m11'], state['m02'], state['m13'], state['m24']][i]
    mn1 = [state['m11'], state['m02'], state['m13'], state['m24'], 0.01        ][i]

    ### basic calculation ###
    bbool = (0.8 * bc * cb > bb * cc and
             0.8 * bc * cn > bn * cc and
             bc * bc > cc)

    blend = (bbool and
             bc * 5 > cc and
             dbc + dcn > 1.5 * thr and
             (dbb < 7 * dbc or dbb < 8 * dcn) and
             (dnn < 8 * dcn or dnn < 7 * dbc) and
             (mb < mb1 and mb < mc or
              mn < mn1 and mn < mc or
              (dbb + dnn) * 4 < dbc + dcn or
              ((bb * cc * 5 < bc * cb or mb > thr) and
               (bn * cc * 5 < bc * cn or mn > thr) and
               bc > thr)))

    clear = (dbb + dbc > thr and
             dcn + dnn > thr and
             (bc < 2 * bb or bc < 2 * bn) and
             (dbb + dnn) * 2 > dbc + dcn and
             (mc < 0.96 * mb and mc < 0.96 * mn and
              (bb * 2 > cb or bn * 2 > cn) and
              cc > cb and cc > cn or
              frfac > 0.45 and frfac < 0.55 and
              0.8 * mc > mb1 and 0.8 * mc > mn1 and
              mb > 0.8 * mn and mn > 0.8 * mb))

    highd = (dcn > 5 * dbc and dcn > 5 * dnn and
             dcn > thr and dbc < thr and dnn < thr)

    lowd  = (dcn * 5 < dbc and dcn * 5 < dnn and
             dbc > thr and dnn > thr and
             dcn < thr and frfac > 0.35 and
             (frfac < 0.51 or dcn * 5 < dbb))

    res = ((state['d32'] < thr and state['d21'] < thr and
            state['d10'] < thr and state['d01'] < thr and
            state['d12'] < thr and state['d23'] < thr and
            state['d34'] < thr and d43 < thr) or
           (dbc * 4 < dbb and dcn * 4 < dbb and dnn * 4 < dbb and dn2 * 4 < dbb) or
           (dcn * 4 < dbc and dnn * 4 < dbc and dn2 * 4 < dbc))

    ### offset calculation ###
    if blend:
        odm = denm
    elif clear:
        odm = 0
    elif highd:
        odm = denm - numr
    elif lowd:
        odm = 2 * denm - numr
    else:
        odm = cof
    odm += cround((cof - odm) / (2 * denm)) * 2 * denm

    if blend:
        odr = denm - numr
    elif clear or highd:
        odr = numr
    elif frfac < 0.5:
        odr = 2 * numr
    else:
        odr = 2 * (denm - numr)
    odr *= 0.9

    if state['ldet'] >= 0:
        if cof > odm + odr:
            if cof - state['offs'] - odm - odr > denm and res:
                cof = odm + 2 * denm - odr
            else:
                cof = odm + odr
        elif cof < odm - odr:
            if state['offs'] > denm and res:
                cof = odm - 2 * denm + odr
            else:
                cof = odm - odr
        elif state['offs'] < -1.15 * denm and res:
            cof += 2 * denm
        elif state['offs'] > 1.25 * denm and res:
            cof -= 2 * denm

    state['offs'] = 0 if frfac == 1 else cof - cfo - 2 * numr * pos
    state['lpos'] = pos

    if frfac == 1:
        opos = 0
    else:
        opos = -cround(
            (cfo + state['offs'] +
             (denm if bfo and state['offs'] <= -4 * numr else 0))
            / (2 * numr)
        )
    pos = min(max(opos, -2), 2)

    ### frame output calculation - resync - dup ###
    i2   = pos + 2
    dbb2 = [d43,          state['d32'], state['d21'], state['d10'], state['d01']][i2]
    dbc2 = [state['d32'], state['d21'], state['d10'], state['d01'], state['d12']][i2]
    dcn2 = [state['d21'], state['d10'], state['d01'], state['d12'], state['d23']][i2]
    dnn2 = [state['d10'], state['d01'], state['d12'], state['d23'], state['d34']][i2]

    ## dup_hq - merge ##
    if opos != pos or abs(mode) < 2 or abs(mode) == 3:
        dup = 0
    elif (dcn2 * 5 < dbc2 and dnn2 * 5 < dbc2 and
          (dcn2 < 1.25 * thr or (bn < bc and pos == state['lpos'])) or
          (dcn2 * dcn2 < dbc2 or dcn2 * 5 < dbc2) and
          bn < bc and pos == state['lpos'] and dnn2 < 0.9 * dbc2 or
          dnn2 * 9 < dbc2 and dcn2 * 3 < dbc2):
        dup = 1
    elif ((dbc2 * dbc2 < dcn2 or dbc2 * 5 < dcn2) and
          bb < bc and pos == state['lpos'] and dbb2 < 0.9 * dcn2 or
          dbb2 * 9 < dcn2 and dbc2 * 3 < dcn2 or
          dbb2 * 5 < dcn2 and dbc2 * 5 < dcn2 and
          (dbc2 < 1.25 * thr or (bb < bc and pos == state['lpos']))):
        dup = -1
    else:
        dup = 0

    # mer: exact AviSynth port -- no abs(mc)>2 guard (that was a prior bug)
    mer = (
        not bom and
        opos == pos and
        dup == 0 and
        abs(mode) > 2 and
        (
            dbc2 * 8 < dcn2 or
            dbc2 * 8 < dbb2 or
            dcn2 * 8 < dbc2 or
            dcn2 * 8 < dnn2 or
            dbc2 * 2 < thr or
            dcn2 * 2 < thr or
            (dnn2 * 9 < dbc2 and dcn2 * 3 < dbc2) or
            (dbb2 * 9 < dcn2 and dbc2 * 3 < dcn2)
        )
    )

    ### deblend - doubleblend removal - postprocessing ###
    # Inert for omode 6 -- every branch below either needs bom or an
    # omode of 1..5, so mode 6 falls through with dup unchanged.  That
    # is why this block could be left out while only mode 6 existed.
    add = (state['bp1'] * state['cn2'] > state['bn2'] * state['cp1'] * (1 + thr * 0.01) and
           state['bn0'] * state['cn2'] > state['bn2'] * state['cn0'] * (1 + thr * 0.01) and
           state['cn2'] * state['bn1'] > state['cn1'] * state['bn2'] * (1 + thr * 0.01))

    if bom:
        if (state['bn0'] > state['bp2'] and state['bn0'] >= state['bp1'] and
                state['bn0'] > state['bn1'] and state['bn0'] > state['bn2'] and
                state['cn0'] < 125):
            if state['d12'] * state['d12'] < state['d10'] or state['d12'] * 9 < state['d10']:
                dup = 1
            elif state['d10'] * state['d10'] < state['d12'] or state['d10'] * 9 < state['d12']:
                dup = 0
            else:
                dup = 4          # neither neighbour is clean -- rebuild the frame
        elif (state['bp1'] > bp3 and state['bp1'] >= state['bp2'] and
              state['bp1'] > state['bn0'] and state['bp1'] > state['bn1']):
            dup = 1
        else:
            dup = 0
    elif dup == 0:
        if 0 < omode < 5:
            if not bbool:
                dup = 0
            elif ((omode == 4 and state['bp1'] * state['cn1'] < state['bn1'] * state['cp1']) or
                  (omode == 3 and state['d10'] < state['d01']) or
                  omode == 1):
                dup = -1
            else:
                dup = 1
        elif omode == 5:
            if (state['bp1'] * state['cp2'] > state['bp2'] * state['cp1'] * (1 + thr * 0.01) and
                    state['bn0'] * state['cp2'] > state['bp2'] * state['cn0'] * (1 + thr * 0.01) and
                    state['cp2'] * state['bn1'] > state['cn1'] * state['bp2'] * (1 + thr * 0.01) and
                    (not add or state['cp2'] * state['bn2'] > state['cn2'] * state['bp2'])):
                dup = -2
            elif add:
                dup = 2
            elif (state['bn0'] * state['cp1'] > state['bp1'] * state['cn0'] and
                  (state['bn0'] * state['cn1'] < state['bn1'] * state['cn0'] or
                   state['cp1'] * state['bn1'] > state['cn1'] * state['bp1'])):
                dup = -1
            elif state['bn0'] * state['cn1'] > state['bn1'] * state['cn0']:
                dup = 1
            else:
                dup = 0
        else:
            dup = 0

    if dup == 4:
        # srestore outputs fin[n] here and ignores opos entirely.
        return n, False, blend, True

    use_mec = mer and dup == 0
    target  = n + opos + dup - (1 if dup == 0 and mer and dbc2 < dcn2 else 0)

    return target, use_mec, blend, False


# ---------------------------------------------------------------------------
# Engine: sequential stats fetch + decision, computed on demand
# ---------------------------------------------------------------------------

class Engine:
    """
    Sequential decision state, advanced by whichever thread needs it.

    There is deliberately NO background worker thread.  An earlier
    version ran the state machine on its own thread while the
    VapourSynth workers waited for it in the FrameEval callback.  That
    is a genuine circular wait: fetching the detection frames needs a
    VapourSynth worker, and every worker may be parked in the callback.
    With core.num_threads=1 -- which Hybrid emits for HAVC-ExModel and
    which the vsThreadCount option allows directly -- it deadlocked
    reproducibly and produced no output at all.

    Computing inline removes the cycle: the thread that needs the
    decision is the thread that fetches it.  A synchronous get_frame
    from inside a filter callback is served on the calling thread, so
    this is safe even with a single worker.

    Frames are decided strictly in order, so _decisions[0.._done) is
    always a filled prefix.  Concurrent callers serialise on _lock; the
    one that gets there first does the catch-up work for all of them.
    """

    def __init__(
        self,
        num_frames:  int,
        bclp_s:      vs.VideoNode,
        dclp_s:      vs.VideoNode,
        dc_motion_s: Optional[vs.VideoNode],
        frfac:       float,
        numr:        int,
        denm:        int,
        mode:        int,
        thr:         float            = _THR,
        omode:       Union[int, str]  = 6,
        bom:         bool             = False,
    ) -> None:
        self._bclp  = bclp_s
        self._dclp  = dclp_s
        self._dcmot = dc_motion_s
        self._frfac = frfac
        self._numr  = numr
        self._denm  = denm
        self._mode  = mode
        self._thr   = thr
        self._omode = omode
        self._bom   = bom

        self._nb_max = bclp_s.num_frames  - 1
        self._nd_max = dclp_s.num_frames  - 1
        self._nm_max = dc_motion_s.num_frames - 1 if dc_motion_s is not None else 0

        # _decisions[0.._done) is a filled prefix; nothing beyond it is
        # valid.  One list entry per frame is the whole footprint -- the
        # previous version allocated a threading.Event per frame, which
        # on a feature-length 59.94p source meant several hundred
        # thousand Event objects (each with its own Condition and Lock)
        # created at script-eval time.
        self._decisions: list[Optional[tuple[int, bool, bool, bool]]] = [None] * num_frames
        self._done  = 0
        self._state = _initial_state()
        self._lock  = threading.Lock()

    @property
    def thr(self) -> float:
        return self._thr

    def get(self, n: int) -> tuple[int, bool, bool, bool]:
        # Fast path: already decided.  _done is only ever raised after
        # the matching _decisions entry has been written, so reading it
        # without the lock cannot observe a half-filled slot.
        if n < self._done:
            return self._decisions[n]

        with self._lock:
            while self._done <= n:
                k = self._done
                s = self._fetch(k)
                self._decisions[k] = _compute_decision(
                    k, s, self._state,
                    self._frfac, self._numr, self._denm, self._thr, self._mode,
                    self._omode, self._bom,
                )
                self._done = k + 1

        return self._decisions[n]

    def _fetch(self, n: int) -> RawStats:
        fb  = self._bclp.get_frame(min(n, self._nb_max))
        fd  = self._dclp.get_frame(min(n, self._nd_max))
        # The motion metric is only measured for omode 6; every other
        # mode runs on a constant 1 (see _compute_decision), so there is
        # no point in pulling the frame -- that is one full detection
        # frame less per output frame.
        if self._dcmot is None:
            dc_diff = 1.0
        else:
            fdc = self._dcmot.get_frame(min(n, self._nm_max))
            # PlaneStatsDiff is [0.0, 1.0]; scale to [0, 255] to match
            # AviSynth's lumadifference() which works in that range.
            dc_diff = float(fdc.props['PlaneStatsDiff']) * 255.0 + _MAGIC_OFFSET
        return RawStats(
            b_min   = float(fb.props['PlaneStatsMin']),
            b_max   = float(fb.props['PlaneStatsMax']),
            d_max   = float(fd.props['PlaneStatsMax']),
            dc_diff = dc_diff,
        )


# ---------------------------------------------------------------------------
# pp0 - pp3 blend reconstruction  (srestore's "bom" branch)
# ---------------------------------------------------------------------------

def _build_pp_clip(source: vs.VideoNode, omode: str) -> vs.VideoNode:
    """
    Build srestore's 'fin' clip: frame n is frame n of the source with
    the blend removed, rebuilt ("unblended") from its neighbours.

    Verbatim port of srestore's bom postprocessing branch.  The inputs
    are Trim()ed, so the result is up to two frames shorter than the
    source -- callers must clamp the index, exactly as srestore does.

    Unlike the numeric modes this actually synthesises picture content,
    which is why it is only built when a pp mode was asked for.
    """
    fmt  = source.format
    bits = fmt.bits_per_sample
    if fmt.sample_type == vs.FLOAT:
        # srestore is integer-only and would compute 1 << 31 here.
        neutral, peak = 0.5, 1.0
    else:
        neutral, peak = 1 << (bits - 1), (1 << bits) - 1

    EXPR = core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr

    sourceDuplicate = source.std.DuplicateFrames(frames=[0])
    sourceTrim1     = source.std.Trim(first=1)
    sourceTrim2     = source.std.Trim(first=2)

    if omode == 'pp0':
        # No mask at all: -0.5*prev + cur + next - 0.5*next2.
        return EXPR([sourceDuplicate, source, sourceTrim1, sourceTrim2],
                    expr=['x -0.5 * y + z + a -0.5 * +'])

    # The two unblend candidates: extrapolated backwards from the
    # previous frame and forwards from the following one.
    unblend1 = EXPR([sourceDuplicate, source],      expr=['x -1 * y 2 * +'])
    unblend2 = EXPR([sourceTrim1,     sourceTrim2], expr=['x 2 * y -1 * +'])

    matrix = [1, 2, 1, 2, 4, 2, 1, 2, 1]

    if omode == 'pp2':
        qmask1 = core.std.MakeDiff(
            unblend1.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1], planes=[0]),
            unblend1, planes=[0])
        qmask2 = core.std.MakeDiff(
            unblend2.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1], planes=[0]),
            unblend2, planes=[0])
        bmask = EXPR([qmask1, qmask2],
                     expr=[f'x {neutral} - dup * dup y {neutral} - dup * + / {peak} *', ''])
        return core.std.MaskedMerge(unblend1, unblend2,
                                    bmask.std.Convolution(matrix=matrix, planes=[0]),
                                    first_plane=True)

    diffm = core.std.MakeDiff(sourceDuplicate, source, planes=[0]).std.Maximum(planes=[0])
    dexpr = ('x 2 * y < x {i} < and 0 y 2 * x < y {i} < and {peak} x x y + / {j} * {k} + ? ?'
             .format(i=scale_value(4, 8, bits), peak=peak,
                     j=scale_value(200, 8, bits), k=scale_value(28, 8, bits)))
    dmask = EXPR([diffm, diffm.std.Trim(first=2)], expr=[dexpr, ''])

    if omode == 'pp1':
        # Chroma of the mask is pinned to neutral, so U/V come from
        # unblend1 unchanged.
        return core.std.MaskedMerge(
            unblend1, unblend2,
            EXPR(dmask.std.Convolution(matrix=matrix, planes=[0]), expr=['', repr(neutral)]))

    if omode == 'pp3':
        qmask1 = core.std.MakeDiff(
            unblend1.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1], planes=[0]),
            unblend1, planes=[0])
        qmask2 = core.std.MakeDiff(
            unblend2.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1], planes=[0]),
            unblend2, planes=[0])
        bmask = EXPR([qmask1, qmask2],
                     expr=[f'x {neutral} - dup * dup y {neutral} - dup * + / {peak} *', ''])
        pmask = EXPR([dmask, bmask],
                     expr=[f'y 0 > y {peak} < and x 0 = x {peak} = or and x y ?', ''])
        fin = core.std.MaskedMerge(unblend1, unblend2,
                                   pmask.std.Convolution(matrix=matrix, planes=[0]),
                                   first_plane=True)
        return fin.std.Convolution(matrix=matrix, planes=[1, 2])

    raise vs.Error(f"deblendS: unexpected value for omode: {omode!r}")


# ---------------------------------------------------------------------------
# dclip denoiser helper
# ---------------------------------------------------------------------------

def _apply_dclip_denoise(
    dclip:     vs.VideoNode,
    denoise:   str,
    nlmeans_h: float = 7.0,
    mode:      int   = 2,
) -> vs.VideoNode:
    """
    Apply a purely spatial denoise to dclip before detection thumbnail
    construction.  Temporal denoisers must never be used here: they smooth
    the inter-frame diff signal that blend detection depends on.

    Parameters
    ----------
    dclip      Source clip (any YUV format).
    denoise    "RemoveGrain" or "NLMeans" (case-insensitive).
    nlmeans_h  NLMeans h strength (default 7.0).
    mode       Detection mode.  For mode >= 0 only Y reaches the
               detection thumbnails, so chroma is left untouched.  For
               mode < 0 chroma is stacked into the detection clip and
               must be denoised too.
    """
    method    = denoise.strip().lower()
    do_chroma = mode < 0

    if method == "removegrain":
        if not hasattr(core, 'zsmooth') and not hasattr(core, 'rgvs'):
            raise vs.Error(
                "deblendS: denoise='RemoveGrain' requires either the "
                "zsmooth or rgvs (RemoveGrainVS) plugin to be installed."
            )
        # Pass 1: mode 2 -- clip each pixel to the min/max of its 8 neighbours.
        #   Kills isolated spike pixels from blocking and ringing without
        #   blurring edges.
        # Pass 2: mode 12 -- 3x3 weighted average (centre weight 4, edges 2,
        #   corners 1 -- a mild Gaussian-like blur).  Smooths fine grain that
        #   pass 1 leaves behind.
        # U/V mode 0 = passthrough (plane copied unchanged).  Note that
        # RemoveGrain mode 1 is NOT a no-op -- it is the strongest of the
        # neighbourhood-clipping modes; 0 is the copy mode.
        # zsmooth is preferred over rgvs: supports higher bit depths natively
        # and is generally faster.
        _rg = core.zsmooth.RemoveGrain if hasattr(core, 'zsmooth') else core.rgvs.RemoveGrain
        _c1 = 2  if do_chroma else 0
        _c2 = 12 if do_chroma else 0
        dclip = _rg(dclip, mode=[2,  _c1, _c1])
        dclip = _rg(dclip, mode=[12, _c2, _c2])

    elif method == "nlmeans":
        # Plugin priority: nlm_ispc > nlm_cuda > knlm.KNLMeansCL
        # nlm_ispc is preferred because it is consistently faster on CPU
        # than knlm and does not require a specific OpenCL device.
        # nlm_cuda is second choice (GPU, fastest when available).
        # KNLMeansCL (knlm) is the fallback — widely installed, slower.
        #
        # d=0  — purely spatial, NO temporal radius.  Critical: d>0 would
        #        pull information from neighbouring frames and corrupt the
        #        inter-frame diff signal that blend detection measures.
        # a=2  — 5×5 search window (2*a+1 on each axis).  Small enough to
        #        be fast on the full-res dclip; the heavy downscale that
        #        follows makes larger windows redundant.
        # s=3  — 7×7 comparison patch (2*s+1).
        # h    — denoising strength on luma, user-tunable via nlmeans_h.
        #
        # Subsampled formats (4:2:0, 4:2:2) require separate Y and UV
        # passes because the plugin cannot handle mixed subsampling in a
        # single 'YUV' call.  For 4:4:4 a single pass is fine.
        # Chroma is only denoised for mode < 0, where it is stacked into
        # the detection clip; for mode >= 0 it never reaches the
        # thumbnails and denoising it would be pure waste.
        fmt        = dclip.format
        subsampled = (fmt.subsampling_w > 0 or fmt.subsampling_h > 0)

        if subsampled:
            passes = ['Y'] + (['UV'] if do_chroma else [])
        else:
            passes = ['YUV' if do_chroma else 'Y']

        # Common kwargs shared across all backends / passes.
        kw = dict(d=0, a=2, s=3, h=nlmeans_h, wmode=0, wref=1.0)

        if hasattr(core, 'nlm_ispc'):
            for ch in passes:
                dclip = dclip.nlm_ispc.NLMeans(**kw, channels=ch)

        elif hasattr(core, 'nlm_cuda'):
            for ch in passes:
                dclip = dclip.nlm_cuda.NLMeans(**kw, channels=ch)

        elif hasattr(core, 'knlm'):
            # KNLMeansCL has no wmode/wref and only supports 4:4:4 or
            # per-plane calls for subsampled input.
            for ch in passes:
                dclip = dclip.knlm.KNLMeansCL(d=0, a=2, s=3, h=nlmeans_h,
                                              channels=ch)

        else:
            raise vs.Error(
                "deblendS: denoise='NLMeans' requires one of the following "
                "plugins to be installed: nlm_ispc, nlm_cuda, or knlm "
                "(KNLMeansCL)."
            )

    else:
        raise vs.Error(
            f"deblendS: unknown denoise method '{denoise}'. "
            "Valid values: 'RemoveGrain', 'NLMeans'."
        )

    return dclip


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def deblendS(
    clip:                 vs.VideoNode,
    src_fps:              Optional[float]        = None,
    true_fps:             Optional[float]        = None,
    omode:                Union[int, str]        = 6,
    show_debug:           bool                   = False,
    dclip:                Optional[vs.VideoNode] = None,
    mode:                 int                    = 2,
    optical_flow:         bool                   = False,
    optical_flow_engine:  str                    = "mvtools",
    of_pel:               int                    = 2,
    of_blksize:           int                    = 16,
    denoise:              Optional[str]           = None,
    nlmeans_h:            float                  = 7.0,
    thresh:               int                    = 16,
    bsize:                int                    = 32,
) -> vs.VideoNode:
    if clip.format is None or clip.format.color_family != vs.YUV:
        raise vs.Error("deblendS: input must be a YUV clip with fixed format")

    if isinstance(omode, str):
        omode = omode.strip().lower()
        if omode not in ('pp0', 'pp1', 'pp2', 'pp3'):
            raise vs.Error("deblendS: 'omode' must be 1-6 or 'pp0'-'pp3'")
        bom = True
    else:
        if omode not in (1, 2, 3, 4, 5, 6):
            raise vs.Error("deblendS: 'omode' must be 1-6 or 'pp0'-'pp3'")
        bom = False

    if src_fps is None:
        src_fps = clip.fps_num / clip.fps_den
    if true_fps is None:
        true_fps = _nearest_lower_fps(src_fps)
    if dclip is None:
        dclip = clip
    elif not isinstance(dclip, vs.VideoNode):
        raise vs.Error("deblendS: 'dclip' is not a clip")
    elif dclip.format is None or dclip.format.color_family != vs.YUV:
        raise vs.Error("deblendS: 'dclip' must be a YUV clip with fixed format")

    # Apply spatial-only denoise to dclip if requested.
    # This happens after the dclip=None fallback so denoise="RemoveGrain"
    # with no explicit dclip still works correctly (denoise is applied to
    # a copy of clip used only for detection, never touching the output).
    if denoise is not None:
        dclip = _apply_dclip_denoise(dclip, denoise, nlmeans_h, mode)

    # AviSynth srestore:
    #   frfac = bom || abs(omode-3)<2.5 ? 1 : ...
    # omode 1-5 and the pp modes are cadence-neutral: they replace or
    # rebuild frames but never decimate, so the ratio is pinned to 1
    # and true_fps has no effect.  Only omode 6 resolves a cadence.
    if bom or omode != 6:
        frfac = 1.0
    # AviSynth srestore:
    #   frfac = frate*5<irate || frate>irate ? 1 : abs(frate)/irate
    # A target rate above the source rate, or below a fifth of it, is not
    # a cadence srestore can resolve.  Falling back to frfac=1 gives a 1:1
    # passthrough instead of a nonsense numr/denm pair and a frame-
    # duplicating ChangeFPS at the end.
    elif true_fps * 5 < src_fps or true_fps > src_fps:
        frfac = 1.0
    else:
        frfac = abs(true_fps) / src_fps

    if abs(frfac * 1001 - cround(frfac * 1001)) < 0.01:
        numr = cround(frfac * 1001)
    elif abs(1001 / frfac - cround(1001 / frfac)) < 0.01:
        numr = 1001
    else:
        numr = cround(frfac * 9000)

    # AviSynth srestore has a second numr line that was missing here:
    #   numr = isfloat(frate) && abs(irate*numr/round(numr/frfac)-frate) >
    #          abs(irate*round(frate*100)/round(irate*100)-frate)
    #          ? round(frate*100) : numr
    # It switches to a hundredths-based ratio whenever that lands closer
    # to the requested rate than numr/denm does.  frate is always known
    # here (true_fps is filled in above), so the guard is always live.
    # This cannot break the frfac==1 passthrough: denm is derived from
    # numr, so the ratio stays 1:1 whatever numr becomes.
    if (abs(src_fps * numr / cround(numr / frfac) - true_fps) >
            abs(src_fps * cround(true_fps * 100) / cround(src_fps * 100) - true_fps)):
        numr = cround(true_fps * 100)

    denm = cround(numr / frfac)

    # _build_detection_clips returns luma post-trim(2), matching AviSynth's
    # det after .trim(2,0); used for bclp / dclp / dc_motion.
    if bsize < 8 or bsize > 256:
        raise vs.Error("deblendS: 'bsize' must be between 8 and 256")

    bclp, dclp_clip, dclip_small = _build_detection_clips(dclip, bsize=bsize, mode=mode, bom=bom)

    bclp_s = bclp.std.PlaneStats()
    dclp_s = dclp_clip.std.PlaneStats()

    # dc_motion: diff between frame n and frame n+2 on the small trimmed luma.
    # Matches AviSynth: lumadifference(det, det.trim(2,0)) where det is
    # already post-trim(2,0).
    #
    # Only omode 6 measures it -- srestore feeds a constant 1 to every
    # other mode -- so the node is not even built there.
    if not bom and abs(omode) > 5:
        det       = dclip_small
        det_ext   = det + det[-1] * 2   # pad so Trim(first=2) stays in-bounds
        dc_motion = core.std.PlaneStats(det_ext, det_ext.std.Trim(first=2))
    else:
        dc_motion = None

    engine = Engine(
        num_frames  = clip.num_frames,
        bclp_s      = bclp_s,
        dclp_s      = dclp_s,
        dc_motion_s = dc_motion,
        frfac       = frfac,
        numr        = numr,
        denm        = denm,
        mode        = mode,
        # Same derivation as srestore: thr = abs(thresh) + 0.01.  The
        # abs() makes the sign deliberately irrelevant.
        thr         = abs(thresh) + 0.01,
        omode       = omode,
        bom         = bom,
    )

    # srestore builds mec only for "abs(mode)>=2 && !bom": the pp modes
    # rebuild blends outright and never merge two source frames.
    if abs(mode) >= 2 and not bom:
        mec = core.std.Merge(clip, clip.std.Trim(first=1), weight=[0.5, 0.5])
    else:
        mec = clip

    pp_clip = _build_pp_clip(clip, omode) if bom else None

    if optical_flow:
        engine_name = optical_flow_engine.lower().strip()
        if engine_name == "svp":
            of_clip = _build_of_clip_svp(clip)
        else:
            # "mvtools-sf" is an explicit request for MVTools-sf, so skip
            # MVUtensils and fall back to MVTools2 if mvsf is missing.
            of_clip = _build_of_clip(
                clip, pel=of_pel, blksize=of_blksize,
                prefer_sf=engine_name in ("mvtools-sf", "mvtools_sf"),
            )
    else:
        of_clip = None

    def _decide(n: int) -> tuple[int, bool, bool, bool]:
        """
        Resolve output frame n to (source index, use mec, is blend, use pp).

        The blend flag deliberately belongs to the frame being EMITTED,
        not to frame n.  srestore's 'blend' describes the frame at the
        centre of the detection window, and the mapping then shifts the
        output by opos+dup -- so asking about n would flag a frame one
        or two positions away from the one actually written out.  Only
        the emitted frame's own flag makes "this frame is a blend, so
        synthesise it instead" true.
        """
        target, use_mec, _, use_pp = engine.get(n)
        t = max(0, min(target, clip.num_frames - 1))
        return t, use_mec, engine.get(t)[2], use_pp

    def _select(n: int) -> vs.VideoNode:
        t, use_mec, is_blend, use_pp = _decide(n)

        # A pp frame is one the mode could not resolve by picking a
        # neighbour, so it is rebuilt.  That takes precedence over the
        # optical-flow reconstruction: pp IS the reconstruction here.
        if use_pp and pp_clip is not None:
            return pp_clip[max(0, min(t, pp_clip.num_frames - 1))]

        if is_blend and of_clip is not None:
            return of_clip[max(0, min(t, of_clip.num_frames - 1))]

        src = mec if use_mec else clip
        return src[max(0, min(t, src.num_frames - 1))]

    output = clip.std.FrameEval(eval=_select)

    if show_debug:
        def _stamp(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
            target, use_mec, is_blend, use_pp = _decide(n)
            fout = f.copy()
            fout.props["DeblendTarget"]  = target
            fout.props["DeblendUseMec"]  = int(use_mec)
            fout.props["DeblendIsBlend"] = int(is_blend)
            fout.props["DeblendUsePP"]   = int(use_pp)
            fout.props["DeblendThr"]     = engine.thr
            return fout
        output = output.std.ModifyFrame(output, _stamp)

        pre_overlay = output

        def _overlay(n: int) -> vs.VideoNode:
            target, use_mec, is_blend, use_pp = _decide(n)
            lines = [
                f"frame:  {n}",
                f"omode:  {omode}",
                f"target: {target}  (offset {target - n:+d})",
                f"blend:  {'yes' if is_blend else 'no'}",
                f"mec:    {'yes' if use_mec else 'no'}",
                f"pp:     {'yes' if use_pp else 'no'}",
                f"thr:    {engine.thr:.2f}",
            ]
            text = "\n".join(lines)
            return core.text.Text(pre_overlay[n], text, alignment=7)

        output = pre_overlay.std.FrameEval(eval=_overlay)

    # numr == denm for every cadence-neutral mode, so there is nothing
    # to decimate and the source rate is kept untouched.
    if numr != denm:
        out_num = clip.fps_num * numr
        out_den = clip.fps_den * denm
        g = math.gcd(out_num, out_den)
        try:
            import ChangeFPS
            output = ChangeFPS.ChangeFPS(output, out_num // g, out_den // g)
        except ImportError:
            output = core.std.AssumeFPS(output, fpsnum=out_num // g, fpsden=out_den // g)

    return output


def deblendS6(
    clip:                 vs.VideoNode,
    src_fps:              Optional[float]        = None,
    true_fps:             Optional[float]        = None,
    show_debug:           bool                   = False,
    dclip:                Optional[vs.VideoNode] = None,
    mode:                 int                    = 2,
    optical_flow:         bool                   = False,
    optical_flow_engine:  str                    = "mvtools",
    of_pel:               int                    = 2,
    of_blksize:           int                    = 16,
    denoise:              Optional[str]          = None,
    nlmeans_h:            float                  = 7.0,
    thresh:               int                    = 16,
    bsize:                int                    = 32,
) -> vs.VideoNode:
    """
    deblendS with omode=6 -- the cadence-resolving mode this script
    started out as.  Kept so existing scripts keep working; new code
    should call deblendS directly.
    """
    return deblendS(
        clip,
        src_fps             = src_fps,
        true_fps            = true_fps,
        omode               = 6,
        show_debug          = show_debug,
        dclip               = dclip,
        mode                = mode,
        optical_flow        = optical_flow,
        optical_flow_engine = optical_flow_engine,
        of_pel              = of_pel,
        of_blksize          = of_blksize,
        denoise             = denoise,
        nlmeans_h           = nlmeans_h,
        thresh              = thresh,
        bsize               = bsize,
    )


# ---------------------------------------------------------------------------
# Optical-flow blend reconstruction
# ---------------------------------------------------------------------------

def _build_of_clip(
    clip:      vs.VideoNode,
    pel:       int  = 2,
    blksize:   int  = 16,
    prefer_sf: bool = False,
) -> vs.VideoNode:
    """
    Build the motion-compensated clip used to reconstruct blend frames.

    prefer_sf -- set for optical_flow_engine="mvtools-sf".  Moves
                 MVTools-sf to the front of the preference order:
                     core.mvsf -> core.mvu -> core.mv
                 With prefer_sf=False ("mvtools") the order is
                     core.mvu -> core.mvsf -> core.mv
                 Either way MVUtensils is only skipped when the plugin
                 that was explicitly asked for is actually installed.
    """
    fmt_orig = clip.format

    if hasattr(core, 'mvu') and not (prefer_sf and hasattr(core, 'mvsf')):
        mv       = core.mvu
        overlap  = blksize // 2

        needs_conv = False
        if fmt_orig.sample_type == vs.FLOAT and fmt_orig.bits_per_sample != 32:
            target_fmt = fmt_orig.replace(bits_per_sample=32, sample_type=vs.FLOAT)
            needs_conv = True
        elif fmt_orig.sample_type == vs.INTEGER and fmt_orig.bits_per_sample > 16:
            target_fmt = fmt_orig.replace(bits_per_sample=16, sample_type=vs.INTEGER)
            needs_conv = True
        else:
            target_fmt = fmt_orig

        src_work = clip.resize.Bicubic(format=target_fmt.id) if needs_conv else clip

        sup_src = mv.Super(
            src_work, blksize=blksize, overlap=overlap,
            pad=[blksize, blksize], pel=pel,
        )
        bwd = mv.Analyse(sup_src, delta=1)    # positive delta = backward
        fwd = mv.Analyse(sup_src, delta=-1)   # negative delta = forward

        interp = mv.FlowInter(src_work, sup_src, [bwd, fwd], time=50)

    elif hasattr(core, 'mvsf') or hasattr(core, 'mv'):
        # --- MVTools-sf first, MVTools2 as fallback.
        # Reached either because MVUtensils is missing, or because
        # prefer_sf routed an explicit "mvtools-sf" request here.
        if hasattr(core, 'mvsf'):
            mv = core.mvsf
            target_fmt = fmt_orig.replace(bits_per_sample=32, sample_type=vs.FLOAT)
            src_work   = clip.resize.Bicubic(format=target_fmt.id)
            needs_conv = True
        else:
            mv = core.mv
            if fmt_orig.bits_per_sample != 8 or fmt_orig.sample_type != vs.INTEGER:
                target_fmt = fmt_orig.replace(bits_per_sample=8, sample_type=vs.INTEGER)
                src_work   = clip.resize.Bicubic(format=target_fmt.id)
                needs_conv = True
            else:
                src_work   = clip
                needs_conv = False

        sup_src  = mv.Super(src_work, pel=pel, hpad=blksize, vpad=blksize)
        _analyse = mv.Analyze if hasattr(mv, 'Analyze') else mv.Analyse
        bwd      = _analyse(sup_src, isb=True,  blksize=blksize, overlap=blksize // 2)
        fwd      = _analyse(sup_src, isb=False, blksize=blksize, overlap=blksize // 2)

        interp = mv.FlowInter(src_work, sup_src, bwd, fwd, time=50)

    else:
        raise vs.Error(
            "deblendS: optical_flow=True requires MVUtensils (core.mvu), "
            "MVTools-sf (core.mvsf) or MVTools2 (core.mv) to be installed."
        )

    if needs_conv and fmt_orig.id != interp.format.id:
        interp = interp.resize.Bicubic(
            format=fmt_orig.id, dither_type="error_diffusion"
        )
    return interp

def _build_of_clip_svp(clip: vs.VideoNode) -> vs.VideoNode:
    if not hasattr(core, 'svp1') or not hasattr(core, 'svp2'):
        raise vs.Error(
            "deblend: optical_flow_engine='svp' requires the SVP plugin "
            "(core.svp1 and core.svp2) to be installed."
        )
    fmt_orig   = clip.format
    needs_conv = fmt_orig.bits_per_sample != 8 or fmt_orig.sample_type != vs.INTEGER
    if needs_conv:
        src8 = clip.resize.Bicubic(format=fmt_orig.replace(
            bits_per_sample=8, sample_type=vs.INTEGER
        ).id)
    else:
        src8 = clip

    blksize    = 32
    max_levels = max(1, int(math.log2(min(src8.width, src8.height) / blksize)))

    super_params   = '{"pel":2,"gpu":0}'
    analyse_params = (
        f'{{"block":{{"w":{blksize},"h":{blksize}}},'
        f'"main":{{"levels":{max_levels},"search":{{"type":4,"distance":-8}},'
        f'"penalty":{{"lambda":100,"plambda":100}}}},'
        f'"refine":[{{"search":{{"type":4,"distance":2}}}}]}}'
    )
    smooth_params  = '{"rate":{"num":2,"den":1,"abs":false},"algo":21,"mask":{"area":100}}'

    sup     = core.svp1.Super(src8, super_params)
    vecs    = core.svp1.Analyse(sup['clip'], sup['data'], src8, analyse_params)
    doubled = core.svp2.SmoothFps(
        src8, sup['clip'], sup['data'], vecs['clip'], vecs['data'], smooth_params
    )
    interp = doubled.std.SelectEvery(cycle=2, offsets=[1])
    if needs_conv:
        interp = interp.resize.Bicubic(
            format=fmt_orig.id, dither_type="error_diffusion"
        )
    return interp
