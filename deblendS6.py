"""
Stateful blend/telecine frame restorer for VapourSynth
======================================================================
This based on sRestore mode=6.

Handles mixed progressive-blend and telecined sources.  Uses the same
detection signals and decision logic as srestore (bclp/dclp blend
metrics, 7-frame rolling diff/motion/blend history, cadence offset
tracking).

Performance design
------------------
* Checkpoints — the background thread saves a full state snapshot
  every _CHECKPOINT_INTERVAL frames.  Seeking to any point only
  requires replaying from the nearest checkpoint, not from frame 0.
* Demand-driven threading — the playback thread stays at most
  _LOOKAHEAD frames ahead of the consumer.  Nothing is computed until
  a frame is actually requested.

NOTE: There is NO scene-cut based state reset during normal sequential
playback.  AviSynth's srestore has no such mechanism, and resetting on
scene cuts (especially after long black/static sections) causes the
Markov state to diverge badly from the AviSynth reference.
Checkpoints are kept purely for seek support.

Usage
-----
deblendS6(clip, src_fps=60000/1001, true_fps=24000/1001)

Parameters
----------
clip        VideoNode   Input clip. Must be YUV.
src_fps     float       Source fps (default: read from clip).
true_fps    float       Target fps (default: nearest lower common rate).
optical_flow          bool   When True, detected blend frames are reconstructed
                             using motion-compensated interpolation instead of
                             the default simple 50/50 neighbour merge.
                             Default: False.
                             WHEN TO USE: live-action or CG with smooth motion.
                             WHEN NOT TO USE: anime/cel animation — vectors are
                             unreliable on hard-cut discrete motion; the simple
                             merge is better.
optical_flow_engine   str    Which engine to use when optical_flow=True.
                             "mvtools" (default) — uses core.mvsf if available,
                               falls back to core.mv.  Direct control via
                               of_pel and of_blksize.
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
nlmeans_h   float       Strength (h) for NLMeans denoising. Default 7.0.
                        Increase for heavier grain (10-16), decrease for
                        light noise (3-5).  Only used when denoise="NLMeans".
mode        int         Controls duplicate/merge detection and mec usage.
                        Default is 2.  Options:

                         0  Basic frame selection only.  No duplicate or
                            merge detection, no mec clip.
                         1  Duplicate detection active.  No mec clip.
                         2  Default.  Mec clip built, duplicate detection
                            active, merge detection off.
                         3  Mec clip built, but duplicate detection
                            disabled.
                         4  Most aggressive.  Mec clip + duplicate
                            detection + merge detection.  Try this if
                            mode 2 misses blends.
                        -2  Same as 2 with inverted duplicate thresholds.
                        -3  Same as 3 with inverted behaviour.
                        -4  Same as 4 with inverted thresholds.

                        Summary of what each flag enables:
                          abs(mode) >= 2  ->  mec clip is constructed
                          abs(mode) != 3  ->  duplicate detection active
                          abs(mode) >  2  ->  merge (mec) detection active
"""

from __future__ import annotations

import copy
import math
import threading
from dataclasses import dataclass
from typing import Optional

import vapoursynth as vs

core = vs.core

# How far ahead of the consumer the playback thread runs.
_LOOKAHEAD = 80

# Save a state checkpoint every N frames for seek support.
_CHECKPOINT_INTERVAL = 200

# Fixed threshold matching AviSynth srestore default (thresh=16).
# NOT auto-estimated: black/static frames at the start would corrupt any
# bootstrap estimate and produce thr << 16, causing massive over-detection.
_THR = 16.01

_MAGIC_OFFSET = 1.0 / 64.0


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _cround(x: float) -> int:
    return math.floor(x + 0.5) if x > 0 else math.ceil(x - 0.5)


def _nearest_lower_fps(src: float) -> float:
    candidates = [24000/1001, 24.0, 25.0, 30000/1001, 30.0,
                  48000/1001, 48.0, 50.0, 60000/1001, 60.0]
    below = [c for c in candidates if c < src - 0.01]
    return max(below) if below else src


def _get_plane(clip: vs.VideoNode, plane: int = 0) -> vs.VideoNode:
    return core.std.ShufflePlanes(clip, plane, vs.GRAY)


# ---------------------------------------------------------------------------
# Detection clip construction  (identical to srestore)
# ---------------------------------------------------------------------------

def _build_detection_clips(
    dclip: vs.VideoNode,
    bsize: int   = 32,
    srad:  float = 12.0,
) -> tuple[vs.VideoNode, vs.VideoNode, vs.VideoNode]:
    """
    Returns (bclp, dclp, luma_trimmed).

    luma_trimmed -- detection luma with Trim(first=2) applied.
                    Used for bclp / dclp AND for dc_motion, matching
                    AviSynth where det = det.pointresize(...).trim(2,0)
                    and all further operations use that trimmed det.
    """
    if dclip.format.id != vs.YUV420P8:
        dclip = dclip.resize.Bicubic(format=vs.YUV420P8)

    new_w = int(dclip.width  / 2 / srad + 4) * 4
    new_h = int(dclip.height / 2 / srad + 4) * 4
    small = dclip.resize.Point(new_w, new_h)

    luma = _get_plane(small, 0).std.Trim(first=2)

    EXPR = (core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else
            core.akarin.Expr   if hasattr(core, 'akarin')   else
            core.cranexpr.Expr if hasattr(core, 'cranexpr') else
            core.std.Expr)

    diff = core.std.MakeDiff(luma, luma.std.Trim(first=1))

    expr_b = ('x 128 - y 128 - * 0 > '
              'x 128 - abs y 128 - abs < '
              'x 128 - 128 x - * y 128 - 128 y - * ? '
              'x y + 256 - dup * ? 0.25 * 128 +')
    bclp = EXPR([diff, diff.std.Trim(first=1)], expr=[expr_b])
    bclp = bclp.resize.Bilinear(bsize, bsize)

    dclp = (diff.std.Trim(first=1)
               .std.Lut(function=lambda x: max(_cround(abs(x - 128) ** 1.1 - 1), 0))
               .resize.Bilinear(bsize, bsize))

    return bclp, dclp, luma


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
# _compute_decision -- faithful port of srestore mode=6 logic
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
) -> tuple[int, bool, bool]:
    """Mutates state in-place. Returns (target_frame_index, use_mec, is_blend)."""

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
           else -_cround((cfo + state['offs']) / (2 * numr)) if bfo
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
    m_v = s.dc_diff
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
              state['bn1'] = state['bn2'] = b_v
    state['bn3'] = b_v

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
    odm += _cround((cof - odm) / (2 * denm)) * 2 * denm

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
        opos = -_cround(
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

    use_mec = mer and dup == 0
    target  = n + opos + dup - (1 if dup == 0 and mer and dbc2 < dcn2 else 0)

    return target, use_mec, blend


# ---------------------------------------------------------------------------
# Engine: sequential stats fetch + decision, with checkpointing for seeks
# ---------------------------------------------------------------------------

class Engine:
    """
    Single background thread that fetches RawStats and runs
    _compute_decision sequentially, preserving the Markov state across
    all frames without any mid-stream resets.

    Checkpoints are saved every _CHECKPOINT_INTERVAL frames solely to
    support random-access seeks; they do not affect normal playback.
    """

    def __init__(
        self,
        num_frames:  int,
        bclp_s:      vs.VideoNode,
        dclp_s:      vs.VideoNode,
        dc_motion_s: vs.VideoNode,
        frfac:       float,
        numr:        int,
        denm:        int,
        mode:        int,
    ) -> None:
        self._nf    = num_frames
        self._bclp  = bclp_s
        self._dclp  = dclp_s
        self._dcmot = dc_motion_s
        self._frfac = frfac
        self._numr  = numr
        self._denm  = denm
        self._mode  = mode

        self._nb_max = bclp_s.num_frames  - 1
        self._nd_max = dclp_s.num_frames  - 1
        self._nm_max = dc_motion_s.num_frames - 1

        self._decisions: list[Optional[tuple[int, bool, bool]]] = [None] * num_frames
        self._events    = [threading.Event() for _ in range(num_frames)]

        self._checkpoints: dict[int, dict] = {}
        self._cp_lock = threading.Lock()

        self._head      = _LOOKAHEAD
        self._head_lock = threading.Lock()
        self._head_ev   = threading.Event()
        self._cancel    = threading.Event()

        st0 = _initial_state()
        with self._cp_lock:
            self._checkpoints[0] = copy.deepcopy(st0)

        self._thread = threading.Thread(
            target=self._run, args=(0, st0), daemon=True
        )
        self._thread.start()

    @property
    def thr(self) -> float:
        return _THR

    def advance(self, n: int) -> None:
        with self._head_lock:
            want = n + _LOOKAHEAD
            if want > self._head:
                self._head = want
                self._head_ev.set()

    def get(self, n: int) -> tuple[int, bool, bool]:
        self.advance(n)
        if not self._events[n].wait(timeout=0.08):
            self._seek_restart(n)
        self._events[n].wait()
        return self._decisions[n]

    def _seek_restart(self, n: int) -> None:
        with self._cp_lock:
            candidates = [k for k in self._checkpoints if k <= n]
            if not candidates:
                return
            cp_frame = max(candidates)
            cp_state = copy.deepcopy(self._checkpoints[cp_frame])

        if self._events[n].is_set():
            return

        self._cancel.set()
        self._thread.join(timeout=0.5)
        self._cancel = threading.Event()

        with self._head_lock:
            self._head = n + _LOOKAHEAD

        self._thread = threading.Thread(
            target=self._run, args=(cp_frame, cp_state), daemon=True
        )
        self._thread.start()
        self.advance(n)

    def _run(self, start: int, state: dict) -> None:
        cancel = self._cancel

        for n in range(start, self._nf):
            if cancel.is_set():
                return

            while True:
                if cancel.is_set():
                    return
                with self._head_lock:
                    ok = n <= self._head
                if ok:
                    break
                self._head_ev.wait(timeout=0.02)
                self._head_ev.clear()

            s = self._fetch(n)

            # Checkpoint for seek support only -- no effect on playback
            if n % _CHECKPOINT_INTERVAL == 0 or n == start:
                with self._cp_lock:
                    if n not in self._checkpoints:
                        self._checkpoints[n] = copy.deepcopy(state)

            decision = _compute_decision(
                n, s, state,
                self._frfac, self._numr, self._denm, _THR, self._mode,
            )

            self._decisions[n] = decision
            self._events[n].set()

    def _fetch(self, n: int) -> RawStats:
        fb  = self._bclp.get_frame(min(n, self._nb_max))
        fd  = self._dclp.get_frame(min(n, self._nd_max))
        fdc = self._dcmot.get_frame(min(n, self._nm_max))
        return RawStats(
            b_min   = float(fb.props['PlaneStatsMin']),
            b_max   = float(fb.props['PlaneStatsMax']),
            d_max   = float(fd.props['PlaneStatsMax']),
            # PlaneStatsDiff is [0.0, 1.0]; scale to [0, 255] to match
            # AviSynth's lumadifference() which works in that range.
            dc_diff = float(fdc.props['PlaneStatsDiff']) * 255.0 + _MAGIC_OFFSET,
        )


# ---------------------------------------------------------------------------
# dclip denoiser helper
# ---------------------------------------------------------------------------

def _apply_dclip_denoise(
    dclip:     vs.VideoNode,
    denoise:   str,
    nlmeans_h: float = 7.0,
) -> vs.VideoNode:
    """
    Apply a purely spatial denoise to dclip before detection thumbnail
    construction.  Temporal denoisers must never be used here: they smooth
    the inter-frame diff signal that blend detection depends on.

    Parameters
    ----------
    dclip      Source clip (any YUV format).
    denoise    "RemoveGrain" or "NLMeans" (case-insensitive).
    nlmeans_h  NLMeans h strength (luma only, default 7.0).
    """
    method = denoise.strip().lower()

    if method == "removegrain":
        if not hasattr(core, 'zsmooth') and not hasattr(core, 'rgvs'):
            raise vs.Error(
                "deblendS6: denoise='RemoveGrain' requires either the "
                "zsmooth or rgvs (RemoveGrainVS) plugin to be installed."
            )
        # Pass 1: mode 2 -- clip each pixel to the min/max of its 8 neighbours.
        #   Kills isolated spike pixels from blocking and ringing without
        #   blurring edges.
        # Pass 2: mode 12 -- 3x3 weighted average (centre weight 4, edges 2,
        #   corners 1 -- a mild Gaussian-like blur).  Smooths fine grain that
        #   pass 1 leaves behind.
        # U/V mode 1 = passthrough (copy plane unchanged).
        # Chroma is irrelevant because _build_detection_clips extracts only
        # the Y plane before building thumbnails.
        # zsmooth is preferred over rgvs: supports higher bit depths natively
        # and is generally faster.
        _rg = core.zsmooth.RemoveGrain if hasattr(core, 'zsmooth') else core.rgvs.RemoveGrain
        dclip = _rg(dclip, mode=[2,  1, 1])
        dclip = _rg(dclip, mode=[12, 1, 1])

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
        # single 'YUV' call.  For 4:4:4 a single 'YUV' pass is fine.
        # We always denoise chroma too: even though _build_detection_clips
        # discards chroma, leaving it undenoised would make the dclip
        # format inconsistent with what downstream plugins may expect.
        fmt        = dclip.format
        subsampled = (fmt.subsampling_w > 0 or fmt.subsampling_h > 0)

        # Common kwargs shared across all backends / passes.
        kw = dict(d=0, a=2, s=3, h=nlmeans_h, wmode=0, wref=1.0)

        if hasattr(core, 'nlm_ispc'):
            if subsampled:
                dclip = dclip.nlm_ispc.NLMeans(**kw, channels='Y')
                dclip = dclip.nlm_ispc.NLMeans(**kw, channels='UV')
            else:
                dclip = dclip.nlm_ispc.NLMeans(**kw, channels='YUV')

        elif hasattr(core, 'nlm_cuda'):
            if subsampled:
                dclip = dclip.nlm_cuda.NLMeans(**kw, channels='Y')
                dclip = dclip.nlm_cuda.NLMeans(**kw, channels='UV')
            else:
                dclip = dclip.nlm_cuda.NLMeans(**kw, channels='YUV')

        elif hasattr(core, 'knlm'):
            # KNLMeansCL uses 'channels' spelled out differently and has no
            # wmode/wref; it also requires device_type/device_id for OpenCL.
            # It only supports 4:4:4 or per-plane calls for subsampled input.
            if subsampled:
                dclip = dclip.knlm.KNLMeansCL(d=0, a=2, s=3, h=nlmeans_h,
                                               channels='Y')
                dclip = dclip.knlm.KNLMeansCL(d=0, a=2, s=3, h=nlmeans_h,
                                               channels='UV')
            else:
                dclip = dclip.knlm.KNLMeansCL(d=0, a=2, s=3, h=nlmeans_h,
                                               channels='YUV')

        else:
            raise vs.Error(
                "deblendS6: denoise='NLMeans' requires one of the following "
                "plugins to be installed: nlm_ispc, nlm_cuda, or knlm "
                "(KNLMeansCL)."
            )

    else:
        raise vs.Error(
            f"deblendS6: unknown denoise method '{denoise}'. "
            "Valid values: 'RemoveGrain', 'NLMeans'."
        )

    return dclip


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

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
    denoise:              Optional[str]           = None,
    nlmeans_h:            float                  = 7.0,
) -> vs.VideoNode:
    if clip.format is None or clip.format.color_family != vs.YUV:
        raise vs.Error("deblendS6: input must be a YUV clip with fixed format")

    if src_fps is None:
        src_fps = clip.fps_num / clip.fps_den
    if true_fps is None:
        true_fps = _nearest_lower_fps(src_fps)
    if dclip is None:
        dclip = clip

    # Apply spatial-only denoise to dclip if requested.
    # This happens after the dclip=None fallback so denoise="RemoveGrain"
    # with no explicit dclip still works correctly (denoise is applied to
    # a copy of clip used only for detection, never touching the output).
    if denoise is not None:
        dclip = _apply_dclip_denoise(dclip, denoise, nlmeans_h)

    frfac = true_fps / src_fps

    if abs(frfac * 1001 - _cround(frfac * 1001)) < 0.01:
        numr = _cround(frfac * 1001)
    elif abs(1001 / frfac - _cround(1001 / frfac)) < 0.01:
        numr = 1001
    else:
        numr = _cround(frfac * 9000)
    denm = _cround(numr / frfac)

    # _build_detection_clips returns luma post-trim(2), matching AviSynth's
    # det after .trim(2,0); used for bclp / dclp / dc_motion.
    bclp, dclp_clip, dclip_small = _build_detection_clips(dclip)

    bclp_s = bclp.std.PlaneStats()
    dclp_s = dclp_clip.std.PlaneStats()

    # dc_motion: diff between frame n and frame n+2 on the small trimmed luma.
    # Matches AviSynth: lumadifference(det, det.trim(2,0)) where det is
    # already post-trim(2,0).
    det     = dclip_small
    det_ext = det + det[-1] * 2   # pad so Trim(first=2) stays in-bounds
    dc_motion = core.std.PlaneStats(det_ext, det_ext.std.Trim(first=2))

    engine = Engine(
        num_frames  = clip.num_frames,
        bclp_s      = bclp_s,
        dclp_s      = dclp_s,
        dc_motion_s = dc_motion,
        frfac       = frfac,
        numr        = numr,
        denm        = denm,
        mode        = mode,
    )

    if abs(mode) >= 2:
        mec = core.std.Merge(clip, clip.std.Trim(first=1), weight=[0.5, 0.5])
    else:
        mec = clip

    if optical_flow:
        engine_name = optical_flow_engine.lower().strip()
        if engine_name == "svp":
            of_clip = _build_of_clip_svp(clip)
        else:
            mv = _get_mv_plugin()
            of_clip = _build_of_clip(clip, mv, pel=of_pel, blksize=of_blksize)
    else:
        of_clip = None

    def _select(n: int) -> vs.VideoNode:
        engine.advance(n)
        target, use_mec, is_blend = engine.get(n)

        t = max(0, min(target, clip.num_frames - 1))

        if is_blend and of_clip is not None:
            return of_clip[max(0, min(t, of_clip.num_frames - 1))]

        src = mec if use_mec else clip
        return src[max(0, min(t, src.num_frames - 1))]

    output = clip.std.FrameEval(eval=_select)

    if show_debug:
        def _stamp(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
            target, use_mec, is_blend = engine.get(n)
            fout = f.copy()
            fout.props["DeblendTarget"]  = target
            fout.props["DeblendUseMec"]  = int(use_mec)
            fout.props["DeblendIsBlend"] = int(is_blend)
            fout.props["DeblendThr"]     = _THR
            return fout
        output = output.std.ModifyFrame(output, _stamp)

        pre_overlay = output

        def _overlay(n: int) -> vs.VideoNode:
            target, use_mec, is_blend = engine.get(n)
            lines = [
                f"frame:  {n}",
                f"target: {target}  (offset {target - n:+d})",
                f"blend:  {'yes' if is_blend else 'no'}",
                f"mec:    {'yes' if use_mec else 'no'}",
                f"thr:    {_THR:.2f}",
            ]
            text = "\n".join(lines)
            return core.text.Text(pre_overlay[n], text, alignment=7)

        output = pre_overlay.std.FrameEval(eval=_overlay)

    out_num = clip.fps_num * numr
    out_den = clip.fps_den * denm
    g = math.gcd(out_num, out_den)
    try:
        import ChangeFPS
        output = ChangeFPS.ChangeFPS(output, out_num // g, out_den // g)
    except ImportError:
        output = core.std.AssumeFPS(output, fpsnum=out_num // g, fpsden=out_den // g)

    return output


# ---------------------------------------------------------------------------
# Optical-flow blend reconstruction
# ---------------------------------------------------------------------------

def _get_mv_plugin():
    if hasattr(core, 'mvsf'):
        return core.mvsf
    if hasattr(core, 'mv'):
        return core.mv
    raise vs.Error(
        "deblend: optical_flow=True requires MVTools2 (core.mv) or "
        "MVTools-sf (core.mvsf) to be installed."
    )


def _build_of_clip(
    clip:    vs.VideoNode,
    mv:      object,
    pel:     int = 2,
    blksize: int = 16,
) -> vs.VideoNode:
    fmt_orig = clip.format

    if hasattr(core, 'mvsf'):
        target_fmt = fmt_orig.replace(bits_per_sample=32, sample_type=vs.FLOAT)
        src_work   = clip.resize.Bicubic(format=target_fmt.id)
        needs_conv = True
    else:
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