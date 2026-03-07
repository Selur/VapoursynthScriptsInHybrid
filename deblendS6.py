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
* Scene-change checkpoints — the background thread saves a full state
  snapshot at every detected scene cut AND every _CHECKPOINT_INTERVAL
  frames.  Seeking to any point only requires replaying from the
  nearest checkpoint, not from frame 0.
* Demand-driven threading — the playback thread stays at most
  _LOOKAHEAD frames ahead of the consumer.  Nothing is computed until
  a frame is actually requested.
* Inside a scene the cadence state is continuous.  At a scene cut the
  state is reset (cadence re-locks from scratch), which is correct
  because blend patterns don't carry across cuts.

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
                          abs(mode) >= 2  →  mec clip is constructed
                          abs(mode) != 3  →  duplicate detection active
                          abs(mode) >  2  →  merge (mec) detection active
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

# Save a state checkpoint every N frames regardless of scene changes.
_CHECKPOINT_INTERVAL = 200

# A diff spike this many times the running median signals a scene cut.
_SCENE_CUT_RATIO = 8.0

# Number of frames used to refine the auto-threshold estimate.
# The fallback value is used immediately so frame 0 is never stalled.
_THR_BOOTSTRAP   = 60
_THR_FALLBACK    = 16.01   # srestore's default; used until bootstrap completes


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
    if dclip.format.id != vs.YUV420P8:
        dclip = dclip.resize.Bicubic(format=vs.YUV420P8)

    new_w = int(dclip.width  / 2 / srad + 4) * 4
    new_h = int(dclip.height / 2 / srad + 4) * 4
    small = dclip.resize.Point(new_w, new_h)
    luma  = _get_plane(small, 0)
    luma  = luma.std.Trim(first=2)

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
    dc_diff: float   # already multiplied by 255


# ---------------------------------------------------------------------------
# Decision state (the Markov chain that _compute_decision mutates)
# ---------------------------------------------------------------------------

def _initial_state() -> dict:
    v = 0.015625
    return dict(
        lfr=-100, offs=0.0, ldet=-100, lpos=0,
        d32=v, d21=v, d10=v, d01=v, d12=v, d23=v, d34=v,
        m42=v, m31=v, m20=v, m11=v, m02=v, m13=v, m24=v,
        bp2=0.125, bp1=0.125, bn0=0.125, bn1=0.125, bn2=0.125, bn3=0.125,
        cp2=0.125, cp1=0.125, cn0=0.125, cn1=0.125, cn2=0.125, cn3=0.125,
    )


# ---------------------------------------------------------------------------
# _compute_decision — faithful port of srestore's logic
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
) -> tuple[int, bool]:
    """Mutates state in-place. Returns (target_frame_index, use_mec)."""

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

    d_v = s.d_max + 0.015625
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

    m_v = s.dc_diff + 0.015625
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

    b_v = max(128 - s.b_min, 0.125)
    c_v = max(s.b_max - 128, 0.125)

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

    i2   = pos + 2
    dbb2 = [d43,          state['d32'], state['d21'], state['d10'], state['d01']][i2]
    dbc2 = [state['d32'], state['d21'], state['d10'], state['d01'], state['d12']][i2]
    dcn2 = [state['d21'], state['d10'], state['d01'], state['d12'], state['d23']][i2]
    dnn2 = [state['d10'], state['d01'], state['d12'], state['d23'], state['d34']][i2]

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

    mer = (opos == pos and dup == 0 and abs(mode) > 2 and
           (dbc2 * 8 < dcn2 or dbc2 * 8 < dbb2 or
            dcn2 * 8 < dbc2 or dcn2 * 8 < dnn2 or
            dbc2 * 2 < thr or dcn2 * 2 < thr or
            dnn2 * 9 < dbc2 and dcn2 * 3 < dbc2 or
            dbb2 * 9 < dcn2 and dbc2 * 3 < dcn2))

    use_mec  = mer and dup == 0
    target   = n + opos + dup - (1 if dup == 0 and mer and dbc2 < dcn2 else 0)

    return target, use_mec, blend


# ---------------------------------------------------------------------------
# Scene-cut detector
# Lightweight: just watches d_max relative to a rolling median estimate.
# ---------------------------------------------------------------------------

class SceneCutDetector:
    """
    Tracks a running estimate of the 'normal' diff level and flags frames
    where d_max spikes far above it as scene cuts.

    Uses an exponential moving average of the lower half of seen diffs
    (ignoring the top half so that blends/motion don't inflate the
    baseline).  A frame is a cut if d_max > _SCENE_CUT_RATIO * baseline.
    """

    def __init__(self) -> None:
        self._ema   = 8.0   # initial guess; adapts quickly
        self._alpha = 0.05  # EMA decay

    def is_cut(self, d_max: float) -> bool:
        cut = d_max > _SCENE_CUT_RATIO * self._ema
        # Only update EMA with low-motion frames to keep baseline clean
        if d_max < 2 * self._ema:
            self._ema = (1 - self._alpha) * self._ema + self._alpha * d_max
        return cut


# ---------------------------------------------------------------------------
# Engine: stats fetch + decision, with checkpointing and seek support
# ---------------------------------------------------------------------------

class Engine:
    """
    Single background thread that:
      - fetches RawStats for each frame
      - runs _compute_decision sequentially (preserving Markov state)
      - saves checkpoints at scene cuts and every _CHECKPOINT_INTERVAL frames
      - estimates thr from the first _THR_BOOTSTRAP frames it processes
      - supports fast seeks by restarting from the nearest checkpoint

    The playback thread is demand-driven: it only runs up to
    _LOOKAHEAD frames ahead of whatever frame the consumer last requested.
    """

    def __init__(
        self,
        num_frames:  int,
        bclp_s:      vs.VideoNode,   # already PlaneStats'd
        dclp_s:      vs.VideoNode,   # already PlaneStats'd
        dc_motion_s: vs.VideoNode,   # already PlaneStats'd
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

        # Results
        self._decisions: list[Optional[tuple[int, bool, bool]]] = [None] * num_frames
        self._events    = [threading.Event() for _ in range(num_frames)]

        # Checkpoints: frame_index → (deep-copy of state, thr at that point)
        self._checkpoints: dict[int, tuple[dict, float]] = {}
        self._cp_lock = threading.Lock()

        # Auto-threshold — starts at fallback, refined after _THR_BOOTSTRAP frames
        self._thr:         float       = _THR_FALLBACK
        self._thr_samples: list[float] = []

        # Playback-thread control
        self._head      = _LOOKAHEAD
        self._head_lock = threading.Lock()
        self._head_ev   = threading.Event()
        self._cancel    = threading.Event()

        # Save checkpoint at frame 0 immediately
        st0 = _initial_state()
        with self._cp_lock:
            self._checkpoints[0] = (copy.deepcopy(st0), _THR_FALLBACK)

        self._thread = threading.Thread(
            target=self._run, args=(0, st0), daemon=True
        )
        self._thread.start()

    # ------------------------------------------------------------------
    # thr property — always available immediately (fallback until bootstrap completes)
    # ------------------------------------------------------------------

    @property
    def thr(self) -> float:
        return self._thr

    # ------------------------------------------------------------------
    # Consumer API
    # ------------------------------------------------------------------

    def advance(self, n: int) -> None:
        with self._head_lock:
            want = n + _LOOKAHEAD
            if want > self._head:
                self._head = want
                self._head_ev.set()

    def get(self, n: int) -> tuple[int, bool, bool]:
        self.advance(n)
        if not self._events[n].wait(timeout=0.08):
            # Likely a seek — restart from nearest checkpoint
            self._seek_restart(n)
        self._events[n].wait()
        return self._decisions[n]

    # ------------------------------------------------------------------
    # Internal: seek restart
    # ------------------------------------------------------------------

    def _seek_restart(self, n: int) -> None:
        with self._cp_lock:
            candidates = [k for k in self._checkpoints if k <= n]
            if not candidates:
                return
            cp_frame = max(candidates)
            cp_state, cp_thr = self._checkpoints[cp_frame]
            cp_state = copy.deepcopy(cp_state)

        # Already computed by someone else while we were waiting?
        if self._events[n].is_set():
            return

        # Cancel current thread and start a new one from the checkpoint
        self._cancel.set()
        self._thread.join(timeout=0.5)
        self._cancel = threading.Event()

        with self._head_lock:
            self._head = n + _LOOKAHEAD

        # Restore thr from checkpoint if it has been refined
        if cp_thr is not None:
            self._thr = cp_thr

        self._thread = threading.Thread(
            target=self._run, args=(cp_frame, cp_state), daemon=True
        )
        self._thread.start()
        # Give the new thread permission to run to n immediately
        self.advance(n)

    # ------------------------------------------------------------------
    # Background thread
    # ------------------------------------------------------------------

    def _run(self, start: int, state: dict) -> None:
        cut_detector = SceneCutDetector()
        cancel       = self._cancel   # local ref so swap doesn't affect us

        for n in range(start, self._nf):
            if cancel.is_set():
                return

            # Throttle: don't run more than _LOOKAHEAD ahead of consumer
            while True:
                if cancel.is_set():
                    return
                with self._head_lock:
                    ok = n <= self._head
                if ok:
                    break
                self._head_ev.wait(timeout=0.02)
                self._head_ev.clear()

            # Fetch raw stats
            s = self._fetch(n)

            # --- Auto-threshold refinement (non-blocking) ---
            # Accumulate samples; once we have enough, refine thr in-place.
            # Until then the fallback value is used — no frame is ever stalled.
            if len(self._thr_samples) < _THR_BOOTSTRAP:
                self._thr_samples.append(s.d_max)
                if len(self._thr_samples) == _THR_BOOTSTRAP:
                    samples = sorted(self._thr_samples)
                    p20 = samples[max(0, int(len(samples) * 0.20) - 1)]
                    self._thr = max(4.0, min(p20, 64.0)) + 0.01

            thr = self._thr

            # --- Scene cut detection ---
            is_cut = cut_detector.is_cut(s.d_max)
            if is_cut:
                # Reset Markov state at the cut — cadence re-locks from scratch
                state = _initial_state()
                state['lfr'] = n - 1   # so jmp=True on next step

            # --- Checkpoint saving ---
            save_cp = (is_cut or
                       n % _CHECKPOINT_INTERVAL == 0 or
                       n == start)
            if save_cp:
                with self._cp_lock:
                    if n not in self._checkpoints:
                        self._checkpoints[n] = (copy.deepcopy(state), thr)

            # --- Decision ---
            decision = _compute_decision(
                n, s, state,
                self._frfac, self._numr, self._denm, thr, self._mode,
            )

            self._decisions[n] = decision
            self._events[n].set()

    # ------------------------------------------------------------------

    def _fetch(self, n: int) -> RawStats:
        fb  = self._bclp.get_frame(min(n, self._nb_max))
        fd  = self._dclp.get_frame(min(n, self._nd_max))
        fdc = self._dcmot.get_frame(min(n, self._nm_max))
        return RawStats(
            b_min   = float(fb.props['PlaneStatsMin']),
            b_max   = float(fb.props['PlaneStatsMax']),
            d_max   = float(fd.props['PlaneStatsMax']),
            dc_diff = float(fdc.props['PlaneStatsDiff']) * 255.0,
        )


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
) -> vs.VideoNode:
    if clip.format is None or clip.format.color_family != vs.YUV:
        raise vs.Error("deblendS6: input must be a YUV clip with fixed format")

    if src_fps is None:
        src_fps = clip.fps_num / clip.fps_den
    if true_fps is None:
        true_fps = _nearest_lower_fps(src_fps)
    if dclip is None:
        dclip = clip

    frfac = true_fps / src_fps

    if abs(frfac * 1001 - _cround(frfac * 1001)) < 0.01:
        numr = _cround(frfac * 1001)
    elif abs(1001 / frfac - _cround(1001 / frfac)) < 0.01:
        numr = 1001
    else:
        numr = _cround(frfac * 9000)
    denm = _cround(numr / frfac)

    bclp, dclp_clip, dclip_small = _build_detection_clips(dclip)

    # PlaneStats nodes (cheap — just adds stats props, lazy evaluation)
    bclp_s = bclp.std.PlaneStats()
    dclp_s = dclp_clip.std.PlaneStats()

    dclip_nxt  = (dclip_small.std.Trim(first=2) +
                  dclip_small[-1] * 2)
    dc_motion  = core.std.PlaneStats(
        dclip_small,
        dclip_nxt[:dclip_small.num_frames]
    )

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

    # mec clip (mode >= 2)
    if abs(mode) >= 2:
        mec = core.std.Merge(clip, clip.std.Trim(first=1), weight=[0, 0.5])
        mec = core.std.Merge(mec,  clip.std.Trim(first=1), weight=[0.5, 0])
    else:
        mec = clip

    # Optical-flow interpolation clip (built once, evaluated lazily per frame)
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
        # On a detected blend frame, use OF interpolation if available.
        # On non-blend frames always use the source.
        if is_blend and of_clip is not None:
            return of_clip[max(0, min(n, of_clip.num_frames - 1))]
        src = mec if use_mec else clip
        return src[max(0, min(target, src.num_frames - 1))]

    output = clip.std.FrameEval(eval=_select)

    if show_debug:
        def _stamp(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
            target, use_mec, is_blend = engine.get(n)
            fout = f.copy()
            fout.props["DeblendTarget"]  = target
            fout.props["DeblendUseMec"]  = int(use_mec)
            fout.props["DeblendIsBlend"] = int(is_blend)
            fout.props["DeblendThr"]     = engine.thr
            return fout
        output = output.std.ModifyFrame(output, _stamp)

        # Burn a readable text overlay per frame.
        # `pre_overlay` captures the clip *before* FrameEval redefines
        # `output`, breaking the circular reference that caused the
        # IndexError when using `output[n]` inside its own FrameEval.
        pre_overlay = output

        def _overlay(n: int) -> vs.VideoNode:
            target, use_mec, is_blend = engine.get(n)
            thr = engine.thr
            lines = [
                f"frame:  {n}",
                f"target: {target}  (offset {target - n:+d})",
                f"blend:  {'yes' if is_blend else 'no'}",
                f"mec:    {'yes' if use_mec else 'no'}",
                f"thr:    {thr:.2f}",
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
    """
    Return the MVTools plugin namespace to use.
    Prefers mvsf (float, higher quality) over mv (integer).
    Raises vs.Error if neither is available.
    """
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
    """
    Return a clip where every frame is a motion-compensated interpolation
    at t=0.5 between frame[n-1] and frame[n+1].

    Used as a replacement for the simple 50/50 Merge on blend frames.
    FlowInter traces per-pixel motion vectors and warps each neighbour to
    the mid-point, producing a sharper result on frames with lateral motion.

    WHEN NOT TO USE: anime/cel animation — use the default merge instead.

    Parameters
    ----------
    clip     Full-resolution source clip (YUV, any bit depth).
    mv       MVTools plugin namespace (core.mv or core.mvsf).
    pel      Sub-pixel precision: 1=pixel, 2=half-pixel, 4=quarter.
    blksize  Block size for vector search.
    """
  
    fmt_orig = clip.format

    # mvsf requires single-precision float; core.mv requires 8-bit integer.
    # Detect which plugin we have by checking for a known mvsf-only attribute.
    plugin_name = getattr(mv, 'namespace', None) or ''
    is_mvsf = 'mvsf' in plugin_name.lower() or (
        hasattr(core, 'mvsf') and type(mv) is type(core.mvsf)
    )

    # Safest approach: just check what format mvsf.Super actually needs
    # by always converting to float32 when mvsf is loaded at all.
    if hasattr(core, 'mvsf'):
        # mvsf is present — always feed it float32 regardless of which
        # namespace 'mv' points to, since mvsf.Super rejects non-float.
        target_fmt = fmt_orig.replace(bits_per_sample=32, sample_type=vs.FLOAT)
        src_work   = clip.resize.Bicubic(format=target_fmt.id)
        needs_conv = True
    else:
        # Only core.mv available — it needs 8-bit integer
        if fmt_orig.bits_per_sample != 8 or fmt_orig.sample_type != vs.INTEGER:
            target_fmt = fmt_orig.replace(bits_per_sample=8, sample_type=vs.INTEGER)
            src_work   = clip.resize.Bicubic(format=target_fmt.id)
            needs_conv = True
        else:
            src_work   = clip
            needs_conv = False

    sup_src = mv.Super(src_work, pel=pel, hpad=blksize, vpad=blksize)
    _analyse = mv.Analyze if hasattr(mv, 'Analyze') else mv.Analyse
    bwd = _analyse(sup_src, isb=True,  blksize=blksize, overlap=blksize // 2)
    fwd = _analyse(sup_src, isb=False, blksize=blksize, overlap=blksize // 2)

    interp = mv.FlowInter(src_work, sup_src, bwd, fwd, time=50)

    if needs_conv and fmt_orig.id != interp.format.id:
        interp = interp.resize.Bicubic(
            format=fmt_orig.id, dither_type="error_diffusion"
        )
    return interp

def _build_of_clip_svp(clip: vs.VideoNode) -> vs.VideoNode:
    """
    Return a clip where every frame is an SVP motion-interpolated frame
    at the midpoint between frame[n-1] and frame[n+1].

    SVP uses adaptive block sizes and multi-level vector refinement,
    generally producing better results than basic MVTools on live-action.
    Requires core.svp1 and core.svp2.

    SVP's SmoothFps outputs at 2× the source rate.  We double the fps,
    then SelectEvery(2, [1]) to grab only the interpolated odd frames
    (the t=0.5 midpoints), then restore the original fps.
    """
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

    super_params   = '{"pel":2,"gpu":0}'
    analyse_params = '{"block":{"w":32,"h":32},"main":{"levels":5}}'
    smooth_params  = '{"rate":{"num":2,"den":1,"abs":false},"algo":21,"mask":{"area":100}}'

    sup     = core.svp1.Super(src8, super_params)
    vecs    = core.svp1.Analyse(sup['clip'], sup['data'], src8, analyse_params)
    doubled = core.svp2.SmoothFps(
        src8, sup['clip'], sup['data'], vecs['clip'], vecs['data'], smooth_params
    )
    # SelectEvery(2, [1]) picks frames 1, 3, 5 … — the interpolated midpoints
    interp = doubled.std.SelectEvery(cycle=2, offsets=[1])

    if needs_conv:
        interp = interp.resize.Bicubic(
            format=fmt_orig.id, dither_type="error_diffusion"
        )
    return interp