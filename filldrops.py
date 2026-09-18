import importlib.util

import vapoursynth as vs
core = vs.core
from misc import MV, SCDetect


# ---------------------------------------------------------------------------
# internal helpers
# ---------------------------------------------------------------------------

def _check_yuv(clip, name):
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error(name + ': this is not a clip')
    if clip.format is None:
        raise vs.Error(name + ': variable format clips are not supported')
    if clip.format.color_family != vs.YUV:
        raise vs.Error(name + ': requires YUV input')


def _match(clip, interp):
    """Make an interpolated segment spliceable with the source clip."""
    if interp.format.id != clip.format.id or interp.width != clip.width or interp.height != clip.height:
        raise vs.Error('interpolated segment does not match the source clip')
    return core.std.AssumeFPS(interp, src=clip)


def _loop_single(clip, single):
    """Wrap a 1-frame interpolation result for use inside FrameEval.

    FrameEval only ever pulls frame n from the returned clip, so looping one
    frame is equivalent to splicing it back into the full clip - but it costs
    one node instead of four, and it is O(1) instead of O(clip length).
    """
    return core.std.AssumeFPS(core.std.Loop(single, clip.num_frames), src=clip)


# ---------------------------------------------------------------------------
# range interpolators: each returns exactly (end - start + 1) frames
# ---------------------------------------------------------------------------

def fillWithMVToolsM(clip, start, end):
    pair = clip[start - 1:start] + clip[end + 1:end + 2]
    super = MV.Super(pair, pel=2, blksize=8, overlap=0)
    vfe = MV.Analyse(super, truemotion=True, isb=False, delta=1)
    vbe = MV.Analyse(super, truemotion=True, isb=True, delta=1)

    count = end - start + 1
    clips = []

    for i in range(count):
        time = 100.0 * (i + 1) / (count + 1)
        clips.append(MV.FlowInter(pair, super, mvbw=vbe, mvfw=vfe, time=time)[:1])

    return core.std.Splice(clips)


def fillWithRIFEM(clip, start, end, rifeModel=22, rifeTTA=False, rifeUHD=False, sceneThresh=0.15):
    clip1 = core.std.AssumeFPS(clip, fpsnum=1, fpsden=1)
    pair = clip1[start - 1:start] + clip1[end + 1:end + 2]

    # Scene detection has to run on the pair RIFE actually sees, not on the
    # full clip - on the full clip the props describe a different comparison.
    if sceneThresh > 0:
        pair = SCDetect(clip=pair, threshold=sceneThresh)

    pair = core.resize.Bicubic(pair, format=vs.RGBS, matrix_in_s="709")

    count = end - start + 1
    r = core.rife.RIFE(pair, model=rifeModel, factor_num=count + 1, factor_den=1,
                       tta=rifeTTA, uhd=rifeUHD, sc=sceneThresh > 0)
    r = core.resize.Bicubic(r, format=clip.format, matrix_s="709")

    return r[1:count + 1]


def hasGMFSSfortuna():
    """True when the optional vsgmfss_fortuna package can be imported."""
    return importlib.util.find_spec("vsgmfss_fortuna") is not None


def _gmfss_fortuna():
    """Import vsgmfss_fortuna, with a clear message when it is not installed."""
    try:
        from vsgmfss_fortuna import gmfss_fortuna
    except ImportError as e:
        raise vs.Error("gmfssfortuna: vsgmfss_fortuna is not installed") from e
    return gmfss_fortuna


def fillWithGMFSSUnionM(clip, start, end, gmfssModel=0, sceneThresh=0.15):
    gmfss_fortuna = _gmfss_fortuna()

    clip1 = core.std.AssumeFPS(clip, fpsnum=1, fpsden=1)
    pair = clip1[start - 1:start] + clip1[end + 1:end + 2]
    pair = core.resize.Bicubic(pair, format=vs.RGBH, matrix_in_s="709")

    count = end - start + 1
    r = gmfss_fortuna(pair, model=gmfssModel, factor_num=count + 1, factor_den=1,
                      sc_threshold=sceneThresh)
    r = core.resize.Bicubic(r, format=clip.format, matrix_s="709")

    return r[1:count + 1]


def fillWithSVPM(clip, start, end, gpu=False):
    clip1 = core.std.AssumeFPS(clip, fpsnum=1, fpsden=1)
    pair = clip1[start - 1:start] + clip1[end + 1:end + 2]

    super = core.svp1.Super(pair, "{gpu:1}" if gpu else "{gpu:0}")
    vectors = core.svp1.Analyse(super["clip"], super["data"], pair, "{}")

    count = end - start + 1
    params = "{rate:{num:" + str(count + 1) + ",den:1}}"
    r = core.svp2.SmoothFps(pair, super["clip"], super["data"],
                            vectors["clip"], vectors["data"], params)

    return r[1:count + 1]


def _interp_range(clip, start, end, method, rifeModel=22, rifeTTA=False, rifeUHD=False,
                  sceneThresh=0.15, gmfssModel=0):
    if method == "mv":
        interp = fillWithMVToolsM(clip, start, end)
    elif method == "svp":
        interp = fillWithSVPM(clip, start, end)
    elif method == "svp_gpu":
        interp = fillWithSVPM(clip, start, end, gpu=True)
    elif method == "rife":
        interp = fillWithRIFEM(clip, start, end, rifeModel, rifeTTA, rifeUHD, sceneThresh)
    elif method == "gmfssfortuna":
        interp = fillWithGMFSSUnionM(clip, start, end, gmfssModel, sceneThresh)
    else:
        raise vs.Error('unknown method ' + str(method))

    count = end - start + 1
    if interp.num_frames != count:
        raise vs.Error('interpolation returned ' + str(interp.num_frames) +
                       ' frames, expected ' + str(count))

    return _match(clip, interp)


# ---------------------------------------------------------------------------
# single frame wrappers (full-length clips, for use inside FrameEval)
# ---------------------------------------------------------------------------

def fillWithMVTools(clip):
    super = MV.Super(clip, pel=2, blksize=8, overlap=0)
    vfe = MV.Analyse(super, truemotion=True, isb=False, delta=1)
    vbe = MV.Analyse(super, truemotion=True, isb=True, delta=1)
    return MV.FlowInter(clip, super, mvbw=vbe, mvfw=vfe, time=50)


def fillWithRIFE(clip, firstframe=None, rifeModel=22, rifeTTA=False, rifeUHD=False, sceneThresh=0.15):
    single = fillWithRIFEM(clip, firstframe, firstframe, rifeModel, rifeTTA, rifeUHD, sceneThresh)
    return _loop_single(clip, single)


def fillWithGMFSSUnion(clip, firstframe=None, gmfssModel=0, sceneThresh=0.15):
    single = fillWithGMFSSUnionM(clip, firstframe, firstframe, gmfssModel, sceneThresh)
    return _loop_single(clip, single)


def fillWithSVP(clip, firstframe=None, gpu=False, endframe=None):
    if endframe is None:
        single = fillWithSVPM(clip, firstframe, firstframe, gpu=gpu)
        return _loop_single(clip, single)

    if firstframe < 1 or endframe >= clip.num_frames - 1 or firstframe > endframe:
        raise vs.Error("fillWithSVP: invalid interpolation range")

    return fillWithSVPM(clip, firstframe, endframe, gpu=gpu)


# ---------------------------------------------------------------------------
# FillSingleDrops / InsertSingle / ReplaceSingle
# ---------------------------------------------------------------------------

def FillSingleDrops(clip, thresh=0.3, method="mv", rifeModel=22, rifeTTA=False, rifeUHD=False,
                    sceneThresh=0.15, gmfssModel=0, debug=False):
    _check_yuv(clip, 'FillSingleDrops')

    last = clip.num_frames - 1

    # Built once, not per frame: this graph does not depend on n.
    static_fill = fillWithMVTools(clip) if method == "mv" else None

    def selectFunc(n, f):
        if f.props['PlaneStatsDiff'] > thresh or n == 0 or n >= last:
            if debug:
                return core.text.Text(clip=clip, text="Org, diff: " + str(f.props['PlaneStatsDiff']), alignment=8)
            return clip

        if static_fill is not None:
            filldrops = static_fill
        elif method == "svp":
            filldrops = fillWithSVP(clip, n)
        elif method == "svp_gpu":
            filldrops = fillWithSVP(clip, n, gpu=True)
        elif method == "rife":
            filldrops = fillWithRIFE(clip, n, rifeModel, rifeTTA, rifeUHD, sceneThresh)
        elif method == "gmfssfortuna":
            filldrops = fillWithGMFSSUnion(clip, n, gmfssModel, sceneThresh)
        else:
            raise vs.Error('FillSingleDrops: Unknown method ' + str(method))

        if debug:
            return core.text.Text(clip=filldrops, text=method + ", diff: " + str(f.props['PlaneStatsDiff']), alignment=8)
        return filldrops

    diffclip = core.std.PlaneStats(clip, clip[0] + clip)
    return core.std.FrameEval(clip, selectFunc, prop_src=diffclip)


def InsertSingle(clip, afterEveryX=2, method="mv", rifeModel=0, rifeTTA=False, rifeUHD=False,
                 sceneThresh=0.15, gmfssModel=0, debug=False):
    _check_yuv(clip, 'InsertSingle')

    static_fill = fillWithMVTools(clip) if method == "mv" else None
    if static_fill is not None:
        static_fill = core.text.Text(static_fill, text="Interpolated", alignment=8)

    def selectFunc(n):
        if n == 0 or n % afterEveryX != 0:
            return clip

        if static_fill is not None:
            return static_fill

        if method == "svp":
            insertFrame = fillWithSVP(clip, n)
        elif method == "svp_gpu":
            insertFrame = fillWithSVP(clip, n, gpu=True)
        elif method == "rife":
            insertFrame = fillWithRIFE(clip, n, rifeModel, rifeTTA, rifeUHD, sceneThresh)
        elif method == "gmfssfortuna":
            insertFrame = fillWithGMFSSUnion(clip, n, gmfssModel, sceneThresh)
        else:
            raise vs.Error('InsertSingle: Unknown method ' + str(method))

        return core.text.Text(insertFrame, text="Interpolated", alignment=8)

    return core.std.FrameEval(clip, selectFunc)


def ReplaceSingle(clip, frameList, method="mv", rifeModel=0, rifeTTA=False, rifeUHD=False,
                  sceneThresh=0.15, gmfssModel=0, debug=False):
    _check_yuv(clip, 'ReplaceSingle')

    # Set lookup instead of a linear scan per frame.
    frameSet = frozenset(frameList)

    static_fill = fillWithMVTools(clip) if method == "mv" else None

    def selectFunc(n):
        if n in frameSet:
            if static_fill is not None:
                insertFrame = static_fill
            elif method == "svp":
                insertFrame = fillWithSVP(clip, n)
            elif method == "svp_gpu":
                insertFrame = fillWithSVP(clip, n, gpu=True)
            elif method == "rife":
                insertFrame = fillWithRIFE(clip, n, rifeModel, rifeTTA, rifeUHD, sceneThresh)
            elif method == "gmfssfortuna":
                insertFrame = fillWithGMFSSUnion(clip, n, gmfssModel, sceneThresh)
            else:
                raise vs.Error('ReplaceSingle: Unknown method ' + str(method))

            if debug:
                return core.text.Text(clip=insertFrame, text=method + ", replaced frame: " + str(n), alignment=8)
            return insertFrame

        if debug:
            return core.text.Text(clip=clip, text="kept frame: " + str(n), alignment=8)
        return clip

    return core.std.FrameEval(clip, selectFunc)


# ---------------------------------------------------------------------------
# flicker detection
# ---------------------------------------------------------------------------

def _brighter(min_diff):
    """Is a frame clearly brighter than a reference frame?

    min_diff = 0 is the plain local maximum test - brighter than the
    neighbours, by any amount. A larger value asks for that much margin on
    top, which is what separates a flash from ordinary luma noise.
    """
    if min_diff > 0:
        return lambda value, reference: value >= reference + min_diff
    return lambda value, reference: value > reference


def _luma_values(clip):
    """Average luma per frame, decoded with VapourSynth's own prefetching."""
    stats = core.std.PlaneStats(clip, plane=0)
    return [f.props["PlaneStatsAverage"] for f in stats.frames(close=True)]


def _apply_flags(clip, use_interp):
    """Attach _UseInterp without touching pixel data.

    ModifyFrame would copy every frame just to set one integer prop; splicing
    two pre-tagged versions costs nothing per frame.
    """
    num_frames = len(use_interp)
    if num_frames == 0:
        return clip

    off = core.std.SetFrameProp(clip, prop="_UseInterp", intval=0)
    on = core.std.SetFrameProp(clip, prop="_UseInterp", intval=1)

    segments = []
    start = 0
    for i in range(1, num_frames + 1):
        if i == num_frames or use_interp[i] != use_interp[start]:
            segments.append((on if use_interp[start] else off)[start:i])
            start = i

    return segments[0] if len(segments) == 1 else core.std.Splice(segments)


def _ranges_from_props(clip):
    """Read _UseInterp back off a clip in one prefetched pass."""
    ranges = []
    start = None

    for n, f in enumerate(clip.frames(close=True)):
        if bool(f.props.get("_UseInterp", 0)):
            if start is None:
                start = n
        elif start is not None:
            ranges.append((start, n - 1))
            start = None

    if start is not None:
        ranges.append((start, clip.num_frames - 1))

    return ranges


def flickerFlag(clip, max_flash_frames=5, min_diff=0.0, return_ranges=False, luma=None):
    """
    Detects short brightness flashes and marks the affected frames with the
    frame property "_UseInterp" (1 = frame should be replaced/interpolated,
    0 = keep). The clip itself is not modified.

    max_flash_frames: maximum length of a flash in frames.
    min_diff:         luma margin (0..1) a frame needs over its neighbours to
                      count as a flash. 0 takes every local maximum.
    return_ranges:    also return the detected (start, end) ranges, so that
                      ReplaceFlagged does not have to rediscover them.
    luma:             per frame luma from an earlier _luma_values() run, to
                      save a second pass over the clip.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("flickerFlag: 'clip' must be a clip")
    if clip.format is None or clip.format.color_family not in (vs.YUV, vs.GRAY):
        raise vs.Error("flickerFlag: clip must be YUV or GRAY (convert RGB first)")

    num_frames = clip.num_frames

    # Calculate luma values once, at full resolution so detection stays exact.
    if luma is None:
        luma = _luma_values(clip)

    # Frames that should be replaced.
    use_interp = [False] * num_frames
    ranges = []

    brighter = _brighter(min_diff)

    n = 1
    while n < num_frames - 1:

        # A flash must start with a strong increase
        # compared to the previous frame.
        if not brighter(luma[n], luma[n - 1]):
            n += 1
            continue

        found = False

        # Only look for short flashes.
        for length in range(1, max_flash_frames + 1):

            end = n + length - 1

            # Need one frame after the flash as reference.
            if end >= num_frames - 1:
                break

            before = luma[n - 1]
            after = luma[end + 1]

            # The last flash frame must also be clearly brighter
            # than the frame after the flash.
            if not brighter(luma[end], after):
                continue

            # The surrounding frames define the normal brightness.
            baseline = max(before, after)

            # Every frame of the flash must be clearly above that baseline.
            if all(brighter(luma[i], baseline) for i in range(n, end + 1)):
                for i in range(n, end + 1):
                    use_interp[i] = True

                ranges.append((n, end))
                found = True

                # Continue after the detected flash.
                n = end + 1
                break

        if not found:
            n += 1

    flagged = _apply_flags(clip, use_interp)

    return (flagged, ranges) if return_ranges else flagged


# ---------------------------------------------------------------------------
# ReplaceFlagged
# ---------------------------------------------------------------------------

def ReplaceFlagged(clip, ranges=None, method="mv", rifeModel=22, rifeTTA=False, rifeUHD=False,
                   sceneThresh=0.15, gmfssModel=0, debug=False):
    """
    Replaces every run of frames flagged with _UseInterp by an interpolation
    between the frames surrounding the run.

    ranges: optional list of (start, end) tuples, e.g. from
            flickerFlag(..., return_ranges=True). If omitted, the flags are
            read off the clip in a single prefetched pass.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('ReplaceFlagged: this is not a clip')

    if ranges is None:
        ranges = _ranges_from_props(clip)

    num_frames = clip.num_frames
    segments = []
    pos = 0

    for start, end in sorted(ranges):
        # Runs touching the clip borders have no reference frame on one side.
        if start < 1 or end >= num_frames - 1 or start > end or start < pos:
            continue

        interp = _interp_range(clip, start, end, method, rifeModel, rifeTTA, rifeUHD,
                               sceneThresh, gmfssModel)

        # The segment is built from the frames around the run, so it inherits
        # their _UseInterp=0 - retag it as replaced.
        interp = core.std.SetFrameProp(interp, prop="_UseInterp", intval=1)

        if debug:
            interp = core.text.Text(interp, text="INTERPOLATED " + str(start) + "-" + str(end), alignment=8)

        if start > pos:
            segments.append(clip[pos:start])
        segments.append(interp)
        pos = end + 1

    if pos < num_frames:
        segments.append(clip[pos:])

    if not segments:
        return clip

    return segments[0] if len(segments) == 1 else core.std.Splice(segments, mismatch=False)


# ---------------------------------------------------------------------------
# lazy detection: no full decode while the graph is built
# ---------------------------------------------------------------------------

def _shift(clip, k):
    """clip moved by k frames, same length, edges clamped."""
    if k == 0:
        return clip
    num_frames = clip.num_frames
    if k > 0:
        return clip[k:] + clip[num_frames - 1:] * k
    return clip[:1] * (-k) + clip[:num_frames + k]


def _flash_range_at(luma, n, num_frames, max_flash_frames, brighter, backoff):
    """The flash range containing frame n, or None.

    Runs the same greedy scan as flickerFlag(), but starts at n - backoff
    instead of at frame 1. That gives the same answer because two flashes can
    never touch: a flash starts on a rise from the preceding frame and ends on
    a drop to the following one, so at least one normal frame separates them
    and a locally started scan resynchronises on it. That holds for the plain
    local maximum test as well, a frame cannot be brighter and darker than the
    same neighbour.
    """
    m = max(1, n - backoff)

    while m < num_frames - 1 and m <= n:

        if not brighter(luma(m), luma(m - 1)):
            m += 1
            continue

        found = False

        for length in range(1, max_flash_frames + 1):

            end = m + length - 1

            if end >= num_frames - 1:
                break

            if not brighter(luma(end), luma(end + 1)):
                continue

            baseline = max(luma(m - 1), luma(end + 1))

            if all(brighter(luma(i), baseline) for i in range(m, end + 1)):
                if m <= n <= end:
                    return m, end
                found = True
                m = end + 1
                break

        if not found:
            m += 1

    return None


def _luma_label(n, value, previous):
    """Debug line: frame number, its luma and the step from the frame before.

    The step is what min_diff is compared against, so both numbers together
    say why a frame was taken for a flash or passed over.
    """
    return "n %d  luma %.4f  d %+.4f" % (n, value, value - previous)


def _label_luma(clip, luma):
    """Write the per frame luma under an already built clip."""
    def selectFunc(n):
        return core.text.Text(clip, text=_luma_label(n, luma[n], luma[n - 1] if n else luma[n]),
                              alignment=2)

    return core.std.FrameEval(clip, selectFunc)


def _fixFlickerLazy(clip, max_flash_frames, min_diff, method, rifeModel, rifeTTA,
                    rifeUHD, sceneThresh, gmfssModel, debug):
    """FixFlicker without the analysis pass, see FixFlicker(lazy=True)."""
    brighter = _brighter(min_diff)
    backoff = 2 * max_flash_frames + 2
    lo, hi = -(backoff + 1), max_flash_frames + 1
    num_frames = clip.num_frames

    # One PlaneStats node, read through shifted views: frame n of view k
    # carries the luma of frame n + k, so the whole window is available
    # without ever touching a frame outside it.
    stats = core.std.PlaneStats(clip, plane=0)
    prop_src = [_shift(stats, k) for k in range(lo, hi + 1)]
    zero = -lo

    off = core.std.SetFrameProp(clip, prop="_UseInterp", intval=0)
    segments = {}

    def selectFunc(n, f):
        window = {n + (i - zero): fr.props["PlaneStatsAverage"] for i, fr in enumerate(f)}

        def luma(i):
            return window[min(max(i, n + lo), n + hi)]

        found = _flash_range_at(luma, n, num_frames, max_flash_frames, brighter, backoff)

        if found is None:
            out = off
        else:
            start, end = found
            interp = segments.get(found)

            # Built once per range, not once per frame: with RIFE the range
            # interpolation computes every intermediate frame anyway.
            if interp is None:
                interp = _interp_range(clip, start, end, method, rifeModel, rifeTTA, rifeUHD,
                                       sceneThresh, gmfssModel)
                interp = core.std.SetFrameProp(interp, prop="_UseInterp", intval=1)
                if debug:
                    interp = core.text.Text(interp, text="INTERPOLATED " + str(start) + "-" + str(end),
                                            alignment=8)
                segments[found] = interp

            out = _loop_single(clip, interp[n - start:n - start + 1])

        if debug:
            out = core.text.Text(out, text=_luma_label(n, luma(n), luma(n - 1)), alignment=2)

        return out

    return core.std.FrameEval(clip, selectFunc, prop_src=prop_src)


# ---------------------------------------------------------------------------
# convenience wrapper
# ---------------------------------------------------------------------------

def FixFlicker(clip: vs.VideoNode, max_flash_frames: int = 1, min_diff: float = 0.0,
               method: str = "mv", rifeModel: int = 22, rifeTTA: bool = False,
               rifeUHD: bool = False, sceneThresh: float = 0.15, gmfssModel: int = 0,
               debug: bool = False, lazy: bool = False) -> vs.VideoNode:
    """
    Detects short brightness flashes and replaces them by an interpolation
    between the frames surrounding the flash.

    max_flash_frames: maximum length of a flash in frames.
    min_diff:         luma margin (0..1) a frame needs over its neighbours to
                      count as a flash. 0 takes every local maximum.
    method:           mv, svp, svp_gpu, rife or gmfssfortuna.
    debug:            label the replaced frames.
    lazy:             decide per frame instead of scanning the clip up front.
                      Nothing is decoded while the graph is built, which is what
                      a preview wants; the detected flashes are the same.

    The returned clip keeps the _UseInterp property, so the detection result
    stays inspectable downstream.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("FixFlicker: 'clip' must be a clip")
    if clip.format is None or clip.format.color_family not in (vs.YUV, vs.GRAY):
        raise vs.Error('FixFlicker: clip must be YUV or GRAY (convert RGB first)')
    if max_flash_frames < 1:
        raise vs.Error('FixFlicker: max_flash_frames must be at least 1')

    if lazy:
        return _fixFlickerLazy(clip, max_flash_frames, min_diff, method, rifeModel,
                               rifeTTA, rifeUHD, sceneThresh, gmfssModel, debug)

    # Needed twice with debug on - for the detection and for the labels.
    luma = _luma_values(clip) if debug else None

    flagged, ranges = flickerFlag(clip,
                                  max_flash_frames=max_flash_frames,
                                  min_diff=min_diff,
                                  return_ranges=True,
                                  luma=luma)

    if not ranges:
        return _label_luma(flagged, luma) if debug else flagged

    out = ReplaceFlagged(flagged, ranges,
                         method=method,
                         rifeModel=rifeModel,
                         rifeTTA=rifeTTA,
                         rifeUHD=rifeUHD,
                         sceneThresh=sceneThresh,
                         gmfssModel=gmfssModel,
                         debug=debug)

    return _label_luma(out, luma) if debug else out