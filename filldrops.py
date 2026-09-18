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


def fillWithGMFSSUnionM(clip, start, end, gmfssModel=0, sceneThresh=0.15):
    from vsgmfss_fortuna import gmfss_fortuna

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


def flickerFlag(clip, max_flash_frames=5, min_diff=0.05, return_ranges=False):
    """
    Detects short brightness flashes and marks the affected frames with the
    frame property "_UseInterp" (1 = frame should be replaced/interpolated,
    0 = keep). The clip itself is not modified.

    max_flash_frames: maximum length of a flash in frames.
    min_diff:         minimum luma difference (0..1) required to count as a flash.
    return_ranges:    also return the detected (start, end) ranges, so that
                      ReplaceFlagged does not have to rediscover them.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("flickerFlag: 'clip' must be a clip")
    if clip.format is None or clip.format.color_family not in (vs.YUV, vs.GRAY):
        raise vs.Error("flickerFlag: clip must be YUV or GRAY (convert RGB first)")

    num_frames = clip.num_frames

    # Calculate luma values once, at full resolution so detection stays exact.
    luma = _luma_values(clip)

    # Frames that should be replaced.
    use_interp = [False] * num_frames
    ranges = []

    n = 1
    while n < num_frames - 1:

        # A flash must start with a strong increase
        # compared to the previous frame.
        if luma[n] < luma[n - 1] + min_diff:
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
            if luma[end] < after + min_diff:
                continue

            # The surrounding frames define the normal brightness.
            baseline = max(before, after)

            # Every frame of the flash must be clearly above that baseline.
            if all(luma[i] >= baseline + min_diff for i in range(n, end + 1)):
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
# convenience wrapper
# ---------------------------------------------------------------------------

def FixFlicker(clip: vs.VideoNode, max_flash_frames: int = 1, min_diff: float = 0.05,
               method: str = "mv", rifeModel: int = 22, rifeTTA: bool = False,
               rifeUHD: bool = False, sceneThresh: float = 0.15, gmfssModel: int = 0,
               debug: bool = False) -> vs.VideoNode:
    """
    Detects short brightness flashes and replaces them by an interpolation
    between the frames surrounding the flash.

    max_flash_frames: maximum length of a flash in frames.
    min_diff:         minimum luma difference (0..1) required to count as a flash.
    method:           mv, svp, svp_gpu, rife or gmfssfortuna.
    debug:            label the replaced frames.

    The returned clip keeps the _UseInterp property, so the detection result
    stays inspectable downstream.
    """
    flagged, ranges = flickerFlag(clip,
                                  max_flash_frames=max_flash_frames,
                                  min_diff=min_diff,
                                  return_ranges=True)

    if not ranges:
        return flagged

    return ReplaceFlagged(flagged, ranges,
                          method=method,
                          rifeModel=rifeModel,
                          rifeTTA=rifeTTA,
                          rifeUHD=rifeUHD,
                          sceneThresh=sceneThresh,
                          gmfssModel=gmfssModel,
                          debug=debug)