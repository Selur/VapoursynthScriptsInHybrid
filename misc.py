from vapoursynth import core
import vapoursynth as vs

from typing import Optional, Union, Sequence, List

def Overlay(
    base: vs.VideoNode,
    overlay: vs.VideoNode,
    x: int = 0,
    y: int = 0,
    mask: Optional[vs.VideoNode] = None,
    opacity: float = 1.0,
    mode: str = 'normal',
    planes: Optional[Union[int, Sequence[int]]] = None,
    mask_first_plane: bool = True,
) -> vs.VideoNode:
    '''
    Puts clip overlay on top of clip base using different blend modes, and with optional x,y positioning, masking and opacity.

    Parameters:
        base: This clip will be the base, determining the size and all other video properties of the result.

        overlay: This is the image that will be placed on top of the base clip.

        x, y: Define the placement of the overlay image on the base clip, in pixels. Can be positive or negative.

        mask: Optional transparency mask. Must be the same size as overlay. Where mask is darker, overlay will be more transparent.

        opacity: Set overlay transparency. The value is from 0.0 to 1.0, where 0.0 is transparent and 1.0 is fully opaque.
            This value is multiplied by mask luminance to form the final opacity.

        mode: Defines how your overlay should be blended with your base image. Available blend modes are:
            addition, average, burn, darken, difference, divide, dodge, exclusion, extremity, freeze, glow, grainextract, grainmerge, hardlight, hardmix, heat,
            lighten, linearlight, multiply, negation, normal, overlay, phoenix, pinlight, reflect, screen, softlight, subtract, vividlight

        planes: Specifies which planes will be processed. Any unprocessed planes will be simply copied.

        mask_first_plane: If true, only the mask's first plane will be used for transparency.
    '''
    if not (isinstance(base, vs.VideoNode) and isinstance(overlay, vs.VideoNode)):
        raise vs.Error('Overlay: this is not a clip')

    if mask is not None:
        if not isinstance(mask, vs.VideoNode):
            raise vs.Error('Overlay: mask is not a clip')

        if mask.width != overlay.width or mask.height != overlay.height or mask.format.bits_per_sample != overlay.format.bits_per_sample :
            raise vs.Error('Overlay: mask must have the same dimensions and bit depth as overlay')

    if base.format.sample_type == vs.INTEGER:
        bits = base.format.bits_per_sample 
        neutral = 1 << (bits - 1)
        peak = (1 << bits) - 1
        factor = 1 << bits
    else:
        neutral = 0.5
        peak = factor = 1.0

    plane_range = range(base.format.num_planes)

    if planes is None:
        planes = list(plane_range)
    elif isinstance(planes, int):
        planes = [planes]

    if base.format.subsampling_w > 0 or base.format.subsampling_h > 0:
        base_orig = base
        base = base.resize.Point(format=base.format.replace(subsampling_w=0, subsampling_h=0))
    else:
        base_orig = None

    if overlay.format.id != base.format.id:
        overlay = overlay.resize.Point(format=base.format)

    if mask is None:
        mask = overlay.std.BlankClip(format=overlay.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0), color=peak)
    elif mask.format.id != overlay.format.id and mask.format.color_family != vs.GRAY:
        mask = mask.resize.Point(format=overlay.format, range_s='full')

    opacity = min(max(opacity, 0.0), 1.0)
    mode = mode.lower()

    # Calculate padding sizes
    l, r = x, base.width - overlay.width - x
    t, b = y, base.height - overlay.height - y

    # Split into crop and padding values
    cl, pl = min(l, 0) * -1, max(l, 0)
    cr, pr = min(r, 0) * -1, max(r, 0)
    ct, pt = min(t, 0) * -1, max(t, 0)
    cb, pb = min(b, 0) * -1, max(b, 0)

    # Crop and padding
    overlay = overlay.std.Crop(left=cl, right=cr, top=ct, bottom=cb)
    overlay = overlay.std.AddBorders(left=pl, right=pr, top=pt, bottom=pb)
    mask = mask.std.Crop(left=cl, right=cr, top=ct, bottom=cb)
    mask = mask.std.AddBorders(left=pl, right=pr, top=pt, bottom=pb, color=[0] * mask.format.num_planes)
    EXPR = core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr
    if opacity < 1:
        mask = EXPR(mask, expr=f'x {opacity} *')

    if mode == 'normal':
        pass
    elif mode == 'addition':
        expr = f'x y +'
    elif mode == "linearadd":
        expr = f'x 2 pow y 2 pow + sqrt' # same as Gimp's 2.10 subtract blending mode    
    elif mode == 'average':
        expr = f'x y + 2 /'
    elif mode == 'burn':
        expr = f'x 0 <= x {peak} {peak} y - {factor} * x / - ?'
    elif mode == 'darken':
        expr = f'x y min'
    elif mode == 'difference':
        expr = f'x y - abs'
    elif mode == 'divide':
        expr = f'y 0 <= {peak} {peak} x * y / ?'
    elif mode == 'dodge':
        expr = f'x {peak} >= x y {factor} * {peak} x - / ?'
    elif mode == 'exclusion':
        expr = f'x y + 2 x * y * {peak} / -'
    elif mode == 'extremity':
        expr = f'{peak} x - y - abs'
    elif mode == 'freeze':
        expr = f'y 0 <= 0 {peak} {peak} x - dup * y / {peak} min - ?'
    elif mode == 'glow':
        expr = f'x {peak} >= x y y * {peak} x - / ?'
    elif mode == 'grainextract':
        expr = f'x y - {neutral} +'
    elif mode == 'grainmerge':
        expr = f'x y + {neutral} -'
    elif mode == 'hardlight':
        expr = f'y {neutral} < 2 y x * {peak} / * {peak} 2 {peak} y - {peak} x - * {peak} / * - ?'
    elif mode == 'hardmix':
        expr = f'x {peak} y - < 0 {peak} ?'
    elif mode == 'heat':
        expr = f'x 0 <= 0 {peak} {peak} y - dup * x / {peak} min - ?'
    elif mode == 'lighten':
        expr = f'x y max'
    elif mode == 'linearlight':
        expr = f'y {neutral} < y 2 x * + {peak} - y 2 x {neutral} - * + ?'
    elif mode == 'multiply':
        expr = f'x y * {peak} /'
    elif mode == 'negation':
        expr = f'{peak} {peak} x - y - abs -'
    elif mode == 'overlay':
        expr = f'x {neutral} < 2 x y * {peak} / * {peak} 2 {peak} x - {peak} y - * {peak} / * - ?'
    elif mode == 'phoenix':
        expr = f'x y min x y max - {peak} +'
    elif mode == 'pinlight':
        expr = f'y {neutral} < x 2 y * min x 2 y {neutral} - * max ?'
    elif mode == 'reflect':
        expr = f'y {peak} >= y x x * {peak} y - / ?'
    elif mode == 'screen':
        expr = f'{peak} {peak} x - {peak} y - * {peak} / -'
    elif mode == 'softlight':
        expr = f'x {neutral} > y {peak} y - x {neutral} - * {neutral} / 0.5 y {neutral} - abs {peak} / - * + y y {neutral} x - {neutral} / * 0.5 y {neutral} - abs {peak} / - * - ?'
    elif mode == 'subtract':
        expr = f'x y -'
    elif mode == "linearsubtract":
        expr = f'x 2 pow y 2 pow - sqrt' # same as Gimp's 2.10 subtract blending mode
    elif mode == 'vividlight':
        expr = f'x {neutral} < x 0 <= 2 x * {peak} {peak} y - {factor} * 2 x * / - ? 2 x {neutral} - * {peak} >= 2 x {neutral} - * y {factor} * {peak} 2 x {neutral} - * - / ? ?'
    else:
        raise vs.Error('Overlay: invalid mode specified')

    if mode != 'normal':
        overlay = EXPR([overlay, base], expr=[expr if i in planes else '' for i in plane_range])

    # Return padded clip
    last = core.std.MaskedMerge(base, overlay, mask, planes=planes, first_plane=mask_first_plane)
    if base_orig is not None:
        last = last.resize.Point(format=base_orig.format)
    return last

def ShiftLinesHorizontally(clip: vs.VideoNode, shift: int, ymin: int, ymax: int) -> vs.VideoNode:
    # Validate clip format and subsampling
    if clip.format.color_family != vs.YUV or clip.format.subsampling_w != 0 or clip.format.subsampling_h != 0:
        raise ValueError("ShiftLinesHorizontalRange: only YUV444 input is supported.")
    
    # Ensure ymin and ymax are within valid range
    if ymin < 0 or ymin >= clip.height:
        raise ValueError(f"ShiftLinesHorizontalRange: ymin ({ymin}) is out of range.")
    if ymax < ymin or ymax >= clip.height:
        raise ValueError(f"ShiftLinesHorizontalRange: ymax ({ymax}) is out of range.")
    
    # If no shift is needed, return original clip
    if shift == 0:
        return clip

    width = clip.width
    height = clip.height

    # Create shifted version of just the target lines
    mid = clip.std.CropAbs(width=width, height=ymax-ymin+1, left=0, top=ymin)
    black = core.std.BlankClip(mid, width=abs(shift), height=mid.height, color=[0, 128, 128])
    
    if shift > 0:
        shifted_mid = core.std.StackHorizontal([black, mid.std.CropRel(right=shift)])
    else:
        shifted_mid = core.std.StackHorizontal([mid.std.CropRel(left=-shift), black])
    
    shifted_mid = shifted_mid.resize.Point(width=width, height=mid.height)

    # Build the output clip by stacking:
    # 1. Lines above ymin (unchanged)
    # 2. Shifted lines (ymin to ymax)
    # 3. Lines below ymax (unchanged)
    parts = []
    
    if ymin > 0:
        parts.append(clip.std.CropAbs(width=width, height=ymin, left=0, top=0))
    
    parts.append(shifted_mid)
    
    if ymax < height - 1:
        parts.append(clip.std.CropAbs(width=width, height=height-ymax-1, left=0, top=ymax+1))
    
    return core.std.StackVertical(parts)

def SCDetect(clip: vs.VideoNode, threshold: float = 0.1, plane: int = 0) -> vs.VideoNode:
    """
    Scene change detection with _SceneChangePrev/_SceneChangeNext frame properties.
    Uses core.misc.SCDetect or core.scd.Detect if available (plane=0 only), otherwise falls back to
    a std.PlaneStats-based reimplementation.

    Args:
        clip      : Input clip
        threshold : Scene change threshold (default: 0.1, must be 0.0–1.0)
        plane     : Plane to analyze; only honoured in fallback path —
                    misc.SCDetect always uses plane 0

    Returns:
        Clip with _SceneChangePrev and _SceneChangeNext frame properties set.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('SCDetect: this is not a clip')
    if not (0.0 <= threshold <= 1.0):
        raise vs.Error('SCDetect: threshold must be between 0.0 and 1.0')
    if clip.num_frames < 2:
        raise vs.Error('SCDetect: clip must have more than one frame')

    if hasattr(core,'scd'):
      if clip.format.color_family == vs.RGB:
            sc = clip.resize.Point(format=vs.GRAY8, matrix_s='709')
            sc = core.misc.SCDetect(sc, threshold=threshold)

            def _copy_props(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
                fout = f[0].copy()
                fout.props['_SceneChangePrev'] = f[1].props['_SceneChangePrev']
                fout.props['_SceneChangeNext'] = f[1].props['_SceneChangeNext']
                return fout

            return clip.std.ModifyFrame(clips=[clip, sc], selector=_copy_props)

      return core.scd.Detect(clip, thresh=threshold)
    elif hasattr(core, 'misc') and plane == 0:
      if clip.format.color_family == vs.RGB:
        sc = clip.resize.Point(format=vs.GRAY8, matrix_s='709')
        sc = core.misc.SCDetect(sc, threshold=threshold)

        def _copy_props(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
          fout = f[0].copy()
          fout.props['_SceneChangePrev'] = f[1].props['_SceneChangePrev']
          fout.props['_SceneChangeNext'] = f[1].props['_SceneChangeNext']
          return fout

        return clip.std.ModifyFrame(clips=[clip, sc], selector=_copy_props)

      return core.misc.SCDetect(clip, threshold=threshold)

    # prev_stats[n] = diff(frame_{n-1}, frame_n)  → SceneChangePrev
    # next_stats[n] = diff(frame_n,     frame_{n+1}) → SceneChangeNext
    prev_shifted = clip.std.DuplicateFrames(0).std.Trim(last=clip.num_frames - 1)
    prev_stats = core.std.PlaneStats(prev_shifted, clip, plane=plane)
    next_stats = core.std.PlaneStats(clip, clip.std.Trim(first=1), plane=plane)

    def _set_sc_props(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        fout = f[0].copy()
        fout.props['_SceneChangePrev'] = int(float(f[1].props.get('PlaneStatsDiff', 0.0)) > threshold)
        fout.props['_SceneChangeNext'] = int(float(f[2].props.get('PlaneStatsDiff', 0.0)) > threshold)
        return fout

    return clip.std.ModifyFrame(
        clips=[clip, prev_stats, next_stats],
        selector=_set_sc_props
    )

def scene_aware(
    clip: vs.VideoNode,
    filter_func,
    sc_threshold: float = 0.1,
    min_scene_len: int = 5,
    color_matrix: str = "709",
    **filter_kwargs
) -> vs.VideoNode:
    """
    Automatically split a clip by scene changes and apply a filter separately per scene.
    """

    if not isinstance(clip, vs.VideoNode):
        raise TypeError("scene_aware: 'clip' must be a VideoNode")

    # --- SCDetect: clip must be constant format and of integer 8-16 bit type or 32 bit float
    sc_src = clip
    if clip.format.color_family == vs.RGB:
        sc_src = core.resize.Bicubic(clip, format=vs.YUV420P8, matrix_s=color_matrix)  # convert to YUV8 for SCDetect
    elif clip.format.sample_type == vs.FLOAT and clip.format.bits_per_sample != 32:
        sc_src = core.resize.Bicubic(clip, format=vs.YUV420P8)

    if hasattr(core,'scd'):
      sc = core.scd.Detect(sc_src, thresh=sc_threshold)
    elif hasattr(core,'misc'):
      sc = core.misc.SCDetect(sc_src, threshold=sc_threshold)
    else: 
      sc = SCDetect(sc_src, threshold=sc_threshold)
    sc_frames = [i for i in range(clip.num_frames) if sc.get_frame(i).props._SceneChangePrev == 1]

    # --- Remove very short segments
    clean_frames = []
    prev = 0
    for f in sc_frames:
        if f - prev >= min_scene_len:
            clean_frames.append(f)
            prev = f
    sc_frames = clean_frames

    # --- Build scene ranges
    start = 0
    ranges = []
    for f in sc_frames:
        ranges.append((start, f - 1))
        start = f
    ranges.append((start, clip.num_frames - 1))

    # --- Apply filter per scene
    processed_segments = []
    for i, (s, e) in enumerate(ranges):
        sub = clip[s:e+1]
        out = filter_func(sub, **filter_kwargs)
        processed_segments.append(out)

    # --- Join them back
    result = core.std.Splice(processed_segments)
    return result

# define ShowFramesAround here (correct FrameEval usage)
def ShowFramesAround(src_clip: vs.VideoNode, count: int = 3) -> vs.VideoNode:
    """
    Return a clip that shows `count` consecutive frames horizontally,
    centered around each frame of the source. `count` must be odd.
    This implementation builds a stacked TEMPLATE clip and uses FrameEval
    on that template so returned frames match expected dimensions.
    """
    if count < 1 or (count % 2) == 0:
        raise ValueError("ShowFramesAround: count must be an odd integer >= 1")

    radius = count // 2
    last = src_clip.num_frames - 1

    # Build a template whose frames already have the stacked (wider) dimensions:
    # stack `count` copies of the source clip horizontally. The template has the same
    # number of frames as src_clip and the desired output dimensions.
    template = core.std.StackHorizontal([src_clip] * count)

    # callback for FrameEval: given frame index n, gather frames around n from src_clip
    def _select(n: int) -> vs.VideoNode:
        frames = []
        for offset in range(-radius, radius + 1):
            i = n + offset
            # clamp to first/last frame
            if i < 0:
                i = 0
            elif i > last:
                i = last
            # index into the original source clip to get the requested frame
            frames.append(src_clip[i])
        # stack the selected frames horizontally; dimensions match the template
        return core.std.StackHorizontal(frames)

    # Evaluate on the template so FrameEval sees matching dimensions
    return core.std.FrameEval(template, _select)


def AddVerticalLines(clip: vs.VideoNode, interval_ms: int = 10, color: float = 1.0) -> vs.VideoNode:
    """
    Draw vertical lines every `interval_ms` milliseconds.
    """
    width, height = clip.width, clip.height
    fps = clip.fps_num / clip.fps_den
    pixels_per_interval = max(1, int(width * interval_ms / (1000 / fps)))

    # start with a blank clip of the same size as clip
    lines_clip = core.std.BlankClip(
        clip=clip,
        color=0.0,
        width=width,
        height=height
    )

    for x in range(0, width, pixels_per_interval):
        # create a single-pixel-wide vertical line
        line = core.std.BlankClip(clip=clip, color=color, width=1, height=height)
        # shift it into position
        left = x
        right = width - x - 1
        line = core.std.AddBorders(line, left=left, right=right, top=0, bottom=0)
        # overlay this line on lines_clip
        lines_clip = core.std.MergeDiff(lines_clip, line)

    # overlay the vertical lines on the original clip
    return core.std.MergeDiff(clip, lines_clip)
    
def DelayAudio(audio_clip: vs.AudioNode, delay_ms: float) -> vs.AudioNode:
    """
    Delay an AudioNode by delay_ms milliseconds.
    Positive delay_ms prepends silence (audio plays later),
    Negative delay_ms trims the start (audio plays earlier).
    """
    if delay_ms == 0:
        return audio_clip

    sr = audio_clip.sample_rate  # sample rate of the audio
    delay_samples = abs(int(sr * (delay_ms / 1000)))

    if delay_ms > 0: # prepend silence
        silence = core.std.BlankAudio(clip=audio_clip, length=delay_samples)
        return core.std.AudioSplice([silence, audio_clip])
    else: # negative delay → trim start
        return core.std.AudioTrim(audio_clip, first=delay_samples)

def AverageFrames(
    clip: vs.VideoNode, weights: Union[float, Sequence[float]], scenechange: Optional[float] = None, planes: Optional[Union[int, Sequence[int]]] = None
) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('AverageFrames: this is not a clip')

    if scenechange:
        clip = SCDetect(clip, threshold=scenechange)
    return clip.std.AverageFrames(weights=weights, scenechange=scenechange, planes=planes)
    
# =============================================================================
# Motion-vector plugin wrapper (mvtools / mvsf / mvutensils)
# =============================================================================
#
# Every other script in this repository that needs motion estimation/compensation
# calls into this wrapper instead of `core.mv.*` / `core.mvsf.*` directly. That gives
# us one place to optionally route calls to mvutensils (https://github.com/myrsloik/mvutensils,
# namespace `core.mvu`) while keeping every call site written in the familiar
# mvtools/mvsf argument style (isb=, delta=, hpad=, lambda_=, dct=, thsadc=, ...).
#
# Behaviour:
#   - If core.mvu is NOT loaded: calls are forwarded to core.mv / core.mvsf unchanged
#     (core.mvsf is used for float clips when available, exactly like the old code did).
#   - If core.mvu IS loaded (and not explicitly disabled): arguments are translated
#     to mvutensils' conventions per https://github.com/myrsloik/mvutensils#porting-from-mvtools
#     and the call is forwarded to core.mvu.
#
# Known lossy/approximate translations (documented inline at each call site below):
#   - Analyse/Recalculate: `dct` (0-10) only maps cleanly to mvu's boolean `satd`
#     for dct in {0, 5}; other dct modes have no mvutensils equivalent and are
#     approximated as satd=True for dct>=5, satd=False otherwise.
#   - Analyse: mvu removed the `truemotion` preset. When `lambda_`/`lsad`/`pnew` are
#     not explicitly given we fall back to mvu's own defaults (mvlambda=1000, lsad=400,
#     pnew=25), which match old truemotion=True except lsad (was 1200 under
#     truemotion=True, mvu always uses 400).
#   - Degrain family: `limit`/`limitc` (0-255 int, 255 == "off") are converted to
#     mvu's float `limit` (per-plane, inf == "off"), scaled to the clip's peak value.
#   - Mask: mvu splits `Mask(kind=0/1/2)` into three separate functions and drops the
#     `clip`/`ysc` arguments, returning a single grayscale plane instead of a
#     clip-shaped/UV-colored mask. Code relying on the old multi-plane mask shape
#     needs to be checked when switching a given call site over.
#   - thscd2: rescaled from the old 0-256 integer to mvu's 0-100 float percentage.
#
# Usage in other files:
#   from misc import MV
#   sup   = MV.Super(clip, hpad=16, vpad=16, pel=2, blksize=8, overlap=2)
#   bvec  = MV.Analyse(sup, blksize=8, overlap=2, isb=True,  delta=1)
#   fvec  = MV.Analyse(sup, blksize=8, overlap=2, isb=False, delta=1)
#   den   = MV.Degrain1(clip, sup, bvec, fvec, thsad=400)
#
# `blksize`/`overlap` are new, *optional* keyword-only additions to `Super()` (mvtools'
# Super never took them). They are required only when mvutensils is actually in use
# (mvutensils pads the super clip itself and needs to know the block geometry up
# front) and should be passed the same values used in the matching Analyse() call.


def has_mvutensils() -> bool:
    '''Returns True if the mvutensils plugin (core.mvu) is loaded.'''
    return hasattr(core, 'mvu')


def _mvu_scale_thscd2(thscd2: float) -> float:
    '''mvtools/mvsf thscd2 is a 0-256 int; mvutensils thscd2 is a 0-100 float percentage.'''
    return max(0.0, min(100.0, thscd2 * 100.0 / 256.0))


def _mvu_search_mode(search: int) -> int:
    '''mvtools search modes 0-7 -> mvutensils 0-5 (old modes 0/1 dropped, rest shifted by -2).'''
    if search in (0, 1):
        # No 1:1 equivalent (old logarithmic/one-time-search modes were dropped).
        # Fall back to mvutensils' closest remaining option (0 = logarithmic/diamond).
        return 0
    return max(0, min(5, search - 2))


def _mvu_rfilter(rfilter: int) -> int:
    '''mvtools rfilter 0-4 -> mvutensils rfilter 0-2 (old modes 1 and 3 dropped).'''
    return {0: 0, 1: 0, 2: 1, 3: 1, 4: 2}.get(rfilter, 1)


def _mvu_dct_to_satd(dct: int) -> bool:
    '''mvtools dct (0-10) -> mvutensils boolean satd. Only dct in {0, 5} map cleanly.'''
    return dct >= 5


def _mvu_plane_to_planes(plane: int, clip: vs.VideoNode) -> List[int]:
    '''mvtools DegrainN `plane` (0=Y,1=U,2=V,3=UV,4=YUV) -> mvutensils `planes` list.'''
    if plane == 4:
        return list(range(clip.format.num_planes))
    if plane == 3:
        return [1, 2]
    if plane in (0, 1, 2):
        return [plane]
    raise vs.Error(f'MV: unsupported plane value {plane!r}')


def _mvu_limit_to_float(limit: Optional[float], clip: vs.VideoNode) -> float:
    '''mvtools DegrainN `limit`/`limitc` (0-255 int, 255 = off) -> mvutensils float limit (inf = off).'''
    if limit is None or limit >= 255:
        return float('inf')
    if clip.format.sample_type == vs.FLOAT:
        return limit / 255.0
    peak = (1 << clip.format.bits_per_sample) - 1
    return limit * peak / 255.0


class MotionVectors:
    '''
    mvtools/mvsf-style wrapper that optionally routes to mvutensils (core.mvu).

    See the module-level comment above `has_mvutensils()` for the full rationale and
    list of approximate/lossy argument translations. Instantiate once per preference
    (the default `MV` singleton below auto-detects mvutensils) and call the familiar
    mvtools method names/argument names on it; the correct backend is picked
    automatically per call based on clip format and availability.
    '''

    def __init__(self, prefer_mvutensils: Optional[bool] = None):
        '''
        prefer_mvutensils:
            None  -> use mvutensils automatically whenever core.mvu is available (default).
            True  -> require mvutensils; raises if core.mvu isn't loaded.
            False -> always use legacy core.mv / core.mvsf, even if core.mvu is available.

        NOTE: availability is checked live on every call
        '''
        self._prefer_mvutensils = prefer_mvutensils

    @property
    def use_mvu(self) -> bool:
        if self._prefer_mvutensils is True:
            if not has_mvutensils():
                raise vs.Error('MotionVectors: mvutensils (core.mvu) was requested but is not loaded')
            return True
        if self._prefer_mvutensils is False:
            return False
        return has_mvutensils()
    # -- internal helpers ----------------------------------------------------

    def _legacy_ns(self, clip: vs.VideoNode):
        '''Picks core.mvsf for float clips (if available) or core.mv, exactly like the old code did.'''
        if clip.format.sample_type == vs.FLOAT and hasattr(core, 'mvsf'):
            return core.mvsf
        return core.mv

    def _legacy_analyse_func(self, ns):
        # Some mvsf builds expose "Analyze", others "Analyse"; mv always uses "Analyse".
        return getattr(ns, 'Analyse', None) or getattr(ns, 'Analyze')

    # -- Super -----------------------------------------------------------

    def Super(
        self,
        clip: vs.VideoNode,
        hpad: int = 8,
        vpad: int = 8,
        pel: int = 2,
        levels: int = 0,
        chroma: bool = True,
        sharp: int = 2,
        rfilter: int = 2,
        pelclip: Optional[vs.VideoNode] = None,
        *,
        blksize: Optional[int] = None,
        blksizev: Optional[int] = None,
        overlap: Optional[int] = None,
        overlapv: Optional[int] = None,
    ) -> vs.VideoNode:
        '''
        blksize/overlap are only required when mvutensils is in use (it pads the super
        clip itself and needs the block geometry up front); pass the same values you
        use in the matching Analyse() call. They are ignored by the legacy backend.
        '''
        if self.use_mvu:
            if blksize is None or overlap is None:
                raise vs.Error(
                    'MV.Super: mvutensils requires blksize and overlap to be passed '
                    '(use the same values as the matching Analyse() call)'
                )
            return core.mvu.Super(
                clip,
                blksize=[blksize, blksizev or blksize],
                overlap=[overlap, overlapv or overlap],
                pad=[max(1, hpad), max(1, vpad)], # pad must be positive
                pel=pel,
                sharp=sharp,
                rfilter=_mvu_rfilter(rfilter),
                onelevel=(levels == 1),
                pelclip=pelclip,
            )
        ns = self._legacy_ns(clip)
        return ns.Super(clip, hpad=hpad, vpad=vpad, pel=pel, levels=levels, chroma=chroma, sharp=sharp, rfilter=rfilter, pelclip=pelclip)

    # -- Analyse / Analyze -------------------------------------------------

    def _analyse(
        self,
        super: vs.VideoNode,
        blksize: int = 8,
        blksizev: Optional[int] = None,
        levels: int = 0,
        search: int = 4,
        searchparam: int = 2,
        pelsearch: int = 0,
        isb: bool = False,
        lambda_: Optional[int] = None,
        chroma: bool = True,
        delta: int = 1,
        truemotion: bool = True,
        lsad: Optional[int] = None,
        plevel: Optional[int] = None,
        global_: Optional[bool] = None,
        pnew: Optional[int] = None,
        pzero: Optional[int] = None,
        pglobal: int = 0,
        overlap: int = 0,
        overlapv: Optional[int] = None,
        divide: int = 0,
        badsad: int = 10000,
        badrange: int = 24,
        meander: bool = True,
        trymany: bool = False,
        fields: bool = False,
        tff: Optional[bool] = None,
        search_coarse: int = 3,
        dct: int = 0,
    ) -> vs.VideoNode:
        if self.use_mvu:
            kwargs = dict(
                blksize=[blksize, blksizev or blksize],
                overlap=[overlap, overlapv or overlap],
                levels=levels,
                search=_mvu_search_mode(search),
                searchparam=searchparam,
                # isb=True (old "is backward") -> positive delta; isb=False (forward) -> negative delta.
                delta=delta if isb else -delta,
                mvlambda=(lambda_ if lambda_ is not None else (1000 if truemotion else 0)),
                chroma=chroma,
                lsad=(lsad if lsad is not None else 400),
                plevel=(plevel if plevel is not None else 1),
                globalmv=(global_ if global_ is not None else True),
                pnew=(pnew if pnew is not None else 25),
                pglobal=pglobal,
                badsad=badsad,
                badrange=badrange,
                meander=meander,
                trymany=(2 if trymany else 0),
                fields=fields,
                tff=bool(tff),
                satd=_mvu_dct_to_satd(dct),
            )
            kwargs['pzero'] = pzero if pzero is not None else kwargs['pnew']
            if pelsearch:
                kwargs['pelsearch'] = pelsearch
            return core.mvu.Analyse(super, **kwargs)
        ns = self._legacy_ns(super)
        func = self._legacy_analyse_func(ns)
        return func(
            super, blksize=blksize, blksizev=blksizev, levels=levels, search=search, searchparam=searchparam,
            pelsearch=pelsearch, isb=isb, lambda_=lambda_, chroma=chroma, delta=delta, truemotion=truemotion,
            lsad=lsad, plevel=plevel, global_=global_, pnew=pnew, pzero=pzero, pglobal=pglobal, overlap=overlap,
            overlapv=overlapv, divide=divide, badsad=badsad, badrange=badrange, meander=meander, trymany=trymany,
            fields=fields, tff=tff, search_coarse=search_coarse, dct=dct,
        )

    def Analyse(self, *args, **kwargs) -> vs.VideoNode:
        return self._analyse(*args, **kwargs)

    def Analyze(self, *args, **kwargs) -> vs.VideoNode:
        return self._analyse(*args, **kwargs)

    # -- Recalculate -------------------------------------------------------

    def Recalculate(
        self,
        super: vs.VideoNode,
        vectors,
        thsad: float = 200.0,
        smooth: int = 1,
        blksize: int = 8,
        blksizev: Optional[int] = None,
        search: int = 4,
        searchparam: int = 2,
        lambda_: Optional[int] = None,
        chroma: bool = True,
        truemotion: bool = True,
        pnew: Optional[int] = None,
        overlap: int = 0,
        overlapv: Optional[int] = None,
        divide: int = 0,
        meander: bool = True,
        fields: bool = False,
        tff: Optional[bool] = None,
        dct: int = 0,
    ):
        if self.use_mvu:
            return core.mvu.Recalculate(
                super, vectors,
                thsad=thsad,
                smooth=bool(smooth),
                blksize=[blksize, blksizev or blksize],
                search=_mvu_search_mode(search),
                searchparam=searchparam,
                mvlambda=(lambda_ if lambda_ is not None else (1000 if truemotion else 0)),
                chroma=chroma,
                pnew=(pnew if pnew is not None else 25),
                overlap=[overlap, overlapv or overlap],
                meander=meander,
                fields=fields,
                tff=bool(tff),
                satd=_mvu_dct_to_satd(dct),
            )
        ns = self._legacy_ns(super)
        return ns.Recalculate(
            super, vectors, thsad=thsad, smooth=smooth, blksize=blksize, blksizev=blksizev, search=search,
            searchparam=searchparam, lambda_=lambda_, chroma=chroma, truemotion=truemotion, pnew=pnew,
            overlap=overlap, overlapv=overlapv, divide=divide, meander=meander, fields=fields, tff=tff, dct=dct,
        )

    # -- Compensate ----------------------------------------------------------

    def Compensate(
        self,
        clip: vs.VideoNode,
        super: vs.VideoNode,
        vectors,
        scbehavior: int = 1,
        thsad: float = 10000.0,
        fields: bool = False,
        time: float = 100.0,
        thscd1: float = 400.0,
        thscd2: float = 130.0,
        tff: Optional[bool] = None,
    ) -> vs.VideoNode:
        if self.use_mvu:
            # mvutensils dropped `scbehavior`.
            return core.mvu.Compensate(
                clip, super, vectors, thsad=thsad, fields=fields, time=time,
                thscd1=thscd1, thscd2=_mvu_scale_thscd2(thscd2), tff=bool(tff),
            )
        ns = self._legacy_ns(clip)
        return ns.Compensate(clip, super, vectors, scbehavior=scbehavior, thsad=thsad, fields=fields, time=time, thscd1=thscd1, thscd2=thscd2, tff=tff)

    # -- Degrain / Degrain1..N ----------------------------------------------

        # -- Degrain / Degrain1..N ----------------------------------------------

        # -- Degrain / Degrain1..N ----------------------------------------------

    def _degrain(
        self,
        clip: vs.VideoNode,
        super: vs.VideoNode,
        *vectors,
        mvbw: Optional[vs.VideoNode] = None,
        mvfw: Optional[vs.VideoNode] = None,
        mvbw2: Optional[vs.VideoNode] = None,
        mvfw2: Optional[vs.VideoNode] = None,
        mvbw3: Optional[vs.VideoNode] = None,
        mvfw3: Optional[vs.VideoNode] = None,
        mvbw4: Optional[vs.VideoNode] = None,
        mvfw4: Optional[vs.VideoNode] = None,
        mvbw5: Optional[vs.VideoNode] = None,
        mvfw5: Optional[vs.VideoNode] = None,
        mvbw6: Optional[vs.VideoNode] = None,
        mvfw6: Optional[vs.VideoNode] = None,
        thsad: float = 400.0,
        thsadc: Optional[float] = None,
        plane: int = 4,
        limit: float = 255.0,
        limitc: Optional[float] = None,
        thscd1: float = 400.0,
        thscd2: float = 130.0,
        opt: bool = True,
    ) -> vs.VideoNode:
        """
        Degrain wrapper supporting both positional and named vector arguments.

        Positional (legacy):
            MV.Degrain2(clip, super, bw1, fw1, bw2, fw2)
        Named:
            MV.Degrain2(clip, super, mvbw=bw1, mvfw=fw1, mvbw2=bw2, mvfw2=fw2)
        Mixed:
            MV.Degrain2(clip, super, bw1, fw1, mvbw2=bw2, mvfw2=fw2)

        Named vectors override positional ones at the same index.
        For radii > 6, pass vectors positionally or use the generic Degrain().
        """
        vec_list = list(vectors)

        # Map named parameters to their slot index in the vector list.
        # Order: mvbw(0), mvfw(1), mvbw2(2), mvfw2(3), ...
        named_slots = [
            (mvbw, 0), (mvfw, 1),
            (mvbw2, 2), (mvfw2, 3),
            (mvbw3, 4), (mvfw3, 5),
            (mvbw4, 6), (mvfw4, 7),
            (mvbw5, 8), (mvfw5, 9),
            (mvbw6, 10), (mvfw6, 11),
        ]

        for vec, idx in named_slots:
            if vec is not None:
                while len(vec_list) <= idx:
                    vec_list.append(None)
                vec_list[idx] = vec

        # Detect gaps (e.g. user gave mvbw but forgot mvfw)
        for i, v in enumerate(vec_list):
            if v is None:
                raise vs.Error(
                    f'MV.Degrain: missing vector at position {i} '
                    f'(expected pairs: mvbw, mvfw, mvbw2, mvfw2, ...)'
                )

        if len(vec_list) % 2 != 0:
            raise vs.Error(
                'MV.Degrain: expected an even number of vector clips '
                '(bw1, fw1, bw2, fw2, ...).'
            )

        if self.use_mvu:
            return core.mvu.Degrain(
                clip, super, vec_list,
                thsad=[thsad, thsadc if thsadc is not None else thsad],
                planes=_mvu_plane_to_planes(plane, clip),
                limit=[
                    _mvu_limit_to_float(limit, clip),
                    _mvu_limit_to_float(limitc if limitc is not None else limit, clip),
                ],
                thscd1=thscd1,
                thscd2=_mvu_scale_thscd2(thscd2),
            )

        ns = self._legacy_ns(clip)
        radius = len(vec_list) // 2
        func = getattr(ns, f'Degrain{radius}')
        return func(
            clip, super, *vec_list,
            thsad=thsad,
            thsadc=thsadc if thsadc is not None else thsad,
            plane=plane,
            limit=limit,
            limitc=limitc if limitc is not None else limit,
            thscd1=thscd1,
            thscd2=thscd2,
            opt=opt,
        )

    def Degrain(self, clip, super, *vectors, **kwargs):
        return self._degrain(clip, super, *vectors, **kwargs)

    def __getattr__(self, name: str):
        # Handles Degrain1..Degrain24 (and any future DegrainN) without hand-writing each one.
        if name.startswith('Degrain') and (name[len('Degrain'):] == '' or name[len('Degrain'):].isdigit()):
            return lambda clip, super, *vectors, **kwargs: self._degrain(clip, super, *vectors, **kwargs)
        raise AttributeError(f'MotionVectors has no attribute {name!r}')

    # -- Flow / FlowInter / FlowFPS / FlowBlur -------------------------------

    def Flow(
        self, clip: vs.VideoNode, super: vs.VideoNode, vectors,
        time: float = 100.0, mode: int = 0, fields: bool = False,
        thscd1: float = 400.0, thscd2: float = 130.0, tff: Optional[bool] = None,
    ) -> vs.VideoNode:
        if self.use_mvu:
            # mvutensils dropped `mode` (only the old mode=0 behaviour remains).
            return core.mvu.Flow(clip, super, vectors, time=time, fields=fields, thscd1=thscd1, thscd2=_mvu_scale_thscd2(thscd2), tff=bool(tff))
        ns = self._legacy_ns(clip)
        return ns.Flow(clip, super, vectors, time=time, mode=mode, fields=fields, thscd1=thscd1, thscd2=thscd2, tff=tff)

    def FlowInter(
        self, clip: vs.VideoNode, super: vs.VideoNode, mvbw, mvfw,
        time: float = 50.0, ml: float = 100.0, blend: bool = True,
        thscd1: float = 400.0, thscd2: float = 130.0,
    ) -> vs.VideoNode:
        if self.use_mvu:
            return core.mvu.FlowInter(clip, super, [mvbw, mvfw], time=time, ml=ml, blend=blend, thscd1=thscd1, thscd2=_mvu_scale_thscd2(thscd2))
        ns = self._legacy_ns(clip)
        return ns.FlowInter(clip, super, mvbw, mvfw, time=time, ml=ml, blend=blend, thscd1=thscd1, thscd2=thscd2)

    def FlowFPS(
        self, clip: vs.VideoNode, super: vs.VideoNode, mvbw, mvfw,
        num: int = 25, den: int = 1, mask: int = 2, ml: float = 100.0, blend: bool = True,
        thscd1: float = 400.0, thscd2: float = 130.0,
    ) -> vs.VideoNode:
        if self.use_mvu:
            # old mask=1 -> extramask=False, old mask=2 (default) -> extramask=True.
            return core.mvu.FlowFPS(clip, super, [mvbw, mvfw], num=num, den=den, extramask=(mask != 1), ml=ml, blend=blend, thscd1=thscd1, thscd2=_mvu_scale_thscd2(thscd2))
        ns = self._legacy_ns(clip)
        return ns.FlowFPS(clip, super, mvbw, mvfw, num=num, den=den, mask=mask, ml=ml, blend=blend, thscd1=thscd1, thscd2=thscd2)

    def FlowBlur(
        self, clip: vs.VideoNode, super: vs.VideoNode, mvbw, mvfw,
        blur: float = 50.0, prec: int = 1, thscd1: float = 400.0, thscd2: float = 130.0,
    ) -> vs.VideoNode:
        if self.use_mvu:
            return core.mvu.FlowBlur(clip, super, [mvbw, mvfw], blur=blur, prec=prec, thscd1=thscd1, thscd2=_mvu_scale_thscd2(thscd2))
        ns = self._legacy_ns(clip)
        return ns.FlowBlur(clip, super, mvbw, mvfw, blur=blur, prec=prec, thscd1=thscd1, thscd2=thscd2)

    # -- Mask ------------------------------------------------------------

    def Mask(
        self, clip: vs.VideoNode, vectors,
        ml: float = 100.0, gamma: float = 1.0, kind: int = 0, time: float = 100.0, ysc: int = 0,
        thscd1: float = 400.0, thscd2: float = 130.0,
    ) -> vs.VideoNode:
        '''
        NOTE: mvutensils replaces Mask(kind=0/1/2) with three separate functions and drops
        `clip`/`ysc`, returning a single grayscale plane in the vector clip's format instead
        of a clip-shaped/UV-colored mask. If a call site relies on the old mask's exact shape
        (e.g. feeding it straight into MaskedMerge against `clip`), double check the result
        when porting that specific call.
        '''
        if self.use_mvu:
            func = {0: core.mvu.VectorLengthMask, 1: core.mvu.SADMask, 2: core.mvu.OcclusionMask}.get(kind)
            if func is None:
                raise vs.Error(f'MV.Mask: unsupported kind {kind!r}')
            return func(vectors, ml=ml, gamma=gamma, time=time, thscd1=thscd1, thscd2=_mvu_scale_thscd2(thscd2))
        ns = self._legacy_ns(clip)
        return ns.Mask(clip, vectors, ml=ml, gamma=gamma, kind=kind, time=time, ysc=ysc, thscd1=thscd1, thscd2=thscd2)

    # -- SCDetection -------------------------------------------------------

    def SCDetection(self, clip: vs.VideoNode, vectors, thscd1: float = 400.0, thscd2: float = 130.0) -> vs.VideoNode:
        if self.use_mvu:
            return core.mvu.SCDetection(clip, vectors, thscd1=thscd1, thscd2=_mvu_scale_thscd2(thscd2))
        ns = self._legacy_ns(clip)
        return ns.SCDetection(clip, vectors, thscd1=thscd1, thscd2=thscd2)

    # -- Depan family (integer-only in both mvtools and mvutensils) ---------

    def DepanEstimate(self, clip: vs.VideoNode, trust: float = 4.0, winx: int = 0, winy: int = 0, wleft: int = -1, wtop: int = -1,
                       dxmax: int = -1, dymax: int = -1, zoommax: float = 0.0, stab: float = 1.0, pixaspect: float = 1.0,
                       info: bool = False, show: bool = False, fields: bool = False, tff: Optional[bool] = None) -> vs.VideoNode:
        ns = core.mvu if self.use_mvu else core.mv
        return ns.DepanEstimate(clip, trust=trust, winx=winx, winy=winy, wleft=wleft, wtop=wtop, dxmax=dxmax, dymax=dymax,
                                 zoommax=(zoommax or 1.0) if self.use_mvu else zoommax, stab=stab, pixaspect=pixaspect,
                                 info=info, show=show, fields=fields, tff=bool(tff) if self.use_mvu else tff)

    def DepanAnalyse(self, clip: vs.VideoNode, vectors, mask: Optional[vs.VideoNode] = None, zoom: bool = True, rot: bool = True,
                      pixaspect: float = 1.0, error: float = 15.0, info: bool = False, wrong: float = 10.0, zerow: float = 0.05,
                      thscd1: float = 400.0, thscd2: float = 130.0, fields: bool = False, tff: Optional[bool] = None) -> vs.VideoNode:
        ns = core.mvu if self.use_mvu else core.mv
        return ns.DepanAnalyse(clip, vectors, mask=mask, zoom=zoom, rot=rot, pixaspect=pixaspect, error=error, info=info,
                                wrong=wrong, zerow=zerow, thscd1=thscd1,
                                thscd2=(_mvu_scale_thscd2(thscd2) if self.use_mvu else thscd2),
                                fields=fields, tff=bool(tff) if self.use_mvu else tff)

    def DepanStabilise(self, clip: vs.VideoNode, data: vs.VideoNode, cutoff: float = 1.0, damping: float = 0.9, initzoom: float = 1.0,
                        addzoom: bool = False, prev: int = 0, next: int = 0, mirror: int = 0, blur: int = 0, dxmax: float = 60.0,
                        dymax: float = 30.0, zoommax: float = 0.0, rotmax: float = 1.0, subpixel: int = 2, pixaspect: float = 1.0,
                        fitlast: int = 0, tzoom: float = 3.0, info: bool = False, method: int = 0, fields: bool = False) -> vs.VideoNode:
        ns = core.mvu if self.use_mvu else core.mv
        return ns.DepanStabilise(clip, data, cutoff=cutoff, damping=damping, initzoom=initzoom, addzoom=addzoom, prev=prev,
                                  next=next, mirror=mirror, blur=blur, dxmax=dxmax, dymax=dymax,
                                  zoommax=(zoommax or 1.05) if self.use_mvu else zoommax, rotmax=rotmax, subpixel=subpixel,
                                  pixaspect=pixaspect, fitlast=fitlast, tzoom=tzoom, info=info, method=method, fields=fields)

    def DepanCompensate(self, clip: vs.VideoNode, data: vs.VideoNode, offset: float = 0.0, subpixel: int = 2, pixaspect: float = 1.0,
                         matchfields: bool = True, mirror: int = 0, blur: int = 0, info: bool = False, fields: bool = False,
                         tff: Optional[bool] = None) -> vs.VideoNode:
        ns = core.mvu if self.use_mvu else core.mv
        return ns.DepanCompensate(clip, data, offset=offset, subpixel=subpixel, pixaspect=pixaspect, matchfields=matchfields,
                                   mirror=mirror, blur=blur, info=info, fields=fields, tff=bool(tff) if self.use_mvu else tff)


# Ready-made singleton: auto-detects mvutensils. Import this from other files:
#   from misc import MV
MV = MotionVectors()