from vapoursynth import core
import vapoursynth as vs

from typing import Optional, Union, Sequence

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
    EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr
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

    sc = core.misc.SCDetect(sc_src, threshold=sc_threshold)
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