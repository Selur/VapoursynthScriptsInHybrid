import vapoursynth as vs
from vapoursynth import core
from typing import Callable

class _DuplicateAwareProcessor:
    """
    Internal helper class to manage the state for the duplicate-aware process.
    An instance of this class is created by the main `duplicateAware` function.
    It holds the state of the previously processed frame and applies the
    user-defined filter function as needed.
    """
    def __init__(self, clip: vs.VideoNode, filter_func: Callable[[vs.VideoNode], vs.VideoNode], thresh: float, debug: bool):
        self.clip = clip
        self.filter_func = filter_func
        self.thresh = thresh
        self.debug = debug
        self.previous_processed_frame: vs.VideoNode = None

    def process(self, n: int, f: vs.VideoFrame) -> vs.VideoNode:
        """
        This is the callback function invoked by FrameEval for each frame.

        It checks if the current frame 'n' is a duplicate based on the PlaneStatsDiff
        property from the frame 'f'. If it's not a duplicate, it processes the
        frame from the original clip; otherwise, it returns the last processed frame.
        """
        # Frame 0 is always processed. For subsequent frames, check the difference threshold.
        if n == 0 or f.props['PlaneStatsDiff'] > self.thresh:
            # Not a duplicate: apply the user-provided filter to the current frame.
            processed = self.filter_func(self.clip[n])
            self.previous_processed_frame = processed
        # If it is a duplicate, we do nothing and simply re-use self.previous_processed_frame.

        out = self.previous_processed_frame

        # If debugging is enabled, overlay the frame number and diff value.
        if self.debug:
            is_duplicate = n > 0 and f.props['PlaneStatsDiff'] <= self.thresh
            diff_text = (f"Frame: {n}\n"
                         f"Duplicate: {'Yes' if is_duplicate else 'No'}\n"
                         f"PlaneStatsDiff: {f.props.get('PlaneStatsDiff', 0):.6f}")
            
            original_format = out.format
            original_cf = original_format.color_family
            matrix = f.props.get('_Matrix', 1) if original_cf == vs.YUV else None
            
            if original_cf == vs.RGB and original_format.bits_per_sample > 8:
                 debug_clip = core.resize.Bicubic(out, format=vs.RGBS, dither_type='error_diffusion')
                 debug_clip = core.text.Text(clip=debug_clip, text=diff_text, alignment=7)
                 out = core.resize.Bicubic(debug_clip, format=original_format.id)
            elif original_cf == vs.YUV:
                 debug_clip = core.resize.Bicubic(out, format=vs.RGBS, matrix_in=matrix, dither_type='error_diffusion')
                 debug_clip = core.text.Text(clip=debug_clip, text=diff_text, alignment=7)
                 out = core.resize.Bicubic(debug_clip, format=original_format.id, matrix=matrix)
            else:
                 out = core.text.Text(clip=out, text=diff_text, alignment=7)
                 
        return out

def duplicateAware(
    clip: vs.VideoNode,
    func: Callable[[vs.VideoNode], vs.VideoNode],
    thresh: float = 0.0001,
    debug: bool = False
) -> vs.VideoNode:
    """
    Applies a VapourSynth filter function only to frames that are not duplicates
    of the preceding frame, significantly speeding up slow filters on content
    with repeated frames.
    """
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("duplicateAware: 'clip' must be a VapourSynth VideoNode.")
    if not callable(func):
        raise TypeError("duplicateAware: 'func' must be a callable function.")

    stats_clip = clip
    if clip.format.id == vs.RGBH:
        stats_clip = core.resize.Bicubic(clip, format=vs.RGBS)

    prop_clip = core.std.PlaneStats(stats_clip, stats_clip[0] + stats_clip)
   
    # 1. Process the first frame to discover the output properties (format, resolution, etc.)
    processed_frame_zero = func(clip[0])

    # 2. Create a BlankClip that has the properties of the processed frame,
    #    but crucially, the length of the original input clip. This is the template.
    template_clip = core.std.BlankClip(
        clip=processed_frame_zero,
        length=clip.num_frames
    )

    processor = _DuplicateAwareProcessor(clip, func, thresh, debug)
    
    return core.std.FrameEval(template_clip, processor.process, prop_src=prop_clip)