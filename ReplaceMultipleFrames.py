import vapoursynth as vs
from vapoursynth import core

class ReplaceMultipleFrames:
    def __init__(self, clip: vs.VideoNode, intervals: list, method: str = 'SVP', rifeModel: int = 22, rifeTTA=False, rifeUHD=False, device_index: int = 0, debug: bool = False):
        """Initialize the frame replacement/interpolation class.
        
        Args:
            clip: Input video clip
            intervals: List of frame intervals to replace (each as [start, end])
            method: Interpolation method ('SVP', 'RIFE', or 'MV')
            rifeModel: Model to use for RIFE interpolation
            rifeTTA: Whether to use test-time augmentation for RIFE
            rifeUHD: Whether to use UHD mode for RIFE
            device_index: GPU device index to use
            debug: Whether to show debug information on frames
        """
        self.clip = clip
        self.method = method
        self.intervals = self.validate_intervals(intervals)
        self.device_index = device_index
        self.rifeModel = rifeModel
        self.rifeTTA = rifeTTA
        self.rifeUHD = rifeUHD
        self.smooth = None  # Cache for interpolated frames
        self.smooth_start = -1  # Start frame of cached interpolation
        self.smooth_end = -1  # End frame of cached interpolation
        self.debug = debug

    def validate_intervals(self, intervals):
        """Validate that intervals are properly formatted and within acceptable range.
        
        Args:
            intervals: List of frame intervals to validate
            
        Returns:
            List of validated intervals
            
        Raises:
            ValueError: If any interval is invalid
        """
        validated = []
        for interval in intervals:
            # Check interval format
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                raise ValueError(f"Invalid interval format: {interval}")
            start, end = interval
            # Check interval length
            if end - start + 1 < 1:
                raise ValueError(f"Interval too short: {interval}")
            if end - start + 1 > 10:
                raise ValueError(f"Interval too long: {interval}")
            validated.append((start, end))
        return validated

    def interpolate(self, n, f):
        """Main interpolation function called for each frame.
        
        Args:
            n: Frame number being processed
            f: Frame properties (unused)
            
        Returns:
            Original frame if outside interval, interpolated frame if inside interval
        """
        for start, end in self.intervals:
            # If frame is within a replacement interval, interpolate it
            if start <= n and n <= end:
                before = start - 1  # Frame before interval
                after = end + 1  # Frame after interval
                # Validate surrounding frames exist
                if before < 0 or after >= self.clip.num_frames:
                    raise ValueError(f"Cannot interpolate: invalid surrounding frames for interval [{start}, {end}]")
                # Interpolate between the surrounding frames
                return self.interpolate_between_frames(before, after, n - before, end - start + 1)
        # Return original frame if not in any interval
        return self.clip[n]

    def interpolate_between_frames(self, start, end, index, num):
        """Create interpolation between two frames and cache the result.
        
        Args:
            start: First frame number to interpolate from
            end: Second frame number to interpolate to
            index: Which interpolated frame to return
            num: Total number of frames to interpolate between start and end
            
        Returns:
            The requested interpolated frame
        """
        clip = self.clip[start] + self.clip[end]
        clip = clip.std.AssumeFPS(fpsnum=1, fpsden=1)
        if self.smooth is None or self.smooth_start != start or self.smooth_end != end:
            if self.method.lower() == 'svp':
                self.smooth = self.interpolateWithSVP(clip, num+1)  # Note: num+1
            elif self.method.lower() == 'rife':
                self.smooth = self.interpolateWithRIFE(clip, num+1)  # Note: num+1
            elif self.method.lower() == 'mv':
                self.smooth = self.interpolateWithMV(clip, num+1)  # Note: num+1
            else:
                raise ValueError(f"Unsupported method: {self.method}")
            
            if self.debug:
                self.smooth = core.text.Text(self.smooth, f"{self.method} {start}", alignment=8)
            
            self.smooth_start = start
            self.smooth_end = end
        
        # Return the correct interpolated frame
        return self.smooth[index]

    def interpolateWithSVP(self, clip, num):
        """Interpolate frames using SVP (SmoothVideo Project) method.
        
        Args:
            clip: Input clip (should contain exactly 2 frames)
            num: Number of frames to interpolate between the input frames
            
        Returns:
            Clip with interpolated frames
        """
        if clip.format.id != vs.YUV420P8:
            raise ValueError("SVP requires YUV420P8 format")
        super = core.svp1.Super(clip, f"{{gpu:{1 if self.method.lower() == 'svp' else 0}}}")
        vectors = core.svp1.Analyse(super["clip"], super["data"], clip, "{}")
        return core.svp2.SmoothFps(
            clip, super["clip"], super["data"], vectors["clip"], vectors["data"],
            f"{{rate:{{num:{num},den:1,abs:true}}}}"
        )

    def interpolateWithRIFE(self, clip, num):
        """Interpolate frames using RIFE (Real-Time Intermediate Flow Estimation) method.
        
        Args:
            clip: Input clip (should contain exactly 2 frames)
            num: Number of frames to interpolate between the input frames
            
        Returns:
            Clip with interpolated frames
        """
        if clip.format.id != vs.RGBS:
            raise ValueError("RIFE requires RGBS format")
        return core.rife.RIFE(clip, model=self.rifeModel, factor_num=num, tta=self.rifeTTA, uhd=self.rifeUHD, gpu_id=self.device_index)

    def interpolateWithMV(self, clip, num):
        """Interpolate frames using motion vectors method.
        
        Args:
            clip: Input clip (should contain exactly 2 frames)
            num: Number of frames to interpolate between the input frames
            
        Returns:
            Clip with interpolated frames
        """
        sup = core.mv.Super(clip, pel=2, hpad=0, vpad=0)
        bvec = core.mv.Analyse(sup, blksize=16, isb=True, chroma=True, search=3, searchparam=1)
        fvec = core.mv.Analyse(sup, blksize=16, isb=False, chroma=True, search=3, searchparam=1)
        return core.mv.FlowFPS(clip, sup, bvec, fvec, num=num, den=1, mask=2)

    @property
    def out(self):
        """Property that returns the processed clip with interpolated frames."""
        return core.std.FrameEval(self.clip, self.interpolate, prop_src=self.clip)