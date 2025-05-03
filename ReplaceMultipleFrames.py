import vapoursynth as vs
from vapoursynth import core

class ReplaceMultipleFrames:
    def __init__(self, clip: vs.VideoNode, intervals: list, method: str = 'SVP', rifeModel: int = 22, rifeTTA=False, rifeUHD=False, device_index: int = 0, debug: bool = False):
        self.clip = clip
        self.method = method
        self.intervals = self.validate_intervals(intervals)
        self.device_index = device_index
        self.rifeModel = rifeModel
        self.rifeTTA = rifeTTA
        self.rifeUHD = rifeUHD
        self.smooth = None
        self.smooth_start = -1
        self.smooth_end = -1
        self.debug = debug

    def validate_intervals(self, intervals):
        validated = []
        for interval in intervals:
            if not isinstance(interval, (list, tuple)) or len(interval) != 2:
                raise ValueError(f"Invalid interval format: {interval}")
            start, end = interval
            if end - start + 1 < 1:
                raise ValueError(f"Interval too short: {interval}")
            if end - start + 1 > 10:
              raise ValueError(f"Interval too long: {interval}")
            validated.append((start, end))
        return validated

    def interpolate(self, n, f):
        for start, end in self.intervals:
            # Wenn n im Intervall [start, end] liegt, interpolieren
            if start <= n and n <= end:  # Interpolieren aller Frames im Intervall [start, end]
                before = start - 1
                after = end + 1
                if before < 0 or after >= self.clip.num_frames:
                    raise ValueError(f"Cannot interpolate: invalid surrounding frames for interval [{start}, {end}]")
                # Interpolation zwischen den benachbarten Frames before und after
                return self.interpolate_between_frames(before, after, n - before, end - start + 1)
        # Wenn n außerhalb des Intervalls liegt, den originalen Frame zurückgeben
        return self.clip[n]

    def interpolate_between_frames(self, start, end, index, num):
        clip = self.clip[start] + self.clip[end]
        clip = clip.std.AssumeFPS(fpsnum=1, fpsden=1)
        if self.smooth is None or self.smooth_start != start or self.smooth_end != end:
            if self.method.lower() == 'svp':
                self.smooth = self.interpolateWithSVP(clip, num)
            elif self.method.lower() == 'rife':
                self.smooth = self.interpolateWithRIFE(clip, num)
            elif self.method.lower() == 'mv':
                self.smooth = self.interpolateWithMV(clip, num)
            else:
                raise ValueError(f"Unsupported method: {self.method}")
            if self.debug:
                #self.smooth = core.text.Text(self.smooth, f"{self.method} {start}", alignment=8)
                self.smooth = core.text.Text(self.smooth, f"{self.method}", alignment=8)
            self.smooth_start = start
            self.smooth_end = end
        return self.smooth[index]

    def interpolateWithSVP(self, clip, num):
        if clip.format.id != vs.YUV420P8:
            raise ValueError("SVP requires YUV420P8 format")
        super = core.svp1.Super(clip, f"{{gpu:{1 if self.method.lower() == 'svp' else 0}}}")
        vectors = core.svp1.Analyse(super["clip"], super["data"], clip, "{}")
        return core.svp2.SmoothFps(
            clip, super["clip"], super["data"], vectors["clip"], vectors["data"],
            f"{{rate:{{num:{num},den:1,abs:true}}}}"
        )

    def interpolateWithRIFE(self, clip, num):
        if clip.format.id != vs.RGBS:
            raise ValueError("RIFE requires RGBS format")
        return core.rife.RIFE(clip, model=self.rifeModel, factor_num=num, tta=self.rifeTTA, uhd=self.rifeUHD, gpu_id=self.device_index)

    def interpolateWithMV(self, clip, num):
        sup = core.mv.Super(clip, pel=2, hpad=0, vpad=0)
        bvec = core.mv.Analyse(sup, blksize=16, isb=True, chroma=True, search=3, searchparam=1)
        fvec = core.mv.Analyse(sup, blksize=16, isb=False, chroma=True, search=3, searchparam=1)
        return core.mv.FlowFPS(clip, sup, bvec, fvec, num=num, den=1, mask=2)

    @property
    def out(self):
        return core.std.FrameEval(self.clip, self.interpolate, prop_src=self.clip)
