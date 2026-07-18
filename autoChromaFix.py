import vapoursynth as vs
from functools import partial

core = vs.core

class ChromaFixer:
    def __init__(self, clip, referenceframe, accuracy, maxshift, nodisplay, nofix):
        if clip.format.color_family != vs.YUV:
            raise ValueError("Input must be YUV format")
        
        self.clip = clip
        self.referenceframe = referenceframe
        self.accuracy = accuracy
        self.maxshift = maxshift
        self.nodisplay = nodisplay
        self.nofix = nofix
        self.shifts = {}  # Stores {'u': (h,v), 'v': (h,v)} for each frame
        self.ref_shift = None
        
        # Get chroma subsampling factors
        self.ssw = clip.format.subsampling_w
        self.ssh = clip.format.subsampling_h
        
        # Pre-calculate dimensions
        self.luma_width = clip.width
        self.luma_height = clip.height
        self.chroma_width = clip.width >> self.ssw
        self.chroma_height = clip.height >> self.ssh

    def process_frame(self, n):
        frame = self.clip[n]
        
        # If using reference frame and we've already calculated it
        if self.referenceframe >= 0 and self.ref_shift is not None:
            return self.apply_shifts(frame, self.ref_shift)
        
        # Calculate new shifts if needed
        if self.referenceframe < 0 or n == self.referenceframe:
            # Extract planes
            y = frame.std.ShufflePlanes(planes=0, colorfamily=vs.GRAY)
            u = frame.std.ShufflePlanes(planes=1, colorfamily=vs.GRAY)
            v = frame.std.ShufflePlanes(planes=2, colorfamily=vs.GRAY)
            
            # Find optimal shifts
            u_shift = self.find_chroma_shift(y, u)
            v_shift = self.find_chroma_shift(y, v)
            
            # Store results
            shifts = {'u': u_shift, 'v': v_shift}
            if self.referenceframe >= 0:
                self.ref_shift = shifts
            else:
                self.shifts[n] = shifts
            
            return self.apply_shifts(frame, shifts)
        
        return frame

    def find_chroma_shift(self, y_plane, chroma_plane):
        """Find optimal shift for one chroma plane relative to luma"""
        best_diff = float('inf')
        best_shift = (0, 0)
        
        # Convert luma to chroma resolution for comparison
        y_down = y_plane.resize.Point(
            width=self.chroma_width,
            height=self.chroma_height,
            format=vs.GRAY16
        )
        
        # Coarse search
        for h in range(-self.maxshift, self.maxshift + 1):
            for v in range(-self.maxshift, self.maxshift + 1):
                shifted = core.resize.Bicubic(
                    chroma_plane,
                    src_left=h,
                    src_top=v,
                    format=vs.GRAY16
                )
                diff = self.calculate_alignment_diff(y_down, shifted)
                if diff < best_diff:
                    best_diff = diff
                    best_shift = (h, v)
        
        # Fine search around best match
        best_diff = float('inf')
        h_base, v_base = best_shift
        for h in [h_base + x*self.accuracy for x in range(-2, 3)]:
            for v in [v_base + x*self.accuracy for x in range(-2, 3)]:
                shifted = core.resize.Bicubic(
                    chroma_plane,
                    src_left=h,
                    src_top=v,
                    format=vs.GRAY16
                )
                diff = self.calculate_alignment_diff(y_down, shifted)
                if diff < best_diff:
                    best_diff = diff
                    best_shift = (h, v)
        
        return best_shift

    def calculate_alignment_diff(self, y, chroma):
        """Calculate alignment difference between luma and chroma edges"""
        # Edge detection
        y_edges = core.std.Prewitt(y)
        chroma_edges = core.std.Prewitt(chroma)
        
        # Calculate absolute difference
        EXPR = core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr
        diff = EXPR([y_edges, chroma_edges], "x y - abs")
        return core.std.PlaneStats(diff).get_frame(0).props.PlaneStatsAverage

    def apply_shifts(self, frame, shifts):
        """Apply the calculated shifts to chroma planes"""
        if self.nofix:
            result = frame
        else:
            # Get original planes
            y = frame.std.ShufflePlanes(planes=0, colorfamily=vs.GRAY)
            
            # Shift chroma planes
            u_shifted = core.resize.Bicubic(
                frame.std.ShufflePlanes(planes=1, colorfamily=vs.GRAY),
                src_left=shifts['u'][0],
                src_top=shifts['u'][1]
            )
            v_shifted = core.resize.Bicubic(
                frame.std.ShufflePlanes(planes=2, colorfamily=vs.GRAY),
                src_left=shifts['v'][0],
                src_top=shifts['v'][1]
            )
            
            # Recombine planes - CORRECTED PLANE COMBINATION
            result = core.std.ShufflePlanes(
                clips=[y, u_shifted, v_shifted],
                planes=[0, 0, 0],
                colorfamily=vs.YUV
            )
        
        if not self.nodisplay:
            result = core.text.Text(
                result,
                f"U: h={shifts['u'][0]:.2f} (x), v={shifts['u'][1]:.2f} (y)\n"
                f"V: h={shifts['v'][0]:.2f} (x), v={shifts['v'][1]:.2f} (y)"
            )
        
        return result

def AutoChromaFix(clip, referenceframe=-1, nodisplay=False, nofix=False, accuracy=0.25, maxshift=2):
    """
    Correct chroma alignment in YUV video
    Parameters:
        clip: YUV input clip
        referenceframe: Frame to use as reference (-1 = per-frame)
        nodisplay: Hide shift values
        nofix: Don't apply correction
        accuracy: Search precision (0.25 = quarter-pixel)
        maxshift: Maximum shift to search (pixels)
    """
    fixer = ChromaFixer(clip, referenceframe, accuracy, maxshift, nodisplay, nofix)
    return core.std.FrameEval(clip, lambda n: fixer.process_frame(n))

# example use: 
# import autoChromaFix
# clip = autoChromaFix.AutoChromaFix(clip=clip, nodisplay=True)