from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import vapoursynth as vs

core = vs.core

Shift = Tuple[float, float]
FrameShifts = Dict[str, Shift]

class ChromaFixer:
    def __init__(self, clip, referenceframe, accuracy, maxshift, nodisplay, nofix):
        if clip.format.color_family != vs.YUV:
            raise ValueError("Input must be YUV format")
        if maxshift < 0:
            raise ValueError("maxshift must be >= 0")
        if accuracy <= 0:
            raise ValueError("accuracy must be > 0")

        self.clip = clip
        self.referenceframe = referenceframe
        self.accuracy = accuracy
        self.maxshift = maxshift
        self.nodisplay = nodisplay
        self.nofix = nofix

        # Get chroma subsampling factors
        self.ssw = clip.format.subsampling_w
        self.ssh = clip.format.subsampling_h

        # Pre-calculate dimensions
        self.chroma_width = clip.width >> self.ssw
        self.chroma_height = clip.height >> self.ssh

    @staticmethod
    def edge_mask(plane: vs.VideoNode) -> vs.VideoNode:
        PREWITT = core.edgemasks.ExPrewitt if hasattr(core, "edgemasks") else core.std.Prewitt
        return PREWITT(plane)

    def siting_offsets(self, source: vs.VideoNode) -> Shift:
        """Where the chroma samples sit relative to the luma grid, in luma pixels

        Without this the downscaled luma lands between the chroma samples and the
        search reports a shift that is not in the picture - half a chroma pixel
        horizontally for the usual left-sited 4:2:0.
        """
        location = source.get_frame(0).props.get('_ChromaLocation', 0)  # 0 = left, the common default
        sw, sh = 1 << self.ssw, 1 << self.ssh
        src_left = 0.5 - 0.5 * sw if location in (0, 2, 4) else 0.0     # left / top-left / bottom-left are co-sited
        if location in (2, 3):                                          # top-left / top
            src_top = 0.5 - 0.5 * sh
        elif location in (4, 5):                                        # bottom-left / bottom
            src_top = 0.5 * sh - 0.5
        else:                                                           # left / center: vertically centered
            src_top = 0.0
        return src_left, src_top

    def compute_shifts(self, source: vs.VideoNode) -> FrameShifts:
        """Measure the shift of both chroma planes against the luma of one frame"""
        y = source.std.ShufflePlanes(planes=0, colorfamily=vs.GRAY)
        u = source.std.ShufflePlanes(planes=1, colorfamily=vs.GRAY)
        v = source.std.ShufflePlanes(planes=2, colorfamily=vs.GRAY)

        # Luma at chroma resolution, sampled where the chroma samples actually
        # sit. Bicubic rather than Point: point sampling a 2:1 downscale lands
        # exactly between two luma columns and picks one of them, which on its
        # own is worth a whole luma pixel of bias.
        # The edge mask is built once here and reused for every candidate - it
        # does not depend on the shift being tested.
        src_left, src_top = self.siting_offsets(source)
        y_down = y.resize.Bicubic(
            width=self.chroma_width,
            height=self.chroma_height,
            src_left=src_left,
            src_top=src_top,
            format=vs.GRAY16
        )
        y_edges = self.edge_mask(y_down)

        return {'u': self.find_chroma_shift(y_edges, u), 'v': self.find_chroma_shift(y_edges, v)}

    def process_frame(self, n: int) -> vs.VideoNode:
        frame = self.clip[n]
        return self.apply_shifts(frame, self.compute_shifts(frame))

    def find_chroma_shift(self, y_edges: vs.VideoNode, chroma_plane: vs.VideoNode) -> Shift:
        """Find optimal shift for one chroma plane relative to luma"""
        # Coarse search over whole pixels
        whole = range(-self.maxshift, self.maxshift + 1)
        h_base, v_base = self.search(y_edges, chroma_plane,
                                     [(float(h), float(v)) for h in whole for v in whole])

        # Fine search around the best match, never leaving the requested range
        fine_h = self.clamped([h_base + x * self.accuracy for x in range(-2, 3)])
        fine_v = self.clamped([v_base + x * self.accuracy for x in range(-2, 3)])
        return self.search(y_edges, chroma_plane, [(h, v) for h in fine_h for v in fine_v])

    def clamped(self, values: Iterable[float]) -> List[float]:
        """Keep candidates inside +/-maxshift and drop duplicates created by clamping"""
        limit = float(self.maxshift)
        return sorted({min(limit, max(-limit, round(value, 6))) for value in values})

    def search(self, y_edges: vs.VideoNode, chroma_plane: vs.VideoNode, candidates: Iterable[Shift]) -> Shift:
        """Pick the candidate with the lowest score, smallest shift first

        The order matters: a strictly smaller score is required to replace the
        incumbent, so candidates that score exactly the same - which is what
        happens where there is no chroma detail to lock onto - keep the smallest
        shift instead of whichever one the loop happened to visit first. No
        tolerance is applied on top of that; the real gain of the correct shift
        can be as small as 5e-4 on detailed material, so anything coarser would
        hide genuine misalignment.
        """
        ordered = sorted(candidates, key=lambda hv: (abs(hv[0]) + abs(hv[1]), abs(hv[0]), abs(hv[1]), hv))
        best_shift = ordered[0]
        best_diff = self.calculate_alignment_diff(y_edges, chroma_plane, *best_shift)
        for shift in ordered[1:]:
            diff = self.calculate_alignment_diff(y_edges, chroma_plane, *shift)
            if diff < best_diff:
                best_diff = diff
                best_shift = shift
        return best_shift

    def calculate_alignment_diff(self, y_edges: vs.VideoNode, chroma_plane: vs.VideoNode, h: float, v: float) -> float:
        """Mean absolute difference between the luma edges and the shifted chroma edges"""
        shifted = core.resize.Bicubic(
            chroma_plane,
            src_left=h,
            src_top=v,
            format=vs.GRAY16
        )
        # PlaneStats with two clips reports the mean absolute difference directly,
        # which is exactly what Expr("x y - abs") + PlaneStatsAverage produced.
        stats = core.std.PlaneStats(y_edges, self.edge_mask(shifted))
        return stats.get_frame(0).props['PlaneStatsDiff']

    def apply_shifts(self, clip: vs.VideoNode, shifts: FrameShifts) -> vs.VideoNode:
        """Apply the calculated shifts to chroma planes"""
        if self.nofix:
            result = clip
        else:
            # Get original planes
            y = clip.std.ShufflePlanes(planes=0, colorfamily=vs.GRAY)

            # Shift chroma planes
            u_shifted = core.resize.Bicubic(
                clip.std.ShufflePlanes(planes=1, colorfamily=vs.GRAY),
                src_left=shifts['u'][0],
                src_top=shifts['u'][1]
            )
            v_shifted = core.resize.Bicubic(
                clip.std.ShufflePlanes(planes=2, colorfamily=vs.GRAY),
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

def AutoChromaFix(
    clip: vs.VideoNode,
    referenceframe: int = -1,
    nodisplay: bool = False,
    nofix: bool = False,
    accuracy: float = 0.25,
    maxshift: int = 2,
) -> vs.VideoNode:
    """
    Correct chroma alignment in YUV video
    From my experience so far this mainly works with animes/cartoons.
    Parameters:
        clip: YUV input clip
        referenceframe: Frame to use as reference (-1 = per-frame)
        nodisplay: Hide shift values
        nofix: Don't apply correction
        accuracy: Search precision (0.25 = quarter-pixel)
        maxshift: Maximum shift to search (pixels)

    With a reference frame the shift is measured once, before any frame is
    rendered, and then applied to the whole clip - the result must not depend on
    the order in which the renderer asks for frames.
    Per-frame mode (-1) measures every frame separately and is a lot slower.
    """
    fixer = ChromaFixer(clip=clip, referenceframe=referenceframe, accuracy=accuracy, maxshift=maxshift, nodisplay=nodisplay, nofix=nofix)
    if referenceframe >= 0:
        reference = min(referenceframe, clip.num_frames - 1)
        return fixer.apply_shifts(clip, fixer.compute_shifts(clip[reference]))
    return core.std.FrameEval(clip, lambda n: fixer.process_frame(n))

# example use:
# import autoChromaFix
# clip = autoChromaFix.AutoChromaFix(clip=clip, nodisplay=True)
