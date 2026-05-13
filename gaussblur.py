import math
from typing import Sequence, Union, Optional
import vapoursynth as vs

# based on vs-jetpack

def _norm_planes(clip: vs.VideoNode, planes):
    if planes is None:
        return list(range(clip.format.num_planes))
    elif isinstance(planes, int):
        return [planes]
    return list(planes)


def _gauss_kernel(sigma: float, radius: int):
    """Normalized 1D Gaussian kernel."""
    kernel = [math.exp(-(i * i) / (2.0 * sigma * sigma)) for i in range(-radius, radius + 1)]
    total = sum(kernel)
    return [v / total for v in kernel]


def _boxblur_impl():
    """Pick the best available BoxBlur function."""
    core = vs.core
    if hasattr(core, 'vszip'):
        return core.vszip.BoxBlur
    return core.std.BoxBlur


def GaussBlur(
    clip: vs.VideoNode,
    sigma: Union[float, Sequence[float]] = 0.5,
    radius: Optional[int] = None,
    mode: str = "hv",
    planes: Optional[Union[int, Sequence[int]]] = None,
    max_conv_radius: int = 12,
) -> vs.VideoNode:
    """
    Standalone Gaussian blur. Drop-in replacement for:
      core.tcanny.TCanny(..., mode=-1, sigma=sigma, planes=planes)

    - Small sigma  -> exact Gaussian via std.Convolution
    - Large sigma  -> 3-pass BoxBlur approximation
    """
    if not isinstance(clip, vs.VideoNode):
        raise TypeError("GaussBlur: clip must be a VideoNode")

    planes = _norm_planes(clip, planes)
    mode = mode.lower()

    # Per-plane sigma: split -> process -> join
    if isinstance(sigma, Sequence):
        if len(sigma) != len(planes):
            raise ValueError("GaussBlur: per-plane sigma list length must match planes")
        if len(set(sigma)) == 1:
            sigma = sigma[0]
        else:
            return _gauss_blur_per_plane(clip, sigma, radius, mode, planes, max_conv_radius)

    if sigma <= 0:
        return clip

    if radius is None:
        radius = max(1, math.ceil(sigma * 3))
        if radius % 2 == 0:
            radius += 1

    # ---------- Large-sigma fallback ----------
    if radius > max_conv_radius:
        box_radius = max(1, round(sigma))
        blur_fn = _boxblur_impl()

        out = clip
        if "h" in mode:
            out = blur_fn(out, planes=planes,
                          hradius=box_radius, hpasses=3,
                          vradius=0, vpasses=0)
        if "v" in mode:
            out = blur_fn(out, planes=planes,
                          hradius=0, hpasses=0,
                          vradius=box_radius, vpasses=3)
        return out

    # ---------- Exact small-kernel path ----------
    kernel = _gauss_kernel(sigma, radius)

    out = clip
    if "h" in mode:
        out = out.std.Convolution(matrix=kernel, mode='h', planes=planes)
    if "v" in mode:
        out = out.std.Convolution(matrix=kernel, mode='v', planes=planes)
    return out


def _gauss_blur_per_plane(
    clip: vs.VideoNode,
    sigmas: Sequence[float],
    radius: Optional[int],
    mode: str,
    planes: Sequence[int],
    max_conv_radius: int,
) -> vs.VideoNode:
    """Process selected planes individually and recombine."""
    core = vs.core
    all_planes = list(range(clip.format.num_planes))

    gray_clips = [core.std.ShufflePlanes(clip, i, vs.GRAY) for i in all_planes]

    for idx, p in enumerate(planes):
        gray_clips[p] = GaussBlur(
            gray_clips[p],
            sigma=sigmas[idx],
            radius=radius,
            mode=mode,
            planes=[0],
            max_conv_radius=max_conv_radius,
        )

    cf = clip.format.color_family
    if cf == vs.YUV:
        return core.std.ShufflePlanes(gray_clips, planes=[0, 0, 0], color_family=vs.YUV)
    elif cf == vs.RGB:
        return core.std.ShufflePlanes(gray_clips, planes=[0, 0, 0], color_family=vs.RGB)
    return gray_clips[0]