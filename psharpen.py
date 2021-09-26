from vapoursynth import core, GRAY

__version__ = "1.1.1"

# from https://gist.github.com/4re/2545a281e3f17ba6ef82#file-psharpen-py-L64

def _clamp(minimum, value, maximum):
    return int(max(minimum, min(round(value), maximum)))


def _m4(value, mult=4.0):
    return 16 if value < 16 else int(round(value / mult) * mult)


def psharpen(clip, strength=25, threshold=75, ss_x=1.0, ss_y=1.0, dest_x=None, dest_y=None):
    """From http://forum.doom9.org/showpost.php?p=683344&postcount=28

    Sharpening function similar to LimitedSharpenFaster.

    Args:
        clip (clip): Input clip.
        strength (int): Strength of the sharpening.
        threshold (int): Controls "how much" to be sharpened.
        ss_x (float): Supersampling factor (reduce aliasing on edges).
        ss_y (float): Supersampling factor (reduce aliasing on edges).
        dest_x (int): Output resolution after sharpening.
        dest_y (int): Output resolution after sharpening.
    """

    src = clip

    if dest_x is None:
        dest_x = src.width
    if dest_y is None:
        dest_y = src.height

    strength = _clamp(0, strength, 100) / 100.0
    threshold = _clamp(0, threshold, 100) / 100.0

    if ss_x < 1.0:
        ss_x = 1.0
    if ss_y < 1.0:
        ss_y = 1.0

    if ss_x != 1.0 or ss_y != 1.0:
        clip = core.resize.Lanczos(clip, width=_m4(src.width * ss_x), height=_m4(src.height * ss_y))

    resz = clip

    if src.format.num_planes != 1:
        clip = core.std.ShufflePlanes(clips=clip, planes=[0], colorfamily=GRAY)

    max_ = core.std.Maximum(clip)
    min_ = core.std.Minimum(clip)

    nmax = core.std.Expr([max_, min_], ["x y -"])
    nval = core.std.Expr([clip, min_], ["x y -"])

    expr0 = threshold * (1.0 - strength) / (1.0 - (1.0 - threshold) * (1.0 - strength))
    epsilon = 0.000000000000001
    scl = (1 << clip.format.bits_per_sample) // 256
    x = f"x {scl} /" if scl != 1 else "x"
    y = f"y {scl} /" if scl != 1 else "y"

    expr = (
        f"{x} {y} {epsilon} + / 2 * 1 - abs {expr0} < {strength} 1 = {x} {y} 2 / = 0 {y} 2 / ? "
        f"{x} {y} {epsilon} + / 2 * 1 - abs 1 {strength} - / ? {x} {y} {epsilon} + / 2 * 1 - abs 1 {threshold} - "
        f"* {threshold} + ? {x} {y} 2 / > 1 -1 ? * 1 + {y} * 2 / {scl} *"
    )

    nval = core.std.Expr([nval, nmax], [expr])

    clip = core.std.Expr([nval, min_], ["x y +"])

    if src.format.num_planes != 1:
        clip = core.std.ShufflePlanes(
            clips=[clip, resz], planes=[0, 1, 2], colorfamily=src.format.color_family
        )

    if ss_x != 1.0 or ss_y != 1.0 or dest_x != src.width or dest_y != src.height:
        clip = core.resize.Lanczos(clip, width=dest_x, height=dest_y)

    return clip
