
from __future__ import annotations

import vapoursynth as vs
core = vs.core

import math
from typing import Optional, Union, Sequence, Dict, Any

class Range:
    LIMITED = vs.RANGE_LIMITED
    FULL = vs.RANGE_FULL

def resolve_range(value, name: str) -> int | None:
    if value is None:
        return None

    if value in (vs.RANGE_LIMITED, vs.RANGE_FULL):
        return value

    if isinstance(value, str):
        value = value.lower()
        if value == "limited":
            return vs.RANGE_LIMITED
        if value == "full":
            return vs.RANGE_FULL

    raise ValueError(f'"{name}" must be RANGE_LIMITED, RANGE_FULL, "limited", or "full".')

def GetPlane(clip: vs.VideoNode, plane: int = 0) -> vs.VideoNode:
    """
    Extract a single plane from a clip as a GRAY clip.

    Parameters
    ----------
    clip
        Source clip.
    plane
        Plane index to extract. Defaults to ``0``.

    Returns
    -------
    vs.VideoNode
        A single-plane GRAY clip.

    Raises
    ------
    TypeError
        If ``clip`` is not a VideoNode or ``plane`` is not an integer.
    ValueError
        If the clip has no constant format or the plane index is invalid.
    """
    if not isinstance(clip, vs.VideoNode):
        raise TypeError('"clip" must be a vs.VideoNode.')

    if not isinstance(plane, int):
        raise TypeError('"plane" must be an int.')

    fmt = clip.format
    if fmt is None:
        raise ValueError('"clip" must have a constant format.')

    if not 0 <= plane < fmt.num_planes:
        raise ValueError(
            f'"plane" must be in the range [0, {fmt.num_planes}).'
        )

    # No-op for already single-plane clips.
    if fmt.num_planes == 1 and plane == 0:
        return clip

    return core.std.ShufflePlanes(clip, plane, vs.GRAY)
    
def cround(x: float) -> int:
    return math.floor(x + 0.5) if x > 0 else math.ceil(x - 0.5)

def m4(x: Union[float, int]) -> int:
    return 16 if x < 16 else cround(x / 4) * 4

def scale(value, peak):
    return cround(value * peak / 255) if peak != 1 else value / 255

def clamp(minimum, value, maximum):
    return int(max(minimum, min(round(value), maximum)))

def scale_value(value: Union[int, float],
                input_depth: int,
                output_depth: int,
                range_in: int = 0,
                range: Optional[int] = None,
                scale_offsets: bool = False,
                chroma: bool = False,
                ) -> Union[int, float]:
    """Scales a given numeric value between bit depths, sample types, and/or ranges.

    >>> scale_value(16, 8, 32, range_in=Range.LIMITED)
    0.0730593607305936
    >>> scale_value(16, 8, 32, range_in=Range.LIMITED, scale_offsets=True)
    0.0
    >>> scale_value(16, 8, 32, range_in=Range.LIMITED, scale_offsets=True, chroma=True)
    -0.5

    :param value:          Numeric value to be scaled.
    :param input_depth:    Bit depth of the `value` parameter. Use ``32`` for float sample type.
    :param output_depth:   Bit depth to scale the input `value` to.
    :param range_in:       Pixel range of the input `value`. No clamping is performed. See :class:`Range`.
    :param range:          Pixel range of the output `value`. No clamping is performed. See :class:`Range`.
    :param scale_offsets:  Whether or not to apply YUV offsets to float chroma and/or TV range integer values.
        (When scaling a TV range value of ``16`` to float, setting this to ``True`` will return ``0.0``
        rather than ``0.073059...``)
    :param chroma:        Whether or not to treat values as chroma instead of luma.

    :return:              Scaled numeric value.
    """
    range_in = resolve_range(range_in, "range_in")
    range = resolve_range(range, "range")
    range = range_in if range is None else range

    if input_depth == 32:
        range_in = 1

    if output_depth == 32:
        range = 1

    def peak_pixel_value(bits: int, range_: Union[int, types.Range], chroma_: bool) -> int:
        """
        _
        """
        if bits == 32:
            return 1
        if range_:
            return (1 << bits) - 1
        return (224 if chroma_ else 219) << (bits - 8)

    input_peak = peak_pixel_value(input_depth, range_in, chroma)

    output_peak = peak_pixel_value(output_depth, range, chroma)

    if input_depth == output_depth and range_in == range:
        return value

    if scale_offsets:
        if output_depth == 32 and chroma:
            value -= 128 << (input_depth - 8)
        elif range and not range_in:
            value -= 16 << (input_depth - 8)

    value *= output_peak / input_peak

    if scale_offsets:
        if input_depth == 32 and chroma:
            value += 128 << (output_depth - 8)
        elif range_in and not range:
            value += 16 << (output_depth - 8)

    return value

def Depth(
    src: vs.VideoNode,
    bits: int,
    dither_type: str = 'error_diffusion',
    range: int | str | None = None,
    range_in: int | str | None = None
) -> vs.VideoNode:
    """
    Change the bit depth of a VapourSynth clip.

    Args:
        src: Input clip.
        bits: Target bit depth.
        dither_type: Dithering algorithm used for conversion.
        range: Output range ('limited'/'full' or VapourSynth range constant).
        range_in: Input range ('limited'/'full' or VapourSynth range constant).

    Returns:
        Converted VapourSynth clip.
    """
    src_f = src.format
    src_cf = src_f.color_family
    src_bits = src_f.bits_per_sample
    src_sw = src_f.subsampling_w
    src_sh = src_f.subsampling_h

    dst_st = vs.INTEGER if bits < 32 else vs.FLOAT

    if range == 'limited':
        range = vs.RANGE_LIMITED
    elif range == 'full':
        range = vs.RANGE_FULL

    if range_in == 'limited':
        range_in = vs.RANGE_LIMITED
    elif range_in == 'full':
        range_in = vs.RANGE_FULL

    if (src_bits, range_in) == (bits, range):
        return src

    _is_api4 = hasattr(vs, "__api_version__") and vs.__api_version__.api_major == 4
    query_video_format = core.query_video_format if _is_api4 else core.register_format

    out_f = query_video_format(src_cf, dst_st, bits, src_sw, src_sh)

    return core.resize.Point(src, format=out_f.id, dither_type=dither_type, range=range, range_in=range_in)


def BoxFilter(input: vs.VideoNode, radius: int = 16, radius_v: Optional[int] = None, planes: Optional[Union[int, Sequence[int]]] = None,
              fmtc_conv: int = 0, radius_thr: Optional[int] = None,
              resample_args: Optional[Dict[str, Any]] = None, keep_bits: bool = True,
              depth_args: Optional[Dict[str, Any]] = None
              ) -> vs.VideoNode:
    '''Box filter

    Performs a box filtering on the input clip.
    Box filtering consists in averaging all the pixels in a square area whose center is the output pixel.
    You can approximate a large gaussian filtering by cascading a few box filters.

    Args:
        input: Input clip to be filtered.

        radius, radius_v: (int) Size of the averaged square. The size is (radius*2-1) * (radius*2-1).
            If "radius_v" is None, it will be set to "radius".
            Default is 16.

        planes: (int []) Whether to process the corresponding plane. By default, every plane will be processed.
            The unprocessed planes will be copied from the source clip, "input".

        fmtc_conv: (0~2) Whether to use fmtc.resample for convolution.
            It's recommended to input clip without chroma subsampling when using fmtc.resample, otherwise the output may be incorrect.
            0: False. 1: True (except both "radius" and "radius_v" is strictly smaller than 4).
                2: Auto, determined by radius_thr (exclusive).
            Default is 0.

        radius_thr: (int) Threshold of wheter to use fmtc.resample when "fmtc_conv" is 2.
            Default is 11 for integer input and 21 for float input.
            Only works when "fmtc_conv" is enabled.

        resample_args: (dict) Additional parameters passed to core.fmtc.resample in the form of dict.
            It's recommended to set "flt" to True for higher precision, like:
                flt = muf.BoxFilter(src, resample_args=dict(flt=True))
            Only works when "fmtc_conv" is enabled.
            Default is {}.

        keep_bits: (bool) Whether to keep the bitdepth of the output the same as input.
            Only works when "fmtc_conv" is enabled and input is integer.

        depth_args: (dict) Additional parameters passed to mvf.Depth in the form of dict.
            Only works when "fmtc_conv" is enabled, input is integer and "keep_bits" is True.
            Default is {}.

    '''

    funcName = 'BoxFilter'

    if not isinstance(input, vs.VideoNode):
        raise TypeError(funcName + ': \"input\" must be a clip!')

    if planes is None:
        planes = list(range(input.format.num_planes))
    elif isinstance(planes, int):
        planes = [planes]

    if radius_v is None:
        radius_v = radius

    if radius == radius_v == 1:
        return input

    if radius_thr is None:
        radius_thr = 21 if input.format.sample_type == vs.FLOAT else 11 # Values are measured from my experiment

    if resample_args is None:
        resample_args = {}

    if depth_args is None:
        depth_args = {}

    planes2 = [(3 if i in planes else 2) for i in range(input.format.num_planes)]
    width = radius * 2 - 1
    width_v = radius_v * 2 - 1
    kernel = [1 / width] * width
    kernel_v = [1 / width_v] * width_v

    # process
    if input.format.sample_type == vs.FLOAT:
        if core.core_version.release_major < 33:
            raise NotImplementedError(funcName + (': Please update your VapourSynth.'
                'BoxBlur on float sample has not yet been implemented on current version.'))
        elif radius == radius_v == 2 or radius == radius_v == 3:
            return core.std.Convolution(input, [1] * ((radius * 2 - 1) * (radius * 2 - 1)), planes=planes, mode='s')

        else:
            if fmtc_conv == 1 or (fmtc_conv != 0 and radius > radius_thr): # Use fmtc.resample for convolution
                flt = core.fmtc.resample(input, kernel='impulse', impulseh=kernel, impulsev=kernel_v, planes=planes2,
                    cnorm=False, fh=-1, fv=-1, center=False, **resample_args)
                return flt # No bitdepth conversion is required since fmtc.resample outputs the same bitdepth as input

            elif hasattr(core, 'vszip'):
                return core.vszip.BoxBlur(input, hradius=radius-1, vradius=radius_v-1, planes=planes)

            elif core.core_version.release_major >= 39:
                return core.std.BoxBlur(input, hradius=radius-1, vradius=radius_v-1, planes=planes)

            else: # BoxBlur on float sample has not been implemented
                if radius > 1:
                    input = core.std.Convolution(input, [1] * (radius * 2 - 1), planes=planes, mode='h')
                if radius_v > 1:
                    input = core.std.Convolution(input, [1] * (radius_v * 2 - 1), planes=planes, mode='v')
                return input

    else: # input.format.sample_type == vs.INTEGER
        if radius == radius_v == 2 or radius == radius_v == 3:
            return core.std.Convolution(input, [1] * ((radius * 2 - 1) * (radius * 2 - 1)), planes=planes, mode='s')

        else:
            if fmtc_conv == 1 or (fmtc_conv != 0 and radius > radius_thr): # Use fmtc.resample for convolution
                flt = core.fmtc.resample(input, kernel='impulse', impulseh=kernel, impulsev=kernel_v, planes=planes2,
                    cnorm=False, fh=-1, fv=-1, center=False, **resample_args)
                if keep_bits and input.format.bits_per_sample != flt.format.bits_per_sample:
                    flt = mvf.Depth(flt, depth=input.format.bits_per_sample, **depth_args)
                return flt

            elif hasattr(core, 'vszip'):
                return core.vszip.BoxBlur(input, hradius=radius-1, vradius=radius_v-1, planes=planes)

            elif hasattr(core.std, 'BoxBlur'):
                return core.std.BoxBlur(input, hradius=radius-1, vradius=radius_v-1, planes=planes)

            else: # BoxBlur was not found
                if radius > 1:
                    input = core.std.Convolution(input, [1] * (radius * 2 - 1), planes=planes, mode='h')
                if radius_v > 1:
                    input = core.std.Convolution(input, [1] * (radius_v * 2 - 1), planes=planes, mode='v')
                return input