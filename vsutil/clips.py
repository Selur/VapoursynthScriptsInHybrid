"""
Functions that modify/return a clip.
"""
__all__ = ['depth', 'frame2clip', 'get_y', 'insert_clip', 'join', 'plane', 'split']

from typing import Any, List, Optional, Sequence, Union, cast

import vapoursynth as vs

from . import func, info, types

core = vs.core


@func.disallow_variable_format
def depth(clip: vs.VideoNode,
          bitdepth: int,
          /,
          sample_type: Optional[Union[int, vs.SampleType]] = None,
          *,
          range: Optional[Union[int, types.Range]] = None,
          range_in: Optional[Union[int, types.Range]] = None,
          dither_type: Optional[Union[types.Dither, str]] = None,
          ) -> vs.VideoNode:
    """A bit depth converter only using ``vapoursynth.core.resize()`` and ``vapoursynth.Format.replace()``.
    By default, outputs ``vapoursynth.FLOAT`` sample type for 32-bit and ``vapoursynth.INTEGER`` for anything else.

    >>> src_8 = vs.core.std.BlankClip(format=vs.YUV420P8)
    >>> src_10 = depth(src_8, 10)
    >>> src_10.format.name
    'YUV420P10'

    >>> src2_10 = vs.core.std.BlankClip(format=vs.RGB30)
    >>> src2_8 = depth(src2_10, 8, dither_type=Dither.RANDOM)  # override default dither behavior
    >>> src2_8.format.name
    'RGB24'

    :param clip:         Input clip.
    :param bitdepth:     Desired `bits_per_sample` of output clip.
    :param sample_type:  Desired `sample_type` of output clip. Allows overriding default float/integer behavior.
                         Accepts ``vapoursynth.SampleType`` enums ``vapoursynth.INTEGER`` and ``vapoursynth.FLOAT``
                         or their values, ``0`` and ``1`` respectively.
    :param range:        Output pixel range (defaults to input `clip`'s range). See :class:`Range`.
    :param range_in:     Input pixel range (defaults to input `clip`'s range). See :class:`Range`.
    :param dither_type:  Dithering algorithm. Allows overriding default dithering behavior. See :class:`Dither`.

        Defaults to :attr:`Dither.ERROR_DIFFUSION`, or Floyd-Steinberg error diffusion, when downsampling,
        converting between ranges, or upsampling full range input.
        Defaults to :attr:`Dither.NONE`, or round to nearest, otherwise.
        See `_should_dither()` comments for more information.

    :return:             Converted clip with desired bit depth and sample type. ``ColorFamily`` will be same as input.
    """
    sample_type = types.resolve_enum(vs.SampleType, sample_type, 'sample_type', depth)
    range = types.resolve_enum(types.Range, range, 'range', depth)
    range_in = types.resolve_enum(types.Range, range_in, 'range_in', depth)
    dither_type = types.resolve_enum(types.Dither, dither_type, 'dither_type', depth)

    curr_depth = info.get_depth(clip)
    sample_type = func.fallback(sample_type, vs.FLOAT if bitdepth == 32 else vs.INTEGER)

    if (curr_depth, clip.format.sample_type, range_in) == (bitdepth, sample_type, range):
        return clip

    should_dither = _should_dither(curr_depth, bitdepth, range_in, range, clip.format.sample_type, sample_type)
    dither_type = func.fallback(dither_type, types.Dither.ERROR_DIFFUSION if should_dither else types.Dither.NONE)

    new_format = clip.format.replace(bits_per_sample=bitdepth, sample_type=sample_type).id

    return clip.resize.Point(format=new_format, range=range, range_in=range_in, dither_type=dither_type)


_unused: Any = []


def frame2clip(frame: vs.VideoFrame, /, *, enforce_cache=_unused) -> vs.VideoNode:
    """Converts a VapourSynth frame to a clip.

    :param frame:          The frame to convert.
    :param enforce_cache:  Forcibly add a cache, even if the ``vapoursynth`` module has this feature disabled.

    :return: A one-frame clip that yields the `frame` passed to the function.
    """
    if enforce_cache is not _unused:
        import warnings
        warnings.warn("enforce_cache is deprecated.", DeprecationWarning)

    bc = core.std.BlankClip(
        width=frame.width,
        height=frame.height,
        length=1,
        fpsnum=1,
        fpsden=1,
        format=frame.format.id
    )
    frame = frame.copy()
    result = bc.std.ModifyFrame([bc], lambda n, f: frame.copy())
    return result


@func.disallow_variable_format
def get_y(clip: vs.VideoNode, /) -> vs.VideoNode:
    """Helper to get the luma plane of a clip.

    If passed a single-plane ``vapoursynth.GRAY`` clip, :func:`plane` will assume it to `be` the luma plane
    itself and returns the `clip` (no-op).

    :param clip: Input clip.

    :return:     Luma plane of the input `clip`. Will return the input `clip` if it is a single-plane grayscale clip.
    """
    if clip.format.color_family not in (vs.YUV, vs.GRAY):
        raise ValueError('The clip must have a luma plane.')
    return plane(clip, 0)


def insert_clip(clip: vs.VideoNode, /, insert: vs.VideoNode, start_frame: int) -> vs.VideoNode:
    """Convenience method to insert a shorter clip into a longer one.

    The `insert` clip cannot go beyond the last frame of the source `clip` or an exception is raised.
    The `insert` clip frames replace the `clip` frames, unlike a normal splice-in.

    :param clip:         Longer clip to insert shorter clip into.
    :param insert:       Insert clip.
    :param start_frame:  First frame of the longer `clip` to replace.

    :return:             Longer clip with frames replaced by the shorter clip.
    """
    if start_frame == 0:
        return insert + clip[insert.num_frames:]
    pre = clip[:start_frame]
    frame_after_insert = start_frame + insert.num_frames
    if frame_after_insert > clip.num_frames:
        raise ValueError('Inserted clip is too long.')
    if frame_after_insert == clip.num_frames:
        return pre + insert
    post = clip[start_frame + insert.num_frames:]
    return pre + insert + post


def join(planes: Sequence[vs.VideoNode], family: vs.ColorFamily = vs.YUV) -> vs.VideoNode:
    """Joins the supplied sequence of planes into a single VideoNode (defaults to YUV).

    >>> planes = [Y, U, V]
    >>> clip_YUV = join(planes)
    >>> plane = core.std.BlankClip(format=vs.GRAY8)
    >>> clip_GRAY = join([plane], family=vs.GRAY)

    :param planes:  Sequence of one-plane ``vapoursynth.GRAY`` clips to merge.
    :param family:  Output color family.

    :return:        Merged clip of the supplied `planes`.
    """
    return planes[0] if len(planes) == 1 and family == vs.GRAY \
        else core.std.ShufflePlanes(planes, [0, 0, 0], family)


@func.disallow_variable_format
def plane(clip: vs.VideoNode, planeno: int, /) -> vs.VideoNode:
    """Extracts the plane with the given index from the input clip.

    If given a one-plane clip and ``planeno=0``, returns `clip` (no-op).

    >>> src = vs.core.std.BlankClip(format=vs.YUV420P8)
    >>> V = plane(src, 2)

    :param clip:     The clip to extract the plane from.
    :param planeno:  The index of which plane to extract.

    :return:         A grayscale clip that only contains the given plane.
    """
    if clip.format.num_planes == 1 and planeno == 0:
        return clip
    return core.std.ShufflePlanes(clip, planeno, vs.GRAY)


@func.disallow_variable_format
def split(clip: vs.VideoNode, /) -> List[vs.VideoNode]:
    """Returns a list of planes (VideoNodes) from the given input clip.

    >>> src = vs.core.std.BlankClip(format=vs.RGB27)
    >>> R, G, B = split(src)
    >>> src2 = vs.core.std.BlankClip(format=vs.GRAY8)
    >>> split(src2)
    [<vapoursynth.VideoNode object>]  # always returns a list, even if single plane

    :param clip:  Input clip.

    :return:      List of planes from the input `clip`.
    """
    return [clip] if clip.format.num_planes == 1 else cast(List[vs.VideoNode], clip.std.SplitPlanes())


def _should_dither(in_bits: int,
                   out_bits: int,
                   in_range: Optional[types.Range] = None,
                   out_range: Optional[types.Range] = None,
                   in_sample_type: Optional[vs.SampleType] = None,
                   out_sample_type: Optional[vs.SampleType] = None,
                   ) -> bool:
    """
    Determines whether dithering is needed for a given depth/range/sample_type conversion.

    If an input range is specified, and output range *should* be specified otherwise it assumes a range conversion.

    For an explanation of when dithering is needed:
        - Dithering is NEVER needed if the conversion results in a float sample type.
        - Dithering is ALWAYS needed for a range conversion (i.e. full to limited or vice-versa).
        - Dithering is ALWAYS needed to convert a float sample type to an integer sample type.
        - Dithering is needed when upsampling full range content with the exception of 8 -> 16 bit upsampling,
          as this is simply (0-255) * 257 -> (0-65535).
        - Dithering is needed when downsampling limited or full range.

    Dithering is theoretically needed when converting from an integer depth greater than 10 to half float,
    despite the higher bit depth, but zimg's internal resampler currently does not dither for float output.
    """
    out_sample_type = func.fallback(out_sample_type, vs.FLOAT if out_bits == 32 else vs.INTEGER)
    in_sample_type = func.fallback(in_sample_type, vs.FLOAT if in_bits == 32 else vs.INTEGER)

    if out_sample_type == vs.FLOAT:
        return False

    range_conversion = in_range != out_range
    float_to_int = in_sample_type == vs.FLOAT
    upsampling = in_bits < out_bits
    downsampling = in_bits > out_bits

    return bool(range_conversion
                or float_to_int
                or (in_range == types.Range.FULL and upsampling and (in_bits, out_bits) != (8, 16))
                or downsampling)
