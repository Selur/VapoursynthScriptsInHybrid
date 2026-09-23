
from __future__ import annotations

import vapoursynth as vs
core = vs.core

import math
import warnings
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

def Padding(clip: vs.VideoNode, left: int = 0, right: int = 0, top: int = 0, bottom: int = 0) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('Padding: this is not a clip')

    if left < 0 or right < 0 or top < 0 or bottom < 0:
        raise vs.Error('Padding: border size to pad must not be negative')

    width = clip.width + left + right
    height = clip.height + top + bottom

    return clip.resize.Point(width, height, src_left=-left, src_top=-top, src_width=width, src_height=height)

def DitherLumaRebuild(src: vs.VideoNode, s0: float = 2.0, c: float = 0.0625, chroma: bool = True, tv_range: bool = True) -> vs.VideoNode:
    '''Converts luma (and chroma) to PC levels, and optionally allows tweaking for pumping up the darks. (for the clip to be fed to motion search only)'''
    # tv_range=False: the input is already full range, only the dark-boosting curve is applied and chroma is left alone.
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('DitherLumaRebuild: this is not a clip')

    if src.format.color_family == vs.RGB:
        raise vs.Error('DitherLumaRebuild: RGB format is not supported')

    is_gray = src.format.color_family == vs.GRAY
    is_integer = src.format.sample_type == vs.INTEGER

    bits = src.format.bits_per_sample
    neutral = 1 << (bits - 1)
    peak = (1 << bits) - 1

    k = (s0 - 1) * c
    if not is_integer:
        t, out = 'x 0 max 1 min', ''
    elif tv_range:
        t, out = f'x {scale_value(16, 8, bits)} - {scale_value(219, 8, bits)} / 0 max 1 min', f'{scale_value(256, 8, bits)} *'
    else:
        t, out = f'x {peak} / 0 max 1 min', f'{peak} *'
    e = f'{k} {1 + c} {(1 + c) * c} {t} {c} + / - * {t} 1 {k} - * + ' + out
    EXPR = get_expr()
    return EXPR(src, expr=e if is_gray else [e, f'x {neutral} - 128 * 112 / {neutral} +' if chroma and is_integer and tv_range else ''])

def is_limited_range(clip: vs.VideoNode, default: bool = True) -> bool:
    '''True if frame 0 is tagged as limited (TV) range; `default` if the clip carries no range property.'''
    prop_name = '_Range' if core.core_version.release_major >= 74 else '_ColorRange'
    with clip.get_frame(0) as frame:
        value = frame.props.get(prop_name)
    return default if value is None else value == vs.RANGE_LIMITED

def get_expr():
    '''Return the best Expr backend available, in order of preference: akarin, cranexpr, std.'''
    if hasattr(core, 'akarin'):
        return core.akarin.Expr
    if hasattr(core, 'cranexpr'):
        return core.cranexpr.Expr
    return core.std.Expr

def get_rg(is_float: bool = False):
    '''Return the best RemoveGrain implementation available, in order of preference: zsmooth, rgsf (float only), rgvs.'''
    if hasattr(core, 'zsmooth'):
        return core.zsmooth.RemoveGrain
    if is_float and hasattr(core, 'rgsf'):
        return core.rgsf.RemoveGrain
    return core.rgvs.RemoveGrain

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

        depth_args: (dict) Additional parameters passed to Depth in the form of dict.
            Only works when "fmtc_conv" is enabled, input is integer and "keep_bits" is True.
            Keys should match helpers.Depth's signature (range, range_in, dither_type)"
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
                    flt = Depth(flt, bits=input.format.bits_per_sample, **depth_args)
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
                
## NNEDI3 ----------------------------------------------------------------------------------------
# (namespace, function, name of its device argument, parameters only it understands)
_NNEDI3_IMPLS = (
    ('sneedif',  'NNEDI3',     'device',       ('dw', 'transpose_first')),
    ('nnedi3vk', 'NNEDI3',     'device_index', ('coopvec', 'num_streams')),
    ('vszipcu',  'NNEDI3',     'device_id',    ('num_streams',)),
    ('nnedi3cl', 'NNEDI3CL',   'device',       ('dw',)),
    ('znedi3',   'nnedi3',     None,           ('opt', 'int16_prescreener', 'int16_predictor', 'exp', 'show_mask')),
    ('nnedi3',   'nnedi3',     None,           ('opt', 'int16_prescreener', 'int16_predictor', 'exp')),
)
_NNEDI3_GPU = ('sneedif', 'nnedi3vk', 'vszipcu', 'nnedi3cl')
_NNEDI3_CPU = ('znedi3', 'nnedi3')
# Parameters every implementation understands; anything else is dropped unless it is listed as an
# extra of the chosen one.
_NNEDI3_COMMON = ('field', 'dh', 'planes', 'nsize', 'nns', 'qual', 'etype', 'pscrn')


def _nnedi3CanRun(namespace: str, kwargs: Dict[str, Any]) -> bool:
    if not hasattr(core, namespace):
        return False
    # sneedif is an OpenCL port whose prescreener only covers 1 and 2; anything else does not fail
    # cleanly but blows up while building the kernel.
    if namespace == 'sneedif' and kwargs.get('pscrn') not in (None, 1, 2):
        return False
    # Only sneedif (and nnedi3cl) can double the width in one call.
    if kwargs.get('dw') and namespace not in ('sneedif', 'nnedi3cl'):
        return False
    return True


def NNEDI3(clip: vs.VideoNode, gpu: Optional[bool] = None, device: Optional[int] = None,
           **kwargs) -> vs.VideoNode:
    '''Calls the NNEDI3 implementation that is loaded.

    Looked for in this order, the first one that is loaded and can serve the call wins:
      gpu true/None : sneedif, nnedi3vk, vszipcu, nnedi3cl, znedi3, nnedi3
      gpu false     : znedi3, nnedi3, sneedif, nnedi3vk, vszipcu, nnedi3cl

    It is a preference, not a restriction - with `gpu=False` and only a GPU port loaded that port
    is still used. Which of them exist is decided by whoever loaded the plugins.

    Args:
        gpu: The caller's own opencl/gpu switch. Only reorders the search.
        device: Device number, passed on under whatever name the implementation uses.

    `pscrn` above 2 and `dw` are not available everywhere - a call using them skips the
    implementations that cannot serve it.
    '''
    order = (_NNEDI3_CPU + _NNEDI3_GPU) if gpu is False else (_NNEDI3_GPU + _NNEDI3_CPU)
    known = {ns: (fn, dev, extra) for ns, fn, dev, extra in _NNEDI3_IMPLS}
    for namespace in order:
        if not _nnedi3CanRun(namespace, kwargs):
            continue
        function, deviceArg, extras = known[namespace]
        args = {k: v for k, v in kwargs.items() if k in _NNEDI3_COMMON or k in extras}
        # Callers use -1 for "let it choose"; nnedi3vk and vszipcu reject negative ids outright,
        # so that is expressed by leaving the argument out.
        if deviceArg is not None and device is not None and device >= 0:
            args[deviceArg] = device
        return getattr(getattr(core, namespace), function)(clip, **args)
    raise vs.Error('NNEDI3: no usable implementation loaded '
                   '(looked for %s)' % ', '.join(order))


## EEDI3 -----------------------------------------------------------------------------------------
# (namespace, function, name of its device argument, parameters beyond the common set)
# Older plugins stay in the list for compatibility, even where they are rarely offered.
_EEDI3_IMPLS = (
    ('eedi3vk2', 'EEDI3',   'device_index', ('planes', 'mclip', 'list_device', 'num_streams')),
    ('vszipcu',  'EEDI3',   'device_id',    ('num_streams',)),
    ('vszipcl',  'EEDI3',   'device_id',    ('num_streams', 'tune')),
    ('eedi3vk',  'EEDI3',   'device_index', ('planes', 'mclip')),
    ('eedi3m',   'EEDI3CL', 'device',       ('planes', 'mclip')),
    ('vszip',    'EEDI3',   None,           ('mclip',)),
    ('eedi3m',   'EEDI3',   None,           ('planes', 'mclip', 'ucubic', 'cost3', 'opt')),
)
_EEDI3_GPU = (('eedi3vk2', 'EEDI3'), ('vszipcu', 'EEDI3'), ('vszipcl', 'EEDI3'),
              ('eedi3vk', 'EEDI3'), ('eedi3m', 'EEDI3CL'))
_EEDI3_CPU = (('vszip', 'EEDI3'), ('eedi3m', 'EEDI3'))
_EEDI3_COMMON = ('field', 'dh', 'alpha', 'beta', 'gamma', 'nrad', 'mdis', 'hp',
                 'vcheck', 'vthresh0', 'vthresh1', 'vthresh2', 'sclip')
# What each implementation expects `mclip` to look like - and the two flavours are mutually
# exclusive: eedi3m/eedi3vk want the clip's own format ("mclip's format and dimensions don't
# match"), vszip wants Gray ("mclip must be Gray"). Switching implementations behind the caller's
# back would therefore turn a valid mask into an invalid one.
_EEDI3_MCLIP = {'vszip': 'gray', 'eedi3m': 'same', 'eedi3vk2': 'same', 'eedi3vk': 'same'}


def _hasFunction(namespace: str, function: str) -> bool:
    return hasattr(core, namespace) and hasattr(getattr(core, namespace), function)


def EEDI3(clip: vs.VideoNode, gpu: Optional[bool] = None, device: Optional[int] = None,
          **kwargs) -> vs.VideoNode:
    '''Calls the EEDI3 implementation that is loaded.

    Looked for in this order, the first one that is loaded and can serve the call wins:
      gpu true/None : eedi3vk2, vszipcu, vszipcl, eedi3vk, eedi3m.EEDI3CL, vszip, eedi3m
      gpu false     : vszip, eedi3m, then the GPU ones

    It is a preference, not a restriction. Which of them exist is decided by whoever loaded the
    plugins.

    Not every implementation takes every argument - vszip and its GPU siblings have no `planes`
    (they always process every plane), and the CUDA/OpenCL ones no `mclip`. A call that uses one of
    those skips the implementations that cannot serve it, so the picture stays the same. A `planes`
    that names *all* planes is dropped instead: it asks for what those implementations do anyway.
    '''
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    allPlanes = list(range(clip.format.num_planes))
    order = (_EEDI3_CPU + _EEDI3_GPU) if gpu is False else (_EEDI3_GPU + _EEDI3_CPU)
    known = {(ns, fn): (dev, extra) for ns, fn, dev, extra in _EEDI3_IMPLS}
    for namespace, function in order:
        if not _hasFunction(namespace, function):
            continue
        deviceArg, extras = known[(namespace, function)]
        allowed = _EEDI3_COMMON + extras
        args = dict(kwargs)
        if 'planes' not in extras and 'planes' in args:
            wanted = args['planes']
            if isinstance(wanted, int): wanted = [wanted]
            if sorted(wanted) != allPlanes:
                continue          # a real restriction this implementation cannot express
            del args['planes']    # asks for every plane, which is what it does anyway
        if any(k not in allowed for k in args):
            continue
        if 'mclip' in args:
            wantsGray = _EEDI3_MCLIP.get(namespace) == 'gray'
            isGray = args['mclip'].format.color_family == vs.GRAY
            if wantsGray != isGray:
                continue      # this one would reject the mask the caller built
        if deviceArg is not None and device is not None and device >= 0:
            args[deviceArg] = device
        return getattr(getattr(core, namespace), function)(clip, **args)
    raise vs.Error('EEDI3: no usable implementation loaded for this call '
                   '(looked for %s)' % ', '.join('%s.%s' % nf for nf in order))


## DFTTest ---------------------------------------------------------------------------------------
# Parameters core.dfttest.DFTTest knows but the GPU implementations do not.
_DFTTEST_NOT_IN_VSZIPCU = ('smode', 'tmode', 'tosize', 'nlocation', 'alpha', 'opt')
_DFTTEST_NOT_IN_DFTTEST2 = ('tmode', 'tosize', 'opt')
# dfttest2 backends that are fixed to sbsize=16 and tbsize in (1, 3, 5, 7).
_DFTTEST2_FIXED_BLOCK = ('dfttest2_nvrtc', 'dfttest2_hiprtc', 'dfttest2_cpu', 'dfttest2_gcc')
# ... and the ones that take any block size.
_DFTTEST2_ANY_BLOCK = ('dfttest2_cuda', 'dfttest2_hip')


def _vszipcuCanRun(kwargs: Dict[str, Any]) -> bool:
    if not hasattr(core, 'vszipcu'):
        return False
    # An NVRTC port: 16x16 spatial window only, and it rejects an even temporal window.
    if kwargs.get('sbsize', 16) != 16 or kwargs.get('tbsize', 3) % 2 == 0:
        return False
    return not any(name in kwargs for name in _DFTTEST_NOT_IN_VSZIPCU)


def _dfttest2CanRun(kwargs: Dict[str, Any]) -> bool:
    if any(name in kwargs for name in _DFTTEST_NOT_IN_DFTTEST2):
        return False
    if kwargs.get('sbsize', 16) == 16 and kwargs.get('tbsize', 3) in (1, 3, 5, 7):
        return any(hasattr(core, name) for name in _DFTTEST2_FIXED_BLOCK + _DFTTEST2_ANY_BLOCK)
    return any(hasattr(core, name) for name in _DFTTEST2_ANY_BLOCK)


def DFTTest(clip: vs.VideoNode, cuda: Optional[bool] = None, **kwargs) -> vs.VideoNode:
    '''Calls the first DFTTest implementation that is loaded, GPU ones first.

    Looked for in this order: core.vszipcu.DFTTest (CUDA), dfttest2.DFTTest (CUDA) and
    core.dfttest.DFTTest (CPU). Which of them is available is decided by whoever loaded the
    plugins.

    Args:
        cuda: False forces the CPU implementation, True and None look for a GPU one first.
            Kept for the callers that have their own cuda/opencl/gpu switch.

    An implementation is only used when it can actually handle the call: neither GPU one takes
    every parameter, and most of their backends are fixed to a spatial block size of 16. Calls
    they cannot serve use the CPU version, which then has to be loaded as well.
    '''
    if cuda is not False:
        if _vszipcuCanRun(kwargs):
            return core.vszipcu.DFTTest(clip, **kwargs)
        if _dfttest2CanRun(kwargs):
            import dfttest2
            # Let dfttest2 pick its backend: NVRTC where it fits, cuFFT for everything else.
            return dfttest2.DFTTest(clip, **kwargs)
    return core.dfttest.DFTTest(clip, **kwargs)


# The BM3D implementations with the BM3DCUDA interface, in the order they are preferred when more than one is loaded.
_BM3D_IMPLEMENTATIONS = ('bm3dcuda', 'bm3dhip', 'bm3dmetal', 'bm3dcpu')


def BM3D(clip: vs.VideoNode, sigma: Sequence[float], radius: int = 0, block_step: int = 8, bm_range: int = 9,
         ps_num: int = 2, ps_range: int = 4, chroma: bool = False, device_id: Optional[int] = None,
         backend: Optional[str] = None) -> vs.VideoNode:
    '''Basic BM3D estimate (aggregated when radius > 0) on the BM3D plugin that is loaded.

    The clip must be 32-bit float, chroma=True (CBM3D, block matching on luma) needs YUV444PS; the result is
    32-bit float. Looked for in this order: bm3dcuda, bm3dhip, bm3dmetal, bm3dcpu, then core.bm3d. backend
    ('bm3dcuda', 'bm3dhip', 'bm3dmetal', 'bm3dcpu' or 'bm3d') moves that implementation to the front, a GPU one
    followed by bm3dcpu; it matters where every plugin is autoloaded (Linux, macOS), with explicit loading only
    the chosen one is there anyway. A GPU plugin that cannot create its filter (no usable card, e.g. an AMD iGPU the bundled ROCm runtime does not support) is skipped with
    a warning, so a GPU choice should be loaded together with bm3dcpu. With the BM3DCUDA plugins, planes with
    sigma 0 are undefined unless chroma=True.
    '''
    if clip.format.sample_type != vs.FLOAT or clip.format.bits_per_sample != 32:
        raise vs.Error('BM3D: the clip must be 32-bit float')
    order = _BM3D_IMPLEMENTATIONS + ('bm3d',)
    if backend is not None:
        if backend not in order:
            raise vs.Error(f'BM3D: unknown backend "{backend}", use one of {", ".join(order)}')
        if not hasattr(core, backend):
            warnings.warn(f'BM3D: {backend} is not loaded, trying the next implementation')
        # A GPU choice falls back to bm3dcpu first, as where only the chosen port and bm3dcpu are loaded.
        first = (backend, 'bm3dcpu') if backend in ('bm3dcuda', 'bm3dhip', 'bm3dmetal') else (backend,)
        order = first + tuple(name for name in order if name not in first)
    for name in order:
        if not hasattr(core, name):
            continue
        if name == 'bm3d':
            if radius == 0:
                return core.bm3d.Basic(clip, sigma=list(sigma), block_step=block_step, bm_range=bm_range)
            basic = core.bm3d.VBasic(clip, sigma=list(sigma), radius=radius, block_step=block_step,
                                     bm_range=bm_range, ps_num=ps_num, ps_range=ps_range)
            return core.bm3d.VAggregate(basic, radius=radius, sample=1)
        kwargs = dict(sigma=list(sigma), radius=radius, block_step=block_step, bm_range=bm_range, ps_num=ps_num,
                      ps_range=ps_range, chroma=chroma)
        if name != 'bm3dcpu' and device_id is not None and device_id >= 0:
            kwargs['device_id'] = device_id
        try:
            return getattr(core, name).BM3Dv2(clip, **kwargs)
        except vs.Error as error:
            warnings.warn(f'BM3D: {name} cannot be used ({str(error).splitlines()[0]}), trying the next implementation')
    raise vs.Error('BM3D: no usable BM3D plugin is loaded')


# The NLMeans implementations, in the order they are preferred when more than one is loaded.
# The name of the call differs only for KNLMeansCL.
_NLMEANS_IMPLEMENTATIONS = (('nlm_ispc', 'NLMeans'), ('nlm_cuda', 'NLMeans'), ('vszipcu', 'NLMeans'),
                            ('vszipcl', 'NLMeans'), ('knlm', 'KNLMeansCL'))


def NLMeans(
    clip: vs.VideoNode,
    d: Optional[int] = None,
    a: Optional[int] = None,
    s: Optional[int] = None,
    h: Optional[float] = None,
    channels: Optional[str] = None,
    wmode: Optional[int] = None,
    wref: Optional[float] = None,
    rclip: Optional[vs.VideoNode] = None,
    device_type: Optional[str] = None,
    device_id: Optional[int] = None,
) -> vs.VideoNode:
    """Calls the NLMeans implementation that is loaded, with the arguments it understands.

    d, a, s, h, channels, wmode, wref and rclip mean the same everywhere. device_type exists in
    knlm alone, device_id in everything but nlm_ispc; a negative device_id means "the plugin
    picks" and is left out.
    """
    for namespace, name in _NLMEANS_IMPLEMENTATIONS:
        if not hasattr(core, namespace):
            continue
        kwargs = {'d': d, 'a': a, 's': s, 'h': h, 'channels': channels, 'wmode': wmode,
                  'wref': wref, 'rclip': rclip}
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        if device_id is not None and device_id >= 0 and namespace != 'nlm_ispc':
            kwargs['device_id'] = device_id
        if device_type is not None and namespace == 'knlm':
            kwargs['device_type'] = device_type
        return getattr(getattr(core, namespace), name)(clip, **kwargs)
    raise vs.Error('NLMeans: none of %s is loaded'
                   % ', '.join(namespace for namespace, _ in _NLMEANS_IMPLEMENTATIONS))


# Taken from havsfunc
def KNLMeansCL(
    clip: vs.VideoNode,
    d: Optional[int] = None,
    a: Optional[int] = None,
    s: Optional[int] = None,
    h: Optional[float] = None,
    wmode: Optional[int] = None,
    wref: Optional[float] = None,
    device_type: Optional[str] = None,
    device_id: Optional[int] = None,
) -> vs.VideoNode:
    """Runs NLMeans over every plane, in the one or two passes the clip's sampling calls for."""
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('KNLMeansCL: this is not a clip')

    if clip.format.color_family not in (vs.YUV, vs.GRAY):
        raise vs.Error('KNLMeansCL: this wrapper is intended to be used only for YUV and GRAY format')

    args = dict(d=d, a=a, s=s, h=h, wmode=wmode, wref=wref, device_type=device_type, device_id=device_id)
    # GRAY has no chroma to walk over, and every implementation only accepts channels='Y'
    # there - 'YUV' wants 4:4:4 and 'UV' wants a YUV clip.
    if clip.format.color_family == vs.GRAY:
        return NLMeans(clip, channels='Y', **args)

    # 'YUV' measures the patch distance over all three planes at once and therefore needs one
    # chroma pixel per luma pixel; subsampled clips get luma and chroma in two passes instead.
    if clip.format.subsampling_w > 0 or clip.format.subsampling_h > 0:
        return NLMeans(NLMeans(clip, channels='Y', **args), channels='UV', **args)
    return NLMeans(clip, channels='YUV', **args)
