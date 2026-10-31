from __future__ import annotations

import vapoursynth as vs
core = vs.core

from typing import List, Optional, Mapping, Sequence
from helpers import get_expr, is_limited_range

PLACEMENTS = ('above', 'below', 'overlay_top', 'overlay_bottom')
# BT.709 luma weights and chroma scales of the coloured trace
_KR, _KG, _KB = 0.2126, 0.7152, 0.0722
_CB_SCALE, _CR_SCALE = 2.0 * (1.0 - _KB), 2.0 * (1.0 - _KR)
# linear light to the BT.709 transfer curve, clamped to 0..1 (the beam exceeds 1 when it is intense)
_CURVE = 'x 0 max 1 min dup 0.018 < swap 4.5 * x 0 max 1 min 0.45 pow 1.099 * 0.099 - ?'


def _plane_exprs(fmt: vs.VideoFormat, limited: bool, color: Sequence[float]) -> List[str]:
    '''One expression per plane of `fmt` on the curved beam brightness `t`: colour, matrix and range in a single pass.'''
    peak = 1.0 if fmt.sample_type == vs.FLOAT else float((1 << fmt.bits_per_sample) - 1)
    r, g, b = (float(c) for c in color)
    if fmt.color_family == vs.RGB:
        return [f'{_CURVE} {c * peak} *' for c in (r, g, b)]
    y = _KR * r + _KG * g + _KB * b
    cb, cr = (b - y) / _CB_SCALE, (r - y) / _CR_SCALE
    if fmt.sample_type == vs.FLOAT:
        scale_y, scale_c, offset_y, offset_c = 1.0, 1.0, 0.0, 0.0
    elif limited:
        unit = float(1 << (fmt.bits_per_sample - 8))
        scale_y, scale_c, offset_y, offset_c = 219.0 * unit, 224.0 * unit, 16.0 * unit, 128.0 * unit
    else:
        scale_y, scale_c, offset_y, offset_c = peak, peak, 0.0, float(1 << (fmt.bits_per_sample - 1))
    exprs = [f'{_CURVE} {y * scale_y} * {offset_y} +']
    if fmt.num_planes == 3:
        exprs += [f'{_CURVE} {cb * scale_c} * {offset_c} +', f'{_CURVE} {cr * scale_c} * {offset_c} +']
    return exprs


def scope(clip: vs.VideoNode, audio: vs.AudioNode, mode: int = 2, placement: str = 'below', height: Optional[int] = None,
          color: Sequence[float] = (1.0, 1.0, 1.0), beam_size: Optional[float] = None, beam_gain: float = 1.0,
          y_gain: float = 1.0, y_ofs: float = 0.0, tools: Optional[Mapping[str, str]] = None) -> vs.VideoNode:
    '''Analog oscilloscope of `audio` (fw.scope, Faveworm) stacked onto or laid over `clip`, in sync with its frames.

    mode: fw.scope mode, 0 trigger, 1 free-running, 2 scrolling, 3 X-Y; X-Y with mono audio falls back to scrolling.
    placement: 'above' or 'below' stacks a scope of `height` lines onto the picture, 'overlay_top' or 'overlay_bottom'
        draws the trace over the picture's top or bottom `height` lines.
    height: scope height in pixels; default a quarter of the clip, mod 4, at least 64.
    color: red, green and blue factors on the beam brightness.
    beam_size: beam diameter in pixels; default grows with the scope height (height / 34, between 1.5 and 8: 8 on a
        quarter of an HD picture), the fw.scope default of 2 is hardly visible there.
    beam_gain: linear gain on the beam brightness; multiplied by fps / 50 above 50 fps, because fw.scope integrates
        the beam over one frame and a 960 fps frame collects a twentieth of the light of a 50 fps frame.
    y_gain: linear gain on the signal (vertical scale), 1 maps full scale to the scope height.
    y_ofs: vertical shift of the trace, -1 bottom, 0 centre, 1 top.
    tools: tools['expr'] picks the Expr implementation.

    The audio is padded with silence to the clip's length: faveworm aborts the whole process when a frame lies past
    the end of the audio. A clip without a frame rate comes back unchanged. The beam brightness goes from linear light
    to BT.709 and straight into the clip's format with one expression per plane (colour, matrix, range and chroma
    location of the clip; frame 0's _Range and _ChromaLocation, YUV limited and GRAY/RGB full when untagged); the
    scope carries the clip's frame properties, an overlay keeps the picture visible where the beam is dark.
    '''
    if placement not in PLACEMENTS:
        raise vs.Error(f'audioscope.scope: placement "{placement}" is unknown, use one of {", ".join(PLACEMENTS)}')
    if clip.fps.numerator == 0 or clip.fps.denominator == 0:
        return clip
    if height is None:
        height = max(64, (clip.height // 4) & ~3)
    if beam_size is None:
        beam_size = min(8.0, max(1.5, height / 34))
    if mode == 3 and audio.num_channels < 2:
        mode = 2
    beam_gain *= max(1.0, clip.fps.numerator / clip.fps.denominator / 50.0)
    expr = get_expr(tools)
    missing = clip.num_frames * audio.sample_rate * clip.fps.denominator // clip.fps.numerator + 1 - audio.num_samples
    if missing > 0:
        audio = audio + core.std.BlankAudio(clip=audio, length=missing)
    trace = core.fw.scope(clip=core.std.BlankClip(clip=clip, height=height, format=vs.GRAYS), audio=audio, mode=mode,
                          beam_size=beam_size, beam_gain=beam_gain, y_gain=y_gain, y_ofs=y_ofs)[:clip.num_frames]
    fmt = clip.format
    limited = is_limited_range(clip, fmt.color_family == vs.YUV)
    gray = core.query_video_format(vs.GRAY, fmt.sample_type, fmt.bits_per_sample, 0, 0).id
    exprs = _plane_exprs(fmt, limited, color)
    # the chroma planes come from the trace at their own size, so no plane is converted twice
    planes = [expr([trace], [exprs[0]], format=gray)]
    if fmt.num_planes == 3:
        source = trace
        if fmt.subsampling_w or fmt.subsampling_h:
            source = core.resize.Bilinear(clip=trace, width=trace.width >> fmt.subsampling_w, height=trace.height >> fmt.subsampling_h)
        planes += [expr([source], [exprs[i]], format=gray) for i in (1, 2)]
    picture = core.std.ShufflePlanes(clips=planes, planes=[0] * fmt.num_planes, colorfamily=fmt.color_family)
    if picture.format.id != fmt.id:
        picture = core.resize.Point(clip=picture, format=fmt.id)
    picture = core.std.CopyFrameProps(clip=picture, prop_src=clip)
    if placement == 'above':
        return core.std.StackVertical(clips=[picture, clip])
    if placement == 'below':
        return core.std.StackVertical(clips=[clip, picture])
    # only the strip is merged, the rest of the picture is stacked back unchanged: padding picture and mask to the
    # full frame and merging all of it costs four times the strip on a 4K picture; the mask is the curved brightness
    mask = expr([trace], [f'{_CURVE} {1.0 if fmt.sample_type == vs.FLOAT else float((1 << fmt.bits_per_sample) - 1)} *'], format=gray)
    rest = clip.height - height
    if placement == 'overlay_top':
        strip = core.std.MaskedMerge(clipa=core.std.Crop(clip=clip, bottom=rest), clipb=picture, mask=mask, first_plane=True)
        return core.std.StackVertical(clips=[strip, core.std.Crop(clip=clip, top=height)])
    strip = core.std.MaskedMerge(clipa=core.std.Crop(clip=clip, top=rest), clipb=picture, mask=mask, first_plane=True)
    return core.std.StackVertical(clips=[core.std.Crop(clip=clip, bottom=height), strip])
