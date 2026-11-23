from __future__ import annotations

import functools
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import vapoursynth as vs
from vapoursynth import core

from helpers import Deen, get_expr, pick_tool, tool_function
from misc import get_mv

# HybridFuPP 0.992b by FuPP, ported from AviSynth
#
# dependencies:
# deen (https://github.com/Selur/Deen), neo_smo or Expr, picked by tools['deen'] (helpers.Deen): Deen("a2d")
#   and Deen("a3d") of the original
# deblock (https://github.com/HomeOfVapourSynthEvolution/VapourSynth-Deblock), only with Deblock=True
# mvutensils or MVTools, picked by tools['mv'], only with motion processing or D_Mode=2
# neo_fft, neo_fft3d or fft3dfilter, picked by tools['fft3d'], only with D_Mode=3
# noise or grain, picked by tools['grain'], only for grain (E_Str_N, Enh_Str with Skin/DS_Enh)
# subtext or core.text, picked by tools['text'], only with show
# vszip or std BoxBlur, picked by tools['boxblur']
#
# Differences to the AviSynth original:
# - MaskTools logic on masks uses min/max instead of bitwise operations; all masks are binary or smoothed binary.
# - UnFilter is the 3-tap kernel [-k, 1+2k, -k] with k = strength/400, strength limited to -100..100.
# - The filter chains N1, N2, M, LP, DR, E1, E2 are callables clip -> clip instead of AviSynth strings; False disables one.
# - N1 runs at the source size with its own motion vectors (the original fed it the vectors of the resized clip).
# - Skin processing with edge processing uses skin areas without edges (the original combined edges without skin, which
#   the later edge merge overwrote).
# - Any 8-16 bit integer YUV clip is accepted; thresholds stay on the 8-bit scale.

Chain = Optional[Union[Callable[[vs.VideoNode], vs.VideoNode], bool]]

_PRESETS = ('medium', 'high', 'very high', 'low', 'very low', 'anime1', 'anime2', 'hybridq')


def _row(*values: Union[int, float, bool]) -> Dict[str, Union[int, float, bool]]:
    return dict(zip(_PRESETS, values))


#                            MED    HIGH   VHIGH  LOW    VLOW   AN1    AN2    HYBQ
_DEFAULTS = {
    'Resizer':    _row(3,     6,     8,     2,     0,     2,     0,     6),
    'ARAdjust':   _row(False, False, False, False, False, False, False, True),
    'Fast':       _row(True,  True,  True,  True,  True,  True,  True,  True),
    'D_Mode':     _row(2,     2,     2,     2,     2,     1,     1,     2),
    'S_Radius':   _row(1,     1,     1,     1,     1,     1,     2,     1),
    'S_Dist':     _row(0.1,   0.1,   0.1,   0.1,   0.1,   0.1,   0.1,   0.1),
    'T_Str':      _row(3,     3,     2,     5,     5,     6,     15,    5),
    'C_Str':      _row(8,     8,     8,     8,     8,     9,     9,     8),
    'MP_Mode':    _row(1,     1,     1,     1,     1,     0,     0,     1),
    'ME_Resizer': _row(0,     0,     0,     0,     0,     0,     0,     0),
    'M_Thr':      _row(5,     5,     5,     5,     5,     5,     5,     5),
    'M_SCD':      _row(260,   260,   260,   260,   260,   260,   260,   260),
    'M_Str':      _row(120,   120,   80,    120,   120,   0,     0,     120),
    'M_TRadius':  _row(1,     1,     1,     1,     1,     0,     0,     1),
    'EM_Mode':    _row(1,     1,     1,     1,     1,     2,     2,     1),
    'E_Thr':      _row(135,   135,   135,   135,   135,   135,   135,   135),
    'E_Auto_Str': _row(False, False, False, False, False, False, False, True),
    'E_Str_X':    _row(0,     0,     0,     0,     0,     46,    46,    30),
    'E_Str_Y':    _row(0,     0,     0,     0,     0,     46,    46,    30),
    'E_Str_B':    _row(0,     0,     0,     0,     0,     -2,    -2,    0),
    'E_Str_N':    _row(0,     0,     0,     0,     0,     0,     0,     7),
    'Dering':     _row(False, False, False, False, False, True,  False, False),
    'DR_Radius':  _row(1,     1,     1,     1,     1,     1,     1,     1),
    'DR_Str':     _row(30,    30,    30,    30,    30,    60,    60,    30),
    'D_Thr':      _row(28,    28,    28,    32,    32,    30,    30,    24),
    'B_Thr':      _row(0,     0,     0,     0,     0,     0,     0,     180),
    'LP_Str':     _row(30,    30,    30,    30,    30,    40,    40,    30),
    'DS_Enh':     _row(False, False, False, False, False, False, False, True),
    'DS_Enh_Thr': _row(48,    48,    48,    48,    48,    48,    48,    48),
    'Deblock':    _row(False, False, False, False, False, False, False, False),
    'DB_M_Only':  _row(False, False, False, False, False, False, False, False),
    'DB_D_Only':  _row(False, False, False, False, False, False, False, False),
    'DB_Q':       _row(25,    25,    25,    25,    25,    25,    25,    25),
    'DB_Off_a':   _row(0,     0,     0,     0,     0,     0,     0,     0),
    'DB_Off_b':   _row(0,     0,     0,     0,     0,     0,     0,     0),
    'Skin':       _row(False, False, False, False, False, False, False, True),
    'Enh_Str':    _row(3,     3,     3,     3,     3,     3,     3,     3),
}
# S_Str depends on the denoiser.
_S_STR = {
    1: _row(10, 5, 7, 12, 12, 7, 20, 6),
    2: _row(0, 0, 0, 0, 0, 0, 0, 0),
    3: _row(2, 2, 1.7, 2, 2, 3, 3, 2),
}
_RESIZER_NAMES = ('Bilinear', 'Bic(0.3,0.3)', 'Bic(0.2,0.4)', 'Bic(0,0.5)', 'Bic(0,0.6)', 'Lanczos 3', 'Spline 16',
                  'Spline 36', 'Spline 64')
# Show: mask letter -> (label, RGB colour, opacity)
_SHOW_MASKS = {
    'E': ('Edges', 0xFF0808, 0.6),
    'M': ('Motion', 0xFFFF0F, 0.3),
    'R': ('DeRinging', 0xFFBF0A, 0.7),
    'D': ('Dark areas', 0x00FFF0, 0.4),
    'X': ('Dark Scenes Enhancement', 0x005FFF, 0.2),
    'B': ('Bright areas', 0x00FF0A, 1.0),
    'S': ('Skin', 0xFF00C0, 0.2),
    'A': ('Auto Resizer Adjustment', 0xFFA0F0, 0.4),
}


def resize(clip: vs.VideoNode, width: int, height: int, resizer: int) -> vs.VideoNode:
    '''HybridFuPP resizer 0-8: bilinear, four bicubic variants, lanczos 3, spline 16/36/64.'''
    if resizer == 0:
        return core.resize.Bilinear(clip, width, height)
    bicubic = {1: (1 / 3, 1 / 3), 2: (0.2, 0.4), 3: (0.0, 0.5), 4: (0.0, 0.6)}
    if resizer in bicubic:
        b, c = bicubic[resizer]
        return core.resize.Bicubic(clip, width, height, filter_param_a=b, filter_param_b=c)
    if resizer == 5:
        return core.resize.Lanczos(clip, width, height, filter_param_a=3)
    if resizer in (6, 7, 8):
        return getattr(core.resize, ('Spline16', 'Spline36', 'Spline64')[resizer - 6])(clip, width, height)
    raise vs.Error(f'HybridFuPP: resizer {resizer} does not exist, use 0-8')


def _convolve3(clip: vs.VideoNode, kernel: Sequence[float], mode: str) -> vs.VideoNode:
    '''3-tap convolution with a float kernel summing to 1; integer clips need integer coefficients.'''
    if clip.format.sample_type == vs.FLOAT:
        return clip.std.Convolution(list(kernel), mode=mode)
    taps = [round(k * 256) for k in kernel]
    return clip.std.Convolution(taps, divisor=sum(taps), mode=mode)


def sharpen(clip: vs.VideoNode, amount: float) -> vs.VideoNode:
    '''AviSynth Sharpen(amount): kernel [(1-2^a)/2, 2^a, (1-2^a)/2] in both directions; negative values blur.'''
    if amount == 0:
        return clip
    center = 2.0 ** amount
    side = (1.0 - center) / 2.0
    return _convolve3(_convolve3(clip, [side, center, side], 'h'), [side, center, side], 'v')


def unfilter(clip: vs.VideoNode, hsharp: float, vsharp: float) -> vs.VideoNode:
    '''UnFilter(hsharp, vsharp): 3-tap sharpen (positive) or soften (negative), strength -100..100 per direction.'''
    for strength, mode in ((hsharp, 'h'), (vsharp, 'v')):
        k = max(-100.0, min(100.0, strength)) / 400.0
        if k:
            clip = _convolve3(clip, [-k, 1.0 + 2.0 * k, -k], mode)
    return clip


def autostr(rw: float, rh: float, resizer: int, axis: str) -> int:
    '''E_Auto_Str: edge sharpening strength from the downscale ratios (0 = upscale or same size) and the resizer.'''
    ref = rw if axis == 'X' else rh
    if ref == 0:
        return 0
    big, small = max(rw, rh), min(rw, rh)
    base = -180 * ref + 180 if big == ref else (-180 * big + 180) * (small / big)
    return min(60, max(0, int((1 - (resizer / 10 - 0.2)) * base)))


class _Masks:
    '''MaskTools-style operations on GRAY clips of one bit depth; thresholds on the 8-bit scale.'''

    def __init__(self, bits: int, tools: Optional[Mapping[str, str]]) -> None:
        self.shift = bits - 8
        self.peak = (1 << bits) - 1
        self.half = 1 << (bits - 1)
        self.tools = tools
        self.expr = get_expr(tools)
        self.gray = core.query_video_format(vs.GRAY, vs.INTEGER, bits, 0, 0).id

    def scale(self, value: float) -> float:
        return value * (1 << self.shift)

    def binarize(self, clip: vs.VideoNode, threshold: float, upper: bool = False) -> vs.VideoNode:
        '''mt_binarize: x > threshold -> peak (upper: -> 0); threshold already in the clip's scale.'''
        hit, miss = (0, self.peak) if upper else (self.peak, 0)
        return self.expr(clip, f'x {threshold} > {hit} {miss} ?')

    def logic(self, a: vs.VideoNode, b: vs.VideoNode, mode: str) -> vs.VideoNode:
        expr = {'and': 'x y min', 'or': 'x y max', 'xor': 'x y - abs', 'andn': f'x {self.peak} y - min'}[mode]
        return self.expr([a, b], expr)

    def black(self, like: vs.VideoNode) -> vs.VideoNode:
        return core.std.BlankClip(like, format=self.gray, color=0, keep=True)

    def white(self, like: vs.VideoNode) -> vs.VideoNode:
        return core.std.BlankClip(like, format=self.gray, color=self.peak, keep=True)

    def select(self, stats_src: vs.VideoNode, when: Callable[[float], bool], a: vs.VideoNode, b: vs.VideoNode,
               plane: int = 0) -> vs.VideoNode:
        '''Per frame `a` when `when(average)` holds for the normalized average of `plane` of `stats_src`, else `b`.'''
        stats = core.std.PlaneStats(stats_src, plane=plane)
        return core.std.FrameEval(a, lambda n, f: a if when(f.props['PlaneStatsAverage']) else b, prop_src=stats)


def _luma(clip: vs.VideoNode) -> vs.VideoNode:
    return core.std.ShufflePlanes(clip, 0, vs.GRAY)


def _repeat(clip: vs.VideoNode, func: Callable[[vs.VideoNode], vs.VideoNode], count: int) -> vs.VideoNode:
    for _ in range(count):
        clip = func(clip)
    return clip


class _Motion:
    '''Super clip and vectors of one clip for the temporal radius 1..3.'''

    def __init__(self, clip: vs.VideoNode, radius: int, blksize: int, mv: object) -> None:
        overlap = max(blksize // 2, 2)
        self.mv = mv
        geometry = dict(blksize=blksize, overlap=overlap)
        self.super = mv.Super(clip, hpad=16, vpad=16, **geometry)
        self.bv = [mv.Analyse(self.super, isb=True, delta=d, lambda_=1000, **geometry) for d in range(1, radius + 1)]
        self.fv = [mv.Analyse(self.super, isb=False, delta=d, lambda_=1000, **geometry) for d in range(1, radius + 1)]

    def compensated(self, clip: vs.VideoNode, radius: int, func: Callable[[vs.VideoNode], vs.VideoNode],
                    scd: int) -> vs.VideoNode:
        '''func on the motion compensated neighbours: frames n-radius..n+radius interleaved, the centre one kept.'''
        bc = [self.mv.Compensate(clip, self.super, v, thscd1=scd) for v in self.bv[:radius]]
        fc = [self.mv.Compensate(clip, self.super, v, thscd1=scd) for v in self.fv[:radius]]
        frames = bc[::-1] + [clip] + fc
        return core.std.SelectEvery(func(core.std.Interleave(frames)), 2 * radius + 1, radius)

    def degrain(self, clip: vs.VideoNode, radius: int, scd: int) -> vs.VideoNode:
        vectors = {}
        for d in range(radius):
            suffix = '' if d == 0 else str(d + 1)
            vectors['mvbw' + suffix] = self.bv[d]
            vectors['mvfw' + suffix] = self.fv[d]
        return self.mv.Degrain(clip, self.super, thscd1=scd, thscd2=135, plane=4, centre_from_clip=True, **vectors)

    def mask(self, clip: vs.VideoNode, masks: _Masks, m_thr: int, scd: int) -> vs.VideoNode:
        '''Moving areas: both vector length masks above 30, shrunk once.'''
        parts = []
        # MVTools' Mask takes only 8-bit clips and uses the clip for its size and format only.
        shape = core.resize.Point(clip, format=vs.GRAY8) if clip.format.bits_per_sample > 8 else clip
        for vectors in (self.fv[0], self.bv[0]):
            m = _luma(self.mv.Mask(shape, vectors, ml=m_thr, gamma=50, kind=0, thscd1=scd, thscd2=135))
            if (m.width, m.height) != (clip.width, clip.height):
                m = core.resize.Point(m, clip.width, clip.height)
            parts.append(m)
        mask_bits = parts[0].format.bits_per_sample
        thr = 30 * (1 << (mask_bits - 8)) if parts[0].format.sample_type == vs.INTEGER else 30 / 255
        return masks.expr(parts, f'x y min {thr} > {masks.peak} 0 ?', format=masks.gray).std.Minimum()


def _edge_mask(clip: vs.VideoNode, masks: _Masks, mode: int, e_thr: int) -> vs.VideoNode:
    '''Dark edges: pixels darker than their 5x5 blur by more than e_thr-128; mode 2 adds weaker edges in dark areas.'''
    y = _luma(clip)
    box = tool_function(masks.tools, 'boxblur', 'BoxBlur')
    diff = core.std.MakeDiff(box(box(y, hradius=1, vradius=1), hradius=1, vradius=1), y)
    strong = _repeat(masks.binarize(diff, masks.half + masks.scale(e_thr - 128)), core.std.Inflate, 2)
    if mode != 2:
        return strong
    weak = _repeat(masks.binarize(diff, masks.half + masks.scale(e_thr - 132)), core.std.Inflate, 2)
    dark = masks.binarize(y, masks.scale(62), upper=True)
    return masks.logic(strong, masks.logic(weak, dark, 'and'), 'or')


def _dark_mask(clip: vs.VideoNode, masks: _Masks, d_thr: int) -> vs.VideoNode:
    return masks.binarize(_luma(clip), masks.scale(d_thr), upper=True).std.Deflate()


def _bright_mask(clip: vs.VideoNode, masks: _Masks, b_thr: int) -> vs.VideoNode:
    return masks.binarize(_luma(clip), masks.scale(b_thr)).std.Minimum()


def _dark_scene_mask(clip: vs.VideoNode, masks: _Masks, ds_thr: int, d_thr: int) -> vs.VideoNode:
    '''In frames with average luma <= 46: the band between the dark masks of ds_thr and d_thr+3.'''
    band = masks.logic(_dark_mask(clip, masks, ds_thr), _dark_mask(clip, masks, d_thr + 3), 'xor')
    y = _luma(clip)
    scene = masks.select(y, lambda avg: avg * 255 <= 46, masks.white(y), masks.black(y))
    return masks.logic(scene, band, 'and')


def _resizer_mask(clip: vs.VideoNode, masks: _Masks) -> vs.VideoNode:
    '''ARAdjust: flat areas (no pixel above its soft blur in the 3x3 neighbourhood), limit 1 in brighter frames.'''
    y = _luma(clip)
    diff = core.std.MakeDiff(y, unfilter(y, -100, -100)).std.Maximum()
    loose = masks.binarize(diff, masks.half + masks.scale(1), upper=True)
    tight = masks.binarize(diff, masks.half, upper=True)
    return masks.select(y, lambda avg: avg * 255 > 56, loose, tight)


def _skin_mask(clip: vs.VideoNode, masks: _Masks) -> vs.VideoNode:
    '''Skin: luma 52..190 and chroma in a band around the frame's average; empty when more than 68/255 is skin.'''
    y = _luma(clip)
    u = core.std.ShufflePlanes(clip, 1, vs.GRAY)
    v = core.std.ShufflePlanes(clip, 2, vs.GRAY)
    luma_ok = masks.expr(y, f'x {masks.scale(52)} <= x {masks.scale(190)} >= or 0 {masks.peak} ?')

    @functools.lru_cache(maxsize=64)
    def skin(avu: float, avv: float) -> vs.VideoNode:
        band = 'x {} <= x {} >= or 0 ' + str(masks.peak) + ' ?'
        um = masks.expr(u, band.format(avu * 0.84, avu * 0.99))
        vm = masks.expr(v, band.format(avv * 1.05, avv * 1.18))
        uv = core.resize.Point(masks.logic(um, vm, 'and'), y.width, y.height)
        return masks.logic(uv, luma_ok, 'and')

    stats = core.std.PlaneStats(clip, plane=1, prop='U')
    stats = core.std.PlaneStats(stats, plane=2, prop='V')
    detected = core.std.FrameEval(y, lambda n, f: skin(round(f.props['UAverage'] * masks.peak, 2),
                                                       round(f.props['VAverage'] * masks.peak, 2)), prop_src=stats)
    return masks.select(detected, lambda avg: avg * 255 <= 68, detected, masks.black(y))


def HybridFuPP(input: vs.VideoNode, width: Optional[int] = None, height: Optional[int] = None, Preset: str = 'high',
               Resizer: Optional[int] = None, Fast: Optional[bool] = None, ARAdjust: Optional[bool] = None,
               Dering: Optional[bool] = None, Deblock: Optional[bool] = None, DB_M_Only: Optional[bool] = None,
               DB_D_Only: Optional[bool] = None, Skin: Optional[bool] = None, Enh_Str: Optional[int] = None,
               D_Mode: Optional[int] = None, MP_Mode: Optional[int] = None, S_Str: Optional[float] = None,
               T_Str: Optional[int] = None, S_Radius: Optional[int] = None, S_Dist: Optional[float] = None,
               C_Str: Optional[int] = None, M_Thr: Optional[int] = None, M_SCD: Optional[int] = None,
               M_Str: Optional[int] = None, M_TRadius: Optional[int] = None, Bl_Size: Optional[int] = None,
               ME_Resizer: Optional[int] = None, E_Auto_Str: Optional[bool] = None, EM_Mode: Optional[int] = None,
               E_Thr: Optional[int] = None, E_Str_X: Optional[int] = None, E_Str_Y: Optional[int] = None,
               E_Str_B: Optional[int] = None, E_Str_N: Optional[int] = None, DR_Str: Optional[int] = None,
               DR_Radius: Optional[int] = None, D_Thr: Optional[int] = None, B_Thr: Optional[int] = None,
               LP_Str: Optional[int] = None, DS_Enh: Optional[bool] = None, DS_Enh_Thr: Optional[int] = None,
               DB_Q: Optional[int] = None, DB_Off_a: Optional[int] = None, DB_Off_b: Optional[int] = None,
               show: str = '', N1: Chain = None, N2: Chain = None, M: Chain = None, E1: Chain = None,
               E2: Chain = None, LP: Chain = None, DR: Chain = None,
               tools: Optional[Mapping[str, str]] = None) -> vs.VideoNode:
    """HybridFuPP: resize with area-dependent processing (denoise, motion, edges, dark/bright areas, skin, deblock).

    Every parameter left at None takes the value of the preset. Thresholds and strengths are on the 8-bit scale.

    width, height: output size, mod 8; default the source size
    Preset:        'very low', 'low', 'medium', 'high' (default), 'very high', 'HybridQ', 'anime1', 'anime2'
    Resizer:       0 bilinear, 1-4 bicubic (soft to sharp), 5 lanczos 3, 6/7/8 spline 16/36/64
    Fast:          masks from the denoised clip (True) or the plainly resized source (False)
    ARAdjust:      use Resizer2 (a softer resizer) in flat areas
    D_Mode:        denoiser 1 Deen a3d (S_Dist: its min), 2 MDegrain (M_TRadius 1-3), 3 FFT3D (sigma S_Str)
    S_Str, S_Radius, T_Str, C_Str: spatial/temporal/chroma strength and radius (D_Mode 1; S_Str also D_Mode 3)
    MP_Mode:       0 no motion processing, 1 motion compensated denoising, unprocessed motion areas,
                   2 spatial denoising (M_Str) in motion areas
    M_TRadius:     motion compensation radius 0-3; Bl_Size 4/8/16; M_Thr motion threshold; M_SCD scene change threshold
    ME_Resizer:    resizer of the clip used for motion search (and MDegrain's pixels)
    EM_Mode:       edge mask 1 film, 2 anime; E_Thr edge threshold (128 neutral)
    E_Str_X/Y:     edge sharpening -50..50 (E_Auto_Str: from the scale ratio); E_Str_B edge brightness; E_Str_N edge grain
    Dering:        Deen a2d (DR_Str) around edges grown by DR_Radius+1
    D_Thr, B_Thr:  dark/bright area thresholds (0 = off), filtered with LP_Str
    DS_Enh:        dark scene detail enhancement (DS_Enh_Thr); Skin: skin enhancement; Enh_Str 0-5 for both
    Deblock:       Deblock(DB_Q, DB_Off_a, DB_Off_b) on the source, DB_M_Only/DB_D_Only restrict it to motion/dark areas
    show:          letters E M D B R X S A overlay masks, P parameters, F filter chains, - | / compare with the source
    N1, N2, M, LP, DR, E1, E2: filter chains clip -> clip replacing the defaults (N1 source denoiser before resizing,
                   N2 denoiser, M motion, LP dark/bright areas, DR deringing, E1 edges before, E2 after resizing);
                   False disables a chain
    tools:         {'deen', 'mv', 'expr', 'fft3d', 'grain', 'boxblur', 'text'} pick the implementations
    """
    if not isinstance(input, vs.VideoNode):
        raise vs.Error('HybridFuPP: this is not a clip')
    fmt = input.format
    if fmt.color_family != vs.YUV or fmt.sample_type != vs.INTEGER or fmt.bits_per_sample > 16:
        raise vs.Error('HybridFuPP: only 8-16 bit integer YUV is supported')

    preset = Preset.lower()
    if preset not in _PRESETS:
        raise vs.Error(f'HybridFuPP: preset "{Preset}" does not exist, use one of {", ".join(_PRESETS)}')

    given = dict(Resizer=Resizer, Fast=Fast, ARAdjust=ARAdjust, Dering=Dering, Deblock=Deblock, DB_M_Only=DB_M_Only,
                 DB_D_Only=DB_D_Only, Skin=Skin, Enh_Str=Enh_Str, D_Mode=D_Mode, MP_Mode=MP_Mode, T_Str=T_Str,
                 S_Radius=S_Radius, S_Dist=S_Dist, C_Str=C_Str, M_Thr=M_Thr, M_SCD=M_SCD, M_Str=M_Str,
                 M_TRadius=M_TRadius, ME_Resizer=ME_Resizer, E_Auto_Str=E_Auto_Str, EM_Mode=EM_Mode, E_Thr=E_Thr,
                 E_Str_X=E_Str_X, E_Str_Y=E_Str_Y, E_Str_B=E_Str_B, E_Str_N=E_Str_N, DR_Str=DR_Str,
                 DR_Radius=DR_Radius, D_Thr=D_Thr, B_Thr=B_Thr, LP_Str=LP_Str, DS_Enh=DS_Enh, DS_Enh_Thr=DS_Enh_Thr,
                 DB_Q=DB_Q, DB_Off_a=DB_Off_a, DB_Off_b=DB_Off_b)
    p = {name: _DEFAULTS[name][preset] if value is None else value for name, value in given.items()}
    defaults = {name: _DEFAULTS[name][preset] for name in given}
    p['S_Str'] = _S_STR.get(p['D_Mode'], _row(*[0] * 8))[preset] if S_Str is None else S_Str
    defaults['S_Str'] = _S_STR[defaults['D_Mode']][preset]
    p['Bl_Size'] = (8 if p['D_Mode'] == 2 else 16) if Bl_Size is None else Bl_Size
    defaults['Bl_Size'] = 8 if defaults['D_Mode'] == 2 else 16

    swidth, sheight = input.width, input.height
    width = swidth if width is None else width
    height = sheight if height is None else height
    rwidth = 0 if width / swidth > 1 else width / swidth
    rheight = 0 if height / sheight > 1 else height / sheight
    resizer2 = 0 if p['Resizer'] < 2 else (p['Resizer'] - 2) // 2
    modified = p != defaults
    if p['E_Auto_Str']:
        p['E_Str_X'] = autostr(rwidth, rheight, p['Resizer'], 'X')
        p['E_Str_Y'] = autostr(rwidth, rheight, p['Resizer'], 'Y')
    if p['MP_Mode'] == 2:
        p['M_TRadius'] = 1

    show = show.upper()
    m_process = p['M_TRadius'] != 0 and p['MP_Mode'] != 0

    if swidth % 2 or sheight % 2:
        raise vs.Error('HybridFuPP: source width and height must be mod 2, crop if necessary')
    if width % 8 or height % 8:
        raise vs.Error('HybridFuPP: output width and height must be mod 8')
    if p['Deblock'] and (swidth % 8 or sheight % 8):
        raise vs.Error('HybridFuPP: source width and height must be mod 8 for deblocking')
    if p['Bl_Size'] not in (4, 8, 16):
        raise vs.Error('HybridFuPP: Bl_Size must be 4, 8 or 16')
    if p['D_Mode'] not in (1, 2, 3):
        raise vs.Error('HybridFuPP: D_Mode must be 1, 2 or 3')
    if not 0 <= p['M_TRadius'] <= 3:
        raise vs.Error('HybridFuPP: M_TRadius must be 0-3')
    if p['D_Mode'] == 2 and p['M_TRadius'] == 0:
        raise vs.Error('HybridFuPP: D_Mode=2 needs M_TRadius 1-3')
    if 'M' in show and not m_process:
        raise vs.Error('HybridFuPP: the motion mask needs M_TRadius > 0 and MP_Mode > 0')
    if not 0 <= p['Enh_Str'] <= 5:
        raise vs.Error('HybridFuPP: Enh_Str must be 0-5')

    masks = _Masks(fmt.bits_per_sample, tools)
    mv = get_mv(tools)

    def grain(clip: vs.VideoNode, var: float) -> vs.VideoNode:
        return tool_function(tools, 'grain', 'Add')(clip, var=var, uvar=0, hcorr=0, vcorr=0) if var else clip

    def deen2d(radius: int, threshold: int) -> Callable[[vs.VideoNode], vs.VideoNode]:
        def run(clip: vs.VideoNode) -> vs.VideoNode:
            return Deen(clip, 'a2d', radius, threshold, 0, 0, 0, 0.0, tools=tools)
        run.__name__ = f'Deen("a2d", {radius}, {threshold}, 0, 0, 0, 0)'
        return run

    def deen3d(clip: vs.VideoNode) -> vs.VideoNode:
        return Deen(clip, 'a3d', max(1, min(4, p['S_Radius'])), p['S_Str'], 3 * p['C_Str'], p['T_Str'],
                    p['C_Str'], p['S_Dist'], 9.0, tools=tools)

    def fft3d(clip: vs.VideoNode) -> vs.VideoNode:
        return tool_function(tools, 'fft3d', 'FFT3D')(clip, sigma=p['S_Str'], bt=3, planes=[0, 1, 2])

    def chain(value: Chain, default: Optional[Callable[[vs.VideoNode], vs.VideoNode]]) \
            -> Optional[Callable[[vs.VideoNode], vs.VideoNode]]:
        if value is False:
            return None
        return default if value is None or value is True else value

    def default_e1(clip: vs.VideoNode) -> vs.VideoNode:
        return grain(unfilter(clip, p['E_Str_X'], p['E_Str_Y']), p['E_Str_N'])

    def default_e2(clip: vs.VideoNode) -> vs.VideoNode:
        return masks.expr(clip, [f'x {masks.scale(p["E_Str_B"])} +', '', '']) if p['E_Str_B'] else clip

    default_e1.__name__ = f'UnFilter({p["E_Str_X"]}, {p["E_Str_Y"]}).AddGrain({p["E_Str_N"]})'
    default_e2.__name__ = f'ColorYUV(off_y={p["E_Str_B"]})'
    deen3d.__name__ = (f'Deen("a3d", {max(1, min(4, p["S_Radius"]))}, {p["S_Str"]}, {3 * p["C_Str"]}, {p["T_Str"]}, '
                       f'{p["C_Str"]}, {p["S_Dist"]}, 9)')
    fft3d.__name__ = f'FFT3D(sigma={p["S_Str"]}, bt=3)'
    mdegrain_name = f'MDegrain{p["M_TRadius"]}'

    n1 = chain(N1, None)
    n2 = chain(N2, {1: deen3d, 3: fft3d}.get(p['D_Mode']))
    n2_mdegrain = N2 is None and p['D_Mode'] == 2
    e1 = chain(E1, default_e1)
    e2 = chain(E2, default_e2)
    m_chain = chain(M, deen2d(1, p['M_Str']))
    lp = chain(LP, deen2d(2, p['LP_Str']))
    dr = chain(DR, deen2d(2, p['DR_Str']))

    d_process2 = p['DS_Enh'] and p['DS_Enh_Thr'] != 0 and p['Enh_Str'] != 0
    d_process = lp is not None and p['LP_Str'] != 0 and p['D_Thr'] != 0
    b_process = lp is not None and p['LP_Str'] != 0 and p['B_Thr'] != 0
    e_process = (e1 is not None or e2 is not None) and \
        (p['E_Str_X'] != 0 or p['E_Str_Y'] != 0 or p['E_Str_B'] != 0 or p['E_Auto_Str'])
    db_m_only = p['Deblock'] and p['DB_M_Only']
    db_d_only = p['Deblock'] and p['DB_D_Only']

    # Deblocking of the source, optionally only in moving or dark areas.
    deblocked = core.deblock.Deblock(input, quant=p['DB_Q'], aoffset=p['DB_Off_a'], boffset=p['DB_Off_b']) \
        if p['Deblock'] else input
    basic = resize(deblocked, width, height, p['ME_Resizer'])
    motion = _Motion(basic, max(1, p['M_TRadius']), p['Bl_Size'], mv) \
        if m_process or db_m_only or n2_mdegrain else None
    dark = _dark_mask(basic, masks, p['D_Thr']) if 'D' in show or d_process or db_d_only else None
    dark_scene = _dark_scene_mask(basic, masks, p['DS_Enh_Thr'], p['D_Thr']) if d_process2 or db_d_only else None
    src = deblocked
    if db_m_only or db_d_only:
        area = masks.black(basic)
        if db_m_only:
            area = masks.logic(area, motion.mask(basic, masks, p['M_Thr'], p['M_SCD']), 'or')
        if db_d_only:
            area = masks.logic(area, masks.logic(dark, dark_scene, 'or'), 'or')
        src = core.std.MaskedMerge(input, deblocked, core.resize.Bilinear(area, swidth, sheight), first_plane=True)

    resizer_mask = core.resize.Point(_resizer_mask(src, masks), width, height) \
        if 'A' in show or p['ARAdjust'] else None

    # Denoising: N1 at the source size, resizing (two resizers with ARAdjust), N2 at the output size.
    def denoise(clip: vs.VideoNode, func: Callable[[vs.VideoNode], vs.VideoNode], moved: Optional[_Motion]) \
            -> vs.VideoNode:
        if moved is not None and p['MP_Mode'] != 2:
            return moved.compensated(clip, p['M_TRadius'], func, p['M_SCD'])
        return func(clip)

    before = src
    if n1 is not None:
        source_motion = _Motion(src, p['M_TRadius'], p['Bl_Size'], mv) if m_process else None
        before = denoise(src, n1, source_motion)
    resized = resize(before, width, height, p['Resizer'])
    if p['ARAdjust']:
        resized = core.std.MaskedMerge(resized, resize(before, width, height, resizer2), resizer_mask, first_plane=True)
    if n2_mdegrain:
        denoised = motion.degrain(resized, p['M_TRadius'], p['M_SCD'])
    elif n2 is not None:
        denoised = denoise(resized, n2, motion if m_process else None)
    else:
        denoised = resized

    clip = denoised if p['Fast'] else resize(src, width, height, p['Resizer'])

    bright = _bright_mask(clip, masks, p['B_Thr']) if show or b_process else None
    edge = _edge_mask(clip, masks, p['EM_Mode'], p['E_Thr']) if any(c in show for c in 'ER') or e_process or p['Dering'] \
        else None
    motion_mask = motion.mask(clip, masks, p['M_Thr'], p['M_SCD']) if 'M' in show or m_process else None
    ringing = _repeat(edge, core.std.Maximum, p['DR_Radius'] + 1) if 'R' in show or p['Dering'] else None
    skin = None
    if 'S' in show or p['Skin']:
        skin = _skin_mask(clip, masks)
        if e_process:
            skin = masks.logic(skin, edge, 'andn')

    final = denoised
    if b_process or d_process:
        area = masks.logic(dark, bright, 'or') if b_process and d_process else (dark if d_process else bright)
        final = core.std.MaskedMerge(final, lp(denoised), area, first_plane=True)
    enhanced = resize(grain(sharpen(src, p['Enh_Str'] * 0.2), p['Enh_Str']), width, height, 6) \
        if p['Skin'] or d_process2 else None
    if d_process2:
        final = core.std.MaskedMerge(final, enhanced, dark_scene, first_plane=True)
    if p['Dering']:
        final = core.std.MaskedMerge(final, dr(clip), ringing, first_plane=True)
    if p['Skin']:
        final = core.std.MaskedMerge(final, enhanced, skin, first_plane=True)
    if e_process:
        edges = resize(e1(src) if e1 is not None else src, width, height, p['Resizer'])
        edges = e2(edges) if e2 is not None else edges
        final = core.std.MaskedMerge(final, edges, edge, first_plane=True)
    if m_process and p['MP_Mode'] == 2:
        blurred = m_chain(denoised) if m_chain is not None else denoised
        if db_m_only:
            blurred = core.deblock.Deblock(blurred, quant=p['DB_Q'], aoffset=p['DB_Off_a'], boffset=p['DB_Off_b'])
        final = core.std.MaskedMerge(final, blurred, motion_mask, first_plane=True)
    elif m_process and p['MP_Mode'] == 1:
        final = core.std.MaskedMerge(final, clip, motion_mask, first_plane=True)

    if not show:
        return final

    def name(func: Optional[Callable[[vs.VideoNode], vs.VideoNode]], fallback: str = '') -> str:
        return getattr(func, '__name__', repr(func)) if func is not None else fallback

    chains = [('N1', name(n1)), ('N2', mdegrain_name if n2_mdegrain else name(n2)), ('M', name(m_chain)),
              ('LP', name(lp)), ('E1', name(e1)), ('E2', name(e2)), ('DR', name(dr))]
    return _show(final, input, show, dict(E=edge, M=motion_mask, R=ringing, D=dark, X=dark_scene, B=bright, S=skin,
                                          A=resizer_mask), masks, p, Preset + (' (mod)' if modified else ''),
                 resizer2, chains)


def _text(clip: vs.VideoNode, text: str, alignment: int, tools: Optional[Mapping[str, str]]) -> vs.VideoNode:
    """Monospaced white text at numpad `alignment`; tools['text'] picks subtext ('sub') or core.text ('std')."""
    if pick_tool(tools, 'text', ('sub', 'std')) != 'sub':
        return core.text.Text(clip, text, alignment=alignment)
    lines = text.split('\n')
    # largest font that fits all lines (Courier New: about 0.6 em per character, 1.2 em per line)
    size = int(min(clip.height / (1.2 * len(lines) + 1), clip.width / (0.6 * max(len(line) for line in lines) + 2), 24))
    style = (f'Courier New,{max(size, 8)},&H00FFFFFF,&H00FFFFFF,&H00000000,&H00000000,1,0,0,0,100,100,0,0,1,1,0,'
             f'{alignment},8,8,8,1')
    return core.sub.Subtitle(clip, text.replace(' ', '\\h').replace('\n', '\\N'), style=style)


def _show(final: vs.VideoNode, source: vs.VideoNode, show: str, mask_clips: Dict[str, Optional[vs.VideoNode]],
          masks: _Masks, p: Dict[str, object], preset: str, resizer2: int,
          chains: Sequence[Tuple[str, str]]) -> vs.VideoNode:
    '''Mask overlays, parameter or filter chain text and source comparison.'''
    matrix = '709' if final.width > 1024 or final.height > 576 else '470bg'
    legend: List[str] = []
    for letter, (label, rgb, opacity) in _SHOW_MASKS.items():
        mask = mask_clips.get(letter)
        if letter not in show or mask is None:
            continue
        if letter == 'B':
            mask = masks.logic(mask, mask.std.Minimum(), 'xor')
        color = core.std.BlankClip(final, format=vs.RGB24, color=[rgb >> 16, (rgb >> 8) & 0xFF, rgb & 0xFF])
        color = core.resize.Bicubic(color, format=final.format.id, matrix_s=matrix)
        final = core.std.MaskedMerge(final, color, masks.expr(mask, f'x {opacity} *'), first_plane=True)
        legend.append(label)
    if legend:
        final = _text(final, '\n'.join(legend), 1, masks.tools)

    if 'P' in show:
        def values(keys: Sequence[str]) -> List[str]:
            return [f'{key:<10} = {p[key] if p["Deblock"] or not key.startswith("DB_") else "N/A"}' for key in keys]

        general = [f'Preset     = {preset}', f'Fast       = {p["Fast"]}', f'ARAdjust   = {p["ARAdjust"]}',
                   f'Resizer    = {p["Resizer"]}:{_RESIZER_NAMES[p["Resizer"]]}',
                   f'Resizer2   = {f"{resizer2}:{_RESIZER_NAMES[resizer2]}" if p["ARAdjust"] else "N/A"}',
                   f'Enh_Str    = {p["Enh_Str"]}']
        edges = values(('EM_Mode', 'E_Thr', 'E_Auto_Str', 'E_Str_X', 'E_Str_Y', 'E_Str_B', 'E_Str_N', 'Dering',
                        'DR_Str', 'DR_Radius'))
        deblock = values(('Deblock', 'DB_M_Only', 'DB_D_Only', 'DB_Q', 'DB_Off_a', 'DB_Off_b'))
        luma = values(('D_Thr', 'B_Thr', 'LP_Str', 'DS_Enh', 'DS_Enh_Thr', 'Skin'))
        denoising = values(('D_Mode', 'MP_Mode', 'ME_Resizer', 'Bl_Size', 'S_Str', 'T_Str', 'S_Radius', 'S_Dist',
                            'C_Str', 'M_TRadius', 'M_Thr', 'M_SCD', 'M_Str'))
        sizes = [f'Input  = {source.width} x {source.height}', f'Output = {final.width} x {final.height}']
        # three columns like the original overlay
        columns = [general + [''] + edges, deblock + [''] + luma, denoising + [''] + sizes]
        widths = [max(len(row) for row in column) + 2 for column in columns]
        rows = [''.join((column[i] if i < len(column) else '').ljust(w) for column, w in zip(columns, widths)).rstrip()
                for i in range(max(len(column) for column in columns))]
        final = _text(final, '\n'.join(rows), 7, masks.tools)
    elif 'F' in show:
        final = _text(final, '\n'.join(f'{key:<3} = {value}' for key, value in chains), 7, masks.tools)

    if not any(c in show for c in '-|/'):
        return final
    src = core.resize.Bicubic(source, format=final.format.id) if source.format.id != final.format.id else source
    gray = [masks.half, masks.half, masks.half]

    def pad(clip: vs.VideoNode, w: int, h: int) -> vs.VideoNode:
        dw, dh = w - clip.width, h - clip.height
        if dw > 0 or dh > 0:
            clip = core.std.AddBorders(clip, max(dw, 0) // 2, max(dw, 0) - max(dw, 0) // 2, max(dh, 0) // 2,
                                       max(dh, 0) - max(dh, 0) // 2, color=gray)
        return clip

    w, h = max(src.width, final.width), max(src.height, final.height)
    src = _text(pad(src, w, h), 'Source', 3, masks.tools)
    out = _text(pad(final, w, h), 'HybridFuPP', 3, masks.tools)
    if '-' in show:
        return core.std.StackVertical([src, out])
    if '|' in show:
        return core.std.StackHorizontal([src, out])
    return core.std.Interleave([src, out])
