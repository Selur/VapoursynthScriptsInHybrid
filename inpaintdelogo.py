from __future__ import annotations
import os
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import vapoursynth as vs
from vapoursynth import core

# Port of InpaintDelogo 3.7 by VoodooFX (public domain): https://github.com/Purfview/InpaintDelogo
# Needs AvsInpaint >= 1.4 (core.avsinpaint) and akarin; BestSource for mask files, neo_fft for negative oPP, addGrain for GrainPP.
# The helpers below rebuild AviSynth+ 3.7.5 (AVX2) and MaskTools2 bit-exactly where the original relies on them.

_NAME = 'InpaintDelogo'

# Parameters of later port stages; a value other than these defaults raises an error.
_UNSUPPORTED: Dict[str, Any] = {
    'Analyze': None, 'AnalyzeTh': None, 'NoBorderAt': 'none', 'ReAnalyze': 0, 'Automask': 0, 'aMix': 0,
    'Interp': None, 'IntSpd': None, 'InterpM': None, 'iTune': 0, 'iTinflate': 0, 'oTune': 0, 'oTinflate': 0,
    'dPP': None, 'dPPm': None, 'diPP': None, 'diPPm': None, 'EdgePP': 0, 'EdgePos': 0, 'EdgeWide': 1,
    'Optimize2': None, 'DynMask': 0, 'Show': 0, 'Extract': 0,
}


def _error(msg: str) -> vs.Error:
    return vs.Error(f'{_NAME}: {msg}')


def _require(namespace: str, purpose: str) -> None:
    if not hasattr(core, namespace):
        raise _error(f'the {namespace} plugin is required for {purpose}.')


def _expr(clips: Union[vs.VideoNode, Sequence[vs.VideoNode]], expr: Union[str, Sequence[str]], fmt: Optional[int] = None) -> vs.VideoNode:
    if fmt is None:
        return core.akarin.Expr(clips, expr)
    return core.akarin.Expr(clips, expr, format=fmt)


def _plane(c: vs.VideoNode, p: int) -> vs.VideoNode:
    return core.std.ShufflePlanes(c, p, vs.GRAY)


def _gray_to_yuv420(y: vs.VideoNode) -> vs.VideoNode:
    # ConvertToYV12 of a Y8 clip: neutral chroma
    blank = core.std.BlankClip(y, format=vs.YUV420P8, color=[0, 128, 128])
    return core.std.ShufflePlanes([y, blank, blank], [0, 1, 2], vs.YUV)


def _blur(c: vs.VideoNode, amount: float) -> vs.VideoNode:
    # AviSynth+ 3.7.5 Blur(amount): 7-bit SIMD weights, 16-bit C formula behind the last full SIMD block of a row
    half = int(32768 * 2.0 ** -amount + 0.5)
    t = (half + 256) >> 9
    kc = 'x ' + str(2 * half) + ' * {a} {b} + ' + str(32768 - half) + ' * + 32768 + 65536 / floor'
    ks = 'x ' + str(2 * t) + ' * {a} {b} + ' + str(64 - t) + ' * + 64 + 128 / floor'
    pv: List[str] = []
    ph: List[str] = []
    for p in range(c.format.num_planes):
        w = c.width >> (c.format.subsampling_w if p else 0)
        pv.append((ks if w >= 16 else kc).format(a='x[0,-1]', b='x[0,1]'))
        if w > 16:
            mod = 32 if w > 32 else 16
            lim = w if w % mod == 0 else (w // mod) * mod
            ph.append(f'X {lim} < ' + ks.format(a='x[-1,0]', b='x[1,0]') + ' ' + kc.format(a='x[-1,0]', b='x[1,0]') + ' ?')
        else:
            ph.append(kc.format(a='x[-1,0]', b='x[1,0]'))
    return _expr(_expr(c, pv), ph)


def _expand(c: vs.VideoNode, n: int = 1, both: bool = False) -> vs.VideoNode:
    for _ in range(n):
        c = core.std.Maximum(c, coordinates=[0, 1, 0, 1, 1, 0, 1, 0]) if both else core.std.Maximum(c)
    return c


def _inpand(c: vs.VideoNode, n: int = 1, both: bool = False) -> vs.VideoNode:
    for _ in range(n):
        c = core.std.Minimum(c, coordinates=[0, 1, 0, 1, 1, 0, 1, 0]) if both else core.std.Minimum(c)
    return c


def _thr(c: vs.VideoNode, t: int) -> vs.VideoNode:
    # mt_binarize(t) and Levels(t,1,t+1,0,255) on RGB32 (alpha included, no coring)
    return core.std.Expr(c, f'x {t} > 255 0 ?')


def _dist(c: vs.VideoNode, scale: float) -> vs.VideoNode:
    return core.avsinpaint.DistanceFunction(c, scale=float(scale))


def _balloon_px(m: vs.VideoNode, mode: int) -> vs.VideoNode:
    # balloonPX(); masks are GRAY8, the alpha channel of the original RGB32 masks
    if mode == 0:
        return m
    if mode in (1, 2, 3, 4):
        scale, t = {1: (128, 30), 2: (64, 70), 3: (48, 70), 4: (38, 70)}[mode]
        return _thr(_dist(m, scale), t)
    if mode in (-1, -2, -3, -4):
        for _ in range(-mode):
            m = _thr(_dist(m, -96), 150)
        return m
    if mode == -7:
        return _thr(_dist(m, -32), 240)
    if mode == 99:
        return _thr(_dist(_balloon_px(m, 3), -32), 110)
    if mode == 88:
        return _thr(_dist(_balloon_px(m, -1), 128), 30)
    raise _error(f'internal balloonPX mode {mode}.')


def _balloon_gr(m: vs.VideoNode, mode: int) -> vs.VideoNode:
    if mode == 0:
        return m
    return _dist(m, {1: 64, 2: 44, 3: 32}[mode])


# neo_fft.FFT3D rejects planes smaller than its block padding, which small "Loc" chroma planes are
_FFT_PAD = 24


def _mirror_pad(c: vs.VideoNode, pad: int) -> vs.VideoNode:
    sub = 1 << max(c.format.subsampling_w, c.format.subsampling_h) if c.format.num_planes > 1 else 1
    p = min(pad, c.width, c.height) // sub * sub
    if p == 0:
        return c
    c = core.std.StackHorizontal([core.std.FlipHorizontal(core.std.CropAbs(c, p, c.height, 0, 0)), c,
                                  core.std.FlipHorizontal(core.std.CropAbs(c, p, c.height, c.width - p, 0))])
    return core.std.StackVertical([core.std.FlipVertical(core.std.CropAbs(c, c.width, p, 0, 0)), c,
                                   core.std.FlipVertical(core.std.CropAbs(c, c.width, p, 0, c.height - p))])


def _fft3d(c: vs.VideoNode, sigma: float) -> vs.VideoNode:
    # Neo_FFT3D(sigma, y=3, u=3, v=3) plane by plane; only a plane FFT3D rejects is mirror-padded
    planes = [c] if c.format.num_planes == 1 else [_plane(c, p) for p in range(3)]
    out = []
    for pl in planes:
        try:
            out.append(core.neo_fft.FFT3D(pl, sigma=sigma))
        except vs.Error:
            padded = _mirror_pad(pl, _FFT_PAD)
            p = (padded.width - pl.width) // 2
            out.append(core.std.CropAbs(core.neo_fft.FFT3D(padded, sigma=sigma), pl.width, pl.height, p, p))
    return out[0] if len(out) == 1 else core.std.ShufflePlanes(out, [0, 0, 0], vs.YUV)


def _pp(c: vs.VideoNode, x: int, grey: bool = False) -> vs.VideoNode:
    # ppFX(): blur chain by abs(x), FFT3D denoise for negative x
    mode = abs(x)
    if mode == 0:
        return c
    if grey:
        c = _plane(c, 0)
    chain = {1: [0.3], 2: [0.7], 3: [1.58], 4: [1.0, 1.58], 5: [1.58] * 3, 6: [1.58] * 4, 7: [1.58] * 18, 8: [1.58] * 60}[mode]
    for amount in chain:
        c = _blur(c, amount)
    if x < 0:
        _require('neo_fft', 'negative oPP')
        c = _fft3d(c, 8.0 if x < -4 else 4.0)
    return _gray_to_yuv420(c) if grey else c


def _overlay(base: vs.VideoNode, ov: vs.VideoNode, mask: vs.VideoNode) -> vs.VideoNode:
    # Overlay(base, ov, mask=Y8) on YV12: float blend; chroma mask = vertical pairs rounded, then horizontal mean truncated
    blend = 'x y x - z 0.0039215688593685627 * * + 0.5 + trunc'
    w, h = mask.width // 2, mask.height // 2
    taps = [core.resize.Point(mask, w, h, src_left=dx, src_top=dy) for dx in (-0.5, 0.5) for dy in (-0.5, 0.5)]
    mc = _expr(taps, 'x y + 1 + 2 / floor z a + 1 + 2 / floor + 2 / floor')
    planes = [_expr([_plane(base, p), _plane(ov, p), mc if p else mask], blend) for p in range(3)]
    return core.std.ShufflePlanes(planes, [0, 0, 0], vs.YUV)


def _paste(base: vs.VideoNode, patch: vs.VideoNode, x: int, y: int) -> vs.VideoNode:
    # Overlay(base, patch, x=, y=) without mask, plane by plane
    if base.format.num_planes == 1:
        planes = [(base, patch, 0, 0)]
    else:
        planes = [(_plane(base, p), _plane(patch, p), base.format.subsampling_w if p else 0, base.format.subsampling_h if p else 0) for p in range(3)]
    out = []
    for bp, pp, sw, sh in planes:
        px, py = x >> sw, y >> sh
        borders = dict(left=px, top=py, right=bp.width - px - pp.width, bottom=bp.height - py - pp.height)
        big = core.std.AddBorders(pp, **borders)
        m = core.std.AddBorders(core.std.BlankClip(pp, color=(1 << pp.format.bits_per_sample) - 1), color=0, **borders)
        out.append(core.std.MaskedMerge(bp, big, m))
    return out[0] if len(out) == 1 else core.std.ShufflePlanes(out, [0, 0, 0], base.format.color_family)


def _range_props(plane: vs.VideoNode, mask: vs.VideoNode, name: str) -> Tuple[vs.VideoNode, vs.VideoNode]:
    # RT_YPlaneMinMaxDifference(mask=): pixels with 128 <= mask <= 255
    hi = core.std.PlaneStats(core.std.Expr([plane, mask], 'y 128 >= x 0 ?'), prop=name + 'h')
    lo = core.std.PlaneStats(core.std.Expr([plane, mask], 'y 128 >= x 255 ?'), prop=name + 'l')
    return hi, lo


def _read_mask(path: str, w: int, h: int, left: int, top: int, loc_w: int, loc_h: int) -> vs.VideoNode:
    # ImageSource(mask).Greyscale.ConvertToRGB32.Crop(Loc).Levels(254,1,255,0,255)
    _require('bs', 'mask files')
    if not os.path.isfile(path):
        raise _error('"mask" file not found!')
    if os.path.splitext(path)[1].lower() == '.ebmp':
        raise _error('ebmp masks are not supported.')
    img = core.bs.VideoSource(path, cachemode=0)[0]
    if (img.width, img.height) != (w, h):
        raise _error('Clip and "mask" image must have same dimensions.')
    if img.format.color_family == vs.GRAY:
        g = img if img.format.id == vs.GRAY8 else core.resize.Point(img, format=vs.GRAY8)
    else:
        if img.format.id != vs.RGB24:
            img = core.resize.Point(img, format=vs.RGB24, matrix_in_s='170m')
        g = _expr([_plane(img, p) for p in range(3)], 'x 9798 * y 19234 * + z 3736 * + 16384 + 32768 / floor')
    return _thr(core.std.CropAbs(g, loc_w, loc_h, left, top), 254)


def _parse_loc(loc: str, w: int, h: int) -> Tuple[int, int, int, int]:
    parts = loc.replace(' ', '').split(',')
    if len(parts) != 4:
        raise _error('Incorrect "Loc".')
    try:
        a, b, c, d = (int(float(p)) for p in parts)
    except ValueError:
        raise _error('Incorrect "Loc".') from None
    a = w + a if a < 0 else a
    b = h + b if b < 0 else b
    c = a + c - w if c > 0 else c
    d = b + d - h if d > 0 else d
    if c > 0 or d > 0 or a < 0 or b < 0:
        raise _error('Incorrect "Loc".')
    if a % 2 or b % 2 or c % 2 or d % 2:
        raise _error('Use even numbers for "Loc".')
    loc_w, loc_h = w - a + c, h - b + d
    if loc_w <= 0 or loc_h <= 0:
        raise _error('Incorrect "Loc".')
    return a, b, loc_w, loc_h


def _to_yv12(c: vs.VideoNode) -> vs.VideoNode:
    if c.format.id == vs.YUV420P8:
        return c
    if c.format.subsampling_w == 1 and c.format.subsampling_h == 1:
        return core.resize.Point(c, format=vs.YUV420P8, dither_type='none')
    return core.resize.Bicubic(c, format=vs.YUV420P8, dither_type='none')


def _from_yv12(c: vs.VideoNode, fmt: vs.VideoFormat) -> vs.VideoNode:
    if fmt.subsampling_w == 1 and fmt.subsampling_h == 1:
        return core.resize.Point(c, format=fmt.id)
    return core.resize.Bicubic(c, format=fmt.id)


def _optimize1(inyv12: vs.VideoNode, imgmask: vs.VideoNode, opaque: vs.VideoNode, turbo: int, grey: bool) -> vs.VideoNode:
    # per-frame preset choice by the luma/chroma range on a ring around the logo
    opt_bm = _expand(imgmask, 3)
    opt_m = _thr(core.std.Expr([_expand(opt_bm, 6), opt_bm], 'x y - 128 +'), 128)
    opt_mc = core.resize.Point(opt_m, inyv12.width // 2, inyv12.height // 2, src_left=-0.5, src_top=-0.5)
    opt_cb = _blur(inyv12, 1.58)

    def inpaint(src: vs.VideoNode, m: vs.VideoNode, **kw: float) -> vs.VideoNode:
        if grey:
            return _gray_to_yuv420(core.avsinpaint.InpaintLogo(_plane(src, 0), m, **kw))
        return core.avsinpaint.InpaintLogo(src, m, **kw)
    oo = [inpaint(opt_cb, opt_bm, sharpness=1.5, preblur=0.0, postblur=0.0, radius=r) for r in (2.0, 4.0, 6.0)]
    oo += [inpaint(inyv12, opaque, sharpness=45.0, preblur=pb, postblur=po, radius=r) for pb, po, r in ((6.5, 4.0, 8.0), (4.0, 8.0, 16.0), (4.0, 12.0, 16.0))]
    props = list(_range_props(_plane(opt_cb, 0), opt_m, 'Y'))
    names = ['x', 'y']
    if not grey:
        props += list(_range_props(_plane(opt_cb, 1), opt_mc, 'U')) + list(_range_props(_plane(opt_cb, 2), opt_mc, 'V'))
        names += ['z', 'a', 'b', 'c']

    def cond(t: int) -> str:
        parts = [f'{names[i]}.{n}hMax {names[i + 1]}.{n}lMin - {t} <' for i, n in zip(range(0, len(names), 2), 'YUV')]
        return ' '.join(parts) + ' and' * (len(parts) - 1)
    lo5 = '4' if turbo == -1 else f'{cond(45)} 4 5 ?'
    lo4 = '3' if turbo == 0 else f'{cond(35)} 3 {lo5} ?'
    expr = f'{cond(6)} 0 {cond(10)} 1 {cond(20)} 2 {lo4} ? ? ?'
    return core.akarin.Select(oo, props, expr)


def InpaintLoc(clp: vs.VideoNode, Loc: str = '0,0,0,0') -> vs.VideoNode:
    '''Highlight the "Loc" area (Left,Top,-Right,-Bottom or Left,Top,Width,Height) to find the crop values.'''
    if clp.format is None or clp.format.color_family != vs.YUV:
        raise _error('InpaintLoc needs a YUV clip.')
    a, b, w, h = _parse_loc(Loc, clp.width, clp.height)
    neutral = 1 << (clp.format.bits_per_sample - 1)
    area = core.std.CropAbs(clp, w, h, a, b)
    area = core.std.Expr(area, ['', f'x {neutral} - 2 / {neutral} +', f'x {neutral} - 2 / {neutral} + {neutral // 2} +'])
    return _paste(clp, area, a, b)


def InpaintDelogo(clp: vs.VideoNode, mask: Union[str, vs.VideoNode, None] = None, Loc: str = '', Mode: str = 'Inpaint',
                  Inflate: int = 1, KillShape: int = 0, Turbo: int = 0, Radius: float = 8.0, Sharpness: float = 45.0,
                  PreBlur: float = 6.5, PostBlur: float = 4.0, prePP: int = 0, oPP: int = -5, oPPm: int = 3, GrainPP: int = 0,
                  mShow: int = 0, Optimize1: Optional[int] = None, Greyscale: int = 0, **kwargs: Any) -> vs.VideoNode:
    '''
    Logo removal by inpainting (InpaintDelogo 3.7, Inpaint mode with a static mask).
    clp: YUV 8..16 bit; only the "Loc" area is processed in YUV420P8, pixels the delogo does not change keep the source values.
    mask: path to a bmp/png mask of the whole frame (logo 255, rest < 255) or a one-frame GRAY clip of the "Loc" size.
    Loc: "Left,Top,-Right,-Bottom" or "Left,Top,Width,Height", even numbers, 10-20 px border around the logo.
    The other parameters have the meaning and the defaults of the original script.
    '''
    for key, value in kwargs.items():
        if key not in _UNSUPPORTED:
            raise _error(f'unknown parameter "{key}".')
        if value != _UNSUPPORTED[key]:
            raise _error(f'"{key}" is not supported yet.')
    _require('avsinpaint', 'inpainting')
    _require('akarin', 'the mask helpers')
    fmt = clp.format
    if fmt is None or fmt.color_family != vs.YUV or fmt.sample_type != vs.INTEGER or fmt.bits_per_sample > 16:
        raise _error('only constant-format YUV 8..16 bit integer clips are supported.')
    if Mode.lower() != 'inpaint':
        raise _error(f'Mode "{Mode}" is not supported yet.')
    if not Loc:
        raise _error('You must define "Loc".')
    if Loc.replace(' ', '') == '100,100,-100,-100':
        raise _error('Use "InpaintLoc" function to get crop values for "Loc".')
    if mask is None or mask == '':
        raise _error('You have to define the "mask" string')
    checks = [('Inflate', Inflate, 0, 2), ('KillShape', KillShape, 0, 1), ('Turbo', Turbo, -2, 3), ('prePP', prePP, 0, 1),
              ('oPP', oPP, -8, 8), ('oPPm', oPPm, 0, 3), ('GrainPP', GrainPP, 0, 3), ('mShow', mShow, 0, 9), ('Greyscale', Greyscale, 0, 2)]
    for name, value, lo, hi in checks:
        if value < lo or value > hi:
            raise _error(f'"{name}" value must be between {lo} and {hi}.')
    if 4 < mShow < 9:
        raise _error('Calculating the meaning of life and everything. Meanwhile RTM.')
    if Optimize1 is None:
        Optimize1 = 0 if Turbo > 0 else 1
    if Optimize1 not in (0, 1):
        raise _error('"Optimize1" value must be between 0 and 1.')
    if Optimize1 and Turbo > 0:
        raise _error('"Optimize1" can\'t be used with positive "Turbo".')
    if Optimize1 and (Sharpness != 45 or PreBlur != 6.5 or PostBlur != 4 or Radius != 8):
        raise _error('"Optimize1" can\'t be used with custom inpainting settings.')
    grey = Greyscale == 1

    oppm = 0 if oPP == 0 else oPPm
    pre = 3 if prePP else 0
    opp = {1: 6, 2: 7, 3: 7}.get(Turbo, oPP)
    radius = {1: 6.0, 2: 4.0, 3: 2.0, -1: 16.0, -2: 16.0}.get(Turbo, Radius)
    sharpness = 0.0 if Turbo > 0 else Sharpness
    preblur = 4.0 if Turbo < 0 else 0.0 if Turbo > 0 else PreBlur
    postblur = {-1: 8.0, -2: 12.0}.get(Turbo, 0.0 if Turbo > 0 else PostBlur)

    a, b, loc_w, loc_h = _parse_loc(Loc, clp.width, clp.height)
    if fmt.subsampling_w and a % (1 << fmt.subsampling_w) or fmt.subsampling_h and b % (1 << fmt.subsampling_h):
        raise _error('"Loc" does not fit the chroma subsampling.')
    loc_src = core.std.CropAbs(clp, loc_w, loc_h, a, b)
    inyv12 = _to_yv12(loc_src)

    if isinstance(mask, vs.VideoNode):
        if mask.num_frames != 1:
            raise _error('dynamic masks are not supported yet.')
        if (mask.width, mask.height) != (loc_w, loc_h):
            raise _error('Clip and "mask" image must have same dimensions.')
        imgmask = core.resize.Point(_plane(mask, 0) if mask.format.color_family != vs.GRAY else mask, format=vs.GRAY8)
    else:
        imgmask = _read_mask(str(mask), clp.width, clp.height, a, b, loc_w, loc_h)
    imgmask = imgmask * clp.num_frames

    basemask_p = _balloon_px(imgmask, 99) if KillShape else imgmask
    basemask = _balloon_px(basemask_p, Inflate)
    opaque = basemask
    inpaint_mpp = _balloon_gr(opaque, oppm)

    if Optimize1:
        delogo = _optimize1(inyv12, imgmask, opaque, Turbo, grey)
    else:
        src = _pp(_plane(inyv12, 0) if grey else inyv12, pre)
        kw = dict(radius=float(radius), sharpness=float(sharpness), preblur=float(preblur), postblur=float(postblur))
        delogo = core.avsinpaint.InpaintLogo(src, opaque, **kw)
        delogo = _gray_to_yuv420(delogo) if grey else delogo

    out = _overlay(inyv12, _pp(delogo, opp, grey), inpaint_mpp)

    if GrainPP:
        from addGrain import GrainFactory3
        strength = {1: (1, 0, 0), 2: (2, 1, 1), 3: (2, 2, 2)}[GrainPP]
        grained = GrainFactory3(out, g1str=strength[0], g2str=strength[1], g3str=strength[2], temp_avg=60)
        out = _overlay(out, grained, _balloon_gr(basemask, 3))

    if mShow:
        red = core.resize.Point(core.std.BlankClip(out, format=vs.RGB24, color=[255, 0, 0]), format=vs.YUV420P8, matrix_s='170m')
        show_mask = {1: opaque, 2: inpaint_mpp, 3: basemask, 4: basemask, 9: inpaint_mpp}[mShow]
        out = _overlay(inyv12 if mShow in (4, 9) else out, red, show_mask)

    if fmt.id == vs.YUV420P8:
        result = _paste(clp, out, a, b)
    else:
        changed = _expr([_from_yv12(out, fmt), _from_yv12(inyv12, fmt), loc_src], 'x y = z x ?')
        result = _paste(clp, changed, a, b)
    if grey:
        neutral = 1 << (fmt.bits_per_sample - 1)
        result = core.std.Expr(result, ['', str(neutral), str(neutral)])
    return result
