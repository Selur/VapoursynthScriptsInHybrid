from __future__ import annotations
import math
import os
import struct
import sys
import zlib
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy
import vapoursynth as vs
from vapoursynth import core

# Port of InpaintDelogo 3.7 by VoodooFX (public domain): https://github.com/Purfview/InpaintDelogo
# Needs AvsInpaint >= 1.4 (core.avsinpaint) and akarin; BestSource for mask files, neo_fft for negative oPP/dPP, addGrain for GrainPP.
# The helpers below rebuild AviSynth+ 3.7.5 (AVX2) and MaskTools2 bit-exactly where the original relies on them.

_NAME = 'InpaintDelogo'
_VERSION = 1

# Parameters of later port stages; a value other than these defaults raises an error.
_UNSUPPORTED: Dict[str, Any] = {'DynMask': 0, 'Extract': 0}


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


def _check_inflate_border(before: vs.VideoNode, after: vs.VideoNode, left: int, top: int, right: int, bottom: int) -> None:
    # InpaintLogo fills the mask from the pixels around it; a mask grown by Inflate onto a "Loc" side inside the frame has none there
    old, new = _to_array(before[0]) > 127, _to_array(after[0]) > 127
    sides = {'top': (top, 0), 'bottom': (bottom, -1), 'left': (left, None), 'right': (right, None)}
    for side, (gap, row) in sides.items():
        if gap == 0:
            continue
        edge_old = old[row] if row is not None else old[:, 0 if side == 'left' else -1]
        edge_new = new[row] if row is not None else new[:, 0 if side == 'left' else -1]
        if edge_new.any() and not edge_old.any():
            raise _error(f'the mask grown by "Inflate" reaches the {side} "Loc" border; enlarge "Loc" or lower "Inflate".')


def _balloon_px(m: vs.VideoNode, mode: int) -> vs.VideoNode:
    # balloonPX(); masks are GRAY8, the alpha channel of the original RGB32 masks
    if mode == 0:
        return m
    if mode in (1, 2, 3, 4):
        scale, t = {1: (128, 30), 2: (64, 70), 3: (48, 70), 4: (38, 70)}[mode]
        return _thr(_dist(m, scale), t)
    if 5 <= mode <= 8:
        # Inflate beyond the original's 2: DistanceFunction falls by scale per pixel, so 128 / (n + 1) keeps exactly n pixels above 127
        return _thr(_dist(m, 128.0 / (mode + 1)), 127)
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
    if base.format.num_planes == 1:
        return _expr([base, ov, mask], blend)
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


def _limited_range(c: vs.VideoNode) -> bool:
    # _Range (R74+): RANGE_LIMITED; _ColorRange (older): 1 = limited; no property: YUV counts as limited
    props = c.get_frame(0).props
    if '_Range' in props:
        return int(props['_Range']) == int(getattr(vs, 'RANGE_LIMITED', 0))
    if '_ColorRange' in props:
        return int(props['_ColorRange']) == 1
    return c.format is not None and c.format.color_family == vs.YUV


def _read_mask(path: str, w: int, h: int, left: int, top: int, loc_w: int, loc_h: int, white: int = 255) -> vs.VideoNode:
    # ImageSource(mask).Greyscale.ConvertToRGB32.Crop(Loc).Levels(254,1,255,0,255); white=235 for TV-range clips
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
    area = core.std.CropAbs(g, loc_w, loc_h, left, top)
    peak = int(_to_array(area).max())
    if peak < white:
        raise _error(f'"mask" has no white ({white}+) pixel inside "Loc" (brightest value: {peak}); the logo must be white, the rest darker.')
    return _thr(area, white - 1)


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




def _opacity(base: vs.VideoNode, ov: vs.VideoNode, opacity: float) -> vs.VideoNode:
    # Overlay(base, ov, opacity=) without mask: float blend
    f = float(numpy.float32(opacity))
    return _expr([base, ov], f'x y x - {f!r} * + 0.5 + trunc')


def _lut(c: vs.VideoNode, table: Sequence[int]) -> vs.VideoNode:
    return core.std.Lut(c, lut=list(table))


# ConvertToRGB32 of a Y8 clip: limited range (Rec601/Rec709 give the same grey) and PC.601
_LIMITED_TO_RGB = [min(max(((v - 16) * 4777 + 2048) >> 12, 0), 255) for v in range(256)]
_PC601_TO_RGB = [min(v + (1 if v >= 232 else 0), 255) for v in range(256)]


def _amod(a: int, b: int) -> int:
    # AviSynth % keeps the sign of the dividend
    return int(math.fmod(a, b))


def _mtballoon2px(c: vs.VideoNode, mode: int) -> vs.VideoNode:
    return _expand(c, mode) if mode > 0 else _inpand(c, -mode)


def _ring(bm: vs.VideoNode) -> vs.VideoNode:
    # MTballoon2PX(6).mt_makediff(bm).mt_binarize(): ring around the mask
    return _thr(core.std.Expr([_expand(bm, 6), bm], 'x y - 128 +'), 128)


def _halve_point(c: vs.VideoNode) -> vs.VideoNode:
    return core.resize.Point(c, c.width // 2, c.height // 2, src_left=-0.5, src_top=-0.5)


class _Ring:
    # per-frame range of a blurred clip on a ring, as props for akarin.Select (RT_YPlaneMinMaxDifference(mask=))

    def __init__(self, cb: vs.VideoNode, ring: vs.VideoNode, grey: bool) -> None:
        self.props = list(_range_props(_plane(cb, 0), ring, 'Y'))
        self.names = ['x', 'y']
        if not grey:
            ringc = _halve_point(ring)
            self.props += list(_range_props(_plane(cb, 1), ringc, 'U')) + list(_range_props(_plane(cb, 2), ringc, 'V'))
            self.names += ['z', 'a', 'b', 'c']

    def cond(self, t: int) -> str:
        parts = [f'{self.names[i]}.{n}hMax {self.names[i + 1]}.{n}lMin - {t} <' for i, n in zip(range(0, len(self.names), 2), 'YUV')]
        return ' '.join(parts) + ' and' * (len(parts) - 1)

    def select(self, clips: Sequence[vs.VideoNode], expr: str) -> vs.VideoNode:
        return core.akarin.Select(list(clips), self.props, expr)


def _inpaint(src: vs.VideoNode, m: vs.VideoNode, grey: bool, radius: float, sharpness: float, preblur: float, postblur: float) -> vs.VideoNode:
    kw = dict(radius=float(radius), sharpness=float(sharpness), preblur=float(preblur), postblur=float(postblur))
    if grey:
        return _gray_to_yuv420(core.avsinpaint.InpaintLogo(_plane(src, 0), m, **kw))
    return core.avsinpaint.InpaintLogo(src, m, **kw)


def _optimize1(inyv12: vs.VideoNode, imgmask: vs.VideoNode, opaque: vs.VideoNode, turbo: int, grey: bool) -> vs.VideoNode:
    # Inpaint: per-frame preset choice by the luma/chroma range on a ring around the logo
    opt_bm = _expand(imgmask, 3)
    opt_cb = _blur(inyv12, 1.58)
    ring = _Ring(opt_cb, _ring(opt_bm), grey)
    oo = [_inpaint(opt_cb, opt_bm, grey, r, 1.5, 0, 0) for r in (2, 4, 6)]
    oo += [_inpaint(inyv12, opaque, grey, r, 45, pb, po) for pb, po, r in ((6.5, 4, 8), (4, 8, 16), (4, 12, 16))]
    lo5 = '4' if turbo == -1 else f'{ring.cond(45)} 4 5 ?'
    lo4 = '3' if turbo == 0 else f'{ring.cond(35)} 3 {lo5} ?'
    return ring.select(oo, f'{ring.cond(6)} 0 {ring.cond(10)} 1 {ring.cond(20)} 2 {lo4} ? ? ?')


def _frame_clip(arr: numpy.ndarray, length: int = 1) -> vs.VideoNode:
    blank = core.std.BlankClip(format=vs.GRAY8, width=arr.shape[1], height=arr.shape[0], length=1)

    def put(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        out = f.copy()
        numpy.copyto(numpy.asarray(out[0]), arr)
        return out
    return core.std.ModifyFrame(blank, blank, put) * length


def _to_array(c: vs.VideoNode, n: int = 0) -> numpy.ndarray:
    return numpy.asarray(c.get_frame(n)[0]).copy()


def _write_image(path: str, arr: numpy.ndarray) -> None:
    # grey image as 24-bit bmp or 8-bit png, chosen by the extension
    h, w = arr.shape
    if path.lower().endswith('.png'):
        raw = b''.join(b'\x00' + arr[y].tobytes() for y in range(h))

        def chunk(tag: bytes, data: bytes) -> bytes:
            return struct.pack('>I', len(data)) + tag + data + struct.pack('>I', zlib.crc32(tag + data) & 0xFFFFFFFF)
        data = b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0)) + chunk(b'IDAT', zlib.compress(raw, 9)) + chunk(b'IEND', b'')
    else:
        stride = (w * 3 + 3) & ~3
        rows = numpy.zeros((h, stride), numpy.uint8)
        rows[:, :w * 3] = numpy.repeat(arr, 3, axis=1)
        pixels = rows[::-1].tobytes()
        data = b'BM' + struct.pack('<IHHI', 54 + len(pixels), 0, 0, 54) + struct.pack('<IiiHHIIiiII', 40, w, h, 1, 24, 0, len(pixels), 2835, 2835, 0, 0) + pixels
    with open(path, 'wb') as fh:
        fh.write(data)


def _progress(what: str, done: int, total: int, last: List[int]) -> None:
    pct = done * 100 // max(total, 1)
    if pct != last[0]:
        last[0] = pct
        sys.stderr.write(f'{_NAME}: {what} {pct}%\n')
        sys.stderr.flush()


def _region(w: int, h: int, rects: Sequence[Tuple[int, int, int, int]]) -> numpy.ndarray:
    m = numpy.zeros((h, w), numpy.uint8)
    for x, y, rw, rh in rects:
        m[y:y + rh, x:x + rw] = 255
    return m


def _scan_clip(inyv12: vs.VideoNode, analyze: int, basemask_p: vs.VideoNode, noborder: str) -> vs.VideoNode:
    # AnalyzeMethods() 2: the border strips as a region mask; 1: the clip whose range decides about a frame
    vw, vh = inyv12.width, inyv12.height
    if analyze == 2:
        strips = {'top': (4, 4, vw - 8, 2), 'bottom': (4, vh - 6, vw - 8, 2), 'left': (4, 4, 2, vh - 8), 'right': (vw - 6, 4, 2, vh - 8)}
        swap = {'top': 'bottom', 'bottom': 'top', 'left': 'right', 'right': 'left'}
        if noborder in swap:
            strips[noborder] = strips[swap[noborder]]
        return _frame_clip(_region(vw, vh, list(strips.values())), inyv12.num_frames)
    m = _expand(basemask_p, 2)
    arr = _to_array(m)
    ys, xs = numpy.nonzero(arr == 255)
    if not len(xs):
        raise _error('Mask image is invalid.')
    mx, my, mw, mh = int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
    ml, mt, mr, mb = mx, my, -vw + mw + mx, -vh + my + mh
    if ml < 8 and -mr < 8 or mt < 8 and -mb < 8:
        raise _error('Mask image or Loc is invalid.')
    ml, mt, mr, mb = ml + _amod(ml, 2), mt + _amod(mt, 2), mr - _amod(mr, 2), mb - _amod(mb, 2)
    mid = int((vw - ml + mr) / 2) + ml
    mid += _amod(mid, 2)
    top = mt - 2 if mt >= -mb else vh + mb
    block = core.resize.Point(core.std.CropAbs(inyv12, 2, 2, mid - 2, top), vw, vh)
    ring = core.std.Expr([_expand(m, 2), m], 'x y = 0 255 ?')
    return _overlay(block, inyv12, ring)


def _select_frames(inyv12: vs.VideoNode, analyze: int, basemask_p: vs.VideoNode, th: int, noborder: str, grey: bool) -> List[int]:
    scan = _scan_clip(inyv12, analyze, basemask_p, noborder)
    stats = core.std.BlankClip(inyv12, format=vs.GRAY8, width=8, height=8)
    keys: List[Tuple[str, str]] = []
    if analyze == 2:
        planes = [(_plane(inyv12, 0), scan)]
        if not grey:
            planes += [(_plane(inyv12, 1), _halve_point(scan)), (_plane(inyv12, 2), _halve_point(scan))]
        for i, (pl, region) in enumerate(planes):
            n = 'YUV'[i]
            hi, lo = _range_props(pl, region, n)
            stats = core.std.CopyFrameProps(core.std.CopyFrameProps(stats, hi, props=[f'{n}hMax']), lo, props=[f'{n}lMin'])
            keys.append((f'{n}hMax', f'{n}lMin'))
    else:
        for i in range(1 if grey else 3):
            n = 'YUV'[i]
            stats = core.std.CopyFrameProps(stats, core.std.PlaneStats(scan, plane=i, prop=n), props=[f'{n}Max', f'{n}Min'])
            keys.append((f'{n}Max', f'{n}Min'))
    limits = [th, th // 2, th // 2]
    total = stats.num_frames
    selected: List[int] = []
    last = [-1]
    for frame_no, f in enumerate(stats.frames(close=True)):
        p = f.props
        if all(p[hi] - p[lo] <= limits[i] for i, (hi, lo) in enumerate(keys)):
            selected.append(frame_no)
        _progress('scan', frame_no + 1, total, last)
    return selected


def _write_frames(path: str, frames: List[int]) -> None:
    runs: List[List[int]] = []
    for n in frames:
        if runs and runs[-1][0] + runs[-1][1] == n:
            runs[-1][1] += 1
        else:
            runs.append([n, 1])
    with open(path, 'w', encoding='ascii', newline='\n') as fh:
        fh.write(f'# Total frames to analyze: {len(frames)}\n')
        fh.writelines(f'{s},{-c}\n' for s, c in runs)


def _read_frames(path: str, total: int) -> List[int]:
    frames: List[int] = []
    with open(path, encoding='ascii') as fh:
        for line in fh:
            line = line.split('#')[0].strip()
            if not line:
                continue
            s, e = (int(v) for v in line.split(','))
            end = s - e - 1 if e < 0 else (total - 1 if e == 0 else e)
            frames.extend(range(s, end + 1))
    return frames


def _loop_count(n: int) -> int:
    for limit, k in ((10, 200), (20, 40), (40, 20), (80, 10), (160, 5), (320, 2)):
        if n < limit:
            return k
    return 1


def _pick(c: vs.VideoNode, frames: List[int]) -> vs.VideoNode:
    blank = core.std.BlankClip(c, length=len(frames))
    return core.std.FrameEval(blank, lambda n: c[frames[n]])


def _analysis_frames(inyv12: vs.VideoNode, analyze: int, basemask_p: vs.VideoNode, th: int, noborder: str, grey: bool,
                     logo_id: str, reanalyze: int) -> List[int]:
    if analyze == 3:
        return list(range(inyv12.num_frames)) * _loop_count(inyv12.num_frames)
    txt = logo_id + '.txt'
    if os.path.isfile(txt) and not reanalyze:
        frames = _read_frames(txt, inyv12.num_frames)
    else:
        frames = _select_frames(inyv12, analyze, basemask_p, th, noborder, grey)
        if len(frames) < 2:
            where = 'with a flat "Loc" border' if analyze == 2 else 'with a flat area next to the logo'
            raise _error(f'No frames {where} found to analyze; increase "AnalyzeTh", use a smaller "Loc" or Analyze=3.')
        _write_frames(txt, frames)
    return frames * _loop_count(len(frames))


_LOGO_KEYS = ('cy', 'ay', 'cu', 'au', 'cv', 'av', 'base')


def _analyze(inyv12: vs.VideoNode, imgmask: vs.VideoNode, analyze: int, th: int, noborder: str, grey: bool,
             logo_id: str, reanalyze: int) -> Dict[str, numpy.ndarray]:
    # positive Analyze: frame selection, AnalyzeLogo of Y and of YUV; results cached in <logo_id>.npz
    npz = logo_id + '.npz'
    w, h = inyv12.width, inyv12.height
    if os.path.isfile(npz) and not reanalyze:
        with numpy.load(npz) as data:
            if int(data['version']) == _VERSION and tuple(int(v) for v in data['size']) == (w, h):
                return {k: data[k].copy() for k in _LOGO_KEYS}
    frames = _analysis_frames(inyv12, analyze, imgmask, th, noborder, grey, logo_id, reanalyze)
    sel = _pick(inyv12, frames)
    sys.stderr.write(f'{_NAME}: analyzing the logo on {len(frames)} frames\n')
    basemask_y = _expand(imgmask[0], 2)
    ym = _to_array(core.avsinpaint.AnalyzeLogo(_plane(sel, 0), basemask_y))
    res = {'cy': ym[:h], 'ay': ym[h:], 'base': _to_array(imgmask)}
    if grey:
        res.update(cu=numpy.full((h // 2, w // 2), 128, numpy.uint8), cv=numpy.full((h // 2, w // 2), 128, numpy.uint8))
        res.update(au=numpy.zeros((h // 2, w // 2), numpy.uint8), av=numpy.zeros((h // 2, w // 2), numpy.uint8))
    else:
        full = core.avsinpaint.AnalyzeLogo(sel, basemask_y)[0]
        color = core.std.CropAbs(full, w, h, 0, 0)
        uva = _to_array(core.resize.Spline36(_plane(core.std.CropAbs(full, w, h, 0, h), 0), w // 2, h // 2))
        res.update(cu=_to_array(_plane(color, 1)), cv=_to_array(_plane(color, 2)), au=uva, av=uva.copy())
    arrays: Dict[str, Any] = dict(res, version=_VERSION, size=numpy.array([w, h]))
    # written next to the target and renamed, so a job waiting for the file never reads it half-written
    tmp = npz[:-4] + '.tmp.npz'
    numpy.savez(tmp, **arrays)
    os.replace(tmp, npz)
    return res


def _automask(clp: vs.VideoNode, inyv12: vs.VideoNode, mask_path: str, a: int, b: int, whole: bool, analyze: int, th: int,
              noborder: str, reanalyze: int, amix: int, logo_id: str) -> vs.VideoNode:
    # Automask=1: estimate the logo alpha and write the base mask to "mask"; the output shows mask and alpha
    if analyze not in (2, 3):
        raise _error('"Automask" needs "Analyze" 2 or 3.')
    folder = os.path.dirname(mask_path)
    if folder and not os.path.isdir(folder):
        raise _error(folder + ' folder not found!')
    w, h = inyv12.width, inyv12.height
    frame = numpy.zeros((clp.height, clp.width), numpy.uint8)
    if whole:
        frame[b + 1:b + h - 1, a + 1:a + w - 1] = 255
    else:
        frame[b:b + h, a:a + w] = 255
    amerge = core.std.CropAbs(_inpand(_frame_clip(frame), 6 if whole else 7), w, h, a, b)
    base, ext = os.path.splitext(mask_path)
    auto_id = f'{base}_AutomaskTemp_A{analyze}{ext}'
    if os.path.isfile(auto_id) and not reanalyze:
        img = core.bs.VideoSource(auto_id, cachemode=0)[0]
        alpha = _to_array(core.resize.Point(_plane(img, 0), format=vs.GRAY8))
    else:
        if analyze == 2:
            frames = _analysis_frames(inyv12, 2, amerge, th, noborder, True, logo_id, reanalyze)
            sel = _pick(_plane(inyv12, 0), frames)
        else:
            sel = _plane(inyv12, 0)
        found = core.avsinpaint.AnalyzeLogo(sel, amerge)[0]
        alpha = _to_array(_lut(core.std.CropAbs(found, w, h, 0, h), _LIMITED_TO_RGB))
        _write_image(auto_id, alpha)
    full = numpy.zeros((clp.height, clp.width), numpy.uint8)
    full[b:b + h, a:a + w] = alpha
    binary = numpy.where(full > int(-amix * 3 + 19), 255, 0).astype(numpy.uint8)
    _write_image(mask_path, binary)
    return _gray_to_yuv420(core.std.StackVertical([_frame_clip(binary[b:b + h, a:a + w]), _frame_clip(alpha)]))


def _edit_masks(logo: Dict[str, numpy.ndarray], imgmask: vs.VideoNode, opts: Dict[str, Any]) -> Dict[str, vs.VideoNode]:
    # ebmp mask editing of the original (expert parameters), on one-frame GRAY clips
    clips = {k: _frame_clip(v) for k, v in logo.items()}
    w, h = logo['cy'].shape[1], logo['cy'].shape[0]
    ext = opts['YAlphaExternal']
    if isinstance(ext, vs.VideoNode):
        if (ext.width, ext.height) != (w, h):
            raise _error('Wrong "YAlphaExternal" resolution.')
        clips['ay'] = core.resize.Point(_plane(ext[0], 0) if ext.format.color_family != vs.GRAY else ext[0], format=vs.GRAY8)
    if opts['UVAlphaFromY']:
        clips['au'] = clips['av'] = core.resize.Spline64(clips['ay'], w // 2, h // 2)
    for key, name in (('ay', 'YAlphaTweak'), ('au', 'UAlphaTweak'), ('av', 'VAlphaTweak')):
        if opts[name]:
            clips[key] = core.std.Expr(clips[key], f'x {opts[name]} +')
    for key, name in (('cy', 'YColorSolid'), ('cu', 'UColorSolid'), ('cv', 'VColorSolid')):
        if opts[name] > -1:
            clips[key] = core.std.BlankClip(clips[key], color=opts[name])
    solid = [f'{p}{k}Solid{io}' for p in 'YUV' for k in ('Alpha', 'Color') for io in ('Outs', 'Ins')]
    if any(opts[s] > -1 for s in solid):
        y = imgmask[0]
        masks = {'Y': (_mtballoon2px(y, opts['YOffsetOuts']), _mtballoon2px(y, opts['YOffsetIns'])),
                 'UV': (_halve_point(_mtballoon2px(y, opts['UVOffsetOuts'])), _halve_point(_mtballoon2px(y, opts['UVOffsetIns'])))}
        for p in 'YUV':
            mo, mi = masks['Y' if p == 'Y' else 'UV']
            for k, key in (('Color', 'c' + p.lower()), ('Alpha', 'a' + p.lower())):
                outs, ins = opts[f'{p}{k}SolidOuts'], opts[f'{p}{k}SolidIns']
                cur = clips[key]
                if outs > -1:
                    cur = _overlay(core.std.BlankClip(cur, color=outs), cur, mo)
                if ins > -1:
                    cur = _overlay(cur, core.std.BlankClip(cur, color=ins), mi)
                clips[key] = cur
    if opts['ShiftMasksVe'] or opts['ShiftMasksHo']:
        ho, ve = opts['ShiftMasksHo'], opts['ShiftMasksVe']
        for key in ('cy', 'ay'):
            clips[key] = core.resize.Spline64(clips[key], w, h, src_left=ho, src_top=ve, src_width=w, src_height=h)
        for key in ('cu', 'au', 'cv', 'av'):
            clips[key] = core.resize.Spline64(clips[key], w // 2, h // 2, src_left=ho / 2.0, src_top=ve / 2.0, src_width=w // 2, src_height=h // 2)
    return clips


def _edge_mask(source: vs.VideoNode, wide: int) -> vs.VideoNode:
    # EdgePP: ring between balloonPX(+k) and balloonPX(-k) of the source mask
    k = wide + 1
    return core.std.Expr([_balloon_px(source, k), _balloon_px(source, -k)], 'y 255 = 0 x ?')


# --- Temporal: hidden pixels from neighbour frames before the spatial inpainting (extension of this port, not in the original)

def _dilate_np(m: numpy.ndarray, r: int) -> numpy.ndarray:
    # round dilation by r pixels of a boolean array
    out = m.copy()
    h, w = m.shape
    for dy in range(-r, r + 1):
        for dx in range(-r, r + 1):
            if dy * dy + dx * dx > r * r or (dy == 0 and dx == 0):
                continue
            out[max(0, dy):h - max(0, -dy), max(0, dx):w - max(0, -dx)] |= m[max(0, -dy):h - max(0, dy), max(0, -dx):w - max(0, dx)]
    return out


def _phase_prep(a: numpy.ndarray, hidden: numpy.ndarray) -> numpy.ndarray:
    a = a.astype(numpy.float32)
    vis = ~hidden
    mean = float(a[vis].mean()) if vis.any() else float(a.mean())
    a[hidden] = mean
    a -= mean
    win = numpy.outer(numpy.hanning(a.shape[0]), numpy.hanning(a.shape[1])).astype(numpy.float32)
    return numpy.fft.rfft2(a * win)


def _phase_offset(fa: numpy.ndarray, fb: numpy.ndarray) -> Tuple[float, float, float]:
    # where the content of a is found in b (subpixel by a parabola fit) and the correlation peak
    cross = fa * numpy.conj(fb)
    cross /= numpy.abs(cross) + 1e-9
    corr = numpy.fft.irfft2(cross)
    h, w = corr.shape
    y, x = numpy.unravel_index(int(numpy.argmax(corr)), corr.shape)

    def sub(c: numpy.ndarray, i: int, n: int) -> float:
        m, p = c[(i - 1) % n], c[(i + 1) % n]
        return float((m - p) / (2 * (m - 2 * c[i] + p) + 1e-12))
    dy = y + sub(corr[:, x], y, h)
    dx = x + sub(corr[y, :], x, w)
    dy = dy - h if dy > h / 2 else dy
    dx = dx - w if dx > w / 2 else dx
    return -dx, -dy, float(corr[y, x])


def _bilinear(plane: numpy.ndarray, ys: numpy.ndarray, xs: numpy.ndarray) -> numpy.ndarray:
    x0 = numpy.floor(xs).astype(int)
    y0 = numpy.floor(ys).astype(int)
    fx, fy = xs - x0, ys - y0
    x1 = numpy.minimum(x0 + 1, plane.shape[1] - 1)
    y1 = numpy.minimum(y0 + 1, plane.shape[0] - 1)
    p = plane
    return p[y0, x0] * (1 - fx) * (1 - fy) + p[y0, x1] * fx * (1 - fy) + p[y1, x0] * (1 - fx) * fy + p[y1, x1] * fx * fy


class _TemporalFill:
    # window = Loc plus a margin (YUV420P8); every tile of the hole takes the nearest neighbour frame whose shifted
    # content matches on a ring around the logo; the shift is the window's or the tile's own, whichever fits the ring better
    TILE, TILE_MARGIN, RING, MAX_DIFF, MIN_PEAK, TILE_FALLBACK, DIRECT, HALF = 48, 48, 8, 8.0, 0.12, 8, True, True

    def __init__(self, win: vs.VideoNode, hole: numpy.ndarray, logo: numpy.ndarray, loc_box: Tuple[int, int, int, int], radius: int) -> None:
        import threading
        self.win, self.hole, self.logo, self.radius = win, hole, logo, radius
        h, w = hole.shape
        ringm = _dilate_np(logo, self.RING) & ~logo
        self.ring = numpy.nonzero(ringm)
        ring_id = numpy.full(hole.shape, -1, numpy.int64)
        ring_id[self.ring] = numpy.arange(len(self.ring[0]))
        self.tiles = []
        l, t, r, b = loc_box
        for ty in range(t, b, self.TILE):
            for tx in range(l, r, self.TILE):
                box = (tx, ty, min(tx + self.TILE, r), min(ty + self.TILE, b))
                if not hole[box[1]:box[3], box[0]:box[2]].any():
                    continue
                m = self.TILE_MARGIN
                tw = (max(0, tx - m), max(0, ty - m), min(w, box[2] + m), min(h, box[3] + m))
                ring = ringm[tw[1]:tw[3], tw[0]:tw[2]]
                if ring.sum() < 50:
                    ring = ~logo[tw[1]:tw[3], tw[0]:tw[2]]
                ry, rx = numpy.nonzero(ring)
                ry, rx = ry + tw[1], rx + tw[0]
                ids = ring_id[ry, rx]
                # a tile whose ring lies in the whole ring reads its error from there; otherwise it measures its own
                self.tiles.append((box, tw, (ry, rx), ids if (ids >= 0).all() else None))
        self.lock = threading.Lock()
        self.steps: Dict[int, Tuple[float, float]] = {}
        self.direct: Dict[Tuple[int, int], Tuple[float, float]] = {}
        self.busy: Dict[int, Any] = {}
        self.planes: Dict[int, List[numpy.ndarray]] = {}
        self.ffts: Dict[Tuple[int, int], numpy.ndarray] = {}
        self.results: Dict[int, Tuple[List[numpy.ndarray], numpy.ndarray]] = {}

    def _get(self, n: int) -> List[numpy.ndarray]:
        n = min(max(n, 0), self.win.num_frames - 1)
        with self.lock:
            if n in self.planes:
                return self.planes[n]
        f = self.win.get_frame(n)
        planes = [numpy.asarray(f[p]).astype(numpy.float32) for p in range(3)]
        with self.lock:
            self.planes[n] = planes
            while len(self.planes) > 2 * self.radius + 16:
                self.planes.pop(next(iter(self.planes)))
        return planes

    def _fft(self, n: int, i: int) -> numpy.ndarray:
        key = (n, i)
        with self.lock:
            if key in self.ffts:
                return self.ffts[key]
        y = self._get(n)[0]
        if i < 0 and self.HALF:
            # the window at half size: 2x2 means, a 2x2 block counts as hidden if any pixel is
            hh, ww = y.shape[0] // 2 * 2, y.shape[1] // 2 * 2
            small = y[:hh, :ww].reshape(hh // 2, 2, ww // 2, 2).mean(axis=(1, 3))
            hidden = self.logo[:hh, :ww].reshape(hh // 2, 2, ww // 2, 2).any(axis=(1, 3))
            value = _phase_prep(small, hidden)
        elif i < 0:
            value = _phase_prep(y, self.logo)
        else:
            l, t, r, b = self.tiles[i][1]
            value = _phase_prep(y[t:b, l:r], self.logo[t:b, l:r])
        with self.lock:
            self.ffts[key] = value
            while len(self.ffts) > (2 * self.radius + 16) * (len(self.tiles) + 1):
                self.ffts.pop(next(iter(self.ffts)))
        return value

    def _step(self, n: int) -> Tuple[float, float]:
        # window shift from frame n to n + 1, measured once
        with self.lock:
            if n in self.steps:
                return self.steps[n]
        dx, dy, _ = _phase_offset(self._fft(n, -1), self._fft(n + 1, -1))
        if self.HALF:
            dx, dy = 2 * dx, 2 * dy
        with self.lock:
            self.steps[n] = (dx, dy)
        return dx, dy

    def _offset(self, n: int, m: int, chain: Dict[int, Tuple[float, float]]) -> Tuple[float, float]:
        # where the content of frame n is in frame m: the chained window steps say whether it moved, a direct measurement says how far
        lo, hi = min(n, m), max(n, m)
        if self.DIRECT:
            cx, cy = chain[m][0] - chain[n][0], chain[m][1] - chain[n][1]
            if abs(cx) < 0.5 and abs(cy) < 0.5:
                return 0.0, 0.0
            key = (lo, hi)
            with self.lock:
                known = self.direct.get(key)
            if known is None:
                known = _phase_offset(self._fft(lo, -1), self._fft(hi, -1))[:2]
                if self.HALF:
                    known = (2 * known[0], 2 * known[1])
                with self.lock:
                    self.direct[key] = known
                    while len(self.direct) > (2 * self.radius + 16) ** 2:
                        self.direct.pop(next(iter(self.direct)))
            return known if m > n else (-known[0], -known[1])
        return chain[m][0] - chain[n][0], chain[m][1] - chain[n][1]

    def _ring_error(self, cur: List[numpy.ndarray], nb: List[numpy.ndarray], ry: numpy.ndarray, rx: numpy.ndarray,
                    dx: float, dy: float) -> Optional[float]:
        h, w = cur[0].shape
        sy, sx = ry + dy, rx + dx
        inside = (sy >= 0) & (sy <= h - 2) & (sx >= 0) & (sx <= w - 2)
        if inside.sum() < 0.8 * len(ry):
            return None
        return float(numpy.abs(_bilinear(nb[0], sy[inside], sx[inside]) - cur[0][ry[inside], rx[inside]]).mean())

    def fill(self, n: int) -> Tuple[List[numpy.ndarray], numpy.ndarray]:
        import threading
        with self.lock:
            if n in self.results:
                return self.results[n]
            busy = self.busy.setdefault(n, threading.Lock())
        # the picture and the mask clip ask for the same frame, often from two threads: one computes, the other waits
        with busy:
            with self.lock:
                if n in self.results:
                    return self.results[n]
            res = self._fill(n)
            with self.lock:
                self.results[n] = res
                while len(self.results) > 16:
                    self.results.pop(next(iter(self.results)))
                self.busy.pop(n, None)
        return res

    def _fill(self, n: int) -> Tuple[List[numpy.ndarray], numpy.ndarray]:
        cur = self._get(n)
        h, w = cur[0].shape
        todo = self.hole.copy()
        out = [c.copy() for c in cur]
        order = sorted((k for k in range(-self.radius, self.radius + 1) if k and 0 <= n + k < self.win.num_frames), key=abs)
        # chained window positions of the neighbours relative to frame n
        lo, hi = max(0, n - self.radius), min(self.win.num_frames - 1, n + self.radius)
        chain: Dict[int, Tuple[float, float]] = {n: (0.0, 0.0)}
        for i in range(n, hi):
            sx, sy = self._step(i)
            chain[i + 1] = (chain[i][0] + sx, chain[i][1] + sy)
        for i in range(n - 1, lo - 1, -1):
            sx, sy = self._step(i)
            chain[i] = (chain[i + 1][0] - sx, chain[i + 1][1] - sy)
        ring_diff: Dict[int, numpy.ndarray] = {}
        for i, ((cl, ct, cr, cb), _, (ry, rx), ids) in enumerate(self.tiles):
            for k in order:
                tile_todo = todo[ct:cb, cl:cr]
                if not tile_todo.any():
                    break
                m = n + k
                dx, dy = self._offset(n, m, chain)
                if abs(dx) < 0.5 and abs(dy) < 0.5:
                    continue  # the background did not move: nothing new to see
                nb = self._get(m)
                if ids is not None:
                    if m not in ring_diff:
                        # per-pixel ring error of the window shift, once per neighbour; NaN where the shifted ring leaves the window
                        sy, sx = self.ring[0] + dy, self.ring[1] + dx
                        inside = (sy >= 0) & (sy <= h - 2) & (sx >= 0) & (sx <= w - 2)
                        diff = numpy.full(len(sy), numpy.nan, numpy.float32)
                        diff[inside] = numpy.abs(_bilinear(nb[0], sy[inside], sx[inside]) - cur[0][self.ring[0][inside], self.ring[1][inside]])
                        ring_diff[m] = diff
                    part = ring_diff[m][ids]
                    valid = ~numpy.isnan(part)
                    err = float(part[valid].mean()) if valid.sum() >= 0.8 * len(part) else None
                else:
                    err = self._ring_error(cur, nb, ry, rx, dx, dy)
                if (err is None or err > self.MAX_DIFF) and abs(k) <= self.TILE_FALLBACK:
                    # the window shift does not fit this tile (parallax): its own shift, if the correlation is clear
                    tdx, tdy, peak = _phase_offset(self._fft(n, i), self._fft(m, i))
                    if peak < self.MIN_PEAK or (abs(tdx) < 0.5 and abs(tdy) < 0.5):
                        continue
                    err = self._ring_error(cur, nb, ry, rx, tdx, tdy)
                    if err is None or err > self.MAX_DIFF:
                        continue
                    dx, dy = tdx, tdy
                elif err is None or err > self.MAX_DIFF:
                    continue
                ty, tx = numpy.nonzero(tile_todo)
                ty, tx = ty + ct, tx + cl
                sy, sx = ty + dy, tx + dx
                ok = (sy >= 0) & (sy <= h - 2) & (sx >= 0) & (sx <= w - 2)
                ok &= ~self.logo[numpy.clip(numpy.rint(sy).astype(int), 0, h - 1), numpy.clip(numpy.rint(sx).astype(int), 0, w - 1)]
                if not ok.any():
                    continue
                ty, tx, sy, sx = ty[ok], tx[ok], sy[ok], sx[ok]
                out[0][ty, tx] = _bilinear(nb[0], sy, sx)
                cy, cx = ty // 2, tx // 2
                for p in (1, 2):
                    out[p][cy, cx] = _bilinear(nb[p], numpy.clip(cy + dy / 2, 0, nb[p].shape[0] - 2), numpy.clip(cx + dx / 2, 0, nb[p].shape[1] - 2))
                todo[ty, tx] = False
        return [numpy.clip(numpy.rint(o), 0, 255).astype(numpy.uint8) for o in out], todo


def _temporal_fill(clp: vs.VideoNode, fmt: vs.VideoFormat, loc_src: vs.VideoNode, inyv12: vs.VideoNode, basemask: vs.VideoNode,
                   a: int, b: int, loc_w: int, loc_h: int, radius: int) -> Tuple[vs.VideoNode, vs.VideoNode]:
    # the clip with the hidden pixels other frames show, and the Loc-sized mask of the pixels still to inpaint
    margin = 160
    wl, wt = max(0, a - margin) & ~1, max(0, b - margin) & ~1
    wr, wb = min(clp.width, a + loc_w + margin) & ~1, min(clp.height, b + loc_h + margin) & ~1
    win = _to_yv12(core.std.CropAbs(clp, wr - wl, wb - wt, wl, wt))
    hole = numpy.zeros((wb - wt, wr - wl), bool)
    hole[b - wt:b - wt + loc_h, a - wl:a - wl + loc_w] = _to_array(basemask[0]) > 127
    tf = _TemporalFill(win, hole, _dilate_np(hole, 2), (a - wl, b - wt, a - wl + loc_w, b - wt + loc_h), radius)
    ox, oy = a - wl, b - wt

    def put_planes(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        o = f.copy()
        planes, _ = tf.fill(n)
        numpy.copyto(numpy.asarray(o[0]), planes[0][oy:oy + loc_h, ox:ox + loc_w])
        for p in (1, 2):
            numpy.copyto(numpy.asarray(o[p]), planes[p][oy // 2:(oy + loc_h) // 2, ox // 2:(ox + loc_w) // 2])
        return o

    def put_rest(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        o = f.copy()
        _, todo = tf.fill(n)
        numpy.copyto(numpy.asarray(o[0]), (todo[oy:oy + loc_h, ox:ox + loc_w] * 255).astype(numpy.uint8))
        return o
    filled = core.std.ModifyFrame(inyv12, inyv12, put_planes)
    rest = core.std.ModifyFrame(core.std.BlankClip(inyv12, format=vs.GRAY8), core.std.BlankClip(inyv12, format=vs.GRAY8), put_rest)
    if fmt.id == vs.YUV420P8:
        changed = filled
    else:
        changed = _expr([_from_yv12(filled, fmt), _from_yv12(inyv12, fmt), loc_src], 'x y = z x ?')
    return _paste(clp, changed, a, b), rest


def _tile(c: vs.VideoNode, label: str) -> vs.VideoNode:
    # one labelled tile of the Show grids, RGB24: GRAY masks as grey, YUV through BT.601 TV range
    if c.format.color_family == vs.GRAY:
        rgb = core.std.ShufflePlanes([c, c, c], [0, 0, 0], vs.RGB)
    else:
        rgb = core.resize.Point(c, format=vs.RGB24, matrix_in_s='170m', range_in_s='limited')
    # text.Text needs at least 40x48 pixels; the half-size chroma tiles of a small Loc stay unlabelled
    return core.text.Text(rgb, label, alignment=7) if label and rgb.width >= 40 and rgb.height >= 48 else rgb


def _show_grid(rows: List[List[Tuple[vs.VideoNode, str]]], frames: int, border: int = 0) -> vs.VideoNode:
    def fit(c: vs.VideoNode) -> vs.VideoNode:
        c = c * frames if c.num_frames == 1 else c
        return core.std.AddBorders(c, right=border, bottom=border, color=[0xb7, 0xb7, 0x15]) if border else c
    return core.std.StackVertical([core.std.StackHorizontal([fit(_tile(c, label)) for c, label in row]) for row in rows])


# --- Track: a logo that moves, found per frame by its shape (extension of this port, not in the original)

def _box_mean(a: numpy.ndarray, r: int) -> numpy.ndarray:
    p = numpy.pad(a, r, mode='edge')
    c = numpy.pad(numpy.cumsum(numpy.cumsum(p, 0, dtype=numpy.float64), 1), ((1, 0), (1, 0)))
    k = 2 * r + 1
    return ((c[k:, k:] - c[:-k, k:] - c[k:, :-k] + c[:-k, :-k]) / (k * k)).astype(numpy.float32)


def _highpass(a: numpy.ndarray, r: int = 3) -> numpy.ndarray:
    return a - _box_mean(a, r)


def _half(a: numpy.ndarray) -> numpy.ndarray:
    h, w = a.shape[0] // 2 * 2, a.shape[1] // 2 * 2
    return a[:h, :w].reshape(h // 2, 2, w // 2, 2).mean(axis=(1, 3)).astype(numpy.float32)


class _LogoTracker:
    # normalized cross-correlation of high-pass luma with the high-passed logo shape: a semi-transparent logo adds edges, its brightness
    # depends on the background; coarse search at half size over the area (FFT), then +-3 pixels at full size
    MARGIN, REFINE = 4, 3

    def __init__(self, shape: numpy.ndarray, area: Tuple[int, int]) -> None:
        ys, xs = numpy.nonzero(shape)
        m = self.MARGIN
        t, l = max(0, int(ys.min()) - m), max(0, int(xs.min()) - m)
        self.shape = shape[t:int(ys.max()) + 1 + m, l:int(xs.max()) + 1 + m]
        h, w = area
        if self.shape.shape[0] + 2 * self.REFINE >= h or self.shape.shape[1] + 2 * self.REFINE >= w:
            raise _error('the logo of "mask" does not fit into "Loc" for "Track".')
        coarse = _highpass(_half(self.shape.astype(numpy.float32) * 255))
        coarse -= coarse.mean()
        self.cn = float(numpy.sqrt((coarse.astype(numpy.float64) ** 2).sum()))
        self.ch, self.cw = coarse.shape
        self.H, self.W = h // 2, w // 2
        pad = numpy.zeros((self.H, self.W), numpy.float32)
        pad[:self.ch, :self.cw] = coarse
        self.f_t = numpy.conj(numpy.fft.rfft2(pad))
        fine = _highpass(self.shape.astype(numpy.float32) * 255)
        self.fine = fine - fine.mean()
        self.fn = float(numpy.sqrt((self.fine.astype(numpy.float64) ** 2).sum()))

    def find(self, y: numpy.ndarray) -> Tuple[int, int, float]:
        hp = _highpass(_half(y.astype(numpy.float32)))
        corr = numpy.fft.irfft2(numpy.fft.rfft2(hp) * self.f_t, s=hp.shape)
        e = numpy.pad(numpy.cumsum(numpy.cumsum(numpy.pad(hp.astype(numpy.float64) ** 2, ((0, self.ch), (0, self.cw))), 0), 1), ((1, 0), (1, 0)))
        ch, cw, H, W = self.ch, self.cw, self.H, self.W
        energy = e[ch:ch + H, cw:cw + W] - e[:H, cw:cw + W] - e[ch:ch + H, :W] + e[:H, :W]
        ncc = corr / (self.cn * numpy.sqrt(numpy.maximum(energy, 1e-6)))
        ncc[H - ch:, :] = -1
        ncc[:, W - cw:] = -1
        iy, ix = numpy.unravel_index(int(numpy.argmax(ncc)), ncc.shape)
        return self._refine(y, 2 * int(ix), 2 * int(iy))

    def _refine(self, y: numpy.ndarray, x0: int, y0: int) -> Tuple[int, int, float]:
        h, w = self.fine.shape
        r = self.REFINE
        best = (x0, y0, -1.0)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                px, py = x0 + dx, y0 + dy
                if px < 0 or py < 0 or py + h > y.shape[0] or px + w > y.shape[1]:
                    continue
                l, t = max(0, px - 4), max(0, py - 4)
                patch = _highpass(y[t:py + h + 4, l:px + w + 4].astype(numpy.float32))
                win = patch[py - t:py - t + h, px - l:px - l + w]
                v = float((win * self.fine).sum() / (self.fn * numpy.sqrt((win.astype(numpy.float64) ** 2).sum()) + 1e-6))
                if v > best[2]:
                    best = (px, py, v)
        return best


def TrackLogo(clp: vs.VideoNode, mask: str, Loc: str = '0,0,0,0', Grow: int = 3, TrackTh: float = 0.35) -> vs.VideoNode:
    """
    Mask clip (GRAY8, "Loc" size, 255 = logo) of a logo that moves: "mask" is a whole-frame image with the logo in white at any
    place, it is searched in every frame inside "Loc"; frames whose best match is below TrackTh (0..1) stay empty.
    Grow: pixels the found shape is grown by (shadows, edges). InpaintDelogo(..., Track=1) uses it with Grow=Inflate.
    """
    a, b, loc_w, loc_h = _parse_loc(Loc, clp.width, clp.height)
    template = _to_array(_read_mask(str(mask), clp.width, clp.height, 0, 0, clp.width, clp.height, 235 if _limited_range(clp) else 255)) > 127
    tracker = _LogoTracker(template, (loc_h, loc_w))
    shape = _dilate_np(numpy.pad(tracker.shape, Grow), Grow) if Grow > 0 else tracker.shape
    luma = core.resize.Point(_plane(core.std.CropAbs(clp, loc_w, loc_h, a, b), 0), format=vs.GRAY8)
    blank = core.std.BlankClip(luma, format=vs.GRAY8, color=0)

    def put(n: int, f: List[vs.VideoFrame]) -> vs.VideoFrame:
        out = f[1].copy()
        dst = numpy.asarray(out[0])
        dst[:] = 0
        x, y, peak = tracker.find(numpy.asarray(f[0][0]))
        if peak >= TrackTh:
            x, y = x - Grow, y - Grow
            sh, sw = shape.shape
            t0, l0 = max(0, -y), max(0, -x)
            t1, l1 = min(sh, loc_h - y), min(sw, loc_w - x)
            dst[y + t0:y + t1, x + l0:x + l1][shape[t0:t1, l0:l1]] = 255
        out.props['InpaintDelogoTrackPeak'] = peak
        return out
    return core.std.ModifyFrame(blank, [luma, blank], put)


def _fit_view(view: vs.VideoNode, clp: vs.VideoNode) -> vs.VideoNode:
    # a view of another size and format, fitted top left on black into a clip of the input's size, format, length and rate
    scale = min(clp.width / view.width, clp.height / view.height)
    if scale >= 1:
        view = core.resize.Point(view, view.width * int(scale), view.height * int(scale))
    else:
        view = core.resize.Bilinear(view, max(2, int(view.width * scale) & ~1), max(2, int(view.height * scale) & ~1))
    rgb = view.format.color_family == vs.RGB
    view = core.std.AddBorders(view, right=clp.width - view.width, bottom=clp.height - view.height, color=[0, 0, 0] if rgb else [16, 128, 128])
    if clp.format.color_family == vs.GRAY:
        view = core.std.ShufflePlanes(view, 0, vs.GRAY) if not rgb else core.resize.Point(view, format=vs.GRAY8, matrix_s='709')
    view = core.resize.Point(view, format=clp.format.id, matrix_s='709') if rgb and clp.format.color_family != vs.RGB else core.resize.Point(view, format=clp.format.id)
    view = view * clp.num_frames if view.num_frames == 1 else view
    return core.std.CopyFrameProps(core.std.AssumeFPS(view, src=clp), clp)


def PreviewView(clp: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
    '''InpaintDelogo whose output is a view (Show, Automask) of another size, fitted top left into a clip of the input's size and format.'''
    return _fit_view(InpaintDelogo(clp, **kwargs), clp)


def AutomaskView(clp: vs.VideoNode, **kwargs: Any) -> vs.VideoNode:
    '''InpaintDelogo with Automask=1; the view (mask above alpha) fitted into the input's size and format.'''
    kwargs.pop('Automask', None)
    return PreviewView(clp, Automask=1, **kwargs)


def InpaintLoc(clp: vs.VideoNode, Loc: str = '0,0,0,0') -> vs.VideoNode:
    '''Highlight the "Loc" area (Left,Top,-Right,-Bottom or Left,Top,Width,Height) to find the crop values.'''
    if clp.format is None or clp.format.color_family != vs.YUV:
        raise _error('InpaintLoc needs a YUV clip.')
    a, b, w, h = _parse_loc(Loc, clp.width, clp.height)
    neutral = 1 << (clp.format.bits_per_sample - 1)
    area = core.std.CropAbs(clp, w, h, a, b)
    area = core.std.Expr(area, ['', f'x {neutral} - 2 / {neutral} +', f'x {neutral} - 2 / {neutral} + {neutral // 2} +'])
    return _paste(clp, area, a, b)


_EDIT_DEFAULTS: Dict[str, Any] = {
    'YAlphaExternal': None, 'UVAlphaFromY': 0, 'YAlphaTweak': 0, 'UAlphaTweak': 0, 'VAlphaTweak': 0,
    'YColorSolid': -1, 'UColorSolid': -1, 'VColorSolid': -1, 'YColorSolidOuts': -1, 'UColorSolidOuts': -1, 'VColorSolidOuts': -1,
    'YColorSolidIns': -1, 'UColorSolidIns': -1, 'VColorSolidIns': -1, 'YAlphaSolidOuts': -1, 'UAlphaSolidOuts': -1, 'VAlphaSolidOuts': -1,
    'YAlphaSolidIns': -1, 'UAlphaSolidIns': -1, 'VAlphaSolidIns': -1, 'YOffsetIns': -1, 'YOffsetOuts': 1, 'UVOffsetIns': -2,
    'UVOffsetOuts': 2, 'ShiftMasksVe': 0, 'ShiftMasksHo': 0,
}
_EDIT_RANGES: Dict[str, Tuple[int, int]] = {
    'UVAlphaFromY': (0, 1), 'YAlphaTweak': (-10, 10), 'UAlphaTweak': (-10, 10), 'VAlphaTweak': (-10, 10), 'YOffsetIns': (-4, 0),
    'YOffsetOuts': (0, 4), 'UVOffsetIns': (-4, 0), 'UVOffsetOuts': (0, 4), 'ShiftMasksVe': (-1, 1), 'ShiftMasksHo': (-1, 1),
}


def InpaintDelogo(clp: vs.VideoNode, mask: Union[str, vs.VideoNode, None] = None, Loc: str = '', Mode: str = 'Inpaint',
                  Analyze: Optional[int] = None, AnalyzeTh: Optional[int] = None, NoBorderAt: str = 'none', ReAnalyze: int = 0,
                  Automask: int = 0, aMix: int = 0, Inflate: int = 1, KillShape: int = 0, Turbo: int = 0, Radius: float = 8.0,
                  Sharpness: float = 45.0, PreBlur: float = 6.5, PostBlur: float = 4.0, prePP: int = 0,
                  Interp: int = 2, IntSpd: int = 1, InterpM: int = 0, iTune: int = 0, iTinflate: int = 0, oTune: int = 0, oTinflate: int = 0,
                  dPP: int = -3, dPPm: int = 3, diPP: int = -5, diPPm: int = 3, oPP: int = -5, oPPm: int = 3,
                  EdgePP: int = 0, EdgePos: int = 0, EdgeWide: int = 1, GrainPP: int = 0, mShow: int = 0,
                  Optimize1: Optional[int] = None, Optimize2: int = 3, Greyscale: int = 0, WholeDeblend: int = 0,
                  TriggerDynamic: int = 0, DumpMasks: int = 0, LogoPath: str = '', Temporal: int = 0, Show: int = 0, Track: int = 0,
                  TrackTh: float = 0.35,
                  **kwargs: Any) -> vs.VideoNode:
    '''
    Logo removal by inpainting and deblending (InpaintDelogo 3.7).
    clp: YUV 8..16 bit; only the "Loc" area is processed in YUV420P8, pixels the delogo does not change keep the source values.
    mask: path to a bmp/png mask of the whole frame (logo 255, rest < 255) or a one-frame GRAY clip of the "Loc" size.
    Loc: "Left,Top,-Right,-Bottom" or "Left,Top,Width,Height", even numbers, 10-20 px border around the logo.
    LogoPath: base path of the analysis files (.txt frame list, .npz masks); default: the original's name next to the mask.
    Inflate: 0..8 pixels; the original allows 0..2, values above 2 are an extension of this port.
    Track: 1 = the logo moves: "mask" is searched in every frame inside "Loc" (TrackLogo(), Inflate grows the found shape, TrackTh is the
      lowest match that counts, 0..1); Inpaint mode only, an extension of this port.
    Show: 1 or 2, the original's grids of masks and delogo steps (Loc area, RGB24, another size than the input); 3..12 are not ported.
    Temporal: 0..100 neighbour frames on each side that may show the hidden pixels (motion around Loc, block-wise); only the rest
      is inpainted. Inpaint mode with a static mask only; an extension of this port.
    The other parameters have the meaning and the defaults of the original; ebmp files and negative Analyze are not supported.
    '''
    opts = dict(_EDIT_DEFAULTS)
    dyn_opts = dict(_DYN_DEFAULTS)
    for key, value in kwargs.items():
        if key in _EDIT_DEFAULTS:
            opts[key] = value
        elif key in _DYN_DEFAULTS:
            dyn_opts[key] = value
        elif key in _UNSUPPORTED:
            if value != _UNSUPPORTED[key]:
                raise _error(f'"{key}" is not supported yet.')
        else:
            raise _error(f'unknown parameter "{key}".')
    _require('avsinpaint', 'inpainting')
    _require('akarin', 'the mask helpers')
    fmt = clp.format
    if fmt is None or fmt.color_family != vs.YUV or fmt.sample_type != vs.INTEGER or fmt.bits_per_sample > 16:
        raise _error('only constant-format YUV 8..16 bit integer clips are supported.')
    mode = Mode.lower()
    if mode not in ('inpaint', 'deblend', 'both'):
        raise _error('This mode doesn\'t exist. Change "Mode" value.')
    if not Loc:
        if not Automask:
            raise _error('You must define "Loc".')
        Loc = '0,0,0,0'  # Automask on the whole frame
    if Loc.replace(' ', '') == '100,100,-100,-100':
        raise _error('Use "InpaintLoc" function to get crop values for "Loc".')
    if Track:
        # the mask image is the template; the per-frame mask clip of the found logo runs as a dynamic mask (DynMask=1)
        if mode != 'inpaint' or Automask or not isinstance(mask, str):
            raise _error('"Track" needs "Inpaint" mode and "mask" as the path of a mask image (the logo to search for).')
        if Temporal:
            raise _error('"Track" can\'t be used with "Temporal".')
        mask = TrackLogo(clp, mask, Loc=Loc, Grow=Inflate, TrackTh=TrackTh)
    dyn_opts['KillShape'] = KillShape
    dyn = _dyn_settings(dyn_opts, mode, mask)
    dynmask = dyn['DynMask']
    if (mask is None or mask == '') and dynmask in (0, 1) or (mask is None or mask == '') and dynmask == 2 and mode == 'both':
        raise _error('You have to define the "mask" string')
    if isinstance(mask, str) and mask.lower().endswith('.ebmp'):
        raise _error('ebmp masks are not supported.')
    if Analyze is None:
        Analyze = 2
    if AnalyzeTh is None:
        AnalyzeTh = 30 if clp.num_frames < 45000 else 16
    checks = [('Analyze', Analyze, 1, 3), ('AnalyzeTh', AnalyzeTh, 1, 255), ('ReAnalyze', ReAnalyze, 0, 1), ('Automask', Automask, 0, 1),
              ('aMix', aMix, -50, 6), ('Inflate', Inflate, 0, 8), ('KillShape', KillShape, 0, 1), ('Turbo', Turbo, -2, 3),
              ('prePP', prePP, 0, 1), ('Interp', Interp, 0, 4), ('IntSpd', IntSpd, 1, 4), ('InterpM', InterpM, -1, 2),
              ('iTune', iTune, -10, 10), ('iTinflate', iTinflate, 0, 3), ('oTune', oTune, -10, 10), ('oTinflate', oTinflate, 0, 3),
              ('dPP', dPP, -8, 8), ('dPPm', dPPm, 1, 3), ('diPP', diPP, -8, 8), ('diPPm', diPPm, 0, 3), ('oPP', oPP, -8, 8),
              ('oPPm', oPPm, 0, 3), ('EdgePP', EdgePP, 0, 3), ('EdgePos', EdgePos, -1, 1), ('EdgeWide', EdgeWide, 0, 3),
              ('GrainPP', GrainPP, 0, 3), ('mShow', mShow, 0, 9), ('Optimize2', Optimize2, 0, 3), ('Greyscale', Greyscale, 0, 2),
              ('WholeDeblend', WholeDeblend, 0, 1), ('TriggerDynamic', TriggerDynamic, 0, 1), ('DumpMasks', DumpMasks, 0, 2),
              ('Temporal', Temporal, 0, 100), ('Show', Show, 0, 12), ('Track', Track, 0, 1), ('TrackTh', TrackTh, 0, 1)]
    checks += [(k, opts[k], lo, hi) for k, (lo, hi) in _EDIT_RANGES.items()]
    checks += [(k, opts[k], -1, 255) for k in _EDIT_DEFAULTS if 'Solid' in k]
    for name, value, lo, hi in checks:
        if value < lo or value > hi:
            raise _error(f'"{name}" value must be between {lo} and {hi}.')
    if NoBorderAt not in ('top', 'bottom', 'left', 'right', 'none'):
        raise _error('"NoBorderAt" string must be \'top\', \'bottom\', \'left\', \'right\' or \'none\'.')
    if mode == 'inpaint' and 4 < mShow < 9:
        raise _error('Calculating the meaning of life and everything. Meanwhile RTM.')
    if Show > 2:
        raise _error(f'"Show={Show}" is not supported yet (only 1 and 2).')
    if Show and Temporal:
        raise _error('"Show" can\'t be used with "Temporal".')
    if mode != 'inpaint' and KillShape:
        raise _error('"KillShape" parameter is only for "Inpaint" mode.')
    if mode == 'inpaint' and EdgePP:
        raise _error('"EdgePP" parameter is not for "Inpaint" mode.')
    if mShow == 8 and EdgePP == 0:
        raise _error('Incompatible "EdgePP" and "mShow" parameters.')
    if Automask and Analyze == 1:
        raise _error('"Analyze" 1 & -1 can\'t be used with "Automask=1".')
    whole = Loc.replace(' ', '') in ('0,0,0,0', '0,0,-0,-0')
    if WholeDeblend and not whole:
        raise _error('"WholeDeblend" can be used only on a whole frame.')
    if Optimize1 is None:
        Optimize1 = 0 if Turbo > 0 or dynmask > 0 else 1
    if Optimize1 and dynmask:
        raise _error('"Optimize1" can\'t be used with "DynMask".')
    if Optimize1 not in (0, 1):
        raise _error('"Optimize1" value must be between 0 and 1.')
    if Optimize1 and Turbo > 0:
        raise _error('"Optimize1" can\'t be used with positive "Turbo".')
    if Optimize1 and (Sharpness != 45 or PreBlur != 6.5 or PostBlur != 4 or Radius != 8):
        raise _error('"Optimize1" can\'t be used with custom inpainting settings.')
    grey = Greyscale == 1
    if DumpMasks == 2:
        opts.update(YColorSolidOuts=0, UColorSolidOuts=128, VColorSolidOuts=128, YAlphaSolidOuts=0, UAlphaSolidOuts=0, VAlphaSolidOuts=0)

    analyze = 0 if mode == 'inpaint' and not Automask else Analyze
    interp = 0.0 if mode == 'inpaint' else {1: 0.84, 2: 0.73, 3: 0.59, 4: 0.40}.get(Interp, 0.0)
    interpm = 0 if mode == 'inpaint' or mode == 'both' and InterpM == -1 else InterpM
    interpm_infl = 0 if interpm == -1 else interpm
    dppm = 0 if dPP == 0 and interp < 1 else dPPm
    dippm = 0 if diPP == 0 else diPPm
    oppm = 0 if oPP == 0 else oPPm
    pre = 3 if prePP else 0
    tunelow, itunelow = int(-oTune * 10 + 140), int(-iTune * 10 + 140)
    if mode == 'inpaint':
        opp = {1: 6, 2: 7, 3: 7}.get(Turbo, oPP)
    else:
        opp = 6 if Turbo > 0 and mode == 'both' else oPP
    radius = {1: 6.0, 2: 4.0, 3: 2.0, -1: 16.0, -2: 16.0}.get(Turbo, Radius)
    sharpness = 0.0 if Turbo > 0 else Sharpness
    preblur = 4.0 if Turbo < 0 else 0.0 if Turbo > 0 else PreBlur
    postblur = {-1: 8.0, -2: 12.0}.get(Turbo, 0.0 if Turbo > 0 else PostBlur)

    a, b, loc_w, loc_h = _parse_loc(Loc, clp.width, clp.height)
    if fmt.subsampling_w and a % (1 << fmt.subsampling_w) or fmt.subsampling_h and b % (1 << fmt.subsampling_h):
        raise _error('"Loc" does not fit the chroma subsampling.')
    loc_src = core.std.CropAbs(clp, loc_w, loc_h, a, b)
    inyv12 = _to_yv12(loc_src)

    mask_name = mask if isinstance(mask, str) else 'Clip'
    coords = f'{a}-{b}-{clp.width - a - loc_w}-{clp.height - b - loc_h}'
    suffix = f'_A{analyze}' if analyze == 3 else f'_A{analyze}-{AnalyzeTh}'
    logo_id = LogoPath or f'{mask_name}_InpaintDelogo3_{coords}{suffix}'

    if Automask:
        if not isinstance(mask, str):
            raise _error('"Automask" needs "mask" as the path of the image to create.')
        return _automask(clp, inyv12, mask, a, b, whole, analyze, AnalyzeTh, NoBorderAt, ReAnalyze, aMix, logo_id)

    if dynmask == 1 or dynmask > 1 and not (dynmask == 2 and mode == 'both'):
        # the original's placeholder base mask for dynamic masks is a "Dummy" caption; an empty mask here
        imgmask = core.std.BlankClip(inyv12, format=vs.GRAY8, length=1)
    elif isinstance(mask, vs.VideoNode):
        if (mask.width, mask.height) != (loc_w, loc_h):
            raise _error('Clip and "mask" image must have same dimensions.')
        imgmask = core.resize.Point(_plane(mask, 0) if mask.format.color_family != vs.GRAY else mask, format=vs.GRAY8)
    else:
        imgmask = _read_mask(str(mask), clp.width, clp.height, a, b, loc_w, loc_h, 235 if _limited_range(clp) else 255)
    imgmask = imgmask * clp.num_frames

    basemask_p = _balloon_px(imgmask, 99) if KillShape else imgmask
    basemask = _balloon_px(basemask_p, Inflate)
    if Inflate > 2 and not (isinstance(mask, vs.VideoNode) and mask.num_frames > 1):
        _check_inflate_border(basemask_p, basemask, a, b, clp.width - a - loc_w, clp.height - b - loc_h)
    if Temporal:
        if mode != 'inpaint' or dynmask or (isinstance(mask, vs.VideoNode) and mask.num_frames > 1):
            raise _error('"Temporal" works only in "Inpaint" mode with a static mask.')
        pre, rest = _temporal_fill(clp, fmt, loc_src, inyv12, basemask, a, b, loc_w, loc_h, Temporal)
        # the rest mask is already grown by Inflate; it is a per-frame mask clip, so the second pass runs with DynMask=1
        return InpaintDelogo(pre, mask=rest, Loc=Loc, Inflate=0, KillShape=KillShape, Turbo=Turbo, Radius=Radius, Sharpness=Sharpness,
                             PreBlur=PreBlur, PostBlur=PostBlur, prePP=prePP, oPP=oPP, oPPm=oPPm, GrainPP=GrainPP, Optimize1=0,
                             Greyscale=Greyscale)
    interp_mask = _balloon_px(basemask, interpm_infl)

    planes: Dict[str, vs.VideoNode] = {}
    if analyze:
        logo = _analyze(inyv12, imgmask, analyze, AnalyzeTh, NoBorderAt, grey, logo_id, ReAnalyze)
        planes = _edit_masks(logo, imgmask, opts)
        if DumpMasks:
            top = numpy.hstack([_to_array(planes['cy']), numpy.vstack([_to_array(planes['cu']), _to_array(planes['au'])])])
            mid = numpy.hstack([_to_array(planes['ay']), numpy.vstack([_to_array(planes['cv']), _to_array(planes['av'])])])
            bot = numpy.hstack([logo['base'], numpy.zeros((loc_h, loc_w // 2), numpy.uint8)])
            _write_image(logo_id + '_dump.png', numpy.vstack([top, mid, bot]))
        if TriggerDynamic:
            planes = {k: v * 2 for k, v in planes.items()}
        logo_alpha = _lut(planes['ay'][0], _PC601_TO_RGB) * clp.num_frames
    else:
        logo_alpha = imgmask
    opaque_tune = _balloon_px(_thr(logo_alpha, tunelow), oTinflate)
    interp_tune_p = _thr(logo_alpha, itunelow)
    interp_tune = _balloon_px(interp_tune_p, iTinflate)
    if interpm == -1:
        interp_mask = interp_tune
    opaque = basemask if mode == 'inpaint' else opaque_tune
    deblend_mpp = _balloon_gr(basemask, dppm)
    inpaint_mpp = _balloon_gr(opaque, oppm)
    itune_pp = _balloon_gr(interp_tune, dippm)
    edge_mask = None
    if EdgePP and mode != 'inpaint':
        edge_mask = _edge_mask(_balloon_px(interp_tune_p if interpm < 0 else basemask_p, EdgePos), EdgeWide)

    if mode != 'inpaint':
        outs = [core.avsinpaint.DeblendLogo(_plane(inyv12, 0), planes['cy'], planes['ay'])]
        for p, c, al in ((1, 'cu', 'au'), (2, 'cv', 'av')):
            outs.append(_plane(inyv12, p) if grey else core.avsinpaint.DeblendLogo(_plane(inyv12, p), planes[c], planes[al]))
        debl_pre = core.std.ShufflePlanes(outs, [0, 0, 0], vs.YUV)
        inpa_pre: Optional[vs.VideoNode] = None
        if interp > 0 and IntSpd > 2 and Optimize1:
            opt2_v = debl_pre if WholeDeblend or interpm == -1 else inyv12
            opt2_bm = _expand(interp_mask, 2)
            opt2_cb = _blur(opt2_v, 1.58)
            ring2 = _Ring(opt2_cb, _ring(opt2_bm), grey)
            ooo = [_inpaint(opt2_cb, opt2_bm, grey, r, 1.5, 0, 0) for r in (2, 6)]
            ooo += [_inpaint(inyv12, interp_mask, grey, r, 45, pb, po) for pb, po, r in ((6.5, 4, 8), (4, 8, 16))]
            lo3 = '2' if IntSpd == 3 else f'{ring2.cond(60)} 2 3 ?'
            inpa_pre = ring2.select(ooo, f'{ring2.cond(15)} 0 {ring2.cond(40)} 1 {lo3} ? ?')
        elif interp > 0:
            inin1 = debl_pre if WholeDeblend or interpm == -1 else inyv12
            r, sh, pb, po = {1: (2, 1.5, 0, 0), 2: (6, 1.5, 0, 0), 3: (8, 45, 6.5, 4), 4: (16, 45, 4, 8)}[IntSpd]
            inpa_pre = _inpaint(inin1, interp_mask, grey, r, sh, pb, po)
        if Optimize2 and interp and inpa_pre is not None:
            ring3 = _Ring(_blur(inyv12, 1.58), _ring(_expand(imgmask, 3)), grey)
            oooo = [_opacity(inpa_pre, debl_pre, 0.40), _opacity(inpa_pre, debl_pre, interp), debl_pre]
            lo1 = '1' if Optimize2 == 2 else f'{ring3.cond(30)} 1 2 ?'
            logo_deblend = ring3.select(oooo, lo1 if Optimize2 == 1 else f'{ring3.cond(15)} 0 {lo1} ?')
        elif interp and inpa_pre is not None:
            logo_deblend = _opacity(inpa_pre, debl_pre, interp)
        else:
            logo_deblend = debl_pre
        if dynmask:
            dyn_mask = _dynamic_mask(inyv12, mask, mode, dyn, clp.width > 1024, logo_deblend)
            inpaint_mpp = _mtballoon_gr(dyn_mask, oppm)
        delogo = logo_deblend if mode == 'deblend' else _inpaint(logo_deblend, dyn_mask if dynmask else opaque, grey, radius, sharpness, preblur, postblur)
    elif Optimize1:
        delogo = _optimize1(inyv12, imgmask, opaque, Turbo, grey)
    else:
        target = opaque
        if dynmask:
            target = _dynamic_mask(inyv12, mask, mode, dyn, clp.width > 1024, None)
            inpaint_mpp = _mtballoon_gr(target, oppm)
        src = _pp(_plane(inyv12, 0) if grey else inyv12, pre)
        kw = dict(radius=float(radius), sharpness=float(sharpness), preblur=float(preblur), postblur=float(postblur))
        delogo = core.avsinpaint.InpaintLogo(src, target, **kw)
        delogo = _gray_to_yuv420(delogo) if grey else delogo

    delogo2 = delogo
    if edge_mask is not None:
        r, sh, pb, po = {1: (2, 1.5, 0, 0), 2: (6, 1.5, 0, 0), 3: (8, 45, 6.5, 4)}[EdgePP]
        delogo2 = _inpaint(delogo, edge_mask, grey, r, sh, pb, po)

    if mode == 'inpaint':
        out = _overlay(inyv12, _pp(delogo, opp, grey), inpaint_mpp)
    elif not WholeDeblend:
        if dPP:
            out = _overlay(inyv12, _pp(delogo2, dPP, grey), deblend_mpp)
        else:
            out = _overlay(inyv12, delogo2, deblend_mpp if dppm else basemask)
    else:
        out = _overlay(delogo2, _pp(delogo2, dPP, grey), deblend_mpp) if dPP else delogo2
    if mode == 'deblend' and InterpM == -1 and diPP:
        out = _overlay(out, _pp(out, diPP, grey), itune_pp)
    if mode == 'both' and oPP:
        out = _overlay(out, _pp(out, opp, grey), inpaint_mpp)

    if GrainPP:
        from addGrain import GrainFactory3
        strength = {1: (1, 0, 0), 2: (2, 1, 1), 3: (2, 2, 2)}[GrainPP]
        grained = GrainFactory3(out, g1str=strength[0], g2str=strength[1], g3str=strength[2], temp_avg=60)
        out = _overlay(out, grained, _balloon_gr(basemask, 3))

    delogo_pp = out
    if mShow:
        red = core.resize.Point(core.std.BlankClip(out, format=vs.RGB24, color=[255, 0, 0]), format=vs.YUV420P8, matrix_s='170m')
        masks = {1: opaque, 2: inpaint_mpp, 3: basemask, 4: basemask, 5: deblend_mpp, 6: opaque_tune, 7: interp_mask, 8: edge_mask, 9: inpaint_mpp}
        show_mask = masks[mShow]
        assert show_mask is not None
        out = _overlay(inyv12 if mShow in (4, 5, 6, 9) else out, red, show_mask)

    if Show in (1, 2):
        # the original's Show 1/2 grids of the Loc area (RGB24, other size than the input: PreviewView fits it into the clip)
        n = clp.num_frames
        blank = core.std.BlankClip(imgmask, format=vs.GRAY8, color=0)
        dyn_view = dyn_mask if dynmask and mode != 'inpaint' else (target if dynmask and mode == 'inpaint' and not Optimize1 else None)
        if Show == 1:
            logo_color = None
            if planes:
                logo_color = core.std.ShufflePlanes([planes['cy'], planes['cu'], planes['cv']], [0, 0, 0], vs.YUV)
            mask1 = (imgmask, 'base mask') if mode == 'inpaint' else (logo_alpha, 'logo alpha')
            mask2 = (inpaint_mpp, 'opaque pp') if mode == 'inpaint' else (deblend_mpp, 'deblend pp')
            if mode == 'deblend':
                mask3 = (interp_mask, 'interp')
            elif mode == 'inpaint':
                mask3 = (dyn_view, 'dynamic mask') if dyn_view is not None else (opaque, 'opaque mask')
            else:
                mask3 = (inpaint_mpp, 'opaque pp')
            if mode == 'both':
                mask6 = (dyn_view, 'dynamic mask') if dyn_view is not None else (opaque, 'otune mask')
            else:
                mask6 = (edge_mask, 'edge pp') if edge_mask is not None else (blank, 'disabled')
            rows = [[(inyv12, 'source'), (delogo, 'delogo'), (delogo_pp, 'delogo pp')], [mask1, mask2, mask3]]
            if mode != 'inpaint':
                rows.append([(logo_color if logo_color is not None else blank, 'logo color'), (basemask, 'base mask'), mask6])
            if mode == 'both':
                rows.append([(blank, 'disabled'), (interp_mask, 'interp on') if interp else (blank, 'interp off'), (blank, 'disabled')])
            return _show_grid(rows, n)
        if analyze and planes:
            chroma = [[(planes['cu'], 'U c'), (planes['cv'], 'V c')], [(planes['au'], 'U a'), (planes['av'], 'V a')]]
            chroma_tile = core.resize.Point(_show_grid(chroma, n), format=vs.YUV420P8, matrix_s='170m')
            rows = [[(inyv12, '0: source'), (debl_pre, '1: deblend'), (inpa_pre, f'2: IntSpd={IntSpd}') if interp and inpa_pre is not None else (blank, '2: Interp Off')],
                    [(out, '4: final'), (delogo, '3: pre-PP'),
                     (core.std.ShufflePlanes([planes['cy'], planes['cu'], planes['cv']], [0, 0, 0], vs.YUV), 'logo demo')],
                    [(planes['ay'], 'Y alpha'), (planes['cy'], 'Y color'), (chroma_tile, '')],
                    [(imgmask, 'basemask'), (interp_mask, 'interp mask'), (opaque, 'oTune mask') if mode == 'both' else (blank, 'disabled')]]
            return _show_grid(rows, n, border=2)
        if dyn_view is not None:
            return _show_grid([[(delogo, 'delogo'), (dyn_view, 'dynamic mask')], [(out, 'final'), (inpaint_mpp, 'dynamic pp mask')]], n)
        return _show_grid([[(delogo, 'delogo')], [(out, 'final')]], n)

    if whole:
        result = out if fmt.id == vs.YUV420P8 else _expr([_from_yv12(out, fmt), _from_yv12(inyv12, fmt), clp], 'x y = z x ?')
    elif fmt.id == vs.YUV420P8:
        result = _paste(clp, out, a, b)
    else:
        changed = _expr([_from_yv12(out, fmt), _from_yv12(inyv12, fmt), loc_src], 'x y = z x ?')
        result = _paste(clp, changed, a, b)
    if grey:
        neutral = 1 << (fmt.bits_per_sample - 1)
        result = core.std.Expr(result, ['', str(neutral), str(neutral)])
    return result


def _inflate(c: vs.VideoNode, n: int = 1) -> vs.VideoNode:
    # mt_inflate(): mean of the 8 neighbours, truncated, never below the centre
    for _ in range(n):
        c = _expr(c, 'x[-1,-1] x[0,-1] x[1,-1] x[-1,0] x[1,0] x[-1,1] x[0,1] x[1,1] + + + + + + + 8 / floor x max')
    return c


def _expand_thr(c: vs.VideoNode, th: int) -> vs.VideoNode:
    # mt_expand(thY=, mode="square"): separable, vertical pass then horizontal, each capped at centre + thY
    c = _expr(c, f'x[0,-1] x[0,1] max x max x {th} + min')
    return _expr(c, f'x[-1,0] x[1,0] max x max x {th} + min')


def _mtballoon_gr(c: vs.VideoNode, mode: int) -> vs.VideoNode:
    if mode == 0:
        return c
    th, n = {1: (170, 3), 2: (180, 8), 3: (190, 16)}[mode]
    return _inflate(_expand_thr(c, th), n)


def _hysteresis4(seed: vs.VideoNode, big: vs.VideoNode) -> vs.VideoNode:
    # mt_hysteresis(corners=false): seed pixels grown inside big with 4-neighbour connectivity
    def grow(n: int, f: List[vs.VideoFrame]) -> vs.VideoFrame:
        bm = numpy.asarray(f[1][0]) > 0
        g = (numpy.asarray(f[0][0]) > 0) & bm
        while True:
            h = g.copy()
            h[1:, :] |= g[:-1, :]
            h[:-1, :] |= g[1:, :]
            h[:, 1:] |= g[:, :-1]
            h[:, :-1] |= g[:, 1:]
            h &= bm
            if numpy.array_equal(h, g):
                break
            g = h
        out = f[0].copy()
        numpy.copyto(numpy.asarray(out[0]), numpy.where(g, 255, 0).astype(numpy.uint8))
        return out
    return core.std.ModifyFrame(seed, [seed, big], grow)


def _hysteresis8(seed: vs.VideoNode, big: vs.VideoNode) -> vs.VideoNode:
    _require('hysteresis', 'dynamic masks')
    return core.hysteresis.Hysteresis(seed, big)


def _removegrain2(c: vs.VideoNode) -> vs.VideoNode:
    # RemoveGrain(2) leaves the 1-pixel frame border untouched
    _require('zsmooth', 'dynamic masks')
    return _expr([core.zsmooth.RemoveGrain(c, 2), c], 'X 0 = Y 0 = X width 1 - = Y height 1 - = or or or y x ?')


def _mtballoon_px(c: vs.VideoNode, mode: int) -> vs.VideoNode:
    if mode < -1:
        c = _inpand(_expand(c, 1, True), 1, True)
    if mode == 0:
        return c
    if mode == -1:
        return _expand(c, 1, True)
    if mode in (1, 2, 3, 4, 99):
        px = _expand(c)
        if mode >= 2:
            px = _expand(px, 1, True)
        if mode >= 3:
            px = _expand(px, 1, True)
        if mode == 4:
            return _expand(px)
        if mode == 99:
            return _inpand(_inpand(px), 2, True)
        return px
    k = abs(mode)
    if k == 81:
        return _removegrain2(c)
    if k == 82:
        return _removegrain2(_removegrain2(c))
    if k == 83:
        return _hysteresis4(_inpand(c, 1, True), c)
    kill = _removegrain2(c) if k == 84 else _removegrain2(_removegrain2(c))
    return _hysteresis4(_inpand(kill, 1, True), kill)


def _kill_balloon(c: vs.VideoNode, mode: int, pre: int = 0, h2: int = 0, h3: int = 0) -> vs.VideoNode:
    # KillBalloonFX(): removes blobs smaller (negative mode) or thinner (positive mode) than the given size
    clp = {0: c, 1: _expand(c, 1, True), 2: _expand(c), -1: _inpand(_expand(c, 1, True), 1, True), -2: _inpand(_expand(c))}[pre]
    if mode < 0:
        msk = _hysteresis4(_inpand(clp, -mode), clp)
    elif mode > 0:
        msk = _expand(_inpand(clp, mode), mode)
    else:
        msk = clp
    msk = _expand(msk, h2)
    ooo = _thr(core.std.Expr([clp, msk], 'x y - 128 +'), 128)
    if pre == 1:
        return _inpand(ooo, 1, True)
    if pre == 2:
        return _inpand(ooo)
    return _inpand(ooo, h3)


def _clipblend(c: vs.VideoNode, delay: int) -> vs.VideoNode:
    # ClipBlend(delay): rounded mean of the frames max(0, n-delay)..n
    shifted = [c] + [(c[0] * k + c)[:c.num_frames] for k in range(1, delay + 1)]
    terms = ['src0'] + [f'src{k} N {k} >= *' for k in range(1, delay + 1)]
    expr = ' '.join(terms) + ' +' * delay + f' 2 * N {delay} min 1 + + N {delay} min 1 + 2 * / floor'
    return core.akarin.Expr(shifted, expr)


def _shift(c: vs.VideoNode, k: int) -> vs.VideoNode:
    # Trim(k, 999999) seen from the original length: the last frame repeats
    if k == 0:
        return c
    return (c[k:] + c[-1] * k) if k < c.num_frames else c[-1] * c.num_frames


def _trimergage(c: vs.VideoNode, seq: int, clpblend: int = 0, left: int = 0) -> vs.VideoNode:
    # TrimergageFX(): ORs the mask with itself shifted forward, recursively, so a moved logo keeps its mask
    if left == 0:
        trimy = clpblend if seq // 2 >= clpblend else seq - clpblend
        left = clpblend - trimy
    else:
        trimy = seq if seq <= left else left
        left -= trimy
    merged = core.std.Expr([c, _shift(c, trimy)], 'x y + 1 + 2 / 20 > 255 0 ?')
    return _trimergage(merged, trimy * 2, 0, left) if left > 0 else merged


def _repass(c: vs.VideoNode, seq: int, clpblend: int, killnoise: int = 0) -> vs.VideoNode:
    re = _trimergage(_thr(_clipblend(c, clpblend), 254), seq, clpblend)
    return _mtballoon_px(re, killnoise) if killnoise else re


def _patch(c: vs.VideoNode, path: str) -> vs.VideoNode:
    # PatchFuncFX(): paints the white parts of an image into the mask (mt_merge with a white clip)
    _require('bs', 'maskPatch')
    if not os.path.isfile(path):
        raise _error('"maskPatch" file not found!')
    img = core.bs.VideoSource(path, cachemode=0)[0]
    if (img.width, img.height) != (c.width, c.height):
        raise _error('"maskPatch" image must have the size of the dynamic mask.')
    planes = [img] if img.format.color_family == vs.GRAY else [_plane(core.resize.Point(img, format=vs.RGB24, matrix_in_s='170m'), p) for p in range(3)]
    planes = [_thr(core.resize.Point(p, format=vs.GRAY8), 127) for p in planes]
    patch = planes[0] if len(planes) == 1 else _expr(planes, 'x 6966 * y 23436 * + z 2366 * + 16384 + 32768 / floor')
    return _expr([c, patch * c.num_frames], 'y y 128 / floor + M! x 256 M@ - * 255 M@ * + 128 + 256 / floor')


_DYN_DEFAULTS: Dict[str, Any] = {
    'DynMask': 0, 'DynTune': 200, 'DynTuneInf': 0, 'DynColor': 0, 'DynColorTol': 10, 'DynColorInf': 1, 'DynTEdge': 0,
    'DynTEdgeThY': 20, 'DynTEdgeInf': 2, 'DynInflate': 0, 'KillNoise': 0, 'KillBlobs': 0, 'preBlobs': 0, 'RePass': 0,
    'maskPatch': '', 'Dyn3Seq': 12, 'ClpBlend': None, 'DynPostTune': None, 'Dyn3buffer': 0, 'DynMask3H': 0, 'ModeSH': 1,
    'TuneH1': 0, 'TuneH2': 0, 'TuneH3': 0, 'KillNoiseH': 0, 'rePassH': 0, 'DynMask4H': 60, 'DynMaskUp': 0, 'DynLocUp': 1,
}
_DYN_RANGES: Dict[str, Tuple[int, int]] = {
    'DynMask': (0, 7), 'DynTuneInf': (-1, 4), 'DynColorTol': (1, 140), 'DynColorInf': (0, 4), 'DynTEdge': (0, 1),
    'DynTEdgeThY': (2, 100), 'DynTEdgeInf': (0, 8), 'DynInflate': (0, 10), 'KillNoise': (-5, 5), 'KillBlobs': (-8, 8),
    'preBlobs': (-2, 2), 'RePass': (0, 2), 'Dyn3Seq': (8, 9999), 'Dyn3buffer': (0, 1), 'DynMask3H': (0, 200), 'ModeSH': (0, 1),
    'TuneH1': (0, 7), 'TuneH2': (0, 4), 'TuneH3': (0, 3), 'KillNoiseH': (0, 2), 'rePassH': (0, 1), 'DynMask4H': (1, 200),
    'DynMaskUp': (-1, 1), 'DynLocUp': (1, 1),
}


def _dyn_settings(d: Dict[str, Any], mode: str, mask: Any) -> Dict[str, Any]:
    if isinstance(mask, vs.VideoNode) and mask.num_frames > 1:
        d['DynMask'] = 1
    if d['ClpBlend'] is None:
        d['ClpBlend'] = d['Dyn3Seq'] - 2 if d['Dyn3Seq'] < 54 else 52
    if d['DynPostTune'] is None:
        d['DynPostTune'] = 0 if d['ClpBlend'] == 0 else 1
    for name, (lo, hi) in _DYN_RANGES.items():
        if d[name] < lo or d[name] > hi:
            if name in ('DynMaskUp', 'DynLocUp'):
                raise _error(f'"{name}" is only for the subtitle extraction, which is not supported.')
            raise _error(f'"{name}" value must be between {lo} and {hi}.')
    if d['DynPostTune'] < 0 or d['DynPostTune'] > 5:
        raise _error('"DynPostTune" value must be between 0 and 5.')
    if d['ClpBlend'] != 0 and (d['ClpBlend'] < 6 or d['ClpBlend'] >= d['Dyn3Seq'] - 1):
        raise _error('"ClpBlend" is out of range.')
    dyn = d['DynMask'] if mode != 'deblend' else 0
    d['DynMask'] = dyn
    if dyn == 1 and mode != 'inpaint':
        raise _error('"mask" as a dynamic mask works only in "Inpaint" mode.')
    if dyn == 1 and not (isinstance(mask, vs.VideoNode) and mask.num_frames > 1):
        raise _error('"mask" is not a dynamic mask.')
    if dyn == 0 and isinstance(mask, vs.VideoNode) and mask.num_frames > 1:
        raise _error('"mask" is a dynamic mask when expected a static mask.')
    if dyn == 3 and (d['ClpBlend'] == 0 or d['DynPostTune'] == 0):
        raise _error('There is nothing to do with these settings, fine tune this mask in "Show=3".')
    if d['RePass'] and dyn != 3:
        raise _error('"RePass" parameter is only for "DynMask=3" method.')
    if d['DynMask3H'] and dyn != 3:
        raise _error('"DynMask3H" parameter is only for "DynMask=3" method.')
    tune = d['DynTune']
    if isinstance(tune, str):
        if ' - ' not in tune:
            raise _error('Wrong "DynTune" value!')
        lo, hi = int(float(tune[:3])), int(float(tune[-3:]))
        if lo >= hi or not (1 <= lo <= 254 and 1 <= hi <= 254):
            raise _error('"DynTune" values must be between 10 and 254.')
        d['LumaRange'] = (lo, hi)
        d['DynTune'] = lo
    elif not 1 <= tune <= 254:
        raise _error('"DynTune" value must be between 10 and 254.')
    color = d['DynColor']
    if isinstance(color, str):
        if '$' not in color:
            raise _error('Wrong "DynColor" value!')
    elif color and not 0 < color <= 0xFFFFFF:
        raise _error('"DynColor" value must be between $000000 and $FFFFFF.')
    if color and dyn < 2:
        raise _error('"DynColor" can be used only with internal "DynMask" methods.')
    if dyn == 5 and not color:
        raise _error('Nothing to do for "DynMask=5" if "DynColor=0".')
    if d['DynTEdge'] and dyn < 2:
        raise _error('"DynTEdge" can be used only with internal "DynMask" methods.')
    if dyn == 7 and not d['DynTEdge']:
        raise _error('Nothing to do for "DynMask=7" if "DynTEdge=0".')
    # the original sets DynInflate to 0 for 5/6/7; this port lets it grow the colour mask (5), e.g. over the dark edge of a coloured title
    if dyn in (6, 7):
        d['DynInflate'] = 0
    kn = d['KillNoise']
    d['KillNoise'] = (80 + abs(kn)) * (1 if kn > 0 else -1) if kn else 0
    d['KillNoiseH'] = 80 + d['KillNoiseH'] if d['KillNoiseH'] else 0
    return d


def _color_mask(inyv12: vs.VideoNode, d: Dict[str, Any], is_hd: bool) -> vs.VideoNode:
    color = d['DynColor']
    tol = d['DynColorTol']
    if isinstance(color, str):
        cols = [int(t.strip().lstrip('$'), 16) for t in color.replace('/*', ' ').split() if t.strip().startswith('$')]
        yuv = core.resize.Bicubic(inyv12, format=vs.YUV444P8)
        f = numpy.float32
        kr, kb = (f(0.2126), f(0.0722)) if is_hd else (f(0.299), f(0.114))
        conds_y, conds_u, conds_v = [], [], []
        for col in cols:
            r, g, b = (f((col >> sh) & 0xFF) / f(255) for sh in (16, 8, 0))
            yy = kr * r + (f(1) - kr - kb) * g + kb * b
            conds_y.append(f'x {16 + int(f(219) * yy + f(0.5))} - abs {tol} <')
            conds_u.append(f'y {128 + int(f(112) * (b - yy) / (f(1) - kb) + f(0.5))} - abs {tol // 2} <')
            conds_v.append(f'z {128 + int(f(112) * (r - yy) / (f(1) - kr) + f(0.5))} - abs {tol // 2} <')

        def anyof(parts: List[str]) -> str:
            return ' '.join(parts) + ' or' * (len(parts) - 1)
        m = _expr([_plane(yuv, p) for p in range(3)], f'{anyof(conds_y)} {anyof(conds_u)} and {anyof(conds_v)} and 255 0 ?')
    else:
        rgb = core.resize.Bicubic(inyv12, format=vs.RGB24, matrix_in_s='709' if is_hd else '170m')
        r, g, b = (color >> 16) & 0xFF, (color >> 8) & 0xFF, color & 0xFF
        m = _expr([_plane(rgb, p) for p in range(3)], f'x {r} - abs {tol} <= y {g} - abs {tol} <= and z {b} - abs {tol} <= and 255 0 ?')
    return _mtballoon2px(m, d['DynColorInf'])


def _dynamic_mask(inyv12: vs.VideoNode, mask: Any, mode: str, d: Dict[str, Any], is_hd: bool,
                  logo_deblend: Optional[vs.VideoNode]) -> vs.VideoNode:
    # DynamicMask of the original, without the upscaling variants of the subtitle extraction
    dyn = d['DynMask']
    iny8 = _plane(inyv12, 0)
    if dyn == 1:
        m = mask if mask.format.color_family == vs.GRAY else _plane(mask, 0)
        if (m.width, m.height) != (inyv12.width, inyv12.height):
            raise _error('Clip and "mask" image must have same dimensions.')
        dm = core.resize.Point(m, format=vs.GRAY8)
    elif dyn == 2 and mode == 'both':
        assert logo_deblend is not None
        dm = _thr(_plane(logo_deblend, 0), d['DynTune'])
    else:
        lvl = col = tedge = None
        if dyn in (2, 3, 4, 6):
            if 'LumaRange' in d:
                lo, hi = d['LumaRange']
                lvl = _mtballoon_px(core.std.Expr(iny8, f'x {lo} > x {hi} < and 255 0 ?'), d['DynTuneInf'])
            else:
                lvl = _mtballoon_px(_thr(iny8, d['DynTune']), d['DynTuneInf'])
        if dyn in (2, 3, 4, 5) and d['DynColor']:
            col = _color_mask(inyv12, d, is_hd)
        if dyn in (2, 3, 4, 7) and d['DynTEdge']:
            t = int(round(d['DynTEdgeThY'] ** 2 * 4))
            tedge = _expr(iny8, f'X 0 = Y 0 = X width 1 - = Y height 1 - = or or or 0 x[1,0] x[-1,0] - dup * x[0,-1] x[0,1] - dup * + {t} > 255 0 ? ?')
            if d['DynTEdgeInf']:
                tedge = _inpand(_expand(_mtballoon2px(tedge, d['DynTEdgeInf'])))
        if dyn == 5:
            assert col is not None
            dm = col
        elif dyn == 6:
            assert lvl is not None
            dm = lvl
        elif dyn == 7:
            assert tedge is not None
            dm = tedge
        elif dyn == 4:
            white = _thr(iny8, d['DynTune'])
            dark = _thr(iny8, d['DynMask4H'])
            frame = numpy.full((inyv12.height, inyv12.width), 255, numpy.uint8)
            frame[8:-8, 8:-8] = 0
            border = _hysteresis4(_frame_clip(frame, inyv12.num_frames), dark)
            dm = _thr(core.std.Expr([white, border], 'x y - 128 +'), 128)
        else:
            assert lvl is not None
            combo = lvl
            for extra in (col, tedge):
                if extra is not None:
                    combo = core.std.Expr([combo, extra], 'x y min')
            if dyn == 2:
                dm = combo
            else:
                m3 = _thr(_clipblend(combo, d['ClpBlend']), 255 - d['DynPostTune'])
                dm = _trimergage(m3, d['Dyn3Seq'], d['ClpBlend'])
    if d['KillBlobs']:
        dm = _kill_balloon(dm, d['KillBlobs'], d['preBlobs'])
    if d['KillNoise']:
        dm = _mtballoon_px(dm, d['KillNoise'])
    if dyn == 3:
        for _ in range(d['RePass']):
            dm = _repass(dm, d['Dyn3Seq'], d['ClpBlend'], d['KillNoise'])
        if d['DynMask3H']:
            if d['ModeSH'] == 1:
                dark = core.std.Invert(_thr(iny8, d['DynMask3H']))
                if d['TuneH1'] == 0:
                    halo = dark
                else:
                    halo = core.std.Expr([dark, _mtballoon2px(_thr(iny8, d['DynTune']), d['TuneH1'])], 'x y min 255 = 0 255 ?')
            else:
                halo = _trimergage(_thr(_clipblend(_thr(iny8, d['DynMask3H']), d['ClpBlend']), 254), d['Dyn3Seq'], d['ClpBlend'])
            if d['TuneH1']:
                halo = _kill_balloon(halo, d['TuneH1'], 0, d['TuneH2'], d['TuneH3'])
            if d['KillNoiseH']:
                halo = _mtballoon_px(halo, d['KillNoiseH'])
            if d['rePassH']:
                halo = _repass(halo, d['Dyn3Seq'], d['ClpBlend'], d['KillNoiseH'])
            dm = _repass(_hysteresis8(halo, dm), d['Dyn3Seq'], d['ClpBlend'])
    if d['KillShape']:
        dm = _mtballoon_px(dm, 99)
    if d['DynInflate']:
        dm = _mtballoon2px(dm, d['DynInflate'])
    if d['maskPatch']:
        dm = _patch(dm, d['maskPatch'])
    return dm
