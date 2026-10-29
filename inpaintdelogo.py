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
_UNSUPPORTED: Dict[str, Any] = {'DynMask': 0, 'Show': 0, 'Extract': 0}


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
            raise _error('No frames found to analyze, increase "AnalyzeTh".')
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
    numpy.savez(npz, **arrays)
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
                  TriggerDynamic: int = 0, DumpMasks: int = 0, LogoPath: str = '', **kwargs: Any) -> vs.VideoNode:
    '''
    Logo removal by inpainting and deblending (InpaintDelogo 3.7).
    clp: YUV 8..16 bit; only the "Loc" area is processed in YUV420P8, pixels the delogo does not change keep the source values.
    mask: path to a bmp/png mask of the whole frame (logo 255, rest < 255) or a one-frame GRAY clip of the "Loc" size.
    Loc: "Left,Top,-Right,-Bottom" or "Left,Top,Width,Height", even numbers, 10-20 px border around the logo.
    LogoPath: base path of the analysis files (.txt frame list, .npz masks); default: the original's name next to the mask.
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
        raise _error('You must define "Loc".')
    if Loc.replace(' ', '') == '100,100,-100,-100':
        raise _error('Use "InpaintLoc" function to get crop values for "Loc".')
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
              ('aMix', aMix, -50, 6), ('Inflate', Inflate, 0, 2), ('KillShape', KillShape, 0, 1), ('Turbo', Turbo, -2, 3),
              ('prePP', prePP, 0, 1), ('Interp', Interp, 0, 4), ('IntSpd', IntSpd, 1, 4), ('InterpM', InterpM, -1, 2),
              ('iTune', iTune, -10, 10), ('iTinflate', iTinflate, 0, 3), ('oTune', oTune, -10, 10), ('oTinflate', oTinflate, 0, 3),
              ('dPP', dPP, -8, 8), ('dPPm', dPPm, 1, 3), ('diPP', diPP, -8, 8), ('diPPm', diPPm, 0, 3), ('oPP', oPP, -8, 8),
              ('oPPm', oPPm, 0, 3), ('EdgePP', EdgePP, 0, 3), ('EdgePos', EdgePos, -1, 1), ('EdgeWide', EdgeWide, 0, 3),
              ('GrainPP', GrainPP, 0, 3), ('mShow', mShow, 0, 9), ('Optimize2', Optimize2, 0, 3), ('Greyscale', Greyscale, 0, 2),
              ('WholeDeblend', WholeDeblend, 0, 1), ('TriggerDynamic', TriggerDynamic, 0, 1), ('DumpMasks', DumpMasks, 0, 2)]
    checks += [(k, opts[k], lo, hi) for k, (lo, hi) in _EDIT_RANGES.items()]
    checks += [(k, opts[k], -1, 255) for k in _EDIT_DEFAULTS if 'Solid' in k]
    for name, value, lo, hi in checks:
        if value < lo or value > hi:
            raise _error(f'"{name}" value must be between {lo} and {hi}.')
    if NoBorderAt not in ('top', 'bottom', 'left', 'right', 'none'):
        raise _error('"NoBorderAt" string must be \'top\', \'bottom\', \'left\', \'right\' or \'none\'.')
    if mode == 'inpaint' and 4 < mShow < 9:
        raise _error('Calculating the meaning of life and everything. Meanwhile RTM.')
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

    if mShow:
        red = core.resize.Point(core.std.BlankClip(out, format=vs.RGB24, color=[255, 0, 0]), format=vs.YUV420P8, matrix_s='170m')
        masks = {1: opaque, 2: inpaint_mpp, 3: basemask, 4: basemask, 5: deblend_mpp, 6: opaque_tune, 7: interp_mask, 8: edge_mask, 9: inpaint_mpp}
        show_mask = masks[mShow]
        assert show_mask is not None
        out = _overlay(inyv12 if mShow in (4, 5, 6, 9) else out, red, show_mask)

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
    if dyn in (5, 6, 7):
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
