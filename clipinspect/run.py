'''Frames as PNG and per-frame metrics of a VapourSynth script: python run.py <script.vpy> <request.json>.

Every message is one stdout line "INSPECT <json>" with a 'type': loaded, progress, scan, frame, done, failed.
Only the standard library and the VapourSynth core are used; metrics are computed on the luma plane as GRAYS (0..1).
'''
import json
import math
import os
import runpy
import statistics
import struct
import sys
import time
import zlib
from typing import Any, Callable, Dict, List, Optional, Tuple

import vapoursynth as vs

core = vs.core

PREFIX = 'INSPECT '
SCAN_WIDTH = 320
PROGRESS_INTERVAL = 0.5
_last_progress = 0.0


class InspectError(Exception):
    pass


def emit(message: Dict[str, Any]) -> None:
    print(PREFIX + json.dumps(message), flush=True)


def progress(phase: str, done: int, total: int, force: bool = False) -> None:
    global _last_progress
    now = time.perf_counter()
    if not force and now - _last_progress < PROGRESS_INTERVAL:
        return
    _last_progress = now
    emit({'type': 'progress', 'phase': phase, 'done': done, 'total': total})


def last_line(error: BaseException) -> str:
    text = str(error).strip()
    return text.splitlines()[-1] if text else type(error).__name__


def number(value: float) -> float:
    return round(float(value), 6)


def load(script: str, output_map: Dict[str, int], wanted: List[str]) -> Dict[str, vs.VideoNode]:
    runpy.run_path(script, run_name='__vapoursynth__')
    outputs = vs.get_outputs()
    nodes: Dict[str, vs.VideoNode] = {}
    for name in wanted:
        if name not in output_map:
            raise InspectError("unknown output '%s'" % name)
        index = output_map[name]
        if index not in outputs:
            raise InspectError("the script sets no output %d ('%s')" % (index, name))
        node = getattr(outputs[index], 'clip', outputs[index])
        if not isinstance(node, vs.VideoNode):
            raise InspectError("output %d ('%s') is no video clip" % (index, name))
        if node.format is None or node.width == 0 or node.height == 0:
            raise InspectError("output %d ('%s') has a variable format or size" % (index, name))
        nodes[name] = node
    return nodes


def clip_info(node: vs.VideoNode, index: int) -> Dict[str, Any]:
    props = node.get_frame(0).props
    family = node.format.color_family
    matrix = props.get('_Matrix')
    guessed = False
    if family == vs.YUV and (matrix is None or matrix == 2):
        matrix = 1 if node.height > 576 else 6
        guessed = True
    info: Dict[str, Any] = {
        'index': index, 'width': node.width, 'height': node.height, 'format': node.format.name,
        'frames': node.num_frames, 'fpsNum': node.fps.numerator, 'fpsDen': node.fps.denominator,
        'matrix': matrix if family == vs.YUV else props.get('_Matrix'), 'transfer': props.get('_Transfer'),
        'range': range_name(props, family), 'matrixGuessed': guessed,
        'sarNum': props.get('_SARNum', 1), 'sarDen': props.get('_SARDen', 1)}
    return info


def range_name(props: Any, family: int) -> str:
    '''full or limited from _Range (vs.RANGE_FULL/RANGE_LIMITED), without it the default of resize.'''
    if '_Range' in props:
        return 'full' if int(props['_Range']) == int(vs.RANGE_FULL) else 'limited'
    return 'full' if family == vs.RGB else 'limited'


def float_clip(node: vs.VideoNode, info: Dict[str, Any]) -> vs.VideoNode:
    '''The clip in 32 bit float, same family and subsampling (not for RGB).'''
    fmt = node.format.replace(sample_type=vs.FLOAT, bits_per_sample=32)
    if fmt.id == node.format.id:
        return node
    # resize reads the range per frame from the frame properties
    return core.resize.Point(node, format=fmt.id)


def luma(node: vs.VideoNode, info: Dict[str, Any]) -> vs.VideoNode:
    family = node.format.color_family
    if family == vs.RGB:
        return core.resize.Bicubic(node, format=vs.GRAYS, matrix_s='709')
    return core.std.ShufflePlanes(float_clip(node, info), planes=0, colorfamily=vs.GRAY)


def next_clip(y: vs.VideoNode) -> vs.VideoNode:
    '''Frame n holds frame n + 1, the last frame itself.'''
    if y.num_frames < 2:
        return y
    return y[1:] + y[-1]


def frame_props(nodes: List[vs.VideoNode], n: int) -> List[Any]:
    futures = [node.get_frame_async(n) for node in nodes]
    return [future.result().props for future in futures]


def target_size(info: Dict[str, Any], crop: List[int], square: bool, max_width: int) -> Tuple[float, float, int, int, int, int]:
    '''src_left, src_top, src_width, src_height and the output width and height of one picture.'''
    if crop:
        left, top, width, height = crop
    else:
        left, top, width, height = 0, 0, info['width'], info['height']
    out_width = float(width)
    out_height = float(height)
    if square and info['sarNum'] > 0 and info['sarDen'] > 0 and info['sarNum'] != info['sarDen']:
        out_width = out_width * info['sarNum'] / info['sarDen']
    if max_width > 0 and out_width > max_width:
        out_height = out_height * max_width / out_width
        out_width = float(max_width)
    return left, top, width, height, max(1, int(round(out_width))), max(1, int(round(out_height)))


def scaled_crop(crop: List[int], reference: Dict[str, Any], info: Dict[str, Any]) -> List[int]:
    '''A crop in pixels of 'reference' on a clip of another size.'''
    if not crop:
        return crop
    fx = info['width'] / reference['width']
    fy = info['height'] / reference['height']
    left = int(round(crop[0] * fx))
    top = int(round(crop[1] * fy))
    width = max(1, min(info['width'] - left, int(round(crop[2] * fx))))
    height = max(1, min(info['height'] - top, int(round(crop[3] * fy))))
    return [left, top, width, height]


def to_rgb(node: vs.VideoNode, info: Dict[str, Any], crop: List[int], square: bool, max_width: int) -> vs.VideoNode:
    left, top, width, height, out_width, out_height = target_size(info, crop, square, max_width)
    args: Dict[str, Any] = {'width': out_width, 'height': out_height, 'format': vs.RGB24, 'dither_type': 'error_diffusion',
                            'src_left': left, 'src_top': top, 'src_width': width, 'src_height': height}
    if node.format.color_family == vs.YUV:
        args['matrix_in'] = info['matrix']
    return core.resize.Bicubic(node, **args)


def to_gray8(y: vs.VideoNode, info: Dict[str, Any], crop: List[int], square: bool, max_width: int) -> vs.VideoNode:
    left, top, width, height, out_width, out_height = target_size(info, crop, square, max_width)
    return core.resize.Bicubic(y, width=out_width, height=out_height, format=vs.GRAY8, range_s='full', src_left=left,
                               src_top=top, src_width=width, src_height=height)


def png_chunk(kind: bytes, data: bytes) -> bytes:
    return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data) & 0xffffffff)


def write_png(path: str, frame: vs.VideoFrame) -> None:
    '''8 bit RGB (color type 2) or 8 bit gray (color type 0), filter type 0, zlib level 3.'''
    width = frame.width
    height = frame.height
    planes = frame.format.num_planes
    if planes == 3:
        pixels = bytearray(width * height * 3)
        for plane in range(3):
            pixels[plane::3] = frame[plane].tobytes()
        stride = width * 3
        color_type = 2
    else:
        pixels = bytearray(frame[0].tobytes())
        stride = width
        color_type = 0
    raw = b''.join(b'\x00' + bytes(pixels[row * stride:(row + 1) * stride]) for row in range(height))
    header = struct.pack('>IIBBBBB', width, height, 8, color_type, 0, 0, 0)
    with open(path, 'wb') as handle:
        handle.write(b'\x89PNG\r\n\x1a\n')
        handle.write(png_chunk(b'IHDR', header))
        handle.write(png_chunk(b'IDAT', zlib.compress(raw, 3)))
        handle.write(png_chunk(b'IEND', b''))


def map_frame(n: int, source: Dict[str, Any], target: Dict[str, Any]) -> int:
    '''Frame number of 'target' at the time of frame n of 'source'.'''
    if source['fpsNum'] > 0 and target['fpsNum'] > 0:
        seconds = n * source['fpsDen'] / source['fpsNum']
        mapped = int(round(seconds * target['fpsNum'] / target['fpsDen']))
    else:
        mapped = int(round(n * target['frames'] / max(1, source['frames'])))
    return max(0, min(target['frames'] - 1, mapped))


def same_timing(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    return a['frames'] == b['frames'] and a['fpsNum'] * b['fpsDen'] == b['fpsNum'] * a['fpsDen']


class Clip:
    '''One output with its luma and the nodes the metrics read.'''

    def __init__(self, name: str, node: vs.VideoNode, info: Dict[str, Any]) -> None:
        self.name = name
        self.node = node
        self.info = info
        self.y = luma(node, info)
        self.stats = core.std.PlaneStats(self.y, next_clip(self.y), prop='S')
        self.chroma: Optional[vs.VideoNode] = None
        if node.format.color_family == vs.YUV:
            self.chroma = float_clip(node, info)


def scan(clip: Clip, filtered: Dict[str, Any], options: Dict[str, Any], first: int, last: int, count: int) -> Dict[str, Any]:
    '''Average and change to the next frame on a 320 pixel wide luma; suggests frames of 'filtered' (§4.5 of the plan).'''
    info = clip.info
    mapped = clip.name != 'filtered' and not same_timing(info, filtered)
    scan_first = map_frame(first, filtered, info) if mapped else first
    scan_last = map_frame(last, filtered, info) if mapped else last
    y = clip.y
    if y.width > SCAN_WIDTH:
        height = max(2, int(round(y.height * SCAN_WIDTH / y.width / 2)) * 2)
        y = core.resize.Bilinear(y, width=SCAN_WIDTH, height=height)
    stats = core.std.PlaneStats(y, next_clip(y), prop='S')
    length = scan_last - scan_first + 1
    step = int(options.get('step', 0) or 0)
    if step <= 0:
        step = max(1, int(math.ceil(length / max(1, int(options.get('maxSamples', 2000))))))
    positions = list(range(scan_first, scan_last + 1, step))
    deadline = time.perf_counter() + float(options.get('scanSeconds', 120))
    depth = max(1, core.num_threads)
    values: List[Tuple[int, float, float]] = []
    pending: List[Tuple[int, Any]] = []
    truncated = False
    index = 0
    while index < len(positions) or pending:
        while index < len(positions) and len(pending) < depth:
            pending.append((positions[index], stats.get_frame_async(positions[index])))
            index += 1
        position, future = pending.pop(0)
        props = future.result().props
        values.append((position, float(props['SAverage']), float(props['SDiff'])))
        progress('scanning', len(values), len(positions))
        if time.perf_counter() > deadline:
            truncated = True
            for _, rest in pending:
                try:
                    rest.result()
                except Exception:
                    pass
            break
    progress('scanning', len(values), len(positions), True)

    def back(n: int) -> int:
        return map_frame(n, info, filtered) if mapped else n

    result: Dict[str, Any] = {'output': clip.name, 'step': step, 'samples': len(values), 'truncated': truncated}
    if not values:
        result.update({'avg': None, 'scenes': [first], 'suggested': [], 'darkest': None, 'brightest': None, 'busiest': None})
        return result
    averages = [value[1] for value in values]
    changes = [value[2] for value in values]
    threshold = max(0.06, 4 * statistics.median(changes))
    scenes = sorted({first} | {back(min(scan_last, position + 1)) for position, _, change in values if change > threshold})
    bounds = scenes + [last + 1]
    segments = sorted(((bounds[i + 1] - bounds[i], bounds[i]) for i in range(len(scenes))), key=lambda s: (-s[0], s[1]))
    suggested: List[int] = []
    for size, start in segments:
        if len(suggested) >= count:
            break
        middle = start + (size - 1) // 2
        if first <= middle <= last and middle not in suggested:
            suggested.append(middle)
    total = last - first + 1
    slot = 0
    while len(suggested) < min(count, total) and slot < count * 4:
        candidate = first + int((slot % count + 0.5) * total / count) + slot // count
        slot += 1
        if candidate <= last and candidate not in suggested:
            suggested.append(candidate)
    quiet = [value for value in values if value[2] <= threshold]
    result.update({
        'avg': {'min': number(min(averages)), 'max': number(max(averages)), 'mean': number(statistics.fmean(averages))},
        'scenes': scenes, 'suggested': sorted(suggested),
        'darkest': back(min(values, key=lambda v: v[1])[0]), 'brightest': back(max(values, key=lambda v: v[1])[0]),
        'busiest': back(max(quiet, key=lambda v: v[2])[0]) if quiet else None})
    return result


def measure(clip: Clip, n: int) -> Dict[str, Any]:
    nodes = [clip.stats,
             core.std.PlaneStats(core.std.Sobel(clip.y[n]), prop='E'),
             core.std.PlaneStats(clip.y[n], core.std.Convolution(clip.y[n], [1, 2, 1, 2, 4, 2, 1, 2, 1]), prop='H')]
    futures = [clip.stats.get_frame_async(n), nodes[1].get_frame_async(0), nodes[2].get_frame_async(0)]
    chroma: List[Any] = []
    if clip.chroma is not None:
        single = clip.chroma[n]
        chroma = [core.std.PlaneStats(single, plane=1, prop='U').get_frame_async(0),
                  core.std.PlaneStats(single, plane=2, prop='V').get_frame_async(0)]
    stats, edge, high = [future.result().props for future in futures]
    entry: Dict[str, Any] = {'avg': number(stats['SAverage']), 'min': number(stats['SMin']), 'max': number(stats['SMax'])}
    if chroma:
        entry['avgU'] = number(chroma[0].result().props['UAverage'] + 0.5)
        entry['avgV'] = number(chroma[1].result().props['VAverage'] + 0.5)
    entry['edge'] = number(edge['EAverage'])
    entry['high'] = number(high['HDiff'])
    entry['next'] = number(stats['SDiff']) if n + 1 < clip.node.num_frames else 0.0
    return entry


def compare(filtered: Clip, original: Clip, n: int, m: int) -> Dict[str, Any]:
    if filtered.y.width != original.y.width or filtered.y.height != original.y.height:
        return {'ref': None, 'psnr': None, 'refReason': 'original differs in size'}
    a = filtered.y[n]
    b = original.y[m]
    ref = core.std.PlaneStats(a, b, prop='R').get_frame_async(0)
    mse = core.std.PlaneStats(core.std.Expr([a, b], 'x y - dup *'), prop='M').get_frame_async(0)
    error = float(mse.result().props['MAverage'])
    entry: Dict[str, Any] = {'ref': number(ref.result().props['RDiff'])}
    if error <= 0.0:
        entry['psnr'] = None
        entry['identical'] = True
    else:
        entry['psnr'] = round(10 * math.log10(1.0 / error), 3)
    return entry


def window_values(clip: Clip, center: int, size: int) -> Dict[str, List[float]]:
    last = clip.node.num_frames - 1
    frames = list(range(max(0, center - size), min(last, center + size) + 1))
    futures = [clip.stats.get_frame_async(n) for n in frames]
    props = [future.result().props for future in futures]
    return {'frames': frames, 'avg': [number(p['SAverage']) for p in props],
            'next': [number(p['SDiff']) if n < last else 0.0 for n, p in zip(frames, props)]}


def render(clips: Dict[str, Clip], request: Dict[str, Any], frames: List[int], directory: str) -> None:
    filtered = clips['filtered']
    names = [name for name in request['outputs'] if name in clips]
    crop: List[int] = list(request.get('crop') or [])
    square = bool(request.get('square', True))
    max_width = int(request.get('maxWidth', 0) or 0)
    want_images = bool(request.get('images', True))
    want_metrics = bool(request.get('metrics', True))
    window = int(request.get('window', 0) or 0)
    rgb: Dict[str, vs.VideoNode] = {}
    if want_images:
        for name in names:
            clip = clips[name]
            rgb[name] = to_rgb(clip.node, clip.info, scaled_crop(crop, filtered.info, clip.info), square, max_width)
    diff: Optional[Callable[[int, int], vs.VideoNode]] = None
    if want_images and request.get('diff') and 'original' in clips and 'original' in names:
        original = clips['original']
        if filtered.y.width == original.y.width and filtered.y.height == original.y.height:
            gain = float(request.get('diffGain', 8))

            def diff_node(n: int, m: int) -> vs.VideoNode:
                picture = core.std.Expr([filtered.y[n], original.y[m]], 'x y - abs %s * 0 max 1 min' % gain)
                return to_gray8(picture, filtered.info, crop, square, max_width)
            diff = diff_node

    def original_frame(n: int) -> int:
        info = clips['original'].info
        return n if same_timing(info, filtered.info) else map_frame(n, filtered.info, info)

    for done, n in enumerate(frames):
        progress('rendering', done, len(frames))
        entry: Dict[str, Any] = {'frame': n}
        if filtered.info['fpsNum'] > 0:
            entry['time'] = round(n * filtered.info['fpsDen'] / filtered.info['fpsNum'], 6)
        numbers: Dict[str, int] = {}
        for name in names:
            clip = clips[name]
            mapped = name != 'filtered' and (name == 'source' or not same_timing(clip.info, filtered.info))
            m = map_frame(n, filtered.info, clip.info) if mapped else n
            numbers[name] = m
            values: Dict[str, Any] = {}
            if mapped:
                values['sourceFrame' if name == 'source' else 'mappedFrame'] = m
            if want_images:
                path = os.path.join(directory, '%s_%06d.png' % (name, n))
                write_png(path, rgb[name].get_frame(m))
                values['png'] = path
            if want_metrics:
                values.update(measure(clip, m))
                if name == 'filtered':
                    if 'original' in clips and 'original' in names:
                        values.update(compare(filtered, clips['original'], n, original_frame(n)))
                    else:
                        values.update({'ref': None, 'psnr': None, 'refReason': 'original not requested'})
            entry[name] = values
        if diff is not None:
            path = os.path.join(directory, 'diff_%06d.png' % n)
            write_png(path, diff(n, original_frame(n)).get_frame(0))
            entry['diff'] = {'png': path}
        if want_metrics and window > 0:
            block: Dict[str, Any] = {}
            flicker: Dict[str, float] = {}
            for name in names:
                values_window = window_values(clips[name], numbers[name], window)
                block[name] = values_window
                flicker[name] = number(statistics.pstdev(values_window['avg'])) if len(values_window['avg']) > 1 else 0.0
            block['flicker'] = flicker
            if 'filtered' in block and 'original' in block:
                calm = statistics.fmean(block['original']['next'])
                block['motionRatio'] = number(statistics.fmean(block['filtered']['next']) / calm) if calm > 0 else None
            entry['window'] = block
        emit(dict({'type': 'frame'}, **entry))
    progress('rendering', len(frames), len(frames), True)


def main() -> None:
    started = time.perf_counter()
    with open(sys.argv[2], 'r', encoding='utf-8') as handle:
        request: Dict[str, Any] = json.load(handle)
    output_map: Dict[str, int] = {str(k): int(v) for k, v in request.get('outputMap', {'filtered': 0, 'original': 1, 'source': 2}).items()}
    outputs: List[str] = list(request.get('outputs') or ['filtered', 'original'])
    request['outputs'] = outputs
    directory = request.get('dir') or os.path.dirname(os.path.abspath(sys.argv[2]))
    os.makedirs(directory, exist_ok=True)
    scan_options: Dict[str, Any] = dict(request.get('scan') or {})
    scan_only = bool(request['scanOnly']) if 'scanOnly' in request else bool(request.get('scan'))
    auto = not isinstance(request.get('frames', 'auto'), list)
    scan_output = str(scan_options.get('output') or '')
    wanted = ['filtered'] + [name for name in outputs if name != 'filtered']
    if (auto or scan_only) and scan_output and scan_output not in wanted:
        wanted.append(scan_output)
    if (auto or scan_only) and not scan_output:
        scan_output = 'original' if 'original' in outputs else 'filtered'
        if scan_output not in wanted:
            wanted.append(scan_output)
    try:
        nodes = load(sys.argv[1], output_map, wanted)
    except InspectError:
        raise
    except BaseException as error:
        raise InspectError('the script does not load: ' + last_line(error))
    clips: Dict[str, Clip] = {}
    infos: Dict[str, Any] = {}
    for name, node in nodes.items():
        info = clip_info(node, output_map[name])
        clips[name] = Clip(name, node, info)
        infos[name] = dict(info)
    filtered = clips['filtered'].info
    for name, info in infos.items():
        if name == 'source':
            info['sourceMapping'] = 'time'
            if request.get('sourceNote'):
                info['sourceNote'] = request['sourceNote']
        elif name != 'filtered':
            info['mapped'] = not same_timing(clips[name].info, filtered)
    emit({'type': 'loaded', 'clips': infos})
    total = filtered['frames']
    span = request.get('range') or []
    first = int(span[0]) if span else 0
    last = int(span[1]) if span else total - 1
    if span and (first < 0 or last >= total or first > last):
        raise InspectError('range %d..%d is outside the clip (%d frames)' % (first, last, total))
    count = int(request.get('count', 8))
    frames: List[int] = []
    if auto or scan_only:
        result = scan(clips[scan_output], filtered, scan_options, first, last, count)
        emit(dict({'type': 'scan'}, scan=result))
        frames = result['suggested']
    if not scan_only:
        if not auto:
            frames = [int(n) for n in request.get('frames') or []]
            for n in frames:
                if n < 0 or n >= total:
                    raise InspectError('frame %d is outside the clip (%d frames)' % (n, total))
        render(clips, request, frames, directory)
    emit({'type': 'done', 'seconds': round(time.perf_counter() - started, 3)})


if __name__ == '__main__':
    code = 0
    try:
        main()
    except InspectError as error:
        emit({'type': 'failed', 'reason': str(error)})
        code = 2
    except BaseException as error:
        emit({'type': 'failed', 'reason': last_line(error)})
        code = 2
    sys.stdout.flush()
    # frames still computing must not keep the process alive
    os._exit(code)
