'''Frames per second of every output of a VapourSynth script: python measure.py <script> <seconds>.

Per output: warm-up (20 frames or 1 s), then at least <seconds> and two rounds of parallel requests (core.num_threads each),
10 s at most; outputs above 1920x1088 run past that until one round is done. Prints "CLASS <index>" and
"RESULT <index> <fps|reason>" per output, "FAILED <reason>" when the script does not load or a tool fell back.
'''
import collections
import concurrent.futures
import os
import runpy
import sys
import time
import warnings

import vapoursynth as vs

SECONDS = float(sys.argv[2])
WARMUP_SECONDS = 1.0
WARMUP_FRAMES = 20
CAP_SECONDS = 10.0
DRAIN_SECONDS = 10.0

fallbacks = []
_show = warnings.showwarning


def show(message, category, filename, lineno, file=None, line=None):
    if 'tools:' in str(message):
        fallbacks.append(str(message).strip().splitlines()[0])
    _show(message, category, filename, lineno, file, line)


def last_line(error):
    text = str(error).strip()
    return text.splitlines()[-1] if text else type(error).__name__


def rate(frames, seconds):
    return '%.1f' % (frames / seconds) if frames >= 2 and seconds > 0 else None


warnings.showwarning = show
try:
    runpy.run_path(sys.argv[1], run_name='__vapoursynth__')
except Exception as error:
    print('FAILED error: ' + last_line(error), flush=True)
    sys.exit(2)
if fallbacks:
    print('FAILED fallback: ' + fallbacks[0], flush=True)
    sys.exit(0)

depth = max(1, vs.core.num_threads)
MIN_FRAMES = 2 * depth
outputs = vs.get_outputs()
for index in sorted(outputs):
    node = getattr(outputs[index], 'clip', outputs[index])
    # UHD runs past the cap until one full round of parallel requests is done: slow values deliver it as one burst
    least = depth if node.width * node.height > 1920 * 1088 else 0
    print('CLASS %d' % index, flush=True)
    pending = collections.deque()
    requested = 0
    result = None
    try:
        start = time.perf_counter()
        first = mark = None
        done = marked = 0
        while True:
            while len(pending) < depth and requested < node.num_frames:
                pending.append(node.get_frame_async(requested))
                requested += 1
            if not pending:
                result = 'error: the clip ended'
                break
            try:
                left = CAP_SECONDS - (time.perf_counter() - start)
                pending[0].result(timeout=left if left > 0.01 else 1.0)
            except concurrent.futures.TimeoutError:
                pass
            now = time.perf_counter()
            if pending[0].done():
                pending.popleft().result()
                done += 1
                if first is None:
                    first = now
                if mark is None:
                    if done >= WARMUP_FRAMES or now - first >= WARMUP_SECONDS:
                        mark, marked = now, done
                elif now - mark >= SECONDS and done - marked >= MIN_FRAMES:
                    result = rate(done - marked, now - mark)
                    break
            if now - start >= CAP_SECONDS and done >= least:
                # too slow for the full window: the timed part over at least one round, else every frame since the class started
                timed = rate(done - marked, now - mark) if mark is not None and done - marked >= depth else None
                result = timed or rate(done, now - start) or 'timeout'
                break
    except Exception as error:
        result = 'error: ' + last_line(error)
    print('RESULT %d %s' % (index, result), flush=True)
    deadline = time.perf_counter() + DRAIN_SECONDS
    for future in pending:
        try:
            future.result(timeout=max(0.01, deadline - time.perf_counter()))
        except Exception:
            pass
# frames still computing must not keep the process alive
os._exit(0)
