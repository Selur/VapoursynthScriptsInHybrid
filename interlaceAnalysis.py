from __future__ import annotations
import threading
from vapoursynth import core
import vapoursynth as vs

# field matching and bob motion statistics for an interlace check
# analyse() returns the analysed frames unchanged and writes two lines per requested frame to 'log':
#   <frame> <match tff> <combed tff> <drop tff> <match bff> <combed bff> <drop bff>
#   b <frame> <move across> <move inside> <reverse across> <reverse inside> for tff, the same four for bff
# match: VFM VFMMatch (0 = p, 1 = c, 2 = n), combed: VFM _Combed (0/1), drop: VDecimate dry run VDecimateDrop (0/1)
# bob steps: 'across' goes from the second field of the previous frame to the first field of the frame, 'inside' between its fields
# move: per mille of the blocks moving in that step; reverse: per mille of the blocks moving in that step and the next one whose motion turns back
# -1 where the step has no motion vectors, reverse also where fewer than 1% of the blocks move in both steps
# the frame number counts the returned clip, so frame // length is the segment it belongs to

_BLOCK = 16


def _segmentStarts(frames: int, segments: int, length: int) -> list[int]:
  # each segment is centered in its share of the clip
  share = frames / segments
  return [max(0, min(frames - length, int(share * i + (share - length) / 2))) for i in range(segments)]


def _select(clip: vs.VideoNode, starts: list[int], length: int) -> vs.VideoNode:
  if not starts:
    return clip
  parts = [clip[start:start + length] for start in starts]
  return parts[0] if len(parts) == 1 else core.std.Splice(parts)


def _matching(clip: vs.VideoNode, order: int) -> tuple[vs.VideoNode, vs.VideoNode]:
  matched = core.vivtc.VFM(clip, order=order, mode=1)
  return matched, core.vivtc.VDecimate(matched, dryrun=True)


def _moving(x: str, y: str) -> str:
  # a vector component of 128 is no motion
  return '{0} 128 - abs {1} 128 - abs + 2 >='.format(x, y)


def _bobMotion(clip: vs.VideoNode, fieldBased: int) -> list[vs.VideoNode]:
  """Per frame of 'clip': PlaneStats clips of the across and inside bob steps with the given field order."""
  bob = core.resize.Bob(core.std.SetFieldBased(clip, fieldBased))
  if bob.width > 1024:
    bob = core.resize.Bilinear(bob, bob.width // 4 * 2, bob.height // 4 * 2)
  vectors = core.mv.Analyse(core.mv.Super(bob, pel=1), isb=False, delta=1, blksize=_BLOCK)

  def component(kind: int) -> vs.VideoNode:
    mask = core.std.ShufflePlanes(core.mv.Mask(bob, vectors, kind=kind), 0, vs.GRAY)
    return core.resize.Point(mask, max(1, mask.width // _BLOCK), max(1, mask.height // _BLOCK))

  x, y = component(3), component(4)
  acrossX, acrossY = x[0::2], y[0::2]
  insideX, insideY = x[1::2], y[1::2]
  nextX, nextY = acrossX[1:] + acrossX[-1], acrossY[1:] + acrossY[-1]
  moved = _moving('x', 'y') + ' 255 0 ?'
  both = _moving('x', 'y') + ' ' + _moving('z', 'a') + ' and'
  turned = 'x 128 - z 128 - * y 128 - a 128 - * + 0 < ' + both + ' and 255 0 ?'
  both += ' 255 0 ?'

  def share(clips: list[vs.VideoNode], expr: str) -> vs.VideoNode:
    return core.std.PlaneStats(core.std.Expr(clips, expr))

  step = [acrossX, acrossY, insideX, insideY]
  following = [insideX, insideY, nextX, nextY]
  return [share([acrossX, acrossY], moved), share([insideX, insideY], moved),
          share(step, turned), share(step, both), share(following, turned), share(following, both),
          core.std.PlaneStats(acrossX), core.std.PlaneStats(insideX), core.std.PlaneStats(nextX)]


def _bobRow(f: list[vs.VideoFrame]) -> str:
  # f: the nine frames of _bobMotion; a step without vectors comes out of mv.Mask all black
  def average(frame: vs.VideoFrame) -> float:
    return float(frame.props['PlaneStatsAverage'])

  def valid(frame: vs.VideoFrame) -> bool:
    return int(frame.props['PlaneStatsMax']) > 0

  def reverse(turned: vs.VideoFrame, both: vs.VideoFrame, first: vs.VideoFrame, second: vs.VideoFrame) -> int:
    if not valid(first) or not valid(second) or average(both) < 0.01:
      return -1
    return int(1000 * average(turned) / average(both) + 0.5)

  across = int(1000 * average(f[0]) + 0.5) if valid(f[6]) else -1
  inside = int(1000 * average(f[1]) + 0.5) if valid(f[7]) else -1
  return '%d %d %d %d' % (across, inside, reverse(f[2], f[3], f[6], f[7]), reverse(f[4], f[5], f[7], f[8]))


def analyse(clip: vs.VideoNode, log: str, segments: int = 5, length: int = 300) -> vs.VideoNode:
  """Field match and bob 'segments' parts of 'length' frames with both field orders; segments=0 analyses the whole clip."""
  if not isinstance(clip, vs.VideoNode):
    raise vs.Error('interlaceAnalysis.analyse: clip must be a VideoNode')
  length = max(1, length)
  frames = clip.num_frames
  starts: list[int] = []
  if segments > 0 and segments * length < frames:
    starts = _segmentStarts(frames, segments, length)

  # VFM, VDecimate and the vector masks take 8 bit YUV
  work = clip
  if clip.format.color_family != vs.YUV or clip.format.bits_per_sample != 8 or clip.format.sample_type != vs.INTEGER:
    work = core.resize.Point(clip, format=vs.YUV420P8, matrix_s='709' if clip.format.color_family == vs.RGB else None,
                             dither_type='none')

  topMatched, topDecimated = _matching(work, 1)
  bottomMatched, bottomDecimated = _matching(work, 0)
  nodes = [topMatched, topDecimated, bottomMatched, bottomDecimated] + _bobMotion(work, 2) + _bobMotion(work, 1)
  props = [_select(node, starts, length) for node in nodes]
  output = _select(clip, starts, length)

  lock = threading.Lock()
  with open(log, 'w', encoding='utf-8'):
    pass

  def _row(match: vs.VideoFrame, drop: vs.VideoFrame) -> str:
    return '%d %d %d' % (int(match.props.get('VFMMatch', 1)), int(match.props.get('_Combed', 0)),
                         int(drop.props.get('VDecimateDrop', 0)))

  def _log(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
    lines = '%d %s %s\nb %d %s %s\n' % (n, _row(f[1], f[2]), _row(f[3], f[4]), n, _bobRow(f[5:14]), _bobRow(f[14:23]))
    with lock:
      with open(log, 'a', encoding='utf-8') as file:
        file.write(lines)
    return f[0]

  return core.std.ModifyFrame(output, [output] + props, _log)
