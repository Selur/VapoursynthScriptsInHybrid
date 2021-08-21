import vapoursynth as vs
from vapoursynth import core
import functools
import math

def ChangeFPS(clp, target_fps_num, target_fps_den):
  attribute_clip = core.std.BlankClip(clp, length=math.floor(len(clp) * target_fps_num / target_fps_den * clp.fps_den / clp.fps_num), fpsnum=target_fps_num, fpsden=target_fps_den)
  adjusted_clip = core.std.FrameEval(attribute_clip, functools.partial(frame_adjuster, clip=clp, target_fps_num=target_fps_num, target_fps_den=target_fps_den))
  return adjusted_clip

def frame_adjuster(n, clip, target_fps_num, target_fps_den):
    real_n = math.floor(n / (target_fps_num / target_fps_den * clip.fps_den / clip.fps_num))
    one_frame_clip = clip[real_n] * (len(clip) + 100)
    return one_frame_clip

