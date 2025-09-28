import vapoursynth as vs
import math
import functools

core = vs.core

def ChangeFPS(clip: vs.VideoNode, target_fps_num: int, target_fps_den: int = 1) -> vs.VideoNode:
    """
    Convert the framerate of a clip, efficiently for very long clips.
    Uses a precomputed lookup table to avoid per-frame calculations.

    :param clip: Input clip
    :param target_fps_num: Numerator of target framerate
    :param target_fps_den: Denominator of target framerate
    :return: Clip with framerate converted
    """
    # Berechnung des Faktors
    factor = (target_fps_num / target_fps_den) * (clip.fps_den / clip.fps_num)
    new_length = round(len(clip) * factor)

    # Precompute lookup table der Frame-Indizes
    lookup = [min(round(n / factor), len(clip) - 1) for n in range(new_length)]

    # FrameEval-Funktion
    def frame_adjuster(n, clip, lookup):
        return clip[lookup[n]]

    # BlankClip für neue Länge
    attribute_clip = core.std.BlankClip(
        clip, length=new_length, fpsnum=target_fps_num, fpsden=target_fps_den
    )

    # FrameEval mit vorberechnetem Lookup
    return core.std.FrameEval(
        attribute_clip,
        functools.partial(frame_adjuster, clip=clip, lookup=lookup)
    )
