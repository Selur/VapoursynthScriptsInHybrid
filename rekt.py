from __future__ import annotations
import vapoursynth as vs
from vapoursynth import core

def rekt(src: vs.VideoNode, fix: vs.VideoNode, left: int = 0, top: int = 0, right: int = 0, bottom: int = 0) -> vs.VideoNode:
    '''Creates a rectangular "mask" for a fix to be applied to.'''

    if left > 0 or right > 0:
        m = core.std.Crop(fix, left=left, right=right)
        l = core.std.Crop(src, right=src.width - left) if left > 0 else 0
        r = core.std.Crop(src, left=src.width - right) if right > 0 else 0
        params = [x for x in [l, m, r] if x != 0]
        m = core.std.StackHorizontal(params)
    else:
        m = fix
    if top > 0 or bottom > 0:
        t = core.std.Crop(src, bottom=src.height - top) if top > 0 else 0
        m = core.std.Crop(m, bottom=bottom, top=top)
        b = core.std.Crop(src, top=src.height - bottom) if bottom > 0 else 0
        params = [x for x in [t, m, b] if x != 0]
        m = core.std.StackVertical(params)
    return m
