from __future__ import annotations
from typing import Any, Dict, Mapping, Optional, Sequence, Union
import vapoursynth as vs
from misc import get_mv

# dependencies:
# Dust (https://github.com/Selur/Dust), namespace dust
# mvutensils (https://github.com/Jaded-Encoding-Thaumaturgy/mvutensils) or MVTools (https://github.com/dubhater/vapoursynth-mvtools), picked by tools['mv']

# based on Dust.py of https://github.com/Selur/Dust 1.0.0

_DEFAULT_LIMIT = {"faery": 2, "pixie": 5, "gold": 8, "space": 6}
_FUNCTION = {"faery": "FaeryDust", "pixie": "PixieDust", "gold": "GoldDust", "space": "SpaceDust"}


def DustDenoise(clip: vs.VideoNode, mode: str = "pixie", limit: Optional[int] = None, limit_edge: int = -1,
                edge_thresh: int = 32, tr: int = 1, temporal: int = 0, planes: Optional[Union[int, Sequence[int]]] = None,
                scthresh: int = 20, interlaced: bool = False, mc: bool = True, thsad: int = 300, blksize: int = 8,
                overlap: int = 4, search: int = 4, pel: int = 2, truemotion: bool = True, opt: int = -1,
                tools: Optional[Mapping[str, str]] = None) -> vs.VideoNode:
    """Limit-clipping spatiotemporal denoising with motion compensated neighbours.

    mode:        "faery" (temporal only), "pixie" (temporal + spatial, default), "gold" (spatial-weighted) or
                 "space" (spatial only, no motion compensation)
    limit:       maximum change per sample, 8-bit scale; default per mode 2 / 5 / 8 / 6
    limit_edge:  limit on edges (edge-aware limit map), -1 = off, at most limit
    edge_thresh: Sobel magnitude counted as a full edge
    tr:          temporal radius 1..5; temporal: 0 = mean, 1 = median
    planes:      plane indices to process, default all
    scthresh:    scene change threshold (mean absolute luma difference, 8-bit scale); 0 = only the frame
                 properties _SceneChangePrev/_SceneChangeNext, -1 = off
    interlaced:  field mode; the motion compensation then works per field (the field order comes from
                 _FieldBased, see std.SeparateFields)
    mc:          False = no motion compensation (Dust uses the plain neighbour frames)
    thsad, blksize, overlap, search, pel, truemotion: motion vector settings
    opt:         instruction set, -1 = best available
    tools:       {'mv': 'mvutensils' | 'mvgpu' | 'mv' | 'mvsf'} picks the motion vector implementation
    """
    core = vs.core
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("DustDenoise: clip must be a video clip")
    if not hasattr(core, "dust"):
        raise vs.Error("DustDenoise: the Dust plugin (namespace dust) is not loaded")
    mode = mode.lower()
    if mode not in _FUNCTION:
        raise vs.Error('DustDenoise: mode must be "faery", "pixie", "gold" or "space"')
    if not 1 <= tr <= 5:
        raise vs.Error("DustDenoise: tr must be 1..5")
    if isinstance(planes, int):
        planes = [planes]
    func = getattr(core.dust, _FUNCTION[mode])
    args: Dict[str, Any] = dict(limit=_DEFAULT_LIMIT[mode] if limit is None else limit, limit_edge=limit_edge,
                                edge_thresh=edge_thresh, scthresh=scthresh, interlaced=int(interlaced), opt=opt)
    if planes is not None:
        args["planes"] = list(planes)
    if mode == "space":
        return func(clip, **args)
    args.update(tr=tr, temporal=temporal)
    if not mc:
        return func(clip, **args)

    # Motion compensation on the frames, or per field (the same field of frame n +- k is field n +- 2k).
    MV = get_mv(tools)
    chroma = clip.format.num_planes > 1 and (planes is None or any(p in (1, 2) for p in planes))
    src = core.std.SeparateFields(clip) if interlaced else clip
    step = 2 if interlaced else 1
    sup = MV.Super(src, hpad=16, vpad=16, pel=pel, sharp=2, blksize=blksize, overlap=overlap)
    prev, nxt = [], []
    for k in range(1, tr + 1):
        common: Dict[str, Any] = dict(delta=k * step, blksize=blksize, overlap=overlap, search=search,
                                      truemotion=truemotion, chroma=chroma)
        p = MV.Compensate(src, sup, MV.Analyse(sup, isb=False, **common), thsad=thsad)
        n = MV.Compensate(src, sup, MV.Analyse(sup, isb=True, **common), thsad=thsad)
        if interlaced:
            p = core.std.DoubleWeave(p)[::2]
            n = core.std.DoubleWeave(n)[::2]
        prev.append(p)
        nxt.append(n)
    return func(clip, prev=prev, next=nxt, **args)
