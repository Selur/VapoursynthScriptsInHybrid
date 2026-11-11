from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional
from vapoursynth import core
import math
import vapoursynth as vs

import misc
from helpers import pick_tool, tool_function, zsmooth_namespace

def _stab_expand_luma(clip: vs.VideoNode, luma_exp: int) -> vs.VideoNode:
    # luma expansion for the motion estimation only (Stab3s Luma_Exp): 1 = limited -> full levels, 2 = AutoAdjust
    if luma_exp == 1:
        shift = clip.format.bits_per_sample - 8
        return core.std.Levels(clip, min_in=16 << shift, max_in=235 << shift, min_out=0, max_out=(1 << clip.format.bits_per_sample) - 1, planes=[0])
    if luma_exp == 2:
        if not hasattr(core, 'autoadjust'):
            raise vs.Error('Stab: luma_exp=2 needs the AutoAdjust plugin')
        return core.autoadjust.AutoAdjust(clip, temporal_radius=10, auto_balance=0, auto_gain=1, use_interp=1, avg_safety=0.25,
                                          dark_limit=10, bright_limit=10, gamma_limit=1.0, dark_exclude=0.05, bright_exclude=0.05,
                                          chroma_process=0, scd_threshold=16, input_tv=0, output_tv=0, debug_view=0)
    return clip


def _stab_fix_false_positives(stabilized: vs.VideoNode, clp: vs.VideoNode, dxmax: int, tools: Optional[Mapping[str, str]]) -> vs.VideoNode:
    # Stab3s FixFalPos: a frame whose edge strip (2*dxmax wide) holds a black border wider than 2 px is replaced by the source frame
    if pick_tool(tools, 'planestats', ('vszip',), lambda name: hasattr(core, 'vszip')) != 'vszip':
        raise vs.Error('Stab: fixfalpos needs vszip (PlaneMinMax)')
    luma = core.std.ShufflePlanes(stabilized, 0, vs.GRAY)
    width, height = luma.width, luma.height
    strip = dxmax * 2
    if strip <= 2 or strip >= min(width, height):
        return stabilized
    ignore = max(0, math.ceil(99 - 2.0 / strip * 100)) / 100.0
    strips = [core.std.Crop(luma, bottom=height - strip), core.std.Crop(luma, top=height - strip),
              core.std.Crop(luma, right=width - strip), core.std.Crop(luma, left=width - strip)]
    stats = [core.vszip.PlaneMinMax(s, maxthr=ignore, prop='StabFalPos') for s in strips]

    def select(n: int, f: List[vs.VideoFrame]) -> vs.VideoNode:
        return clp if any(frame.props['StabFalPosMax'] <= 0 for frame in f) else stabilized

    return core.std.FrameEval(stabilized, select, prop_src=stats, clip_src=[stabilized, clp])


def Stab(clp: vs.VideoNode, dxmax: int = 8, dymax: int = 8, mirror: int = 5, scd_thscd1: float = 400.0, scd_thscd2: float = 130.0,
         ts: int = 7, zoom: float = 1.0, pixaspect: float = 1.0, luma_exp: int = 0, fixfalpos: bool = False,
         prefilter: Optional[vs.VideoNode] = None, tools: Optional[Mapping[str, str]] = None) -> vs.VideoNode:
    if not isinstance(clp, vs.VideoNode):
        raise TypeError('Stab: This is not a clip')
    if not 1 <= ts <= 7:
        raise ValueError('Stab: ts must be between 1 and 7')
    if luma_exp not in (0, 1, 2):
        raise ValueError('Stab: luma_exp must be 0, 1 or 2')
    if prefilter is not None and (prefilter.format.id != clp.format.id or prefilter.width != clp.width
                                  or prefilter.height != clp.height or prefilter.num_frames != clp.num_frames):
        raise ValueError('Stab: prefilter must match the clip in format, size and length')

    source = clp
    pref = misc.SCDetect(clip=clp if prefilter is None else prefilter, threshold=0.25, scd_thscd1=scd_thscd1, scd_thscd2=scd_thscd2, tools=tools)

    soften = zsmooth_namespace(tools, 'temporalsoften', ('zsmooth',))
    if soften:
        temp = getattr(core, soften).TemporalSoften(pref, radius=ts, threshold=[255], scenechange=-1, scalep=True)
        temp2 = getattr(core, soften).TemporalSoften(pref, radius=1, threshold=[255], scenechange=-1, scalep=True)
    else:
        temp = misc.AverageFrames(pref, weights=[1] * (2 * ts + 1), scd_thscd1=scd_thscd1, scd_thscd2=scd_thscd2, tools=tools)
        temp2 = misc.AverageFrames(pref, weights=[1] * 3, scd_thscd1=scd_thscd1, scd_thscd2=scd_thscd2, tools=tools)
    rep = tool_function(tools, 'rg', 'Repair')(temp, temp2, 1)
    inter = core.std.Interleave([rep, pref])
    estimate_input = _stab_expand_luma(inter, luma_exp)
    compensate_input = inter if prefilter is None else core.std.Interleave([rep, source])

    estimate_args: Dict[str, Any] = {'trust': 0, 'dxmax': dxmax, 'dymax': dymax}
    compensate_args: Dict[str, Any] = {'offset': -1, 'mirror': mirror}
    if zoom != 1.0:
        estimate_args['zoommax'] = zoom
    if pixaspect != 1.0:
        estimate_args['pixaspect'] = pixaspect
        compensate_args['pixaspect'] = pixaspect

    # mvutensils (https://github.com/myrsloik/mvutensils) carries the Depan family over from
    # mvtools largely unchanged: DePanEstimate -> DepanEstimate, DePan -> DepanCompensate, same
    # parameter meanings. Neither this mvtools build nor mvutensils takes `range`: in depan that only widened the neighbourhood of
    # frames estimated (and cached) per call, while frame n's own dx/dy still came solely from
    # the n-1 <-> n correlation, so dropping it does not change the estimate.
    depan = core.mvu if pick_tool(tools, 'mv', ('mvutensils', 'mv')) == 'mvutensils' else core.mv
    mdata = depan.DepanEstimate(estimate_input, **estimate_args)
    stabilized = depan.DepanCompensate(compensate_input, data=mdata, **compensate_args)[::2]

    if fixfalpos:
        stabilized = _stab_fix_false_positives(stabilized, source, dxmax, tools)
    return stabilized

    