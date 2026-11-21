from __future__ import annotations
from typing import Any, Dict, List, Mapping, Optional
from vapoursynth import core
import math
import vapoursynth as vs

import misc
from helpers import _EXPR_VARIABLES, expr_backend, get_expr, pick_tool, tool_function, zsmooth_namespace

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


def _stab_fix_false_positives(stabilized: vs.VideoNode, measured: vs.VideoNode, clp: vs.VideoNode, dxmax: int, thick: float,
                              tools: Optional[Mapping[str, str]]) -> vs.VideoNode:
    # Stab3s FixFalPos: a frame whose edge strip (2*dxmax wide) of `measured` holds a black border wider than `thick` px is replaced by the source frame
    if pick_tool(tools, 'planestats', ('vszip',), lambda name: hasattr(core, 'vszip')) != 'vszip':
        raise vs.Error('Stab: fixfalpos needs vszip (PlaneMinMax)')
    luma = core.std.ShufflePlanes(measured, 0, vs.GRAY)
    width, height = luma.width, luma.height
    strip = dxmax * 2
    if strip <= 2 or strip >= min(width, height):
        return stabilized
    ignore = max(0, math.ceil(99 - thick / strip * 100)) / 100.0
    strips = [core.std.Crop(luma, bottom=height - strip), core.std.Crop(luma, top=height - strip),
              core.std.Crop(luma, right=width - strip), core.std.Crop(luma, left=width - strip)]
    stats = [core.vszip.PlaneMinMax(s, maxthr=ignore, prop='StabFalPos') for s in strips]

    def select(n: int, f: List[vs.VideoFrame]) -> vs.VideoNode:
        return clp if any(frame.props['StabFalPosMax'] <= 0 for frame in f) else stabilized

    return core.std.FrameEval(stabilized, select, prop_src=stats, clip_src=[stabilized, clp])


def _stab_border_mask(compensate_input: vs.VideoNode, depan: Any, mdata: vs.VideoNode, compensate_args: Dict[str, Any]) -> vs.VideoNode:
    # GRAY8 mask of the area the compensation left uncovered (255), like DePan's alpha with alphainvert=true in Stab3s
    white = core.std.BlankClip(compensate_input, format=vs.GRAY8, color=255, keep=True)
    shifted = depan.DepanCompensate(white, data=mdata, **dict(compensate_args, mirror=0))[::2]
    return core.std.Expr(shifted, 'x 255 < 255 0 ?')


def _stab_fill_borders(stabilized: vs.VideoNode, source: vs.VideoNode, mask: vs.VideoNode, tools: Optional[Mapping[str, str]]) -> vs.VideoNode:
    # Stab3s FillBorder=1: the uncovered area takes the source pixels, then InpaintDelogo inpaints it with the per-frame mask
    import inpaintdelogo
    fmt = stabilized.format
    merge_mask = mask if fmt.bits_per_sample == 8 else core.std.Expr(mask, f'x 255 = {(1 << fmt.bits_per_sample) - 1} 0 ?', format=fmt.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0).id)
    pre = core.std.MaskedMerge(stabilized, source, merge_mask, planes=list(range(fmt.num_planes)), first_plane=True)
    return inpaintdelogo.InpaintDelogo(pre, mask=mask, Loc='0,0,0,0', Inflate=0, DynMask=1, tools=tools)


def _stab_sample_aspect_ratio(clip: vs.VideoNode) -> float:
    # _SARNum/_SARDen of the first frame, 1.0 when unset
    props = clip.get_frame(0).props
    num, den = props.get('_SARNum', 0), props.get('_SARDen', 0)
    return num / den if num > 0 and den > 0 else 1.0


def Stab(clp: vs.VideoNode, dxmax: int = 8, dymax: int = 8, mirror: int = 5, scd_thscd1: float = 400.0, scd_thscd2: float = 130.0,
         ts: int = 7, zoom: float = 1.0, pixaspect: Optional[float] = None, luma_exp: int = 0, fixfalpos: bool = False,
         fillborder: int = 0, prefilter: Optional[vs.VideoNode] = None, tools: Optional[Mapping[str, str]] = None) -> vs.VideoNode:
    if not isinstance(clp, vs.VideoNode):
        raise TypeError('Stab: This is not a clip')
    if not 1 <= ts <= 7:
        raise ValueError('Stab: ts must be between 1 and 7')
    if luma_exp not in (0, 1, 2):
        raise ValueError('Stab: luma_exp must be 0, 1 or 2')
    if fillborder not in (0, 1):
        raise ValueError('Stab: fillborder must be 0 or 1')
    if fillborder and clp.format.color_family != vs.YUV:
        raise ValueError('Stab: fillborder needs a YUV clip')
    if prefilter is not None and (prefilter.format.id != clp.format.id or prefilter.width != clp.width
                                  or prefilter.height != clp.height or prefilter.num_frames != clp.num_frames):
        raise ValueError('Stab: prefilter must match the clip in format, size and length')

    if pixaspect is None:
        pixaspect = _stab_sample_aspect_ratio(clp)
    source = clp
    pref = misc.SCDetect(clip=clp if prefilter is None else prefilter, threshold=0.25, scd_thscd1=scd_thscd1, scd_thscd2=scd_thscd2, tools=tools)

    focus2 = (tools or {}).get('temporalsoften') == 'focus2' and hasattr(core, 'focus2')
    soften = None if focus2 else zsmooth_namespace(tools, 'temporalsoften', ('zsmooth',))
    if focus2:
        # scenechange > 0 makes focus2 read the _SceneChangePrev/Next of SCDetect above, like zsmooth's -1
        temp = core.focus2.TemporalSoften2(pref, radius=ts, luma_threshold=255, chroma_threshold=255, scenechange=1, mode=2)
        temp2 = core.focus2.TemporalSoften2(pref, radius=1, luma_threshold=255, chroma_threshold=255, scenechange=1, mode=2)
    elif soften:
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
    compensate_args: Dict[str, Any] = {'offset': -1, 'mirror': 0 if fillborder else mirror}
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

    result = stabilized
    if fillborder:
        result = _stab_fill_borders(stabilized, source, _stab_border_mask(compensate_input, depan, mdata, compensate_args), tools)
    if fixfalpos:
        result = _stab_fix_false_positives(result, stabilized, source, dxmax, 3.0 if fillborder else 2.0, tools)
    return result

    

def _vbd_shift(clip: vs.VideoNode, d: int) -> vs.VideoNode:
    # frame n shows frame n + d; beyond the clip ends the first/last frame is repeated with _VBDPad=1 (set on every frame: akarin does not read a missing prop as NaN)
    n = clip.num_frames
    k = min(abs(d), n)
    body = core.std.SetFrameProps(clip, _VBDPad=0)
    if d < 0:
        pad = core.std.SetFrameProps(clip[0], _VBDPad=1) * k
        return pad if k == n else core.std.Splice([pad, body[:n - k]])
    pad = core.std.SetFrameProps(clip[n - 1], _VBDPad=1) * k
    return pad if k == n else core.std.Splice([body[k:], pad])


def _vbd_neighbour_median(clip: vs.VideoNode, radius: int, step: int, mc: bool, thsad: int, tools: Optional[Mapping[str, str]]) -> vs.VideoNode:
    # median of the frames n +- k*step (k = 1..radius); one outside the clip or behind a scene change takes its counterpart on the other side, or frame n if that is unusable too
    if mc:
        MV = misc.get_mv(tools)
        c16 = core.resize.Point(clip, format=vs.GRAY16)
        geometry = dict(blksize=16, overlap=8)
        search = MV.Super(c16, pel=2, sharp=2, chroma=False, **geometry)
        render = MV.Super(c16, pel=2, sharp=2, levels=1, chroma=False, **geometry)

        def neighbour(d: int) -> vs.VideoNode:
            vectors = MV.Analyse(search, isb=d > 0, delta=abs(d), search=4, dct=5, chroma=False, **geometry)
            comp = core.resize.Point(MV.Compensate(c16, render, vectors, thsad=thsad), format=vs.GRAYS)
            return core.std.CopyFrameProps(comp, _vbd_shift(clip, d))
    else:
        def neighbour(d: int) -> vs.VideoNode:
            return _vbd_shift(clip, d)
    clips = [clip]
    names: Dict[int, List[str]] = {-1: [], 1: []}
    bad: Dict[int, List[str]] = {-1: [], 1: []}
    for direction, toward, away in ((-1, '_SceneChangePrev', '_SceneChangeNext'), (1, '_SceneChangeNext', '_SceneChangePrev')):
        # a scene change between n and the k-th neighbour: the toward-prop of the previous neighbour (x for k=1), and with step 2 also the away-prop of the neighbour itself
        cut, inner = None, 'x'
        for k in range(1, radius + 1):
            clips.append(neighbour(direction * k * step))
            v = _EXPR_VARIABLES[len(clips) - 1]
            hit = f'{inner}.{toward} 0.5 >'
            if step == 2:
                hit = f'{hit} {v}.{away} 0.5 > or'
            cut = hit if cut is None else f'{cut} {hit} or'
            names[direction].append(v)
            bad[direction].append(f'{cut} {v}._VBDPad 0.5 > or')
            inner = v
    terms = []
    for k in range(radius):
        for side in (-1, 1):
            own, other = names[side][k], names[-side][k]
            terms.append(f'{bad[side][k]} {bad[-side][k]} x {other} ? {own} ?')
    expr = ' '.join(terms) + f' sort{2 * radius}'
    if radius > 1:
        expr += f' drop{radius - 1}'
    expr += ' + 2 /'
    if radius > 1:
        expr += f' swap{radius - 1} drop{radius - 1}'
    return get_expr(tools)(clips, expr)


def _vbd_analyse(clip: vs.VideoNode, radius: int, width: int, motion: float, smooth: int, maxwidth: int, limit: float, strength: float,
                 scthresh: float, strips: int, mc: bool, thsad: int, step: int, tools: Optional[Mapping[str, str]], name: str) -> Dict[str, Any]:
    # shared analysis of VerticalBandDeflicker and its mask: analysis-size difference to the reference and still mask, full-size correction (8-bit levels)
    fmt = clip.format
    if fmt.color_family not in (vs.YUV, vs.GRAY):
        raise vs.Error(f'{name}: YUV or GRAY input required')
    if not 1 <= radius <= 12:
        raise vs.Error(f'{name}: radius must be 1..12')
    if step not in (1, 2):
        raise vs.Error(f'{name}: step must be 1 or 2')
    if strips < 1 or width < 16:
        raise vs.Error(f'{name}: strips must be >= 1 and width >= 16')
    if expr_backend(tools) not in ('akarin', 'cranexpr'):
        raise vs.Error(f'{name}: needs akarin or cranexpr as tools["expr"] (sort and frame properties)')
    expr = get_expr(tools)
    height = max(16, round(clip.height * width / clip.width / 2) * 2)
    luma = core.std.ShufflePlanes(clip, 0, vs.GRAY) if fmt.color_family == vs.YUV else clip
    small = core.resize.Bilinear(luma, width, height, format=vs.GRAYS)
    # scene detection on an 8-bit copy: scd.Detect takes integer clips only
    scenes = misc.SCDetect(core.resize.Point(small, format=vs.GRAY8), threshold=scthresh, tools=tools)
    small = core.std.CopyFrameProps(small, scenes, props=['_SceneChangePrev', '_SceneChangeNext'])
    ref = _vbd_neighbour_median(small, radius, step, mc, thsad, tools)
    # difference to the reference and a mask of the pixels that do not move
    diff = expr([small, ref], 'x y - 255 *')
    still = expr([diff], f'x abs {motion} < 1 0 ?')
    # masked mean per column and strip: downscaling to `strips` rows averages each strip
    msum = core.resize.Bilinear(expr([diff, still], 'x y *'), width, strips)
    mcnt = core.resize.Bilinear(still, width, strips)
    prof = expr([msum, mcnt], 'y 0.333 > x y / 0 ?')
    if smooth > 0:
        prof = core.std.BoxBlur(prof, hradius=smooth, vradius=0)
    wide = core.std.BoxBlur(prof, hradius=maxwidth // 2, hpasses=2, vradius=0)
    corr = core.resize.Bilinear(expr([prof, wide], f'x y - {strength} * {-limit} max {limit} min'), clip.width, clip.height)
    return {'luma': luma, 'diff': diff, 'still': still, 'corr': corr, 'expr': expr}


def VerticalBandDeflickerMask(clip: vs.VideoNode, mode: int = 1, scale: float = 10.0, radius: int = 12, width: int = 480, motion: float = 10.0,
                              smooth: int = 2, maxwidth: int = 150, limit: float = 8.0, strength: float = 1.0, scthresh: float = 0.1, strips: int = 6,
                              mc: bool = True, thsad: int = 2000, step: int = 2, tools: Optional[Mapping[str, str]] = None) -> vs.VideoNode:
    # debug view of VerticalBandDeflicker in the input format, chroma neutral: mode 1 correction, 2 measurement mask (white = measured, black = motion), 3 difference to the temporal reference; 1 and 3 are mid gray +- value * scale
    if mode not in (1, 2, 3):
        raise vs.Error('VerticalBandDeflickerMask: mode must be 1, 2 or 3')
    parts = _vbd_analyse(clip, radius, width, motion, smooth, maxwidth, limit, strength, scthresh, strips, mc, thsad, step, tools, 'VerticalBandDeflickerMask')
    expr, luma = parts['expr'], parts['luma']
    full = lambda c: core.resize.Point(c, clip.width, clip.height) if mode == 2 else core.resize.Bilinear(c, clip.width, clip.height)
    # value on the 8-bit scale (0..255) for the visible image
    if mode == 1:
        view, value = parts['corr'], f'x {scale} * 128 +'
    elif mode == 2:
        view, value = full(parts['still']), 'x 255 *'
    else:
        view, value = full(parts['diff']), f'x {scale} * 128 +'
    fmt = luma.format
    if fmt.sample_type == vs.INTEGER:
        level, peak = 1 << (fmt.bits_per_sample - 8), (1 << fmt.bits_per_sample) - 1
        shown = expr([view], f'{value} {level} * 0 max {peak} min', format=fmt.id)
    else:
        shown = expr([view], f'{value} 255 / 0 max 1 min', format=fmt.id)
    if clip.format.color_family == vs.GRAY:
        return shown
    neutral = 0 if clip.format.sample_type == vs.FLOAT else 1 << (clip.format.bits_per_sample - 1)
    return core.std.ShufflePlanes([shown, core.std.BlankClip(clip, color=[0, neutral, neutral], keep=True)], [0, 1, 2], vs.YUV)


def VerticalBandDeflicker(clip: vs.VideoNode, radius: int = 12, width: int = 480, motion: float = 10.0, smooth: int = 2,
                          maxwidth: int = 150, limit: float = 8.0, strength: float = 1.0, scthresh: float = 0.1, strips: int = 6,
                          mc: bool = True, thsad: int = 2000, step: int = 2, debug: int = 0, tools: Optional[Mapping[str, str]] = None) -> vs.VideoNode:
    # removes vertical luma bands that change over time (film density flicker in columns); levels are on the 8-bit scale, widths in analysis pixels, the reference reaches radius*step frames; debug 1..3 returns the VerticalBandDeflickerMask view of that mode
    if debug:
        return VerticalBandDeflickerMask(clip, mode=debug, radius=radius, width=width, motion=motion, smooth=smooth, maxwidth=maxwidth, limit=limit,
                                         strength=strength, scthresh=scthresh, strips=strips, mc=mc, thsad=thsad, step=step, tools=tools)
    fmt = clip.format
    parts = _vbd_analyse(clip, radius, width, motion, smooth, maxwidth, limit, strength, scthresh, strips, mc, thsad, step, tools, 'VerticalBandDeflicker')
    expr, luma, corr = parts['expr'], parts['luma'], parts['corr']
    if fmt.sample_type == vs.INTEGER:
        scale, peak = 1 << (fmt.bits_per_sample - 8), (1 << fmt.bits_per_sample) - 1
        fixed = expr([luma, corr], f'x y {scale} * - 0 max {peak} min', format=luma.format.id)
    else:
        fixed = expr([luma, corr], 'x y 255 / -', format=luma.format.id)
    if fmt.color_family == vs.GRAY:
        return fixed
    return core.std.ShufflePlanes([fixed, clip], [0, 1, 2], vs.YUV)
