'''One measuring call per tool family: the family's main function, run through the same dispatcher a filter uses.

call(clip, family, value, device) returns `clip` filtered by implementation `value` of `family`; every value of a family
gets the same arguments, so the frame rates compare the implementations and nothing else. `clip` is YUV 4:2:0, 8 or
16 bit integer; a family whose implementations need float converts inside its call, for every value alike. `device`
is the GPU number of values that open their own device, None otherwise.
'''
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import vapoursynth as vs

import helpers
import misc

core = vs.core


def _float(clip: vs.VideoNode) -> vs.VideoNode:
    return core.resize.Point(clip, format=clip.format.replace(sample_type=vs.FLOAT, bits_per_sample=32))


def _luma(clip: vs.VideoNode) -> vs.VideoNode:
    return core.std.ShufflePlanes(clip, 0, vs.GRAY)


def _peak(clip: vs.VideoNode) -> int:
    return (1 << clip.format.bits_per_sample) - 1


def _device(device: Optional[int]) -> Dict[str, Any]:
    return {} if device is None else {'device_id': device}


def _mv(clip, tools, device):
    mv = misc.get_mv(tools)
    sup = mv.Super(clip, pel=2, sharp=2, blksize=16, overlap=8)
    backward = mv.Analyse(sup, isb=True, delta=1, blksize=16, overlap=8)
    forward = mv.Analyse(sup, isb=False, delta=1, blksize=16, overlap=8)
    return mv.Degrain1(clip, sup, backward, forward, thsad=400) if hasattr(mv, 'Degrain1') \
        else mv.Degrain(clip, sup, backward, forward, thsad=400)


def _expr(clip, tools, device):
    expr = helpers.get_expr(tools)
    return expr([clip, core.std.BoxBlur(clip)], 'x y - abs 2 * x +')


def _nnedi3(clip, tools, device):
    return helpers.NNEDI3(clip, field=1, device=device, tools=tools)


def _eedi3(clip, tools, device):
    return helpers.EEDI3(clip, field=1, device=device, tools=tools)


def _dfttest(clip, tools, device):
    return helpers.DFTTest(clip, sigma=8.0, tbsize=3, tools=tools, device_id=device)


def _nlmeans(clip, tools, device):
    return helpers.NLMeans(clip, d=1, a=2, s=4, h=1.2, channels='Y', device_id=device, tools=tools)


def _bm3d(clip, tools, device):
    # luma only: bm3dvk2 filters no YUV chroma
    return helpers.BM3D(_float(_luma(clip)), sigma=[3.0], radius=1, device_id=device, tools=tools)


def _bilateral(clip, tools, device):
    # the dispatcher of the vsscripts: vsfeel runs at 16 bit there, its GPU frames come back as normal frames
    import derainbow
    return derainbow._bilateral(clip, sigmaS=3.0, sigmaR=0.02, tools=tools, **_device(device))


def _wnnm(clip, tools, device):
    # no wnnm entry in helpers.TOOLS: the value is the namespace
    # vsfeel returns GPU-resident frames; Hybrid's WNNM downloads them like this
    return helpers.gpu_download(tools['wnnm'], getattr(core, tools['wnnm']).WNNM(_float(_luma(clip)), sigma=3.0, radius=1))


def _placebodeband(clip, tools, device):
    # no placebodeband entry in helpers.TOOLS: the value is the namespace
    value = tools['placebodeband']
    # libplacebo takes a plane bit mask, the vszip ports plane indices
    arguments = {'planes': 1 | 2 | 4} if value == 'placebo' else {'planes': [0, 1, 2]}
    if value != 'placebo' and device is not None:
        arguments['device_id'] = device
    return getattr(core, value).Deband(clip, iterations=2, threshold=4.0, radius=16.0, grain=0.0, **arguments)


def _rg(clip, tools, device):
    removegrain = helpers.tool_function(tools, 'rg', 'RemoveGrain')
    return removegrain(removegrain(clip, 11), 20)


def _median(clip, tools, device):
    namespace = helpers.pick_zsmooth(tools, 'median', ('zsmooth', 'std'), lambda name: name == 'std' or hasattr(core, name))
    return core.std.Median(clip) if namespace == 'std' else getattr(core, namespace).Median(clip, radius=2)


def _ttempsmooth(clip, tools, device):
    namespace = helpers.pick_zsmooth(tools, 'ttempsmooth', ('zsmooth', 'ttmpsm'), lambda name: hasattr(core, name))
    # scthresh -1: no scene change detection, it needs MiscFilters
    arguments = {} if namespace == 'ttmpsm' else {'scthresh': -1}
    return getattr(core, namespace).TTempSmooth(clip, maxr=3, **arguments)


def _temporalsoften(clip, tools, device):
    import derainbow
    # scenechange 0: the clip carries no scene change properties
    return derainbow._temporal_soften(clip, 4, 4, 8, 0, 2, tools=tools)


def _boxblur(clip, tools, device):
    return helpers.BoxFilter(clip, radius=8, tools=tools)


def _limiter(clip, tools, device):
    floating = _float(clip)
    if helpers.pick_tool(tools, 'limiter', ('vszip', 'std'), lambda name: name == 'std' or hasattr(core, 'vszip')) == 'vszip':
        return floating.vszip.Limiter()
    return floating.std.Limiter()


def _planestats(clip, tools, device):
    if helpers.pick_tool(tools, 'planestats', ('vszip', 'std'), lambda name: name == 'std' or hasattr(core, 'vszip')) == 'vszip':
        return core.vszip.PlaneMinMax(clip)
    return core.std.PlaneStats(clip)


def _edgemasks(clip, tools, device):
    prewitt = core.edgemasks.ExPrewitt if helpers.pick_tool(tools, 'edgemasks', ('edgemasks', 'std')) == 'edgemasks' else core.std.Prewitt
    return prewitt(clip)


def _tcanny(clip, tools, device):
    # TCanny as the Gaussian blur most scripts use it for; std: their stand-in, three box blur passes of radius round(1.5 * sigma)
    sigma = 1.5
    if helpers.pick_tool(tools, 'tcanny', ('tcanny', 'std')) == 'tcanny':
        return core.tcanny.TCanny(clip, sigma=sigma, mode=-1)
    radius = max(1, round(sigma * 1.5))
    return helpers.tool_function(None, 'boxblur', 'BoxBlur')(clip, hradius=radius, hpasses=3, vradius=radius, vpasses=3)


def _scd(clip, tools, device):
    # 'mv' on mvtools, the plugin the caller loads for it
    if tools['scd'] == 'mv':
        tools = dict(tools, mv='mv')
    return misc.SCDetect(clip, threshold=0.1, tools=tools)


def _hysteresis(clip, tools, device):
    luma = _luma(clip)
    peak = _peak(luma)
    strong = core.std.Binarize(core.std.Sobel(luma), threshold=peak // 4)
    weak = core.std.Binarize(core.std.Sobel(luma), threshold=peak // 16)
    return helpers.tool_function(tools, 'hysteresis', 'Hysteresis')(strong, weak)


def _dctfilter(clip, tools, device):
    factors = [1.0, 1.0, 1.0, 1.0, 0.75, 0.5, 0.25, 0.0]
    return helpers.tool_function(tools, 'dctfilter', 'DCTFilter')(clip, factors=factors)


def _warp(clip, tools, device):
    import sharpen
    return sharpen.AWarpSharp2(clip, thresh=128, blur=2, tools=tools)


def _cnr(clip, tools, device):
    namespace = helpers.pick_zsmooth(tools, 'cnr', ('zsmooth', 'cnr2'), lambda name: hasattr(core, name))
    return getattr(core, namespace).Cnr4(clip, scenechange=False)


def _ccd(clip, tools, device):
    # both implementations take RGB only
    rgb = core.resize.Point(clip, format=vs.RGB24 if clip.format.bits_per_sample == 8 else vs.RGB48, matrix_in_s='709')
    return helpers.CCD(rgb, threshold=8.0, tools=tools)


def _f3kdb(clip, tools, device):
    if tools['f3kdb'] == 'vszip':
        # vszip takes an 8 bit scale where the neo_f3kdb family takes 14 bit
        threshold = 64 * 255 / ((1 << 14) - 1)
        return core.vszip.Deband(clip, range=15, thr=(threshold, threshold, threshold), grain=(0, 0))
    return helpers.f3kdb_function(tools['f3kdb'])(clip, range=15, y=64, cb=64, cr=64, grainy=0, grainc=0)


def _vinverse(clip, tools, device):
    import residual
    return residual.Vinverse(clip, tools=tools)


CALLS: Dict[str, Callable[[vs.VideoNode, Dict[str, str], Optional[int]], vs.VideoNode]] = {
    'mv': _mv, 'expr': _expr, 'nnedi3': _nnedi3, 'eedi3': _eedi3, 'dfttest': _dfttest, 'nlmeans': _nlmeans,
    'bm3d': _bm3d, 'bilateral': _bilateral, 'wnnm': _wnnm, 'placebodeband': _placebodeband,
    'rg': _rg, 'median': _median, 'ttempsmooth': _ttempsmooth, 'temporalsoften': _temporalsoften,
    'boxblur': _boxblur, 'limiter': _limiter, 'planestats': _planestats, 'edgemasks': _edgemasks,
    'tcanny': _tcanny, 'scd': _scd, 'hysteresis': _hysteresis, 'dctfilter': _dctfilter, 'warp': _warp,
    'cnr': _cnr, 'ccd': _ccd, 'f3kdb': _f3kdb, 'vinverse': _vinverse,
}


def call(clip: vs.VideoNode, family: str, value: str, device: Optional[int] = None) -> vs.VideoNode:
    '''`clip` through implementation `value` of `family`.'''
    if family not in CALLS:
        raise vs.Error(f'benchmark: no measuring call for "{family}" (known: {", ".join(sorted(CALLS))})')
    return CALLS[family](clip, {family: value}, device)
