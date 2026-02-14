import vapoursynth as vs
from vapoursynth import core

import math
from functools import partial

def Deblock_QED(
    clp: vs.VideoNode, quant1: int = 24, quant2: int = 26, aOff1: int = 1, bOff1: int = 2, aOff2: int = 1, bOff2: int = 2, uv: int = 3
) -> vs.VideoNode:
    '''
    A postprocessed Deblock: Uses full frequencies of Deblock's changes on block borders, but DCT-lowpassed changes on block interiours.

    Parameters:
        clp: Clip to process.

        quant1: Strength of block edge deblocking.

        quant2: Strength of block internal deblocking.

        aOff1: Halfway "sensitivity" and halfway a strength modifier for borders.

        bOff1: "Sensitivity to detect blocking" for borders.

        aOff2: Halfway "sensitivity" and halfway a strength modifier for block interiors.

        bOff2: "Sensitivity to detect blocking" for block interiors.

        uv:
            3 = use proposed method for chroma deblocking
            2 = no chroma deblocking at all (fastest method)
            1 = directly use chroma debl. from the normal Deblock()
            -1 = directly use chroma debl. from the strong Deblock()
    '''
    if not isinstance(clp, vs.VideoNode):
        raise vs.Error('Deblock_QED: this is not a clip')

    is_gray = clp.format.color_family == vs.GRAY
    planes = [0, 1, 2] if uv > 2 and not is_gray else 0

    if clp.format.sample_type == vs.INTEGER:
        bits = clp.format.bits_per_sample
        neutral = 1 << (bits - 1)
        peak = (1 << bits) - 1
    else:
        neutral = 0.0
        peak = 1.0

    # add borders if clp is not mod 8
    w = clp.width
    h = clp.height
    padX = 8 - w % 8 if w & 7 else 0
    padY = 8 - h % 8 if h & 7 else 0
    if padX or padY:
        clp = clp.resize.Point(w + padX, h + padY, src_width=w + padX, src_height=h + padY)

    # block
    block = clp.std.BlankClip(width=6, height=6, format=clp.format.replace(color_family=vs.GRAY, subsampling_w=0, subsampling_h=0), length=1, color=0)
    block = block.std.AddBorders(1, 1, 1, 1, color=peak)
    block = core.std.StackHorizontal([block for _ in range(clp.width // 8)])
    block = core.std.StackVertical([block for _ in range(clp.height // 8)])
    if not is_gray:
        blockc = block.std.CropAbs(width=clp.width >> clp.format.subsampling_w, height=clp.height >> clp.format.subsampling_h)
        block = core.std.ShufflePlanes([block, blockc], planes=[0, 0, 0], colorfamily=clp.format.color_family)
    block = block.std.Loop(times=clp.num_frames)

    # create normal deblocking (for block borders) and strong deblocking (for block interiour)
    normal = clp.deblock.Deblock(quant=quant1, aoffset=aOff1, boffset=bOff1, planes=[0, 1, 2] if uv != 2 and not is_gray else 0)
    strong = clp.deblock.Deblock(quant=quant2, aoffset=aOff2, boffset=bOff2, planes=[0, 1, 2] if uv != 2 and not is_gray else 0)

    # build difference maps of both
    normalD = core.std.MakeDiff(clp, normal, planes=planes)
    strongD = core.std.MakeDiff(clp, strong, planes=planes)

    # separate border values of the difference maps, and set the interiours to '128'
    expr = f'y {peak} = x {neutral} ?'
    EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr
    normalD2 = EXPR([normalD, block], expr=expr if uv > 2 or is_gray else [expr, ''])
    strongD2 = EXPR([strongD, block], expr=expr if uv > 2 or is_gray else [expr, ''])

    # interpolate the border values over the whole block: DCTFilter can do it. (Kiss to Tom Barry!)
    # (Note: this is not fully accurate, but a reasonable approximation.)
    # add borders if clp is not mod 16
    sw = strongD2.width
    sh = strongD2.height
    remX = 16 - sw % 16 if sw & 15 else 0
    remY = 16 - sh % 16 if sh & 15 else 0
    if remX or remY:
        strongD2 = strongD2.resize.Point(sw + remX, sh + remY, src_width=sw + remX, src_height=sh + remY)
    expr = f'x {neutral} - 1.01 * {neutral} +'
    strongD3 = (
        EXPR(strongD2, expr=expr if uv > 2 or is_gray else [expr, ''])
        .dctf.DCTFilter(factors=[1, 1, 0, 0, 0, 0, 0, 0], planes=planes)
        .std.Crop(right=remX, bottom=remY)
    )

    # apply compensation from "normal" deblocking to the borders of the full-block-compensations calculated from "strong" deblocking ...
    expr = f'y {neutral} = x y ?'
    strongD4 = EXPR([strongD3, normalD2], expr=expr if uv > 2 or is_gray else [expr, ''])

    # ... and apply it.
    deblocked = core.std.MakeDiff(clp, strongD4, planes=planes)

    # simple decisions how to treat chroma
    if not is_gray:
        if uv < 0:
            deblocked = core.std.ShufflePlanes([deblocked, strong], planes=[0, 1, 2], colorfamily=clp.format.color_family)
        elif uv < 2:
            deblocked = core.std.ShufflePlanes([deblocked, normal], planes=[0, 1, 2], colorfamily=clp.format.color_family)

    # remove mod 8 borders
    return deblocked.std.Crop(right=padX, bottom=padY)

"""
based on https://github.com/Irrational-Encoding-Wizardry/fvsfunc/blob/076dbde68227f6cca91304a447b2a02b0e95413e/fvsfunc.py#L773
VapourSynth port of AutoDeblock2. Original script by joletb, vinylfreak89, eXmendiC and Gebbi.

The purpose of this script is to automatically remove MPEG2 artifacts.

Supports 8..16 bit integer YUV formats

Adjusted by Selur to use faster libraries for speed
"""
def AutoDeblock(src: vs.VideoNode, edgevalue: int = 24, db1: int = 1, db2: int = 6, db3: int = 15, deblocky: bool = True, deblockuv: bool = True, debug: bool = False, redfix: bool = False, fastdeblock: bool = False, adb1: int = 3, adb2: int = 4, adb3: int = 8, adb1d: int = 2, adb2d: int = 7, adb3d: int = 11, planes: Optional[List[int]] = None) -> vs.VideoNode:
    """
    Automatically deblocks a YUV clip using adaptive thresholds and optional red-area correction.

    Parameters
    ----------
    src : vs.VideoNode
        Input clip, must be YUV color family, 8-16 bit integer format.
    edgevalue : int
        Threshold for detecting edges, scaled internally for bit depth.
    db1, db2, db3 : int
        Deblocking strengths for weak, medium, and strong deblocking passes.
    deblocky : bool
        Whether to deblock the luma plane.
    deblockuv : bool
        Whether to deblock the chroma planes.
    debug : bool
        If True, overlays debug information showing differences for each frame.
    redfix : bool
        If True, avoids over-deblocking red-colored areas. Cannot be combined with `fastdeblock`.
    fastdeblock : bool
        If True, applies a faster, single-pass deblock method based on thresholds.
    adb1, adb2, adb3 : int
        Adaptive thresholds for OrigDiff to determine deblocking strength.
    adb1d, adb2d, adb3d : int
        Adaptive thresholds for YNextDiff to determine deblocking strength.
    planes : Optional[List[int]]
        List of planes to process; defaults to `[0]` if deblocky or `[1,2]` if deblockuv

    Returns
    -------
    vs.VideoNode

    Raises
    ------
    TypeError, ValueError
    """
    if src.format.color_family not in [vs.YUV]: raise TypeError("AutoDeblock: src must be YUV color family!")
    if src.format.bits_per_sample < 8 or src.format.bits_per_sample > 16 or src.format.sample_type != vs.INTEGER:
        raise TypeError("AutoDeblock: src must be between 8 and 16 bit integer format")

    shift = src.format.bits_per_sample - 8
    edgevalue <<= shift
    maxvalue = (1 << src.format.bits_per_sample) - 1

    def to8bit(f: float) -> float: return f * 255

    def sub_props(src: vs.VideoNode, f: List[vs.VideoNode], name: str) -> vs.VideoNode:
        OrigDiff_str = str(to8bit(f[0].props.OrigDiff))
        YNextDiff_str = str(to8bit(f[1].props.YNextDiff))
        return core.sub.Subtitle(src, name + f"\nOrigDiff: {OrigDiff_str}\nYNextDiff: {YNextDiff_str}")

    def eval_deblock_strength(n: int, f: List[vs.VideoNode], fastdeblock: bool, debug: bool, unfiltered: vs.VideoNode, fast: vs.VideoNode, weakdeblock: vs.VideoNode, mediumdeblock: vs.VideoNode, strongdeblock: vs.VideoNode) -> vs.VideoNode:
        unfiltered = sub_props(unfiltered, f, "unfiltered") if debug else unfiltered
        out = unfiltered
        if fastdeblock:
            if to8bit(f[0].props.OrigDiff) > adb1 and to8bit(f[1].props.YNextDiff) > adb1d:
                return sub_props(fast, f, "deblock") if debug else fast
            else: return unfiltered
        if to8bit(f[0].props.OrigDiff) > adb1 and to8bit(f[1].props.YNextDiff) > adb1d:
            out = sub_props(weakdeblock, f, "weakdeblock") if debug else weakdeblock
        if to8bit(f[0].props.OrigDiff) > adb2 and to8bit(f[1].props.YNextDiff) > adb2d:
            out = sub_props(mediumdeblock, f, "mediumdeblock") if debug else mediumdeblock
        if to8bit(f[0].props.OrigDiff) > adb3 and to8bit(f[1].props.YNextDiff) > adb3d:
            out = sub_props(strongdeblock, f, "strongdeblock") if debug else strongdeblock
        return out

    def fix_red(n: int, f: List[vs.VideoNode], unfiltered: vs.VideoNode, autodeblock: vs.VideoNode) -> vs.VideoNode:
        if 50 < to8bit(f[0].props.YAverage) < 130 and 95 < to8bit(f[1].props.UAverage) < 130 and 130 < to8bit(f[2].props.VAverage) < 155: return unfiltered
        return autodeblock

    if redfix and fastdeblock: raise ValueError('AutoDeblock: You cannot set both "redfix" and "fastdeblock" to True!')

    if planes is None:
        planes = []
        if deblocky: planes.append(0)
        if deblockuv: planes.extend([1, 2])

    orig = core.std.Prewitt(src)
    EXPR = core.llvmexpr.Expr if hasattr(core, 'llvmexpr') else core.akarin.Expr if hasattr(core, 'akarin') else core.cranexpr.Expr if hasattr(core, 'cranexpr') else core.std.Expr
    orig = EXPR(orig, f"x {edgevalue} >= {maxvalue} x ?")

    isFLOAT = src.format.sample_type == vs.FLOAT
    RG = core.zsmooth.RemoveGrain if hasattr(core, 'zsmooth') else core.rgsf.RemoveGrain if hasattr(core, 'rgsf') and isFLOAT else core.rgvs.RemoveGrain

    orig_d = RG(orig, 4)
    orig_d = RG(orig_d, 4)
    src_d = RG(src, 2)
    src_d = RG(src_d, 2)

    unfiltered = src
    predeblock = Deblock_QED(src_d)

    if hasattr(core, 'dfttest2_nvrtc'):
        import dfttest2
        backend = dfttest2.Backend.NVRTC
        DFTTest = partial(dfttest2.DFTTest, backend=backend)
    else: DFTTest = core.dfttest.DFTTest

    fast = DFTTest(predeblock, tbsize=1)
    weakdeblock = DFTTest(predeblock, sigma=db1, tbsize=1, planes=planes)
    mediumdeblock = DFTTest(predeblock, sigma=db2, tbsize=1, planes=planes)
    strongdeblock = DFTTest(predeblock, sigma=db3, tbsize=1, planes=planes)

    difforig = core.std.PlaneStats(orig, orig_d, prop='Orig')
    diffnext = core.std.PlaneStats(src, src.std.DeleteFrames([0]), prop='YNext')

    autodeblock = core.std.FrameEval(
        unfiltered,
        partial(eval_deblock_strength, fastdeblock=fastdeblock, debug=debug, unfiltered=unfiltered,
                fast=fast, weakdeblock=weakdeblock, mediumdeblock=mediumdeblock, strongdeblock=strongdeblock),
        prop_src=[difforig, diffnext]
    )

    if redfix:
        src_y = core.std.PlaneStats(src, prop='Y')
        src_u = core.std.PlaneStats(src, plane=1, prop='U')
        src_v = core.std.PlaneStats(src, plane=2, prop='V')
        autodeblock = core.std.FrameEval(
            unfiltered,
            partial(fix_red, unfiltered=unfiltered, autodeblock=autodeblock),
            prop_src=[src_y, src_u, src_v]
        )

    return autodeblock