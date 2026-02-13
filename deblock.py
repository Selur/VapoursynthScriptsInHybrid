import vapoursynth as vs
from vapoursynth import core

import math

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
