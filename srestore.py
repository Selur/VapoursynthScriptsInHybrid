from __future__ import annotations
import importlib

import math
from functools import partial
from typing import Any, Mapping, Optional, Sequence, Union

import vapoursynth as vs
from vsutil import Dither, depth, fallback, get_depth, get_y, join, plane, scale_value

import functools
import ChangeFPS

core = vs.core

    
###### srestore v2.7e ######
# sRestore from old havsfunc
def sRestore(source, frate=None, omode=6, speed=None, mode=2, thresh=16, dclip=None):
    if not isinstance(source, vs.VideoNode):
        raise vs.Error('srestore: this is not a clip')

    if source.format.color_family != vs.YUV:
        raise vs.Error('srestore: only YUV format is supported')

    if dclip is None:
        dclip = source
    elif not isinstance(dclip, vs.VideoNode):
        raise vs.Error("srestore: 'dclip' is not a clip")
    elif dclip.format.color_family != vs.YUV:
        raise vs.Error('srestore: only YUV format is supported')

    neutral = 1 << (source.format.bits_per_sample - 1)
    peak = (1 << source.format.bits_per_sample) - 1

    ###### parameters & other necessary vars ######
    srad = math.sqrt(abs(speed)) * 4 if speed is not None and abs(speed) >= 1 else 12
    irate = source.fps_num / source.fps_den
    bsize = 16 if speed is not None and speed > 0 else 32
    bom = isinstance(omode, str)
    thr = abs(thresh) + 0.01

    if bom or abs(omode - 3) < 2.5:
        frfac = 1
    elif frate is not None:
        if frate * 5 < irate or frate > irate:
            frfac = 1
        else:
            frfac = abs(frate) / irate
    elif cround(irate * 10010) % 30000 == 0:
        frfac = 1001 / 2400
    else:
        frfac = 480 / 1001

    if abs(frfac * 1001 - cround(frfac * 1001)) < 0.01:
        numr = cround(frfac * 1001)
    elif abs(1001 / frfac - cround(1001 / frfac)) < 0.01:
        numr = 1001
    else:
        numr = cround(frfac * 9000)
    if frate is not None and abs(irate * numr / cround(numr / frfac) - frate) > abs(irate * cround(frate * 100) / cround(irate * 100) - frate):
        numr = cround(frate * 100)
    denm = cround(numr / frfac)

    ###### source preparation & lut ######
    if abs(mode) >= 2 and not bom:
        mec = core.std.Merge(core.std.Merge(source, source.std.Trim(first=1), weight=[0, 0.5]), source.std.Trim(first=1), weight=[0.5, 0])

    if dclip.format.id != vs.YUV420P8:
        dclip = dclip.resize.Bicubic(format=vs.YUV420P8)
    dclip = dclip.resize.Point(dclip.width if srad == 4 else int(dclip.width / 2 / srad + 4) * 4, dclip.height if srad == 4 else int(dclip.height / 2 / srad + 4) * 4).std.Trim(first=2)
    if mode < 0:
        dclip = core.std.StackVertical([core.std.StackHorizontal([GetPlane(dclip, 1), GetPlane(dclip, 2)]), GetPlane(dclip, 0)])
    else:
        dclip = GetPlane(dclip, 0)
    if bom:
        dclip = dclip.std.Expr(expr=['x 0.5 * 64 +'])

    expr1 = 'x 128 - y 128 - * 0 > x 128 - abs y 128 - abs < x 128 - 128 x - * y 128 - 128 y - * ? x y + 256 - dup * ? 0.25 * 128 +'
    expr2 = 'x y - dup * 3 * x y + 256 - dup * - 128 +'
    diff = core.std.MakeDiff(dclip, dclip.std.Trim(first=1))
    if not bom:
        bclp = core.std.Expr([diff, diff.std.Trim(first=1)], expr=[expr1]).resize.Bilinear(bsize, bsize)
    else:
        bclp = core.std.Expr([diff.std.Trim(first=1), core.std.MergeDiff(diff, diff.std.Trim(first=2))], expr=[expr2]).resize.Bilinear(bsize, bsize)
    dclp = diff.std.Trim(first=1).std.Lut(function=lambda x: max(cround(abs(x - 128) ** 1.1 - 1), 0)).resize.Bilinear(bsize, bsize)

    ###### postprocessing ######
    if bom:
        sourceDuplicate = source.std.DuplicateFrames(frames=[0])
        sourceTrim1 = source.std.Trim(first=1)
        sourceTrim2 = source.std.Trim(first=2)

        unblend1 = core.std.Expr([sourceDuplicate, source], expr=['y 2 * x -'])
        unblend2 = core.std.Expr([sourceTrim1, sourceTrim2], expr=['x 2 * y -'])

        qmask1 = core.std.MakeDiff(unblend1.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1], planes=[0]), unblend1, planes=[0])
        qmask2 = core.std.MakeDiff(unblend2.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1], planes=[0]), unblend2, planes=[0])
        diffm = core.std.MakeDiff(sourceDuplicate, source, planes=[0]).std.Maximum(planes=[0])
        bmask = core.std.Expr([qmask1, qmask2], expr=[f'x {neutral} - dup * dup y {neutral} - dup * + / {peak} *', ''])
        expr = 'x 2 * y < x {i} < and 0 y 2 * x < y {i} < and {peak} x x y + / {j} * {k} + ? ?'.format(i=scale(4, peak), peak=peak, j=scale(200, peak), k=scale(28, peak))
        dmask = core.std.Expr([diffm, diffm.std.Trim(first=2)], expr=[expr, ''])
        pmask = core.std.Expr([dmask, bmask], expr=[f'y 0 > y {peak} < and x 0 = x {peak} = or and x y ?', ''])

        matrix = [1, 2, 1, 2, 4, 2, 1, 2, 1]

        omode = omode.lower()
        if omode == 'pp0':
            fin = core.std.Expr([sourceDuplicate, source, sourceTrim1, sourceTrim2], expr=['y x 2 / - z a 2 / - +'])
        elif omode == 'pp1':
            fin = core.std.MaskedMerge(unblend1, unblend2, dmask.std.Convolution(matrix=matrix, planes=[0]).std.Expr(expr=['', repr(neutral)]))
        elif omode == 'pp2':
            fin = core.std.MaskedMerge(unblend1, unblend2, bmask.std.Convolution(matrix=matrix, planes=[0]), first_plane=True)
        elif omode == 'pp3':
            fin = core.std.MaskedMerge(unblend1, unblend2, pmask.std.Convolution(matrix=matrix, planes=[0]), first_plane=True).std.Convolution(matrix=matrix, planes=[1, 2])
        else:
            raise vs.Error('srestore: unexpected value for omode')

    ###### initialise variables ######
    lfr = -100
    offs = 0
    ldet = -100
    lpos = 0
    d32 = d21 = d10 = d01 = d12 = d23 = d34 = None
    m42 = m31 = m20 = m11 = m02 = m13 = m24 = None
    bp2 = bp1 = bn0 = bn1 = bn2 = bn3 = None
    cp2 = cp1 = cn0 = cn1 = cn2 = cn3 = None

    def srestore_inside(n, f):
        nonlocal lfr, offs, ldet, lpos, d32, d21, d10, d01, d12, d23, d34, m42, m31, m20, m11, m02, m13, m24, bp2, bp1, bn0, bn1, bn2, bn3, cp2, cp1, cn0, cn1, cn2, cn3

        ### preparation ###
        jmp = lfr + 1 == n
        cfo = ((n % denm) * numr * 2 + denm + numr) % (2 * denm) - denm
        bfo = cfo > -numr and cfo <= numr
        lfr = n
        offs = offs + 2 * denm if bfo and offs <= -4 * numr else offs - 2 * denm if bfo and offs >= 4 * numr else offs
        pos = 0 if frfac == 1 else -cround((cfo + offs) / (2 * numr)) if bfo else lpos
        cof = cfo + offs + 2 * numr * pos
        ldet = -1 if n + pos == ldet else n + pos

        ### diff value shifting ###
        d_v = f[1].props['PlaneStatsMax'] + 0.015625
        if jmp:
            d43 = d32
            d32 = d21
            d21 = d10
            d10 = d01
            d01 = d12
            d12 = d23
            d23 = d34
        else:
            d43 = d32 = d21 = d10 = d01 = d12 = d23 = d_v
        d34 = d_v

        m_v = f[2].props['PlaneStatsDiff'] * 255 + 0.015625 if not bom and abs(omode) > 5 else 1
        if jmp:
            m53 = m42
            m42 = m31
            m31 = m20
            m20 = m11
            m11 = m02
            m02 = m13
            m13 = m24
        else:
            m53 = m42 = m31 = m20 = m11 = m02 = m13 = m_v
        m24 = m_v

        ### get blend and clear values ###
        b_v = 128 - f[0].props['PlaneStatsMin']
        if b_v < 1:
            b_v = 0.125
        c_v = f[0].props['PlaneStatsMax'] - 128
        if c_v < 1:
            c_v = 0.125

        ### blend value shifting ###
        if jmp:
            bp3 = bp2
            bp2 = bp1
            bp1 = bn0
            bn0 = bn1
            bn1 = bn2
            bn2 = bn3
        else:
            bp3 = b_v - c_v if bom else b_v
            bp2 = bp1 = bn0 = bn1 = bn2 = bp3
        bn3 = b_v - c_v if bom else b_v

        ### clear value shifting ###
        if jmp:
            cp3 = cp2
            cp2 = cp1
            cp1 = cn0
            cn0 = cn1
            cn1 = cn2
            cn2 = cn3
        else:
            cp3 = cp2 = cp1 = cn0 = cn1 = cn2 = c_v
        cn3 = c_v

        ### used detection values ###
        bb = [bp3, bp2, bp1, bn0, bn1][pos + 2]
        bc = [bp2, bp1, bn0, bn1, bn2][pos + 2]
        bn = [bp1, bn0, bn1, bn2, bn3][pos + 2]

        cb = [cp3, cp2, cp1, cn0, cn1][pos + 2]
        cc = [cp2, cp1, cn0, cn1, cn2][pos + 2]
        cn = [cp1, cn0, cn1, cn2, cn3][pos + 2]

        dbb = [d43, d32, d21, d10, d01][pos + 2]
        dbc = [d32, d21, d10, d01, d12][pos + 2]
        dcn = [d21, d10, d01, d12, d23][pos + 2]
        dnn = [d10, d01, d12, d23, d34][pos + 2]
        dn2 = [d01, d12, d23, d34, d34][pos + 2]

        mb1 = [m53, m42, m31, m20, m11][pos + 2]
        mb = [m42, m31, m20, m11, m02][pos + 2]
        mc = [m31, m20, m11, m02, m13][pos + 2]
        mn = [m20, m11, m02, m13, m24][pos + 2]
        mn1 = [m11, m02, m13, m24, 0.01][pos + 2]

        ### basic calculation ###
        bbool = 0.8 * bc * cb > bb * cc and 0.8 * bc * cn > bn * cc and bc * bc > cc
        blend = bbool and bc * 5 > cc and dbc + dcn > 1.5 * thr and (dbb < 7 * dbc or dbb < 8 * dcn) and (dnn < 8 * dcn or dnn < 7 * dbc) and (mb < mb1 and mb < mc or mn < mn1 and mn < mc or (dbb + dnn) * 4 < dbc + dcn or (bb * cc * 5 < bc * cb or mb > thr) and (bn * cc * 5 < bc * cn or mn > thr) and bc > thr)
        clear = dbb + dbc > thr and dcn + dnn > thr and (bc < 2 * bb or bc < 2 * bn) and (dbb + dnn) * 2 > dbc + dcn and (mc < 0.96 * mb and mc < 0.96 * mn and (bb * 2 > cb or bn * 2 > cn) and cc > cb and cc > cn or frfac > 0.45 and frfac < 0.55 and 0.8 * mc > mb1 and 0.8 * mc > mn1 and mb > 0.8 * mn and mn > 0.8 * mb)
        highd = dcn > 5 * dbc and dcn > 5 * dnn and dcn > thr and dbc < thr and dnn < thr
        lowd = dcn * 5 < dbc and dcn * 5 < dnn and dbc > thr and dnn > thr and dcn < thr and frfac > 0.35 and (frfac < 0.51 or dcn * 5 < dbb)
        res = d43 < thr and d32 < thr and d21 < thr and d10 < thr and d01 < thr and d12 < thr and d23 < thr and d34 < thr or dbc * 4 < dbb and dcn * 4 < dbb and dnn * 4 < dbb and dn2 * 4 < dbb or dcn * 4 < dbc and dnn * 4 < dbc and dn2 * 4 < dbc

        ### offset calculation ###
        if blend:
            odm = denm
        elif clear:
            odm = 0
        elif highd:
            odm = denm - numr
        elif lowd:
            odm = 2 * denm - numr
        else:
            odm = cof
        odm += cround((cof - odm) / (2 * denm)) * 2 * denm

        if blend:
            odr = denm - numr
        elif clear or highd:
            odr = numr
        elif frfac < 0.5:
            odr = 2 * numr
        else:
            odr = 2 * (denm - numr)
        odr *= 0.9

        if ldet >= 0:
            if cof > odm + odr:
                if cof - offs - odm - odr > denm and res:
                    cof = odm + 2 * denm - odr
                else:
                    cof = odm + odr
            elif cof < odm - odr:
                if offs > denm and res:
                    cof = odm - 2 * denm + odr
                else:
                    cof = odm - odr
            elif offs < -1.15 * denm and res:
                cof += 2 * denm
            elif offs > 1.25 * denm and res:
                cof -= 2 * denm

        offs = 0 if frfac == 1 else cof - cfo - 2 * numr * pos
        lpos = pos
        opos = 0 if frfac == 1 else -cround((cfo + offs + (denm if bfo and offs <= -4 * numr else 0)) / (2 * numr))
        pos = min(max(opos, -2), 2)

        ### frame output calculation - resync - dup ###
        dbb = [d43, d32, d21, d10, d01][pos + 2]
        dbc = [d32, d21, d10, d01, d12][pos + 2]
        dcn = [d21, d10, d01, d12, d23][pos + 2]
        dnn = [d10, d01, d12, d23, d34][pos + 2]

        ### dup_hq - merge ###
        if opos != pos or abs(mode) < 2 or abs(mode) == 3:
            dup = 0
        elif dcn * 5 < dbc and dnn * 5 < dbc and (dcn < 1.25 * thr or bn < bc and pos == lpos) or (dcn * dcn < dbc or dcn * 5 < dbc) and bn < bc and pos == lpos and dnn < 0.9 * dbc or dnn * 9 < dbc and dcn * 3 < dbc:
            dup = 1
        elif (dbc * dbc < dcn or dbc * 5 < dcn) and bb < bc and pos == lpos and dbb < 0.9 * dcn or dbb * 9 < dcn and dbc * 3 < dcn or dbb * 5 < dcn and dbc * 5 < dcn and (dbc < 1.25 * thr or bb < bc and pos == lpos):
            dup = -1
        else:
            dup = 0
        mer = not bom and opos == pos and dup == 0 and abs(mode) > 2 and (dbc * 8 < dcn or dbc * 8 < dbb or dcn * 8 < dbc or dcn * 8 < dnn or dbc * 2 < thr or dcn * 2 < thr or dnn * 9 < dbc and dcn * 3 < dbc or dbb * 9 < dcn and dbc * 3 < dcn)

        ### deblend - doubleblend removal - postprocessing ###
        add = bp1 * cn2 > bn2 * cp1 * (1 + thr * 0.01) and bn0 * cn2 > bn2 * cn0 * (1 + thr * 0.01) and cn2 * bn1 > cn1 * bn2 * (1 + thr * 0.01)
        if bom:
            if bn0 > bp2 and bn0 >= bp1 and bn0 > bn1 and bn0 > bn2 and cn0 < 125:
                if d12 * d12 < d10 or d12 * 9 < d10:
                    dup = 1
                elif d10 * d10 < d12 or d10 * 9 < d12:
                    dup = 0
                else:
                    dup = 4
            elif bp1 > bp3 and bp1 >= bp2 and bp1 > bn0 and bp1 > bn1:
                dup = 1
            else:
                dup = 0
        elif dup == 0:
            if omode > 0 and omode < 5:
                if not bbool:
                    dup = 0
                elif omode == 4 and bp1 * cn1 < bn1 * cp1 or omode == 3 and d10 < d01 or omode == 1:
                    dup = -1
                else:
                    dup = 1
            elif omode == 5:
                if bp1 * cp2 > bp2 * cp1 * (1 + thr * 0.01) and bn0 * cp2 > bp2 * cn0 * (1 + thr * 0.01) and cp2 * bn1 > cn1 * bp2 * (1 + thr * 0.01) and (not add or cp2 * bn2 > cn2 * bp2):
                    dup = -2
                elif add:
                    dup = 2
                elif bn0 * cp1 > bp1 * cn0 and (bn0 * cn1 < bn1 * cn0 or cp1 * bn1 > cn1 * bp1):
                    dup = -1
                elif bn0 * cn1 > bn1 * cn0:
                    dup = 1
                else:
                    dup = 0
            else:
                dup = 0

        ### output clip ###
        if dup == 4:
            return fin
        else:
            oclp = mec if mer and dup == 0 else source
            opos += dup - (1 if dup == 0 and mer and dbc < dcn else 0)
            if opos < 0:
                return oclp.std.DuplicateFrames(frames=[0] * -opos)
            else:
                return oclp.std.Trim(first=opos)

    ###### evaluation call & output calculation ######
    bclpYStats = bclp.std.PlaneStats()
    dclpYStats = dclp.std.PlaneStats()
    dclipYStats = core.std.PlaneStats(dclip, dclip.std.Trim(first=2))
    
    last = source.std.FrameEval(eval=srestore_inside, prop_src=[bclpYStats, dclpYStats, dclipYStats])

    last = core.std.Convolution(last, matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1], divisor=9)

    ###### final decimation ######
    return ChangeFPS.ChangeFPS(last, source.fps_num * numr, source.fps_den * denm)

def sRestoreMUVs(
    source: vs.VideoNode,
    frate: Optional[numbers.Real] = None,
    omode: int = 6,
    speed: Optional[int] = None,
    mode: int = 2,
    thresh: int = 16,
    dclip: Optional[vs.VideoNode] = None
) -> vs.VideoNode:

    """ srestore v2.7e
    srestore with serialized execution by explicit node processing dependency

    modified from havsfunc's srestore function
    https://github.com/HomeOfVapourSynthEvolution/havsfunc/blob/e236281cd8c1dd6b1b0cc906844944b79b1b52fa/havsfunc.py#L1899-L2227
    """

    if not isinstance(source, vs.VideoNode):
        raise vs.Error('srestore: this is not a clip')

    if source.format.color_family != vs.YUV:
        raise vs.Error('srestore: only YUV format is supported')

    if dclip is None:
        dclip = source
    elif not isinstance(dclip, vs.VideoNode):
        raise vs.Error("srestore: 'dclip' is not a clip")
    elif dclip.format.color_family != vs.YUV:
        raise vs.Error('srestore: only YUV format is supported')

    bits = source.format.bits_per_sample
    neutral = 1 << (bits - 1)
    peak = (1 << bits) - 1

    ###### parameters & other necessary vars ######
    srad = math.sqrt(abs(speed)) * 4 if speed is not None and abs(speed) >= 1 else 12
    irate = source.fps_num / source.fps_den
    bsize = 16 if speed is not None and speed > 0 else 32
    bom = isinstance(omode, str)
    thr = abs(thresh) + 0.01

    if bom or abs(omode - 3) < 2.5:
        frfac = 1
    elif frate is not None:
        if frate * 5 < irate or frate > irate:
            frfac = 1
        else:
            frfac = abs(frate) / irate
    elif cround(irate * 10010) % 30000 == 0:
        frfac = 1001 / 2400
    else:
        frfac = 480 / 1001

    if abs(frfac * 1001 - cround(frfac * 1001)) < 0.01:
        numr = cround(frfac * 1001)
    elif abs(1001 / frfac - cround(1001 / frfac)) < 0.01:
        numr = 1001
    else:
        numr = cround(frfac * 9000)
    if (
        frate is not None and
        abs(irate * numr / cround(numr / frfac) - frate) > abs(irate * cround(frate * 100) / cround(irate * 100) - frate)
    ):
        numr = cround(frate * 100)
    denm = cround(numr / frfac)

    ###### source preparation & lut ######
    if abs(mode) >= 2 and not bom:
        mec = core.std.Merge(source, source.std.Trim(first=1), weight=[0, 0.5])
        mec = core.std.Merge(mec, source.std.Trim(first=1), weight=[0.5, 0])

    if dclip.format.id != vs.YUV420P8:
        dclip = dclip.resize.Bicubic(format=vs.YUV420P8)
    dclip = dclip.resize.Point(
        dclip.width if srad == 4 else int(dclip.width / 2 / srad + 4) * 4,
        dclip.height if srad == 4 else int(dclip.height / 2 / srad + 4) * 4
    )
    dclip = dclip.std.Trim(first=2)
    if mode < 0:
        dclip = core.std.StackVertical([
            core.std.StackHorizontal([GetPlane(dclip, 1), GetPlane(dclip, 2)]),
            GetPlane(dclip, 0)
        ])
    else:
        dclip = GetPlane(dclip, 0)
    if bom:
        dclip = dclip.std.Expr(expr=['x 0.5 * 64 +'])

    expr1 = 'x 128 - y 128 - * 0 > x 128 - abs y 128 - abs < x 128 - 128 x - * y 128 - 128 y - * ? x y + 256 - dup * ? 0.25 * 128 +'
    expr2 = 'x y - dup * 3 * x y + 256 - dup * - 128 +'
    diff = core.std.MakeDiff(dclip, dclip.std.Trim(first=1))
    if not bom:
        bclp = core.std.Expr([diff, diff.std.Trim(first=1)], expr=[expr1]).resize.Bilinear(bsize, bsize)
    else:
        bclp = core.std.Expr([
            diff.std.Trim(first=1),
            core.std.MergeDiff(diff, diff.std.Trim(first=2))
        ], expr=[expr2])
        bclp = bclp.resize.Bilinear(bsize, bsize)
    dclp = diff.std.Trim(first=1).std.Lut(function=lambda x: max(cround(abs(x - 128) ** 1.1 - 1), 0))
    dclp = dclp.resize.Bilinear(bsize, bsize)

    ###### postprocessing ######
    if bom:
        sourceDuplicate = source.std.DuplicateFrames(frames=[0])
        sourceTrim1 = source.std.Trim(first=1)
        sourceTrim2 = source.std.Trim(first=2)

        unblend1 = core.std.Expr([sourceDuplicate, source], expr=['x -1 * y 2 * +'])
        unblend2 = core.std.Expr([sourceTrim1, sourceTrim2], expr=['x 2 * y -1 * +'])

        qmask1 = core.std.MakeDiff(
            unblend1.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1], planes=[0]),
            unblend1,
            planes=[0]
        )
        qmask2 = core.std.MakeDiff(
            unblend2.std.Convolution(matrix=[1, 1, 1, 1, 0, 1, 1, 1, 1], planes=[0]),
            unblend2,
            planes=[0]
        )
        diffm = core.std.MakeDiff(sourceDuplicate, source, planes=[0]).std.Maximum(planes=[0])
        bmask = core.std.Expr([qmask1, qmask2], expr=[f'x {neutral} - dup * dup y {neutral} - dup * + / {peak} *', ''])
        expr = (
            'x 2 * y < x {i} < and 0 y 2 * x < y {i} < and {peak} x x y + / {j} * {k} + ? ?'
            .format(i=scale(4, bits), peak=peak, j=scale(200, bits), k=scale(28, bits))
        )
        dmask = core.std.Expr([diffm, diffm.std.Trim(first=2)], expr=[expr, ''])
        pmask = core.std.Expr([dmask, bmask], expr=[f'y 0 > y {peak} < and x 0 = x {peak} = or and x y ?', ''])

        matrix = [1, 2, 1, 2, 4, 2, 1, 2, 1]

        omode = omode.lower()
        if omode == 'pp0':
            fin = core.std.Expr([sourceDuplicate, source, sourceTrim1, sourceTrim2], expr=['x -0.5 * y + z + a -0.5 * +'])
        elif omode == 'pp1':
            fin = core.std.MaskedMerge(
                unblend1,
                unblend2,
                dmask.std.Convolution(matrix=matrix, planes=[0]).std.Expr(expr=['', repr(neutral)])
            )
        elif omode == 'pp2':
            fin = core.std.MaskedMerge(
                unblend1,
                unblend2,
                bmask.std.Convolution(matrix=matrix, planes=[0]), first_plane=True
            )
        elif omode == 'pp3':
            fin = core.std.MaskedMerge(
                unblend1,
                unblend2,
                pmask.std.Convolution(matrix=matrix, planes=[0]), first_plane=True
            )
            fin = fin.std.Convolution(matrix=matrix, planes=[1, 2])
        else:
            raise vs.Error('srestore: unexpected value for omode')

    def srestore_inside(n: int, f: List[vs.VideoFrame], real_n: int) -> vs.VideoNode:
        n = real_n

        if n == 0:
            ###### initialise variables ######
            lfr = -100
            offs = 0
            ldet = -100
            lpos = 0
            d32 = d21 = d10 = d01 = d12 = d23 = d34 = None
            m42 = m31 = m20 = m11 = m02 = m13 = m24 = None
            bp2 = bp1 = bn0 = bn1 = bn2 = bn3 = None
            cp2 = cp1 = cn0 = cn1 = cn2 = cn3 = None
        else:
            state = f[3].props

            lfr = state["_lfr"]
            offs = state["_offs"]
            ldet = state["_ldet"]
            lpos = state["_lpos"]
            d32 = state.get("_d32")
            d21 = state.get("_d21")
            d10 = state.get("_d10")
            d01 = state.get("_d01")
            d12 = state.get("_d12")
            d23 = state.get("_d23")
            d34 = state.get("_d34")
            m42 = state.get("_m42")
            m31 = state.get("_m31")
            m20 = state.get("_m20")
            m11 = state.get("_m11")
            m02 = state.get("_m02")
            m13 = state.get("_m13")
            m24 = state.get("_m24")
            bp2 = state.get("_bp2")
            bp1 = state.get("_bp1")
            bn0 = state.get("_bn0")
            bn1 = state.get("_bn1")
            bn2 = state.get("_bn2")
            bn3 = state.get("_bn3")
            cp2 = state.get("_cp2")
            cp1 = state.get("_cp1")
            cn0 = state.get("_cn0")
            cn1 = state.get("_cn1")
            cn2 = state.get("_cn2")
            cn3 = state.get("_cn3")

        ### preparation ###
        jmp = lfr + 1 == n
        cfo = ((n % denm) * numr * 2 + denm + numr) % (2 * denm) - denm
        bfo = cfo > -numr and cfo <= numr
        lfr = n
        if bfo:
            if offs <= -4 * numr:
                offs = offs + 2 * denm
            elif offs >= 4 * numr:
                offs = offs - 2 * denm
        pos = 0 if frfac == 1 else -cround((cfo + offs) / (2 * numr)) if bfo else lpos
        cof = cfo + offs + 2 * numr * pos
        ldet = -1 if n + pos == ldet else n + pos

        ### diff value shifting ###
        d_v = f[1].props['PlaneStatsMax'] + 0.015625
        if jmp:
            d43 = d32
            d32 = d21
            d21 = d10
            d10 = d01
            d01 = d12
            d12 = d23
            d23 = d34
        else:
            d43 = d32 = d21 = d10 = d01 = d12 = d23 = d_v
        d34 = d_v

        m_v = f[2].props['PlaneStatsDiff'] * 255 + 0.015625 if not bom and abs(omode) > 5 else 1
        if jmp:
            m53 = m42
            m42 = m31
            m31 = m20
            m20 = m11
            m11 = m02
            m02 = m13
            m13 = m24
        else:
            m53 = m42 = m31 = m20 = m11 = m02 = m13 = m_v
        m24 = m_v

        ### get blend and clear values ###
        b_v = 128 - f[0].props['PlaneStatsMin']
        if b_v < 1:
            b_v = 0.125
        c_v = f[0].props['PlaneStatsMax'] - 128
        if c_v < 1:
            c_v = 0.125

        ### blend value shifting ###
        if jmp:
            bp3 = bp2
            bp2 = bp1
            bp1 = bn0
            bn0 = bn1
            bn1 = bn2
            bn2 = bn3
        else:
            bp3 = b_v - c_v if bom else b_v
            bp2 = bp1 = bn0 = bn1 = bn2 = bp3
        bn3 = b_v - c_v if bom else b_v

        ### clear value shifting ###
        if jmp:
            cp3 = cp2
            cp2 = cp1
            cp1 = cn0
            cn0 = cn1
            cn1 = cn2
            cn2 = cn3
        else:
            cp3 = cp2 = cp1 = cn0 = cn1 = cn2 = c_v
        cn3 = c_v

        ### used detection values ###
        bb = [bp3, bp2, bp1, bn0, bn1][pos + 2]
        bc = [bp2, bp1, bn0, bn1, bn2][pos + 2]
        bn = [bp1, bn0, bn1, bn2, bn3][pos + 2]

        cb = [cp3, cp2, cp1, cn0, cn1][pos + 2]
        cc = [cp2, cp1, cn0, cn1, cn2][pos + 2]
        cn = [cp1, cn0, cn1, cn2, cn3][pos + 2]

        dbb = [d43, d32, d21, d10, d01][pos + 2]
        dbc = [d32, d21, d10, d01, d12][pos + 2]
        dcn = [d21, d10, d01, d12, d23][pos + 2]
        dnn = [d10, d01, d12, d23, d34][pos + 2]
        dn2 = [d01, d12, d23, d34, d34][pos + 2]

        mb1 = [m53, m42, m31, m20, m11][pos + 2]
        mb = [m42, m31, m20, m11, m02][pos + 2]
        mc = [m31, m20, m11, m02, m13][pos + 2]
        mn = [m20, m11, m02, m13, m24][pos + 2]
        mn1 = [m11, m02, m13, m24, 0.01][pos + 2]

        ### basic calculation ###
        bbool = 0.8 * bc * cb > bb * cc and 0.8 * bc * cn > bn * cc and bc * bc > cc
        blend = (
            bbool and
            bc * 5 > cc and
            dbc + dcn > 1.5 * thr and
            (dbb < 7 * dbc or dbb < 8 * dcn) and
            (dnn < 8 * dcn or dnn < 7 * dbc) and
            (
                mb < mb1 and mb < mc or
                mn < mn1 and mn < mc or
                (dbb + dnn) * 4 < dbc + dcn or
                (bb * cc * 5 < bc * cb or mb > thr) and (bn * cc * 5 < bc * cn or mn > thr) and bc > thr
            )
        )
        clear = (
            dbb + dbc > thr and
            dcn + dnn > thr and
            (bc < 2 * bb or bc < 2 * bn) and
            (dbb + dnn) * 2 > dbc + dcn and
            (
                mc < 0.96 * mb and mc < 0.96 * mn and (bb * 2 > cb or bn * 2 > cn) and cc > cb and cc > cn or
                frfac > 0.45 and frfac < 0.55 and 0.8 * mc > mb1 and 0.8 * mc > mn1 and mb > 0.8 * mn and mn > 0.8 * mb
            )
        )
        highd = dcn > 5 * dbc and dcn > 5 * dnn and dcn > thr and dbc < thr and dnn < thr
        lowd = (
            dcn * 5 < dbc and
            dcn * 5 < dnn and
            dbc > thr and
            dnn > thr and
            dcn < thr and
            frfac > 0.35 and
            (frfac < 0.51 or dcn * 5 < dbb)
        )
        res = (
            d43 < thr and
            d32 < thr and
            d21 < thr and
            d10 < thr and
            d01 < thr and
            d12 < thr and
            d23 < thr and
            d34 < thr or

            dbc * 4 < dbb and
            dcn * 4 < dbb and
            dnn * 4 < dbb and
            dn2 * 4 < dbb or

            dcn * 4 < dbc and
            dnn * 4 < dbc and
            dn2 * 4 < dbc
        )

        ### offset calculation ###
        if blend:
            odm = denm
        elif clear:
            odm = 0
        elif highd:
            odm = denm - numr
        elif lowd:
            odm = 2 * denm - numr
        else:
            odm = cof
        odm += cround((cof - odm) / (2 * denm)) * 2 * denm

        if blend:
            odr = denm - numr
        elif clear or highd:
            odr = numr
        elif frfac < 0.5:
            odr = 2 * numr
        else:
            odr = 2 * (denm - numr)
        odr *= 0.9

        if ldet >= 0:
            if cof > odm + odr:
                if cof - offs - odm - odr > denm and res:
                    cof = odm + 2 * denm - odr
                else:
                    cof = odm + odr
            elif cof < odm - odr:
                if offs > denm and res:
                    cof = odm - 2 * denm + odr
                else:
                    cof = odm - odr
            elif offs < -1.15 * denm and res:
                cof += 2 * denm
            elif offs > 1.25 * denm and res:
                cof -= 2 * denm

        offs = 0 if frfac == 1 else cof - cfo - 2 * numr * pos
        lpos = pos
        if frfac == 1:
            opos = 0
        else:
            opos = -cround((cfo + offs + (denm if bfo and offs <= -4 * numr else 0)) / (2 * numr))
        pos = min(max(opos, -2), 2)

        ### frame output calculation - resync - dup ###
        dbb = [d43, d32, d21, d10, d01][pos + 2]
        dbc = [d32, d21, d10, d01, d12][pos + 2]
        dcn = [d21, d10, d01, d12, d23][pos + 2]
        dnn = [d10, d01, d12, d23, d34][pos + 2]

        ### dup_hq - merge ###
        if opos != pos or abs(mode) < 2 or abs(mode) == 3:
            dup = 0
        elif (
            dcn * 5 < dbc and dnn * 5 < dbc and (dcn < 1.25 * thr or bn < bc and pos == lpos) or
            (dcn * dcn < dbc or dcn * 5 < dbc) and bn < bc and pos == lpos and dnn < 0.9 * dbc or
            dnn * 9 < dbc and dcn * 3 < dbc
        ):
            dup = 1
        elif (
            (dbc * dbc < dcn or dbc * 5 < dcn) and bb < bc and pos == lpos and dbb < 0.9 * dcn or
            dbb * 9 < dcn and dbc * 3 < dcn or
            dbb * 5 < dcn and dbc * 5 < dcn and (dbc < 1.25 * thr or bb < bc and pos == lpos)
        ):
            dup = -1
        else:
            dup = 0

        mer = (
            not bom and
            opos == pos and
            dup == 0 and
            abs(mode) > 2 and
            (
                dbc * 8 < dcn or
                dbc * 8 < dbb or
                dcn * 8 < dbc or
                dcn * 8 < dnn or
                dbc * 2 < thr or
                dcn * 2 < thr or
                dnn * 9 < dbc and dcn * 3 < dbc or
                dbb * 9 < dcn and dbc * 3 < dcn
            )
        )

        ### deblend - doubleblend removal - postprocessing ###
        add = (
            bp1 * cn2 > bn2 * cp1 * (1 + thr * 0.01) and
            bn0 * cn2 > bn2 * cn0 * (1 + thr * 0.01) and
            cn2 * bn1 > cn1 * bn2 * (1 + thr * 0.01)
        )
        if bom:
            if bn0 > bp2 and bn0 >= bp1 and bn0 > bn1 and bn0 > bn2 and cn0 < 125:
                if d12 * d12 < d10 or d12 * 9 < d10:
                    dup = 1
                elif d10 * d10 < d12 or d10 * 9 < d12:
                    dup = 0
                else:
                    dup = 4
            elif bp1 > bp3 and bp1 >= bp2 and bp1 > bn0 and bp1 > bn1:
                dup = 1
            else:
                dup = 0
        elif dup == 0:
            if omode > 0 and omode < 5:
                if not bbool:
                    dup = 0
                elif omode == 4 and bp1 * cn1 < bn1 * cp1 or omode == 3 and d10 < d01 or omode == 1:
                    dup = -1
                else:
                    dup = 1
            elif omode == 5:
                if (
                    bp1 * cp2 > bp2 * cp1 * (1 + thr * 0.01) and
                    bn0 * cp2 > bp2 * cn0 * (1 + thr * 0.01) and
                    cp2 * bn1 > cn1 * bp2 * (1 + thr * 0.01) and
                    (not add or cp2 * bn2 > cn2 * bp2)
                ):
                    dup = -2
                elif add:
                    dup = 2
                elif bn0 * cp1 > bp1 * cn0 and (bn0 * cn1 < bn1 * cn0 or cp1 * bn1 > cn1 * bp1):
                    dup = -1
                elif bn0 * cn1 > bn1 * cn0:
                    dup = 1
                else:
                    dup = 0
            else:
                dup = 0

        ### output clip ###
        if dup == 4:
            ret = fin
        else:
            oclp = mec if mer and dup == 0 else source
            opos += dup - (1 if dup == 0 and mer and dbc < dcn else 0)
            if opos < 0:
                ret = oclp.std.DuplicateFrames(frames=[0] * -opos)
            else:
                ret = oclp.std.Trim(first=opos)

        ret = ret[n]

        temp_kwargs = dict(
            lfr=lfr, offs=offs, ldet=ldet, lpos=lpos,
            d32=d32, d21=d21, d10=d10, d01=d01, d12=d12, d23=d23, d34=d34,
            m42=m42, m31=m31, m20=m20, m11=m11, m02=m02, m13=m13, m24=m24,
            bp2=bp2, bp1=bp1, bn0=bn0, bn1=bn1, bn2=bn2, bn3=bn3,
            cp2=cp2, cp1=cp1, cn0=cn0, cn1=cn1, cn2=cn2, cn3=cn3
        )
        state_kwargs = {f"_{k}": v for k, v in temp_kwargs.items() if v is not None}

        if hasattr(core.std, "SetFrameProps"):
            return core.std.SetFrameProps(ret, **state_kwargs)
        else:
            for k, v in state_kwargs.items():
                ret = core.std.SetFrameProp(ret, prop=k, intval=v)
            return ret

    ###### evaluation call & output calculation ######
    bclpYStats = bclp.std.PlaneStats()
    dclpYStats = dclp.std.PlaneStats()
    dclipYStats = core.std.PlaneStats(dclip, dclip.std.Trim(first=2))

    # https://github.com/vapoursynth/vapoursynth/blob/55e7d0e989359c23782fc1e0d4aa1c0c35838a80/src/core/vsapi.cpp#L151-L152
    def get_frame(clip: vs.VideoNode, n: int) -> vs.VideoNode:
        return clip[min(n, clip.num_frames - 1)]

    last_frames: List[vs.VideoNode] = []
    state: vs.VideoNode

    for n in range(source.num_frames):
        prop_src = [
            get_frame(bclpYStats, n),
            get_frame(dclpYStats, n),
            get_frame(dclipYStats, n)
        ]

        if n > 0:
            prop_src.append(state)

        state = source[n].std.FrameEval(
            eval=functools.partial(srestore_inside, real_n=n),
            prop_src=prop_src
        )
        last_frames.append(state)

    last = core.std.Splice(last_frames)

    ###### final decimation ######
    return ChangeFPS.ChangeFPS(last, source.fps_num * numr, source.fps_den * denm)


def GetPlane(clip, plane=None):
    # input clip
    if not isinstance(clip, vs.VideoNode):
        raise type_error('"clip" must be a clip!')

    # Get properties of input clip
    sFormat = clip.format
    sNumPlanes = sFormat.num_planes

    # Parameters
    if plane is None:
        plane = 0
    elif not isinstance(plane, int):
        raise type_error('"plane" must be an int!')
    elif plane < 0 or plane > sNumPlanes:
        raise value_error(f'valid range of "plane" is [0, {sNumPlanes})!')

    # Process
    return core.std.ShufflePlanes(clip, plane, vs.GRAY)
    
    
def cround(x: float) -> int:
    return math.floor(x + 0.5) if x > 0 else math.ceil(x - 0.5)
        
def cround(x: float) -> int:
    return math.floor(x + 0.5) if x > 0 else math.ceil(x - 0.5)
    
def scale(value, peak):
    return cround(value * peak / 255) if peak != 1 else value / 255