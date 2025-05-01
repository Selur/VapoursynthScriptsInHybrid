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
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    if mode < 0:
        dclip = core.std.StackVertical([
            core.std.StackHorizontal([GetPlane(dclip, 1), GetPlane(dclip, 2)]),
            GetPlane(dclip, 0)
        ])
    else:
        dclip = GetPlane(dclip, 0)
    if bom:
        dclip = EXPR(dclip, expr=['x 0.5 * 64 +'])

    expr1 = 'x 128 - y 128 - * 0 > x 128 - abs y 128 - abs < x 128 - 128 x - * y 128 - 128 y - * ? x y + 256 - dup * ? 0.25 * 128 +'
    expr2 = 'x y - dup * 3 * x y + 256 - dup * - 128 +'
    diff = core.std.MakeDiff(dclip, dclip.std.Trim(first=1))
    if not bom:
        bclp = EXPR([diff, diff.std.Trim(first=1)], expr=[expr1]).resize.Bilinear(bsize, bsize)
    else:
        bclp = EXPR([
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

        unblend1 = EXPR([sourceDuplicate, source], expr=['x -1 * y 2 * +'])
        unblend2 = EXPR([sourceTrim1, sourceTrim2], expr=['x 2 * y -1 * +'])

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
        bmask = EXPR([qmask1, qmask2], expr=[f'x {neutral} - dup * dup y {neutral} - dup * + / {peak} *', ''])
        expr = (
            'x 2 * y < x {i} < and 0 y 2 * x < y {i} < and {peak} x x y + / {j} * {k} + ? ?'
            .format(i=scale(4, bits), peak=peak, j=scale(200, bits), k=scale(28, bits))
        )
        dmask = EXPR([diffm, diffm.std.Trim(first=2)], expr=[expr, ''])
        pmask = EXPR([dmask, bmask], expr=[f'y 0 > y {peak} < and x 0 = x {peak} = or and x y ?', ''])

        matrix = [1, 2, 1, 2, 4, 2, 1, 2, 1]

        omode = omode.lower()
        if omode == 'pp0':
            fin = EXPR([sourceDuplicate, source, sourceTrim1, sourceTrim2], expr=['x -0.5 * y + z + a -0.5 * +'])
        elif omode == 'pp1':
            fin = core.std.MaskedMerge(
                unblend1,
                unblend2,
                EXPR(dmask.std.Convolution(matrix=matrix, planes=[0]), expr=['', repr(neutral)])
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

    class SRestoreState:
        def __init__(self):
            self.lfr = -100
            self.offs = 0
            self.ldet = -100
            self.lpos = 0
            self.d32 = None
            self.d21 = None
            self.d10 = None
            self.d01 = None
            self.d12 = None
            self.d23 = None
            self.d34 = None
            self.m42 = None
            self.m31 = None
            self.m20 = None
            self.m11 = None
            self.m02 = None
            self.m13 = None
            self.m24 = None
            self.bp2 = None
            self.bp1 = None
            self.bn0 = None
            self.bn1 = None
            self.bn2 = None
            self.bn3 = None
            self.cp2 = None
            self.cp1 = None
            self.cn0 = None
            self.cn1 = None
            self.cn2 = None
            self.cn3 = None

        def update_from_frame(self, f: vs.VideoFrame):
            if f is not None and hasattr(f, 'props'):
                props = f.props
                self.lfr = props.get("_lfr", self.lfr)
                self.offs = props.get("_offs", self.offs)
                self.ldet = props.get("_ldet", self.ldet)
                self.lpos = props.get("_lpos", self.lpos)
                self.d32 = props.get("_d32", self.d32)
                self.d21 = props.get("_d21", self.d21)
                self.d10 = props.get("_d10", self.d10)
                self.d01 = props.get("_d01", self.d01)
                self.d12 = props.get("_d12", self.d12)
                self.d23 = props.get("_d23", self.d23)
                self.d34 = props.get("_d34", self.d34)
                self.m42 = props.get("_m42", self.m42)
                self.m31 = props.get("_m31", self.m31)
                self.m20 = props.get("_m20", self.m20)
                self.m11 = props.get("_m11", self.m11)
                self.m02 = props.get("_m02", self.m02)
                self.m13 = props.get("_m13", self.m13)
                self.m24 = props.get("_m24", self.m24)
                self.bp2 = props.get("_bp2", self.bp2)
                self.bp1 = props.get("_bp1", self.bp1)
                self.bn0 = props.get("_bn0", self.bn0)
                self.bn1 = props.get("_bn1", self.bn1)
                self.bn2 = props.get("_bn2", self.bn2)
                self.bn3 = props.get("_bn3", self.bn3)
                self.cp2 = props.get("_cp2", self.cp2)
                self.cp1 = props.get("_cp1", self.cp1)
                self.cn0 = props.get("_cn0", self.cn0)
                self.cn1 = props.get("_cn1", self.cn1)
                self.cn2 = props.get("_cn2", self.cn2)
                self.cn3 = props.get("_cn3", self.cn3)

        def to_frame_props(self) -> dict:
            props = {
                "_lfr": self.lfr,
                "_offs": self.offs,
                "_ldet": self.ldet,
                "_lpos": self.lpos
            }
            if self.d32 is not None: props["_d32"] = self.d32
            if self.d21 is not None: props["_d21"] = self.d21
            if self.d10 is not None: props["_d10"] = self.d10
            if self.d01 is not None: props["_d01"] = self.d01
            if self.d12 is not None: props["_d12"] = self.d12
            if self.d23 is not None: props["_d23"] = self.d23
            if self.d34 is not None: props["_d34"] = self.d34
            if self.m42 is not None: props["_m42"] = self.m42
            if self.m31 is not None: props["_m31"] = self.m31
            if self.m20 is not None: props["_m20"] = self.m20
            if self.m11 is not None: props["_m11"] = self.m11
            if self.m02 is not None: props["_m02"] = self.m02
            if self.m13 is not None: props["_m13"] = self.m13
            if self.m24 is not None: props["_m24"] = self.m24
            if self.bp2 is not None: props["_bp2"] = self.bp2
            if self.bp1 is not None: props["_bp1"] = self.bp1
            if self.bn0 is not None: props["_bn0"] = self.bn0
            if self.bn1 is not None: props["_bn1"] = self.bn1
            if self.bn2 is not None: props["_bn2"] = self.bn2
            if self.bn3 is not None: props["_bn3"] = self.bn3
            if self.cp2 is not None: props["_cp2"] = self.cp2
            if self.cp1 is not None: props["_cp1"] = self.cp1
            if self.cn0 is not None: props["_cn0"] = self.cn0
            if self.cn1 is not None: props["_cn1"] = self.cn1
            if self.cn2 is not None: props["_cn2"] = self.cn2
            if self.cn3 is not None: props["_cn3"] = self.cn3
            return props

    def srestore_inside(n: int, f: List[vs.VideoFrame], real_n: int, state_obj: SRestoreState) -> vs.VideoNode:
        n = real_n

        if n > 0:
            state_obj.update_from_frame(f[3])

        ### preparation ###
        jmp = state_obj.lfr + 1 == n
        cfo = ((n % denm) * numr * 2 + denm + numr) % (2 * denm) - denm
        bfo = cfo > -numr and cfo <= numr
        state_obj.lfr = n
        if bfo:
            if state_obj.offs <= -4 * numr:
                state_obj.offs = state_obj.offs + 2 * denm
            elif state_obj.offs >= 4 * numr:
                state_obj.offs = state_obj.offs - 2 * denm
        pos = 0 if frfac == 1 else -cround((cfo + state_obj.offs) / (2 * numr)) if bfo else state_obj.lpos
        cof = cfo + state_obj.offs + 2 * numr * pos
        state_obj.ldet = -1 if n + pos == state_obj.ldet else n + pos

        ### diff value shifting ###
        d_v = f[1].props['PlaneStatsMax'] + 0.015625
        if jmp:
            d43 = state_obj.d32
            state_obj.d32 = state_obj.d21
            state_obj.d21 = state_obj.d10
            state_obj.d10 = state_obj.d01
            state_obj.d01 = state_obj.d12
            state_obj.d12 = state_obj.d23
            state_obj.d23 = state_obj.d34
        else:
            d43 = state_obj.d32 = state_obj.d21 = state_obj.d10 = state_obj.d01 = state_obj.d12 = state_obj.d23 = d_v
        state_obj.d34 = d_v

        m_v = f[2].props['PlaneStatsDiff'] * 255 + 0.015625 if not bom and abs(omode) > 5 else 1
        if jmp:
            m53 = state_obj.m42
            state_obj.m42 = state_obj.m31
            state_obj.m31 = state_obj.m20
            state_obj.m20 = state_obj.m11
            state_obj.m11 = state_obj.m02
            state_obj.m02 = state_obj.m13
            state_obj.m13 = state_obj.m24
        else:
            m53 = state_obj.m42 = state_obj.m31 = state_obj.m20 = state_obj.m11 = state_obj.m02 = state_obj.m13 = m_v
        state_obj.m24 = m_v

        ### get blend and clear values ###
        b_v = 128 - f[0].props['PlaneStatsMin']
        if b_v < 1:
            b_v = 0.125
        c_v = f[0].props['PlaneStatsMax'] - 128
        if c_v < 1:
            c_v = 0.125

        ### blend value shifting ###
        if jmp:
            bp3 = state_obj.bp2
            state_obj.bp2 = state_obj.bp1
            state_obj.bp1 = state_obj.bn0
            state_obj.bn0 = state_obj.bn1
            state_obj.bn1 = state_obj.bn2
            state_obj.bn2 = state_obj.bn3
        else:
            bp3 = b_v - c_v if bom else b_v
            state_obj.bp2 = state_obj.bp1 = state_obj.bn0 = state_obj.bn1 = state_obj.bn2 = bp3
        state_obj.bn3 = b_v - c_v if bom else b_v

        ### clear value shifting ###
        if jmp:
            cp3 = state_obj.cp2
            state_obj.cp2 = state_obj.cp1
            state_obj.cp1 = state_obj.cn0
            state_obj.cn0 = state_obj.cn1
            state_obj.cn1 = state_obj.cn2
            state_obj.cn2 = state_obj.cn3
        else:
            cp3 = state_obj.cp2 = state_obj.cp1 = state_obj.cn0 = state_obj.cn1 = state_obj.cn2 = c_v
        state_obj.cn3 = c_v

        ### used detection values ###
        bb = [bp3, state_obj.bp2, state_obj.bp1, state_obj.bn0, state_obj.bn1][pos + 2]
        bc = [state_obj.bp2, state_obj.bp1, state_obj.bn0, state_obj.bn1, state_obj.bn2][pos + 2]
        bn = [state_obj.bp1, state_obj.bn0, state_obj.bn1, state_obj.bn2, state_obj.bn3][pos + 2]

        cb = [cp3, state_obj.cp2, state_obj.cp1, state_obj.cn0, state_obj.cn1][pos + 2]
        cc = [state_obj.cp2, state_obj.cp1, state_obj.cn0, state_obj.cn1, state_obj.cn2][pos + 2]
        cn = [state_obj.cp1, state_obj.cn0, state_obj.cn1, state_obj.cn2, state_obj.cn3][pos + 2]

        dbb = [d43, state_obj.d32, state_obj.d21, state_obj.d10, state_obj.d01][pos + 2]
        dbc = [state_obj.d32, state_obj.d21, state_obj.d10, state_obj.d01, state_obj.d12][pos + 2]
        dcn = [state_obj.d21, state_obj.d10, state_obj.d01, state_obj.d12, state_obj.d23][pos + 2]
        dnn = [state_obj.d10, state_obj.d01, state_obj.d12, state_obj.d23, state_obj.d34][pos + 2]
        dn2 = [state_obj.d01, state_obj.d12, state_obj.d23, state_obj.d34, state_obj.d34][pos + 2]

        mb1 = [m53, state_obj.m42, state_obj.m31, state_obj.m20, state_obj.m11][pos + 2]
        mb = [state_obj.m42, state_obj.m31, state_obj.m20, state_obj.m11, state_obj.m02][pos + 2]
        mc = [state_obj.m31, state_obj.m20, state_obj.m11, state_obj.m02, state_obj.m13][pos + 2]
        mn = [state_obj.m20, state_obj.m11, state_obj.m02, state_obj.m13, state_obj.m24][pos + 2]
        mn1 = [state_obj.m11, state_obj.m02, state_obj.m13, state_obj.m24, 0.01][pos + 2]

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
            state_obj.d32 < thr and
            state_obj.d21 < thr and
            state_obj.d10 < thr and
            state_obj.d01 < thr and
            state_obj.d12 < thr and
            state_obj.d23 < thr and
            state_obj.d34 < thr or

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

        if state_obj.ldet >= 0:
            if cof > odm + odr:
                if cof - state_obj.offs - odm - odr > denm and res:
                    cof = odm + 2 * denm - odr
                else:
                    cof = odm + odr
            elif cof < odm - odr:
                if state_obj.offs > denm and res:
                    cof = odm - 2 * denm + odr
                else:
                    cof = odm - odr
            elif state_obj.offs < -1.15 * denm and res:
                cof += 2 * denm
            elif state_obj.offs > 1.25 * denm and res:
                cof -= 2 * denm

        state_obj.offs = 0 if frfac == 1 else cof - cfo - 2 * numr * pos
        state_obj.lpos = pos
        if frfac == 1:
            opos = 0
        else:
            opos = -cround((cfo + state_obj.offs + (denm if bfo and state_obj.offs <= -4 * numr else 0)) / (2 * numr))
        pos = min(max(opos, -2), 2)

        ### frame output calculation - resync - dup ###
        dbb = [d43, state_obj.d32, state_obj.d21, state_obj.d10, state_obj.d01][pos + 2]
        dbc = [state_obj.d32, state_obj.d21, state_obj.d10, state_obj.d01, state_obj.d12][pos + 2]
        dcn = [state_obj.d21, state_obj.d10, state_obj.d01, state_obj.d12, state_obj.d23][pos + 2]
        dnn = [state_obj.d10, state_obj.d01, state_obj.d12, state_obj.d23, state_obj.d34][pos + 2]

        ### dup_hq - merge ###
        if opos != pos or abs(mode) < 2 or abs(mode) == 3:
            dup = 0
        elif (
            dcn * 5 < dbc and dnn * 5 < dbc and (dcn < 1.25 * thr or bn < bc and pos == state_obj.lpos) or
            (dcn * dcn < dbc or dcn * 5 < dbc) and bn < bc and pos == state_obj.lpos and dnn < 0.9 * dbc or
            dnn * 9 < dbc and dcn * 3 < dbc
        ):
            dup = 1
        elif (
            (dbc * dbc < dcn or dbc * 5 < dcn) and bb < bc and pos == state_obj.lpos and dbb < 0.9 * dcn or
            dbb * 9 < dcn and dbc * 3 < dcn or
            dbb * 5 < dcn and dbc * 5 < dcn and (dbc < 1.25 * thr or bb < bc and pos == state_obj.lpos)
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
            state_obj.bp1 * state_obj.cn2 > state_obj.bn2 * state_obj.cp1 * (1 + thr * 0.01) and
            state_obj.bn0 * state_obj.cn2 > state_obj.bn2 * state_obj.cn0 * (1 + thr * 0.01) and
            state_obj.cn2 * state_obj.bn1 > state_obj.cn1 * state_obj.bn2 * (1 + thr * 0.01)
        )
        if bom:
            if state_obj.bn0 > state_obj.bp2 and state_obj.bn0 >= state_obj.bp1 and state_obj.bn0 > state_obj.bn1 and state_obj.bn0 > state_obj.bn2 and state_obj.cn0 < 125:
                if state_obj.d12 * state_obj.d12 < state_obj.d10 or state_obj.d12 * 9 < state_obj.d10:
                    dup = 1
                elif state_obj.d10 * state_obj.d10 < state_obj.d12 or state_obj.d10 * 9 < state_obj.d12:
                    dup = 0
                else:
                    dup = 4
            elif state_obj.bp1 > bp3 and state_obj.bp1 >= state_obj.bp2 and state_obj.bp1 > state_obj.bn0 and state_obj.bp1 > state_obj.bn1:
                dup = 1
            else:
                dup = 0
        elif dup == 0:
            if omode > 0 and omode < 5:
                if not bbool:
                    dup = 0
                elif omode == 4 and state_obj.bp1 * state_obj.cn1 < state_obj.bn1 * state_obj.cp1 or omode == 3 and state_obj.d10 < state_obj.d01 or omode == 1:
                    dup = -1
                else:
                    dup = 1
            elif omode == 5:
                if (
                    state_obj.bp1 * state_obj.cp2 > state_obj.bp2 * state_obj.cp1 * (1 + thr * 0.01) and
                    state_obj.bn0 * state_obj.cp2 > state_obj.bp2 * state_obj.cn0 * (1 + thr * 0.01) and
                    state_obj.cp2 * state_obj.bn1 > state_obj.cn1 * state_obj.bp2 * (1 + thr * 0.01) and
                    (not add or state_obj.cp2 * state_obj.bn2 > state_obj.cn2 * state_obj.bp2)
                ):
                    dup = -2
                elif add:
                    dup = 2
                elif state_obj.bn0 * state_obj.cp1 > state_obj.bp1 * state_obj.cn0 and (state_obj.bn0 * state_obj.cn1 < state_obj.bn1 * state_obj.cn0 or state_obj.cp1 * state_obj.bn1 > state_obj.cn1 * state_obj.bp1):
                    dup = -1
                elif state_obj.bn0 * state_obj.cn1 > state_obj.bn1 * state_obj.cn0:
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

        props = state_obj.to_frame_props()
        if hasattr(core.std, "SetFrameProps"):
            return core.std.SetFrameProps(ret, **props)
        else:
            for k, v in props.items():
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
    state_obj = SRestoreState()

    for n in range(source.num_frames):
        prop_src = [
            get_frame(bclpYStats, n),
            get_frame(dclpYStats, n),
            get_frame(dclipYStats, n)
        ]

        if n > 0:
            prop_src.append(last_frames[-1])

        frame = source[n].std.FrameEval(
            eval=functools.partial(srestore_inside, real_n=n, state_obj=state_obj),
            prop_src=prop_src
        )
        last_frames.append(frame)

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