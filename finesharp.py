# finesharp.py - finesharp module for VapourSynth
# Original author: Didee (http://forum.doom9.org/showthread.php?t=166082)
from vapoursynth import core
import vapoursynth as vs


def spline(x, coordinates):
    def get_matrix(px, py, l):
        matrix = []
        matrix.append([(i == 0) * 1.0 for i in range(l + 1)])
        for i in range(1, l - 1):
            p = [0 for t in range(l + 1)]
            p[i - 1] = px[i] - px[i - 1]
            p[i] = 2 * (px[i + 1] - px[i - 1])
            p[i + 1] = px[i + 1] - px[i]
            p[l] = 6 * (((py[i + 1] - py[i]) / p[i + 1]) - (py[i] - py[i - 1]) / p[i - 1])
            matrix.append(p)
        matrix.append([(i == l - 1) * 1.0 for i in range(l + 1)])
        return matrix

    def equation(matrix, dim):
        for i in range(dim):
            num = matrix[i][i]
            for j in range(dim + 1):
                matrix[i][j] /= num
            for j in range(dim):
                if i != j:
                    a = matrix[j][i]
                    for k in range(i, dim + 1):
                        matrix[j][k] -= a * matrix[i][k]

    if not isinstance(coordinates, dict):
        raise TypeError('coordinates must be a dict')

    length = len(coordinates)
    if length < 3:
        raise ValueError('coordinates require at least three pairs')

    px = [key for key in coordinates.keys()]
    py = [val for val in coordinates.values()]
    matrix = get_matrix(px, py, length)
    equation(matrix, length)
    for i in range(length + 1):
        if x >= px[i] and x <= px[i + 1]:
            break
    j = i + 1
    h = px[j] - px[i]
    s = matrix[j][length] * (x - px[i]) ** 3
    s -= matrix[i][length] * (x - px[j]) ** 3
    s /= 6 * h
    s += (py[j] / h - h * matrix[j][length] / 6) * (x - px[i])
    s -= (py[i] / h - h * matrix[i][length] / 6) * (x - px[j])
    return s


def sharpen(clip, mode=1, sstr=2.0, cstr=None, xstr=0.19, lstr=1.49, pstr=1.272, ldmp=None):
    """Small and relatively fast realtime-sharpening function, for 1080p,
    or after scaling 720p -> 1080p during playback
    (to make 720p look more like being 1080p)
    It's a generic sharpener. Only for good quality sources!
    (If the source is crap, FineSharp will happily sharpen the crap.) ;)
    Noise/grain will be enhanced, too. The method is GENERIC.

    Modus operandi: A basic nonlinear sharpening method is performed,
    then the *blurred* sharp-difference gets subtracted again.

    Args:
        clip (clip): YUV or GRAY video.
        mode (int): 1 to 3, weakest to strongest. When negative -1 to -3,
                a broader kernel for equalisation is used.
        sstr (float): strength of sharpening, 0.0 up to ??
        cstr (float): strength of equalisation, 0.0 to ??
                (recomm. 0.5 to 1.25, default AUTO)
        xstr (float): strength of XSharpen-style final sharpening, 0.0 to 1.0
                (but, better don't go beyond 0.249 ...)
        lstr (float): modifier for non-linear sharpening
        pstr (float): exponent for non-linear sharpening
        ldmp (float): "low damp", to not overenhance very small differences
                (noise coming out of flat areas, default sstr + 0.1)
    """
    bd = clip.format.bits_per_sample
    mid = 1 << (bd - 1)
    scl = 1 << (bd - 8)
    x = 'x {} /'.format(scl)
    y = 'y {} /'.format(scl)

    src = clip

    if core.version_number() < 33:
        raise EnvironmentError('VapourSynth version should be 33 or greater.')

    if src.format.color_family != vs.YUV and src.format.color_family != vs.GRAY:
        raise ValueError('clip must be YUV or GRAY color family.')

    if bd < 8 or bd > 16:
        raise ValueError('clip must be 8..16 bits.')

    mode = int(mode)
    if abs(mode) > 3 or mode == 0:
        raise ValueError('mode must be 1, 2, 3, -1, -2 or -3.')

    sstr = float(sstr)
    if sstr < 0.0:
        raise ValueError('sstr must be positive.')

    if src.format.color_family != vs.GRAY:
        clip = core.std.ShufflePlanes(clips=clip, planes=0, colorfamily=vs.GRAY)

    if cstr is None:
        cstr = spline(sstr, {0: 0, 0.5: 0.1, 1: 0.6, 2: 0.9, 2.5: 1, 3: 1.09,
                             3.5: 1.15, 4: 1.19, 8: 1.249, 255: 1.5})
        if mode > 0:
            cstr **= 0.8
    cstr = float(cstr)

    xstr = float(xstr)
    if xstr < 0.0:
        raise ValueError('xstr must be positive.')

    lstr = float(lstr)

    pstr = float(pstr)

    if ldmp is None:
      ldmp = sstr + 0.1
    else:
      ldmp = float(ldmp)
        
    if sstr < 0.01 and cstr < 0.01 and xstr < 0.01:
        return src

    if abs(mode) == 1:
        c2 = core.std.Convolution(clip, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1]).std.Median()
    elif abs(mode) == 2:
        c2 = core.std.Median(clip).std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])
    else:
        c2 = core.std.Median(clip).std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1]).std.Median()

    expr = '{x} {y} - abs {lstr} / 1 {pstr} / pow {sstr} * {x} {y} - {x} {y} - abs 0.001 + / * {x} {y} - 2 pow {x} {y} - 2 pow {ldmp} + / * 128 + {scl} *'
    diff = core.std.Expr(clips=[clip, c2], expr=expr.format(x=x, y=y, lstr=lstr, pstr=pstr, sstr=sstr, ldmp=ldmp, scl=scl))

    shrp = clip
    if sstr >= 0.01:
        shrp = core.std.MergeDiff(clipa=shrp, clipb=diff)

    if cstr >= 0.01:
        expr = 'x {mid} - {cstr} * {mid} +'.format(mid=mid, cstr=cstr)
        diff = core.std.Expr(clips=diff, expr=expr)
        diff = core.std.Convolution(clip=diff, matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1]) if mode > 0 else core.std.Convolution(clip=diff, matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])
        shrp = core.std.MakeDiff(clipa=shrp, clipb=diff)

    if xstr >= 0.01:
        expr = 'x x y - 9.9 * +'
        xyshrp = core.std.Expr(clips=[shrp, core.std.Convolution(shrp, matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1])], expr=expr)
        if hasattr(core, 'zsmooth'):
          rpshrp = core.zsmooth.Repair(clip=xyshrp, repairclip=shrp, mode=[12])
        else:
          rpshrp = core.rgvs.Repair(clip=xyshrp, repairclip=shrp, mode=[12])
        shrp = core.std.Merge(clipa=shrp, clipb=rpshrp, weight=[xstr])

    if src.format.color_family != vs.GRAY:
        shrp = core.std.ShufflePlanes(clips=[shrp, src], planes=[0, 1, 2], colorfamily=src.format.color_family)

    return shrp
