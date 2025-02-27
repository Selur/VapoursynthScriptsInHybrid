import vapoursynth as vs
from vapoursynth import core

from typing import Optional, TypeVar
from functools import partial

T = TypeVar('T')

# Taken from old havsfunc
def daa(
    c: vs.VideoNode,
    nsize: Optional[int] = None,
    nns: Optional[int] = None,
    qual: Optional[int] = None,
    pscrn: Optional[int] = None,
    int16_prescreener: Optional[bool] = None,
    int16_predictor: Optional[bool] = None,
    exp: Optional[int] = None,
    opencl: bool = False,
    device: Optional[int] = None,
) -> vs.VideoNode:
    '''
    Anti-aliasing with contra-sharpening by Didée.

    It averages two independent interpolations, where each interpolation set works between odd-distanced pixels.
    This on its own provides sufficient amount of blurring. Enough blurring that the script uses a contra-sharpening step to counteract the blurring.
    '''
    if not isinstance(c, vs.VideoNode):
        raise vs.Error('daa: this is not a clip')

    if opencl:
        nnedi3 = partial(core.nnedi3cl.NNEDI3CL, nsize=nsize, nns=nns, qual=qual, pscrn=pscrn, device=device)
    else:
        if hasattr(core,'znedi3'):
          nnedi3 = partial(core.znedi3.nnedi3, nsize=nsize, nns=nns, qual=qual, pscrn=pscrn, int16_prescreener=int16_prescreener, int16_predictor=int16_predictor, exp=exp)
        else:
          nnedi3 = partial(core.nnedi3.nnedi3, nsize=nsize, nns=nns, qual=qual, pscrn=pscrn, int16_prescreener=int16_prescreener, int16_predictor=int16_predictor, exp=exp)

    nn = nnedi3(c, field=3)
    dbl = core.std.Merge(nn[::2], nn[1::2])
    dblD = core.std.MakeDiff(c, dbl)
    shrpD = core.std.MakeDiff(dbl, dbl.std.Convolution(matrix=[1, 1, 1, 1, 1, 1, 1, 1, 1] if c.width > 1100 else [1, 2, 1, 2, 4, 2, 1, 2, 1]))
    DD = core.rgvs.Repair(shrpD, dblD, mode=13)
    return core.std.MergeDiff(dbl, DD)

# Taken from old havsfunc
def santiag(
    c: vs.VideoNode,
    strh: int = 1,
    strv: int = 1,
    type: str = 'nnedi3',
    nsize: Optional[int] = None,
    nns: Optional[int] = None,
    qual: Optional[int] = None,
    pscrn: Optional[int] = None,
    int16_prescreener: Optional[bool] = None,
    int16_predictor: Optional[bool] = None,
    exp: Optional[int] = None,
    aa: Optional[int] = None,
    alpha: Optional[float] = None,
    beta: Optional[float] = None,
    gamma: Optional[float] = None,
    nrad: Optional[int] = None,
    mdis: Optional[int] = None,
    vcheck: Optional[int] = None,
    fw: Optional[int] = None,
    fh: Optional[int] = None,
    halfres: bool = False,
    typeh: Optional[str] = None,
    typev: Optional[str] = None,
    opencl: bool = False,
    device: Optional[int] = None,
) -> vs.VideoNode:
    '''
    santiag v1.6
    Simple antialiasing

    type = "nnedi3", "eedi2", "eedi3" or "sangnom"
    '''

    def santiag_dir(c: vs.VideoNode, strength: int, type: str, fw: Optional[int] = None, fh: Optional[int] = None) -> vs.VideoNode:
        fw = fallback(fw, c.width)
        fh = fallback(fh, c.height)

        c = santiag_stronger(c, strength, type)

        return c.resize.Spline36(fw, fh, src_top=0 if halfres else 0.5)

    def santiag_stronger(c: vs.VideoNode, strength: int, type: str) -> vs.VideoNode:
        if opencl:
            nnedi3 = partial(core.nnedi3cl.NNEDI3CL, nsize=nsize, nns=nns, qual=qual, pscrn=pscrn, device=device)
            if hasattr(core, 'EEDI3CL'):
              eedi3 = partial(core.eedi3m.EEDI3CL, alpha=alpha, beta=beta, gamma=gamma, nrad=nrad, mdis=mdis, vcheck=vcheck, device=device)
            else:
              eedi3 = partial(core.eedi3m.EEDI3, alpha=alpha, beta=beta, gamma=gamma, nrad=nrad, mdis=mdis, vcheck=vcheck)
        else:
            if hasattr(core, 'znedi3'):
              nnedi3 = partial(core.znedi3.nnedi3, nsize=nsize, nns=nns, qual=qual, pscrn=pscrn, int16_prescreener=int16_prescreener, int16_predictor=int16_predictor, exp=exp)
            else:
              nnedi3 = partial(core.nnedi3.nnedi3, nsize=nsize, nns=nns, qual=qual, pscrn=pscrn, int16_prescreener=int16_prescreener, int16_predictor=int16_predictor, exp=exp)
            
            if hasattr(core, 'EEDI3CL'):
              eedi3 = partial(core.eedi3m.EEDI3CL, alpha=alpha, beta=beta, gamma=gamma, nrad=nrad, mdis=mdis, vcheck=vcheck, device=device)
            else:
              eedi3 = partial(core.eedi3m.EEDI3, alpha=alpha, beta=beta, gamma=gamma, nrad=nrad, mdis=mdis, vcheck=vcheck)

        strength = max(strength, 0)
        field = strength % 2
        dh = strength <= 0 and not halfres

        if strength > 0:
            c = santiag_stronger(c, strength - 1, type)

        w = c.width
        h = c.height

        if type == 'nnedi3':
            return nnedi3(c, field=field, dh=dh)
        elif type == 'eedi2':
            if not dh:
                c = c.resize.Point(w, h // 2, src_top=1 - field)
            return c.eedi2.EEDI2(field=field)
        elif type == 'eedi3':
            sclip = nnedi3(c, field=field, dh=dh)
            return eedi3(c, field=field, dh=dh, sclip=sclip)
        elif type == 'sangnom':
            if dh:
                c = c.resize.Spline36(w, h * 2, src_top=-0.25)
            return c.sangnom.SangNom(order=field + 1, aa=aa)
        else:
            raise vs.Error('santiag: unexpected value for type')

    if not isinstance(c, vs.VideoNode):
        raise vs.Error('santiag: this is not a clip')

    type = type.lower()
    typeh = type if typeh is None else typeh.lower()
    typev = type if typev is None else typev.lower()

    w = c.width
    h = c.height
    fwh = fw if strv < 0 else w
    fhh = fh if strv < 0 else h

    if strh >= 0:
        c = santiag_dir(c, strh, typeh, fwh, fhh)
    if strv >= 0:
        c = santiag_dir(c.std.Transpose(), strv, typev, fh, fw).std.Transpose()

    fw = fallback(fw, w)
    fh = fallback(fh, h)
    if strh < 0 and strv < 0:
        c = c.resize.Spline36(fw, fh)
    return c

# Taken from sfrom vsutil
def fallback(value: Optional[T], fallback_value: T) -> T:
    """Utility function that returns a value or a fallback if the value is ``None``.

    >>> fallback(5, 6)
    5
    >>> fallback(None, 6)
    6

    :param value:           Argument that can be ``None``.
    :param fallback_value:  Fallback value that is returned if `value` is ``None``.

    :return:                The input `value` or `fallback_value` if `value` is ``None``.
    """
    return fallback_value if value is None else value