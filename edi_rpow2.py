import vapoursynth as vs
import math

core = vs.core

def nnedi3_rpow2(clip, rfactor, correct_shift="fmtconv", nsize=0, nns=3, qual=None, etype=None, pscrn=None, opt=None,
                 int16_prescreener=None, int16_predictor=None, exp=None):
    """
    Scales clip by rfactor (power of two) using nnedi3 or znedi3 if available.
    """
    def edi(clip, field, dh):
        if hasattr(core, 'znedi3'):
            return core.znedi3.nnedi3(clip=clip, field=field, dh=dh, nsize=nsize, nns=nns, qual=qual, etype=etype,
                                      pscrn=pscrn, opt=opt, int16_prescreener=int16_prescreener,
                                      int16_predictor=int16_predictor, exp=exp)
        else:
            return core.nnedi3.nnedi3(clip=clip, field=field, dh=dh, nsize=nsize, nns=nns, qual=qual, etype=etype,
                                     pscrn=pscrn, opt=opt, int16_prescreener=int16_prescreener,
                                     int16_predictor=int16_predictor, exp=exp)

    return edi_rpow2(clip=clip, rfactor=rfactor, correct_shift=correct_shift, edi=edi)


def nnedi3cl_rpow2(clip, rfactor, correct_shift="fmtconv", nsize=0, nns=3, qual=None, etype=None, pscrn=None):
    """
    Scales clip by rfactor (power of two) using nnedi3cl or sneedif.
    """
    def edi(clip, field, dh):
        if hasattr(core, 'sneedif'):
          return sneedif_rpow2(clip, rfactor, correct_shift, nsize, nns, qual, etype, pscrn)
        return core.nnedi3cl.NNEDI3CL(clip=clip, field=field, dh=dh, nsize=nsize, nns=nns, qual=qual, etype=etype, pscrn=pscrn)

    return edi_rpow2(clip=clip, rfactor=rfactor, correct_shift=correct_shift, edi=edi)

def sneedif_rpow2(clip, rfactor, correct_shift="fmtconv", nsize=0, nns=3, qual=None, etype=None, pscrn=None):
    """
    Scales clip by rfactor (power of two) using sneedif.
    """
    def edi(clip, field, dh):
        return core.sneedif.NNEDI3(clip=clip, field=field, dh=dh, nsize=nsize, nns=nns, qual=qual, etype=etype, pscrn=pscrn)

    return edi_rpow2(clip=clip, rfactor=rfactor, correct_shift=correct_shift, edi=edi)


def edi_rpow2(clip, rfactor, correct_shift, edi):
    """
    Core function to scale clip by rfactor (power of two) using edi() function.
    """
    if not (rfactor != 0 and ((rfactor & (rfactor - 1)) == 0)):
        raise ValueError('rfactor must be a power of two')

    steps = int(math.log2(rfactor))

    if correct_shift not in [None, "fmtconv", "zimg"]:
        raise ValueError('correct_shift must be None, "fmtconv" or "zimg"')

    clip = core.std.Transpose(clip)
    for _ in range(steps):
        clip = edi(clip, field=1, dh=1)
    clip = core.std.Transpose(clip)
    clip = edi(clip, field=1, dh=1)
    for _ in range(steps - 1):
        clip = edi(clip, field=0, dh=1)

    if correct_shift in ("fmtconv", "zimg"):
        clip = correct_edi_shift(clip, rfactor=rfactor, plugin=correct_shift)

    return clip


def correct_edi_shift(clip, rfactor, plugin):
    """
    Corrects subpixel shift introduced by EDI scaling depending on plugin.
    """
    if clip.format.subsampling_w == 1:
        hshift = -rfactor / 2 + 0.5
    else:
        hshift = -0.5

    if plugin == "fmtconv":
        bits = clip.format.bits_per_sample
        if clip.format.subsampling_h == 0:
            clip = core.fmtc.resample(clip=clip, sx=hshift, sy=-0.5)
        else:
            clip = core.fmtc.resample(clip=clip, sx=hshift, sy=-0.5, planes=[3, 2, 2])
            clip = core.fmtc.resample(clip=clip, sx=hshift, sy=-1, planes=[2, 3, 3])
        if bits != 16:
            clip = core.fmtc.bitdepth(clip=clip, bits=bits)

    elif plugin == "zimg":
        if clip.format.subsampling_h == 0:
            clip = core.z.Subresize(clip=clip, resample_filter="spline36", width=clip.width, height=clip.height,
                                    shift_w=hshift, shift_h=-0.5)
        else:
            Y = core.std.ShufflePlanes(clips=clip, planes=0, colorfamily=vs.GRAY)
            U = core.std.ShufflePlanes(clips=clip, planes=1, colorfamily=vs.GRAY)
            V = core.std.ShufflePlanes(clips=clip, planes=2, colorfamily=vs.GRAY)
            Y = core.z.Subresize(clip=Y, resample_filter="spline36", width=clip.width, height=clip.height,
                                 shift_w=hshift, shift_h=-0.5)
            U = core.z.Subresize(clip=U, resample_filter="spline36", width=clip.width, height=clip.height,
                                 shift_w=hshift / 2, shift_h=-0.5)
            V = core.z.Subresize(clip=V, resample_filter="spline36", width=clip.width, height=clip.height,
                                 shift_w=hshift / 2, shift_h=-0.5)
            clip = core.std.ShufflePlanes(clips=[Y, U, V], planes=[0, 0, 0], colorfamily=vs.YUV)

    return clip
