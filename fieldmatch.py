import vapoursynth as vs
core = vs.core
import qtgmc

# FieldMatch: Experimental manual field matching function
#
# This function implements a custom, heuristic-based field matching algorithm.
#
# It generates multiple field-matching candidates by combining fields from
# the previous, current, and next frames. Each candidate is evaluated using
# a combing detection function (IsCombed), and the first non-combed result
# is selected in a predefined priority order.
#
# If all candidates are detected as combed (i.e. interlacing artifacts remain),
# the function falls back to QTGMC-based deinterlacing for that frame.
#
# Candidate priority order:
#   1. Current frame (cc)
#   2. Next frame (nn)
#   3. Current + Next (cn)
#   4. Next + Current (nc)
#   5. Previous frame (pp)
#   6. Current + Previous (cp)
#   7. Previous + Current (pc)
#   8. QTGMC fallback (deint)
#
# Notes:
# - This is not a full replacement for TFM and lacks motion analysis.
# - Performance may be lower due to FrameEval and multiple clip evaluations.
# - Quality depends heavily on IsCombed thresholds and source characteristics.
#
# Requirements:
# - tdm (for IsCombed)
# - QTGMC
#
# Author: no clue where I got this from originally 

def FieldMatch(c: vs.VideoNode) -> vs.VideoNode:
    pp = core.std.DuplicateFrames(c, frames=[0])
    cc = c
    nn = core.std.DeleteFrames(c, frames=[0])

    # Separate fields
    p2 = core.std.SeparateFields(pp)
    c2 = core.std.SeparateFields(cc)
    n2 = core.std.SeparateFields(nn)

    # Create field match candidates
    pc = core.std.DoubleWeave(core.std.Interleave([p2[::2], c2[1::2]])).std.SelectEvery(2, 0)
    cp = core.std.DoubleWeave(core.std.Interleave([c2[::2], p2[1::2]])).std.SelectEvery(2, 0)
    cn = core.std.DoubleWeave(core.std.Interleave([c2[::2], n2[1::2]])).std.SelectEvery(2, 0)
    nc = core.std.DoubleWeave(core.std.Interleave([n2[::2], c2[1::2]])).std.SelectEvery(2, 0)

    deint = qtgmc.QTGMC(cc, Preset="Fast").std.SelectEvery(2, 0)

    # List of all candidates in priority order
    candidates = [cc, nn, cn, nc, pp, cp, pc, deint]

    # Precompute IsCombed for each clip
    combed_flags = [core.tdm.IsCombed(clip, cthresh=12, chroma=True, blockx=16, blocky=32) for clip in candidates]

    def select_frame(n, f):
        for i in range(len(combed_flags) - 1):  # skip deint until end
            if not f[i].props._Combed:
                return candidates[i]
        return candidates[-1]  # fallback to deint if all are combed

    return core.std.FrameEval(cc, eval=lambda n: select_frame(n, [cf.get_frame(n) for cf in combed_flags]))


# Field matching implementation based on:
# https://forum.doom9.org/showthread.php?t=185104 (by Ceppo)
#
# Requirements:
#   vinverse: https://github.com/Asd-g/vinverse or https://github.com/Selur/VapoursynthScriptsInHybrid/blob/master/residual.py#L13
#   VapourSynth-VMAF: https://github.com/HomeOfVapourSynthEvolution/VapourSynth-VMAF
#   BoxBlur (optional): https://github.com/dnjulek/vapoursynth-zip/
def cFieldMatch(clip: vs.VideoNode, chroma: bool=False, vinverse: bool=False):
    # Determine field order from frame properties:
    # _FieldBased = 2 → Top Field First (TFF)
    # _FieldBased = 1 → Bottom Field First (BFF)
    field_based = clip.get_frame(0).props.get('_FieldBased', 0)
    tff = field_based == 2

    # Split clip into individual fields
    fields = core.std.SeparateFields(clip=clip, tff=tff)
    first  = fields[0::2]   # First field in display order (Bottom for BFF, Top for TFF)
    second = fields[1::2]   # Second field in display order

    # c: current match → combine fields from the same frame (i)
    c = core.std.DoubleWeave(clip=fields, tff=tff)[0::2]

    # n: next match → first[i] + second[i+1]
    second_next = second[1:] + second[-1:]  # replicate last field to maintain length
    n_fields = core.std.Interleave(clips=[first, second_next])
    n = core.std.DoubleWeave(clip=n_fields, tff=tff)[0::2]

    # p: previous match → first[i] + second[i-1]
    second_prev = second[0:1] + second[:-1]  # replicate first field
    p_fields = core.std.Interleave(clips=[first, second_prev])
    p = core.std.DoubleWeave(clip=p_fields, tff=tff)[0::2]

    # First frame of 'p' is invalid (no previous field available),
    # replace it with the corresponding frame from 'n'
    p = n[0:1] + p[1:]

    # Compute quality metric (PSNR via VMAF plugin)
    if vinverse:
        # Use vinverse as distortion source if available
        if hasattr(core, 'vinverse'):
            c_m = core.vmaf.Metric(reference=c, distorted=core.vinverse.vinverse(c), feature=[0])
            n_m = core.vmaf.Metric(reference=n, distorted=core.vinverse.vinverse(n), feature=[0])
            p_m = core.vmaf.Metric(reference=p, distorted=core.vinverse.vinverse(p), feature=[0])
        else:
            # Fallback implementation
            import residual
            c_m = core.vmaf.Metric(reference=c, distorted=residual.Vinverse(c), feature=[0])
            n_m = core.vmaf.Metric(reference=n, distorted=residual.Vinverse(n), feature=[0])
            p_m = core.vmaf.Metric(reference=p, distorted=residual.Vinverse(p), feature=[0])
    else:
        # Use slight blur as reference distortion if vinverse is disabled
        BOX = core.vszip.BoxBlur if hasattr(core, 'vszip') else core.std.BoxBlur
        c_m = core.vmaf.Metric(reference=c, distorted=BOX(c, hradius=0), feature=[0])
        n_m = core.vmaf.Metric(reference=n, distorted=BOX(n, hradius=0), feature=[0])
        p_m = core.vmaf.Metric(reference=p, distorted=BOX(p, hradius=0), feature=[0])

    out_clips  = [c, n, p]
    prop_clips = [c_m, n_m, p_m]

    def select(n, f):
        # Aggregate PSNR score:
        #   luma only (default) or luma + chroma if enabled
        A = f[0].props["psnr_y"] + (f[0].props["psnr_cb"] + f[0].props["psnr_cr"] if chroma else 0)
        B = f[1].props["psnr_y"] + (f[1].props["psnr_cb"] + f[1].props["psnr_cr"] if chroma else 0)
        C = f[2].props["psnr_y"] + (f[2].props["psnr_cb"] + f[2].props["psnr_cr"] if chroma else 0)

        # Select the best field match based on highest PSNR
        if B > A and B > C:
            return out_clips[1]   # next match (n)
        elif C > A:
            return out_clips[2]   # previous match (p)
        else:
            return out_clips[0]   # current match (c)

    # Per-frame selection using computed metrics
    return core.std.FrameEval(clip, eval=select, prop_src=prop_clips)
