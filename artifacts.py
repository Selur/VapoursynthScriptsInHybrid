from __future__ import annotations
from typing import Mapping, Optional, Sequence, Tuple
from vapoursynth import core
import vapoursynth as vs

from misc import get_mv
from helpers import get_expr, pick_tool, tool_function

# VS port of a script by Didée http://forum.doom9.net/showthread.php?p=1402690#post1402690
# In my experience this filter works very good as a prefilter for SMDegrain(). 
# Filtering only luma seems to help to avoid ghost artefacts.
def DeSpot(o: vs.VideoNode, tools: Optional[Mapping[str, str]] = None) -> vs.VideoNode:
  MV = get_mv(tools)
  osup = MV.Super(o, pel=2, sharp=2, blksize=8, overlap=4)
  bv1  = MV.Analyse(osup, isb=True, delta=1, blksize=8, overlap=4, search=4)
  fv1  = MV.Analyse(osup, isb=False,delta=1, blksize=8, overlap=4, search=4)
  bc1  = MV.Compensate(o, osup, bv1)
  fc1  = MV.Compensate(o, osup, fv1)

  clip = core.std.Interleave([fc1, o, bc1])

  clip = tool_function(tools, 'rg', 'Clense')(clip)
    
  return clip.std.SelectEvery(cycle=3, offsets=1)

import vapoursynth as vs
core = vs.core

# Requires
# zsmooth: https://github.com/adworacz/zsmooth
# RemoveDirt: https://github.com/pinterf/RemoveDirt
def RemoveSpots(clip: vs.VideoNode, grey: bool = False, limit: int = 16, tools: Optional[Mapping[str, str]] = None) -> vs.VideoNode:
    """
    Temporal spot/dirt removal filter using zsmooth and RemoveDirt.
    
    Detects and removes transient spots, dirt, and impulse noise by comparing
    forward/backward temporal neighbours and applying motion-aware restoration.
    This is effective for film restoration and cleaning up analogue captures.
    
    Args:
        clip: Input clip. Should be 8-bit or higher.
        grey: If True, process luma only (planes=[0]). If False, process all planes.
        limit: Repair mode strength for zsmooth.Repair. Higher values = more aggressive
               replacement of detected spots. Default 16 is a balanced starting point.
               Lower values preserve more original detail but may miss spots.
    
    Returns:
        vs.VideoNode: Cleaned clip with spots and transient dirt removed.
    
    Dependencies:
        - zsmooth (Clense, ForwardClense, BackwardClense, Repair)
        - RemoveDirt (SCSelect, RestoreMotionBlocks) as 'removedirt' or 'rdvs'
        tools['rg'] and tools['removedirt'] pick the implementations.
    
    Notes:
        - Uses a 3-frame temporal window (prev, current, next) for detection.
        - SCSelect chooses the best candidate between forward/backward clensed
          frames and the temporal median (clensed).
        - RestoreMotionBlocks does the final motion-compensated restoration,
          using the original clip as a neighbour reference and repair-filtered
          versions as alternatives.
    """
    planes = [0] if grey else [0, 1, 2]
    
    # Temporal median of 3 frames: reduces spots that appear on single frames
    clensed = tool_function(tools, 'rg', 'Clense')(clip, planes=planes)

    # Forward clense: temporal filter looking ahead
    sbegin  = tool_function(tools, 'rg', 'ForwardClense')(clip, planes=planes)

    # Backward clense: temporal filter looking behind
    send    = tool_function(tools, 'rg', 'BackwardClense')(clip, planes=planes)

    # Scene change detection: pick the best of the three temporal candidates
    # SCSelect avoids blending across scene boundaries
    removedirt = core.removedirt if pick_tool(tools, 'removedirt', ('removedirt', 'rdvs')) == 'removedirt' else core.rdvs
    scenechange = removedirt.SCSelect(clip, sbegin, send, clensed)
    RESTORE     = removedirt.RestoreMotionBlocks

    # Repair mode: how aggressively to replace pixels. 0 = no repair for that plane
    rep_mode = [limit if p in planes else 0 for p in range(clip.format.num_planes)]
    
    # Alternative restoration path using scene-change-selected frame
    alt     = tool_function(tools, 'rg', 'Repair')(scenechange, clip, mode=rep_mode)

    # Another restoration path using the temporal median
    restore = tool_function(tools, 'rg', 'Repair')(clensed, clip, mode=rep_mode)

    # Final motion-aware restoration:
    # - clensed: temporal median as base
    # - restore: repaired temporal median
    # - neighbour=clip: original frame for motion reference
    # - alternative=alt: scenechange-repaired frame as fallback
    corrected = RESTORE(
        clensed, restore,
        neighbour=clip,
        alternative=alt,
        gmthreshold=70,   # Global motion threshold (higher = more tolerant of motion)
        dist=1,           # Spatial distance for block matching
        dmode=2,          # Degrain mode / restoration mode
        noise=10,         # Noise threshold for detection
        noisy=12,         # Noise threshold for restoration
        grey=grey,
    )
    return corrected


# Requires
# zsmooth: https://github.com/adworacz/zsmooth
# RemoveDirt: https://github.com/pinterf/RemoveDirt
# mvtools: https://github.com/Mr-Z-2697/vapoursynth-mvtools
def RemoveSpotsMCX(clip: vs.VideoNode, limit: int = 6, grey: bool = False, runs: int = 3, tools: Optional[Mapping[str, str]] = None) -> vs.VideoNode:
    """
    Motion-compensated temporal spot removal using mvtools + RemoveSpots.

    Creates motion-compensated forward and backward references, interleaves them
    with the source, then applies RemoveSpots `runs` times for aggressive cleaning.
    This version is much stronger than plain RemoveSpots and handles motion better,
    but is significantly slower and may soften fine detail.

    Args:
        clip: Input clip.
        limit: Repair mode strength for RemoveSpots. Default 6 is lower than the
               standalone RemoveSpots default (16) because the multi-pass and
               motion compensation already provide significant cleaning.
        grey: If True, process luma only. Saves speed on YUV sources.
        runs: Number of RemoveSpots passes. Default 3.

    Returns:
        vs.VideoNode: Heavily cleaned clip with motion-compensated spot removal.

    Dependencies:
        - mvtools (Super, Analyse, Flow)
        - zsmooth (via RemoveSpots)
        - RemoveDirt (via RemoveSpots)

    Notes:
        - pel=2: Half-pixel precision motion vectors. Good balance of quality/speed.
        - blksize=8: Small block size for fine motion detail.
        - truemotion=True: Uses smoother, more realistic motion estimation.
        - The clip is interleaved as [backward, source, forward] so RemoveSpots
          sees motion-compensated neighbours instead of raw temporal neighbours.
        - RemoveSpots is applied `runs` times because the interleaved pattern means
          each frame is processed in context with its motion-compensated neighbours.
        - SelectEvery(cycle=3, offsets=[1]) extracts only the original source
          frames after processing, discarding the motion-compensated helpers.

    Performance:
        - Much slower than RemoveSpots due to mvtools motion estimation and
          multi-pass filtering. Best used for difficult sources where plain
          RemoveSpots leaves visible spots.
    """
    # Create superclip for motion estimation at half-pixel precision
    MV = get_mv(tools)
    sup  = MV.Super(clip, pel=2, blksize=8, overlap=4)

    # Analyse backward motion (next frame -> current)
    bvec = MV.Analyse(sup, isb=False, blksize=8, overlap=4, delta=1, truemotion=True)

    # Analyse forward motion (prev frame -> current)
    fvec = MV.Analyse(sup, isb=True,  blksize=8, overlap=4, delta=1, truemotion=True)

    # Motion-compensate: create frames warped to match current frame's motion
    backw = MV.Flow(clip, sup, bvec)  # Next frame compensated to current
    forw  = MV.Flow(clip, sup, fvec)  # Previous frame compensated to current

    # Interleave as [backward, source, forward] so RemoveSpots processes
    # a motion-compensated sequence
    clp = core.std.Interleave([backw, clip, forw])

    # Multi-pass spot removal on the motion-compensated interleaved clip
    for _ in range(runs):
        clp = RemoveSpots(clp, grey=grey, limit=limit, tools=tools)

    # Extract only the source frames (offset 1 in each 3-frame cycle)
    clp = core.std.SelectEvery(clp, cycle=3, offsets=[1])
    return clp

# DeScratch on an amplified horizontal high-pass of the luma: finds faint vertical scratches below the grain; the repair is scaled back and added at full depth.
def DescratchHighPass(clip: vs.VideoNode, gain: float = 6.0, hradius: int = 8, passes: Sequence[Tuple[int, int, int]] = ((4, 3, 3), (4, 7, 3)),
                      limit: float = 4.0, blurlen: int = 100, minlen: int = 300, maxlen: Optional[int] = None, maxgap: Optional[int] = None,
                      maxangle: Optional[float] = None, asym: Optional[int] = None, keep: Optional[int] = None, border: Optional[int] = None,
                      minwidth: Optional[int] = None, mark: bool = False, tools: Optional[Mapping[str, str]] = None) -> vs.VideoNode:
    # passes: (mindif, maxwidth, modey) per DeScratch call on the amplified copy, modey 1 dark, 2 light, 3 both; limit: largest correction in 8-bit levels; mark: luma shows the amplified copy with the detections drawn in, format unchanged
    fmt = clip.format
    if fmt.color_family not in (vs.YUV, vs.GRAY):
        raise vs.Error('DescratchHighPass: YUV or GRAY input required')
    if gain <= 0 or hradius < 1 or not passes:
        raise vs.Error('DescratchHighPass: gain > 0, hradius >= 1 and at least one pass required')
    if not hasattr(core, 'descratch'):
        raise vs.Error('DescratchHighPass: needs the DeScratch plugin')
    expr = get_expr(tools)
    boxblur = tool_function(tools, 'boxblur', 'BoxBlur')
    # one 8-bit level in the units of the clip
    level = 1.0 / 255 if fmt.sample_type == vs.FLOAT else float(1 << (fmt.bits_per_sample - 8))
    luma = core.std.ShufflePlanes(clip, 0, vs.GRAY) if fmt.color_family == vs.YUV else clip
    low = boxblur(luma, hradius=hradius, hpasses=2, vradius=0)
    amp = expr([luma, low], f'x y - {gain} * {level} / 128 +', format=vs.GRAY8)
    fixed = core.std.ShufflePlanes([amp, amp, amp], [0, 0, 0], vs.YUV)
    optional = {'maxlen': maxlen, 'maxgap': maxgap, 'maxangle': maxangle, 'asym': asym, 'keep': keep, 'border': border, 'minwidth': minwidth}
    args = {k: v for k, v in optional.items() if v is not None}
    for mindif, maxwidth, modey in passes:
        fixed = core.descratch.DeScratch(fixed, mindif=mindif, maxwidth=maxwidth, modey=modey, modeu=0, modev=0, blurlen=blurlen,
                                         minlen=minlen, mark=int(mark), **args)
    fixed = core.std.ShufflePlanes(fixed, 0, vs.GRAY)
    if mark:
        out = core.resize.Point(fixed, format=luma.format.id, range_in=0, range=0)
    else:
        lim = limit * level
        out = expr([luma, fixed, amp], f'x y z - {level} * {gain} / {-lim} max {lim} min +')
    if fmt.color_family == vs.GRAY:
        return out
    return core.std.ShufflePlanes([out, clip], [0, 1, 2], vs.YUV)
