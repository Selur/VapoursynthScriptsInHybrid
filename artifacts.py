from vapoursynth import core
import vapoursynth as vs

# VS port of a script by Didée http://forum.doom9.net/showthread.php?p=1402690#post1402690
# In my experience this filter works very good as a prefilter for SMDegrain(). 
# Filtering only luma seems to help to avoid ghost artefacts.
def DeSpot(o):
  osup = o.mv.Super(pel=2, sharp=2)
  bv1  = osup.mv.Analyse(isb=True, delta=1, blksize=8, overlap=4, search=4)
  fv1  = osup.mv.Analyse(isb=False,delta=1, blksize=8, overlap=4, search=4)
  bc1  = o.mv.Compensate(osup, bv1)
  fc1  = o.mv.Compensate(osup, fv1)
  
  clip = core.std.Interleave([fc1, o, bc1])
  
  if hasattr(core,'zsmooth'):
    clip = clip.zsmooth.Clense()
  else:
    clip = clip.rgvs.Clense()
  
  return clip.std.SelectEvery(cycle=3, offsets=1)

import vapoursynth as vs
core = vs.core

# Requires
# zsmooth: https://github.com/adworacz/zsmooth
# RemoveDirt: https://github.com/pinterf/RemoveDirt
def RemoveSpots(clip: vs.VideoNode, grey: bool = False, limit: int = 16) -> vs.VideoNode:
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
        - RemoveDirt (SCSelect, RestoreMotionBlocks) as 'removedirt' or 'rmd'
    
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
    clensed = core.zsmooth.Clense(clip, planes=planes)
    
    # Forward clense: temporal filter looking ahead
    sbegin  = core.zsmooth.ForwardClense(clip, planes=planes)
    
    # Backward clense: temporal filter looking behind
    send    = core.zsmooth.BackwardClense(clip, planes=planes)

    # Scene change detection: pick the best of the three temporal candidates
    # SCSelect avoids blending across scene boundaries
    if hasattr(core, 'removedirt') and hasattr(core.removedirt, 'SCSelect'):
        scenechange = core.removedirt.SCSelect(clip, sbegin, send, clensed)
        RESTORE     = core.removedirt.RestoreMotionBlocks
    else:
        scenechange = core.rmd.SCSelect(clip, sbegin, send, clensed)
        RESTORE     = core.rmd.RestoreMotionBlocks

    # Repair mode: how aggressively to replace pixels. 0 = no repair for that plane
    rep_mode = [limit if p in planes else 0 for p in range(clip.format.num_planes)]
    
    # Alternative restoration path using scene-change-selected frame
    alt     = core.zsmooth.Repair(scenechange, clip, mode=rep_mode)
    
    # Another restoration path using the temporal median
    restore = core.zsmooth.Repair(clensed, clip, mode=rep_mode)

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
def RemoveSpotsMCX(clip: vs.VideoNode, limit: int = 6, grey: bool = False, runs: int = 3) -> vs.VideoNode:
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
    sup  = core.mv.Super(clip, pel=2)

    # Analyse backward motion (next frame -> current)
    bvec = core.mv.Analyse(sup, isb=False, blksize=8, delta=1, truemotion=True)

    # Analyse forward motion (prev frame -> current)
    fvec = core.mv.Analyse(sup, isb=True,  blksize=8, delta=1, truemotion=True)

    # Motion-compensate: create frames warped to match current frame's motion
    backw = core.mv.Flow(clip, sup, bvec)  # Next frame compensated to current
    forw  = core.mv.Flow(clip, sup, fvec)  # Previous frame compensated to current

    # Interleave as [backward, source, forward] so RemoveSpots processes
    # a motion-compensated sequence
    clp = core.std.Interleave([backw, clip, forw])

    # Multi-pass spot removal on the motion-compensated interleaved clip
    for _ in range(runs):
        clp = RemoveSpots(clp, grey=grey, limit=limit)

    # Extract only the source frames (offset 1 in each 3-frame cycle)
    clp = core.std.SelectEvery(clp, cycle=3, offsets=[1])
    return clp