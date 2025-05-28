import vapoursynth as vs
core = vs.core

def addDup(clip: vs.VideoNode, thresh: float = 0.3, debug: bool = False) -> vs.VideoNode:
    """
    Detects frames with low temporal difference compared to their predecessor and duplicates the previous frame
    if the difference is below the given threshold. Optionally adds debug metadata.

    Parameters:
        clip (vs.VideoNode): Input video clip.
        thresh (float): PlaneStatsDiff threshold for duplication.
        debug (bool): If True, adds debug props to output frames.

    Returns:
        vs.VideoNode: Modified clip with duplicates inserted when differences are below threshold.
    """
    if clip.format.color_family != vs.YUV and clip.format.color_family != vs.GRAY:
        raise ValueError("addDup: Only YUV or GRAY input clips are supported.")

    # Create shifted versions of the clip for frame difference calculation
    prev = core.std.Trim(clip, 0, clip.num_frames - 1)  # Frame N-1
    curr = core.std.Trim(clip, 1, clip.num_frames)      # Frame N

    # Calculate framewise difference between consecutive frames
    diffclip = core.std.PlaneStats(curr, prev)

    def selectFunc(n: int, f: vs.VideoFrame) -> vs.VideoNode:
        diff = f.props['PlaneStatsDiff']
        if diff > thresh:
            return clip.std.SetFrameProps(_DupApplied=False, _Diff=diff)
        else:
            modified = addingDups(clip, n + 1)  # n+1 because diffclip starts from frame 1
            return modified.std.SetFrameProps(_DupApplied=True, _Diff=diff)

    # Apply the frame selection logic using FrameEval, aligned with original clip
    result = core.std.FrameEval(curr, selectFunc, prop_src=diffclip)

    # Prepend the first frame which has no previous for comparison
    first_frame = core.std.Trim(clip, 0, 1).std.SetFrameProps(_DupApplied=False, _Diff=1.0)
    return core.std.Splice([first_frame, result], mismatch=True)


def addingDups(clip: vs.VideoNode, insert_at: int) -> vs.VideoNode:
    """
    Duplicates the frame just before `insert_at` and inserts it at `insert_at`, pushing the rest forward.

    Parameters:
        clip (vs.VideoNode): Input clip.
        insert_at (int): Index at which to insert a duplicate of the previous frame.

    Returns:
        vs.VideoNode: Clip with duplicate frame inserted.
    """
    if insert_at <= 0 or insert_at >= clip.num_frames:
        return clip

    pre = core.std.Trim(clip, 0, insert_at - 1)           # Frames before insertion
    dup = core.std.Trim(clip, insert_at - 1, insert_at - 1)  # Duplicate of previous frame
    post = core.std.Trim(clip, insert_at, None)           # Frames after insertion

    return core.std.Splice([pre, dup, post], mismatch=True)


def addingMultipleDups(clip: vs.VideoNode, positions: list[int]) -> vs.VideoNode:
    """
    Inserts duplicate frames after specified positions. Duplicated frames are copies of the preceding frame.

    Parameters:
        clip (vs.VideoNode): Input clip.
        positions (list[int]): Frame indices after which to insert duplicates.

    Returns:
        vs.VideoNode: Modified clip with duplicates inserted.
    """
    # Sort descending to avoid shifting issues while inserting
    positions = sorted(set(positions), reverse=True)

    for pos in positions:
        if pos < 0 or pos >= clip.num_frames - 1:
            raise ValueError(f"Invalid duplicate insertion position: {pos}")
        clip = addingDups(clip, pos + 1)

    return clip
