import vapoursynth as vs
# collection of Mask filters.

# Use retinex to greatly improve the accuracy of the edge detection in dark scenes.
# draft=True is a lot faster, albeit less accurate
# from https://blog.kageru.moe/legacy/edgemasks.html
def retinex_edgemask(src: vs.VideoNode, sigma: int=1, draft: bool=False) -> vs.VideoNode:
    core = vs.core
    import mvsfunc as mvf
    src = mvf.Depth(src, 16)
    luma = mvf.GetPlane(src, 0)
    if draft:
        ret = core.std.Expr(luma, 'x 65535 / sqrt 65535 *')
    else:
        ret = core.retinex.MSRCP(luma, sigma=[50, 200, 350], upper_thr=0.005)
    mask = core.std.Expr([kirsch(luma), ret.tcanny.TCanny(mode=1, sigma=sigma).std.Minimum(coordinates=[1, 0, 1, 0, 0, 1, 0, 1])], 'x y +')
    return mask


# Kirsch edge detection. This uses 8 directions, so it's slower but better than Sobel (4 directions).
# more information: https://ddl.kageru.moe/konOJ.pdf
# from https://blog.kageru.moe/legacy/edgemasks.html
def kirsch(src: vs.VideoNode) -> vs.VideoNode:
    core = vs.core
    w = [5]*3 + [-3]*5
    weights = [w[-i:] + w[:-i] for i in range(4)]
    c = [src.std.Convolution((w[:4]+[0]+w[4:]), saturate=False) for w in weights]
    return core.std.Expr(c, 'x y max z max a max')


# should behave similar to std.Sobel() but faster since it has no additional high-/lowpass or gain.
# the internal filter is also a little brighter
# from https://blog.kageru.moe/legacy/edgemasks.html
def fast_sobel(src: vs.VideoNode) -> vs.VideoNode:
    core = vs.core
    sx = src.std.Convolution([-1, -2, -1, 0, 0, 0, 1, 2, 1], saturate=False)
    sy = src.std.Convolution([-1, 0, 1, -2, 0, 2, -1, 0, 1], saturate=False)
    return core.std.Expr([sx, sy], 'x y max')


# a weird kind of edgemask that draws around the edges. probably needs more tweaking/testing
# maybe useful for edge cleaning?
# from https://blog.kageru.moe/legacy/edgemasks.html
def bloated_edgemask(src: vs.VideoNode) -> vs.VideoNode:
    return src.std.Convolution(matrix=[1,  2,  4,  2, 1,
                                       2, -3, -6, -3, 2,
                                       4, -6,  0, -6, 4,
                                       2, -3, -6, -3, 2,
                                       1,  2,  4,  2, 1], saturate=False)

# https://github.com/DeadNews/dnfunc/blob/f5d22057e424fb3b8bd80d1aadd0c2ed2b7e71d5/dnfunc.py#L1212                                                                              
def kirsch2(clip_y: vs.VideoNode) -> vs.VideoNode:
    core = vs.core
    n = core.std.Convolution(clip_y, [5, 5, 5, -3, 0, -3, -3, -3, -3], divisor=3, saturate=False)
    nw = core.std.Convolution(clip_y, [5, 5, -3, 5, 0, -3, -3, -3, -3], divisor=3, saturate=False)
    w = core.std.Convolution(clip_y, [5, -3, -3, 5, 0, -3, 5, -3, -3], divisor=3, saturate=False)
    sw = core.std.Convolution(clip_y, [-3, -3, -3, 5, 0, -3, 5, 5, -3], divisor=3, saturate=False)
    s = core.std.Convolution(clip_y, [-3, -3, -3, -3, 0, -3, 5, 5, 5], divisor=3, saturate=False)
    se = core.std.Convolution(clip_y, [-3, -3, -3, -3, 0, 5, -3, 5, 5], divisor=3, saturate=False)
    e = core.std.Convolution(clip_y, [-3, -3, 5, -3, 0, 5, -3, -3, 5], divisor=3, saturate=False)
    ne = core.std.Convolution(clip_y, [-3, 5, 5, -3, 0, 5, -3, -3, -3], divisor=3, saturate=False)
    return core.std.Expr(
        [n, nw, w, sw, s, se, e, ne],
        ["x y max z max a max b max c max d max e max"],
    )
# from https://github.com/theChaosCoder/lostfunc/blob/master/lostfunc.py -> mfToon2/MfTurd
def scale8(x, newmax):
        return x * newmax // 0xFF

def CartoonEdges(clip, low=0, high=255):
    core = vs.core
    """Should behave like mt_edge(mode="cartoon")"""
    valuerange = (1 << clip.format.bits_per_sample)
    maxvalue = valuerange - 1
    
    low = scale8(low, maxvalue)
    high = scale8(high, maxvalue)
    edges = core.std.Convolution(clip, matrix=[0,-2,1,0,1,0,0,0,0], saturate=True)
    return core.std.Expr(edges, ['x {high} >= {maxvalue} x {low} <= 0 x ? ?'
                                 .format(low=low, high=high, maxvalue=maxvalue), ''])

def RobertsEdges(clip, low=0, high=255):
    core = vs.core
    """Should behave like mt_edge(mode="roberts")"""
    valuerange = (1 << clip.format.bits_per_sample)
    maxvalue = valuerange - 1
    
    low = scale8(low, maxvalue)
    high = scale8(high, maxvalue)
    edges = core.std.Convolution(clip, matrix=[0,0,0,0,2,-1,0,-1,0], divisor=2, saturate=False)
    return core.std.Expr(edges, ['x {high} >= {maxvalue} x {low} <= 0 x ? ?'
                                 .format(low=low, high=high, maxvalue=maxvalue), ''])
