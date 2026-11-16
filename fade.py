from __future__ import annotations
import vapoursynth as vs
core = vs.core

# one black frame in the clip's format; limited-range integer YUV/GRAY gets video black (16), limited=None reads _Range of frame 0
def _black(inputClip: vs.VideoNode, limited: bool | None = None) -> vs.VideoNode:
	fmt = inputClip.format
	if fmt.sample_type != vs.INTEGER or fmt.color_family not in (vs.YUV, vs.GRAY):
		return core.std.BlankClip(clip=inputClip, length=1)
	if limited is None:
		props = inputClip.get_frame(0).props
		limited = props.get('_Range', props.get('_ColorRange', 1)) == 1
	shift = fmt.bits_per_sample - 8
	color = [16 << shift if limited else 0]
	if fmt.color_family == vs.YUV:
		color += [128 << shift, 128 << shift]
	return core.std.BlankClip(clip=inputClip, length=1, color=color)

# based on: http://forum.doom9.org/archive/index.php/t-165771-p-4.html
def fadeout(inputClip: vs.VideoNode, fadeframes: int, limited: bool | None = None) -> vs.VideoNode:
	beginframes = inputClip.num_frames-1 - fadeframes
	blank = _black(inputClip, limited)
	fade_frames = []
	for i in range(inputClip.num_frames):
		if (i <= beginframes) :
			fade_frames.append(inputClip[i])
		else :
			fade_frames.append(core.std.Merge(clipa=inputClip[i], clipb=blank, weight=[(i-beginframes)/fadeframes]))
	fade_output = core.std.Splice(clips=fade_frames)
	return fade_output

def fadein(inputClip: vs.VideoNode, fadeframes: int, limited: bool | None = None) -> vs.VideoNode:
	blank = _black(inputClip, limited)
	fade_frames = []
	for i in range(inputClip.num_frames):
		if (i > fadeframes) :
			fade_frames.append(inputClip[i])
		else :
			fade_frames.append(core.std.Merge(clipa=inputClip[i], clipb=blank, weight=[(fadeframes-i)/fadeframes]))
	fade_output = core.std.Splice(clips=fade_frames)
	return fade_output
