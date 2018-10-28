import vapoursynth as vs
core = vs.core

# based on: http://forum.doom9.org/archive/index.php/t-165771-p-4.html
def fadeout(inputClip, fadeframes):
	beginframes = inputClip.num_frames-1 - fadeframes
	blank = core.std.BlankClip(clip=inputClip, length=1)
	fade_frames = []
	for i in range(inputClip.num_frames):
		if (i <= beginframes) :
			fade_frames.append(inputClip[i])
		else :
			fade_frames.append(core.std.Merge(clipa=inputClip[i], clipb=blank, weight=[(i-beginframes)/fadeframes]))
	fade_output = core.std.Splice(clips=fade_frames)
	return fade_output

def fadein(inputClip, fadeframes):
	blank = core.std.BlankClip(clip=inputClip, length=1)
	fade_frames = []
	for i in range(inputClip.num_frames):
		if (i > fadeframes) :
			fade_frames.append(inputClip[i])
		else :
			fade_frames.append(core.std.Merge(clipa=inputClip[i], clipb=blank, weight=[(fadeframes-i)/fadeframes]))
	fade_output = core.std.Splice(clips=fade_frames)
	return fade_output