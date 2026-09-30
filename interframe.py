from __future__ import annotations
import re
from typing import Optional
from vapoursynth import core
import vapoursynth as vs


def _option(text: str, key: str, default: int) -> int:
    match = re.search(r'(?<![A-Za-z_])"?%s"?\s*:\s*(\d+)' % key, text)
    return int(match.group(1)) if match else default


def _is_open_svpflow() -> bool:
    # open-svpflow registers through API 4, SVPflow through API 3 (no return signature).
    return getattr(core.svp2.SmoothFps, 'return_signature', 'any') != 'any'


def _cpu_render_padding(height: int, SuperString: str, VectorsString: str) -> int:
    # open-svpflow's CPU renderer panics on heights that are no multiple of the vertical block step; the rows returned fill that step.
    if re.search(r'"?gpu"?\s*:\s*1', SuperString) or not _is_open_svpflow():
        return 0
    block = re.search(r'"?block"?\s*:\s*\{([^{}]*)\}', VectorsString)
    block_options = block.group(1) if block else ''
    block_height = _option(block_options, 'h', _option(block_options, 'w', 16))
    overlap = _option(block_options, 'overlap', 2)
    if overlap == 0:
        return 0
    if overlap == 3:
        raise vs.Error('InterFrame: block overlap 3 is not supported with CPU rendering, use GPU or an overlap of 0-2')
    step = block_height - block_height // (8 if overlap == 1 else 4)
    if step % 2:
        step *= 2
    return (-height) % step


def _pad_bottom(clip: vs.VideoNode, rows: int) -> vs.VideoNode:
    return core.std.StackVertical([clip, core.std.Crop(clip, top=clip.height - rows)])


def _smooth(clip: vs.VideoNode, SuperString: str, VectorsString: str, SmoothString: str) -> vs.VideoNode:
    rows = _cpu_render_padding(clip.height, SuperString, VectorsString)
    padded = _pad_bottom(clip, rows) if rows else clip
    Super = padded.svp1.Super(SuperString)
    Vectors = core.svp1.Analyse(Super['clip'], Super['data'], padded, VectorsString)
    smooth = core.svp2.SmoothFps(padded, Super['clip'], Super['data'], Vectors['clip'], Vectors['data'], SmoothString)
    return core.std.Crop(smooth, bottom=rows) if rows else smooth

#------------------------------------------------------------------------------#
#                                                                              #
#                         InterFrame 2.8.2 by SubJunk                          #
#                                                                              #
#         A frame interpolation script that makes accurate estimations         #
#                   about the content of non-existent frames                   #
#      Its main use is to give videos higher framerates like newer TVs do      #
# changes:                                                                     #
#  20201108 - added  overwriteSuper, overwriteVectors, overwriteSmooth  (Selur)#
#------------------------------------------------------------------------------#
def InterFrameCustom(Input: vs.VideoNode, Preset: str = 'Medium', Tuning: str = 'Film', NewNum: Optional[int] = None, NewDen: int = 1, GPU: bool = False, InputType: str = '2D', OverrideAlgo: Optional[int] = None, OverrideArea: Optional[int] = None, FrameDouble: bool = False, overwriteSuper: str = '', overwriteVectors: str = '', overwriteSmooth: str = '') -> vs.VideoNode:
    if not isinstance(Input, vs.VideoNode):
        raise vs.Error('InterFrame: This is not a clip')

    # Validate inputs
    Preset = Preset.lower()
    Tuning = Tuning.lower()
    InputType = InputType.upper()

    if Preset not in ['medium', 'fast', 'faster', 'fastest']:
        raise vs.Error(f"InterFrame: '{Preset}' is not a valid preset")

    if Tuning not in ['film', 'smooth', 'animation', 'weak']:
        raise vs.Error(f"InterFrame: '{Tuning}' is not a valid tuning")

    if InputType not in ['2D', 'SBS', 'OU', 'HSBS', 'HOU']:
        raise vs.Error(f"InterFrame: '{InputType}' is not a valid InputType")

    def InterFrameProcess(clip: vs.VideoNode, overwriteSuper: str = '', overwriteVectors: str = '', overwriteSmooth: str = '') -> vs.VideoNode:
        if overwriteSuper == '':
          # Create SuperString
          if Preset in ['fast', 'faster', 'fastest']:
              SuperString = '{pel:1,'
          else:
              SuperString = '{'
          SuperString += 'gpu:1}' if GPU else 'gpu:0}'
        else:
          SuperString = overwriteSuper

        # Create VectorsString
        if overwriteVectors == '':
          if Tuning == 'animation' or Preset == 'fastest':
              VectorsString = '{block:{w:32,'
          elif Preset in ['fast', 'faster'] or not GPU:
              VectorsString = '{block:{w:16,'
          else:
              VectorsString = '{block:{w:8,'

          if Tuning == 'animation' or Preset == 'fastest':
              VectorsString += 'overlap:0'
          elif Preset == 'faster' and GPU:
              VectorsString += 'overlap:1'
          else:
              VectorsString += 'overlap:2'

          if Tuning == 'animation':
              VectorsString += '},main:{search:{coarse:{type:2,'
          elif Preset == 'faster':
              VectorsString += '},main:{search:{coarse:{'
          else:
              VectorsString += '},main:{search:{distance:0,coarse:{'

          if Tuning == 'animation':
              VectorsString += 'distance:-6,satd:false},distance:0,'
          elif Tuning == 'weak':
              VectorsString += 'distance:-1,trymany:true,'
          else:
              VectorsString += 'distance:-10,'

          if Tuning == 'animation' or Preset in ['faster', 'fastest']:
              VectorsString += 'bad:{sad:2000}}}}}'
          elif Tuning == 'weak':
              VectorsString += 'bad:{sad:2000}}}},refine:[{thsad:250,search:{distance:-1,satd:true}}]}'
          else:
              VectorsString += 'bad:{sad:2000}}}},refine:[{thsad:250}]}'
        else: 
          VectorsString = overwriteVectors

        # Create SmoothString
        if overwriteSmooth == '':
          if NewNum is not None:
              SmoothString = '{rate:{num:' + repr(NewNum) + ',den:' + repr(NewDen) + ',abs:true},'
          elif clip.fps_num / clip.fps_den in [15, 25, 30] or FrameDouble:
              SmoothString = '{rate:{num:2,den:1,abs:false},'
          else:
              SmoothString = '{rate:{num:60000,den:1001,abs:true},'

          if OverrideAlgo is not None:
              SmoothString += 'algo:' + repr(OverrideAlgo) + ',mask:{cover:80,'
          elif Tuning == 'animation':
              SmoothString += 'algo:2,mask:{'
          elif Tuning == 'smooth':
              SmoothString += 'algo:23,mask:{'
          else:
              SmoothString += 'algo:13,mask:{cover:80,'

          if OverrideArea is not None:
              SmoothString += f'area:{OverrideArea}'
          elif Tuning == 'smooth':
              SmoothString += 'area:150'
          else:
              SmoothString += 'area:0'

          if Tuning == 'weak':
              SmoothString += ',area_sharp:1.2},scene:{blend:true,mode:0,limits:{blocks:50}}}'
          else:
              SmoothString += ',area_sharp:1.2},scene:{blend:true,mode:0}}'
        else:
          SmoothString = overwriteSmooth

        return _smooth(clip, SuperString, VectorsString, SmoothString)

    # Get either 1 or 2 clips depending on InputType
    if InputType == 'SBS':
        FirstEye = InterFrameProcess(Input.std.Crop(right=Input.width // 2), overwriteSuper, overwriteVectors, overwriteSmooth)
        SecondEye = InterFrameProcess(Input.std.Crop(left=Input.width // 2), overwriteSuper, overwriteVectors, overwriteSmooth)
        return core.std.StackHorizontal([FirstEye, SecondEye])
    elif InputType == 'OU':
        FirstEye = InterFrameProcess(Input.std.Crop(bottom=Input.height // 2), overwriteSuper, overwriteVectors, overwriteSmooth)
        SecondEye = InterFrameProcess(Input.std.Crop(top=Input.height // 2), overwriteSuper, overwriteVectors, overwriteSmooth)
        return core.std.StackVertical([FirstEye, SecondEye])
    elif InputType == 'HSBS':
        FirstEye = InterFrameProcess(Input.std.Crop(right=Input.width // 2).resize.Spline36(Input.width, Input.height), overwriteSuper, overwriteVectors, overwriteSmooth)
        SecondEye = InterFrameProcess(Input.std.Crop(left=Input.width // 2).resize.Spline36(Input.width, Input.height), overwriteSuper, overwriteVectors, overwriteSmooth)
        return core.std.StackHorizontal([FirstEye.resize.Spline36(Input.width // 2, Input.height), SecondEye.resize.Spline36(Input.width // 2, Input.height)])
    elif InputType == 'HOU':
        FirstEye = InterFrameProcess(Input.std.Crop(bottom=Input.height // 2).resize.Spline36(Input.width, Input.height), overwriteSuper, overwriteVectors, overwriteSmooth)
        SecondEye = InterFrameProcess(Input.std.Crop(top=Input.height // 2).resize.Spline36(Input.width, Input.height), overwriteSuper, overwriteVectors, overwriteSmooth)
        return core.std.StackVertical([FirstEye.resize.Spline36(Input.width, Input.height // 2), SecondEye.resize.Spline36(Input.width, Input.height // 2)])
    else:
        return InterFrameProcess(Input, overwriteSuper, overwriteVectors, overwriteSmooth)


#------------------------------------------------------------------------------#
#                                                                              #
#                         InterFrame 2.8.2 by SubJunk                          #
#                                                                              #
#         A frame interpolation script that makes accurate estimations         #
#                   about the content of non-existent frames                   #
#      Its main use is to give videos higher framerates like newer TVs do      #
#------------------------------------------------------------------------------#
def InterFrame(Input: vs.VideoNode, Preset: str = 'Medium', Tuning: str = 'Film', NewNum: Optional[int] = None, NewDen: int = 1, GPU: bool = False, InputType: str = '2D', OverrideAlgo: Optional[int] = None, OverrideArea: Optional[int] = None, FrameDouble: bool = False) -> vs.VideoNode:
    if not isinstance(Input, vs.VideoNode):
        raise vs.Error('InterFrame: this is not a clip')

    # Validate inputs
    Preset = Preset.lower()
    Tuning = Tuning.lower()
    InputType = InputType.upper()

    if Preset not in ['medium', 'fast', 'faster', 'fastest']:
        raise vs.Error(f"InterFrame: '{Preset}' is not a valid preset")

    if Tuning not in ['film', 'smooth', 'animation', 'weak']:
        raise vs.Error(f"InterFrame: '{Tuning}' is not a valid tuning")

    if InputType not in ['2D', 'SBS', 'OU', 'HSBS', 'HOU']:
        raise vs.Error(f"InterFrame: '{InputType}' is not a valid InputType")

    def InterFrameProcess(clip: vs.VideoNode) -> vs.VideoNode:
        # Create SuperString
        if Preset in ['fast', 'faster', 'fastest']:
            SuperString = '{pel:1,'
        else:
            SuperString = '{'

        SuperString += 'gpu:1}' if GPU else 'gpu:0}'

        # Create VectorsString
        if Tuning == 'animation' or Preset == 'fastest':
            VectorsString = '{block:{w:32,'
        elif Preset in ['fast', 'faster'] or not GPU:
            VectorsString = '{block:{w:16,'
        else:
            VectorsString = '{block:{w:8,'

        if Tuning == 'animation' or Preset == 'fastest':
            VectorsString += 'overlap:0'
        elif Preset == 'faster' and GPU:
            VectorsString += 'overlap:1'
        else:
            VectorsString += 'overlap:2'

        if Tuning == 'animation':
            VectorsString += '},main:{search:{coarse:{type:2,'
        elif Preset == 'faster':
            VectorsString += '},main:{search:{coarse:{'
        else:
            VectorsString += '},main:{search:{distance:0,coarse:{'

        if Tuning == 'animation':
            VectorsString += 'distance:-6,satd:false},distance:0,'
        elif Tuning == 'weak':
            VectorsString += 'distance:-1,trymany:true,'
        else:
            VectorsString += 'distance:-10,'

        if Tuning == 'animation' or Preset in ['faster', 'fastest']:
            VectorsString += 'bad:{sad:2000}}}}}'
        elif Tuning == 'weak':
            VectorsString += 'bad:{sad:2000}}}},refine:[{thsad:250,search:{distance:-1,satd:true}}]}'
        else:
            VectorsString += 'bad:{sad:2000}}}},refine:[{thsad:250}]}'

        # Create SmoothString
        if NewNum is not None:
            SmoothString = '{rate:{num:' + repr(NewNum) + ',den:' + repr(NewDen) + ',abs:true},'
        elif clip.fps_num / clip.fps_den in [15, 25, 30] or FrameDouble:
            SmoothString = '{rate:{num:2,den:1,abs:false},'
        else:
            SmoothString = '{rate:{num:60000,den:1001,abs:true},'

        if OverrideAlgo is not None:
            SmoothString += 'algo:' + repr(OverrideAlgo) + ',mask:{cover:80,'
        elif Tuning == 'animation':
            SmoothString += 'algo:2,mask:{'
        elif Tuning == 'smooth':
            SmoothString += 'algo:23,mask:{'
        else:
            SmoothString += 'algo:13,mask:{cover:80,'

        if OverrideArea is not None:
            SmoothString += f'area:{OverrideArea}'
        elif Tuning == 'smooth':
            SmoothString += 'area:150'
        else:
            SmoothString += 'area:0'

        if Tuning == 'weak':
            SmoothString += ',area_sharp:1.2},scene:{blend:true,mode:0,limits:{blocks:50}}}'
        else:
            SmoothString += ',area_sharp:1.2},scene:{blend:true,mode:0}}'

        return _smooth(clip, SuperString, VectorsString, SmoothString)

    # Get either 1 or 2 clips depending on InputType
    if InputType == 'SBS':
        FirstEye = InterFrameProcess(Input.std.Crop(right=Input.width // 2))
        SecondEye = InterFrameProcess(Input.std.Crop(left=Input.width // 2))
        return core.std.StackHorizontal([FirstEye, SecondEye])
    elif InputType == 'OU':
        FirstEye = InterFrameProcess(Input.std.Crop(bottom=Input.height // 2))
        SecondEye = InterFrameProcess(Input.std.Crop(top=Input.height // 2))
        return core.std.StackVertical([FirstEye, SecondEye])
    elif InputType == 'HSBS':
        FirstEye = InterFrameProcess(Input.std.Crop(right=Input.width // 2).resize.Spline36(Input.width, Input.height))
        SecondEye = InterFrameProcess(Input.std.Crop(left=Input.width // 2).resize.Spline36(Input.width, Input.height))
        return core.std.StackHorizontal([FirstEye.resize.Spline36(Input.width // 2, Input.height), SecondEye.resize.Spline36(Input.width // 2, Input.height)])
    elif InputType == 'HOU':
        FirstEye = InterFrameProcess(Input.std.Crop(bottom=Input.height // 2).resize.Spline36(Input.width, Input.height))
        SecondEye = InterFrameProcess(Input.std.Crop(top=Input.height // 2).resize.Spline36(Input.width, Input.height))
        return core.std.StackVertical([FirstEye.resize.Spline36(Input.width, Input.height // 2), SecondEye.resize.Spline36(Input.width, Input.height // 2)])
    else:
        return InterFrameProcess(Input)
