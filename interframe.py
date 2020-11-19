from vapoursynth import core
import vapoursynth as vs

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
def InterFrame(Input, Preset='Medium', Tuning='Film', NewNum=None, NewDen=1, GPU=False, InputType='2D', OverrideAlgo=None, OverrideArea=None, FrameDouble=False, overwriteSuper='', overwriteVectors='', overwriteSmooth=''):
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

    def InterFrameProcess(clip, overwriteSuper='', overwriteVectors='', overwriteSmooth=''):
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

        # Make interpolation vector clip
        Super = clip.svp1.Super(SuperString)
        Vectors = core.svp1.Analyse(Super['clip'], Super['data'], clip, VectorsString)

        # Put it together
        return core.svp2.SmoothFps(clip, Super['clip'], Super['data'], Vectors['clip'], Vectors['data'], SmoothString)

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
