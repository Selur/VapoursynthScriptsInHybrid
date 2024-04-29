from vapoursynth import core
import vapoursynth as vs

# collection of small helper functions to validate parameters

def matrixIsInvalid(clip: vs.VideoNode) -> bool:
  frame = clip.get_frame(0)
  value = frame.props.get('_Matrix', None)
  return value in [None, 2, 3] or value not in vs.MatrixCoefficients.__members__.values()

def transferIsInvalid(clip: vs.VideoNode) -> bool:
  frame = clip.get_frame(0)
  value = frame.props.get('_Transfer', None)
  return value in [None, 0, 2, 3] or value not in vs.TransferCharacteristics.__members__.values()
  
def primariesIsInvalid(clip: vs.VideoNode) -> bool:
  frame = clip.get_frame(0)
  value = frame.props.get('_Primaries', None)
  return value in [None, 2] or value not in vs.ColorPrimaries.__members__.values()

def rangeIsInvalid(clip: vs.VideoNode) -> bool:
  frame = clip.get_frame(0)
  value = frame.props.get('_ColorRange', None)
  return value is None or value not in vs.ColorRange.__members__.values()

def fieldBaseIsInvalid(clip: vs.VideoNode) -> bool:
  frame = clip.get_frame(0)
  value = frame.props.get('_FieldBased', None)
  return value is None or value not in vs.FieldBased.__members__.values()
