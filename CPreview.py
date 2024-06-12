# ===============================================================================
# ===============================================================================
#            CPreview 2024-06-10
# ===============================================================================
# ===============================================================================

import math as m
from functools import partial
import vapoursynth as vs
core = vs.core

# -------------------------------------------------------------------------------

def CPreview(Source, CL, CR, CT, CB, Frame=False, Time=False, Type=1, CropText=True):

    if not isinstance(Source, vs.VideoNode):
        raise vs.Error('CPreview: Source must be a video')

# -------------------------------------------------------------------------------

    Source_Width = Source.width
    F_Source_Width = float(Source_Width)
    Source_Height = Source.height
    F_Source_Height = float(Source_Height)

    if (Source_Width == 0) or (Source_Height == 0):
        raise vs.Error('CPreview: Only video with a constant width and height is supported')

    SCFamily = Source.format.color_family
    SFName = Source.format.name
    Source_Bits = Source.format.bits_per_sample
    Source_Sample_Type = Source.format.sample_type

    Is444 = (Source.format.subsampling_w == 0 == Source.format.subsampling_h)

    PropsNum = Source.get_frame(0).props.get('_SARNum', 0)
    PropsDen = Source.get_frame(0).props.get('_SARDen', 0)

    CropLine = 1 if (Source_Width <= 1920) and (Source_Height <= 1080) else 2

    if not ((isinstance(Type, bool) is False) and isinstance(Type, int) and (CropLine >= 1)):
        raise vs.Error('CPreview: CropLine must be an integer greater than or equal to 1')

    CW = Source_Width - CL - CR
    CH = Source_Height - CT - CB
    CLineL = min(CL, CropLine)
    CLineR = min(CR, CropLine)
    CLineT = min(CT, CropLine)
    CLineB = min(CB, CropLine)

    Source444 = \
      core.resize.Bicubic(Source, format=vs.RGB48) \
      if (SCFamily == vs.RGB) and SFName.endswith("H") else \
      core.resize.Bicubic(Source, format=vs.GRAY16) \
      if (SCFamily == vs.GRAY) and SFName.endswith("H") else \
      core.resize.Bicubic(Source, format=vs.YUV444P16) \
      if SFName.endswith("H") else \
      core.resize.Bicubic(Source, format=vs.YUV444PS) \
      if (SCFamily == vs.YUV) and (Is444 is False) and SFName.endswith("S") else \
      eval("core.resize.Bicubic(Source, format=vs.YUV444P" + str(Source_Bits) + ")") \
      if (SCFamily == vs.YUV) and (Is444 is False) else Source

    CropTest = None if (SCFamily == vs.RGB) or (SCFamily == vs.GRAY) or (Is444 is True) else \
      core.std.Crop(Source, CL, CR, CT, CB)
    Cropped444 = core.std.Crop(Source444, CL, CR, CT, CB)

    Scale = 2 ** (Source444.format.bits_per_sample - 8)
    MinIn = 16 * Scale
    MaxInL = 235 * Scale
    MaxInC = 240 * Scale
    MaxOut = 255 * Scale

    SubText = "" if (CropText is False) else \
      f'Source Resolution\n{Source_Width} x {Source_Height}' + \
      '\n\nCropping\nLeft ' + str(CL) + ', Right ' + str(CR) + \
      ', Top ' + str(CT) + ', Bottom ' + str(CB) + \
      '\n\nCropped Resolution\n' + f'{Cropped444.width} x {Cropped444.height}' + \
      '\n\nFrame Properties SAR  ' + (f'{PropsNum}:{PropsDen}' if (PropsNum > 0 < PropsDen) else 'None')

    BVid = core.std.BlankClip(Cropped444)

    if (Type == 1):

        OVid = BVid.std.AddBorders(CLineL, CLineR, CLineT, CLineB, color=RGBColor(Source444, 'FFFF00'))
        OVid = OVid.std.AddBorders(CL - CLineL, CR - CLineR, CT - CLineT, CB - CLineB)
        MVid = BVid.std.AddBorders(CLineL, CLineR, CLineT, CLineB, color=RGBColor(Source444, 'B1B1B1'))
        MVid = MVid.std.AddBorders(CL - CLineL, CR - CLineR, CT - CLineT, CB - CLineB)

    elif (Type == 2):

        OVid = BVid.std.AddBorders(CL, CR, CT, CB, color=RGBColor(Source444, 'FFFF00'))
        MVid = BVid.std.AddBorders(CL, CR, CT, CB, color=RGBColor(Source444, '4B4B4B'))

    else:

        OVid = Cropped444.std.AddBorders(CL, CR, CT, CB)
        MVid = core.std.BlankClip(Cropped444, color=RGBColor(Source444, 'FFFFFF'))
        MVid = MVid.std.AddBorders(CL, CR, CT, CB)

    MVid = MVid if (SCFamily == vs.RGB) or SFName.endswith("S") else \
      MVid.std.Levels(min_in=MinIn, max_in=MaxInL, min_out=0, max_out=MaxOut, planes=0)
    MVid = MVid if (SCFamily == vs.RGB) or (SCFamily == vs.GRAY) or SFName.endswith("S") else \
      MVid.std.Levels(min_in=MinIn, max_in=MaxInC, min_out=0, max_out=MaxOut, planes=[1, 2])

    Output = Source444 if (0 == CL == CT == CR == CB) else \
      core.std.MaskedMerge(Source444, OVid, mask=MVid, first_plane=True) \
      if (Type == 1) or (Type == 2) else \
      core.std.MaskedMerge(core.std.Invert(Source444), OVid, mask=MVid, first_plane=True)

    Output = Output if (CropText is False) or (Frame is True) or (Time is True) else \
      core.text.Text(Output, SubText, alignment=5)

    return Output if (Frame is False) and (Time is False) else \
      CP_Position(Output, Frame, Time, SubText)

# ===============================================================================
# ===============================================================================
#            Wrapper Functions
# ===============================================================================
# ===============================================================================

# -------------------------------------------------------------------------------
#            pCrop / pCropf / pCropt / pCropp
# -------------------------------------------------------------------------------

def pCrop(Source, CL, CR, CT, CB):
    return CPreview(Source, CL, CR, CT, CB, False, False, 1)

def pCropf(Source, CL, CR, CT, CB):
    return CPreview(Source, CL, CR, CT, CB, True, False, 1)

def pCropt(Source, CL, CR, CT, CB):
    return CPreview(Source, CL, CR, CT, CB, False, True, 1)

def pCropp(Source, CL, CR, CT, CB):
    return CPreview(Source, CL, CR, CT, CB, True, True, 1)

# -------------------------------------------------------------------------------
#            ppCrop / ppCropf / ppCropt / ppCropp
# -------------------------------------------------------------------------------

def ppCrop(Source, CL, CR, CT, CB):
    return CPreview(Source, CL, CR, CT, CB, False, False, 2)

def ppCropf(Source, CL, CR, CT, CB):
    return CPreview(Source, CL, CR, CT, CB, True, False, 2)

def ppCropt(Source, CL, CR, CT, CB):
    return CPreview(Source, CL, CR, CT, CB, False, True, 2)

def ppCropp(Source, CL, CR, CT, CB):
    return CPreview(Source, CL, CR, CT, CB, True, True, 2)

# -------------------------------------------------------------------------------
#            pppCrop / pppCropf / pppCropt / pppCropp
# -------------------------------------------------------------------------------

def pppCrop(Source, CL, CR, CT, CB):
    return CPreview(Source, CL, CR, CT, CB, False, False, 3)

def pppCropf(Source, CL, CR, CT, CB):
    return CPreview(Source, CL, CR, CT, CB, True, False, 3)

def pppCropt(Source, CL, CR, CT, CB):
    return CPreview(Source, CL, CR, CT, CB, False, True, 3)

def pppCropp(Source, CL, CR, CT, CB):
    return CPreview(Source, CL, CR, CT, CB, True, True, 3)

# -------------------------------------------------------------------------------
#            qCrop / qCropf / qCropt / qCropp
# -------------------------------------------------------------------------------

def qCrop(Source, CL, CR, CT, CB):
    return CPreview(Source, CL, CR, CT, CB, False, False, 1, False)

def qCropf(Source, CL, CR, CT, CB):
    return CPreview(Source, CL, CR, CT, CB, True, False, 1, False)

def qCropt(Source, CL, CR, CT, CB):
    return CPreview(Source, CL, CR, CT, CB, False, True, 1, False)

def qCropp(Source, CL, CR, CT, CB):
    return CPreview(Source, CL, CR, CT, CB, True, True, 1, False)

# -------------------------------------------------------------------------------
#            qqCrop / qqCropf / qqCropt / qqCropp
# -------------------------------------------------------------------------------

def qqCrop(Source, CL, CR, CT, CB):
    return CPreview(Source, CL, CR, CT, CB, False, False, 2, False)

def qqCropf(Source, CL, CR, CT, CB):
    return CPreview(Source, CL, CR, CT, CB, True, False, 2, False)

def qqCropt(Source, CL, CR, CT, CB):
    return CPreview(Source, CL, CR, CT, CB, False, True, 2, False)

def qqCropp(Source, CL, CR, CT, CB):
    return CPreview(Source, CL, CR, CT, CB, True, True, 2, False)

# -------------------------------------------------------------------------------
#            qqqCrop / qqqCropf / qqqCropt / qqqCropp
# -------------------------------------------------------------------------------

def qqqCrop(Source, CL, CR, CT, CB):
    return CPreview(Source, CL, CR, CT, CB, False, False, 3, False)

def qqqCropf(Source, CL, CR, CT, CB):
    return CPreview(Source, CL, CR, CT, CB, True, False, 3, False)

def qqqCropt(Source, CL, CR, CT, CB):
    return CPreview(Source, CL, CR, CT, CB, False, True, 3, False)

def qqqCropp(Source, CL, CR, CT, CB):
    return CPreview(Source, CL, CR, CT, CB, True, True, 3, False)

# -------------------------------------------------------------------------------
#            Cropf / Cropt / Cropp
# -------------------------------------------------------------------------------

def Cropf(Source, CL, CR, CT, CB):
    Output = core.std.Crop(Source, CL, CR, CT, CB)
    return CP_Position(Output, True, False)

def Cropt(Source, CL, CR, CT, CB):
    Output = core.std.Crop(Source, CL, CR, CT, CB)
    return CP_Position(Output, False, True)

def Cropp(Source, CL, CR, CT, CB):
    Output = core.std.Crop(Source, CL, CR, CT, CB)
    return CP_Position(Output, True, True)

# ===============================================================================
# ===============================================================================
#            Helper Functions
# ===============================================================================
# ===============================================================================

def Crop(Source, CL, CR, CT, CB):

    if not isinstance(Source, vs.VideoNode):
        raise vs.Error('CPreview: Source must be a video')

    return core.std.Crop(Source, CL, CR, CT, CB)

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
#            Position
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

def CP_Position(Source, Frame, Time, SubText=""):

    SCFamily = Source.format.color_family
    SFName = Source.format.name

    Source = \
      core.resize.Bicubic(Source, format=vs.RGB48) \
      if (SCFamily == vs.RGB) and SFName.endswith("H") else \
      core.resize.Bicubic(Source, format=vs.GRAY16) \
      if (SCFamily == vs.GRAY) and SFName.endswith("H") else \
      core.resize.Bicubic(Source, format=vs.YUV444P16) \
      if SFName.endswith("H") else Source

    return core.std.FrameEval(Source, \
      partial(CP_Pos, Source=Source, Frame=Frame, Time=Time, SubText=SubText))

# ---------------------------------------

def CP_Pos(n, Source, Frame, Time, SubText):

    FRateNum = Source.fps.numerator
    FRateDen = Source.fps.denominator
    Time = Time if (FRateNum > 0 < FRateDen) else False

    if (Time is True):

        F_Position = n * FRateDen / FRateNum
        HH = m.floor(F_Position / 3600)
        MM = m.floor(F_Position / 60) % 60
        SS = m.floor(F_Position) % 60
        MS = m.floor((F_Position - m.floor(F_Position)) * 1000)
        Pos = "{:02.0f}:{:02.0f}:{:02.0f}.{:03.0f}".format(HH, MM, SS, MS) + "\n\n"

    PosText = str(n) + "\n" + Pos if (Frame is True) and (Time is True) else \
      str(n) + "\n\n" if (Frame is True) else Pos if (Time is True) else ""

    return core.text.Text(Source, PosText + SubText, alignment=5)

# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------
#            RGBColor
# -------------------------------------------------------------------------------
# -------------------------------------------------------------------------------

def RGBColor(clip, color=None, matrix=None, range=None):

    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('RGBColor: clip must be a video')
    if not ((color is None) or isinstance(color, str)):
        raise vs.Error('RGBColor: color must be a string, for example "darkblue" or "00008B"')
    if not ((matrix is None) or isinstance(matrix, str) or \
      (isinstance(matrix, int) and (isinstance(matrix, bool) is False))):
        raise vs.Error('RGBColor: matrix must be an integer or string')
    if not ((range is None) or (isinstance(range, str) and \
      ((range.lower().strip() == 'full') or (range.lower().strip() == 'limited') or \
      (range.lower().strip() == 'f') or (range.lower().strip() == 'l')))):
        raise vs.Error('RGBColor: range must be "full" or "f", or "limited" or "l"')
    if (clip.format.name).endswith("H"):
        raise vs.Error(f'RGBColor: {clip.format.name} is not supported')

    color = None if (color is None) else color.lower().strip()

    colors = {'aliceblue'            : 'F0F8FF',
              'antiquewhite'         : 'FAEBD7',
              'aqua'                 : '00FFFF',
              'aquamarine'           : '7FFFD4',
              'azure'                : 'F0FFFF',
              'beige'                : 'F5F5DC',
              'bisque'               : 'FFE4C4',
              'black'                : '000000',
              'blanchedalmond'       : 'FFEBCD',
              'blue'                 : '0000FF',
              'blueviolet'           : '8A2BE2',
              'brown'                : 'A52A2A',
              'burlywood'            : 'DEB887',
              'cadetblue'            : '5F9EA0',
              'chartreuse'           : '7FFF00',
              'chocolate'            : 'D2691E',
              'coral'                : 'FF7F50',
              'cornflowerblue'       : '6495ED',
              'cornsilk'             : 'FFF8DC',
              'crimson'              : 'DC143C',
              'cyan'                 : '00FFFF',
              'darkblue'             : '00008B',
              'darkcyan'             : '008B8B',
              'darkgoldenrod'        : 'B8860B',
              'darkgray'             : 'A9A9A9',
              'darkgrey'             : 'A9A9A9',
              'darkgreen'            : '006400',
              'darkkhaki'            : 'BDB76B',
              'darkmagenta'          : '8B008B',
              'darkolivegreen'       : '556B2F',
              'darkorange'           : 'FF8C00',
              'darkorchid'           : '9932CC',
              'darkred'              : '8B0000',
              'darksalmon'           : 'E9967A',
              'darkseagreen'         : '8FBC8F',
              'darkslateblue'        : '483D8B',
              'darkslategray'        : '2F4F4F',
              'darkslategrey'        : '2F4F4F',
              'darkturquoise'        : '00CED1',
              'darkviolet'           : '9400D3',
              'deeppink'             : 'FF1493',
              'deepskyblue'          : '00BFFF',
              'dimgray'              : '696969',
              'dimgrey'              : '696969',
              'dodgerblue'           : '1E90FF',
              'firebrick'            : 'B22222',
              'floralwhite'          : 'FFFAF0',
              'forestgreen'          : '228B22',
              'fuchsia'              : 'FF00FF',
              'gainsboro'            : 'DCDCDC',
              'ghostwhite'           : 'F8F8FF',
              'gold'                 : 'FFD700',
              'goldenrod'            : 'DAA520',
              'gray'                 : '808080',
              'grey'                 : '808080',
              'green'                : '008000',
              'greenyellow'          : 'ADFF2F',
              'honeydew'             : 'F0FFF0',
              'hotpink'              : 'FF69B4',
              'indianred'            : 'CD5C5C',
              'indigo'               : '4B0082',
              'ivory'                : 'FFFFF0',
              'khaki'                : 'F0E68C',
              'lavender'             : 'E6E6FA',
              'lavenderblush'        : 'FFF0F5',
              'lawngreen'            : '7CFC00',
              'lemonchiffon'         : 'FFFACD',
              'lightblue'            : 'ADD8E6',
              'lightcoral'           : 'F08080',
              'lightcyan'            : 'E0FFFF',
              'lightgoldenrodyellow' : 'FAFAD2',
              'lightgreen'           : '90EE90',
              'lightgray'            : 'D3D3D3',
              'lightgrey'            : 'D3D3D3',
              'lightpink'            : 'FFB6C1',
              'lightsalmon'          : 'FFA07A',
              'lightseagreen'        : '20B2AA',
              'lightskyblue'         : '87CEFA',
              'lightslategray'       : '778899',
              'lightslategrey'       : '778899',
              'lightsteelblue'       : 'B0C4DE',
              'lightyellow'          : 'FFFFE0',
              'lime'                 : '00FF00',
              'limegreen'            : '32CD32',
              'linen'                : 'FAF0E6',
              'magenta'              : 'FF00FF',
              'maroon'               : '800000',
              'mediumaquamarine'     : '66CDAA',
              'mediumblue'           : '0000CD',
              'mediumorchid'         : 'BA55D3',
              'mediumpurple'         : '9370DB',
              'mediumseagreen'       : '3CB371',
              'mediumslatenlue'      : '7B68EE',
              'mediumspringgreen'    : '00FA9A',
              'mediumturquoise'      : '48D1CC',
              'mediumvioletred'      : 'C71585',
              'midnightblue'         : '191970',
              'mintcream'            : 'F5FFFA',
              'mistyrose'            : 'FFE4E1',
              'moccasin'             : 'FFE4B5',
              'navajowhite'          : 'FFDEAD',
              'navy'                 : '000080',
              'oldlace'              : 'FDF5E6',
              'olive'                : '808000',
              'olivedrab'            : '6B8E23',
              'orange'               : 'FFA500',
              'orangered'            : 'FF4500',
              'orchid'               : 'DA70D6',
              'palegoldenrod'        : 'EEE8AA',
              'palegreen'            : '98FB98',
              'paleturquoise'        : 'AFEEEE',
              'palevioletred'        : 'DB7093',
              'papayawhip'           : 'FFEFD5',
              'peachpuff'            : 'FFDAB9',
              'peru'                 : 'CD853F',
              'pink'                 : 'FFC0CB',
              'plum'                 : 'DDA0DD',
              'powderblue'           : 'B0E0E6',
              'purple'               : '800080',
              'red'                  : 'FF0000',
              'rosybrown'            : 'BC8F8F',
              'royalblue'            : '4169E1',
              'saddlebrown'          : '8B4513',
              'salmon'               : 'FA8072',
              'sandybrown'           : 'F4A460',
              'seagreen'             : '2E8B57',
              'seashell'             : 'FFF5EE',
              'sienna'               : 'A0522D',
              'silver'               : 'C0C0C0',
              'skyblue'              : '87CEEB',
              'slateblue'            : '6A5ACD',
              'slategray'            : '708090',
              'slategrey'            : '708090',
              'snow'                 : 'FFFAFA',
              'springgreen'          : '00FF7F',
              'steelblue'            : '4682B4',
              'tan'                  : 'D2B48C',
              'teal'                 : '008080',
              'thistle'              : 'D8BFD8',
              'tomato'               : 'FF6347',
              'turquoise'            : '40E0D0',
              'violet'               : 'EE82EE',
              'wheat'                : 'F5DEB3',
              'white'                : 'FFFFFF',
              'whitesmoke'           : 'F5F5F5',
              'yellow'               : 'FFFF00',
              'yellowgreen'          : '9ACD32',
              'gray10'               : '191919',
              'grey10'               : '191919',
              'gray20'               : '323232',
              'grey20'               : '323232',
              'gray30'               : '4B4B4B',
              'grey30'               : '4B4B4B',
              'gray40'               : '656565',
              'grey40'               : '656565',
              'gray50'               : '7F7F7F',
              'grey50'               : '7F7F7F',
              'gray60'               : '989898',
              'grey60'               : '989898',
              'gray70'               : 'B1B1B1',
              'grey70'               : 'B1B1B1',
              'gray80'               : 'CACACA',
              'grey80'               : 'CACACA',
              'gray90'               : 'E4E4E4',
              'grey90'               : 'E4E4E4'}

    if (colors.get(color) is None) and (color is not None) and (len(color) != 6):
        raise vs.Error('RGBColor: Invalid color string specified')

    elif (colors.get(color) is None):
        try: v0, v1, v2 = tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))
        except: raise vs.Error('RGBColor: Invalid color string specified')

    else: v0, v1, v2 = tuple(int(colors.get(color)[i : i + 2], 16) for i in (0, 2, 4))

    if (clip.format.color_family != vs.RGB):
        matrix = matrix if (matrix is None) or isinstance(matrix, int) else \
          matrix.lower().strip()

        if (matrix is not None) and isinstance(matrix, str):
            MatrixNum = \
              1 if (matrix == '709') else 4 if (matrix == 'fcc') else \
              5 if (matrix == '470bg') else 6 if (matrix == '170m') else \
              7 if (matrix == '240m') else 8 if (matrix == 'ycgco') else \
              9 if (matrix == '2020ncl') else 10 if (matrix == '2020cl') else -1

        if (matrix is not None) and isinstance(matrix, int):
            MatrixNum = matrix if (matrix == 1) or (4 <= matrix <= 10) else -1

        if (matrix is not None) and (MatrixNum == -1):
            raise vs.Error('RGBColor: Unsupported matrix specified')

        PropMatrix = clip.get_frame(0).props.get("_Matrix", None)

        if (PropMatrix is not None) and (PropMatrix > 10):
            raise vs.Error(f'RGBColor: Video has an unsupported value ({PropMatrix}) ' + \
                     'for "_Matrix" in frame properties')

        PropMatrix = PropMatrix if (PropMatrix is None) else \
          PropMatrix if (PropMatrix == 1) or (4 <= PropMatrix <= 10) else None

        if (matrix is not None) and (PropMatrix is not None) and (MatrixNum != PropMatrix):
            raise vs.Error(f'RGBColor: The value for "_Matrix" ({PropMatrix}) ' + \
                     'in frame properties doesn\'t match the specified matrix')

        matrix = MatrixNum if (matrix is not None) else \
          PropMatrix if (PropMatrix is not None) else \
          5 if (clip.width <= 1024) and (clip.height <= 576) else \
          1 if (clip.width <= 1920) and (clip.height <= 1080) else 9

    range = range if (range is None) else range.lower().strip()
    range = range if (range is None) else \
      0 if (range == 'full') or (range == 'f') else \
      1 if (range == 'limited') or (range == 'l') else None

    PropRange = clip.get_frame(0).props.get("_ColorRange", None)
    PropRange = PropRange if (PropRange is None) else \
      PropRange if (0 <= PropRange <= 1) else None

    if (range is not None) and (PropRange is not None) and (range != PropRange):
        raise vs.Error(f'RGBColor: The value for "_ColorRange" ({PropRange}) ' + \
                 'in frame properties doesn\'t match the specified range')

    range = range if (range is not None) else PropRange

    BlankRGBClip = core.std.BlankClip(color=(v0, v1, v2))
    ColorClip = BlankRGBClip.resize.Point(format=clip.format.id, matrix=matrix, range=range)

    f = ColorClip.get_frame(0)
    p0 = f[0][0,0]

    return p0 if (clip.format.color_family == vs.GRAY) else (p0, f[1][0,0], f[2][0,0])

# ===============================================================================
# ===============================================================================
