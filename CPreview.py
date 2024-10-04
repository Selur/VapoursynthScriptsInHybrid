# ===============================================================================
# ===============================================================================
#            CPreview 2024-10-04
# ===============================================================================
# ===============================================================================

import math as m
from functools import partial
import vapoursynth as vs
core = vs.core

# ===============================================================================

def CPreview(Source, CL, CR, CT, CB, Frame=False, Time=False, Type=1):

    if not isinstance(Source, vs.VideoNode): raise vs.Error('CPreview: Source must be a video')

    Source_Width = Source.width
    Source_Height = Source.height
    F_Source_Width = float(Source_Width)
    F_Source_Height = float(Source_Height)
    Source_Bits = Source.format.bits_per_sample
    Scale = 2 ** (Source_Bits - 8)
    CropLine = P_Line(Source_Width, Source_Height) if (1 <= Type <= 3) else Q_Line(Source_Width, Source_Height)
    IsHalfFloat = Source.format.name.endswith("H")
    IsFloat = Source.format.name.endswith("S")
    IsRGBSource = (Source.format.color_family == vs.RGB)
    IsGraySource = (Source.format.color_family == vs.GRAY)
    IsChromaSS = \
      not ((Source.format.subsampling_w == 0 == Source.format.subsampling_h) or IsRGBSource or IsGraySource)

    if (Source_Width == 0) or (Source_Height == 0):
        raise vs.Error('CPreview: Only video with a constant width and height is supported')
    if not ((isinstance(CL, bool) is False) and isinstance(CL, int) and (CL >= 0)):
        raise vs.Error(f'CPreview: Left cropping must be zero or a positive integer')
    if not ((isinstance(CR, bool) is False) and isinstance(CR, int) and (CR >= 0)):
        raise vs.Error(f'CPreview: Right cropping must be zero or a positive integer')
    if not ((isinstance(CT, bool) is False) and isinstance(CT, int) and (CT >= 0)):
        raise vs.Error(f'CPreview: Top cropping must be zero or a positive integer')
    if not ((isinstance(CB, bool) is False) and isinstance(CB, int) and (CB >= 0)):
        raise vs.Error(f'CPreview: Bottom cropping must be zero or a positive integer')
    if not ((isinstance(CropLine, bool) is False) and isinstance(CropLine, int) and (CropLine >= 1)):
        raise vs.Error('CPreview: The LineThickness functions must return an integer greater than or equal to 1')
    if (1 <= Type <= 3) and IsHalfFloat:
        raise vs.Error('CPreview: The cropping previews beginning with "p" cannot be used with ' + \
          f'{Source.format.name} as half float formats are not supported by the VapourSynth Text function')

    PropNum = Source.get_frame(0).props.get('_SARNum', 0)
    PropDen = Source.get_frame(0).props.get('_SARDen', 0)
    PropRange = Source.get_frame(0).props.get("_ColorRange", -1)
    PropRange = PropRange if (-1 <= PropRange <= 1) else -1
    PropChroma = None if not IsChromaSS else Source.get_frame(0).props.get("_ChromaLocation", None)

    Cropped_Width = Source_Width - CL - CR
    Cropped_Height = Source_Height - CT - CB
    CLineL = min(CL, CropLine)
    CLineR = min(CR, CropLine)
    CLineT = min(CT, CropLine)
    CLineB = min(CB, CropLine)

    CropTest = core.std.Crop(Source, CL, CR, CT, CB)

    Source444 = Source if not (IsChromaSS or IsHalfFloat) else \
      core.resize.Bicubic(Source, format=vs.RGBS) if IsRGBSource and IsHalfFloat else \
      core.resize.Bicubic(Source, format=vs.GRAYS) if IsGraySource and IsHalfFloat else \
      core.resize.Bicubic(Source, format=vs.YUV444PS) if IsHalfFloat or (IsFloat and IsChromaSS) else \
      eval("core.resize.Bicubic(Source, format=vs.YUV444P" + str(Source_Bits) + ")")

    Cropped444 = core.std.Crop(Source444, CL, CR, CT, CB)

# -------------------------------------------------------------------------------

    SubText = "" if (4 <= Type <= 6) else f'Source Resolution\n{Source_Width} x {Source_Height}' + \
      CR_PicMod(Source_Width, Source_Height) + \
      '\n\nCropping\nLeft ' + str(CL) + ', Right ' + str(CR) + ', Top ' + str(CT) + ', Bottom ' + str(CB) + \
      '\n\nCropped Resolution\n' + f'{Cropped444.width} x {Cropped444.height}' + \
      CR_PicMod(Cropped_Width, Cropped_Height) + \
      '\n\nFrame Properties SAR  ' + (f'{PropNum}:{PropDen}' if (PropNum > 0 < PropDen) else 'None')

# -------------------------------------------------------------------------------

    Black = RGBColor(Source444)
    BClip = core.std.BlankClip(Cropped444, color=Black)

    if (Type == 1) or (Type == 4):

        OClip = BClip.std.AddBorders(CLineL, CLineR, CLineT, CLineB, color=RGBColor(Source444,'FFFF00'))
        OClip = OClip.std.AddBorders(CL-CLineL, CR-CLineR, CT-CLineT, CB-CLineB, color=Black)
        MClip = BClip.std.AddBorders(CLineL, CLineR, CLineT, CLineB, color=RGBColor(Source444,'E4E4E4'))
        MClip = MClip.std.AddBorders(CL-CLineL, CR-CLineR, CT-CLineT, CB-CLineB, color=Black)

    elif (Type == 2) or (Type == 5):

        OClip = BClip.std.AddBorders(CL, CR, CT, CB, color=RGBColor(Source444,'FFFF00'))
        MClip = BClip.std.AddBorders(CL, CR, CT, CB, color=RGBColor(Source444,'4B4B4B'))

    else:

        OClip = Cropped444.std.AddBorders(CL, CR, CT, CB, color=Black)
        MClip = core.std.BlankClip(Cropped444, color=RGBColor(Source444,'FFFFFF'))
        MClip = MClip.std.AddBorders(CL, CR, CT, CB, color=Black)

    Planes = [0, 1, 2] if IsRGBSource else 0
    FirstPlane = False if IsRGBSource else True

    MClip = MClip if (PropRange == 0) or ((PropRange == -1) and IsRGBSource) or IsFloat or IsHalfFloat else \
      MClip.std.Levels(min_in=16*Scale, max_in=235*Scale, min_out=0, max_out=255*Scale, planes=Planes)

# -------------------------------------------------------------------------------

    CPreviewVideo = core.std.MaskedMerge(Source444, OClip, mask=MClip, first_plane=FirstPlane) \
      if (1 <= Type <= 2) or (4 <= Type <= 5) else \
      core.std.MaskedMerge(core.std.Invert(Source444), OClip, mask=MClip, first_plane=FirstPlane)

    CPreviewVideo = CPreviewVideo if ((Type == 1) or (Type == 4) or not IsChromaSS) and not IsHalfFloat else \
      core.resize.Bicubic(CPreviewVideo, format=vs.RGBH) if IsRGBSource else \
      core.resize.Bicubic(CPreviewVideo, format=vs.GRAYH) if IsGraySource else \
      core.resize.Bicubic(CPreviewVideo, format=vs.YUV444PH) if (Type == 1) or (Type == 4) else \
      core.resize.Bicubic(CPreviewVideo, format=Source.format.id, chromaloc=PropChroma)

    CPreviewVideo = CPreviewVideo if (Frame or Time) or (4 <= Type <= 6) else \
      core.text.Text(CPreviewVideo, SubText, alignment=5)

    return CPreviewVideo if not (Frame or Time) else CP_Position(CPreviewVideo, Frame, Time, SubText)

# ===============================================================================
# ===============================================================================
#            Wrapper Functions
# ===============================================================================
# ===============================================================================

# -------------------------------------------------------------------------------
#            pCrop / pCropf / pCropt / pCropp
# -------------------------------------------------------------------------------

def pCrop(Source,  CL, CR, CT, CB): return CPreview(Source, CL, CR, CT, CB, False, False, 1)
def pCropf(Source, CL, CR, CT, CB): return CPreview(Source, CL, CR, CT, CB, True,  False, 1)
def pCropt(Source, CL, CR, CT, CB): return CPreview(Source, CL, CR, CT, CB, False, True,  1)
def pCropp(Source, CL, CR, CT, CB): return CPreview(Source, CL, CR, CT, CB, True,  True,  1)

# -------------------------------------------------------------------------------
#            ppCrop / ppCropf / ppCropt / ppCropp
# -------------------------------------------------------------------------------

def ppCrop(Source,  CL, CR, CT, CB): return CPreview(Source, CL, CR, CT, CB, False, False, 2)
def ppCropf(Source, CL, CR, CT, CB): return CPreview(Source, CL, CR, CT, CB, True,  False, 2)
def ppCropt(Source, CL, CR, CT, CB): return CPreview(Source, CL, CR, CT, CB, False, True,  2)
def ppCropp(Source, CL, CR, CT, CB): return CPreview(Source, CL, CR, CT, CB, True,  True,  2)

# -------------------------------------------------------------------------------
#            pppCrop / pppCropf / pppCropt / pppCropp
# -------------------------------------------------------------------------------

def pppCrop(Source,  CL, CR, CT, CB): return CPreview(Source, CL, CR, CT, CB, False, False, 3)
def pppCropf(Source, CL, CR, CT, CB): return CPreview(Source, CL, CR, CT, CB, True,  False, 3)
def pppCropt(Source, CL, CR, CT, CB): return CPreview(Source, CL, CR, CT, CB, False, True,  3)
def pppCropp(Source, CL, CR, CT, CB): return CPreview(Source, CL, CR, CT, CB, True,  True,  3)

# -------------------------------------------------------------------------------
#            qCrop / qCropf / qCropt / qCropp
# -------------------------------------------------------------------------------

def qCrop(Source,  CL, CR, CT, CB): return CPreview(Source, CL, CR, CT, CB, False, False, 4)
def qCropf(Source, CL, CR, CT, CB): return CPreview(Source, CL, CR, CT, CB, True,  False, 4)
def qCropt(Source, CL, CR, CT, CB): return CPreview(Source, CL, CR, CT, CB, False, True,  4)
def qCropp(Source, CL, CR, CT, CB): return CPreview(Source, CL, CR, CT, CB, True,  True,  4)

# -------------------------------------------------------------------------------
#            qqCrop / qqCropf / qqCropt / qqCropp
# -------------------------------------------------------------------------------

def qqCrop(Source,  CL, CR, CT, CB): return CPreview(Source, CL, CR, CT, CB, False, False, 5)
def qqCropf(Source, CL, CR, CT, CB): return CPreview(Source, CL, CR, CT, CB, True,  False, 5)
def qqCropt(Source, CL, CR, CT, CB): return CPreview(Source, CL, CR, CT, CB, False, True,  5)
def qqCropp(Source, CL, CR, CT, CB): return CPreview(Source, CL, CR, CT, CB, True,  True,  5)

# -------------------------------------------------------------------------------
#            qqqCrop / qqqCropf / qqqCropt / qqqCropp
# -------------------------------------------------------------------------------

def qqqCrop(Source,  CL, CR, CT, CB): return CPreview(Source, CL, CR, CT, CB, False, False, 6)
def qqqCropf(Source, CL, CR, CT, CB): return CPreview(Source, CL, CR, CT, CB, True,  False, 6)
def qqqCropt(Source, CL, CR, CT, CB): return CPreview(Source, CL, CR, CT, CB, False, True,  6)
def qqqCropp(Source, CL, CR, CT, CB): return CPreview(Source, CL, CR, CT, CB, True,  True,  6)

# -------------------------------------------------------------------------------
#            Cropf / Cropt / Cropp
# -------------------------------------------------------------------------------

def Cropf(Source, CL, CR, CT, CB): return CP_Position(core.std.Crop(Source, CL, CR, CT, CB), True, False)
def Cropt(Source, CL, CR, CT, CB): return CP_Position(core.std.Crop(Source, CL, CR, CT, CB), False, True)
def Cropp(Source, CL, CR, CT, CB): return CP_Position(core.std.Crop(Source, CL, CR, CT, CB), True,  True)

# ===============================================================================
# ===============================================================================
#            VapourSynth's Crop
# ===============================================================================
# ===============================================================================

def Crop(Source, CL, CR, CT, CB): return core.std.Crop(Source, CL, CR, CT, CB)

# ===============================================================================
# ===============================================================================
#            RGBColor
# ===============================================================================
# ===============================================================================

def RGBColor(clip, color=None, matrix=None, range=None):

    if not isinstance(clip, vs.VideoNode): raise vs.Error('CPreview (RGBColor): clip must be a video')

    PropMatrix = clip.get_frame(0).props.get("_Matrix", -1)
    PropRange = clip.get_frame(0).props.get("_ColorRange", -1)

# -------------------------------------------------------------------------------

    if not ((-1 <= PropMatrix <= 2) or (4 <= PropMatrix <= 10)):
        raise vs.Error('CPreview (RGBColor): Video has an unsupported value ' + \
          f'({PropMatrix}) for "_Matrix" in frame properties')
    if not (-1 <= PropRange <= 1):
        raise vs.Error('CPreview (RGBColor): Video has an unsupported value ' + \
          f'({PropRange}) for "_ColorRange" in frame properties')
    if (clip.format.name).endswith("H"):
        raise vs.Error(f'CPreview (RGBColor): {clip.format.name} is not supported')
    if not ((color is None) or isinstance(color, str)):
        raise vs.Error('CPreview (RGBColor): "color" must be a string, for example "darkblue" or "00008B"')
    if not ((matrix is None) or isinstance(matrix, str) or \
      (isinstance(matrix, int) and (isinstance(matrix, bool) is False))):
        raise vs.Error('CPreview (RGBColor): matrix must be an integer or string')
    if (matrix is not None) and (clip.format.color_family == vs.RGB):
        raise vs.Error('CPreview (RGBColor): A matrix cannot be specified for an RGB source')
    if not ((range is None) or (isinstance(range, str) and \
      ((range.lower().strip() == 'full') or (range.lower().strip() == 'limited') or \
      (range.lower().strip() == 'f') or (range.lower().strip() == 'l')))):
        raise vs.Error('CPreview (RGBColor): ' + \
          'range must be "full" or "f", or "limited" or "l" (not case sensitive)')

# -------------------------------------------------------------------------------
#            Color
# -------------------------------------------------------------------------------

    color = 'black' if (color is None) else color.lower().strip()

    Colors = {'aliceblue'            : 'F0F8FF',
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

    if (Colors.get(color) is None) and (color.upper() not in Colors.values()):
        raise vs.Error('CPreview (RGBColor): Invalid color string specified')

    if (Colors.get(color) is not None):
        v0, v1, v2 = tuple(int(Colors.get(color)[i : i + 2], 16) for i in (0, 2, 4))

    else: v0, v1, v2 = tuple(int(color[i : i + 2], 16) for i in (0, 2, 4))

# -------------------------------------------------------------------------------
#            Matrix For YUV
# -------------------------------------------------------------------------------

    MatrixNum = None

    if (clip.format.color_family != vs.RGB):

        matrix = matrix if (matrix is None) or isinstance(matrix, int) else matrix.lower().strip()

        MatrixNum = None if (matrix is None) else \
          (matrix if (0 <= matrix <= 1) or (4 <= matrix <= 10) else -1) \
          if isinstance(matrix, int) else \
          1 if (matrix == '709') else 4 if (matrix == 'fcc') else \
          5 if (matrix == '470bg') else 6 if (matrix == '170m') else \
          7 if (matrix == '240m') else 8 if (matrix == 'ycgco') else \
          9 if (matrix == '2020ncl') else 10 if (matrix == '2020cl') else -1

        if (matrix is not None) and (MatrixNum == -1):
            raise vs.Error('CPreview (RGBColor): Unsupported matrix specified')

        PropMatrix = None if (PropMatrix == -1) or (PropMatrix == 2) else PropMatrix

        if (matrix is not None) and (PropMatrix is not None) and (MatrixNum != PropMatrix):
            raise vs.Error(f'CPreview (RGBColor): The value for "_Matrix" ({PropMatrix}) ' + \
              'in frame properties doesn\'t match the specified matrix')

        MatrixNum = matrix if (matrix is not None) else \
          PropMatrix if (PropMatrix is not None) else \
          5 if (clip.width <= 1056) and (clip.height < 600) else \
          9 if (clip.width > 1920) and (clip.height > 1080) else 1

# -------------------------------------------------------------------------------
#            Color Range
# -------------------------------------------------------------------------------

    range = None if (range is None) else range.lower().strip()

    range = None if (range is None) else \
      'limited' if (range == 'limited') or (range == 'l') else \
      'full' if (range == 'full') or (range == 'f') else None

    PropRangeStr = "limited" if (PropRange == 1) else "full" if (PropRange == 0) else None

    if (range is not None) and (PropRangeStr is not None) and (range != PropRangeStr):
        raise vs.Error(f'CPreview (RGBColor): The value for "_ColorRange" ({PropRange}) ' + \
          'in frame properties doesn\'t match the specified range')

    range = PropRangeStr if (PropRangeStr is not None) else range

# -------------------------------------------------------------------------------
#            RGBColor Output
# -------------------------------------------------------------------------------

    BlankRGB = core.std.BlankClip(color=(v0, v1, v2))
    ColorClip = BlankRGB.resize.Point(format=clip.format.id, matrix=MatrixNum, range_s=range)

    f = ColorClip.get_frame(0)
    p0 = f[0][0,0]

    return p0 if (clip.format.color_family == vs.GRAY) else (p0, f[1][0,0], f[2][0,0])

# ===============================================================================
# ===============================================================================
#            Picture Mod
# ===============================================================================
# ===============================================================================

def CR_PicMod(W, H):

    WMod = 16 if (W % 16 == 0) else 8 if (W % 8 == 0) else 4 if (W % 4 == 0) else 2 if (W % 2 == 0) else 1
    HMod = 16 if (H % 16 == 0) else 8 if (H % 8 == 0) else 4 if (H % 4 == 0) else 2 if (H % 2 == 0) else 1

    return f'  (Mod {WMod}x{HMod})'

# ===============================================================================
# ===============================================================================
#            Position
# ===============================================================================
# ===============================================================================

def CP_Position(Source, Frame, Time, SubText=""):

    if (Frame or Time) and Source.format.name.endswith("H"):
        raise vs.Error('CPreview: The Cropf, Cropt & Cropp functions cannot be used with ' + \
          f'{Source.format.name} as half float formats are not supported by the VapourSynth Text function')

    return core.std.FrameEval(Source, \
      partial(CP_Pos, Source=Source, Frame=Frame, Time=Time, SubText=SubText))

# ---------------------------------------

def CP_Pos(n, Source, Frame, Time, SubText):

    FRateNum = Source.fps.numerator
    FRateDen = Source.fps.denominator
    Time = Time if (FRateNum > 0 < FRateDen) else False

    if Time:

        F_Position = n * FRateDen / FRateNum
        HH = m.floor(F_Position / 3600)
        MM = m.floor(F_Position / 60) % 60
        SS = m.floor(F_Position) % 60
        MS = m.floor((F_Position - m.floor(F_Position)) * 1000)
        Pos = "{:02.0f}:{:02.0f}:{:02.0f}.{:03.0f}".format(HH, MM, SS, MS) + "\n\n"

    PosText = str(n) + "\n" + Pos if Frame and Time else str(n) + "\n\n" if Frame else Pos if Time else ""

    return core.text.Text(Source, PosText + SubText, alignment=5)

# ===============================================================================
# ===============================================================================
#            Line Thickness For pCrop & qCrop
# ===============================================================================
# ===============================================================================
#
#  Modifying the return value of the functions below will change
#  the thickness of the lines in the pCrop & qCrop previews.
#  These funtions are required and cannot be commented out.
#
# -------------------------------------------------------------------------------

def P_Line(W, H): return 1 if (W <= 1920) and (H <= 1080) else 2
def Q_Line(W, H): return 1 if (W <= 1920) and (H <= 1080) else 2

# ===============================================================================
# ===============================================================================
