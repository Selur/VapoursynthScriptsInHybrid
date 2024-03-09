# ===============================================================================
# ===============================================================================
#            RGBColor 2023-10-02
# ===============================================================================
# ===============================================================================

import vapoursynth as vs
core = vs.core

def RGBColor(clip, color=None, matrix=None, range=None):

    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('RGBColor: clip must be a video')
    if not ((color is None) or isinstance(color, str)):
        raise vs.Error('RGBColor: color must be a string, for example \'darkblue\' or \'00008B\'')
    if not ((matrix is None) or isinstance(matrix, str) or \
      (isinstance(matrix, int) and (isinstance(matrix, bool) is False))):
        raise vs.Error('RGBColor: matrix must be an integer or string')
    if not ((range is None) or (isinstance(range, str) and \
      ((range.lower().strip() == 'full') or (range.lower().strip() == 'limited') or \
      (range.lower().strip() == 'f') or (range.lower().strip() == 'l')))):
        raise vs.Error('RGBColor: range must be \'full\' or \'f\', or \'limited\' or \'l\'')
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
              'gray20'               : '323232',
              'gray30'               : '4B4B4B',
              'gray40'               : '656565',
              'gray50'               : '7F7F7F',
              'gray60'               : '989898',
              'gray70'               : 'B1B1B1',
              'gray80'               : 'CACACA',
              'gray90'               : 'E4E4E4',
              'grey10'               : '191919',
              'grey20'               : '323232',
              'grey30'               : '4B4B4B',
              'grey40'               : '656565',
              'grey50'               : '7F7F7F',
              'grey60'               : '989898',
              'grey70'               : 'B1B1B1',
              'grey80'               : 'CACACA',
              'grey90'               : 'E4E4E4'}

    if (colors.get(color) is None):

        try: v0, v1, v2 = tuple(int(color[i:i+2], 16) for i in (0, 2, 4))
        except: raise vs.Error('RGBColor: Invalid color string specified')

    else:

        v0, v1, v2 = tuple(int(colors.get(color)[i:i+2], 16) for i in (0, 2, 4))

    if (clip.format.color_family != vs.RGB):

        matrix = matrix if (matrix is None) or isinstance(matrix, int) else \
          matrix.lower().strip()

        if (matrix is not None) and isinstance(matrix, str):

            M = \
              1 if (matrix == '709') else \
              4 if (matrix == 'fcc') else \
              5 if (matrix == '470bg') else \
              6 if (matrix == '170m') else \
              7 if (matrix == '240m') else \
              8 if (matrix == 'ycgco') else \
              9 if (matrix == '2020ncl') else \
             10 if (matrix == '2020cl') else -1

        if (matrix is not None) and isinstance(matrix, int):

            M = \
              1 if (matrix == 1) else \
              4 if (matrix == 4) else \
              5 if (matrix == 5) else \
              6 if (matrix == 6) else \
              7 if (matrix == 7) else \
              8 if (matrix == 8) else \
              9 if (matrix == 9) else \
             10 if (matrix == 10) else -1

        if (matrix is not None) and (M == -1):
            raise vs.Error('RGBColor: Unsupported matrix specified')

        MP = clip.get_frame(0).props.get("_Matrix", None)

        if (MP is not None) and (MP > 10):
            raise vs.Error('RGBColor: Video has an unsupported value (' + str(MP) + \
                           ') for "_Matrix" in frame properties')

        MatrixProp = MP if (MP is None) else MP if (MP == 1) or (4 <= MP <= 10) else None

        if (matrix is not None) and (MatrixProp is not None) and (M != MatrixProp):
            raise vs.Error('RGBColor: The value for "_Matrix" (' + str(MatrixProp) + \
                           ') in frame properties doesn\'t match the specified matrix')

        matrix = M if (matrix is not None) else \
          MatrixProp if (MatrixProp is not None) else \
          5 if (clip.width <= 1024) and (clip.height <= 576) else \
          1 if (clip.width <= 1920) and (clip.height <= 1080) else 9

    range = range if (range is None) else range.lower().strip()
    range = range if (range is None) else \
      0 if (range == 'full') or (range == 'f') else \
      1 if (range == 'limited') or (range == 'l') else None

    RP = clip.get_frame(0).props.get("_ColorRange", None)
    RangeProp = RP if (RP is None) else RP if (RP == 0) or (RP == 1) else None

    if (range is not None) and (RangeProp is not None) and (range != RangeProp):
        raise vs.Error('RGBColor: The value for "_ColorRange" (' + str(RangeProp) + \
                       ') in frame properties doesn\'t match the specified range')

    range = range if (range is not None) else RangeProp

    BlankRGBClip = core.std.BlankClip(color=(v0, v1, v2))
    ColorClip = BlankRGBClip.resize.Point(format=clip.format.id, matrix=matrix, range=range)

    f = ColorClip.get_frame(0)
    p0 = f[0][0,0]

    return p0 if (clip.format.color_family == vs.GRAY) else (p0, f[1][0,0], f[2][0,0])

# ===============================================================================
# ===============================================================================
