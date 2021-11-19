from vapoursynth import core, RGB, RGBH, RGBS

__version__ = '2.1.2'


def resample_hq(clip, width=None, height=None, kernel='spline36', matrix=None, matrix_dst=None,
                transfer=None, transfer_dst=None, src_left=None, src_top=None, src_width=None,
                src_height=None, descale=False, filter_param_a=None, filter_param_b=None,
                range_in=None, precision=1, hdr=False):
    """Gamma correct resizing in linear light (RGB).

    Args:
        width (int): The target width.
        height (int): The target height.
        kernel (string): The kernel to use while resizing.
            Default is "spline36".
        matrix (string): The source matrix. Default is automatically decided bases on input clip
            vertical resolution. Ignored if source colorspace is RGB.
        matrix_dst (string): The destination matrix. Default is automatically decided bases on output
            clip vertical resolution. Ignored if source colorspace is RGB.
        transfer (string): The transfer matrix. Default is automatically decided bases on input clip
            vertical resolution.
        transfer_dst (string): The destination transfer matrix. Default is automatically decided bases
            on output clip resolution.
        src_left (int): A sub‐pixel offset to crop the source from the left.
            Default 0.
        src_top (int): A sub‐pixel offset to crop the source from the top.
            Default 0.
        src_width (int): A sub‐pixel width to crop the source to. If negative,
            specifies offset from the right. Default is source width−src_left.
        src_height (int): A sub‐pixel height to crop the source to.
            If negative, specifies offset from the bottom.
            Default is source height − src_top.
        descale (bool): Activates the kernel inversion mode, allowing to “undo” a previous upsizing
            by compensating the loss in high frequencies, giving a sharper and more accurate output
            than classic kernels, closer to the original. Default is False.
        filter_param_a (float): For the bicubic filter, filter_param_a represent the “b” parameter ,
            for the lanczos filter, it represents the number of taps.
        filter_param_b (float): For the bicubic filter, it represent the “c” parameter.
        range_in (bool): Range of the input video, either "limited" or "full". Default is "limited".
        precision (bool): 0 uses half float precision , 1 uses single float precision. Default is 1.
        hdr (bool): If set to True, matrix and transfer hdr coefficients will be used for input and
            output clips. Manually set settings will be used instead of hdr automatic ones.
            Keep in mind this is not a color grading tool and should not be used as such.
    """

    # Cheks

    if kernel == 'point' and descale is True:
        raise ValueError('Descale does not support point resizer.')

    if not isinstance(descale, bool):
        raise ValueError('"descale" must be True or False.')

    if precision < 0 or precision > 1:
        raise ValueError('"precision" must be either 0 (half) or 1 (single).')

    # Var stuff

    if descale is True:
        precision = 1

    kernel = kernel.lower().strip()

    if kernel == 'point':
        scaler = core.resize.Point
    elif kernel == 'linear' or kernel == 'bilinear':
        if descale is False:
            scaler = core.resize.Bilinear
        else:
            scaler = core.descale.Debilinear
    elif kernel == 'cubic' or kernel == 'bicubic':
        if descale is False:
            scaler = core.resize.Bicubic
        else:
            scaler = core.descale.Debicubic
    elif kernel == 'lanczos':
        if descale is False:
            scaler = core.resize.Lanczos
        else:
            scaler = core.descale.Delanczos
    elif kernel == 'spline16':
        if descale is False:
            scaler = core.resize.Spline16
        else:
            scaler = core.descale.Despline16
    elif kernel == 'spline36':
        if descale is False:
            scaler = core.resize.Spline36
        else:
            scaler = core.descale.Despline36

    scaler_opts = dict(width=width, height=height)

    if descale is True:
        scaler_opts.update(src_top=src_top, src_left=src_left)
        if kernel == 'cubic' or kernel == 'bicubic':
            scaler_opts.update(b=filter_param_a, c=filter_param_b)
        elif kernel == 'lanczos':
            scaler_opts.update(taps=filter_param_a)
    else:
        scaler_opts.update(src_left=src_left, src_top=src_top,
                           src_width=src_width, src_height=src_height,
                           filter_param_a=filter_param_a, filter_param_b=filter_param_b)

    if range_in is None:
        if clip.format.color_family == RGB:
            range_in = 'full'
        else:
            range_in = 'limited'

    if hdr is True:
        matrix = '2020ncl' if matrix is None else matrix
        matrix_dst = '2020ncl' if matrix_dst is None else matrix_dst
        transfer = 'st2084' if transfer is None else transfer
        transfer_dst = 'st2084' if transfer_dst is None else transfer_dst
    else:
        if clip.height <= 576:
            matrix = '470bg' if matrix is None else matrix
            transfer = '601' if transfer is None else transfer
        else:
            matrix = '709' if matrix is None else matrix
            transfer = '709' if transfer is None else transfer
        if height <= 576:
            matrix_dst = '470bg' if matrix_dst is None else matrix_dst
            transfer_dst = '601' if transfer_dst is None else transfer_dst
        else:
            matrix_dst = '709' if matrix_dst is None else matrix_dst
            transfer_dst = '709' if transfer_dst is None else transfer_dst

    orig_format = clip.format.id

    if precision == 1:
        tmp_format = RGBS
    else:
        tmp_format = RGBH

    to_tmp_format_opts = dict(format=tmp_format, range_in_s=range_in, range_s='full',
                              transfer_in_s=transfer, transfer_s='linear')

    to_orig_format_opts = dict(format=orig_format, transfer_in_s='linear', range_in_s='full', range_s=range_in,
                               transfer_s=transfer_dst)

    if clip.format.color_family != RGB:
        to_tmp_format_opts.update(matrix_in_s=matrix)
        to_orig_format_opts.update(matrix_s=matrix_dst)

    # Do stuff

    clip = core.resize.Bicubic(clip, **to_tmp_format_opts)

    clip = scaler(clip, **scaler_opts)

    clip = core.resize.Bicubic(clip, **to_orig_format_opts)

    return clip
