import vapoursynth as vs
from vapoursynth import core

import math
import importlib
from functools import partial
from typing import Any, Mapping, Optional, Sequence, Union
from vsutil import Dither, depth, fallback, get_y, join, plane, scale_value

QTGMC_globals = {}


def QTGMC(
    Input: vs.VideoNode,
    Preset: str = 'Slower',
    TR0: Optional[int] = None,
    TR1: Optional[int] = None,
    TR2: Optional[int] = None,
    Rep0: Optional[int] = None,
    Rep1: int = 0,
    Rep2: Optional[int] = None,
    EdiMode: Optional[str] = None,
    RepChroma: bool = True,
    NNSize: Optional[int] = None,
    NNeurons: Optional[int] = None,
    EdiQual: int = 1,
    EdiMaxD: Optional[int] = None,
    ChromaEdi: str = '',
    EdiExt: Optional[vs.VideoNode] = None,
    Sharpness: Optional[float] = None,
    SMode: Optional[int] = None,
    SLMode: Optional[int] = None,
    SLRad: Optional[int] = None,
    SOvs: int = 0,
    SVThin: float = 0.0,
    Sbb: Optional[int] = None,
    SrchClipPP: Optional[int] = None,
    SubPel: Optional[int] = None,
    SubPelInterp: int = 2,
    BlockSize: Optional[int] = None,
    Overlap: Optional[int] = None,
    Search: Optional[int] = None,
    SearchParam: Optional[int] = None,
    PelSearch: Optional[int] = None,
    ChromaMotion: Optional[bool] = None,
    TrueMotion: bool = False,
    Lambda: Optional[int] = None,
    LSAD: Optional[int] = None,
    PNew: Optional[int] = None,
    PLevel: Optional[int] = None,
    GlobalMotion: bool = True,
    DCT: int = 0,
    ThSAD1: int = 640,
    ThSAD2: int = 256,
    ThSCD1: int = 180,
    ThSCD2: int = 98,
    SourceMatch: int = 0,
    MatchPreset: Optional[str] = None,
    MatchEdi: Optional[str] = None,
    MatchPreset2: Optional[str] = None,
    MatchEdi2: Optional[str] = None,
    MatchTR2: int = 1,
    MatchEnhance: float = 0.5,
    Lossless: int = 0,
    NoiseProcess: Optional[int] = None,
    EZDenoise: Optional[float] = None,
    EZKeepGrain: Optional[float] = None,
    NoisePreset: str = 'Fast',
    Denoiser: Optional[str] = None,
    FftThreads: int = 1,
    DenoiseMC: Optional[bool] = None,
    NoiseTR: Optional[int] = None,
    Sigma: Optional[float] = None,
    ChromaNoise: bool = False,
    ShowNoise: Union[bool, float] = 0.0,
    GrainRestore: Optional[float] = None,
    NoiseRestore: Optional[float] = None,
    NoiseDeint: Optional[str] = None,
    StabilizeNoise: Optional[bool] = None,
    InputType: int = 0,
    ProgSADMask: Optional[float] = None,
    FPSDivisor: int = 1,
    ShutterBlur: int = 0,
    ShutterAngleSrc: float = 180.0,
    ShutterAngleOut: float = 180.0,
    SBlurLimit: int = 4,
    Border: bool = False,
    Precise: Optional[bool] = None,
    Tuning: str = 'None',
    ShowSettings: bool = False,
    GlobalNames: str = 'QTGMC',
    PrevGlobals: str = 'Replace',
    ForceTR: int = 0,
    Str: float = 2.0,
    Amp: float = 0.0625,
    FastMA: bool = False,
    ESearchP: bool = False,
    RefineMotion: bool = False,
    TFF: Optional[bool] = None,
    nnedi3_args: Mapping[str, Any] = {},
    eedi3_args: Mapping[str, Any] = {},
    opencl: bool = False,
    device: Optional[int] = None,
) -> vs.VideoNode:
    '''
    QTGMC 3.33

    A very high quality deinterlacer with a range of features for both quality and convenience. These include a simple presets system, extensive noise
    processing capabilities, support for repair of progressive material, precision source matching, shutter speed simulation, etc. Originally based on
    TempGaussMC_beta2 by Didée.

    Parameters:
        Input: Clip to process.

        Preset: Sets a range of defaults for different encoding speeds.
            Select from "Placebo", "Very Slow", "Slower", "Slow", "Medium", "Fast", "Faster", "Very Fast", "Super Fast", "Ultra Fast" & "Draft".

        TR0: Temporal binomial smoothing radius used to create motion search clip. In general 2=quality, 1=speed, 0=don't use.

        TR1: Temporal binomial smoothing radius used on interpolated clip for initial output. In general 2=quality, 1=speed, 0=don't use.

        TR2: Temporal linear smoothing radius used for final stablization / denoising. Increase for smoother output.

        Rep0: Repair motion search clip (0=off): repair unwanted blur after temporal smooth TR0 (see QTGMC_KeepOnlyBobShimmerFixes function for details).

        Rep1: Repair initial output clip (0=off): repair unwanted blur after temporal smooth TR1.

        Rep2: Repair final output clip (0=off): unwanted blur after temporal smooth TR2 (will also repair TR1 blur if Rep1 not used).

        EdiMode: Interpolation method, from "NNEDI3", "EEDI3+NNEDI3" (EEDI3 with sclip from NNEDI3), "EEDI3" or "Bwdif", anything else uses "Bob".

        RepChroma: Whether the repair modes affect chroma.

        NNSize: Area around each pixel used as predictor for NNEDI3. A larger area is slower with better quality, read the NNEDI3 docs to see the area choices.
            Note: area sizes are not in increasing order (i.e. increased value doesn't always mean increased quality).

        NNeurons: Controls number of neurons in NNEDI3, larger = slower and better quality but improvements are small.

        EdiQual: Quality setting for NNEDI3. Higher values for better quality - but improvements are marginal.

        EdiMaxD: Spatial search distance for finding connecting edges in EEDI3.

        ChromaEdi: Interpolation method used for chroma. Set to "" to use EdiMode above (default). Otherwise choose from "NNEDI3", "Bwdif" or "Bob" - all high
            speed variants. This can give a minor speed-up if using a very slow EdiMode (i.e. one of the EEDIx modes).

        EdiExt: Provide externally created interpolated clip rather than use one of the above modes.

        Sharpness: How much to resharpen the temporally blurred clip (default is always 1.0 unlike original TGMC).

        SMode: Resharpening mode.
            0 = none
            1 = difference from 3x3 blur kernel
            2 = vertical max/min average + 3x3 kernel

        SLMode: Sharpness limiting.
            0 = off
            [1 = spatial, 2 = temporal]: before final temporal smooth
            [3 = spatial, 4 = temporal]: after final temporal smooth

        SLRad: Temporal or spatial radius used with sharpness limiting (depends on SLMode). Temporal radius can only be 0, 1 or 3.

        SOvs: Amount of overshoot allowed with temporal sharpness limiting (SLMode=2,4), i.e. allow some oversharpening.

        SVThin: How much to thin down 1-pixel wide lines that have been widened due to interpolation into neighboring field lines.

        Sbb: Back blend (blurred) difference between pre & post sharpened clip (minor fidelity improvement).
            0 = off
            1 = before (1st) sharpness limiting
            2 = after (1st) sharpness limiting
            3 = both

        SrchClipPP: Pre-filtering for motion search clip.
            0 = none
            1 = simple blur
            2 = Gauss blur
            3 = Gauss blur + edge soften

        SubPel: Sub-pixel accuracy for motion analysis.
            1 = 1 pixel
            2 = 1/2 pixel
            4 = 1/4 pixel

        SubPelInterp: Interpolation used for sub-pixel motion analysis.
            0 = bilinear (soft)
            1 = bicubic (sharper)
            2 = Weiner (sharpest)

        BlockSize: Size of blocks that are matched during motion analysis.

        Overlap: How much to overlap motion analysis blocks (requires more blocks, but essential to smooth block edges in motion compenstion).

        Search: Search method used for matching motion blocks - see MVTools2 documentation for available algorithms.

        SearchParam: Parameter for search method chosen. For default search method (hexagon search) it is the search range.

        PelSearch: Search parameter (as above) for the finest sub-pixel level (see SubPel).

        ChromaMotion: Whether to consider chroma when analyzing motion. Setting to false gives good speed-up,
            but may very occasionally make incorrect motion decision.

        TrueMotion: Whether to use the 'truemotion' defaults from MAnalyse (see MVTools2 documentation).

        Lambda: Motion vector field coherence - how much the motion analysis favors similar motion vectors for neighboring blocks.
            Should be scaled by BlockSize*BlockSize/64.

        LSAD: How much to reduce need for vector coherence (i.e. Lambda above) if prediction of motion vector from neighbors is poor,
            typically in areas of complex motion. This value is scaled in MVTools (unlike Lambda).

        PNew: Penalty for choosing a new motion vector for a block over an existing one - avoids choosing new vectors for minor gain.

        PLevel: Mode for scaling lambda across different sub-pixel levels - see MVTools2 documentation for choices.

        GlobalMotion: Whether to estimate camera motion to assist in selecting block motion vectors.

        DCT: Modes to use DCT (frequency analysis) or SATD as part of the block matching process - see MVTools2 documentation for choices.

        ThSAD1: SAD threshold for block match on shimmer-removing temporal smooth (TR1). Increase to reduce bob-shimmer more (may smear/blur).

        ThSAD2: SAD threshold for block match on final denoising temporal smooth (TR2). Increase to strengthen final smooth (may smear/blur).

        ThSCD1: Scene change detection parameter 1 - see MVTools documentation.

        ThSCD2: Scene change detection parameter 2 - see MVTools documentation.

        SourceMatch:
            0 = source-matching off (standard algorithm)
            1 = basic source-match
            2 = refined match
            3 = twice refined match

        MatchPreset: Speed/quality for basic source-match processing, select from "Placebo", "Very Slow", "Slower", "Slow", "Medium", "Fast", "Faster",
            "Very Fast", "Super Fast", "Ultra Fast" ("Draft" is not supported). Ideal choice is the same as main preset,
            but can choose a faster setting (but not a slower setting). Default is 3 steps faster than main Preset.

        MatchEdi: Override default interpolation method for basic source-match. Default method is same as main EdiMode setting (usually NNEDI3).
            Only need to override if using slow method for main interpolation (e.g. EEDI3) and want a faster method for source-match.

        MatchPreset2: Speed/quality for refined source-match processing, select from "Placebo", "Very Slow", "Slower", "Slow", "Medium", "Fast", "Faster",
            "Very Fast", "Super Fast", "Ultra Fast" ("Draft" is not supported). Default is 2 steps faster than MatchPreset.
            Faster settings are usually sufficient but can use slower settings if you get extra aliasing in this mode.

        MatchEdi2: Override interpolation method for refined source-match. Can be a good idea to pick MatchEdi2="Bob" for speed.

        MatchTR2: Temporal radius for refined source-matching. 2=smoothness, 1=speed/sharper, 0=not recommended. Differences are very marginal.
            Basic source-match doesn't need this setting as its temporal radius must match TR1 core setting (i.e. there is no MatchTR1).

        MatchEnhance: Enhance the detail found by source-match modes 2 & 3. A slight cheat - will enhance noise if set too strong. Best set < 1.0.

        Lossless: Puts exact source fields into result & cleans any artefacts. 0=off, 1=after final temporal smooth, 2=before resharpening.
            Adds some extra detail but: mode 1 gets shimmer / minor combing, mode 2 is more stable/tweakable but not exactly lossless.

        NoiseProcess: Bypass mode.
            0 = disable
            1 = denoise source & optionally restore some noise back at end of script [use for stronger denoising]
            2 = identify noise only & optionally restore some after QTGMC smoothing [for grain retention / light denoising]

        EZDenoise: Automatic setting to denoise source. Set > 0.0 to enable. Higher values denoise more. Can use ShowNoise to help choose value.

        EZKeepGrain: Automatic setting to retain source grain/detail. Set > 0.0 to enable. Higher values retain more grain. A good starting point = 1.0.

        NoisePreset: Automatic setting for quality of noise processing. Choices: "Slower", "Slow", "Medium", "Fast", and "Faster".

        Denoiser: Select denoiser to use for noise bypass / denoising. Select from "bm3d", "dfttest", "fft3dfilter" or "knlmeanscl".
            Unknown value selects "fft3dfilter".

        FftThreads: Number of threads to use if using "fft3dfilter" for Denoiser.

        DenoiseMC: Whether to provide a motion-compensated clip to the denoiser for better noise vs detail detection (will be a little slower).

        NoiseTR: Temporal radius used when analyzing clip for noise extraction. Higher values better identify noise vs detail but are slower.

        Sigma: Amount of noise known to be in the source, sensible values vary by source and denoiser, so experiment. Use ShowNoise to help.

        ChromaNoise: When processing noise (NoiseProcess > 0), whether to process chroma noise or not (luma noise is always processed).

        ShowNoise: Display extracted and "deinterlaced" noise rather than normal output. Set to true or false, or set a value (around 4 to 16) to specify
            contrast for displayed noise. Visualising noise helps to determine suitable value for Sigma or EZDenoise - want to see noise and noisy detail,
            but not too much clean structure or edges - fairly subjective.

        GrainRestore: How much removed noise/grain to restore before final temporal smooth. Retain "stable" grain and some detail (effect depends on TR2).

        NoiseRestore: How much removed noise/grain to restore after final temporal smooth. Retains any kind of noise.

        NoiseDeint: When noise is taken from interlaced source, how to 'deinterlace' it before restoring.
            "Bob" & "DoubleWeave" are fast but with minor issues: "Bob" is coarse and "Doubleweave" lags by one frame.
            "Generate" is a high quality mode that generates fresh noise lines, but it is slower. Unknown value selects "DoubleWeave".

        StabilizeNoise: Use motion compensation to limit shimmering and strengthen detail within the restored noise. Recommended for "Generate" mode.

        InputType: Default = 0 for interlaced input. Settings 1, 2 & 3 accept progressive input for deshimmer or repair. Frame rate of progressive source is not
            doubled. Mode 1 is for general progressive material. Modes 2 & 3 are designed for badly deinterlaced material.

        ProgSADMask: Only applies to InputType=2,3. If ProgSADMask > 0.0 then blend InputType modes 1 and 2/3 based on block motion SAD.
            Higher values help recover more detail, but repair less artefacts. Reasonable range about 2.0 to 20.0, or 0.0 for no blending.

        FPSDivisor: 1=Double-rate output, 2=Single-rate output. Higher values can be used too (e.g. 60fps & FPSDivisor=3 gives 20fps output).

        ShutterBlur: 0=Off, 1=Enable, 2,3=Higher precisions (slower). Higher precisions reduce blur "bleeding" into static areas a little.

        ShutterAngleSrc: Shutter angle used in source. If necessary, estimate from motion blur seen in a single frame.
            0=pin-sharp, 360=fully blurred from frame to frame.

        ShutterAngleOut: Shutter angle to simulate in output. Extreme values may be rejected (depends on other settings).
            Cannot reduce motion blur already in the source.

        SBlurLimit: Limit motion blur where motion lower than given value. Increase to reduce blur "bleeding". 0=Off. Sensible range around 2-12.

        Border: Pad a little vertically while processing (doesn't affect output size) - set true you see flickering on the very top or bottom line of the
            output. If you have wider edge effects than that, you should crop afterwards instead.

        Precise: Set to false to use faster algorithms with *very* slight imprecision in places.

        Tuning: Tweaks the defaults for different source types. Choose from "None", "DV-SD", "DV-HD".

        ShowSettings: Display all the current parameter values - useful to find preset defaults.

        GlobalNames: The name used to expose intermediate clips to calling script. QTGMC now exposes its motion vectors and other intermediate clips to the
            calling script through global variables. These globals are uniquely named. By default they begin with the prefix "QTGMC_". The available clips are:
                Backward motion vectors                 bVec1, bVec2, bVec3 (temporal radius 1 to 3)
                Forward motion vectors                  fVec1, fVec2, fVec3
                Filtered clip used for motion analysis  srchClip
                MVTools "super" clip for filtered clip  srchSuper
            Not all these clips are necessarily created - it depends on your QTGMC settings. To ensure motion vector creation to radius X, set ForceTR=X
            Clips can be accessed from other scripts with havsfunc.QTGMC_globals['Prefix_Name']

        PrevGlobals: What to do with global variables from earlier QTGMC call that match above name. Either "Replace", or "Reuse" (for a speed-up).
            Set PrevGlobals="Reuse" to reuse existing similar named globals for this run & not recalculate motion vectors etc. This will improve performance.
            Set PrevGlobals="Replace" to overwrite similar named globals from a previous run. This is the default and easiest option for most use cases.

        ForceTR: Ensure globally exposed motion vectors are calculated to this radius even if not needed by QTGMC.

        Str: With this parameter you control the strength of the brightening of the prefilter clip for motion analysis.
            This is good when problems with dark areas arise.

        Amp: Use this together with Str (active when Str is different from 1.0). This defines the amplitude of the brightening in the luma range,
            for example by using 1.0 all the luma range will be used and the brightening will find its peak at luma value 128 in the original.

        FastMA: Use 8-bit for faster motion analysis when using high bit depth input.

        ESearchP: Use wider search range for hex and umh search method.

        RefineMotion: Refines and recalculates motion data of previously estimated motion vectors with new parameters set (e.g. lesser block size).
            The two-stage method may be also useful for more stable (robust) motion estimation.

        TFF: Since VapourSynth only has a weak notion of field order internally, TFF may have to be set. Setting TFF to true means top field first and false
            means bottom field first. Note that the _FieldBased frame property, if present, takes precedence over TFF.

        nnedi3_args: Additional arguments to pass to NNEDI3.

        eedi3_args: Additional arguments to pass to EEDI3.

        opencl: Whether to use the OpenCL version of NNEDI3 and EEDI3.

        device: Sets target OpenCL device.
    '''
    if not isinstance(Input, vs.VideoNode):
        raise vs.Error('QTGMC: this is not a clip')

    if EdiExt is not None:
        if not isinstance(EdiExt, vs.VideoNode):
            raise vs.Error('QTGMC: EdiExt is not a clip')

        if EdiExt.format.id != Input.format.id:
            raise vs.Error('QTGMC: EdiExt must have the same format as input')

    if InputType != 1 and TFF is None:
        with Input.get_frame(0) as f:
            if (field_based := f.props.get('_FieldBased')) not in [1, 2]:
                raise vs.Error('QTGMC: TFF was not specified and field order could not be determined from frame properties')

        TFF = field_based == 2

    is_gray = Input.format.color_family == vs.GRAY

    bits = Input.format.bits_per_sample
    neutral = 1 << (bits - 1)

    SOvs = scale_value(SOvs, 8, bits)

    # ---------------------------------------
    # Presets

    # Select presets / tuning
    Preset = Preset.lower()
    presets = ['placebo', 'very slow', 'slower', 'slow', 'medium', 'fast', 'faster', 'very fast', 'super fast', 'ultra fast', 'draft']
    try:
        pNum = presets.index(Preset)
    except ValueError:
        raise vs.Error("QTGMC: 'Preset' choice is invalid")

    if MatchPreset is None:
        mpNum1 = min(pNum + 3, 9)
        MatchPreset = presets[mpNum1]
    else:
        try:
            mpNum1 = presets[:10].index(MatchPreset.lower())
        except ValueError:
            raise vs.Error("QTGMC: 'MatchPreset' choice is invalid/unsupported")

    if MatchPreset2 is None:
        mpNum2 = min(mpNum1 + 2, 9)
        MatchPreset2 = presets[mpNum2]
    else:
        try:
            mpNum2 = presets[:10].index(MatchPreset2.lower())
        except ValueError:
            raise vs.Error("QTGMC: 'MatchPreset2' choice is invalid/unsupported")

    try:
        npNum = presets[2:7].index(NoisePreset.lower())
    except ValueError:
        raise vs.Error("QTGMC: 'NoisePreset' choice is invalid")

    try:
        tNum = ['none', 'dv-sd', 'dv-hd'].index(Tuning.lower())
    except ValueError:
        raise vs.Error("QTGMC: 'Tuning' choice is invalid")

    # Tunings only affect blocksize in this version
    bs = [16, 16, 32][tNum]
    bs2 = 32

    # fmt: off
    #                                                 Very                                                        Very      Super     Ultra
    # Preset groups:                        Placebo   Slow      Slower    Slow      Medium    Fast      Faster    Fast      Fast      Fast      Draft
    TR0 = fallback(TR0,                   [ 2,        2,        2,        2,        2,        2,        1,        1,        1,        1,        0      ][pNum])
    TR1 = fallback(TR1,                   [ 2,        2,        2,        1,        1,        1,        1,        1,        1,        1,        1      ][pNum])
    TR2X = fallback(TR2,                  [ 3,        2,        1,        1,        1,        0,        0,        0,        0,        0,        0      ][pNum])
    Rep0 = fallback(Rep0,                 [ 4,        4,        4,        4,        3,        3,        0,        0,        0,        0,        0      ][pNum])
    Rep2 = fallback(Rep2,                 [ 4,        4,        4,        4,        4,        4,        4,        4,        3,        3,        0      ][pNum])
    EdiMode = fallback(EdiMode,           ['NNEDI3', 'NNEDI3', 'NNEDI3', 'NNEDI3', 'NNEDI3', 'NNEDI3', 'NNEDI3', 'NNEDI3', 'NNEDI3', 'Bwdif',  'Bob'   ][pNum]).lower()
    NNSize = fallback(NNSize,             [ 1,        1,        1,        1,        5,        5,        4,        4,        4,        4,        4      ][pNum])
    NNeurons = fallback(NNeurons,         [ 2,        2,        1,        1,        1,        0,        0,        0,        0,        0,        0      ][pNum])
    EdiMaxD = fallback(EdiMaxD,           [ 12,       10,       8,        7,        7,        6,        6,        5,        4,        4,        4      ][pNum])
    SMode = fallback(SMode,               [ 2,        2,        2,        2,        2,        2,        2,        2,        2,        2,        0      ][pNum])
    SLModeX = fallback(SLMode,            [ 2,        2,        2,        2,        2,        2,        2,        2,        0,        0,        0      ][pNum])
    SLRad = fallback(SLRad,               [ 3,        1,        1,        1,        1,        1,        1,        1,        1,        1,        1      ][pNum])
    Sbb = fallback(Sbb,                   [ 3,        1,        1,        0,        0,        0,        0,        0,        0,        0,        0      ][pNum])
    SrchClipPP = fallback(SrchClipPP,     [ 3,        3,        3,        3,        3,        2,        2,        2,        1,        1,        0      ][pNum])
    SubPel = fallback(SubPel,             [ 2,        2,        2,        2,        1,        1,        1,        1,        1,        1,        1      ][pNum])
    BlockSize = fallback(BlockSize,       [ bs,       bs,       bs,       bs,       bs,       bs,       bs2,      bs2,      bs2,      bs2,      bs2    ][pNum])
    bs = BlockSize
    Overlap = fallback(Overlap,           [ bs // 2,  bs // 2,  bs // 2,  bs // 2,  bs // 2,  bs // 2,  bs // 2,  bs // 4,  bs // 4,  bs // 4,  bs // 4][pNum])
    Search = fallback(Search,             [ 5,        4,        4,        4,        4,        4,        4,        4,        0,        0,        0      ][pNum])
    SearchParam = fallback(SearchParam,   [ 2,        2,        2,        2,        2,        2,        2,        1,        1,        1,        1      ][pNum])
    PelSearch = fallback(PelSearch,       [ 2,        2,        2,        2,        1,        1,        1,        1,        1,        1,        1      ][pNum])
    ChromaMotion = fallback(ChromaMotion, [ True,     True,     True,     False,    False,    False,    False,    False,    False,    False,    False  ][pNum])
    Precise = fallback(Precise,           [ True,     True,     False,    False,    False,    False,    False,    False,    False,    False,    False  ][pNum])
    ProgSADMask = fallback(ProgSADMask,   [ 10.0,     10.0,     10.0,     10.0,     10.0,     0.0,      0.0,      0.0,      0.0,      0.0,      0.0    ][pNum])

    if ESearchP and Search in [4, 5]:
        if pNum < 4:
            SearchParam = 24
        elif pNum < 8:
            SearchParam = 16

    # Noise presets                             Slower      Slow       Medium     Fast      Faster
    Denoiser = fallback(Denoiser,             ['dfttest',  'dfttest', 'dfttest', 'fft3df', 'fft3df'][npNum]).lower()
    DenoiseMC = fallback(DenoiseMC,           [ True,       True,      False,     False,    False  ][npNum])
    NoiseTR = fallback(NoiseTR,               [ 2,          1,         1,         1,        0      ][npNum])
    NoiseDeint = fallback(NoiseDeint,         ['Generate', 'Bob',      '',        '',       ''     ][npNum]).lower()
    StabilizeNoise = fallback(StabilizeNoise, [ True,       True,      True,      False,    False  ][npNum])
    # fmt: on

    # The basic source-match step corrects and re-runs the interpolation of the input clip. So it initially uses same interpolation settings as the main preset
    MatchNNSize = NNSize
    MatchNNeurons = NNeurons
    MatchEdiMaxD = EdiMaxD
    MatchEdiQual = EdiQual

    # However, can use a faster initial interpolation when using source-match allowing the basic source-match step to "correct" it with higher quality settings
    if SourceMatch > 0 and mpNum1 < pNum:
        raise vs.Error("QTGMC: 'MatchPreset' cannot use a slower setting than 'Preset'")
    # Basic source-match presets
    if SourceMatch > 0:
        # fmt: off
        #                    Very                                      Very  Super  Ultra
        #           Placebo  Slow  Slower  Slow  Medium  Fast  Faster  Fast  Fast   Fast
        NNSize =   [1,       1,    1,      1,    5,      5,    4,      4,    4,     4    ][mpNum1]
        NNeurons = [2,       2,    1,      1,    1,      0,    0,      0,    0,     0    ][mpNum1]
        EdiMaxD =  [12,      10,   8,      7,    7,      6,    6,      5,    4,     4    ][mpNum1]
        EdiQual =  [1,       1,    1,      1,    1,      1,    1,      1,    1,     1    ][mpNum1]
        # fmt: on
    TempEdi = EdiMode  # Main interpolation is actually done by basic-source match step when enabled, so a little swap and wriggle is needed
    if SourceMatch > 0:
        EdiMode = fallback(MatchEdi, EdiMode if mpNum1 < 9 else 'Bwdif').lower()  # Force Bwdif for "Ultra Fast" basic source match
    MatchEdi = TempEdi

    # fmt: off
    #                                           Very                                                        Very      Super    Ultra
    # Refined source-match presets    Placebo   Slow      Slower    Slow      Medium    Fast      Faster    Fast      Fast     Fast
    MatchEdi2 = fallback(MatchEdi2, ['NNEDI3', 'NNEDI3', 'NNEDI3', 'NNEDI3', 'NNEDI3', 'NNEDI3', 'NNEDI3', 'NNEDI3', 'NNEDI3', ''   ][mpNum2]).lower()
    MatchNNSize2 =                  [ 1,        1,        1,        1,        5,        5,        4,        4,        4,       4    ][mpNum2]
    MatchNNeurons2 =                [ 2,        2,        1,        1,        1,        0,        0,        0,        0,       0    ][mpNum2]
    MatchEdiMaxD2 =                 [ 12,       10,       8,        7,        7,        6,        6,        5,        4,       4    ][mpNum2]
    MatchEdiQual2 =                 [ 1,        1,        1,        1,        1,        1,        1,        1,        1,       1    ][mpNum2]
    # fmt: on

    # ---------------------------------------
    # Settings

    # Core defaults
    TR2 = fallback(TR2, max(TR2X, 1)) if SourceMatch > 0 else TR2X  # ***TR2 defaults always at least 1 when using source-match***

    # Source-match defaults
    MatchTR1 = TR1

    # Sharpness defaults. Sharpness default is always 1.0 (0.2 with source-match), but adjusted to give roughly same sharpness for all settings
    if Sharpness is not None and Sharpness <= 0:
        SMode = 0
    SLMode = fallback(SLMode, 0) if SourceMatch > 0 else SLModeX  # ***Sharpness limiting disabled by default for source-match***
    if SLRad <= 0:
        SLMode = 0
    spatialSL = SLMode in [1, 3]
    temporalSL = SLMode in [2, 4]
    Sharpness = fallback(Sharpness, 0.0 if SMode <= 0 else 0.2 if SourceMatch > 0 else 1.0)  # Default sharpness is 1.0, or 0.2 if using source-match
    sharpMul = 2 if temporalSL else 1.5 if spatialSL else 1  # Adjust sharpness based on other settings
    sharpAdj = Sharpness * (sharpMul * (0.2 + TR1 * 0.15 + TR2 * 0.25) + (0.1 if SMode == 1 else 0))  # [This needs a bit more refinement]
    if SMode <= 0:
        Sbb = 0

    # Noise processing settings
    if EZDenoise is not None and EZDenoise > 0 and EZKeepGrain is not None and EZKeepGrain > 0:
        raise vs.Error("QTGMC: EZDenoise and EZKeepGrain cannot be used together")
    if NoiseProcess is None:
        if EZDenoise is not None and EZDenoise > 0:
            NoiseProcess = 1
        elif (EZKeepGrain is not None and EZKeepGrain > 0) or Preset in ['placebo', 'very slow']:
            NoiseProcess = 2
        else:
            NoiseProcess = 0
    if GrainRestore is None:
        if EZDenoise is not None and EZDenoise > 0:
            GrainRestore = 0.0
        elif EZKeepGrain is not None and EZKeepGrain > 0:
            GrainRestore = 0.3 * math.sqrt(EZKeepGrain)
        else:
            GrainRestore = [0.0, 0.7, 0.3][NoiseProcess]
    if NoiseRestore is None:
        if EZDenoise is not None and EZDenoise > 0:
            NoiseRestore = 0.0
        elif EZKeepGrain is not None and EZKeepGrain > 0:
            NoiseRestore = 0.1 * math.sqrt(EZKeepGrain)
        else:
            NoiseRestore = [0.0, 0.3, 0.1][NoiseProcess]
    if Sigma is None:
        if EZDenoise is not None and EZDenoise > 0:
            Sigma = EZDenoise
        elif EZKeepGrain is not None and EZKeepGrain > 0:
            Sigma = 4.0 * EZKeepGrain
        else:
            Sigma = 2.0
    if isinstance(ShowNoise, bool):
        ShowNoise = 10.0 if ShowNoise else 0.0
    if ShowNoise > 0:
        NoiseProcess = 2
        NoiseRestore = 1.0
    if NoiseProcess <= 0:
        NoiseTR = 0
        GrainRestore = 0.0
        NoiseRestore = 0.0
    totalRestore = GrainRestore + NoiseRestore
    if totalRestore <= 0:
        StabilizeNoise = False
    noiseTD = [1, 3, 5][NoiseTR]
    noiseCentre = scale_value(128.5, 8, bits) if Denoiser in ['fft3df', 'fft3dfilter'] else neutral

    # MVTools settings
    Lambda = fallback(Lambda, (1000 if TrueMotion else 100) * BlockSize * BlockSize // 64)
    LSAD = fallback(LSAD, 1200 if TrueMotion else 400)
    PNew = fallback(PNew, 50 if TrueMotion else 25)
    PLevel = fallback(PLevel, 1 if TrueMotion else 0)

    # Motion blur settings
    if ShutterAngleOut * FPSDivisor == ShutterAngleSrc:  # If motion blur output is same as input
        ShutterBlur = 0

    # Miscellaneous
    PrevGlobals = PrevGlobals.lower()
    ReplaceGlobals = PrevGlobals in ['replace', 'reuse']  # If reusing existing globals put them back afterwards - simplifies logic later
    ReuseGlobals = PrevGlobals == 'reuse'
    if InputType < 2:
        ProgSADMask = 0.0

    # Get maximum temporal radius needed
    maxTR = max(SLRad if temporalSL else 0, MatchTR2, TR1, TR2, NoiseTR)
    if (ProgSADMask > 0 or StabilizeNoise or ShutterBlur > 0) and maxTR < 1:
        maxTR = 1
    maxTR = max(ForceTR, maxTR)

    # ---------------------------------------
    # Pre-Processing

    w = Input.width
    h = Input.height

    # Reverse "field" dominance for progressive repair mode 3 (only difference from mode 2)
    if InputType >= 3:
        TFF = not TFF

    # Pad vertically during processing (to prevent artefacts at top & bottom edges)
    if Border:
        h += 8
        clip = Input.resize.Point(w, h, src_top=-4, src_height=h)
    else:
        clip = Input

    hpad = vpad = BlockSize

    # ---------------------------------------
    # Motion Analysis

    # Bob the input as a starting point for motion search clip
    if InputType <= 0:
        bobbed = clip.resize.Bob(tff=TFF, filter_param_a=0, filter_param_b=0.5)
    elif InputType == 1:
        bobbed = clip
    else:
        bobbed = clip.std.Convolution(matrix=[1, 2, 1], mode='v')

    # If required, get any existing global clips with a matching "GlobalNames" setting. Unmatched values get None
    if ReuseGlobals:
        srchClip = QTGMC_GetUserGlobal(GlobalNames, 'srchClip')
        srchSuper = QTGMC_GetUserGlobal(GlobalNames, 'srchSuper')
        bVec1 = QTGMC_GetUserGlobal(GlobalNames, 'bVec1')
        fVec1 = QTGMC_GetUserGlobal(GlobalNames, 'fVec1')
        bVec2 = QTGMC_GetUserGlobal(GlobalNames, 'bVec2')
        fVec2 = QTGMC_GetUserGlobal(GlobalNames, 'fVec2')
        bVec3 = QTGMC_GetUserGlobal(GlobalNames, 'bVec3')
        fVec3 = QTGMC_GetUserGlobal(GlobalNames, 'fVec3')
    else:
        srchClip = srchSuper = bVec1 = fVec1 = bVec2 = fVec2 = bVec3 = fVec3 = None

    CMplanes = [0, 1, 2] if ChromaMotion and not is_gray else [0]

    # The bobbed clip will shimmer due to being derived from alternating fields. Temporally smooth over the neighboring frames using a binomial kernel. Binomial
    # kernels give equal weight to even and odd frames and hence average away the shimmer. The two kernels used are [1 2 1] and [1 4 6 4 1] for radius 1 and 2.
    # These kernels are approximately Gaussian kernels, which work well as a prefilter before motion analysis (hence the original name for this script)
    # Create linear weightings of neighbors first                                                  -2    -1    0     1     2
    if not isinstance(srchClip, vs.VideoNode):
        if TR0 > 0:
            ts1 = core.std.AverageFrames(bobbed, weights=[1] * 3, scenechange=28 / 255, planes=CMplanes)  # 0.00  0.33  0.33  0.33  0.00
        if TR0 > 1:
            ts2 = core.std.AverageFrames(bobbed, weights=[1] * 5, scenechange=28 / 255, planes=CMplanes)  # 0.20  0.20  0.20  0.20  0.20

    # Combine linear weightings to give binomial weightings - TR0=0: (1), TR0=1: (1:2:1), TR0=2: (1:4:6:4:1)
    if isinstance(srchClip, vs.VideoNode):
        binomial0 = None
    elif TR0 <= 0:
        binomial0 = bobbed
    elif TR0 == 1:
        binomial0 = core.std.Merge(ts1, bobbed, weight=0.25 if ChromaMotion or is_gray else [0.25, 0])
    else:
        binomial0 = core.std.Merge(
            core.std.Merge(ts1, ts2, weight=0.357 if ChromaMotion or is_gray else [0.357, 0]), bobbed, weight=0.125 if ChromaMotion or is_gray else [0.125, 0]
        )

    # Remove areas of difference between temporal blurred motion search clip and bob that are not due to bob-shimmer - removes general motion blur
    if isinstance(srchClip, vs.VideoNode) or Rep0 <= 0:
        repair0 = binomial0
    else:
        repair0 = QTGMC_KeepOnlyBobShimmerFixes(binomial0, bobbed, Rep0, RepChroma and ChromaMotion)

    matrix = [1, 2, 1, 2, 4, 2, 1, 2, 1]
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    # Blur image and soften edges to assist in motion matching of edge blocks. Blocks are matched by SAD (sum of absolute differences between blocks), but even
    # a slight change in an edge from frame to frame will give a high SAD due to the higher contrast of edges
    if not isinstance(srchClip, vs.VideoNode):
        if SrchClipPP == 1:
            spatialBlur = repair0.resize.Bilinear(w // 2, h // 2).std.Convolution(matrix=matrix, planes=CMplanes).resize.Bilinear(w, h)
        elif SrchClipPP >= 2:
            spatialBlur = Gauss(repair0.std.Convolution(matrix=matrix, planes=CMplanes), p=2.35)
            spatialBlur = core.std.Merge(spatialBlur, repair0, weight=0.1 if ChromaMotion or is_gray else [0.1, 0])
        if SrchClipPP <= 0:
            srchClip = repair0
        elif SrchClipPP < 3:
            srchClip = spatialBlur
        else:
            expr = 'x {i3} + y < x {i3} + x {i3} - y > x {i3} - y ? ?'.format(i3=scale_value(3, 8, bits))
            tweaked = EXPR([repair0, bobbed], expr=expr if ChromaMotion or is_gray else [expr, ''])
            expr = 'x {i7} + y < x {i2} + x {i7} - y > x {i2} - x 51 * y 49 * + 100 / ? ?'.format(i7=scale_value(7, 8, bits), i2=scale_value(2, 8, bits))
            srchClip = EXPR([spatialBlur, tweaked], expr=expr if ChromaMotion or is_gray else [expr, ''])
        srchClip = DitherLumaRebuild(srchClip, s0=Str, c=Amp, chroma=ChromaMotion)
        if bits > 8 and FastMA:
            srchClip = depth(srchClip, 8, dither_type=Dither.NONE)

    super_args = dict(pel=SubPel, hpad=hpad, vpad=vpad)
    analyse_args = dict(
        blksize=BlockSize,
        overlap=Overlap,
        search=Search,
        searchparam=SearchParam,
        pelsearch=PelSearch,
        truemotion=TrueMotion,
        lambda_=Lambda,
        lsad=LSAD,
        pnew=PNew,
        plevel=PLevel,
        global_=GlobalMotion,
        dct=DCT,
        chroma=ChromaMotion,
    )
    recalculate_args = dict(
        thsad=ThSAD1 // 2,
        blksize=max(BlockSize // 2, 4),
        search=Search,
        searchparam=SearchParam,
        chroma=ChromaMotion,
        truemotion=TrueMotion,
        pnew=PNew,
        overlap=max(Overlap // 2, 2),
        dct=DCT,
    )

    # Calculate forward and backward motion vectors from motion search clip
    if maxTR > 0:
        if not isinstance(srchSuper, vs.VideoNode):
            srchSuper = srchClip.mv.Super(sharp=SubPelInterp, chroma=ChromaMotion, **super_args)
        if not isinstance(bVec1, vs.VideoNode):
            bVec1 = srchSuper.mv.Analyse(isb=True, delta=1, **analyse_args)
            if RefineMotion:
                bVec1 = core.mv.Recalculate(srchSuper, bVec1, **recalculate_args)
        if not isinstance(fVec1, vs.VideoNode):
            fVec1 = srchSuper.mv.Analyse(isb=False, delta=1, **analyse_args)
            if RefineMotion:
                fVec1 = core.mv.Recalculate(srchSuper, fVec1, **recalculate_args)
    if maxTR > 1:
        if not isinstance(bVec2, vs.VideoNode):
            bVec2 = srchSuper.mv.Analyse(isb=True, delta=2, **analyse_args)
            if RefineMotion:
                bVec2 = core.mv.Recalculate(srchSuper, bVec2, **recalculate_args)
        if not isinstance(fVec2, vs.VideoNode):
            fVec2 = srchSuper.mv.Analyse(isb=False, delta=2, **analyse_args)
            if RefineMotion:
                fVec2 = core.mv.Recalculate(srchSuper, fVec2, **recalculate_args)
    if maxTR > 2:
        if not isinstance(bVec3, vs.VideoNode):
            bVec3 = srchSuper.mv.Analyse(isb=True, delta=3, **analyse_args)
            if RefineMotion:
                bVec3 = core.mv.Recalculate(srchSuper, bVec3, **recalculate_args)
        if not isinstance(fVec3, vs.VideoNode):
            fVec3 = srchSuper.mv.Analyse(isb=False, delta=3, **analyse_args)
            if RefineMotion:
                fVec3 = core.mv.Recalculate(srchSuper, fVec3, **recalculate_args)

    # Expose search clip, motion search super clip and motion vectors to calling script through globals
    if ReplaceGlobals:
        QTGMC_SetUserGlobal(GlobalNames, 'srchClip', srchClip)
        QTGMC_SetUserGlobal(GlobalNames, 'srchSuper', srchSuper)
        QTGMC_SetUserGlobal(GlobalNames, 'bVec1', bVec1)
        QTGMC_SetUserGlobal(GlobalNames, 'fVec1', fVec1)
        QTGMC_SetUserGlobal(GlobalNames, 'bVec2', bVec2)
        QTGMC_SetUserGlobal(GlobalNames, 'fVec2', fVec2)
        QTGMC_SetUserGlobal(GlobalNames, 'bVec3', bVec3)
        QTGMC_SetUserGlobal(GlobalNames, 'fVec3', fVec3)

    # ---------------------------------------
    # Noise Processing

    # Expand fields to full frame size before extracting noise (allows use of motion vectors which are frame-sized)
    if NoiseProcess > 0:
        if InputType > 0:
            fullClip = clip
        else:
            fullClip = clip.resize.Bob(tff=TFF, filter_param_a=0, filter_param_b=1)
    if NoiseTR > 0:
        fullSuper = fullClip.mv.Super(levels=1, chroma=ChromaNoise, **super_args)  # TEST chroma OK?

    CNplanes = [0, 1, 2] if ChromaNoise and not is_gray else [0]

    if NoiseProcess > 0:
        # Create a motion compensated temporal window around current frame and use to guide denoisers
        if not DenoiseMC or NoiseTR <= 0:
            noiseWindow = fullClip
        elif NoiseTR == 1:
            noiseWindow = core.std.Interleave(
                [
                    core.mv.Compensate(fullClip, fullSuper, fVec1, thscd1=ThSCD1, thscd2=ThSCD2),
                    fullClip,
                    core.mv.Compensate(fullClip, fullSuper, bVec1, thscd1=ThSCD1, thscd2=ThSCD2),
                ]
            )
        else:
            noiseWindow = core.std.Interleave(
                [
                    core.mv.Compensate(fullClip, fullSuper, fVec2, thscd1=ThSCD1, thscd2=ThSCD2),
                    core.mv.Compensate(fullClip, fullSuper, fVec1, thscd1=ThSCD1, thscd2=ThSCD2),
                    fullClip,
                    core.mv.Compensate(fullClip, fullSuper, bVec1, thscd1=ThSCD1, thscd2=ThSCD2),
                    core.mv.Compensate(fullClip, fullSuper, bVec2, thscd1=ThSCD1, thscd2=ThSCD2),
                ]
            )
        if Denoiser == 'bm3d':
            dnWindow = mvf.BM3D(noiseWindow, radius1=NoiseTR, sigma=[Sigma if plane in CNplanes else 0 for plane in range(3)])
        elif Denoiser == 'dfttest':
          if opencl:
            if hasattr(core, 'dfttest2_nvrtc'):  
              try:
                 import dfttest2
                 dnWindow = dfttest2.DFTTest(noiseWindow, sigma=Sigma * 4, tbsize=noiseTD, planes=CNplanes)
              except ModuleNotFoundError:
                 dnWindow = noiseWindow.dfttest.DFTTest(sigma=Sigma * 4, tbsize=noiseTD, planes=CNplanes)
            else:
              dnWindow = noiseWindow.dfttest.DFTTest(sigma=Sigma * 4, tbsize=noiseTD, planes=CNplanes)   
          else:
            dnWindow = noiseWindow.dfttest.DFTTest(sigma=Sigma * 4, tbsize=noiseTD, planes=CNplanes)
        elif Denoiser in ['knlm', 'knlmeanscl', 'nlm_cuda', 'nlm_ispc']:
            if ChromaNoise and not is_gray:
                dnWindow = KNLMeansCL(noiseWindow, d=NoiseTR, h=Sigma)
            else:
                if hasattr(core, 'nlm_ispc'):  
                  nlmeans_func = noiseWindow.nlm_ispc.NLMeans
                elif hasattr(core, 'nlm_cuda'):
                  nlmeans_func = noiseWindow.nlm_cuda.NLMeans
                else:
                  nlmeans_func = noiseWindow.knlm.KNLMeansCL
                dnWindow = nlmeans_func(d=NoiseTR, h=Sigma)
        else:
            fft3d_func = noiseWindow.neo_fft3d.FFT3D if hasattr(core, 'neo_fft3d') else noiseWindow.fft3dfilter.FFT3DFilter
            dnWindow = fft3d_func(sigma=Sigma, planes=CNplanes, bt=noiseTD, ncpu=FftThreads)


        # Rework denoised clip to match source format - various code paths here: discard the motion compensation window, discard doubled lines (from point resize)
        # Also reweave to get interlaced noise if source was interlaced (could keep the full frame of noise, but it will be poor quality from the point resize)
        if not DenoiseMC:
            if InputType > 0:
                denoised = dnWindow
            else:
                denoised = Weave(dnWindow.std.SeparateFields(tff=TFF).std.SelectEvery(cycle=4, offsets=[0, 3]), tff=TFF)
        elif InputType > 0:
            if NoiseTR <= 0:
                denoised = dnWindow
            else:
                denoised = dnWindow.std.SelectEvery(cycle=noiseTD, offsets=NoiseTR)
        else:
            denoised = Weave(dnWindow.std.SeparateFields(tff=TFF).std.SelectEvery(cycle=noiseTD * 4, offsets=[NoiseTR * 2, NoiseTR * 6 + 3]), tff=TFF)

        if totalRestore > 0:
            # Get actual noise from difference. Then 'deinterlace' where we have weaved noise - create the missing lines of noise in various ways
            noise = core.std.MakeDiff(clip, denoised, planes=CNplanes)
            if InputType > 0:
                deintNoise = noise
            elif NoiseDeint == 'bob':
                deintNoise = noise.resize.Bob(tff=TFF, filter_param_a=0, filter_param_b=0.5)
            elif NoiseDeint == 'generate':
                deintNoise = QTGMC_Generate2ndFieldNoise(noise, denoised, ChromaNoise, TFF)
            else:
                deintNoise = noise.std.SeparateFields(tff=TFF).std.DoubleWeave(tff=TFF)

            # Motion-compensated stabilization of generated noise
            if StabilizeNoise:
                noiseSuper = deintNoise.mv.Super(sharp=SubPelInterp, levels=1, chroma=ChromaNoise, **super_args)
                mcNoise = core.mv.Compensate(deintNoise, noiseSuper, bVec1, thscd1=ThSCD1, thscd2=ThSCD2)
                expr = f'x {neutral} - abs y {neutral} - abs > x y ? 0.6 * x y + 0.2 * +'
                finalNoise = EXPR([deintNoise, mcNoise], expr=expr if ChromaNoise or is_gray else [expr, ''])
            else:
                finalNoise = deintNoise

    # If NoiseProcess=1 denoise input clip. If NoiseProcess=2 leave noise in the clip and let the temporal blurs "denoise" it for a stronger effect
    innerClip = denoised if NoiseProcess == 1 else clip

    # ---------------------------------------
    # Interpolation

    # Support badly deinterlaced progressive content - drop half the fields and reweave to get 1/2fps interlaced stream appropriate for QTGMC processing
    if InputType > 1:
        ediInput = Weave(innerClip.std.SeparateFields(tff=TFF).std.SelectEvery(cycle=4, offsets=[0, 3]), tff=TFF)
    else:
        ediInput = innerClip

    # Create interpolated image as starting point for output
    if EdiExt is not None:
        edi1 = EdiExt.resize.Point(w, h, src_top=(EdiExt.height - h) // 2, src_height=h)
    else:
        edi1 = QTGMC_Interpolate(
            ediInput, InputType, EdiMode, NNSize, NNeurons, EdiQual, EdiMaxD, bobbed, ChromaEdi.lower(), TFF, nnedi3_args, eedi3_args, opencl, device
        )

    # InputType=2,3: use motion mask to blend luma between original clip & reweaved clip based on ProgSADMask setting. Use chroma from original clip in any case
    if InputType < 2:
        edi = edi1
    elif ProgSADMask <= 0:
        if not is_gray:
            edi = core.std.ShufflePlanes([edi1, innerClip], planes=[0, 1, 2], colorfamily=Input.format.color_family)
        else:
            edi = edi1
    else:
        inputTypeBlend = core.mv.Mask(srchClip, bVec1, kind=1, ml=ProgSADMask)
        edi = core.std.MaskedMerge(innerClip, edi1, inputTypeBlend, planes=0)

    # Get the max/min value for each pixel over neighboring motion-compensated frames - used for temporal sharpness limiting
    if TR1 > 0 or temporalSL:
        ediSuper = edi.mv.Super(sharp=SubPelInterp, levels=1, **super_args)
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    if temporalSL:
        bComp1 = core.mv.Compensate(edi, ediSuper, bVec1, thscd1=ThSCD1, thscd2=ThSCD2)
        fComp1 = core.mv.Compensate(edi, ediSuper, fVec1, thscd1=ThSCD1, thscd2=ThSCD2)
        tMax = EXPR([EXPR([edi, fComp1], expr='x y max'), bComp1], expr='x y max')
        tMin = EXPR([EXPR([edi, fComp1], expr='x y min'), bComp1], expr='x y min')
        if SLRad > 1:
            bComp3 = core.mv.Compensate(edi, ediSuper, bVec3, thscd1=ThSCD1, thscd2=ThSCD2)
            fComp3 = core.mv.Compensate(edi, ediSuper, fVec3, thscd1=ThSCD1, thscd2=ThSCD2)
            tMax = EXPR([EXPR([tMax, fComp3], expr='x y max'), bComp3], expr='x y max')
            tMin = EXPR([EXPR([tMin, fComp3], expr='x y min'), bComp3], expr='x y min')

    # ---------------------------------------
    # Create basic output

    # Use motion vectors to blur interpolated image (edi) with motion-compensated previous and next frames. As above, this is done to remove shimmer from
    # alternate frames so the same binomial kernels are used. However, by using motion-compensated smoothing this time we avoid motion blur. The use of
    # MDegrain1 (motion compensated) rather than TemporalSmooth makes the weightings *look* different, but they evaluate to the same values
    # Create linear weightings of neighbors first                                                               -2    -1    0     1     2
    if TR1 > 0:
        degrain1 = core.mv.Degrain1(edi, ediSuper, bVec1, fVec1, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)  # 0.00  0.33  0.33  0.33  0.00
    if TR1 > 1:
        degrain2 = core.mv.Degrain1(edi, ediSuper, bVec2, fVec2, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)  # 0.33  0.00  0.33  0.00  0.33

    # Combine linear weightings to give binomial weightings - TR1=0: (1), TR1=1: (1:2:1), TR1=2: (1:4:6:4:1)
    if TR1 <= 0:
        binomial1 = edi
    elif TR1 == 1:
        binomial1 = core.std.Merge(degrain1, edi, weight=0.25)
    else:
        binomial1 = core.std.Merge(core.std.Merge(degrain1, degrain2, weight=0.2), edi, weight=0.0625)

    # Remove areas of difference between smoothed image and interpolated image that are not bob-shimmer fixes: repairs residual motion blur from temporal smooth
    if Rep1 <= 0:
        repair1 = binomial1
    else:
        repair1 = QTGMC_KeepOnlyBobShimmerFixes(binomial1, edi, Rep1, RepChroma)

    # Apply source match - use difference between output and source to succesively refine output [extracted to function to clarify main code path]
    if SourceMatch <= 0:
        match = repair1
    else:
        match = QTGMC_ApplySourceMatch(
            repair1,
            InputType,
            ediInput,
            bVec1 if maxTR > 0 else None,
            fVec1 if maxTR > 0 else None,
            bVec2 if maxTR > 1 else None,
            fVec2 if maxTR > 1 else None,
            SubPel,
            SubPelInterp,
            hpad,
            vpad,
            ThSAD1,
            ThSCD1,
            ThSCD2,
            SourceMatch,
            MatchTR1,
            MatchEdi,
            MatchNNSize,
            MatchNNeurons,
            MatchEdiQual,
            MatchEdiMaxD,
            MatchTR2,
            MatchEdi2,
            MatchNNSize2,
            MatchNNeurons2,
            MatchEdiQual2,
            MatchEdiMaxD2,
            MatchEnhance,
            TFF,
            nnedi3_args,
            eedi3_args,
            opencl,
            device,
        )

    # Lossless=2 - after preparing an interpolated, de-shimmered clip, restore the original source fields into it and clean up any artefacts
    # This mode will not give a true lossless result because the resharpening and final temporal smooth are still to come, but it will add further detail
    # However, it can introduce minor combing. This setting is best used together with source-match (it's effectively the final source-match stage)
    if Lossless >= 2:
        lossed1 = QTGMC_MakeLossless(match, innerClip, InputType, TFF)
    else:
        lossed1 = match

    # ---------------------------------------
    # Resharpen / retouch output

    # Resharpen to counteract temporal blurs. Little sharpening needed for source-match mode since it has already recovered sharpness from source
    if SMode <= 0:
        resharp = lossed1
    elif SMode == 1:
        resharp = EXPR([lossed1, lossed1.std.Convolution(matrix=matrix)], expr=f'x x y - {sharpAdj} * +')
    else:
        vresharp1 = core.std.Merge(lossed1.std.Maximum(coordinates=[0, 1, 0, 0, 0, 0, 1, 0]), lossed1.std.Minimum(coordinates=[0, 1, 0, 0, 0, 0, 1, 0]))
        if Precise:  # Precise mode: reduce tiny overshoot
            vresharp = EXPR([vresharp1, lossed1], expr='x y < x {i1} + x y > x {i1} - x ? ?'.format(i1=scale_value(1, 8, bits)))
        else:
            vresharp = vresharp1
        resharp = EXPR([lossed1, vresharp.std.Convolution(matrix=matrix)], expr=f'x x y - {sharpAdj} * +')

    # Slightly thin down 1-pixel high horizontal edges that have been widened into neighboring field lines by the interpolator
    SVThinSc = SVThin * 6.0
    zsmooth = hasattr(core,'zsmooth')
    if SVThin > 0:
        expr = f'y x - {SVThinSc} * {neutral} +'
        if zsmooth:
          vertMedD = EXPR([lossed1, lossed1.zsmooth.VerticalCleaner(mode=1 if is_gray else [1, 0])], expr=expr if is_gray else [expr, ''])
        else:
          vertMedD = EXPR([lossed1, lossed1.rgvs.VerticalCleaner(mode=1 if is_gray else [1, 0])], expr=expr if is_gray else [expr, ''])
        vertMedD = vertMedD.std.Convolution(matrix=[1, 2, 1], planes=0, mode='h')
        expr = f'y {neutral} - abs x {neutral} - abs > y {neutral} ?'
        neighborD = EXPR([vertMedD, vertMedD.std.Convolution(matrix=matrix, planes=0)], expr=expr if is_gray else [expr, ''])
        thin = core.std.MergeDiff(resharp, neighborD, planes=0)
    else:
        thin = resharp

    # Back blend the blurred difference between sharpened & unsharpened clip, before (1st) sharpness limiting (Sbb == 1,3). A small fidelity improvement
    if Sbb not in [1, 3]:
        backBlend1 = thin
    else:
        backBlend1 = core.std.MakeDiff(thin, Gauss(core.std.MakeDiff(thin, lossed1, planes=0).std.Convolution(matrix=matrix, planes=0), p=5), planes=0)

    # Limit over-sharpening by clamping to neighboring (spatial or temporal) min/max values in original
    # Occurs here (before final temporal smooth) if SLMode == 1,2. This location will restrict sharpness more, but any artefacts introduced will be smoothed
    if SLMode == 1:
        if SLRad <= 1:
            sharpLimit1 = core.zsmooth.Repair(backBlend1, edi, mode=1) if zsmooth else core.rgvs.Repair(backBlend1, edi, mode=1)
        else:
            sharpLimit1 = core.zsmooth.Repair(backBlend1, core.zsmooth.Repair(backBlend1, edi, mode=12), mode=1) if zsmooth else core.rgvs.Repair(backBlend1, core.rgvs.Repair(backBlend1, edi, mode=12), mode=1)
    elif SLMode == 2:
        sharpLimit1 = mt_clamp(backBlend1, tMax, tMin, SOvs, SOvs)
    else:
        sharpLimit1 = backBlend1

    # Back blend the blurred difference between sharpened & unsharpened clip, after (1st) sharpness limiting (Sbb == 2,3). A small fidelity improvement
    if Sbb < 2:
        backBlend2 = sharpLimit1
    else:
        backBlend2 = core.std.MakeDiff(
            sharpLimit1, Gauss(core.std.MakeDiff(sharpLimit1, lossed1, planes=0).std.Convolution(matrix=matrix, planes=0), p=5), planes=0
        )

    # Add back any extracted noise, prior to final temporal smooth - this will restore detail that was removed as "noise" without restoring the noise itself
    # Average luma of FFT3DFilter extracted noise is 128.5, so deal with that too
    if GrainRestore <= 0:
        addNoise1 = backBlend2
    else:
        expr = f'x {noiseCentre} - {GrainRestore} * {neutral} +'
        addNoise1 = core.std.MergeDiff(backBlend2, EXPR(finalNoise, expr=expr if ChromaNoise or is_gray else [expr, '']), planes=CNplanes)

    # Final light linear temporal smooth for denoising
    if TR2 > 0:
        stableSuper = addNoise1.mv.Super(sharp=SubPelInterp, levels=1, **super_args)
    if TR2 <= 0:
        stable = addNoise1
    elif TR2 == 1:
        stable = core.mv.Degrain1(addNoise1, stableSuper, bVec1, fVec1, thsad=ThSAD2, thscd1=ThSCD1, thscd2=ThSCD2)
    elif TR2 == 2:
        stable = core.mv.Degrain2(addNoise1, stableSuper, bVec1, fVec1, bVec2, fVec2, thsad=ThSAD2, thscd1=ThSCD1, thscd2=ThSCD2)
    else:
        stable = core.mv.Degrain3(addNoise1, stableSuper, bVec1, fVec1, bVec2, fVec2, bVec3, fVec3, thsad=ThSAD2, thscd1=ThSCD1, thscd2=ThSCD2)

    # Remove areas of difference between final output & basic interpolated image that are not bob-shimmer fixes: repairs motion blur caused by temporal smooth
    if Rep2 <= 0:
        repair2 = stable
    else:
        repair2 = QTGMC_KeepOnlyBobShimmerFixes(stable, edi, Rep2, RepChroma)

    # Limit over-sharpening by clamping to neighboring (spatial or temporal) min/max values in original
    # Occurs here (after final temporal smooth) if SLMode == 3,4. Allows more sharpening here, but more prone to introducing minor artefacts
    if SLMode == 3:
        if SLRad <= 1:
            sharpLimit2 = core.zsmooth.Repair(repair2, edi, mode=1) if zsmooth else core.rgvs.Repair(repair2, edi, mode=1)
        else:
            sharpLimit2 = core.zsmooth.Repair(repair2, core.zsmooth.Repair(repair2, edi, mode=12), mode=1) if zsmooth else core.rgvs.Repair(repair2, core.rgvs.Repair(repair2, edi, mode=12), mode=1)
    elif SLMode >= 4:
        sharpLimit2 = mt_clamp(repair2, tMax, tMin, SOvs, SOvs)
    else:
        sharpLimit2 = repair2

    # Lossless=1 - inject source fields into result and clean up inevitable artefacts. Provided NoiseRestore=0.0 or 1.0, this mode will make the script result
    # properly lossless, but this will retain source artefacts and cause some combing (where the smoothed deinterlace doesn't quite match the source)
    if Lossless == 1:
        lossed2 = QTGMC_MakeLossless(sharpLimit2, innerClip, InputType, TFF)
    else:
        lossed2 = sharpLimit2

    # Add back any extracted noise, after final temporal smooth. This will appear as noise/grain in the output
    # Average luma of FFT3DFilter extracted noise is 128.5, so deal with that too
    if NoiseRestore <= 0:
        addNoise2 = lossed2
    else:
        expr = f'x {noiseCentre} - {NoiseRestore} * {neutral} +'
        addNoise2 = core.std.MergeDiff(lossed2, EXPR(finalNoise, expr=expr if ChromaNoise or is_gray else [expr, '']), planes=CNplanes)

    # ---------------------------------------
    # Post-Processing

    # Shutter motion blur - get level of blur depending on output framerate and blur already in source
    blurLevel = (ShutterAngleOut * FPSDivisor - ShutterAngleSrc) * 100 / 360
    if blurLevel < 0:
        raise vs.Error('QTGMC: cannot reduce motion blur already in source: increase ShutterAngleOut or FPSDivisor')
    if blurLevel > 200:
        raise vs.Error('QTGMC: exceeded maximum motion blur level: decrease ShutterAngleOut or FPSDivisor')

    # ShutterBlur mode 2,3 - get finer resolution motion vectors to reduce blur "bleeding" into static areas
    rBlockDivide = [1, 1, 2, 4][ShutterBlur]
    rBlockSize = max(BlockSize // rBlockDivide, 4)
    rOverlap = max(Overlap // rBlockDivide, 2)
    rBlockDivide = BlockSize // rBlockSize
    rLambda = Lambda // (rBlockDivide * rBlockDivide)
    if ShutterBlur > 1:
        recalculate_args = dict(
            thsad=ThSAD1,
            blksize=rBlockSize,
            overlap=rOverlap,
            search=Search,
            searchparam=SearchParam,
            truemotion=TrueMotion,
            lambda_=rLambda,
            pnew=PNew,
            dct=DCT,
            chroma=ChromaMotion,
        )
        sbBVec1 = core.mv.Recalculate(srchSuper, bVec1, **recalculate_args)
        sbFVec1 = core.mv.Recalculate(srchSuper, fVec1, **recalculate_args)
    elif ShutterBlur > 0:
        sbBVec1 = bVec1
        sbFVec1 = fVec1

    # Shutter motion blur - use MFlowBlur to blur along motion vectors
    if ShutterBlur > 0:
        sblurSuper = addNoise2.mv.Super(sharp=SubPelInterp, levels=1, **super_args)
        sblur = core.mv.FlowBlur(addNoise2, sblurSuper, sbBVec1, sbFVec1, blur=blurLevel, thscd1=ThSCD1, thscd2=ThSCD2)

    # Shutter motion blur - use motion mask to reduce blurring in areas of low motion - also helps reduce blur "bleeding" into static areas, then select blur type
    if ShutterBlur <= 0:
        sblurred = addNoise2
    elif SBlurLimit <= 0:
        sblurred = sblur
    else:
        sbMotionMask = core.mv.Mask(srchClip, bVec1, kind=0, ml=SBlurLimit)
        sblurred = core.std.MaskedMerge(addNoise2, sblur, sbMotionMask)

    # Reduce frame rate
    if FPSDivisor > 1:
        decimated = sblurred.std.SelectEvery(cycle=FPSDivisor, offsets=0)
    else:
        decimated = sblurred

    # Crop off temporary vertical padding
    if Border:
        cropped = decimated.std.Crop(top=4, bottom=4)
    else:
        cropped = decimated

    # Show output of choice + settings
    if ShowNoise <= 0:
        output = cropped
    else:
        expr = f'x {neutral} - {ShowNoise} * {neutral} +'
        output = EXPR(finalNoise, expr=expr if ChromaNoise or is_gray else [expr, repr(neutral)])
    output = output.std.SetFieldBased(value=0)
    if not ShowSettings:
        return output
    else:
        text = (
            f'{TR0=} | {TR1=} | {TR2=} | {Rep0=} | {Rep1=} | {Rep2=} | {RepChroma=} | {EdiMode=} | {NNSize=} | {NNeurons=} | {EdiQual=} | {EdiMaxD=} | '
            + f'{ChromaEdi=} | {Sharpness=} | {SMode=} | {SLMode=} | {SLRad=} | {SOvs=} | {SVThin=} | {Sbb=} | {SrchClipPP=} | {SubPel=} | {SubPelInterp=} | '
            + f'{BlockSize=} | {Overlap=} | {Search=} | {SearchParam=} | {PelSearch=} | {ChromaMotion=} | {TrueMotion=} | {Lambda=} | {LSAD=} | {PNew=} | '
            + f'{PLevel=} | {GlobalMotion=} | {DCT=} | {ThSAD1=} | {ThSAD2=} | {ThSCD1=} | {ThSCD2=} | {SourceMatch=} | {MatchPreset=} | {MatchEdi=} | '
            + f'{MatchPreset2=} | {MatchEdi2=} | {MatchTR2=} | {MatchEnhance=} | {Lossless=} | {NoiseProcess=} | {Denoiser=} | {FftThreads=} | {DenoiseMC=} | '
            + f'{NoiseTR=} | {Sigma=} | {ChromaNoise=} | {ShowNoise=} | {GrainRestore=} | {NoiseRestore=} | {NoiseDeint=} | {StabilizeNoise=} | {InputType=} | '
            + f'{ProgSADMask=} | {FPSDivisor=} | {ShutterBlur=} | {ShutterAngleSrc=} | {ShutterAngleOut=} | {SBlurLimit=} | {Border=} | {Precise=} | '
            + f'{Preset=} | {Tuning=} | {GlobalNames=} | {PrevGlobals=} | {ForceTR=} | {Str=} | {Amp=} | {FastMA=} | {ESearchP=} | {RefineMotion=}'
        )
        return output.text.Text(text=text)


def QTGMC_Interpolate(
    Input: vs.VideoNode,
    InputType: int,
    EdiMode: str,
    NNSize: int,
    NNeurons: int,
    EdiQual: int,
    EdiMaxD: int,
    Fallback: Optional[vs.VideoNode] = None,
    ChromaEdi: str = '',
    TFF: Optional[bool] = None,
    nnedi3_args: Mapping[str, Any] = {},
    eedi3_args: Mapping[str, Any] = {},
    opencl: bool = False,
    device: Optional[int] = None,
) -> vs.VideoNode:
    '''
    Interpolate input clip using method given in EdiMode. Use Fallback or Bob as result if mode not in list. If ChromaEdi string if set then interpolate chroma
    separately with that method (only really useful for EEDIx). The function is used as main algorithm starting point and for first two source-match stages
    '''
    is_gray = Input.format.color_family == vs.GRAY
    if is_gray:
        ChromaEdi = ''
    planes = [0, 1, 2] if ChromaEdi == '' and not is_gray else [0]

    field = 3 if TFF else 2

    
    if opencl:
        nnedi3 = partial(core.nnedi3cl.NNEDI3CL, field=field, device=device, **nnedi3_args)
        if hasattr(core, 'EEDI3CL'):
          eedi3 = partial(core.eedi3m.EEDI3CL, field=field, planes=planes, mdis=EdiMaxD, device=device, **eedi3_args)
        else:
          eedi3 = partial(core.eedi3m.EEDI3, field=field, planes=planes, mdis=EdiMaxD, **eedi3_args)
    else:
       if hasattr(core, 'znedi3'):
         nnedi3 = partial(core.znedi3.nnedi3, field=field, **nnedi3_args)
       else:
         nnedi3 = partial(core.nnedi3.nnedi3, field=field, **nnedi3_args)
  
       if hasattr(core, 'EEDI3CL'):
          eedi3 = partial(core.eedi3m.EEDI3CL, field=field, planes=planes, mdis=EdiMaxD, device=device, **eedi3_args)
       else:
          eedi3 = partial(core.eedi3m.EEDI3, field=field, planes=planes, mdis=EdiMaxD, **eedi3_args)

    if InputType == 1:
        return Input
    elif EdiMode == 'nnedi3':
        interp = nnedi3(Input, planes=planes, nsize=NNSize, nns=NNeurons, qual=EdiQual)
    elif EdiMode == 'eedi3+nnedi3':
        interp = eedi3(Input, sclip=nnedi3(Input, planes=planes, nsize=NNSize, nns=NNeurons, qual=EdiQual))
    elif EdiMode == 'eedi3':
        interp = eedi3(Input)
    elif EdiMode == 'bwdif':
        interp = Input.bwdif.Bwdif(field=field)
    else:
        interp = fallback(Fallback, Input.resize.Bob(tff=TFF, filter_param_a=0, filter_param_b=0.5))

    if ChromaEdi == 'nnedi3':
        interpuv = nnedi3(Input, planes=[1, 2], nsize=4, nns=0, qual=1)
    elif ChromaEdi == 'bwdif':
        interpuv = Input.bwdif.Bwdif(field=field)
    elif ChromaEdi == 'bob':
        interpuv = Input.resize.Bob(tff=TFF, filter_param_a=0, filter_param_b=0.5)
    else:
        return interp

    return core.std.ShufflePlanes([interp, interpuv], planes=[0, 1, 2], colorfamily=Input.format.color_family)


def QTGMC_KeepOnlyBobShimmerFixes(Input: vs.VideoNode, Ref: vs.VideoNode, Rep: int = 1, Chroma: bool = True) -> vs.VideoNode:
    '''
    Helper function: Compare processed clip with reference clip: only allow thin, horizontal areas of difference, i.e. bob shimmer fixes
    Rough algorithm: Get difference, deflate vertically by a couple of pixels or so, then inflate again. Thin regions will be removed
                     by this process. Restore remaining areas of difference back to as they were in reference clip
    '''
    is_gray = Input.format.color_family == vs.GRAY
    planes = [0, 1, 2] if Chroma and not is_gray else [0]

    bits = Input.format.bits_per_sample
    neutral = 1 << (bits - 1)

    # ed is the erosion distance - how much to deflate then reflate to remove thin areas of interest: 0 = minimum to 6 = maximum
    # od is over-dilation level  - extra inflation to ensure areas to restore back are fully caught:  0 = none to 3 = one full pixel
    # If Rep < 10, then ed = Rep and od = 0, otherwise ed = 10s digit and od = 1s digit (nasty method, but kept for compatibility with original TGMC)
    ed = Rep if Rep < 10 else Rep // 10
    od = 0 if Rep < 10 else Rep % 10

    diff = core.std.MakeDiff(Ref, Input)

    coordinates = [0, 1, 0, 0, 0, 0, 1, 0]
    has_zsmooth = hasattr(core, 'zsmooth')
    # Areas of positive difference
    choke1 = diff.std.Minimum(planes=planes, coordinates=coordinates)
    if ed > 2:
        choke1 = choke1.std.Minimum(planes=planes, coordinates=coordinates)
    if ed > 5:
        choke1 = choke1.std.Minimum(planes=planes, coordinates=coordinates)
    if ed % 3 != 0:
        choke1 = choke1.std.Deflate(planes=planes)
    if ed in [2, 5]:
        choke1 = choke1.zsmooth.Median(planes=planes) if has_zsmooth else choke1.std.Median(planes=planes)
    choke1 = choke1.std.Maximum(planes=planes, coordinates=coordinates)
    if ed > 1:
        choke1 = choke1.std.Maximum(planes=planes, coordinates=coordinates)
    if ed > 4:
        choke1 = choke1.std.Maximum(planes=planes, coordinates=coordinates)

    # Over-dilation - extra reflation up to about 1 pixel
    if od == 1:
        choke1 = choke1.std.Inflate(planes=planes)
    elif od == 2:
        choke1 = choke1.std.Inflate(planes=planes).std.Inflate(planes=planes)
    elif od >= 3:
        choke1 = choke1.std.Maximum(planes=planes)

    # Areas of negative difference (similar to above)
    choke2 = diff.std.Maximum(planes=planes, coordinates=coordinates)
    if ed > 2:
        choke2 = choke2.std.Maximum(planes=planes, coordinates=coordinates)
    if ed > 5:
        choke2 = choke2.std.Maximum(planes=planes, coordinates=coordinates)
    if ed % 3 != 0:
        choke2 = choke2.std.Inflate(planes=planes)
    if ed in [2, 5]:
        choke2 = choke2.zsmooth.Median(planes=planes) if has_zsmooth else choke2.std.Median(planes=planes)
    choke2 = choke2.std.Minimum(planes=planes, coordinates=coordinates)
    if ed > 1:
        choke2 = choke2.std.Minimum(planes=planes, coordinates=coordinates)
    if ed > 4:
        choke2 = choke2.std.Minimum(planes=planes, coordinates=coordinates)

    if od == 1:
        choke2 = choke2.std.Deflate(planes=planes)
    elif od == 2:
        choke2 = choke2.std.Deflate(planes=planes).std.Deflate(planes=planes)
    elif od >= 3:
        choke2 = choke2.std.Minimum(planes=planes)

    # Combine above areas to find those areas of difference to restore
    expr1 = f'x {scale_value(129, 8, bits)} < x y {neutral} < {neutral} y ? ?'
    expr2 = f'x {scale_value(127, 8, bits)} > x y {neutral} > {neutral} y ? ?'
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    restore = EXPR(
        [EXPR([diff, choke1], expr=expr1 if Chroma or is_gray else [expr1, '']), choke2], expr=expr2 if Chroma or is_gray else [expr2, '']
    )
    return core.std.MergeDiff(Input, restore, planes=planes)


def QTGMC_Generate2ndFieldNoise(Input: vs.VideoNode, InterleavedClip: vs.VideoNode, ChromaNoise: bool = False, TFF: Optional[bool] = None) -> vs.VideoNode:
    '''
    Given noise extracted from an interlaced source (i.e. the noise is interlaced), generate "progressive" noise with a new "field" of noise injected. The new
    noise is centered on a weighted local average and uses the difference between local min & max as an estimate of local variance
    '''
    is_gray = Input.format.color_family == vs.GRAY
    planes = [0, 1, 2] if ChromaNoise and not is_gray else [0]

    bits = Input.format.bits_per_sample
    neutral = 1 << (bits - 1)

    origNoise = Input.std.SeparateFields(tff=TFF)
    noiseMax = origNoise.std.Maximum(planes=planes).std.Maximum(planes=planes, coordinates=[0, 0, 0, 1, 1, 0, 0, 0])
    noiseMin = origNoise.std.Minimum(planes=planes).std.Minimum(planes=planes, coordinates=[0, 0, 0, 1, 1, 0, 0, 0])
    random = (
        InterleavedClip.std.SeparateFields(tff=TFF)
        .std.BlankClip(color=[neutral] * Input.format.num_planes)
        .grain.Add(var=1800, uvar=1800 if ChromaNoise else 0)
    )
    expr = f'x {neutral} - y * {scale_value(256, 8, bits)} / {neutral} +'
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    varRandom = EXPR([core.std.MakeDiff(noiseMax, noiseMin, planes=planes), random], expr=expr if ChromaNoise or is_gray else [expr, ''])
    newNoise = core.std.MergeDiff(noiseMin, varRandom, planes=planes)
    return Weave(core.std.Interleave([origNoise, newNoise]), tff=TFF)


def QTGMC_MakeLossless(Input: vs.VideoNode, Source: vs.VideoNode, InputType: int, TFF: Optional[bool] = None) -> vs.VideoNode:
    '''
    Insert the source lines into the result to create a true lossless output. However, the other lines in the result have had considerable processing and won't
    exactly match source lines. There will be some slight residual combing. Use vertical medians to clean a little of this away
    '''
    if InputType == 1:
        raise vs.Error('QTGMC: lossless modes are incompatible with InputType=1')

    neutral = 1 << (Input.format.bits_per_sample - 1)

    # Weave the source fields and the "new" fields that have generated in the input
    if InputType <= 0:
        srcFields = Source.std.SeparateFields(tff=TFF)
    else:
        srcFields = Source.std.SeparateFields(tff=TFF).std.SelectEvery(cycle=4, offsets=[0, 3])
    newFields = Input.std.SeparateFields(tff=TFF).std.SelectEvery(cycle=4, offsets=[1, 2])
    processed = Weave(core.std.Interleave([srcFields, newFields]).std.SelectEvery(cycle=4, offsets=[0, 1, 3, 2]), tff=TFF)

    # Clean some of the artefacts caused by the above - creating a second version of the "new" fields
    zsmooth = hasattr(core,'zsmooth')
    vertMedian = processed.zsmooth.VerticalCleaner(mode=1) if zsmooth else processed.rgvs.VerticalCleaner(mode=1)
    vertMedDiff = core.std.MakeDiff(processed, vertMedian)
    vmNewDiff1 = vertMedDiff.std.SeparateFields(tff=TFF).std.SelectEvery(cycle=4, offsets=[1, 2])
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    if zsmooth:
      vmNewDiff2 = EXPR(
        [vmNewDiff1.zsmooth.VerticalCleaner(mode=1), vmNewDiff1], expr=f'x {neutral} - y {neutral} - * 0 < {neutral} x {neutral} - abs y {neutral} - abs < x y ? ?'
      )
    else:
      vmNewDiff2 = EXPR(
        [vmNewDiff1.rgvs.VerticalCleaner(mode=1), vmNewDiff1], expr=f'x {neutral} - y {neutral} - * 0 < {neutral} x {neutral} - abs y {neutral} - abs < x y ? ?'
      )
    RG = core.zsmooth.RemoveGrain if zsmooth else core.rgvs.RemoveGrain
    vmNewDiff3 = core.zsmooth.Repair(vmNewDiff2, RG(vmNewDiff2, mode=2), mode=1) if zsmooth else core.rgvs.Repair(vmNewDiff2, RG(vmNewDiff2, mode=2), mode=1)

    # Reweave final result
    return Weave(core.std.Interleave([srcFields, core.std.MakeDiff(newFields, vmNewDiff3)]).std.SelectEvery(cycle=4, offsets=[0, 1, 3, 2]), tff=TFF)


def QTGMC_ApplySourceMatch(
    Deinterlace: vs.VideoNode,
    InputType: int,
    Source: vs.VideoNode,
    bVec1: Union[vs.VideoNode, None],
    fVec1: Union[vs.VideoNode, None],
    bVec2: Union[vs.VideoNode, None],
    fVec2: Union[vs.VideoNode, None],
    SubPel: int,
    SubPelInterp: int,
    hpad: int,
    vpad: int,
    ThSAD1: int,
    ThSCD1: int,
    ThSCD2: int,
    SourceMatch: int,
    MatchTR1: int,
    MatchEdi: str,
    MatchNNSize: int,
    MatchNNeurons: int,
    MatchEdiQual: int,
    MatchEdiMaxD: int,
    MatchTR2: int,
    MatchEdi2: str,
    MatchNNSize2: int,
    MatchNNeurons2: int,
    MatchEdiQual2: int,
    MatchEdiMaxD2: int,
    MatchEnhance: float,
    TFF: Optional[bool] = None,
    nnedi3_args: Mapping[str, Any] = {},
    eedi3_args: Mapping[str, Any] = {},
    opencl: bool = False,
    device: Optional[int] = None,
) -> vs.VideoNode:
    '''
    Source-match, a three stage process that takes the difference between deinterlaced input and the original interlaced source, to shift the input more towards
    the source without introducing shimmer. All other arguments defined in main script
    '''

    # Basic source-match. Find difference between source clip & equivalent fields in interpolated/smoothed clip (called the "error" in formula below). Ideally
    # there should be no difference, we want the fields in the output to be as close as possible to the source whilst remaining shimmer-free. So adjust the
    # *source* in such a way that smoothing it will give a result closer to the unadjusted source. Then rerun the interpolation (edi) and binomial smooth with
    # this new source. Result will still be shimmer-free and closer to the original source.
    # Formula used for correction is P0' = P0 + (P0-P1)/(k+S(1-k)), where P0 is original image, P1 is the 1st attempt at interpolation/smoothing , P0' is the
    # revised image to use as new source for interpolation/smoothing, k is the weighting given to the current frame in the smooth, and S is a factor indicating
    # "temporal similarity" of the error from frame to frame, i.e. S = average over all pixels of [neighbor frame error / current frame error] . Decreasing
    # S will make the result sharper, sensible range is about -0.25 to 1.0. Empirically, S=0.5 is effective [will do deeper analysis later]
    errorTemporalSimilarity = 0.5  # S in formula described above
    errorAdjust1 = [1.0, 2.0 / (1.0 + errorTemporalSimilarity), 8.0 / (3.0 + 5.0 * errorTemporalSimilarity)][MatchTR1]
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    if SourceMatch < 1 or InputType == 1:
        match1Clip = Deinterlace
    else:
        match1Clip = Weave(Deinterlace.std.SeparateFields(tff=TFF).std.SelectEvery(cycle=4, offsets=[0, 3]), tff=TFF)
    if SourceMatch < 1 or MatchTR1 <= 0:
        match1Update = Source
    else:
        match1Update = EXPR([Source, match1Clip], expr=f'x {errorAdjust1 + 1} * y {errorAdjust1} * -')
    if SourceMatch > 0:
        match1Edi = QTGMC_Interpolate(
            match1Update,
            InputType,
            MatchEdi,
            MatchNNSize,
            MatchNNeurons,
            MatchEdiQual,
            MatchEdiMaxD,
            TFF=TFF,
            nnedi3_args=nnedi3_args,
            eedi3_args=eedi3_args,
            opencl=opencl,
            device=device,
        )
        if MatchTR1 > 0:
            match1Super = match1Edi.mv.Super(pel=SubPel, sharp=SubPelInterp, levels=1, hpad=hpad, vpad=vpad)
            match1Degrain1 = core.mv.Degrain1(match1Edi, match1Super, bVec1, fVec1, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)
        if MatchTR1 > 1:
            match1Degrain2 = core.mv.Degrain1(match1Edi, match1Super, bVec2, fVec2, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)
    if SourceMatch < 1:
        match1 = Deinterlace
    elif MatchTR1 <= 0:
        match1 = match1Edi
    elif MatchTR1 == 1:
        match1 = core.std.Merge(match1Degrain1, match1Edi, weight=0.25)
    else:
        match1 = core.std.Merge(core.std.Merge(match1Degrain1, match1Degrain2, weight=0.2), match1Edi, weight=0.0625)

    if SourceMatch < 2:
        return match1

    # Enhance effect of source-match stages 2 & 3 by sharpening clip prior to refinement (source-match tends to underestimate so this will leave result sharper)
    if SourceMatch > 1 and MatchEnhance > 0:
        match1Shp = EXPR([match1, match1.std.Convolution(matrix=[1, 2, 1, 2, 4, 2, 1, 2, 1])], expr=f'x x y - {MatchEnhance} * +')
    else:
        match1Shp = match1

    # Source-match refinement. Find difference between source clip & equivalent fields in (updated) interpolated/smoothed clip. Interpolate & binomially smooth
    # this difference then add it back to output. Helps restore differences that the basic match missed. However, as this pass works on a difference rather than
    # the source image it can be prone to occasional artefacts (difference images are not ideal for interpolation). In fact a lower quality interpolation such
    # as a simple bob often performs nearly as well as advanced, slower methods (e.g. NNEDI3)
    if SourceMatch < 2 or InputType == 1:
        match2Clip = match1Shp
    else:
        match2Clip = Weave(match1Shp.std.SeparateFields(tff=TFF).std.SelectEvery(cycle=4, offsets=[0, 3]), tff=TFF)
    if SourceMatch > 1:
        match2Diff = core.std.MakeDiff(Source, match2Clip)
        match2Edi = QTGMC_Interpolate(
            match2Diff,
            InputType,
            MatchEdi2,
            MatchNNSize2,
            MatchNNeurons2,
            MatchEdiQual2,
            MatchEdiMaxD2,
            TFF=TFF,
            nnedi3_args=nnedi3_args,
            eedi3_args=eedi3_args,
            opencl=opencl,
            device=device,
        )
        if MatchTR2 > 0:
            match2Super = match2Edi.mv.Super(pel=SubPel, sharp=SubPelInterp, levels=1, hpad=hpad, vpad=vpad)
            match2Degrain1 = core.mv.Degrain1(match2Edi, match2Super, bVec1, fVec1, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)
        if MatchTR2 > 1:
            match2Degrain2 = core.mv.Degrain1(match2Edi, match2Super, bVec2, fVec2, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)
    if SourceMatch < 2:
        match2 = match1
    elif MatchTR2 <= 0:
        match2 = match2Edi
    elif MatchTR2 == 1:
        match2 = core.std.Merge(match2Degrain1, match2Edi, weight=0.25)
    else:
        match2 = core.std.Merge(core.std.Merge(match2Degrain1, match2Degrain2, weight=0.2), match2Edi, weight=0.0625)

    # Source-match second refinement - correct error introduced in the refined difference by temporal smoothing. Similar to error correction from basic step
    errorAdjust2 = [1.0, 2.0 / (1.0 + errorTemporalSimilarity), 8.0 / (3.0 + 5.0 * errorTemporalSimilarity)][MatchTR2]
    if SourceMatch < 3 or MatchTR2 <= 0:
        match3Update = match2Edi
    else:
        match3Update = EXPR([match2Edi, match2], expr=f'x {errorAdjust2 + 1} * y {errorAdjust2} * -')
    if SourceMatch > 2:
        if MatchTR2 > 0:
            match3Super = match3Update.mv.Super(pel=SubPel, sharp=SubPelInterp, levels=1, hpad=hpad, vpad=vpad)
            match3Degrain1 = core.mv.Degrain1(match3Update, match3Super, bVec1, fVec1, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)
        if MatchTR2 > 1:
            match3Degrain2 = core.mv.Degrain1(match3Update, match3Super, bVec2, fVec2, thsad=ThSAD1, thscd1=ThSCD1, thscd2=ThSCD2)
    if SourceMatch < 3:
        match3 = match2
    elif MatchTR2 <= 0:
        match3 = match3Update
    elif MatchTR2 == 1:
        match3 = core.std.Merge(match3Degrain1, match3Update, weight=0.25)
    else:
        match3 = core.std.Merge(core.std.Merge(match3Degrain1, match3Degrain2, weight=0.2), match3Update, weight=0.0625)

    # Apply difference calculated in source-match refinement
    return core.std.MergeDiff(match1Shp, match3)


def QTGMC_SetUserGlobal(Prefix: str, Name: str, Value: Union[vs.VideoNode, None]) -> None:
    '''Set global variable called "Prefix_Name" to "Value".'''
    global QTGMC_globals
    QTGMC_globals[f'{Prefix}_{Name}'] = Value


def QTGMC_GetUserGlobal(Prefix: str, Name: str) -> Union[vs.VideoNode, None]:
    '''Return value of global variable called "Prefix_Name". Returns None if it doesn't exist'''
    global QTGMC_globals
    return QTGMC_globals.get(f'{Prefix}_{Name}')

def Gauss(clip: vs.VideoNode, p: Optional[float] = None, sigma: Optional[float] = None, planes: Optional[Union[int, Sequence[int]]] = None) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('Gauss: this is not a clip')

    if p is None and sigma is None:
        raise vs.Error('Gauss: must have p or sigma')

    if p is not None and not 0.385 <= p <= 64.921:
        raise vs.Error('Gauss: p must be between 0.385 and 64.921 (inclusive)')

    if sigma is not None and not 0.334 <= sigma <= 4.333:
        raise vs.Error('Gauss: sigma must be between 0.334 and 4.333 (inclusive)')

    if sigma is None and p is not None:
        # Translate AviSynth parameter to standard parameter.
        sigma = math.sqrt(1.0 / (2.0 * (p / 10.0) * math.log(2)))

    # 6 * sigma + 1 rule-of-thumb.
    taps = int(math.ceil(sigma * 6 + 1))
    if not taps % 2:
        taps += 1

    # Gaussian kernel.
    kernel = []
    for x in range(int(math.floor(taps / 2))):
        kernel.append(1.0 / (math.sqrt(2.0 * math.pi) * sigma) * math.exp(-(x * x) / (2 * sigma * sigma)))

    # Renormalize to -1023...1023.
    for i in range(1, len(kernel)):
        kernel[i] *= 1023 / kernel[0]
    kernel[0] = 1023

    # Symmetry.
    kernel = kernel[::-1] + kernel[1:]

    return clip.std.Convolution(matrix=kernel, planes=planes, mode='hv')
    
def DitherLumaRebuild(src: vs.VideoNode, s0: float = 2.0, c: float = 0.0625, chroma: bool = True) -> vs.VideoNode:
    '''Converts luma (and chroma) to PC levels, and optionally allows tweaking for pumping up the darks. (for the clip to be fed to motion search only)'''
    if not isinstance(src, vs.VideoNode):
        raise vs.Error('DitherLumaRebuild: this is not a clip')

    if src.format.color_family == vs.RGB:
        raise vs.Error('DitherLumaRebuild: RGB format is not supported')

    is_gray = src.format.color_family == vs.GRAY
    is_integer = src.format.sample_type == vs.INTEGER

    bits = src.format.bits_per_sample
    neutral = 1 << (bits - 1)

    k = (s0 - 1) * c
    t = f'x {scale_value(16, 8, bits)} - {scale_value(219, 8, bits)} / 0 max 1 min' if is_integer else 'x 0 max 1 min'
    e = f'{k} {1 + c} {(1 + c) * c} {t} {c} + / - * {t} 1 {k} - * + ' + (f'{scale_value(256, 8, bits)} *' if is_integer else '')
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    return EXPR(src, expr=e if is_gray else [e, f'x {neutral} - 128 * 112 / {neutral} +' if chroma and is_integer else ''])

    
def Weave(clip: vs.VideoNode, tff: Optional[bool] = None) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('Weave: this is not a clip')

    if tff is None:
        with clip.get_frame(0) as f:
            if f.props.get('_Field') not in [1, 2]:
                raise vs.Error('Weave: tff was not specified and field order could not be determined from frame properties')

    return clip.std.DoubleWeave(tff=tff)[::2]
    
def mt_clamp(
    clip: vs.VideoNode,
    bright_limit: vs.VideoNode,
    dark_limit: vs.VideoNode,
    overshoot: int = 0,
    undershoot: int = 0,
    planes: Optional[Union[int, Sequence[int]]] = None,
) -> vs.VideoNode:
    if not (isinstance(clip, vs.VideoNode) and isinstance(bright_limit, vs.VideoNode) and isinstance(dark_limit, vs.VideoNode)):
        raise vs.Error('mt_clamp: this is not a clip')

    if bright_limit.format.id != clip.format.id or dark_limit.format.id != clip.format.id:
        raise vs.Error('mt_clamp: clips must have the same format')

    plane_range = range(clip.format.num_planes)

    if planes is None:
        planes = list(plane_range)
    elif isinstance(planes, int):
        planes = [planes]
    EXPR = core.akarin.Expr if hasattr(core,'akarin') else core.std.Expr
    return EXPR([clip, bright_limit, dark_limit], expr=[f'x y {overshoot} + min z {undershoot} - max' if i in planes else '' for i in plane_range])
    
    
# Taken from havsfunc
def KNLMeansCL(
    clip: vs.VideoNode,
    d: Optional[int] = None,
    a: Optional[int] = None,
    s: Optional[int] = None,
    h: Optional[float] = None,
    wmode: Optional[int] = None,
    wref: Optional[float] = None,
    device_type: Optional[str] = None,
    device_id: Optional[int] = None,
) -> vs.VideoNode:
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error('KNLMeansCL: this is not a clip')

    if clip.format.color_family != vs.YUV:
        raise vs.Error('KNLMeansCL: this wrapper is intended to be used only for YUV format')

    use_ispc = hasattr(core, 'nlm_ispc')
    use_cuda = hasattr(core, 'nlm_cuda')
    subsampled = clip.format.subsampling_w > 0 or clip.format.subsampling_h > 0
    if use_ispc:
        nlmeans = clip.nlm_ispc.NLMeans
        if subsampled:
          clip = nlmeans(d=d, a=a, s=s, h=h, channels='Y', wmode=wmode, wref=wref)
          return nlmeans(d=d, a=a, s=s, h=h, channels='UV', wmode=wmode, wref=wref)
        else:
          return nlmeans(d=d, a=a, s=s, h=h, channels='YUV', wmode=wmode, wref=wref)
    elif use_cuda:
        nlmeans = clip.nlm_cuda.NLMeans
        if subsampled:
          clip = nlmeans(d=d, a=a, s=s, h=h, channels='Y', wmode=wmode, wref=wref)
          return nlmeans(d=d, a=a, s=s, h=h, channels='UV', wmode=wmode, wref=wref)
        else:
          return nlmeans(d=d, a=a, s=s, h=h, channels='YUV', wmode=wmode, wref=wref)
    else:
      nlmeans = clip.knlm.KNLMeansCL
      if subsampled:
          clip = nlmeans(d=d, a=a, s=s, h=h, channels='Y', wmode=wmode, wref=wref, device_type=device_type, device_id=device_id)
          return nlmeans(d=d, a=a, s=s, h=h, channels='UV', wmode=wmode, wref=wref, device_type=device_type, device_id=device_id)
      else:
          return nlmeans(d=d, a=a, s=s, h=h, channels='YUV', wmode=wmode, wref=wref, device_type=device_type, device_id=device_id)
        