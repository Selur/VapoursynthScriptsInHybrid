# Original: https://github.com/IFeelBloated/Oyster/blob/master/Oyster.py
# modified by Selur, to support Akarin, znedi3, nnedi3cl, sneedif if present

import vapoursynth as vs
import math
import sys
import json

fmtc_args                      = dict(fulls=True, fulld=True)
msuper_args                    = dict(hpad=0, vpad=0, sharp=2, levels=0)
manalyze_args                  = dict(search=3, truemotion=False, trymany=True, levels=0, badrange=-24, divide=0, dct=0)
mrecalculate_args              = dict(truemotion=False, search=3, smooth=1, divide=0, dct=0)
mdegrain_args                  = dict(thscd1=16711680.0, thscd2=255.0)
nnedi_args                     = dict(field=1, dh=True, nns=4, qual=2, etype=1, nsize=0)
dfttest_args                   = dict(smode=0, sosize=0, tbsize=1, tosize=0, tmode=0)

class get_core:
      def __init__(self):
          self.core = vs.core

          # --- motion estimation ---
          self.MSuper       = self.core.mvsf.Super
          self.MAnalyze     = self.core.mvsf.Analyze
          self.MRecalculate = self.core.mvsf.Recalculate
          self.MDegrain     = self.core.mvsf.Degrain

          # --- BM3D selection ---
          if hasattr(self.core, 'bm3d'):
              bm3d_core = self.core.bm3d
              # Standard BM3D has Basic, Final, VBasic, VFinal, RGB2OPP, OPP2RGB
              self.BMBasic   = bm3d_core.VBasic
              self.BMFinal   = bm3d_core.VFinal
              self.Aggregate = bm3d_core.VAggregate
              self.RGB2OPP   = bm3d_core.RGB2OPP
              self.OPP2RGB   = bm3d_core.OPP2RGB
          else:
              raise RuntimeError("No BM3D plugin found.")

          # --- DFTTest selection ---
          self.DFTTest = self.core.dfttest.DFTTest

          # --- KNLMeansCL ---
          if hasattr(self.core, 'nlm_cuda'):
              self.KNLMeansCL = self.core.nlm_cuda.NLMeans
          else:
              self.KNLMeansCL = self.core.knlm.KNLMeansCL

          # --- NNEDI ---
          if hasattr(core, 'sneedif'):
              self.NNEDI = self.core.sneedif.NNEDI3
          if hasattr(self.core, 'nnedi3cl'):
              self.NNEDI = self.core.nnedi3cl.NNEDI3CL
          elif hasattr(self.core, 'znedi3'):
              self.NNEDI = self.core.znedi3.nnedi3
          else:
              self.NNEDI = self.core.nnedi3.nnedi3
              
          # --- SVPFlow plugins ---
          if hasattr(self.core, "svp1"):
              self.svp1 = self.core.svp1
          if hasattr(self.core, "svp2"):
              self.svp2 = self.core.svp2    

          # --- Common functions ---
          self.Resample        = self.core.fmtc.resample
          self.Expr            = (self.core.llvmexpr.Expr if hasattr(self.core, 'llvmexpr')
                                  else self.core.akarin.Expr if hasattr(self.core,'akarin')
                                  else self.core.std.Expr)
          self.MakeDiff        = self.core.std.MakeDiff
          self.MergeDiff       = self.core.std.MergeDiff
          self.Crop            = self.core.std.CropRel
          self.CropAbs         = self.core.std.CropAbs
          self.Transpose       = self.core.std.Transpose
          self.BlankClip       = self.core.std.BlankClip
          self.AddBorders      = self.core.std.AddBorders
          self.StackHorizontal = self.core.std.StackHorizontal
          self.StackVertical   = self.core.std.StackVertical
          self.MaskedMerge     = self.core.std.MaskedMerge
          self.ShufflePlanes   = self.core.std.ShufflePlanes
          self.SetFieldBased   = self.core.std.SetFieldBased

      def FreqMerge(self, low, hi, sbsize, slocation):
          hif                  = self.MakeDiff(hi, self.DFTTest(hi, sbsize=sbsize, slocation=slocation, **dfttest_args))
          clip                 = self.MergeDiff(self.DFTTest(low, sbsize=sbsize, slocation=slocation, **dfttest_args), hif)
          return clip

      def Pad(self, src, left, right, top, bottom):
          w                    = src.width
          h                    = src.height
          clip                 = self.Resample(src, w+left+right, h+top+bottom, -left, -top, w+left+right, h+top+bottom, kernel="point", **fmtc_args)
          return clip

      def NLMeans(self, src, d, a, s, h, rclip, color):
          def duplicate(src):
              if d > 0:
                 blank         = self.Expr(src[0], "0.0") * d
                 clip          = blank + src + blank
              else:
                 clip          = src
              return clip
          pad                  = self.AddBorders(src, a+s, a+s, a+s, a+s)
          pad                  = duplicate(pad)
          if rclip is not None:
             rclip             = self.AddBorders(rclip, a+s, a+s, a+s, a+s)
             rclip             = duplicate(rclip)
          nlm                  = self.KNLMeansCL(pad, d=d, a=a, s=s, h=h, channels="YUV" if color else "Y", wref=1.0, rclip=rclip)
          clip                 = self.Crop(nlm, a+s, a+s, a+s, a+s)
          return clip[d:clip.num_frames - d]

      def ThrMerge(self, flt, src, ref=None, thr=0.0009765625, elast=None):
          ref                  = src if ref is None else ref
          elast                = thr / 2 if elast is None else elast
          BExp                 = ["x {thr} {elast} + z - 2 {elast} * / * y {elast} z + {thr} - 2 {elast} * / * +".format(thr=thr, elast=elast)]
          BDif                 = self.Expr(src, "0.0")
          PDif                 = self.Expr([flt, src], "x y - 0.0 max")
          PRef                 = self.Expr([flt, ref], "x y - 0.0 max")
          PBLD                 = self.Expr([PDif, BDif, PRef], BExp)
          NDif                 = self.Expr([flt, src], "y x - 0.0 max")
          NRef                 = self.Expr([flt, ref], "y x - 0.0 max")
          NBLD                 = self.Expr([NDif, BDif, NRef], BExp)
          BLDD                 = self.MakeDiff(PBLD, NBLD)
          BLD                  = self.MergeDiff(src, BLDD)
          UDN                  = self.Expr([flt, ref, BLD], ["x y - abs {thr} {elast} - > z x ?".format(thr=thr, elast=elast)])
          clip                 = self.Expr([flt, ref, UDN, src], ["x y - abs {thr} {elast} + < z a ?".format(thr=thr, elast=elast)])
          return clip

      def GenBlockMask(self, src):
          clip                 = self.BlankClip(src, 24, 24, color=0.0)
          clip                 = self.AddBorders(clip, 4, 4, 4, 4, color=1.0)
          clip                 = self.StackHorizontal([clip, clip, clip, clip])
          clip                 = self.StackVertical([clip, clip, clip, clip])
          clip                 = self.Resample(clip, 32, 32, kernel="point", **fmtc_args)
          clip                 = self.Expr(clip, ["x 0.0 > 1.0 0.0 ?"])
          clip                 = self.StackHorizontal([clip, clip, clip, clip, clip, clip, clip, clip])
          clip                 = self.StackVertical([clip, clip, clip, clip, clip, clip])
          clip                 = self.StackHorizontal([clip, clip, clip, clip, clip, clip])
          clip                 = self.StackVertical([clip, clip, clip, clip, clip])
          clip                 = self.StackHorizontal([clip, clip, clip, clip, clip, clip])
          clip                 = self.StackVertical([clip, clip, clip, clip, clip])
          clip                 = self.CropAbs(clip, src.width, src.height, 0, 0)
          return clip

class internal:
      def super(core, src, pel):
          src                  = core.Pad(src, 128, 128, 128, 128)
          clip                 = core.Transpose(core.NNEDI(core.Transpose(core.NNEDI(src, **nnedi_args)), **nnedi_args))
          if pel == 4:
             clip              = core.Transpose(core.NNEDI(core.Transpose(core.NNEDI(clip, **nnedi_args)), **nnedi_args))
          return clip

      @staticmethod
      def svp_overlap_from_mv(blksize, mv_overlap):
          """
          Convert MVTools pixel overlap to SVPFlow allowed index.
          Allowed fractions: 0, 1/8, 1/4, 1/2 of block
          Returns: 0,1,2,3
          """
          allowed = [0, blksize // 8, blksize // 4, blksize // 2]
          diffs = [abs(mv_overlap - a) for a in allowed]
          return diffs.index(min(diffs))

      @staticmethod
      def svpflow_process(clip, new_num=None, new_den=1, preset='Medium', tuning='Film',
                          super_clip=None, radius=2, pel=2, sad=400, short_time=False, color=False):

          core = vs.core

          # --- Super clip config ---
          super_opts = {"pel": pel, "gpu": 1}
          super_opts_json = json.dumps(super_opts)

          # --- SAD mapping ---
          svp_sad = max(1, int(sad))
          refine_sad = max(1, int(sad // 8))

          # --- Convert clip to YUV420P8 ---
          svp_input = core.resize.Bicubic(clip, format=vs.YUV420P8)

          # --- Build SVPFlow vectors steps, safe overlaps ---
          if short_time:
              mv_steps = [(8, 4), (4, 2), (2, 1)]
          else:
              mv_steps = [(128, 64), (64, 32), (32, 16), (16, 8), (8, 4), (4, 2)]

          vectors_list = []
          for blksize, mv_overlap in mv_steps:
              vectors_list.append({
                  "block": {"w": blksize, "overlap": internal.svp_overlap_from_mv(blksize, mv_overlap)},
                  "main": {"search": {"distance": -10}, "bad": {"sad": svp_sad}},
                  "refine": [{"thsad": refine_sad}]
              })
          vectors_opts_json = json.dumps({"vectors": vectors_list})

          # --- SmoothFps config ---
          if new_num is not None:
              smooth_opts = {
                  "rate": {"num": new_num, "den": new_den, "abs": True},
                  "algo": 13,
                  "mask": {"cover": 80, "area": 0, "area_sharp": 1.2},
                  "scene": {"blend": True, "mode": 0}
              }
          else:
              smooth_opts = {
                  "rate": {"num": 2, "den": 1, "abs": False},
                  "algo": 13,
                  "mask": {"cover": 80, "area": 0, "area_sharp": 1.2},
                  "scene": {"blend": True, "mode": 0}
              }
          smooth_opts_json = json.dumps(smooth_opts)

          try:
              # --- Call SVPFlow ---
              super_clip_obj = core.svp1.Super(svp_input, super_opts_json)
              vectors_clip = core.svp1.Analyse(super_clip_obj['clip'], super_clip_obj['data'], svp_input, vectors_opts_json)
              out = core.svp2.SmoothFps(
                  svp_input,
                  super_clip_obj['clip'], super_clip_obj['data'],
                  vectors_clip['clip'], vectors_clip['data'],
                  smooth_opts_json
              )

              # Convert back to float YUV444PS
              out_float = core.resize.Bicubic(out, format=vs.YUV444PS)

              # Crop padding if necessary
              if super_clip is None:
                  out_float = core.std.CropRel(out_float, left=128, right=128, top=128, bottom=128)

              return out_float

          except Exception as e:
              # Fallback to MVTools
              # print(f"SVPFlow failed, falling back to MVTools: {e}")
              return None


      def basic(core, src, super_clip=None, radius=2, pel=2, sad=400, short_time=False, color=False, use_svpflow=True):
          plane = 4 if color else 0

          # Ensure src is single-precision float
          if src.format.sample_type != vs.FLOAT:
              src = core.fmtc.bitdepth(src, bits=32, planes=[0,1,2])

          src_padded = core.Pad(src, 128, 128, 128, 128)

          if use_svpflow:
              svp_out = internal.svpflow_process(
                  src_padded, new_num=None, new_den=1, preset='Medium', tuning='Film',
                  super_clip=super_clip, radius=radius, pel=pel, sad=sad,
                  short_time=short_time, color=color
              )
              if svp_out is not None:
                  return core.Crop(svp_out, 128, 128, 128, 128)

          # --- MVTools fallback ---
          if super_clip is None:
              supersoft = core.MSuper(src_padded, rfilter=4, pel=pel, chroma=color, **msuper_args)
              supersharp = core.MSuper(src_padded, rfilter=2, pel=pel, chroma=color, **msuper_args)
          else:
              supersoft = supersharp = super_clip

          # Motion estimation
          if short_time:
              constant = 0.0001989762736579584832432989326
              me_sad = [constant * math.pow(sad, 2.0) * math.log(1.0 + 1.0 / (constant * sad)), sad]
              vmulti = core.MAnalyze(supersoft, radius=radius, chroma=color, overlap=4, blksize=8, **manalyze_args)
              vmulti = core.MRecalculate(supersoft, vmulti, chroma=color, overlap=2, blksize=4, thsad=me_sad[0], **mrecalculate_args)
              vmulti = core.MRecalculate(supersoft, vmulti, chroma=color, overlap=1, blksize=2, thsad=me_sad[1], **mrecalculate_args)
          else:
              constant = 0.0000139144247313257680589719533
              me_sad = constant * math.pow(sad, 2.0) * math.log(1.0 + 1.0 / (constant * sad))
              vmulti = core.MAnalyze(supersoft, radius=radius, chroma=color, overlap=3, blksize=128, **manalyze_args)
              vmulti = core.MRecalculate(supersoft, vmulti, chroma=color, overlap=3, blksize=64, thsad=me_sad, **mrecalculate_args)
              vmulti = core.MRecalculate(supersoft, vmulti, chroma=color, overlap=3, blksize=32, thsad=me_sad, **mrecalculate_args)
              vmulti = core.MRecalculate(supersoft, vmulti, chroma=color, overlap=3, blksize=16, thsad=me_sad, **mrecalculate_args)
              vmulti = core.MRecalculate(supersoft, vmulti, chroma=color, overlap=2, blksize=8, thsad=me_sad, **mrecalculate_args)
              vmulti = core.MRecalculate(supersoft, vmulti, chroma=color, overlap=1, blksize=4, thsad=me_sad, **mrecalculate_args)

          clip = core.MDegrain(src_padded, supersharp, vmulti, thsad=sad, plane=plane, **mdegrain_args)
          return core.Crop(clip, 128, 128, 128, 128)

      def deringing(core, src, ref, radius, h, sigma, \
                    mse, hard_thr, block_size, block_step, group_size, bm_range, bm_step, ps_num, ps_range, ps_step, \
                    lowpass, color, matrix):
          c1                   = 0.1134141984932795312503328847998
          c2                   = 2.8623043756241389436528021745239
          strength             = [h]
          strength            += [h * math.pow(c1 * h, c2) * math.log(1.0 + 1.0 / math.pow(c1 * h, c2))]
          strength            += [None]
          def loop(flt, init, src, n):
              strength[2]      = n * strength[0] / 4 + strength[1] * (1 - n / 4)
              window           = int(32 / math.pow(2, n))
              flt              = init if n == 4 else flt
              dif              = core.MakeDiff(src, flt)
              dif              = core.NLMeans(dif, 0, window, 1, strength[2], flt, color)
              fnl              = core.MergeDiff(flt, dif)
              n               -= 1
              return fnl if n == -1 else loop(fnl, init, src, n)
          ref                  = core.FreqMerge(src, ref, block_size // 2 * 2 + 1, lowpass)
          dif                  = core.MakeDiff(src, ref)
          dif                  = core.BMBasic(dif, ref, radius=radius, th_mse=mse[0], hard_thr=hard_thr, sigma=sigma, \
                                              block_size=block_size, block_step=block_step, group_size=group_size, bm_range=bm_range, bm_step=bm_step, \
                                              ps_num=ps_num, ps_range=ps_range, ps_step=ps_step, matrix=matrix)
          dif                  = core.Aggregate(dif, radius, 1)
          ref                  = core.MergeDiff(ref, dif)
          refined              = loop(None, ref, src, 4)
          bm3d                 = core.BMFinal(refined, ref, radius=radius, th_mse=mse[1], sigma=sigma, \
                                              block_size=block_size, block_step=block_step, group_size=group_size, bm_range=bm_range, bm_step=bm_step, \
                                              ps_num=ps_num, ps_range=ps_range, ps_step=ps_step, matrix=matrix)
          bm3d                 = core.Aggregate(bm3d, radius, 1)
          bm3d                 = core.FreqMerge(refined, bm3d, block_size // 2 * 2 + 1, lowpass)
          clip                 = loop(None, bm3d, refined, 4)
          return clip

      def destaircase(core, src, ref, radius, sigma, \
                      mse, hard_thr, block_size, block_step, group_size, bm_range, bm_step, ps_num, ps_range, ps_step, \
                      thr, elast, lowpass, matrix):
          mask                 = core.GenBlockMask(core.ShufflePlanes(src, 0, vs.GRAY))
          ref                  = core.FreqMerge(src, ref, block_size // 2 * 2 + 1, lowpass)
          ref                  = core.ThrMerge(src, ref, thr=thr, elast=elast)
          dif                  = core.MakeDiff(src, ref)
          dif                  = core.BMBasic(dif, ref, radius=radius, th_mse=mse[0], hard_thr=hard_thr, sigma=sigma, \
                                              block_size=block_size, block_step=block_step, group_size=group_size, bm_range=bm_range, bm_step=bm_step, \
                                              ps_num=ps_num, ps_range=ps_range, ps_step=ps_step, matrix=matrix)
          dif                  = core.Aggregate(dif, radius, 1)
          ref                  = core.MergeDiff(ref, dif)
          dif                  = core.MakeDiff(src, ref)
          dif                  = core.BMFinal(dif, ref, radius=radius, th_mse=mse[1], sigma=sigma, \
                                              block_size=block_size, block_step=block_step, group_size=group_size, bm_range=bm_range, bm_step=bm_step, \
                                              ps_num=ps_num, ps_range=ps_range, ps_step=ps_step, matrix=matrix)
          dif                  = core.Aggregate(dif, radius, 1)
          ref                  = core.MergeDiff(ref, dif)
          clip                 = core.MaskedMerge(src, ref, mask, first_plane=True)
          return clip

      def deblocking(core, src, ref, radius, h, sigma, \
                     mse, hard_thr, block_size, block_step, group_size, bm_range, bm_step, ps_num, ps_range, ps_step, \
                     lowpass, color, matrix):
          mask                 = core.GenBlockMask(core.ShufflePlanes(src, 0, vs.GRAY))
          cleansed             = core.NLMeans(ref, radius, block_size, math.ceil(block_size / 2), h, ref, color)
          dif                  = core.MakeDiff(ref, cleansed)
          dif                  = core.BMBasic(dif, cleansed, radius=radius, th_mse=mse[0], hard_thr=hard_thr, sigma=sigma, \
                                              block_size=block_size, block_step=block_step, group_size=group_size, bm_range=bm_range, bm_step=bm_step, \
                                              ps_num=ps_num, ps_range=ps_range, ps_step=ps_step, matrix=matrix)
          dif                  = core.Aggregate(dif, radius, 1)
          cleansed             = core.MergeDiff(cleansed, dif)
          dif                  = core.MakeDiff(ref, cleansed)
          dif                  = core.BMFinal(dif, cleansed, radius=radius, th_mse=mse[1], sigma=sigma, \
                                              block_size=block_size, block_step=block_step, group_size=group_size, bm_range=bm_range, bm_step=bm_step, \
                                              ps_num=ps_num, ps_range=ps_range, ps_step=ps_step, matrix=matrix)
          dif                  = core.Aggregate(dif, radius, 1)
          cleansed             = core.MergeDiff(cleansed, dif)
          ref                  = core.FreqMerge(cleansed, ref, block_size // 2 * 2 + 1, lowpass)
          src                  = core.FreqMerge(cleansed, src, block_size // 2 * 2 + 1, lowpass)
          clip                 = core.MaskedMerge(src, ref, mask, first_plane=True)
          return clip

def Super(src, pel=4):
    if not isinstance(src, vs.VideoNode):
       raise TypeError("Oyster.Super: src has to be a video clip!")
    elif src.format.sample_type != vs.FLOAT or src.format.bits_per_sample < 32:
       raise TypeError("Oyster.Super: the sample type of src has to be single precision!")
    elif src.format.subsampling_w > 0 or src.format.subsampling_h > 0:
       raise RuntimeError("Oyster.Super: subsampled stuff not supported!")
    if not isinstance(pel, int):
       raise TypeError("Oyster.Super: pel has to be an integer!")
    elif pel != 2 and pel != 4:
       raise RuntimeError("Oyster.Super: pel has to be 2 or 4!")
    core                       = get_core()
    src                        = core.SetFieldBased(src, 0)
    colorspace                 = src.format.color_family
    if colorspace == vs.RGB:
       src                     = core.RGB2OPP(src, 1)
    clip                       = internal.super(core, src, pel)
    del core
    return clip

def Basic(src, super=None, radius=6, pel=4, sad=2000.0, short_time=False):
    if not isinstance(src, vs.VideoNode):
       raise TypeError("Oyster.Basic: src has to be a video clip!")
    elif src.format.sample_type != vs.FLOAT or src.format.bits_per_sample < 32:
       raise TypeError("Oyster.Basic: the sample type of src has to be single precision!")
    elif src.format.subsampling_w > 0 or src.format.subsampling_h > 0:
       raise RuntimeError("Oyster.Basic: subsampled stuff not supported!")
    if not isinstance(super, vs.VideoNode) and super is not None:
       raise TypeError("Oyster.Basic: super has to be a video clip or None!")
    elif super is not None:
       if super.format.sample_type != vs.FLOAT or super.format.bits_per_sample < 32 or super.format.subsampling_w > 0 or super.format.subsampling_h > 0:
          raise RuntimeError("Oyster.Basic: corrupted super clip!")
    if not isinstance(radius, int):
       raise TypeError("Oyster.Basic: radius has to be an integer!")
    elif radius < 1:
       raise RuntimeError("Oyster.Basic: radius has to be greater than 0!")
    if not isinstance(pel, int):
       raise TypeError("Oyster.Basic: pel has to be an integer!")
    elif pel != 1 and pel != 2 and pel != 4:
       raise RuntimeError("Oyster.Basic: pel has to be 1, 2 or 4!")
    if not isinstance(sad, float) and not isinstance(sad, int):
       raise TypeError("Oyster.Basic: sad has to be a real number!")
    elif sad <= 0.0:
       raise RuntimeError("Oyster.Basic: sad has to be greater than 0!")
    if not isinstance(short_time, bool):
       raise TypeError("Oyster.Basic: short_time has to be boolean!")
    core                       = get_core()
    color                      = True
    rgb                        = False
    colorspace                 = src.format.color_family
    if colorspace == vs.RGB:
       src                     = core.RGB2OPP(src, 1)
       rgb                     = True
    if colorspace == vs.GRAY:
       color                   = False
    src                        = core.SetFieldBased(src, 0)
    super                      = core.SetFieldBased(super, 0) if super is not None else None
    clip                       = internal.basic(core, src, super, radius, pel, sad, short_time, color)
    clip                       = core.OPP2RGB(clip, 1) if rgb else clip
    del core
    return clip

def Deringing(src, ref, radius=6, h=6.4, sigma=16.0, \
              mse=[None, None], hard_thr=3.2, block_size=8, block_step=1, group_size=32, bm_range=24, bm_step=1, ps_num=2, ps_range=8, ps_step=1, \
              lowpass=None):
    if not isinstance(src, vs.VideoNode):
       raise TypeError("Oyster.Deringing: src has to be a video clip!")
    elif src.format.sample_type != vs.FLOAT or src.format.bits_per_sample < 32:
       raise TypeError("Oyster.Deringing: the sample type of src has to be single precision!")
    elif src.format.subsampling_w > 0 or src.format.subsampling_h > 0:
       raise RuntimeError("Oyster.Deringing: subsampled stuff not supported!")
    if not isinstance(ref, vs.VideoNode):
       raise TypeError("Oyster.Deringing: ref has to be a video clip!")
    elif ref.format.sample_type != vs.FLOAT or ref.format.bits_per_sample < 32:
       raise TypeError("Oyster.Deringing: the sample type of ref has to be single precision!")
    elif ref.format.subsampling_w > 0 or ref.format.subsampling_h > 0:
       raise RuntimeError("Oyster.Deringing: subsampled stuff not supported!")
    if not isinstance(radius, int):
       raise TypeError("Oyster.Deringing: radius has to be an integer!")
    elif radius < 1:
       raise RuntimeError("Oyster.Deringing: radius has to be greater than 0!")
    if not isinstance(h, float) and not isinstance(h, int):
       raise TypeError("Oyster.Deringing: h has to be a real number!")
    elif h <= 0:
       raise RuntimeError("Oyster.Deringing: h has to be greater than 0!")
    if not isinstance(mse, list):
       raise TypeError("Oyster.Deringing: mse parameter has to be an array!")
    elif len(mse) != 2:
       raise RuntimeError("Oyster.Deringing: mse parameter has to contain 2 elements exactly!")
    for i in range(2):
        if not isinstance(mse[i], float) and not isinstance(mse[i], int) and mse[i] is not None:
           raise TypeError("Oyster.Deringing: elements in mse must be real numbers or None!")
    if not isinstance(lowpass, list) and lowpass is not None:
       raise TypeError("Oyster.Deringing: lowpass has to be a list or None!")
    core                       = get_core()
    rgb                        = False
    color                      = True
    mse[0]                     = sigma * 160.0 + 1200.0 if mse[0] is None else mse[0]
    mse[1]                     = sigma * 120.0 + 800.0 if mse[1] is None else mse[1]
    lowpass                    = [0.0,sigma, 0.48,1024.0, 1.0,1024.0] if lowpass is None else lowpass
    matrix                     = None
    colorspace                 = src.format.color_family
    if colorspace == vs.RGB:
       rgb                     = True
       matrix                  = 100
       src                     = core.RGB2OPP(src, 1)
       ref                     = core.RGB2OPP(ref, 1)
    if colorspace == vs.GRAY:
       color                   = False
    src                        = core.SetFieldBased(src, 0)
    ref                        = core.SetFieldBased(ref, 0)
    clip                       = internal.deringing(core, src, ref, radius, h, sigma, \
                                                    mse, hard_thr, block_size, block_step, group_size, bm_range, bm_step, ps_num, ps_range, ps_step, \
                                                    lowpass, color, matrix)
    clip                       = core.OPP2RGB(clip, 1) if rgb else clip
    del core
    return clip

def Destaircase(src, ref, radius=6, sigma=16.0, \
                mse=[None, None], hard_thr=3.2, block_size=8, block_step=1, group_size=32, bm_range=24, bm_step=1, ps_num=2, ps_range=8, ps_step=1, \
                thr=0.03125, elast=0.015625, lowpass=None):
    if not isinstance(src, vs.VideoNode):
       raise TypeError("Oyster.Destaircase: src has to be a video clip!")
    elif src.format.sample_type != vs.FLOAT or src.format.bits_per_sample < 32:
       raise TypeError("Oyster.Destaircase: the sample type of src has to be single precision!")
    elif src.format.subsampling_w > 0 or src.format.subsampling_h > 0:
       raise RuntimeError("Oyster.Destaircase: subsampled stuff not supported!")
    if not isinstance(ref, vs.VideoNode):
       raise TypeError("Oyster.Destaircase: ref has to be a video clip!")
    elif ref.format.sample_type != vs.FLOAT or ref.format.bits_per_sample < 32:
       raise TypeError("Oyster.Destaircase: the sample type of ref has to be single precision!")
    elif ref.format.subsampling_w > 0 or ref.format.subsampling_h > 0:
       raise RuntimeError("Oyster.Destaircase: subsampled stuff not supported!")
    if not isinstance(radius, int):
       raise TypeError("Oyster.Destaircase: radius has to be an integer!")
    elif radius < 1:
       raise RuntimeError("Oyster.Destaircase: radius has to be greater than 0!")
    if not isinstance(mse, list):
       raise TypeError("Oyster.Destaircase: mse parameter has to be an array!")
    elif len(mse) != 2:
       raise RuntimeError("Oyster.Destaircase: mse parameter has to contain 2 elements exactly!")
    for i in range(2):
        if not isinstance(mse[i], float) and not isinstance(mse[i], int) and mse[i] is not None:
           raise TypeError("Oyster.Destaircase: elements in mse must be real numbers or None!")
    if not isinstance(thr, float) and not isinstance(thr, int):
       raise TypeError("Oyster.Destaircase: thr has to be a real number!")
    elif thr < 0 or thr > 1:
       raise RuntimeError("Oyster.Destaircase: thr has to fall in [0, 1]!")
    if not isinstance(elast, float) and not isinstance(elast, int):
       raise TypeError("Oyster.Destaircase: elast has to be a real number!")
    elif elast < 0 or elast > thr:
       raise RuntimeError("Oyster.Destaircase: elast has to fall in [0, thr]!")
    if not isinstance(lowpass, list) and lowpass is not None:
       raise TypeError("Oyster.Destaircase: lowpass has to be a list or None!")
    core                       = get_core()
    rgb                        = False
    mse[0]                     = sigma * 160.0 + 1200.0 if mse[0] is None else mse[0]
    mse[1]                     = sigma * 120.0 + 800.0 if mse[1] is None else mse[1]
    lowpass                    = [0.0,sigma, 0.48,1024.0, 1.0,1024.0] if lowpass is None else lowpass
    matrix                     = None
    colorspace                 = src.format.color_family
    if colorspace == vs.RGB:
       rgb                     = True
       matrix                  = 100
       src                     = core.RGB2OPP(src, 1)
       ref                     = core.RGB2OPP(ref, 1)
    src                        = core.SetFieldBased(src, 0)
    ref                        = core.SetFieldBased(ref, 0)
    clip                       = internal.destaircase(core, src, ref, radius, sigma, \
                                                      mse, hard_thr, block_size, block_step, group_size, bm_range, bm_step, ps_num, ps_range, ps_step, \
                                                      thr, elast, lowpass, matrix)
    clip                       = core.OPP2RGB(clip, 1) if rgb else clip
    del core
    return clip

def Deblocking(src, ref, radius=6, h=6.4, sigma=16.0, \
               mse=[None, None], hard_thr=3.2, block_size=8, block_step=1, group_size=32, bm_range=24, bm_step=1, ps_num=2, ps_range=8, ps_step=1, \
               lowpass=[0.0,0.0, 0.12,1024.0, 1.0,1024.0]):
    if not isinstance(src, vs.VideoNode):
       raise TypeError("Oyster.Deblocking: src has to be a video clip!")
    elif src.format.sample_type != vs.FLOAT or src.format.bits_per_sample < 32:
       raise TypeError("Oyster.Deblocking: the sample type of src has to be single precision!")
    elif src.format.subsampling_w > 0 or src.format.subsampling_h > 0:
       raise RuntimeError("Oyster.Deblocking: subsampled stuff not supported!")
    if not isinstance(ref, vs.VideoNode):
       raise TypeError("Oyster.Deblocking: ref has to be a video clip!")
    elif ref.format.sample_type != vs.FLOAT or ref.format.bits_per_sample < 32:
       raise TypeError("Oyster.Deblocking: the sample type of ref has to be single precision!")
    elif ref.format.subsampling_w > 0 or ref.format.subsampling_h > 0:
       raise RuntimeError("Oyster.Deblocking: subsampled stuff not supported!")
    if not isinstance(radius, int):
       raise TypeError("Oyster.Deblocking: radius has to be an integer!")
    elif radius < 1:
       raise RuntimeError("Oyster.Deblocking: radius has to be greater than 0!")
    if not isinstance(h, float) and not isinstance(h, int):
       raise TypeError("Oyster.Deblocking: h has to be a real number!")
    elif h <= 0:
       raise RuntimeError("Oyster.Deblocking: h has to be greater than 0!")
    if not isinstance(mse, list):
       raise TypeError("Oyster.Deblocking: mse parameter has to be an array!")
    elif len(mse) != 2:
       raise RuntimeError("Oyster.Deblocking: mse parameter has to contain 2 elements exactly!")
    for i in range(2):
        if not isinstance(mse[i], float) and not isinstance(mse[i], int) and mse[i] is not None:
           raise TypeError("Oyster.Deblocking: elements in mse must be real numbers or None!")
    if not isinstance(lowpass, list):
       raise TypeError("Oyster.Deblocking: lowpass has to be a list!")
    core                       = get_core()
    rgb                        = False
    color                      = True
    mse[0]                     = sigma * 160.0 + 1200.0 if mse[0] is None else mse[0]
    mse[1]                     = sigma * 120.0 + 800.0 if mse[1] is None else mse[1]
    matrix                     = None
    colorspace                 = src.format.color_family
    if colorspace == vs.RGB:
       rgb                     = True
       matrix                  = 100
       src                     = core.RGB2OPP(src, 1)
       ref                     = core.RGB2OPP(ref, 1)
    if colorspace == vs.GRAY:
       color                   = False
    src                        = core.SetFieldBased(src, 0)
    ref                        = core.SetFieldBased(ref, 0)
    clip                       = internal.deblocking(core, src, ref, radius, h, sigma, \
                                                     mse, hard_thr, block_size, block_step, group_size, bm_range, bm_step, ps_num, ps_range, ps_step, \
                                                     lowpass, color, matrix)
    clip                       = core.OPP2RGB(clip, 1) if rgb else clip
    del core
    return clip
