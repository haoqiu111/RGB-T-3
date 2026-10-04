# -*- coding: utf-8 -*-
"""Zero-shot wrappers of published fusion models (official code + official weights, loaded with weights_only).
  cddfuse  : CDDFuse (CVPR 2023)  third_party/MMIF-CDDFuse，models/CDDFuse_IVF.pth
  emma     : EMMA (CVPR 2024)     third_party/MMIF-EMMA，model/EMMA.pth
  textif   : Text-IF (CVPR 2024)  third_party/Text-IF, simple_fusion.pth + the official generic prompt (blind)
  textif_p : Text-IF, degradation-aware: text_fusion_power.pth + the official prompt matching the degradation kind (kind given = oracle prompt)
Interface: fuse(vg, ir)->[1,1,H,W]; Text-IF uses the RGB interface fuse(vt, ir, kind) and has rgb=True."""
import sys
from pathlib import Path

import torch

import _bootstrap  # noqa: F401,E402
from cdkit.paths import THIRD_PARTY, WEIGHTS  # noqa: E402

TP = Path(THIRD_PARTY)
GENERIC = "This is the infrared and visible light image fusion task."
PROMPTS = {
    "vis_underexposure": "In the context of infrared-visible image fusion, visible images are susceptible to extremely low light degradation.",
    "vis_overexposure": "We're tackling the infrared-visible image fusion challenge, dealing with visible images suffering from overexposure degradation.",
    "vis_gauss_noise": "The goal is to effectively fuse infrared and visible light images, mitigating the random noise present in visible images.",
    "vis_blur": "This task involves integrating infrared and visible light images, focusing on the degradation caused by blur in visible images.",
    "ir_low_contrast": "In this challenge, we're addressing the fusion of infrared and visible images, with a specific focus on the low contrast degradation in the infrared images.",
    "ir_stripe_nuc": "This pertains to the fusion of infrared and visible light images, with an emphasis on addressing stripe noise degradation in the infrared images.",
    "ir_gauss_noise": "We're working on the fusion of infrared and visible images, with special consideration for the noise degradation affecting the infrared captures.",
}
for _k in ("real_agc_histeq", "real_agc_plateau", "real_agc_gamma", "real_agc_minmax", "real_agc_stale"):
    PROMPTS[_k] = PROMPTS["ir_low_contrast"]  # the official prompt closest to a gain-control change


def _import_from(repo, module, names):
    """Isolated import: put the official checkout first on sys.path, import, then remove it and purge same-named modules (net/utils clash)."""
    p = str(TP / repo); sys.path.insert(0, p)
    for m in [k for k in list(sys.modules) if k == module.split(".")[0] or k.startswith(module.split(".")[0] + ".")]:
        del sys.modules[m]
    mod = __import__(module, fromlist=names); out = [getattr(mod, n) for n in names]
    sys.path.remove(p)
    for m in [k for k in list(sys.modules) if k == module.split(".")[0] or k.startswith(module.split(".")[0] + ".")]:
        del sys.modules[m]
    return out


def load_cddfuse(device):
    Enc, Dec, Base, Detail = _import_from("MMIF-CDDFuse", "net", ["Restormer_Encoder", "Restormer_Decoder", "BaseFeatureExtraction", "DetailFeatureExtraction"])
    enc, dec = torch.nn.DataParallel(Enc()).to(device), torch.nn.DataParallel(Dec()).to(device)
    base, det = torch.nn.DataParallel(Base(dim=64, num_heads=8)).to(device), torch.nn.DataParallel(Detail(num_layers=1)).to(device)
    ck = torch.load(TP / "MMIF-CDDFuse" / "models" / "CDDFuse_IVF.pth", map_location="cpu", weights_only=True)
    enc.load_state_dict(ck.get("CDDF_Encoder", ck.get("DIDF_Encoder"))); dec.load_state_dict(ck.get("CDDF_Decoder", ck.get("DIDF_Decoder")))
    base.load_state_dict(ck["BaseFuseLayer"]); det.load_state_dict(ck["DetailFuseLayer"])
    for m in (enc, dec, base, det):
        m.eval()

    @torch.no_grad()
    def fuse(vg, ir):
        vb, vd, _ = enc(vg); ib, idt, _ = enc(ir)
        out, _ = dec(vg, base(vb + ib), det(vd + idt))
        return ((out - out.min()) / (out.max() - out.min()).clamp_min(1e-8)).clamp(0, 1)   # min-max normalization as in the official test_IVF.py
    return fuse


def load_emma(device):
    (Ufuser,) = _import_from("MMIF-EMMA", "nets.Ufuser", ["Ufuser"])
    net = Ufuser().to(device); net.load_state_dict(torch.load(TP / "MMIF-EMMA" / "model" / "EMMA.pth", map_location="cpu", weights_only=True)); net.eval()

    @torch.no_grad()
    def fuse(vg, ir):
        out = net(ir, vg)                                       # official test.py: model(ir, vi); sizes must be divisible by 32 (512x640 is)
        return ((out - out.min()) / (out.max() - out.min()).clamp_min(1e-8)).clamp(0, 1)
    return fuse


AGC_PROMPT = ("In the context of infrared-visible image fusion, the infrared image has been remapped by the camera's automatic gain control, "
              "so its brightness and contrast are altered while the visible image is normal.")


def load_textif(device, power=False, agc_prompt=False):
    import clip
    (Text_IF,) = _import_from("Text-IF", "model.Text_IF_model", ["Text_IF"])
    model_clip, _ = clip.load("ViT-B/32", device=device)
    net = Text_IF(model_clip).to(device)
    w = "text_fusion_power.pth" if power else "simple_fusion.pth"
    net.load_state_dict(torch.load(TP / "Text-IF" / "pretrained_weights" / w, map_location="cpu", weights_only=True)["model"]); net.eval()
    cache = {}

    @torch.no_grad()
    def fuse(vt, ir, kind="clean"):
        text = PROMPTS.get(kind, GENERIC) if power else GENERIC
        if agc_prompt and kind.startswith("real_agc"):      # custom AGC prompt (the official prompt set has no gain-control class)
            text = AGC_PROMPT
        if text not in cache:
            cache[text] = clip.tokenize(text).to(device)
        out = net(vt, ir.repeat(1, 3, 1, 1), cache[text]).clamp(0, 1)   # official test: model(vi, ir, text), RGB output
        return (0.299 * out[:, 0:1] + 0.587 * out[:, 1:2] + 0.114 * out[:, 2:3]).clamp(0, 1)
    fuse.rgb = True
    return fuse


def load_controlfusion(device, oracle=False):
    """ControlFusion (NeurIPS 2025). The released default forward pass is "automatic control": the degradation prompt comes from an image-domain adapter
    (spatial + spectral CNN); the CLIP text branch is commented out in forward, so CLIP is not instantiated (model_clip.* keys are skipped, the rest load strictly).
      blind  controlfusion   : both.pth adapter on (ir, vi), no degradation prior;
      prior  controlfusion_p : one.pth adapter fed with the degraded stream (vis_* -> vi, ir_* / real_agc_* -> ir, clean -> vi as in the official test.py).
    The official test.py loads one.pth with strict=False; six fusion_module weights are missing from the file and stay randomly initialized, seeded with 0 here."""
    import argparse
    torch.serialization.add_safe_globals([argparse.Namespace])          # the checkpoint stores its training Namespace; allow only this type
    (CF,) = _import_from("ControlFusion", "model.ControlFusion", ["ControlFusion"])
    (Both,) = _import_from("ControlFusion", "model.Adapter_both", ["Adapter"]); (One,) = _import_from("ControlFusion", "model.Adapter_one", ["Adapter"])
    W = Path(WEIGHTS) / "fusion_weights" / "pretrained_weights"
    net = CF(torch.nn.Identity()).to(device)
    sd = {k: v for k, v in torch.load(W / "final.pth", map_location="cpu", weights_only=True)["model"].items() if not k.startswith("model_clip.")}
    net.load_state_dict(sd, strict=True); net.eval(); del sd
    torch.manual_seed(0)
    ada = (One() if oracle else Both()).to(device)
    ada.load_state_dict(torch.load(W / ("one.pth" if oracle else "both.pth"), map_location="cpu", weights_only=True), strict=False); ada.eval()

    @torch.no_grad()
    def fuse(vt, ir, kind="clean"):
        ir3 = ir.repeat(1, 3, 1, 1)
        if oracle:
            feat = ada(ir3 if (kind.startswith("ir_") or kind.startswith("real_agc")) else vt)
        else:
            feat = ada(ir3, vt)
        out = net(vt, ir3, None, feat).clamp(0, 1)                       # official test: model(vi, ir, text, img_feature), RGB output
        return (0.299 * out[:, 0:1] + 0.587 * out[:, 1:2] + 0.114 * out[:, 2:3]).clamp(0, 1)
    fuse.rgb = True
    return fuse


def load_drmf(device):
    """DRMF (ACM MM 2024), degradation-robust diffusion fusion, "challenging IVIF" setting of the official test_fusion.py: IVIF_degraded/IR_MSRS.pth + VI_LOL.pth
    restoration diffusion priors, Fusion_DPCM.pth weight-allocation network, DDIM 5 steps, eta=0, seed 61, per-channel min-max de-normalization. Networks and
    sampler are imported by file path (models/__init__ pulls in tensorboardX and training code). Blind: no degradation kind is given."""
    import argparse, importlib.util, yaml
    import numpy as np
    torch.serialization.add_safe_globals([argparse.Namespace])
    R = TP / "DRMF"

    def by_path(name, p):
        spec = importlib.util.spec_from_file_location(name, p); m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m); return m
    unet, samp = by_path("drmf_unet", R / "models" / "unet.py"), by_path("drmf_sampling", R / "utils" / "sampling.py")

    def ns(d):
        o = argparse.Namespace()
        for k, v in d.items():
            setattr(o, k, ns(v) if isinstance(v, dict) else v)
        return o
    cfg = ns(yaml.safe_load(open(R / "configs" / "Fusion.yml", encoding="utf-8"))); cfg.device = device
    W = Path(WEIGHTS) / "fusion_weights" / "experiment"
    strip = lambda sd: {k.replace("module.", ""): v for k, v in sd.items()}
    nets = []
    for cls, p in ((unet.DiffusionUNet, W / "IVIF_degraded" / "IR_MSRS.pth"), (unet.DiffusionUNet, W / "IVIF_degraded" / "VI_LOL.pth"), (unet.WeightUNet, W / "Fusion" / "Fusion_DPCM.pth")):
        n = cls(cfg).to(device); ck = torch.load(p, map_location="cpu", weights_only=True, mmap=True)
        n.load_state_dict(strip(ck["state_dict"]), strict=True); n.eval(); nets.append(n); del ck
    m_ir, m_vi, m_w = nets
    betas = torch.from_numpy(np.linspace(cfg.diffusion.beta_start, cfg.diffusion.beta_end, cfg.diffusion.num_diffusion_timesteps, dtype=np.float64)).float().to(device)
    seq = range(0, cfg.diffusion.num_diffusion_timesteps, cfg.diffusion.num_diffusion_timesteps // 5)
    gen = torch.Generator(device=device); gen.manual_seed(61)

    @torch.no_grad()
    def fuse(vt, ir, kind="clean"):
        a, b = 2 * ir.repeat(1, 3, 1, 1) - 1, 2 * vt - 1                     # official data_transform: A = infrared, B = visible; 512x640 is divisible by 16
        x = torch.randn(a.shape, device=device, generator=gen)
        with torch.autocast("cuda", dtype=torch.float16, enabled=fuse.amp):
            out = samp.generalized_steps_multi_weight(x, [a, b], seq, [m_ir, m_vi], model_weight=m_w, b=betas, eta=0.0)["xs"][-1].to(device).float()
        mx, mn = out.amax(dim=(2, 3), keepdim=True), out.amin(dim=(2, 3), keepdim=True)
        out = (out - mn) / (mx - mn).clamp_min(1e-8)                          # official inverse_data_transform: per-channel min-max
        return (0.299 * out[:, 0:1] + 0.587 * out[:, 1:2] + 0.114 * out[:, 2:3]).clamp(0, 1)
    fuse.rgb = True; fuse.amp = False
    return fuse


LOADERS = {"drmf": load_drmf, "controlfusion": lambda d: load_controlfusion(d, False), "controlfusion_p": lambda d: load_controlfusion(d, True),
"cddfuse": load_cddfuse, "emma": load_emma, "textif": lambda d: load_textif(d, False), "textif_p": lambda d: load_textif(d, True), "textif_agc": lambda d: load_textif(d, True, True)}
