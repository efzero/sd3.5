#!/usr/bin/env python
# sd3_large_controlnet_infer.py
#
# Pure inference script: loads SD3 + ControlNet + LoRA checkpoint, runs infer().
# Usage:
#   python sd3_large_controlnet_infer.py \
#       --model=models/sd3.5_large.safetensors \
#       --controlnet_ckpt=models/sd3.5_large_controlnet_blur.safetensors \
#       --lora_checkpoint=outputs/controlnet_grpo/controlnet_grpo_new_step2400_20260221_181544.pt \
#       --lq_dir=/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images_test_CelebA 
#       --prompt_dir=/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_prompts_test_CelebA
#       --out_dir=inference_test
#       --n_images=1


#python sd3_large_controlnet_infer.py --model=models/sd3.5_large.safetensors --controlnet_ckpt=models/sd3.5_large_controlnet_blur.safetensors --lora_checkpoint=outputs/controlnet_grpo/controlnet_grpo-run-l1lpips-seed6-ccs-2.5_step2200_20260224_232334.pt --lq_dir=/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images_test_CelebA --prompt_dir=/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_prompts_test_CelebA --out_dir=grpo_lpips_iter200_groupsize6_lr5e5_cutoff0 --n_images=10


# python sd3_large_controlnet_infer.py --model=models/sd3.5_large.safetensors --controlnet_ckpt=models/sd3.5_large_controlnet_blur.safetensors --lora_checkpoint=outputs/controlnet_grpo/controlnet_grpo-run-l1lpips-seed6-ccs-2.5_step2200_20260224_232334.pt --lq_dir=/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images_test_CelebA --prompt_dir=/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_prompts_test_CelebA --out_dir=grpo_lpips_iter200_groupsize6_lr5e5_cutoff0 --n_images=10

# ███████████████████████████████████████████████████████████████| 1/1 [00:58<00:00, 58.15s/it]
# (DiT) [bowenbw@lh1200 sd3.5]$ python sd3_large_controlnet_infer.py --model=models/sd3.5_large.safetensors --controlnet_ckpt=models/sd3.5_large_controlnet_blur.safetensors --lora_checkpoint=outputs/controlnet_grpo/controlnet_grpo-run-l1lpips-seed6-ccs-2.5_step2200_20260224_232334.pt --lq_dir=/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images_test_DIV2K --prompt_dir=/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_prompts_test_DIV2K/ --out_dir=grpo_lpips_iter200_groupsize6_lr5e5_cutoff0_div2k --n_images=10

# python sd3_large_controlnet_infer.py --model=models/sd3.5_large.safetensors --controlnet_ckpt=models/sd3.5_large_controlnet_blur.safetensors --lora_checkpoint=outputs/controlnet_grpo/controlnet_grpo-run-l1lpips-seed6-ccs-2.5_step2300_20260225_044553.pt --lq_dir=/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images_test_DIV2K --prompt_dir=/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_prompts_test_DIV2K/ --out_dir=grpo_lpips_iter300_groupsize6_lr5e5_cutoff0_div2k --n_images=10

# python sd3_large_controlnet_infer.py --model=models/sd3.5_large.safetensors --controlnet_ckpt=models/sd3.5_large_controlnet_blur.safetensors --lora_checkpoint=outputs/controlnet_grpo/controlnet_grpo-run-lpipsmusiq-seed6-ccs-2.5-cutoff750_step2100_20260225_103804.pt --lq_dir=/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images_test_DIV2K --prompt_dir=/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_prompts_test_DIV2K/ --out_dir=grpo_lpipsmusiq_iter100_groupsize6_lr5e5_cutoff0_true_div2k --n_images=10


# python sd3_large_controlnet_infer.py --model=models/sd3.5_large.safetensors --controlnet_ckpt=models/sd3.5_large_controlnet_blur.safetensors --lora_checkpoint=outputs/controlnet_lora/controlnet_lora_step2000_20260127_074346_large.pt --lq_dir=/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images_test_DIV2K --prompt_dir=/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_prompts_test_DIV2K/ --out_dir=grpo_lpipsmusiq_iter100_groupsize6_lr5e5_cutoff0_true_div2k --n_images=10


# python sd3_large_controlnet_infer.py --model=models/sd3.5_large.safetensors --controlnet_ckpt=models/sd3.5_large_controlnet_blur.safetensors --lora_checkpoint=outputs/controlnet_grpo-run-lpipsmusiq-seed6-ccs-2.5-cutoff750_step2500_20260226_015338.pt --lq_dir=/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images_test_DIV2K --prompt_dir=/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_prompts_test_DIV2K/ --out_dir=grpo_lpipsmusiq_iter500_groupsize6_lr5e5_cutoff0_true_div2k --n_images=10


# python sd3_large_controlnet_infer.py --model=models/sd3.5_large.safetensors --controlnet_ckpt=models/sd3.5_large_controlnet_blur.safetensors --lora_checkpoint=outputs/controlnet_grpo/controlnet_grpo-run-lpipsmusiq-seed6-ccs-2.5-cutoff750_step2500_20260226_015338.pt --lq_dir=/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images_test_CelebA --prompt_dir=/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_prompts_test_CelebA/ --out_dir=grpo_lpipsmusiq_iter500_groupsize6_lr5e5_cutoff0_true_celeba --n_images=10

import os
import math
from dataclasses import dataclass
from glob import glob
from typing import List, Optional
import fire
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from safetensors import safe_open

import sd3_impls
from other_impls import SD3Tokenizer, SDClipModel, SDXLClipG, T5XXLModel
from sd3_impls import SDVAE, BaseModel, SD3LatentFormat, invert_euler, CFGDenoiser, SkipLayerCFGDenoiser
from tqdm import tqdm
WIDTH = 1024
HEIGHT = 1024

DEFAULT_PROMPT = (
    "a high-resolution and sharp image, Cinematic, hyper sharpness, highly detailed, "
    "perfect without deformations, hyper detailed photo - realistic maximum detail"
)


################################################################################
# Model loading helpers
################################################################################

def load_into(ckpt, model, prefix, device, dtype=None, remap=None):
    for key in ckpt.keys():
        model_key = remap.get(key, key) if remap is not None else key
        if model_key.startswith(prefix) and not model_key.startswith("loss."):
            path = model_key[len(prefix):].split(".")
            obj = model
            for p in path:
                if obj is list:
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p, None)
                    if obj is None:
                        break
            if obj is None:
                continue
            tensor = ckpt.get_tensor(key).to(device=device)
            if dtype is not None and tensor.dtype != torch.int32:
                tensor = tensor.to(dtype=dtype)
            obj.requires_grad_(False)
            obj.set_(tensor)


CLIPG_CONFIG = {
    "hidden_act": "gelu",
    "hidden_size": 1280,
    "intermediate_size": 5120,
    "num_attention_heads": 20,
    "num_hidden_layers": 32,
}

class ClipG:
    def __init__(self, model_folder: str, device: str = "cpu"):
        with safe_open(f"{model_folder}/clip_g.safetensors", framework="pt", device="cpu") as f:
            self.model = SDXLClipG(CLIPG_CONFIG, device=device, dtype=torch.float32)
            load_into(f, self.model.transformer, "", device, torch.float32)


CLIPL_CONFIG = {
    "hidden_act": "quick_gelu",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
}

class ClipL:
    def __init__(self, model_folder: str):
        with safe_open(f"{model_folder}/clip_l.safetensors", framework="pt", device="cpu") as f:
            self.model = SDClipModel(
                layer="hidden",
                layer_idx=-2,
                device="cpu",
                dtype=torch.float32,
                layer_norm_hidden_state=False,
                return_projected_pooled=False,
                textmodel_json_config=CLIPL_CONFIG,
            )
            load_into(f, self.model.transformer, "", "cpu", torch.float32)


T5_CONFIG = {
    "d_ff": 10240,
    "d_model": 4096,
    "num_heads": 64,
    "num_layers": 24,
    "vocab_size": 32128,
}

class T5XXL:
    def __init__(self, model_folder: str, device: str = "cpu", dtype=torch.float32):
        with safe_open(f"{model_folder}/t5xxl.safetensors", framework="pt", device="cpu") as f:
            self.model = T5XXLModel(T5_CONFIG, device=device, dtype=dtype)
            load_into(f, self.model.transformer, "", device, dtype)


CONTROLNET_MAP = {
    "time_text_embed.timestep_embedder.linear_1.bias": "t_embedder.mlp.0.bias",
    "time_text_embed.timestep_embedder.linear_1.weight": "t_embedder.mlp.0.weight",
    "time_text_embed.timestep_embedder.linear_2.bias": "t_embedder.mlp.2.bias",
    "time_text_embed.timestep_embedder.linear_2.weight": "t_embedder.mlp.2.weight",
    "pos_embed.proj.bias": "x_embedder.proj.bias",
    "pos_embed.proj.weight": "x_embedder.proj.weight",
    "time_text_embed.text_embedder.linear_1.bias": "y_embedder.mlp.0.bias",
    "time_text_embed.text_embedder.linear_1.weight": "y_embedder.mlp.0.weight",
    "time_text_embed.text_embedder.linear_2.bias": "y_embedder.mlp.2.bias",
    "time_text_embed.text_embedder.linear_2.weight": "y_embedder.mlp.2.weight",
}

class SD3Bundle:
    def __init__(self, model_path, controlnet_ckpt, shift, verbose=False, device="cuda"):
        self.device = device
        self.using_8b_controlnet = False

        with safe_open(model_path, framework="pt", device="cpu") as f:
            control_model_ckpt = safe_open(controlnet_ckpt, framework="pt", device=device) if controlnet_ckpt else None
            self.model = BaseModel(
                shift=shift,
                file=f,
                prefix="model.diffusion_model.",
                device=device,
                dtype=torch.float16,
                control_model_ckpt=control_model_ckpt,
                verbose=verbose,
            ).eval()
            load_into(f, self.model, "model.", device, torch.float16)

        if controlnet_ckpt is not None:
            ck = safe_open(controlnet_ckpt, framework="pt", device=device)
            self.model.control_model = self.model.control_model.to(device)
            load_into(ck, self.model.control_model, "", device, torch.float16, remap=CONTROLNET_MAP)
            self.using_8b_controlnet = (self.model.control_model.y_embedder.mlp[0].in_features == 2048)
            self.model.control_model.using_8b_controlnet = self.using_8b_controlnet


class VAE:
    def __init__(self, model_path, dtype=torch.float16):
        with safe_open(model_path, framework="pt", device="cpu") as f:
            self.model = SDVAE(device="cpu", dtype=dtype).eval().cpu()
            prefix = ""
            if any(k.startswith("first_stage_model.") for k in f.keys()):
                prefix = "first_stage_model."
            load_into(f, self.model, prefix, "cpu", dtype)


################################################################################
# LoRA (must match training script to load checkpoints correctly)
################################################################################

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0, device="cuda"):
        super().__init__()
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.r if self.r > 0 else 0.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.empty(self.r, in_f, dtype=torch.float32, device=device))
        self.lora_B = nn.Parameter(torch.zeros(out_f, self.r, dtype=torch.float32, device=device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

    def forward(self, x):
        y = self.base(x)
        x_d = self.dropout(x).float()
        delta = (x_d @ self.lora_A.t()) @ self.lora_B.t()
        return y + delta.to(y.dtype) * self.scale


def inject_lora_into_linears(module: nn.Module, r: int, alpha: float, dropout: float = 0.0):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
        else:
            inject_lora_into_linears(child, r=r, alpha=alpha, dropout=dropout)


def load_lora_state_dict(module: nn.Module, sd: dict):
    for name, m in module.named_modules():
        if isinstance(m, LoRALinear):
            kA, kB = f"{name}.lora_A", f"{name}.lora_B"
            if kA in sd and kB in sd:
                m.lora_A.data.copy_(sd[kA].to(m.lora_A.device, dtype=m.lora_A.dtype))
                m.lora_B.data.copy_(sd[kB].to(m.lora_B.device, dtype=m.lora_B.dtype))


################################################################################
# Config
################################################################################

@dataclass
class InferConfig:
    # Required paths
    model: str = ""
    controlnet_ckpt: str = ""
    lora_checkpoint: str = ""          # .pt file saved by Trainer._save()

    model_folder: str = "models"       # folder containing clip_l/g + t5xxl safetensors
    vae: Optional[str] = None          # defaults to model path if None

    # LoRA hyper-params (must match training)
    lora_r: int = 128
    lora_alpha: float = 16.0

    shift: float = 3.0

    # Inference defaults
    steps: int = 60
    cfg_scale: float = 4.5
    sampler: str = "euler"
    seed: int = 23
    denoise: float = 1.0
    width: int = 1024
    height: int = 1024


################################################################################
# Inferencer
################################################################################

class Inferencer:
    def __init__(self, cfg: InferConfig):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # Text encoders
        self.tokenizer = SD3Tokenizer()
        self.t5xxl = T5XXL(cfg.model_folder, "cpu", torch.float32)
        self.clip_l = ClipL(cfg.model_folder)
        self.clip_g = ClipG(cfg.model_folder, "cpu")

        # Diffusion model + ControlNet + VAE
        self.sd3 = SD3Bundle(cfg.model, cfg.controlnet_ckpt, cfg.shift, verbose=False, device=self.device)
        self.vae = VAE(cfg.vae or cfg.model, dtype=torch.float16)
        self.latent_fmt = SD3LatentFormat()

        assert self.sd3.model.control_model is not None, "control_model is None — check controlnet_ckpt."

        self.control_type = int(self.sd3.model.control_model.control_type.item())
        self.using_2b = not self.sd3.using_8b_controlnet

        # Inject LoRA structure (must match training config)
        inject_lora_into_linears(self.sd3.model.control_model, cfg.lora_r, cfg.lora_alpha)

        # Load LoRA weights
        if cfg.lora_checkpoint:
            self._load_lora(cfg.lora_checkpoint)

        # Everything is frozen for inference
        for p in self.sd3.model.parameters():
            p.requires_grad_(False)
        self.sd3.model.eval()

    def _load_lora(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        key = "control_lora"
        if key not in ckpt:
            raise KeyError(f"'{key}' not found in checkpoint {path}. Keys: {list(ckpt.keys())}")
        load_lora_state_dict(self.sd3.model.control_model, ckpt[key])
        step = ckpt.get("step", "?")
        print(f"[load] LoRA from {path}  (step={step})")

    @torch.no_grad()
    def encode_prompt(self, prompt: str):
        tokens = self.tokenizer.tokenize_with_weights(prompt)
        l_out, l_pooled = self.clip_l.model.encode_token_weights(tokens["l"])
        g_out, g_pooled = self.clip_g.model.encode_token_weights(tokens["g"])
        t5_out, _ = self.t5xxl.model.encode_token_weights(tokens["t5xxl"])
        lg_out = torch.cat([l_out, g_out], dim=-1)
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
        c = torch.cat([lg_out, t5_out], dim=-2)
        y = torch.cat((l_pooled, g_pooled), dim=-1)
        return c, y   # CPU float32

    @torch.no_grad()
    def _get_empty_latent(self, width: int, height: int, seed: int):
        shape = (1, 16, height // 8, width // 8)
        g = torch.Generator(device=self.device).manual_seed(int(seed))
        return torch.randn(shape, generator=g, device=self.device, dtype=torch.float16)

    @torch.no_grad()
    def _get_noise(self, seed: int, latent: torch.Tensor):
        g = torch.manual_seed(int(seed))
        noise = torch.randn(latent.size(), dtype=torch.float32, generator=g, device="cpu")
        return noise.to(device=latent.device, dtype=latent.dtype)

    @torch.no_grad()
    def _get_sigmas(self, steps: int):
        sampling = self.sd3.model.model_sampling
        start = sampling.timestep(sampling.sigma_max)
        end = sampling.timestep(sampling.sigma_min)
        timesteps = torch.linspace(start, end, steps, device=self.device)
        sigs = [sampling.sigma(ts) for ts in timesteps]
        sigs += [torch.tensor(0.0, device=self.device)]
        return torch.stack(sigs).to(torch.float32)

    @torch.no_grad()
    def _max_denoise(self, sigmas: torch.Tensor):
        max_sigma = float(self.sd3.model.model_sampling.sigma_max)
        sigma0 = float(sigmas[0].item())
        return math.isclose(max_sigma, sigma0, rel_tol=1e-05) or sigma0 > max_sigma

    @torch.no_grad()
    def _fix_cond(self, cond_tuple):
        cond, pooled = cond_tuple
        return {"c_crossattn": cond.half().to(self.device), "y": pooled.half().to(self.device)}

    @torch.no_grad()
    def _image_to_latent(self, image_path: str, width: int, height: int,
                         using_2b: bool = False, control_type: int = 0) -> torch.Tensor:
        im = Image.open(image_path).convert("RGB").resize((width, height), Image.LANCZOS)
        arr = np.array(im).astype(np.float32) / 255.0
        t = torch.from_numpy(np.moveaxis(arr, 2, 0)).unsqueeze(0).to(self.device, dtype=torch.float32)

        if using_2b:
            t = t * 2.0 - 1.0
        elif control_type == 1:  # canny
            t = t * 255 * 0.5 + 0.5
        else:
            t = 2.0 * t - 1.0

        self.vae.model = self.vae.model.to(self.device)
        lat = self.vae.model.encode(t).cpu()
        self.vae.model = self.vae.model.cpu()
        return self.latent_fmt.process_in(lat).to(self.device, dtype=torch.float16)

    @torch.no_grad()
    def _vae_decode(self, latent_out: torch.Tensor) -> Image.Image:
        self.vae.model = self.vae.model.to(self.device)
        img = self.vae.model.decode(latent_out.to(self.device)).float()
        self.vae.model = self.vae.model.cpu()
        img = torch.clamp((img + 1.0) / 2.0, 0.0, 1.0)[0]
        decoded_np = (255.0 * np.moveaxis(img.cpu().numpy(), 0, 2)).astype(np.uint8)
        return Image.fromarray(decoded_np)

    @torch.no_grad()
    def _do_sampling(
        self,
        latent: torch.Tensor,
        seed: int,
        conditioning,
        neg_cond,
        steps: int,
        cfg_scale: float,
        sampler: str = "euler",
        controlnet_cond: Optional[torch.Tensor] = None,
        denoise: float = 1.0,
        init_noise = None,
    ) -> torch.Tensor:
        latent = latent.half().to(self.device)
        self.sd3.model = self.sd3.model.to(self.device)

        noise = self._get_noise(seed, latent)
        sigmas = self._get_sigmas(steps)
        sigmas = sigmas[int(steps * (1 - denoise)):]

        cond = self._fix_cond(conditioning)
        uncond = self._fix_cond(neg_cond)

        extra_args = {
            "cond": cond,
            "uncond": uncond,
            "cond_scale": cfg_scale,
            "controlnet_cond": controlnet_cond,
        }

        noise_scaled = self.sd3.model.model_sampling.noise_scaling(
            sigmas[0], noise, latent, self._max_denoise(sigmas)
        )
        
        if init_noise is not None:
            noise_scaled = init_noise

        sample_fn = getattr(sd3_impls, f"sample_{sampler}")
        latent = sample_fn(
            sd3_impls.CFGDenoiser(self.sd3.model),
            noise_scaled,
            sigmas,
            extra_args=extra_args,
        )
        return self.latent_fmt.process_out(latent)

    
    @torch.no_grad()
    def do_inversion(
        self,
        image,
        conditioning,
        neg_cond,
        steps,
        cfg_scale,
        controlnet_cond=None,
        width=WIDTH,
        height=HEIGHT,
    ) -> torch.Tensor:
        """
        Invert an image to the initial noise that would produce it under the Euler sampler.

        Uses `invert_kdiff_midpoint` (midpoint/RK2 ODE inverter) which walks the sigma
        schedule in reverse: sigma≈0 (clean) → sigma_max (noise).

        Args:
            image: PIL Image or file path (str) of the input image.
            conditioning: (c, y) tuple from get_cond(), same prompt as sampling.
            neg_cond: (c, y) tuple from get_cond("").
            steps: number of inversion steps (should match forward sampling steps).
            cfg_scale: CFG guidance scale (same as sampling for best round-trip).
            controlnet_cond: optional control latent tensor (already VAE-encoded + process_in).
            width, height: target spatial resolution.

        Returns:
            noise_latent: torch.Tensor — inverted latent at sigma_max, same shape as
                          the noisy latent fed into sample_euler during forward sampling.
        """
        # --- 1. Encode input image to clean internal latent (process_in format) ---
        if isinstance(image, str):
            # file path: _image_to_latent encodes + process_in
            latent = self._image_to_latent(image, width, height)
        else:
            # PIL Image
            raw_lat = self.vae_encode(image)                   # raw VAE latent, CPU
            latent = SD3LatentFormat().process_in(raw_lat)     # scale to internal format

        latent = latent.half().cuda()
        self.sd3.model = self.sd3.model.cuda()

        # --- 2. Build the same descending sigma schedule as do_sampling ---
        sigmas = self._get_sigmas(steps).cuda()
        # sigmas: [sigma_max, ..., sigma_min, 0]
        # invert_kdiff_midpoint will walk this backward (index n-1 → 0),
        # i.e., starting from the clean latent at sigmas[-1]≈0 → noise at sigmas[0]=sigma_max

        # --- 3. Build conditioning (same structure as do_sampling extra_args) ---
        conditioning = self._fix_cond(conditioning)
        neg_cond = self._fix_cond(neg_cond)
        extra_args = {
            "cond": conditioning,
            "uncond": neg_cond,
            "cond_scale": cfg_scale,
            "controlnet_cond": controlnet_cond,
        }

        # --- 4. Run midpoint ODE inversion ---
#         noise_latent = sd3_impls.invert_kdiff_midpoint(
#             CFGDenoiser(self.sd3.model),
#             latent,       # clean latent at sigma≈0
#             sigmas,       # descending schedule; inverter walks it in reverse
#             extra_args=extra_args,
#         )
        
        noise_latent = sd3_impls.invert_euler(
            CFGDenoiser(self.sd3.model),
            latent,       # clean latent at sigma≈0
            sigmas,       # descending schedule; inverter walks it in reverse
            extra_args=extra_args,
        )
        return noise_latent
    
    @torch.no_grad()
    def reconstruct(
        self,
        image,
        prompt,
        steps,
        cfg_scale,
        controlnet_cond_image=None,
        width=WIDTH,
        height=HEIGHT,
        sampler="euler",
        save_path="reconstructed.png",
    ) -> Image.Image:
        """
        Invert an image to noise, then re-sample from that noise to reconstruct it.

        A perfect inverter + sampler pair should produce an image nearly identical
        to the input. Useful for verifying inversion quality or as a starting point
        for editing.

        Args:
            image: file path (str) or PIL Image of the input.
            prompt: text prompt used for both inversion and re-sampling.
            steps: number of steps (must be the same for both passes).
            cfg_scale: CFG guidance scale.
            controlnet_cond_image: optional file path for the ControlNet condition image.
            width, height: spatial resolution.
            sampler: sampler name (default "euler", must match what invert_kdiff_midpoint inverts).
            save_path: where to save the reconstructed image.

        Returns:
            Reconstructed PIL Image.
        """
        # 1. Encode prompt + negative
        
        print("reconstructing")
        conditioning = self.get_cond(prompt)
        neg_cond = self.get_cond("")

        # 2. Prepare ControlNet latent (if any)
        controlnet_cond = None
        if controlnet_cond_image:
            using_2b, control_type = False, 0
            if self.sd3.model.control_model is not None:
                using_2b = not self.sd3.using_8b_controlnet
                control_type = int(self.sd3.model.control_model.control_type.item())
            controlnet_cond = self._image_to_latent(
                controlnet_cond_image, width, height, using_2b, control_type
            )

        # 3. Invert: image → noise at sigma_max
        noise_latent = self.do_inversion(
            image, conditioning, neg_cond, steps, 1.5,
            controlnet_cond=controlnet_cond, width=width, height=height,
        )
        
        print(noise_latent.mean(), noise_latent.max(), noise_latent.std(), "noise latent mean max std")
        # noise_latent is already at sigma_max (fully-noised starting point).
        # Do NOT apply noise_scaling again — that would double-noise it.

        # 4. Re-sample: noise → clean latent
        self.sd3.model = self.sd3.model.cuda()
        sigmas = self.get_sigmas(self.sd3.model.model_sampling, steps).cuda()

        extra_args = {
            "cond": self.fix_cond(conditioning),
            "uncond": self.fix_cond(neg_cond),
            "cond_scale": cfg_scale,
            "controlnet_cond": controlnet_cond,
        }

        sample_fn = getattr(sd3_impls, f"sample_{sampler}")
        recon_latent = sample_fn(
            CFGDenoiser(self.sd3.model),
            noise_latent,   # already scaled to sigma_max by inversion
            sigmas,
            extra_args=extra_args,
        )
        recon_latent = SD3LatentFormat().process_out(recon_latent)

        # 5. Decode and save
        out_image = self.vae_decode(recon_latent)
        out_dir = os.path.dirname(save_path)
        if out_dir:
            os.makedirs(out_dir, exist_ok=True)
        out_image.save(save_path)
        print(f"Saved → {save_path}")
        
        return out_image
    
    
    @torch.no_grad()
    def infer(
        self,
        prompts: List[str],
        controlnet_cond_image: Optional[str],
        init_image: Optional[str],
        out_dir: str,
        width: int = None,
        height: int = None,
        steps: int = None,
        cfg_scale: float = None,
        sampler: str = None,
        seed: int = None,
        denoise: float = None,
        save_path: str = None,
    ):
        cfg = self.cfg
        width     = width     or cfg.width
        height    = height    or cfg.height
        steps     = steps     or cfg.steps
        cfg_scale = cfg_scale or cfg.cfg_scale
        sampler   = sampler   or cfg.sampler
        seed      = seed      if seed is not None else cfg.seed
        denoise   = denoise   if denoise is not None else cfg.denoise

        os.makedirs(out_dir, exist_ok=True)

        # Base latent
        if init_image:
            latent = self._image_to_latent(init_image, width, height,
                                           using_2b=False, control_type=0)
        else:
            latent = self._get_empty_latent(width, height, seed)

        # Control latent
        control_lat = None
        if controlnet_cond_image:
            control_lat = self._image_to_latent(
                controlnet_cond_image, width, height,
                using_2b=self.using_2b, control_type=self.control_type,
            )

        neg_cond = self.encode_prompt("")
        import math

        for i, prompt in tqdm(list(enumerate(prompts)), total=len(prompts)):
            conditioning = self.encode_prompt(prompt)
#             init_noise = self.do_inversion(init_image, conditioning, neg_cond, 40, 1.0)
#             init_noise = (math.sin(1.0) * torch.randn_like(init_noise) + math.cos(1.0) * init_noise).to(init_noise.device).to(init_noise.dtype)
#             print(init_noise.mean(), init_noise.std())
            sampled_latent = self._do_sampling(
                latent=latent,
                seed=seed,
                conditioning=conditioning,
                neg_cond=neg_cond,
                steps=steps,
                cfg_scale=cfg_scale,
                sampler=sampler,
                controlnet_cond=control_lat,
                denoise=denoise if init_image else 1.0,
#                 init_noise=init_noise
            )
            img = self._vae_decode(sampled_latent)
            fname = save_path if save_path else f"{i:06d}.png"
            img.save(os.path.join(out_dir, fname))
            print(f"  saved → {os.path.join(out_dir, fname)}")


################################################################################
# Entry point
################################################################################

def main(
    model: str,
    controlnet_ckpt: str,
    lora_checkpoint: str,
    lq_dir: str,
    prompt_dir: str,
    out_dir: str = "infer_output",
    model_folder: str = "models",
    vae: str = None,
    lora_r: int = 128,
    lora_alpha: float = 16.0,
    shift: float = 3.0,
    steps: int = 60,
    cfg_scale: float = 4.5,
    sampler: str = "euler",
    seed: int = 23,
    denoise: float = 1.0,
    n_images: int = None,
):
    cfg = InferConfig(
        model=model,
        controlnet_ckpt=controlnet_ckpt,
        lora_checkpoint=lora_checkpoint,
        model_folder=model_folder,
        vae=vae,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        shift=shift,
        steps=steps,
        cfg_scale=cfg_scale,
        sampler=sampler,
        seed=seed,
        denoise=denoise,
    )
    inferencer = Inferencer(cfg)

    lq_paths = sorted(glob(f"{lq_dir}/*.png"))
    prompt_paths = sorted(glob(f"{prompt_dir}/*.txt"))
    
    supir_paths = sorted(glob("/scratch/liyues_root/liyues/shared_data/xuyuexy/SUPIR/results/lq_images_test_DIV2K/*.png"))

    if n_images is not None:
        lq_paths = lq_paths[:n_images]
        prompt_paths = prompt_paths[:n_images]

    assert len(lq_paths) == len(prompt_paths), (
        f"Mismatch: {len(lq_paths)} LQ images vs {len(prompt_paths)} prompts"
    )

    for i, (lq_path, prompt_path) in enumerate(zip(lq_paths, prompt_paths)):
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.readlines()[0].strip()

        supir_path = supir_paths[i]
        inferencer.infer(
#             prompts=[f"{prompt}, {DEFAULT_PROMPT}"],
            prompts = [DEFAULT_PROMPT],
            controlnet_cond_image=lq_path,
            init_image=supir_path,
            denoise=denoise,
            out_dir=out_dir,
            
            save_path=f"recon_{str(i).zfill(5)}.png",
        )


if __name__ == "__main__":
    fire.Fire(main)
