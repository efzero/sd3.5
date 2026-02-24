#!/usr/bin/env python
# sd3_large_controlnet_grpo.py
#
# GRPO (Group Relative Policy Optimization) training for SD3 ControlNet LoRA.
# Based on sd3_large_controlnet_train_lora.py (SFT version).
# GRPO algorithm reference: flow_grpo/scripts/train_sd3.py
#
# Key differences from SFT:
#   - Samples K=num_seeds trajectories per image with different random seeds (reverse denoising)
#   - Stores full trajectory (latents + log probs) for each of the K seeds
#   - Computes reward via reward_model(generated_image, gt_image) for each generated image
#   - Normalizes rewards to advantages within the K-group per image
#   - Trains with PPO-clipped policy gradient loss on stored trajectory log probs

import os
import math
import time
import random
import datetime
import contextlib
from dataclasses import dataclass
from glob import glob
from typing import List, Optional, Tuple
import fire
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from safetensors import safe_open

import sd3_impls
from other_impls import SD3Tokenizer, SDClipModel, SDXLClipG, T5XXLModel
from sd3_impls import SDVAE, BaseModel, SD3LatentFormat

from tqdm import tqdm
import wandb

DEFAULT_PROMPT = (
    "a high-resolution and sharp image, Cinematic, hyper sharpness, highly detailed, "
    "perfect without deformations, hyper detailed photo - realistic maximum detail"
)


import numpy as np
import torch
import lpips
from PIL import Image

# Create once (global / trainer init), not inside the reward call.
_lpips_model = None

def init_lpips(device="cuda", net="alex"):
    global _lpips_model
    _lpips_model = lpips.LPIPS(net=net).to(device).eval()
    
    
################################################################################
# Datasets  (unchanged from sd3_large_controlnet_train_lora.py)
################################################################################

class ImageDataset(Dataset):
    def __init__(self, image_root: str, lq_image_root: str):
        self.image_root = image_root
        self.lq_image_root = lq_image_root
        img_files = sorted(glob(f"{image_root}/*.png"))
        lq_img_files = sorted(glob(f"{lq_image_root}/*.png"))
        self.img_items = img_files
        self.lqimg_items = lq_img_files

    def __len__(self):
        return len(self.lqimg_items)

    def __getitem__(self, idx):
        path = self.img_items[idx]
        lq_path = self.lqimg_items[idx]
        image = Image.open(path).convert("RGB").resize((1024, 1024), Image.LANCZOS)
        image_np = np.moveaxis(np.array(image).astype(np.float32) / 255.0, 2, 0)
        tensor = torch.from_numpy(image_np) * 2.0 - 1.0
        return tensor, lq_path, path ####the ground truth image, the path to the lq image, the path to the gt imagexx


class ImageCaptionDatasetLarge(Dataset):
    def __init__(self, image_root: str, captions_root: str, lq_image_root: str, imgperprompt=1):
        self.image_root = image_root
        self.lq_image_root = lq_image_root
        self.imgperprompt = imgperprompt
        self.img_items = sorted(glob(f"{image_root}/*.png"))
        self.lqimg_items = sorted(glob(f"{lq_image_root}/*.png"))
        self.items = sorted(glob(f"{captions_root}/*.txt"))

    def __len__(self):
        return len(self.lqimg_items)

    def __getitem__(self, idx):
        path = self.img_items[idx]
        lq_path = self.lqimg_items[idx]
        caption_path = self.items[idx // self.imgperprompt]
        image = Image.open(path).convert("RGB").resize((1024, 1024), Image.LANCZOS)
        with open(caption_path, "r", encoding="utf-8") as f:
            caption = f.readline().strip()
        image_np = np.moveaxis(np.array(image).astype(np.float32) / 255.0, 2, 0)
        tensor = torch.from_numpy(image_np) * 2.0 - 1.0
        return tensor, lq_path, caption


################################################################################
# Model loading helpers  (unchanged)
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
                layer="hidden", layer_idx=-2, device="cpu", dtype=torch.float32,
                layer_norm_hidden_state=False, return_projected_pooled=False,
                textmodel_json_config=CLIPL_CONFIG,
            )
            load_into(f, self.model.transformer, "", "cpu", torch.float32)


T5_CONFIG = {
    "d_ff": 10240, "d_model": 4096, "num_heads": 64,
    "num_layers": 24, "vocab_size": 32128,
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
            control_model_ckpt = (
                safe_open(controlnet_ckpt, framework="pt", device=device)
                if controlnet_ckpt else None
            )
            self.model = BaseModel(
                shift=shift, file=f, prefix="model.diffusion_model.",
                device=device, dtype=torch.bfloat16,
                control_model_ckpt=control_model_ckpt, verbose=verbose,
            ).train()
            load_into(f, self.model, "model.", device, torch.bfloat16)
        if controlnet_ckpt is not None:
            ck = safe_open(controlnet_ckpt, framework="pt", device=device)
            self.model.control_model = self.model.control_model.to(device)
            load_into(ck, self.model.control_model, "", device, torch.bfloat16, remap=CONTROLNET_MAP)
            self.using_8b_controlnet = (
                self.model.control_model.y_embedder.mlp[0].in_features == 2048
            )
            self.model.control_model.using_8b_controlnet = self.using_8b_controlnet


class VAE:
    def __init__(self, model_path, dtype=torch.float16):
        with safe_open(model_path, framework="pt", device="cpu") as f:
            self.model = SDVAE(device="cpu", dtype=dtype).eval().cpu()
            prefix = "first_stage_model." if any(
                k.startswith("first_stage_model.") for k in f.keys()
            ) else ""
            load_into(f, self.model, prefix, "cpu", dtype)


################################################################################
# LoRA  (unchanged)
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


def inject_lora_into_linears(module: nn.Module, r: int, alpha: float, dropout: float):
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            setattr(module, name, LoRALinear(child, r=r, alpha=alpha, dropout=dropout))
        else:
            inject_lora_into_linears(child, r=r, alpha=alpha, dropout=dropout)


def mark_only_lora_trainable(module: nn.Module):
    for p in module.parameters():
        p.requires_grad_(False)
    for m in module.modules():
        if isinstance(m, LoRALinear):
            m.lora_A.requires_grad_(True)
            m.lora_B.requires_grad_(True)


def lora_state_dict(module: nn.Module):
    sd = {}
    for name, m in module.named_modules():
        if isinstance(m, LoRALinear):
            sd[f"{name}.lora_A"] = m.lora_A.detach().cpu()
            sd[f"{name}.lora_B"] = m.lora_B.detach().cpu()
            sd[f"{name}.alpha"] = torch.tensor(m.alpha)
            sd[f"{name}.r"] = torch.tensor(m.r)
    return sd


def load_lora_state_dict(module: nn.Module, sd: dict):
    for name, m in module.named_modules():
        if isinstance(m, LoRALinear):
            kA, kB = f"{name}.lora_A", f"{name}.lora_B"
            if kA in sd and kB in sd:
                m.lora_A.data.copy_(sd[kA].to(m.lora_A.device, dtype=m.lora_A.dtype))
                m.lora_B.data.copy_(sd[kB].to(m.lora_B.device, dtype=m.lora_B.dtype))


################################################################################
# Reward model  (customize with your actual reward function)
################################################################################

def reward_model(generated_image: Image.Image, gt_image: Image.Image) -> float:
    """
    Compute a scalar reward comparing generated image to ground truth.
    Higher reward = better output. Replace this placeholder with your actual
    reward model (e.g. LPIPS, NIQE, HyperIQA, or a learned discriminator).

    Current placeholder: negative MSE in pixel space.
    """
    gen_np = np.array(generated_image).astype(np.float32) / 255.0
    gt_np = np.array(gt_image).astype(np.float32) / 255.0
    return -float(np.mean((gen_np - gt_np) ** 2) * 100)


###############################################################################
# Reward model LPIPS
################################################################################

@torch.no_grad()
def reward_model_lpips_batch(gen_images, gt_images, device="cuda"):
    """
    gen_images: list[PIL] length N
    gt_images:  list[PIL] length N (aligned)
    returns: torch.FloatTensor shape (N,) rewards = -LPIPS
    """
    global _lpips_model
    if _lpips_model is None:
        init_lpips(device=device, net="alex")

    def stack_pil(imgs):
        arr = np.stack([np.array(im.convert("RGB")).astype(np.float32) / 255.0 for im in imgs], axis=0)  # (N,H,W,3)
        t = torch.from_numpy(arr).permute(0, 3, 1, 2)  # (N,3,H,W)
        t = t * 2.0 - 1.0
        return t

    gen = stack_pil(gen_images).to(device)
    gt  = stack_pil(gt_images).to(device)

    d = _lpips_model(gen, gt)          # (N,1,1,1) or (N,1)
    d = d.view(d.shape[0], -1).mean(1) # (N,)
    return -d.detach().cpu()


###############################################################################
# Reward model LPIPS + L1
################################################################################
@torch.no_grad()
def reward_model_lpips_l1_batch(
    gen_images,
    gt_images,
    device="cuda",
    lpips_weight: float = 1.0,
    l1_weight: float = 0.0,
):
    """
    returns: rewards (N,) on CPU, higher is better
    reward = -(lpips_weight * LPIPS + l1_weight * L1)

    Notes:
      - LPIPS expects inputs in [-1, 1]
      - L1 here is mean absolute error in [0, 1] space (more interpretable)
    """
    global _lpips_model
    if _lpips_model is None:
        init_lpips(device=device, net="vgg")

    def stack_pil_01(imgs):
        arr = np.stack(
            [np.array(im.convert("RGB")).astype(np.float32) / 255.0 for im in imgs],
            axis=0,
        )  # (N,H,W,3) in [0,1]
        t = torch.from_numpy(arr).permute(0, 3, 1, 2)  # (N,3,H,W)
        return t

    gen_01 = stack_pil_01(gen_images).to(device)
    gt_01  = stack_pil_01(gt_images).to(device)

    # LPIPS in [-1,1]
    gen_m11 = gen_01 * 2.0 - 1.0
    gt_m11  = gt_01 * 2.0 - 1.0

    lp = _lpips_model(gen_m11, gt_m11)
    lp = lp.view(lp.shape[0], -1).mean(1)  # (N,)

    l1 = (gen_01 - gt_01).abs().mean(dim=(1, 2, 3)) * 5    # (N,)
    print(l1, "l1", lp, "lpips")

    reward = -(lpips_weight * lp + l1_weight * l1)
    return reward.detach().cpu()

################################################################################
# GRPO core: SDE step with log probability computation
################################################################################





import math
from typing import Optional, Tuple
import torch



import math
from typing import Optional, Tuple
import torch

# def euler_sde_step_with_logprob(
#     pred_x0: torch.Tensor,              # (B, C, H, W) model predicts x0
#     x_t: torch.Tensor,                  # (B, C, H, W) current latent x_t
#     sigma_curr: torch.Tensor,           # (B,) current scheduler "sigma" (time-like t in (0,1))
#     sigma_next: torch.Tensor,           # (B,) next scheduler "sigma" (usually smaller in reverse sampling)
#     noise_level: float = 0.7,           # scalar a in std_dev_t = a * sqrt(t/(1-t))
#     x_next_given: Optional[torch.Tensor] = None,  # evaluate log_prob at this point
#     temperature: float = 1.0,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Euler-Maruyama SDE step + Gaussian log-prob, matching the provided scheduler update,
#     but for a model that predicts x0 instead of velocity.

#     Uses:
#         v = (x_t - pred_x0) / t
#         std_dev_t = noise_level * sqrt(t / (1 - t))
#         dt = t_next - t_curr   (typically negative in reverse denoising)

#         mean = x_t * (1 + std_dev_t^2/(2t) * dt)
#              + v   * (1 + std_dev_t^2*(1-t)/(2t)) * dt

#         x_next = mean + std_dev_t * sqrt(-dt) * eps

#     Returns:
#         x_next:   (B, C, H, W)
#         log_prob: (B,)   # per-element mean formulation (same style as your code)
#         mean:     (B, C, H, W)
#         std:      (B, 1, 1, 1)  # actual transition std (includes sqrt(-dt))
#     """
#     pred_x0 = pred_x0.float()
#     x_t = x_t.float()
#     if x_next_given is not None:
#         x_next_given = x_next_given.float()

#     B = x_t.shape[0]
#     ndims = x_t.ndim
#     device = x_t.device
#     dtype = x_t.dtype

#     # Broadcast shape (B,1,1,1,...)
#     t = sigma_curr.view(B, *([1] * (ndims - 1))).to(device=device, dtype=dtype)
#     t_next = sigma_next.view(B, *([1] * (ndims - 1))).to(device=device, dtype=dtype)

#     # Reverse denoising usually => dt < 0
#     dt = t_next - t

#     # Numerical safety near 0/1
#     t_safe = t.clamp(min=1e-6, max=1.0 - 1e-6)

#     # Convert x0 prediction to velocity prediction
#     # x_t = t * eps + (1 - t) * x0  => v = eps - x0 = (x_t - x0) / t
#     v = (x_t - pred_x0) / t_safe

#     # sigma_t from the paper (named std_dev_t in your scheduler code)
#     std_dev_t = noise_level * torch.sqrt(t_safe / (1.0 - t_safe))  # (B,1,1,1,...)

#     # Mean update (same structure as your provided code, but model_output -> v)
#     mean = (
#         x_t * (1.0 + (std_dev_t ** 2) / (2.0 * t_safe) * dt)
#         + v   * (1.0 + (std_dev_t ** 2) * (1.0 - t_safe) / (2.0 * t_safe)) * dt
#     )

#     # Transition std for reverse step: std_dev_t * sqrt(-dt)
#     neg_dt = (-dt).clamp(min=1e-12)
#     trans_std = std_dev_t * torch.sqrt(neg_dt)

#     # Sample / evaluate
#     if x_next_given is not None:
#         x_next = x_next_given
#     elif noise_level > 0:
#         x_next = mean + trans_std * torch.randn_like(x_t)
#     else:
#         x_next = mean.clone()

#     # Gaussian log-prob (mean-reduced version to match your existing style)
#     trans_std_safe = trans_std.clamp(min=1e-12)
#     reduce_dims = tuple(range(1, ndims))
#     sq_err = ((x_next - mean).pow(2) / trans_std_safe.pow(2)).mean(dim=reduce_dims)  # (B,)

#     # each sample has scalar std broadcasted over all non-batch dims
#     log_std_b = trans_std_safe.view(B, -1)[:, 0].log()  # (B,)

#     log_prob = -0.5 * sq_err - log_std_b - 0.5 * math.log(2.0 * math.pi)

#     if temperature != 1.0:
#         log_prob = log_prob / temperature

#     return x_next, log_prob, mean, trans_std

# def euler_sde_step_with_logprob(
#     pred_x0: torch.Tensor,           # (B, C, H, W) model x0 prediction
#     x_t: torch.Tensor,               # (B, C, H, W) current latent at time t
#     sigma_curr: torch.Tensor,        # (B,) interpreted as current time t_curr in (0,1)
#     sigma_next: torch.Tensor,        # (B,) interpreted as next time t_next in (0,1)
#     noise_level: float = 0.7,        # scalar a in sigma_t = a * sqrt(t/(1-t))
#     x_next_given: Optional[torch.Tensor] = None,
#     temperature: float = 5000.0,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Euler-Maruyama step based on Eq. (9):

#         x_{t+dt} = x_t + [ v_theta(x_t,t)
#                            + (sigma_t^2 / (2t)) * (x_t + (1-t) v_theta(x_t,t))
#                          ] * dt
#                    + sigma_t * sqrt(dt) * eps

#     where sigma_t = a * sqrt(t / (1 - t)).

#     IMPORTANT:
#     - Here sigma_curr / sigma_next are treated as *time* t, not SD3 noise sigma.
#     - If you are doing reverse-time denoising (t_next < t_curr), we use:
#         dt = t_next - t_curr   (signed, for drift)
#         std = sigma_t * sqrt(|dt|)  (for Gaussian variance)
#       This preserves a valid Gaussian transition for logging likelihood.

#     Returns:
#         x_next:   (B, C, H, W)
#         log_prob: (B,)
#         mean:     (B, C, H, W)
#         std:      (B, 1, 1, 1)
#     """
#     B, C, H, W = x_t.shape
#     device = x_t.device
#     dtype = x_t.dtype

#     t_curr = sigma_curr.reshape(B, 1, 1, 1).to(device=device, dtype=dtype)
#     t_next = sigma_next.reshape(B, 1, 1, 1).to(device=device, dtype=dtype)

#     # signed step (can be negative in reverse-time sampling)
#     dt = t_next - t_curr

#     # Numerical safety for divisions / sqrt near boundaries
#     t_safe = t_curr.clamp(min=1e-6, max=1.0 - 1e-6)

#     # v_theta(x_t, t) from x0-prediction under x_t = t * eps + (1 - t) * x0
#     # => v = eps - x0 = (x_t - x0) / t
#     v = (x_t - pred_x0) / t_safe

#     # sigma_t = a * sqrt(t / (1 - t))
#     sigma_t = noise_level * torch.sqrt(t_safe / (1.0 - t_safe))  # (B,1,1,1)

#     # Drift term in Eq. (9)
#     drift = v + (sigma_t.pow(2) / (2.0 * t_safe)) * (x_t + (1.0 - t_safe) * v)

#     # EM mean update (use signed dt)
#     mean = x_t + drift * dt

#     # Gaussian noise std for transition (use |dt| so variance is nonnegative)
#     std = sigma_t * torch.sqrt(dt.abs().clamp(min=1e-12))  # (B,1,1,1)

#     # Sample or evaluate log-prob at provided point
#     if x_next_given is not None:
#         x_next = x_next_given
#     elif noise_level > 0:
#         x_next = mean + std * torch.randn_like(x_t)
#     else:
#         x_next = mean.clone()

#     # Gaussian log-prob (your "mean formulation")
#     # log p = -0.5 * mean(((x-mean)/std)^2) - log(std) - 0.5log(2pi)
#     # (per-element average, not summed over dimensions)
#     std_safe = std.clamp(min=1e-12)
#     sq_err = ((x_next - mean).pow(2) / std_safe.pow(2)).mean(dim=(1, 2, 3))  # (B,)
#     log_std_b = std_safe.reshape(B).log()  # (B,)
#     log_prob = -0.5 * sq_err - log_std_b - 0.5 * math.log(2.0 * math.pi)

#     # Optional temperature ......scaling (uncomment if desired)
#     # log_prob = log_prob / temperature

#     return x_next, log_prob, mean, std


#####given by GPT previously###############

def euler_sde_step_with_logprob(
    pred_x0: torch.Tensor,           # (B, C, H, W) float32 – model x0 prediction
    x_t: torch.Tensor,               # (B, C, H, W) float32 – current latent
    sigma_curr: torch.Tensor,        # (B,) float32 – current sigma
    sigma_next: torch.Tensor,        # (B,) float32 – next sigma (smaller)
    noise_level: float = 0.7,
    x_next_given: Optional[torch.Tensor] = None,  # evaluate log_prob at this point
    temperature: float = 5000,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    One Euler SDE denoising step for SD3 flow matching with log probability.

    SD3 forward process:   x_t = sigma_t * eps + (1 - sigma_t) * x0
    Velocity:              v_t = (x_t - x0_pred) / sigma_t
    ODE Euler step:        mean = x_t + (sigma_next - sigma_curr) * v_t
    SDE noise injection:   std  = noise_level * sqrt(2 * |sigma_next - sigma_curr| * sigma_curr)
    Sample:                x_next = mean + std * eps,  eps ~ N(0, I)

    Log prob under this Gaussian:
        log p(x_next | x_t) = -0.5 * sum((x_next - mean)^2 / std^2)
                               - D * (log(std) + 0.5 * log(2*pi))

    Args:
        x_next_given: if provided, evaluate log p(x_next_given | x_t) rather than sampling.

    Returns:
        x_next:   (B, C, H, W)
        log_prob: (B,)
        mean:     (B, C, H, W)
        std:      (B, 1, 1, 1)
    """
    B, C, H, W = x_t.shape

    sc = sigma_curr.reshape(B, 1, 1, 1)   # (B,1,1,1)
    sn = sigma_next.reshape(B, 1, 1, 1)
    dt = sn - sc                           # negative (sigma decreases)

    v = (x_t - pred_x0) / sc
    mean = x_t + dt * v
    
#     sc_eps = sc * x_t + sc * (1 - sc) * v #####the noise component of prediction
    C_0 = 0.6 * noise_level
    
#     x_t_perturbed = math.cos(C_0) * x_t + (1 - sc) * (1 - math.cos(C_0)) * pred_x0 + math.sin(C_0) * sc * torch.randn_like(x_t)

#     std = noise_level * torch.sqrt(2.0 * dt.abs() * sc).clamp(min=1e-6)  # (B,1,1,1)

    ccs_mean = math.cos(C_0) * mean + (1 - sn) * (1 - math.cos(C_0)) * pred_x0
    std = math.sin(C_0) * sn
    if x_next_given is not None:
        x_next = x_next_given
    elif noise_level > 0:
#         x_next = mean + std * torch.randn_like(x_t) ###simply adding gaussian noise
        x_next = ccs_mean + math.sin(C_0) * sn * torch.randn_like(x_t) ###CCS perturbation
    else:
        x_next = mean.clone()

    # Gaussian log prob
    D = C * H * W
#     sq_err = ((x_next - mean).pow(2) / std.pow(2).clamp(min=1e-12)).sum(dim=(1, 2, 3))  # (B,)
#     log_std_b = std.reshape(B).log()  # (B,)
#     log_prob = -0.5 * sq_err - D * (log_std_b + 0.5 * math.log(2.0 * math.pi))  # (B,)
#     log_prob = log_prob / temperature

    #################### mean formulation ##################
    sq_err = ((x_next - ccs_mean).pow(2) / std.pow(2).clamp(min=1e-12)).mean(dim=(1, 2, 3))  # (B,)
    log_std_b = std.reshape(B).log()  # (B,)
#     log_prob = -0.5 * sq_err - 1 * (log_std_b + 0.5 * math.log(2.0 * math.pi))  # (B,)
    log_prob = -0.5 * sq_err - log_std_b - 0.5 * math.log(2.0 * math.pi)

    return x_next, log_prob, mean, std


################################################################################
# Config
################################################################################

@dataclass
class TrainConfig:
    model: str
    controlnet_ckpt: str
    model_folder: str = "models"
    vae: Optional[str] = None
    shift: float = 3.0

    image_root: str = ""
    lq_image_root: str = ""
    captions_file: str = ""
    imgperprompt: int = 1

    out_dir: str = "outputs/controlnet_grpo"
    batch_size: int = 1
    num_workers: int = 2

    steps: int = 4000
    lr: float = 5e-6
    weight_decay: float = 0.0
    grad_clip: float = 1.0

    log_every: int = 1
    save_every: int = 100

    # LoRA
    lora_r: int = 128
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0

    # AMP
#     amp: bool = True
    amp: bool = False
    bf16: bool = True
    imageonly: bool = False

    resume: Optional[str] = None


@dataclass
class GRPOConfig(TrainConfig):
    # ---- GRPO-specific hyperparameters ----
    # Matched to clipscore_sd3() in flow_grpo/config/grpo.py
    num_seeds: int = 10             # K: trajectories per image  (num_image_per_prompt=24)
    clip_range: float = 1e-2       # PPO clipping range for ratio
    noise_level: float = 0.7        # SDE noise injection (0 = pure ODE)
    grpo_steps: int = 20            # Denoising steps for sampling  (num_steps=10)
    cfg_scale: float = 4.5          # Classifier-free guidance scale
    num_inner_epochs: int = 1      # Inner training epochs over collected batch
    adv_clip_max: float = 5.0       # Clip advantages to [-adv_clip_max, adv_clip_max]
    timestep_fraction: float = 0.50 # Fraction of timesteps to train on  (timestep_fraction=0.99)
    kl_beta: float = 0.00           # KL regularization weight  (beta=0.02)
    wandb_project: str = "sd3-grpo-controlnet"
    wandb_run_name: str = "grpo-run-l1lpips-seed6-ccs-2.5"
    save_suffix: str = f"grpo-run-l1lpips-seed6-ccs-2.5"


################################################################################
# Trainer
################################################################################

class GRPOTrainer:
    def __init__(self, cfg: GRPOConfig):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(cfg.out_dir, exist_ok=True)

        wandb.init(project=cfg.wandb_project, name=cfg.wandb_run_name,
                   config=vars(cfg))

        # Text encoders
        self.tokenizer = SD3Tokenizer()
        self.t5xxl = T5XXL(cfg.model_folder, "cpu", torch.float32)
        self.clip_l = ClipL(cfg.model_folder)
        self.clip_g = ClipG(cfg.model_folder, "cpu")

        # Diffusion model + ControlNet + VAE
        self.sd3 = SD3Bundle(cfg.model, cfg.controlnet_ckpt, cfg.shift,
                             verbose=False, device="cuda")
        self.vae = VAE(cfg.vae or cfg.model, dtype=torch.float16)
        self.latent_fmt = SD3LatentFormat()

        assert self.sd3.model.control_model is not None, "control_model is None"

        self.control_type = int(self.sd3.model.control_model.control_type.item())
        self.using_2b = not self.sd3.using_8b_controlnet

        # Inject LoRA into control_model only
        inject_lora_into_linears(
            self.sd3.model.control_model, cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout
        )

        # Freeze everything except LoRA params
        for p in self.sd3.model.diffusion_model.parameters():
            p.requires_grad_(False)
        mark_only_lora_trainable(self.sd3.model.control_model)
        
        
        ##############debug####################
#         for p in self.sd3.model.diffusion_model.parameters():
#             p.requires_grad_(False)
#         for p in self.sd3.model.control_model.parameters():
#             p.requires_grad_(True)
            
        ############################################
         

        train_params = [p for p in self.sd3.model.control_model.parameters()
                        if p.requires_grad]
        
        
        
        if not train_params:
            raise RuntimeError(
                "LoRA injection produced no trainable parameters. "
                "Check that control_model contains nn.Linear layers visible to "
                "inject_lora_into_linears (i.e. direct children, not inside custom wrappers)."
            )
        print(f"[GRPOTrainer] Trainable LoRA params: {len(train_params)}, "
              f"total elements: {sum(p.numel() for p in train_params):,}")
        self.opt = torch.optim.AdamW(train_params, lr=cfg.lr,
                                     weight_decay=cfg.weight_decay)
        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and not cfg.bf16))
        self.global_step = 0
        if cfg.resume:
            self._load(cfg.resume)

    # ------------------------------------------------------------------
    # Helpers (same as the SFT Trainer)
    # ------------------------------------------------------------------

    @torch.no_grad()
    def encode_prompt(self, prompt: str):
        tokens = self.tokenizer.tokenize_with_weights(prompt)
        l_out, l_pooled = self.clip_l.model.encode_token_weights(tokens["l"])
        g_out, g_pooled = self.clip_g.model.encode_token_weights(tokens["g"])
        t5_out, _ = self.t5xxl.model.encode_token_weights(tokens["t5xxl"])
        lg_out = torch.cat([l_out, g_out], dim=-1)
        lg_out = F.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
        c = torch.cat([lg_out, t5_out], dim=-2)
        y = torch.cat((l_pooled, g_pooled), dim=-1)
        return c, y

    @torch.no_grad()
    def vae_encode_gt_tensor(self, gt_tensor_m11: torch.Tensor) -> torch.Tensor:
        self.vae.model = self.vae.model.cuda()
        lat = self.vae.model.encode(gt_tensor_m11.to("cuda", dtype=torch.float32))
        self.vae.model = self.vae.model.cpu()
        return self.latent_fmt.process_in(lat)

    @torch.no_grad()
    def vae_encode_control_from_paths(self, paths: List[str]) -> torch.Tensor:
        ims = [Image.open(p).convert("RGB").resize((1024, 1024), Image.LANCZOS)
               for p in paths]
        t = torch.stack([
            torch.from_numpy(np.moveaxis(np.array(im).astype(np.float32) / 255.0, 2, 0))
            for im in ims
        ]).to("cuda", dtype=torch.float32)
        if self.using_2b:
            t = t * 2.0 - 1.0
        elif self.control_type == 1:
            t = t * 255 * 0.5 + 0.5
        else:
            t = 2.0 * t - 1.0
        self.vae.model = self.vae.model.cuda()
        lat = self.vae.model.encode(t)
        self.vae.model = self.vae.model.cpu()
        return self.latent_fmt.process_in(lat)

    @torch.no_grad()
    def _get_sigmas(self, steps: int) -> torch.Tensor:
        sampling = self.sd3.model.model_sampling
        start = sampling.timestep(sampling.sigma_max)
        end = sampling.timestep(sampling.sigma_min)
        timesteps = torch.linspace(start, end, steps, device="cuda")
        sigs = [sampling.sigma(ts) for ts in timesteps]
        sigs.append(torch.tensor(0.0, device="cuda"))
        return torch.stack(sigs).to(torch.float32)  # (steps+1,)

    @torch.no_grad()
    def _vae_decode(self, latent_out: torch.Tensor) -> Image.Image:
        """latent_out must already be process_out()'d."""
        self.vae.model = self.vae.model.cuda()
        img = self.vae.model.decode(latent_out.cuda()).float()
        self.vae.model = self.vae.model.cpu()
        img = torch.clamp((img + 1.0) / 2.0, 0.0, 1.0)[0]
        return Image.fromarray(
            (255.0 * np.moveaxis(img.cpu().numpy(), 0, 2)).astype(np.uint8)
        )

    def _save(self, path: str):
        torch.save({
            "step": self.global_step,
            "opt": self.opt.state_dict(),
            "control_lora": lora_state_dict(self.sd3.model.control_model),
            "cfg": vars(self.cfg),
        }, path)

    def _load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.global_step = int(ckpt.get("step", 0))
        ########debug, dont load optimizer state dict####
#         if "opt" in ckpt:
#             self.opt.load_state_dict(ckpt["opt"])
        ###############################################
        if "control_lora" in ckpt:
            load_lora_state_dict(self.sd3.model.control_model, ckpt["control_lora"])
        print(f"[resume] {path}  (step={self.global_step})")

    @contextlib.contextmanager
    def _disable_lora(self):
        """Temporarily zero out all LoRA scales (simulates reference / base model)."""
        lora_modules = [
            m for m in self.sd3.model.control_model.modules()
            if isinstance(m, LoRALinear)
        ]
        orig_scales = [m.scale for m in lora_modules]
        for m in lora_modules:
            m.scale = 0.0
        try:
            yield
        finally:
            for m, s in zip(lora_modules, orig_scales):
                m.scale = s

    @contextlib.contextmanager
    def _use_ref_lora(self):
        """Temporarily swap LoRA to the frozen SFT reference weights."""
        modules = {
            name: m
            for name, m in self.sd3.model.control_model.named_modules()
            if isinstance(m, LoRALinear)
        }
        # Save current (GRPO) weights
        saved = {
            name: {
                "lora_A": m.lora_A.data.clone(),
                "lora_B": m.lora_B.data.clone(),
                "scale":  m.scale
            }
            for name, m in modules.items()
        }
        # Swap in SFT reference weights
        for name, m in modules.items():
            m.lora_A.data.copy_(self._ref_lora_weights[name]["lora_A"])
            m.lora_B.data.copy_(self._ref_lora_weights[name]["lora_B"])
            m.scale = self._ref_lora_weights[name]["scale"]
        try:
            yield
        finally:
            for name, m in modules.items():
                m.lora_A.data.copy_(saved[name]["lora_A"])
                m.lora_B.data.copy_(saved[name]["lora_B"])
                m.scale = saved[name]["scale"]
#                 m.scale = self._ref_lora_weights[name]["scale"]

    # ------------------------------------------------------------------
    # GRPO sampling: one trajectory with log probs
    # ------------------------------------------------------------------

    @torch.no_grad()
    def _sample_trajectory_with_logprob(
        self,
        control_lat: torch.Tensor,   # (B, C, H, W) float16 on cuda
        c_cross: torch.Tensor,       # (B, seq, d) float16 on cuda  – positive cond
        y_cond: torch.Tensor,        # (B, d) bfloat16 on cuda
        neg_c: torch.Tensor,         # (B, seq, d) float16 on cuda  – negative cond
        neg_y: torch.Tensor,         # (B, d) bfloat16 on cuda
        seed: int,
        sigmas: torch.Tensor,        # (T+1,) float32 on cuda
        cfg_scale: float,
        noise_level: float,
        num_train_ts: int,           # how many steps to record log probs for
        sign: int,
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Run the full reverse denoising SDE and collect trajectory log probs.

        Returns:
            latents:   list of T+1 tensors, each (B, C, H, W) bfloat16 on CPU
            log_probs: list of `num_train_ts` tensors, each (B,) float32 on CPU
                       (log probs for timesteps 0 .. num_train_ts-1)
        """
        B = control_lat.shape[0]
        T = len(sigmas) - 1

        # Initial noise
        
        print(f"sampling with seed {seed} sign {sign}")
        g = torch.Generator(device="cuda").manual_seed(seed)
        x = torch.randn(control_lat.shape, generator=g,
                        device="cuda", dtype=torch.float32)
        
        ########antithetic sampling, flipping sign of x if sign = 1#########
        if sign == 1:
            x = -x
        x = x * sigmas[0]  # sigma_max ≈ 1 for SD3
        latents = [x.cpu()]
        log_probs = []

        for i in range(T):
            sigma_curr = sigmas[i].view(1).expand(B).cuda()   # (B,) bfloat16
            sigma_next = sigmas[i + 1].view(1).expand(B).cuda()

            # CFG: double the batch  [uncond | cond]
            x_in      = torch.cat([x, x], dim=0)                           # (2B,)
            sigma_in  = sigma_curr.repeat(2)                                # (2B,)
            c_in      = torch.cat([neg_c, c_cross], dim=0)                  # (2B, seq, d)
            y_in      = torch.cat([neg_y, y_cond], dim=0)                   # (2B, d)
            ctrl_in   = torch.cat([control_lat, control_lat], dim=0)        # (2B, C, H, W)

            step_noise_level = noise_level if i < num_train_ts else 0.0

            with torch.cuda.amp.autocast(enabled=True, dtype=torch.bfloat16):
                pred_x0_double = self.sd3.model.apply_model(
                    x_in, sigma_in,
                    c_crossattn=c_in,
                    y=y_in,
                    controlnet_cond=ctrl_in,
                )

            pred_x0_uncond, pred_x0_cond = pred_x0_double.float().chunk(2, dim=0)
            pred_x0 = pred_x0_uncond + cfg_scale * (pred_x0_cond - pred_x0_uncond)

            x_next, log_prob, _, _ = euler_sde_step_with_logprob(
                pred_x0, x.float(), sigma_curr, sigma_next, step_noise_level
            )

            x = x_next
            latents.append(x_next.detach().cpu())

            if i < num_train_ts:
                log_probs.append(log_prob.cpu())

        return latents, log_probs

    # ------------------------------------------------------------------
    # GRPO training: recompute log prob WITH gradients
    # ------------------------------------------------------------------

    def snapshot_ref_lora(self):
        ##############create snapshot of the reference policy lora weights########
        self._ref_lora_weights = {}
        for name, module in self.sd3.model.control_model.named_modules():
            if isinstance(module, LoRALinear):
                self._ref_lora_weights[name] = {
                    "lora_A": module.lora_A.data.clone(),
                    "lora_B": module.lora_B.data.clone(),
                    "scale":  module.scale,
                }
        ###########################################################
            
            
    
    def _recompute_log_prob(
        self,
        x_t: torch.Tensor,           # (B, C, H, W) float32 on cuda
        x_next: torch.Tensor,        # (B, C, H, W) float32 on cuda  – stored target
        sigma_curr: torch.Tensor,    # (B,) float32 on cuda
        sigma_next: torch.Tensor,    # (B,) float32 on cuda
        c_cross: torch.Tensor,
        y_cond: torch.Tensor,
        neg_c: torch.Tensor,
        neg_y: torch.Tensor,
        control_lat: torch.Tensor,
        cfg_scale: float,
        noise_level: float,
        autocast_enabled: bool,
        autocast_dtype: torch.dtype,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass with gradients to recompute log p(x_next | x_t)."""
        x_in     = torch.cat([x_t, x_t], dim=0)
        sigma_in = sigma_curr.repeat(2)
        c_in     = torch.cat([neg_c, c_cross], dim=0)
        y_in     = torch.cat([neg_y, y_cond], dim=0)
        ctrl_in  = torch.cat([control_lat, control_lat], dim=0)

        with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=autocast_dtype):
            pred_x0_double = self.sd3.model.apply_model(
                x_in, sigma_in,
                c_crossattn=c_in,
                y=y_in,
                controlnet_cond=ctrl_in,
            )

        pred_x0_uncond, pred_x0_cond = pred_x0_double.float().chunk(2, dim=0)
        pred_x0 = pred_x0_uncond + cfg_scale * (pred_x0_cond - pred_x0_uncond)

        _, log_prob, mean, std = euler_sde_step_with_logprob(
            pred_x0, x_t.float(), sigma_curr, sigma_next, noise_level,
            x_next_given=x_next,
        )
        return log_prob, mean, std

    # ------------------------------------------------------------------
    # Main GRPO training loop
    # ------------------------------------------------------------------

    def train_grpo(self):
        cfg = self.cfg
        K = cfg.num_seeds
        T = cfg.grpo_steps
        num_train_ts = int(T * cfg.timestep_fraction)

        autocast_dtype = torch.bfloat16 if cfg.bf16 else torch.float16
        autocast_enabled = cfg.amp or cfg.bf16

        # Dataset
        if cfg.imageonly:
            dataset = ImageDataset(cfg.image_root, cfg.lq_image_root)
        else:
            dataset = ImageCaptionDatasetLarge(
                cfg.image_root, cfg.captions_file, cfg.lq_image_root,
                imgperprompt=cfg.imgperprompt,
            )
            
        seed_ = 45
        g2 = torch.Generator()
        g2.manual_seed(seed_)
        
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=cfg.num_workers, drop_last=True, generator=g2,)

        # Precompute negative prompt embeddings once
        neg_c_raw, neg_y_raw = self.encode_prompt("")   # (1, seq, d), (1, d) on CPU

        it = iter(loader)
        start = time.time()

        while self.global_step < cfg.steps:

            # ---- Load one batch ----------------------------------------
            try:
                if cfg.imageonly:
                    images_m11, lq_paths, gt_paths = next(it)
                    captions = None
                else:
                    images_m11, lq_paths, captions = next(it)
            except StopIteration:
                it = iter(loader)
                if cfg.imageonly:
                    images_m11, lq_paths, gt_paths = next(it)
                    captions = None
                else:
                    images_m11, lq_paths, captions = next(it)

            lq_paths = list(lq_paths)
            B = images_m11.shape[0]

            # ---- Encode latents and text cond (no grad) ----------------
            with torch.no_grad():
                x0 = self.vae_encode_gt_tensor(images_m11)          # (B, 16, H, W)
                control_lat = self.vae_encode_control_from_paths(lq_paths)  # (B, 16, H, W)
                captions_list = (
                    [DEFAULT_PROMPT] * B if captions is None
                    else list(captions)
                )
                c_list, y_list = [], []
                for cap in captions_list:
                    u = random.uniform(0, 1)
                    p = DEFAULT_PROMPT if u < 0.4 else f"{cap}, {DEFAULT_PROMPT}"
                    c, y = self.encode_prompt(p)
                    c_list.append(c)
                    y_list.append(y)
                c_cross = torch.cat(c_list, dim=0).to("cuda", dtype=torch.float16)   # (B,seq,d)
                y_cond  = torch.cat(y_list, dim=0).to("cuda", dtype=torch.float16)   # (B,d)
                # Expand negative cond to batch size
                neg_c = neg_c_raw.expand(B, -1, -1).to("cuda", dtype=torch.float16)  # (B,seq,d)
                neg_y = neg_y_raw.expand(B, -1).to("cuda", dtype=torch.float16)       # (B,d)
                # Sigma schedule
                sigmas = self._get_sigmas(T)   # (T+1,) float32 on cuda
                # GT images for reward (decode from x0)
                gt_images = []
                for b in range(B):
                    gt_lat_out = self.latent_fmt.process_out(x0[b:b+1].clone())
                    gt_images.append(self._vae_decode(gt_lat_out))
                    
            # ==================== SAMPLING PHASE ========================
            # Sample K trajectories, each with a different seed.
            # Store latents (CPU) and old log probs (CPU).
            all_latent_seqs = []   # [K] each: list of T+1 tensors (B,C,H,W) float16 CPU
            all_log_prob_seqs = [] # [K] each: list of num_train_ts tensors (B,) float32 CPU
            all_gen_images = []    # [K] each: list of B PIL images

            self.sd3.model.eval()
            with torch.no_grad():
                for k in tqdm(range(K),
                              desc=f"[step {self.global_step}] Sampling",
                              leave=False):
                    ####Antithetic sampling
#                     seed = self.global_step * K + k//2 * 2 #####different seeds
#                     sign = k % 2 ###different seeds
                    
                    seed = self.global_step * K ###test same seed
                    sign = 0
#                     sign = k % 2

                    latents, log_probs = self._sample_trajectory_with_logprob(
                        control_lat, c_cross, y_cond, neg_c, neg_y,
                        seed=seed, sigmas=sigmas,
                        cfg_scale=cfg.cfg_scale, noise_level=cfg.noise_level,
                        num_train_ts=num_train_ts, sign = sign
                    )

                    # Decode final latent to PIL for reward computation
                    final_lat = latents[-1].to("cuda", dtype=torch.bfloat16)
                    final_lat_out = self.latent_fmt.process_out(final_lat.clone())
                    gen_imgs = [self._vae_decode(final_lat_out[b:b+1]) for b in range(B)]

                    all_latent_seqs.append(latents)
                    all_log_prob_seqs.append(log_probs)
                    all_gen_images.append(gen_imgs)

            # ==================== REWARD COMPUTATION ====================
            rewards = torch.zeros(K, B)   # (K, B) for l2 reward
            ########################for L2 reward##################################
#             for k in range(K):
#                 for b in range(B):
#                     rewards[k, b] = reward_model(all_gen_images[k][b], gt_images[b])
                    
            #######################################################################
        
            ###########################different rewards##########################
            for k in range(K):
# #                 rewards[k] = reward_model_lpips_batch(all_gen_images[k], gt_images) ###pure Alex lpips
                rewards[k] = reward_model_lpips_l1_batch(all_gen_images[k], gt_images) ####VGG lpips and L1
                    
                              

            # ==================== ADVANTAGE COMPUTATION =================
            # Normalize within K group per base image
            adv_mean = rewards.mean(dim=0, keepdim=True)              # (1, B)
            adv_std  = rewards.std(dim=0, unbiased=False, keepdim=True).clamp(min=1e-4)  # (1, B)
            
            print(adv_mean, "Adv mean", adv_std, "adv std")
            advantages = (rewards - adv_mean) / adv_std               # (K, B)

            wandb.log({
                "reward/mean":   rewards.mean().item(),
                "reward/std":    rewards.std().item(),
                "adv/abs_mean":  advantages.abs().mean().item(),
            }, step=self.global_step)

            # Skip update if all rewards are identical (no learning signal)
            if advantages.abs().max().item() < 1e-6:
                print(f"[step {self.global_step}] All advantages zero, skipping.")
                continue

            # ==================== TRAINING PHASE ========================
            self.sd3.model.train()
            self.opt.zero_grad(set_to_none=True)

            total_updates = K * num_train_ts * cfg.num_inner_epochs
            total_ppo_loss = 0.0

            for inner_epoch in range(cfg.num_inner_epochs):
                k_order = list(range(K))
                random.shuffle(k_order)

                for k in k_order:
                    adv_k = (
                        advantages[k]
                        .clamp(-cfg.adv_clip_max, cfg.adv_clip_max)
                        .to("cuda")
                    )  # (B,)
                    
                    ###############################debug##################################################
#                     adv_k = 1
                    #####################################################################################

                    for j in range(num_train_ts):
                        x_t    = all_latent_seqs[k][j].to("cuda", dtype=torch.float32)
                        x_next = all_latent_seqs[k][j + 1].to("cuda", dtype=torch.float32)
                        sigma_curr = sigmas[j].view(1).expand(B).cuda()
                        sigma_next = sigmas[j + 1].view(1).expand(B).cuda()
                        old_log_prob = all_log_prob_seqs[k][j].to("cuda")   # (B,)

                        # torch.enable_grad() ensures grad tracking is active even
                        # if a no_grad context leaked from the sampling/encoding phase.
                        with torch.enable_grad():
                            log_prob_new, mean_new, std_new = self._recompute_log_prob(
                                x_t, x_next, sigma_curr, sigma_next,
                                c_cross, y_cond, neg_c, neg_y, control_lat,
                                cfg.cfg_scale, cfg.noise_level,
                                autocast_enabled, autocast_dtype,
                            )

                        # Reference forward pass for KL regularization
                        if cfg.kl_beta > 0:
                            with torch.no_grad():
                                with self._use_ref_lora():
                                    _, mean_ref, _ = self._recompute_log_prob(
                                        x_t, x_next, sigma_curr, sigma_next,
                                        c_cross, y_cond, neg_c, neg_y, control_lat,
                                        cfg.cfg_scale, cfg.noise_level,
                                        autocast_enabled, autocast_dtype,
                                    )

                        # PPO-clipped policy gradient loss
                        ratio = torch.exp(log_prob_new - old_log_prob.detach())
                        unclipped = -adv_k * ratio
                        clipped   = -adv_k * torch.clamp(
                            ratio, 1.0 - cfg.clip_range, 1.0 + cfg.clip_range
                        )
                        
                        print(unclipped.item(), "unclipped", clipped.item(), "clipped")
                        ppo_loss = torch.mean(torch.maximum(unclipped, clipped))
                        print(ppo_loss, "ppo loss")

                        # KL regularization: penalizes drift from the reference model
                        # KL ≈ ||mean_new - mean_ref||² / (2 * std²)  (Gaussian KL, same std)
#                         cfg.kl_beta = 0 ###just for testing
                        if cfg.kl_beta > 0:
                            print("using KL loss")
                            kl_loss = (
                                (mean_new - mean_ref).pow(2)
                                .mean(dim=(1, 2, 3), keepdim=True)
                                / (2.0 * std_new.pow(2))
                            )
                            kl_loss = torch.mean(kl_loss)
                            loss = (ppo_loss + cfg.kl_beta * kl_loss) / total_updates
                        else:
                            print("skip kl loss")
                            kl_loss = None
                            loss = ppo_loss / total_updates

                        print(log_prob_new, "logprobnew", old_log_prob, "oldlogprob")
                        print(ratio.item(), "logprob ratio", "inner epoch", inner_epoch)
#                         print(adv_k.item(), "advantage k", "inner epoch", inner_epoch)
                        delta = (log_prob_new - old_log_prob).detach()
                        print(delta.item(), "delta_logp", torch.exp(delta).item(), "ratio", "inner epoch", inner_epoch)
                        if self.scaler.is_enabled():
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()
                        total_ppo_loss += ppo_loss.item()

                # Gradient clipping and optimizer step
                if cfg.grad_clip > 0:
                    
                    if self.scaler.is_enabled():
                        self.scaler.unscale_(self.opt)
                    nn.utils.clip_grad_norm_(
                        [p for p in self.sd3.model.control_model.parameters()
                         if p.requires_grad],
                        cfg.grad_clip,
                    )

                if self.scaler.is_enabled():
                    self.scaler.step(self.opt)
                    self.scaler.update()
                    print("enable gradient scaler")
                else:
                    print("disable gradient scaler")
                    self.opt.step()
                self.opt.zero_grad(set_to_none=True)

                avg_loss = total_ppo_loss / total_updates
                log_dict = {"train/ppo_loss": avg_loss}
                if kl_loss is not None:
                    log_dict["train/kl_loss"] = kl_loss.item()

            # ---- Image logging (first sample in batch) ------------------
            # Log: GT | best-reward gen | worst-reward gen
            if self.global_step % cfg.log_every == 0:
                b_idx = 0  # log first sample in batch
                best_k  = int(rewards[:, b_idx].argmax().item())
                worst_k = int(rewards[:, b_idx].argmin().item())

                gt_img   = gt_images[b_idx]                    # PIL
                best_img  = all_gen_images[best_k][b_idx]      # PIL
                worst_img = all_gen_images[worst_k][b_idx]     # PIL

                # Resize all to same height for a side-by-side strip
                h = min(gt_img.height, best_img.height, worst_img.height)
                def _resize_h(img, h):
                    w = int(img.width * h / img.height)
                    return img.resize((w, h), resample=Image.LANCZOS)

                gt_img_r    = _resize_h(gt_img, h)
                best_img_r  = _resize_h(best_img, h)
                worst_img_r = _resize_h(worst_img, h)

                total_w = gt_img_r.width + best_img_r.width + worst_img_r.width
                strip = Image.new("RGB", (total_w, h))
                strip.paste(gt_img_r,    (0, 0))
                strip.paste(best_img_r,  (gt_img_r.width, 0))
                strip.paste(worst_img_r, (gt_img_r.width + best_img_r.width, 0))

                log_dict["images/gt_best_worst"] = wandb.Image(
                    strip,
                    caption=(
                        f"GT | best(k={best_k}, r={rewards[best_k, b_idx]:.3f}) | "
                        f"worst(k={worst_k}, r={rewards[worst_k, b_idx]:.3f})"
                    ),
                )

            wandb.log(log_dict, step=self.global_step)

            if self.global_step % cfg.log_every == 0:
                dt = time.time() - start
                print(
                    f"[step {self.global_step:6d}]  "
                    f"ppo_loss={avg_loss:.6f}  "
                    f"reward={rewards.mean():.4f}±{rewards.std():.4f}  "
                    f"({dt:.1f}s)"
                )

            if self.global_step % cfg.save_every == 0:
                stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(
                    cfg.out_dir,
                    f"controlnet_{cfg.save_suffix}_step{self.global_step}_{stamp}.pt",
                )
                print(f"[save] {path}")
                self._save(path)

            self.global_step += 1

        # Final save
        final_path = os.path.join(
            cfg.out_dir,
            f"controlnet_{cfg.save_suffix}_final_step{self.global_step}.pt",
        )
        self._save(final_path)
        print(f"[save final] {final_path}")


################################################################################
# Entry point


##use case: python sd3_large_controlnet_grpo.py --model models/sd3.5_large.safetensors --controlnet_ckpt models/sd3.5_large_controlnet_blur.safetensors --model_folder models --image_root /scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/images_large --lq_image_root /scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images_large --captions_file /scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_prompts_large --num_seeds 6 --grpo_steps 30 --cfg_scale 4.5 --noise_level 0.7 --clip_range 1e-3 --lr 5e-6 --timestep_fraction 0.5
#################################################################

def main(**kwargs):
    cfg = GRPOConfig(**kwargs)
    trainer = GRPOTrainer(cfg)
    trainer._load("outputs/controlnet_lora/controlnet_lora_step2000_20260127_074346_large.pt")
    trainer.snapshot_ref_lora()
    trainer.train_grpo()


if __name__ == "__main__":
    fire.Fire(main)
