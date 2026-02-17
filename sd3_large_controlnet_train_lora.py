#!/usr/bin/env python
# sd3_controlnet_lora_train_with_ImageCaptionDataset.py

import os
import math
import time
import random
import datetime
from dataclasses import dataclass
from glob import glob
from typing import List, Optional
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
import re
import random
import wandb
wandb.init(project="sd3-train-controlnet", name="run-115", config={"lr": 1e-4, "bs": 4})
DEFAULT_PROMPT = "a high-resolution and sharp image, Cinematic, hyper sharpness, highly detailed, perfect without deformations, hyper detailed photo - realistic maximum detail"


################################################################################
# Your dataset (used as-is)
################################################################################
# a high-resolution and sharp image, Cinematic, hyper sharpness, highly detailed, perfect without deformations, hyper detailed photo - realistic maximum detail
# ##########

class ImageDataset(Dataset):
    def __init__(self, image_root: str, lq_image_root: str):
        self.image_root = image_root
        self.items: List[str] = []
        self.lq_image_root = lq_image_root
        

        self.img_items: List[str] = []
        img_files = glob(f'{image_root}/*.png')
        img_files.sort()
        lq_img_files = glob(f'{lq_image_root}/*.png')
        lq_img_files.sort()
        
        self.img_items = img_files
        self.lqimg_items = lq_img_files

    def __len__(self) -> int:
        return len(self.lqimg_items)

    def __getitem__(self, idx: int):
        path = self.img_items[idx]
        lq_path = self.lqimg_items[idx]

        
        image = Image.open(path).convert("RGB")
       

        # Normalize like sd3_infer: [0,1] -> [-1,1]
        image = image.resize((1024, 1024), Image.LANCZOS)
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = np.moveaxis(image_np, 2, 0)
        tensor = torch.from_numpy(image_np)
        tensor = 2.0 * tensor - 1.0

        return tensor, lq_path, path
    
    

    
    
class ImageCaptionDatasetLarge(Dataset):
    """
    
    /scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_prompts_large/
    
    Very small reference dataset:
      - image_root/: image files
      - captions_file: text file of "filename<TAB>caption" per line
    """
    def __init__(self, image_root: str, captions_root: str, lq_image_root: str, imgperprompt=1):
        self.image_root = image_root
        self.items: List[str] = []
        self.lq_image_root = lq_image_root
        self.imgperprompt = imgperprompt

        self.img_items: List[str] = []
        img_files = glob(f'{image_root}/*.png')
        img_files.sort()
        lq_img_files = glob(f'{lq_image_root}/*.png')
        lq_img_files.sort()
        caption_files = glob(f'{captions_root}/*.txt')
        caption_files.sort()
        self.img_items = img_files
        self.items = caption_files
        self.lqimg_items = lq_img_files

    def __len__(self) -> int:
        return len(self.lqimg_items)

    def __getitem__(self, idx: int):
        path = self.img_items[idx]
        lq_path = self.lqimg_items[idx]
        caption_idx = (idx // self.imgperprompt)
        caption_path = self.items[caption_idx]
        image = Image.open(path).convert("RGB")
        caption = ""
        with open(caption_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                caption = line

        # Normalize like sd3_infer: [0,1] -> [-1,1]
        image = image.resize((1024, 1024), Image.LANCZOS)
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = np.moveaxis(image_np, 2, 0)
        tensor = torch.from_numpy(image_np)
        tensor = 2.0 * tensor - 1.0

        return tensor, lq_path, caption
    

class ImageCaptionDataset(Dataset):
    """
    
    /scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_prompts_large/
    
    Very small reference dataset:
      - image_root/: image files
      - captions_file: text file of "filename<TAB>caption" per line
    """
    def __init__(self, image_root: str, captions_file: str, lq_image_root: str, imgperprompt=10):
        self.image_root = image_root
        self.items: List[str] = []
        self.lq_image_root = lq_image_root
        self.imgperprompt = imgperprompt

        self.img_items: List[str] = []
        img_files = glob(f'{image_root}/*.png')
        img_files.sort()
        lq_img_files = glob(f'{lq_image_root}/*.png')
        lq_img_files.sort()
        caption_files = glob(f'{lq_image_root}/*.txt')
        caption_files.sort()
        self.img_items = img_files
        self.items = caption_files
        self.lqimg_items = lq_img_files

    def __len__(self) -> int:
        return len(self.lqimg_items)

    def __getitem__(self, idx: int):
        path = self.img_items[idx]
        lq_path = self.lqimg_items[idx]
        caption_idx = (idx // self.imgperprompt)
        caption_path = self.items[caption_idx]
        image = Image.open(path).convert("RGB")
        caption = ""
        with open(caption_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                caption = line

        # Normalize like sd3_infer: [0,1] -> [-1,1]
        image = image.resize((1024, 1024), Image.LANCZOS)
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = np.moveaxis(image_np, 2, 0)
        tensor = torch.from_numpy(image_np)
        tensor = 2.0 * tensor - 1.0

        return tensor, lq_path, caption


################################################################################
# Loading helpers (same style as your inference script)
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
            ).train()
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
# LoRA for Linear only
################################################################################

class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0, device = "cuda"):
        super().__init__()
        self.base = base
        self.r = int(r)
        self.alpha = float(alpha)
        self.scale = self.alpha / self.r if self.r > 0 else 0.0
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        in_f, out_f = base.in_features, base.out_features
        self.lora_A = nn.Parameter(torch.empty(self.r, in_f, dtype=torch.float32, device = device))
        self.lora_B = nn.Parameter(torch.zeros(out_f, self.r, dtype=torch.float32, device = device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # freeze base weights
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

    def forward(self, x):
        y = self.base(x)
        x_d = self.dropout(x).float()
        
#         print(x_d.device, self.lora_A.device, self.lora_B.device, "xd loraa lorab")
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
# Trainer
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
    imgperprompt: int = 10

    out_dir: str = "outputs/controlnet_lora"
    batch_size: int = 1
    num_workers: int = 2

    steps: int = 2000
    lr: float = 1e-4
    weight_decay: float = 0.0
    grad_accum: int = 1
    grad_clip: float = 1.0

    log_every: int = 2
    save_every: int = 100

    # LoRA
    lora_r: int = 128 ###previous 16
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0

    # noise schedule sampling
    min_t: int = 1
    max_t: int = 1000

    # AMP
    amp: bool = True
    bf16: bool = False
    imageonly: bool = False

    resume: Optional[str] = None


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        os.makedirs(cfg.out_dir, exist_ok=True)

        # Tokenizers + encoders
        self.tokenizer = SD3Tokenizer()
        self.t5xxl = T5XXL(cfg.model_folder, "cpu", torch.float32)
        self.clip_l = ClipL(cfg.model_folder)
        self.clip_g = ClipG(cfg.model_folder, "cpu")

        # SD3 + ControlNet + VAE
        self.sd3 = SD3Bundle(cfg.model, cfg.controlnet_ckpt, cfg.shift, verbose=False, device="cuda")
        self.vae = VAE(cfg.vae or cfg.model, dtype=torch.float16)
        self.latent_fmt = SD3LatentFormat()

        assert self.sd3.model.control_model is not None, "control_model is None. Bad controlnet_ckpt?"

        # control preprocess flags (matching your BaseModel.apply_model / inferencer)
        self.control_type = int(self.sd3.model.control_model.control_type.item())
        self.using_2b = not self.sd3.using_8b_controlnet

        # -------------------- ONLY inject LoRA into control_model --------------------
        inject_lora_into_linears(self.sd3.model.control_model, cfg.lora_r, cfg.lora_alpha, cfg.lora_dropout)
        # ---------------------------------------------------------------------------

        # -------------------- Freeze everything except LoRA params -----------------
        for p in self.sd3.model.diffusion_model.parameters():
            p.requires_grad_(False)
        mark_only_lora_trainable(self.sd3.model.control_model)
        # ---------------------------------------------------------------------------

        train_params = [p for p in self.sd3.model.control_model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(train_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

        self.scaler = torch.cuda.amp.GradScaler(enabled=(cfg.amp and not cfg.bf16))
        self.global_step = 0
        self.micro_step = 0

        if cfg.resume:
            self._load(cfg.resume)

    @torch.no_grad()
    def encode_prompt(self, prompt: str):
        tokens = self.tokenizer.tokenize_with_weights(prompt)
        l_out, l_pooled = self.clip_l.model.encode_token_weights(tokens["l"])
        g_out, g_pooled = self.clip_g.model.encode_token_weights(tokens["g"])
        t5_out, t5_pooled = self.t5xxl.model.encode_token_weights(tokens["t5xxl"])
        lg_out = torch.cat([l_out, g_out], dim=-1)
        lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
        c = torch.cat([lg_out, t5_out], dim=-2)
        y = torch.cat((l_pooled, g_pooled), dim=-1)
        return c, y

    @torch.no_grad()
    def vae_encode_gt_tensor(self, gt_tensor_m11: torch.Tensor) -> torch.Tensor:
        """
        gt_tensor_m11: (B,3,H,W) in [-1,1] float32 from your dataset.
        """
        self.vae.model = self.vae.model.cuda()
        lat = self.vae.model.encode(gt_tensor_m11.to("cuda", dtype=torch.float32))
        self.vae.model = self.vae.model.cpu()
        lat = self.latent_fmt.process_in(lat)
        return lat

    @torch.no_grad()
    def vae_encode_control_from_paths(self, paths: List[str], gt_paths = None) -> torch.Tensor:
        """
        Loads control image(s) from lq_path(s), resizes to 1024, applies inferencer-style preprocessing,
        then VAE encodes -> SD3LatentFormat.process_in.
        """
        ims = []
        
        if gt_paths is None:
            for p in paths:
                im = Image.open(p).convert("RGB").resize((1024, 1024), Image.LANCZOS)
                ims.append(im)
                
        else:
            for p in gt_paths:
                im = Image.open(p).convert("RGB").resize((1024, 1024), Image.LANCZOS)
                ims.append(im)
                

        # [0,1]
        t = torch.stack([torch.from_numpy(np.moveaxis(np.array(im).astype(np.float32) / 255.0, 2, 0)) for im in ims], dim=0)
        t = t.to("cuda", dtype=torch.float32)

        # match inferencer.vae_encode preprocessing
        if self.using_2b:
            t = t * 2.0 - 1.0
        elif self.control_type == 1:  # canny
            t = t * 255 * 0.5 + 0.5
        else:
            t = 2.0 * t - 1.0

        self.vae.model = self.vae.model.cuda()
        lat = self.vae.model.encode(t)
        self.vae.model = self.vae.model.cpu()
        lat = self.latent_fmt.process_in(lat)
        return lat

    def sample_sigmas(self, bsz: int):
        t = torch.randint(self.cfg.min_t, self.cfg.max_t + 1, (bsz,), device="cuda", dtype=torch.int64)
        sigma = self.sd3.model.model_sampling.sigma(t.float()).to(torch.float32)
        return sigma

    def _save(self, path: str):
        ckpt = {
            "step": self.global_step,
            "micro_step": self.micro_step,
            "opt": self.opt.state_dict(),
            "control_lora": lora_state_dict(self.sd3.model.control_model),
            "cfg": vars(self.cfg),
        }
        torch.save(ckpt, path)

    def _load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.global_step = int(ckpt.get("step", 0))
        self.micro_step = int(ckpt.get("micro_step", 0))
        if "opt" in ckpt:
            self.opt.load_state_dict(ckpt["opt"])
        if "control_lora" in ckpt:
            load_lora_state_dict(self.sd3.model.control_model, ckpt["control_lora"])
        print(f"[resume] {path} (step={self.global_step}, micro={self.micro_step})")

    def train(self):
        cfg = self.cfg
        
        img_only = self.cfg.imageonly
        
        if not img_only:
#             dataset = ImageCaptionDataset(cfg.image_root, cfg.captions_file, cfg.lq_image_root, imgperprompt=cfg.imgperprompt)###image caption dataset medium
            
            dataset = ImageCaptionDatasetLarge(cfg.image_root, cfg.captions_file, cfg.lq_image_root, imgperprompt =1)
        else:
            dataset = ImageDataset(cfg.image_root, cfg.lq_image_root) ###image dataset only
            print(len(dataset), "length of the image dataset")
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)

        autocast_dtype = torch.bfloat16 if cfg.bf16 else torch.float16
        autocast_enabled = cfg.amp or cfg.bf16

        self.opt.zero_grad(set_to_none=True)
        start = time.time()

        it = iter(loader)
        DEGRADE_FLAG = False
        while self.global_step < cfg.steps:
            
            if img_only:
                try:
                    images_m11, lq_paths, gt_paths = next(it)
                except StopIteration:
                    it = iter(loader)
                    images_m11, lq_paths, gt_paths = next(it)

                self.micro_step += 1
                captions = None
                lq_paths = list(lq_paths)
            else:
                try:
                    images_m11, lq_paths, captions = next(it)
                    print("using lq_path", lq_paths[0], "caption path", captions[0])
                except StopIteration:
                    it = iter(loader)
                    images_m11, lq_paths, captions = next(it)
                    captions = list(captions)
                    lq_paths = list(lq_paths)

            self.micro_step += 1

            # Encode latents (no grad)
            with torch.no_grad():
                x0 = self.vae_encode_gt_tensor(images_m11)                # (B,16,128,128) typically
                if not DEGRADE_FLAG:
                    control_lat = self.vae_encode_control_from_paths(lq_paths)
                else:
                    control_lat = self.vae_encode_control_from_paths(lq_paths, gt_paths)
                
                # text cond
                c_list, y_list = [], []
                
                if captions is None:
                    print("using DEFAULT CAPTION")
                    captions = [DEFAULT_PROMPT] * len(lq_paths)
                for p in captions:
                    
                    U = random.uniform(0,1)
                    if U < 0.4:
                        p = DEFAULT_PROMPT
                    else:
                        p = f"{p}, {DEFAULT_PROMPT}"
                        
                    c, y = self.encode_prompt(p)
                    c_list.append(c)
                    y_list.append(y)
                c_cross = torch.cat(c_list, dim=0).to("cuda", dtype=torch.float16)
                y = torch.cat(y_list, dim=0).to("cuda", dtype=torch.float16)

            sigma = self.sample_sigmas(x0.shape[0])
            noise = torch.randn_like(x0, dtype=torch.float32).to(x0.dtype)
            
            b = x0.shape[0]
            x = sigma.view(b,1, 1, 1) * noise + (1.0 - sigma).view(b,1,1,1) * x0
#             x = self.sd3.model.model_sampling.noise_scaling(sigma, noise, x0, max_denoise=False)

            # Forward only trains LoRA params in control_model
            with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=autocast_dtype):
                pred_x0 = self.sd3.model.apply_model(
                    x, sigma,
                    c_crossattn=c_cross,
                    y=y,
                    controlnet_cond=control_lat,
                )
                loss = F.mse_loss(pred_x0, x0)
                print(loss.item(), "loss at", self.global_step, " global step")
                wandb.log({"train/loss": float(loss.item())}, step=self.global_step)
                loss_scaled = loss / max(1, cfg.grad_accum)

            if self.scaler.is_enabled():
                self.scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

            # optimizer step every grad_accum microbatches
            if (self.micro_step % max(1, cfg.grad_accum)) == 0:
                if cfg.grad_clip and cfg.grad_clip > 0:
                    if self.scaler.is_enabled():
                        self.scaler.unscale_(self.opt)
                    nn.utils.clip_grad_norm_(
                        [p for p in self.sd3.model.control_model.parameters() if p.requires_grad],
                        cfg.grad_clip
                    )

                if self.scaler.is_enabled():
                    self.scaler.step(self.opt)
                    self.scaler.update()
                else:
                    self.opt.step()

                self.opt.zero_grad(set_to_none=True)
                self.global_step += 1

                if self.global_step % cfg.log_every == 0:
                    dt = time.time() - start
                    print(f"[step {self.global_step:6d}] loss={loss.item():.6f}  ({dt:.1f}s)")

                if self.global_step % cfg.save_every == 0:
                    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    path = os.path.join(cfg.out_dir, f"controlnet_lora_step{self.global_step}_{stamp}_large_promptflip.pt")
                    print(f"[save] {path}")
                    self._save(path)

        final_path = os.path.join(cfg.out_dir, f"controlnet_lora_final_step{self.global_step}._large_promptflip.pt")
        print(f"[save final] {final_path}")
        self._save(final_path)
        
        
    @torch.no_grad()
    def _get_empty_latent(self, batch_size: int, width: int, height: int, seed: int, device: str = "cuda"):
        shape = (batch_size, 16, height // 8, width // 8)
        latents = torch.zeros(shape, device=device, dtype=torch.float16)
        for i in range(batch_size):
            g = torch.Generator(device=device).manual_seed(int(seed + i))
            latents[i] = torch.randn(shape[1:], generator=g, device=device, dtype=torch.float16)
        return latents

    @torch.no_grad()
    def _get_sigmas(self, sampling, steps: int):
        start = sampling.timestep(sampling.sigma_max)
        end = sampling.timestep(sampling.sigma_min)
        timesteps = torch.linspace(start, end, steps, device="cuda")
        sigs = [sampling.sigma(ts) for ts in timesteps]
        sigs += [torch.tensor(0.0, device="cuda")]
        return torch.stack(sigs).to(torch.float32)  # (steps+1,)

    @torch.no_grad()
    def _get_noise(self, seed: int, latent: torch.Tensor):
        # match reference: generate on CPU float32 then cast
        g = torch.manual_seed(int(seed))
        noise = torch.randn(latent.size(), dtype=torch.float32, generator=g, device="cpu")
        return noise.to(device=latent.device, dtype=latent.dtype)

    @torch.no_grad()
    def _max_denoise(self, sigmas: torch.Tensor):
        max_sigma = float(self.sd3.model.model_sampling.sigma_max)
        sigma0 = float(sigmas[0].item())
        return math.isclose(max_sigma, sigma0, rel_tol=1e-05) or sigma0 > max_sigma

    @torch.no_grad()
    def _fix_cond(self, cond_tuple):
        # cond_tuple: (c_crossattn, y)
        cond, pooled = cond_tuple
        return {"c_crossattn": cond.half().cuda(), "y": pooled.half().cuda()}

    @torch.no_grad()
    def _image_to_latent(self, image_path: str, width: int, height: int, using_2b_controlnet: bool, controlnet_type: int):
        # follows inferencer._image_to_latent + inferencer.vae_encode preprocessing
        im = Image.open(image_path).convert("RGB").resize((width, height), Image.LANCZOS)

        image_np = np.array(im).astype(np.float32) / 255.0
        image_np = np.moveaxis(image_np, 2, 0)
        image_t = torch.from_numpy(np.expand_dims(image_np, axis=0)).to("cuda", dtype=torch.float32)

        if using_2b_controlnet:
            image_t = image_t * 2.0 - 1.0
        elif controlnet_type == 1:  # canny
            image_t = image_t * 255 * 0.5 + 0.5
        else:
            image_t = 2.0 * image_t - 1.0

        self.vae.model = self.vae.model.cuda()
        lat = self.vae.model.encode(image_t).cpu()
        self.vae.model = self.vae.model.cpu()

        lat = self.latent_fmt.process_in(lat)
        return lat.to("cuda", dtype=torch.float16)

    @torch.no_grad()
    def _vae_decode(self, latent_out: torch.Tensor) -> Image.Image:
        # latent_out is SD3LatentFormat.process_out(latent) already (same as inferencer.do_sampling)
        self.vae.model = self.vae.model.cuda()
        img = self.vae.model.decode(latent_out.cuda())
        img = img.float()
        self.vae.model = self.vae.model.cpu()

        img = torch.clamp((img + 1.0) / 2.0, 0.0, 1.0)[0]
        decoded_np = (255.0 * np.moveaxis(img.cpu().numpy(), 0, 2)).astype(np.uint8)
        return Image.fromarray(decoded_np)

    @torch.no_grad()
    def _do_sampling(
        self,
        latent: torch.Tensor,
        seed: int,
        conditioning,         # (c_crossattn, y)
        neg_cond,             # (c_crossattn, y)
        steps: int,
        cfg_scale: float,
        sampler: str = "dpmpp_2m",
        controlnet_cond: Optional[torch.Tensor] = None,
        denoise: float = 1.0,
        skip_layer_config: dict = None,
    ) -> torch.Tensor:
        if skip_layer_config is None:
            skip_layer_config = {}

        latent = latent.half().cuda()
        self.sd3.model = self.sd3.model.cuda()

        noise = self._get_noise(seed, latent)
        sigmas = self._get_sigmas(self.sd3.model.model_sampling, steps)
        sigmas = sigmas[int(steps * (1 - denoise)) :]

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

        sample_fn = getattr(sd3_impls, f"sample_{sampler}")

        # keep same behavior as inferencer: CFGDenoiser vs SkipLayerCFGDenoiser
        # If you didn't import these into training file, just always use CFGDenoiser.
        denoiser = sd3_impls.CFGDenoiser if skip_layer_config.get("scale", 0) <= 0 else sd3_impls.SkipLayerCFGDenoiser

        if denoiser is sd3_impls.CFGDenoiser:
            latent = sample_fn(
                denoiser(self.sd3.model),
                noise_scaled,
                sigmas,
                extra_args=extra_args,
            )
        else:
            latent = sample_fn(
                denoiser(self.sd3.model, steps, skip_layer_config),
                noise_scaled,
                sigmas,
                extra_args=extra_args,
            )

        latent = self.latent_fmt.process_out(latent)
        return latent

    @torch.no_grad()
    def infer(
        self,
        prompts: List[str],
        controlnet_cond_image: Optional[str],
        init_image: Optional[str],
        out_dir: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 60,
        cfg_scale: float = 4.5, ###more aggresive like 4.5
        sampler: str = "euler",
        seed: int = 23,
        seed_type: str = "rand",   # "fixed" | "rand" | "roll"
        denoise: float = 1.0,
        skip_layer_config: dict = None,
        save_path: str = None,
    ):
        """
        Inference function to drop into training script.
        Matches sd3_infer.py behavior:
          - init_image optional (img2img-like)
          - controlnet_cond_image optional
          - neg prompt is ""
          - seed_type: fixed/rand/roll
          
          "steps": 60,
        "cfg": 3.5,
        "sampler": "euler" default
        """
        os.makedirs(out_dir, exist_ok=True)

        # Determine controlnet preprocessing flags
        using_2b, control_type = False, 0
        if self.sd3.model.control_model is not None:
            using_2b = not self.sd3.using_8b_controlnet
            control_type = int(self.sd3.model.control_model.control_type.item())

        # Prepare base latent (either from init_image or pure noise)
        if init_image is not None and init_image != "":
            latent = self._image_to_latent(init_image, width, height, using_2b_controlnet=False, controlnet_type=0)
        else:
            latent = self._get_empty_latent(1, width, height, seed, device="cuda")

        # Prepare control latent if provided
        control_lat = None
        if controlnet_cond_image is not None and controlnet_cond_image != "":
            control_lat = self._image_to_latent(
                controlnet_cond_image, width, height,
                using_2b_controlnet=using_2b, controlnet_type=control_type
            )

        neg_cond = self.encode_prompt("")  # returns (c, y) on CPU float32
        # IMPORTANT: encode_prompt returns CPU tensors; _fix_cond will move+half them

        seed_num = None
        for i, prompt in tqdm(list(enumerate(prompts)), total=len(prompts)):
            if seed_type == "roll":
                seed_num = seed if seed_num is None else seed_num + 1
            elif seed_type == "rand":
                seed_num = torch.randint(0, 100000, (1,)).item()
            else:
                seed_num = seed

            conditioning = self.encode_prompt(prompt)  # (c,y) cpu float32
            sampled_latent = self._do_sampling(
                latent=latent,
                seed=seed_num,
                conditioning=conditioning,
                neg_cond=neg_cond,
                steps=steps,
                cfg_scale=cfg_scale,
                sampler=sampler,
                controlnet_cond=control_lat,
                denoise=denoise if (init_image is not None and init_image != "") else 1.0,
                skip_layer_config=skip_layer_config or {},
            )

            img = self._vae_decode(sampled_latent)
            if save_path is None:
                save_path = os.path.join(out_dir, f"{i:06d}_test1_cutoff300sampling_denoise0.9.png")
            else:
                save_path = os.path.join(out_dir, save_path)
            img.save(save_path)


def main(**kwargs):
    cfg = TrainConfig(**kwargs)
    trainer = Trainer(cfg)
#     trainer._load("/scratch/liyues_root/liyues/shared_data/bowenbw/sd3.5/outputs/controlnet_lora/controlnet_lora_step500_20260124_235259.pt") ###with ours lora checkpoint
#     trainer._load("/scratch/liyues_root/liyues/shared_data/bowenbw/sd3.5/outputs/controlnet_lora/controlnet_lora_step600_20260127_010850_large.pt")
    
#     trainer._load("/scratch/liyues_root/liyues/shared_data/bowenbw/sd3.5/outputs/controlnet_lora_step1000_20260127_032137_large.pt")
#     trainer._load("outputs/controlnet_lora/controlnet_lora_final_step2000.pt")
    
#     trainer._load("outputs/controlnet_lora/controlnet_lora_step400_20260127_061049_large_promptflip.pt")

#     trainer._load("outputs/controlnet_lora/controlnet_lora_step2000_20260128_053556_large_promptflip.pt")
    
    
    
#     trainer._load("outputs/controlnet_lora/controlnet_lora_step1400_20260128_002718_large_promptflip.pt")
    
    trainer._load("outputs/controlnet_lora/controlnet_lora_step2000_20260127_074346_large.pt")
    
    ##############Testing########################
#     trainer.infer(prompts = ["A brown teddy bear [[314,128,931,858]] sits next to a blue latex glove [[000,295,422,887]]."], controlnet_cond_image = "/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_medium_images/000000_lq.png", init_image = "", out_dir = "recon")
    
#     trainer.infer(prompts = ["A man [[280,000,633,924]] in a white shirt [[300,071,647,400]] and blue shorts [[334,372,589,593]] is running on a track [[000,800,995,995]]."], controlnet_cond_image = "/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/bowen_im/IMG_6369.jpeg", init_image = "/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/bowen_im/IMG_6369.jpeg", out_dir = "recon_ablation", save_path = f"bowenrunning_cutoff600_denoise0.9.png", denoise = 0.90)
    
    
    
#     trainer.infer(prompts = ["A couch [[000,170,998,872]] with two pillows [[074,182,506,515;498,166,998,518]] is in the foreground, with a white plate [[345,840,692,998]] on a table [[003,846,998,998]] in the background."], controlnet_cond_image = "/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images/000_lq.png", init_image = "", out_dir = "recon")
    
    ################################################
    
    for i in range(0, 50):
#         prompt_file = f"/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images/{str(i).zfill(3)}_lq.txt"
        prompt_file = f"/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_prompts_test_DIV2K/{str(i).zfill(5)}.txt"###DIV2K
#         prompt_file = f"/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_prompts_test_CelebA/{str(i).zfill(5)}.txt" ###CelebA 
#         prompt_file = f"/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/test_VideoLQ_prompts/{str(i).zfill(5)}.txt" ###VideoLQ
#         prompt_file = f"/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/lumi_test/prompts/{str(i).zfill(5)}.txt"
        with open(prompt_file, "r", encoding="utf-8") as f:
            prompt = f.readlines()[0].strip()

        if not prompt:
            raise ValueError(f"Empty prompt in {prompt_file}")
        PROMPT = "a high-resolution and sharp image, Cinematic, hyper sharpness, highly detailed, perfect without deformations, hyper detailed photo - realistic maximum detail"    
            
            
        ###########ControlNet#################
        lq_path = f"/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/test_DIV2K/{str(i).zfill(5)}.png" ###DIV2K
        
#         lq_path = f"/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images_test_CelebA/{str(i).zfill(5)}.png"  ###CelebA
#         lq_path = f"/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/test_VideoLQ/{str(i).zfill(5)}.png"
#         lq_path = f"/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/lumi_test/lq/{str(i).zfill(5)}.png"
        ###########Ours###################
        
        for seed in range(20):
            trainer.infer(
                prompts=[f"{prompt}, {DEFAULT_PROMPT}"],
    #             prompts = [prompt],
    #             prompts = [DEFAULT_PROMPT],
    #             controlnet_cond_image=f"/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images_test_severe/{str(i).zfill(3)}.png",
    #             init_image=f"/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images_test_severe/{str(i).zfill(3)}.png",
                controlnet_cond_image = lq_path,
                init_image = lq_path,
                denoise=0.98,
    #             out_dir = "recon_ours_videolq_hq",
                out_dir="recon_ours_div2k_ablation",
    #             out_dir="recon_large_lumi",
                save_path = f"recon_{str(i).zfill(5)}_seed{seed}.png"
            )
        
        ############Ours#####################

#         trainer.infer(
#             prompts=[prompt],
#             controlnet_cond_image=f"/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images/{str(i).zfill(3)}_lq.png",
#             init_image=f"/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images/{str(i).zfill(3)}_lq.png",
#             denoise=0.95,
#             out_dir="recon_all",
#             save_path = f"recon_{str(i).zfill(3)}_denoise0.95_cutoff450.png"
#         )
        ##############################################
        
#         lq_path = f"/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/test_DIV2K/{str(i).zfill(5)}.png"
# #         lq_path = f"/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/lumi_test/lq/{str(i).zfill(5)}.png"
        
#         ###########Ours###################
#         trainer.infer(
#             prompts=[f"{prompt}, {DEFAULT_PROMPT}"],
# #             controlnet_cond_image=f"/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images_test_severe/{str(i).zfill(3)}.png",
# #             init_image=f"/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images_test_severe/{str(i).zfill(3)}.png",
#             controlnet_cond_image = lq_path,
#             init_image = lq_path,
#             denoise=1.0,
#             out_dir="recon_large_div2k",
# #             out_dir="recon_large_lumi",
#             save_path = f"recon_{str(i).zfill(5)}.png"
#         )
        
        
#         trainer.infer(
#             prompts=[prompt],
#             controlnet_cond_image=f"/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images/{str(i).zfill(3)}_lq.png",
#             init_image=f"/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images/{str(i).zfill(3)}_lq.png",
#             denoise=1.0,
#             out_dir="recon_large_test",
#             save_path = f"recon_{str(i).zfill(3)}_denoise1.0_cutoff0_cfg35_wtprompt_600iter.png"
#         )
            
            
#         trainer.infer(
#             prompts=[prompt],
#             controlnet_cond_image=f"/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images/{str(i).zfill(3)}_lq.png",
#             init_image=f"/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images/{str(i).zfill(3)}_lq.png",
#             denoise=1.0,
#             out_dir="recon_126best",
#             save_path = f"recon_{str(i).zfill(3)}_denoise1.0_cutoff300_cfg45.png"
#         )
        
        
#         trainer.infer(
#             prompts=[prompt],
#             controlnet_cond_image=f"/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images/{str(i).zfill(3)}_lq.png",
#             init_image=f"/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images/{str(i).zfill(3)}_lq.png",
#             denoise=0.80,
#             out_dir="recon_lora",
#             save_path = f"recon_{str(i).zfill(3)}_denoise0.85_cutoff300_cfg45.png"
#         )
        
#         trainer.infer(
#             prompts=[prompt],
#             controlnet_cond_image=f"/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images/{str(i).zfill(3)}_lq.png",
#             init_image=f"/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images/{str(i).zfill(3)}_lq.png",
#             denoise=0.90,
#             out_dir="recon_lora",
#             save_path = f"recon_{str(i).zfill(3)}_denoise0.85_cutoff300_cfg45.png"
#         )
        ######################
        
        
        ###############ControlNet##################
#         trainer.infer(
#             prompts=[prompt],
#             controlnet_cond_image=f"/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images/{str(i).zfill(3)}_lq.png",
#             init_image=f"/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images/{str(i).zfill(3)}_lq.png",
#             denoise=1.0,
#             out_dir="recon_sd3controlnet",
#             save_path = f"recon_{str(i).zfill(3)}.png"
#         )
        ####################################
            
        
        
        
        
        
        
    
#     for si in range(20, 30):
    
#         trainer.infer(prompts = ["A television [[163,154,838,717]] is turned on and showing a horse [[232,225,598,598]] on the screen [[228,200,769,592]]."], controlnet_cond_image = "/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images/001_lq.png", init_image = "/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images/001_lq.png", denoise = 1.0, seed = si, out_dir = "recon_all", save_path = f"recon_1_seed{str(si)}_cutoff300_denoise1.0.png")
    
#     trainer.train()




if __name__ == "__main__":
    fire.Fire(main)
