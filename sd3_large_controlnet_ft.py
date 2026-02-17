#!/usr/bin/env python
# sd3_controlnet_full_finetune_with_ImageCaptionDataset.py
#
# Full fine-tune ControlNet parameters ONLY (no LoRA, no new layers).
# - Freeze: main diffusion_model, VAE, text encoders
# - Train: self.sd3.model.control_model (ALL params)
#
# Uses your ImageCaptionDataset exactly.

import os
import math
import time
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


################################################################################
# Your dataset (used as-is)
################################################################################

class ImageCaptionDataset(Dataset):
    """
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
# Trainer (full finetune controlnet weights only)
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

    out_dir: str = "outputs/controlnet_fullft"
    batch_size: int = 1
    num_workers: int = 2

    steps: int = 2000
    lr: float = 1e-5           # usually lower for full FT than LoRA
    weight_decay: float = 0.0
    grad_accum: int = 1
    grad_clip: float = 1.0

    log_every: int = 20
    save_every: int = 500

    # noise schedule sampling
    min_t: int = 1
    max_t: int = 1000

    # AMP
    amp: bool = True
    bf16: bool = False

    resume: Optional[str] = None


class Trainer:
    def __init__(self, cfg: TrainConfig):
        self.cfg = cfg
        os.makedirs(cfg.out_dir, exist_ok=True)

        # Tokenizers + encoders (always frozen)
        self.tokenizer = SD3Tokenizer()
        self.t5xxl = T5XXL(cfg.model_folder, "cpu", torch.float32)
        self.clip_l = ClipL(cfg.model_folder)
        self.clip_g = ClipG(cfg.model_folder, "cpu")

        # SD3 + ControlNet + VAE
        self.sd3 = SD3Bundle(cfg.model, cfg.controlnet_ckpt, cfg.shift, verbose=False, device="cuda")
        self.vae = VAE(cfg.vae or cfg.model, dtype=torch.float16)
        self.latent_fmt = SD3LatentFormat()

        assert self.sd3.model.control_model is not None, "control_model is None. Bad controlnet_ckpt?"

        # control preprocess flags (matching BaseModel.apply_model / inferencer)
        self.control_type = int(self.sd3.model.control_model.control_type.item())
        self.using_2b = not self.sd3.using_8b_controlnet

        # -------------------- Freeze main diffusion model --------------------
        for p in self.sd3.model.diffusion_model.parameters():
            p.requires_grad_(False)
        # --------------------------------------------------------------------

        # -------------------- Train ALL controlnet params --------------------
        for p in self.sd3.model.control_model.parameters():
            p.requires_grad_(True)
        # --------------------------------------------------------------------

        # Optimizer over ONLY control_model params
        self.opt = torch.optim.AdamW(
            [p for p in self.sd3.model.control_model.parameters() if p.requires_grad],
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

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
        # gt_tensor_m11: (B,3,H,W) in [-1,1] float32 from dataset
        self.vae.model = self.vae.model.cuda()
        lat = self.vae.model.encode(gt_tensor_m11.to("cuda", dtype=torch.float32))
        self.vae.model = self.vae.model.cpu()
        return self.latent_fmt.process_in(lat)

    @torch.no_grad()
    def vae_encode_control_from_paths(self, paths: List[str]) -> torch.Tensor:
        ims = [Image.open(p).convert("RGB").resize((1024, 1024), Image.LANCZOS) for p in paths]
        t = torch.stack(
            [torch.from_numpy(np.moveaxis(np.array(im).astype(np.float32) / 255.0, 2, 0)) for im in ims],
            dim=0
        ).to("cuda", dtype=torch.float32)

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
        return self.latent_fmt.process_in(lat)

    def sample_sigmas(self, bsz: int):
        t = torch.randint(self.cfg.min_t, self.cfg.max_t + 1, (bsz,), device="cuda", dtype=torch.int64)
        return self.sd3.model.model_sampling.sigma(t.float()).to(torch.float32)

    def _save(self, path: str):
        ckpt = {
            "step": self.global_step,
            "micro_step": self.micro_step,
            "opt": self.opt.state_dict(),
            # save FULL controlnet state
            "control_model": self.sd3.model.control_model.state_dict(),
            "cfg": vars(self.cfg),
        }
        torch.save(ckpt, path)

    def _load(self, path: str):
        ckpt = torch.load(path, map_location="cpu")
        self.global_step = int(ckpt.get("step", 0))
        self.micro_step = int(ckpt.get("micro_step", 0))
        if "opt" in ckpt:
            self.opt.load_state_dict(ckpt["opt"])
        if "control_model" in ckpt:
            self.sd3.model.control_model.load_state_dict(ckpt["control_model"], strict=False)
        print(f"[resume] {path} (step={self.global_step}, micro={self.micro_step})")

    def train(self):
        cfg = self.cfg
        dataset = ImageCaptionDataset(cfg.image_root, cfg.captions_file, cfg.lq_image_root, imgperprompt=cfg.imgperprompt)
        loader = DataLoader(dataset, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)

        autocast_dtype = torch.bfloat16 if cfg.bf16 else torch.float16
        autocast_enabled = cfg.amp or cfg.bf16

        self.opt.zero_grad(set_to_none=True)
        start = time.time()

        it = iter(loader)
        while self.global_step < cfg.steps:
            try:
                images_m11, lq_paths, captions = next(it)
            except StopIteration:
                it = iter(loader)
                images_m11, lq_paths, captions = next(it)

            self.micro_step += 1
            captions = list(captions)
            lq_paths = list(lq_paths)

            # Encode latents (no grad)
            with torch.no_grad():
                x0 = self.vae_encode_gt_tensor(images_m11)  # (B,16,H/8,W/8)
                control_lat = self.vae_encode_control_from_paths(lq_paths)

                # text cond
                c_list, y_list = [], []
                for p in captions:
                    c, y = self.encode_prompt(p)
                    c_list.append(c)
                    y_list.append(y)
                c_cross = torch.cat(c_list, dim=0).to("cuda", dtype=torch.float16)
                y = torch.cat(y_list, dim=0).to("cuda", dtype=torch.float16)

            sigma = self.sample_sigmas(x0.shape[0])  # (B,) float32
            noise = torch.randn_like(x0, dtype=torch.float32).to(x0.dtype)

            b = x0.shape[0]
            x = sigma.view(b, 1, 1, 1) * noise + (1.0 - sigma).view(b, 1, 1, 1) * x0

            # Forward trains ALL controlnet params
            with torch.cuda.amp.autocast(enabled=autocast_enabled, dtype=autocast_dtype):
                pred_x0 = self.sd3.model.apply_model(
                    x, sigma,
                    c_crossattn=c_cross,
                    y=y,
                    controlnet_cond=control_lat,
                )
                loss = F.mse_loss(pred_x0, x0)
                loss_scaled = loss / max(1, cfg.grad_accum)

            if self.scaler.is_enabled():
                self.scaler.scale(loss_scaled).backward()
            else:
                loss_scaled.backward()

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
                    path = os.path.join(cfg.out_dir, f"controlnet_fullft_step{self.global_step}_{stamp}.pt")
                    print(f"[save] {path}")
                    self._save(path)

        final_path = os.path.join(cfg.out_dir, f"controlnet_fullft_final_step{self.global_step}.pt")
        print(f"[save final] {final_path}")
        self._save(final_path)


def main(**kwargs):
    cfg = TrainConfig(**kwargs)
    trainer = Trainer(cfg)
    trainer.train()


if __name__ == "__main__":
    fire.Fire(main)
