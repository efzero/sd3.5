import os

os.environ["HF_HOME"] = "/scratch/liyues_root/liyues/shared_data/bowenbw/hf"
# optional overrides
os.environ["HF_HUB_CACHE"] = "/scratch/liyues_root/liyues/shared_data/bowenbw/hf/hub"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/liyues_root/liyues/shared_data/bowenbw/hf/transformers"
os.environ["DIFFUSERS_CACHE"] = "/scratch/liyues_root/liyues/shared_data/bowenbw/hf/diffusers"
os.environ["HF_DATASETS_CACHE"] = "/scratch/liyues_root/liyues/shared_data/bowenbw/hf/datasets"

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import fire
import numpy as np
import torch
from PIL import Image
from glob import glob
from safetensors import safe_open
from torch import nn
from torch.utils.data import Dataset, DataLoader
from transformers import CLIPTextModelWithProjection, CLIPVisionModelWithProjection, CLIPTokenizer, T5EncoderModel, T5TokenizerFast, CLIPImageProcessor
from other_impls import SDXLClipG, SDClipModel, T5XXLModel, SD3Tokenizer
from sd3_impls import BaseModel, SDVAE, SD3LatentFormat, append_dims
import torch.nn.functional as F
import random
import wandb
import torch.nn as nn
from einops import rearrange
from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3


wandb.init(project="sd3-train", name="run-115", config={"lr": 1e-4, "bs": 4})


###################################################################################################
### LoRA Implementation
###################################################################################################


class ClipVisionAdaptor(nn.Module):
    """
    Map CLIP-L/G vision token sequences -> SD3 cross-attn CLIP context: [B, 77, 4096]
    Assumes we take first (77*3)=231 tokens and fold them into 77 tokens.
    """
    def __init__(self, dim_l: int, dim_g: int, out_dim: int = 4096, t: int = 77, k: int = 3):
        super().__init__()
        self.t = t
        self.k = k
        in_dim = (k * dim_l) + (k * dim_g)

        # simple MLP adaptor (works well as a starting point)
        self.net = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Linear(in_dim, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, l_tokens: torch.Tensor, g_tokens: torch.Tensor) -> torch.Tensor:
        # l_tokens: [B, seq_l, dim_l], g_tokens: [B, seq_g, dim_g]
        B = l_tokens.shape[0]
        need = self.t * self.k  # 231

        l_crop = l_tokens[:, :need, :]  # [B,231,dim_l]
        g_crop = g_tokens[:, :need, :]  # [B,231,dim_g]

        l_fold = rearrange(l_crop, "b (t k) d -> b t (k d)", t=self.t, k=self.k)  # [B,77,3*dim_l]
        g_fold = rearrange(g_crop, "b (t k) d -> b t (k d)", t=self.t, k=self.k)  # [B,77,3*dim_g]

        x = torch.cat([l_fold, g_fold], dim=-1)  # [B,77, 3*dim_l + 3*dim_g]
        return self.net(x)  # [B,77,4096]
    

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer for fine-tuning."""
    def __init__(self, in_features: int, out_features: int, rank: int = 32, alpha: float = 1.0, dropout: float = 0.0, device=None, dtype=None):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        
#         self.lora_A = nn.Parameter(torch.randn(rank, in_features, device=device, dtype=dtype) * 0.02)
#         self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=dtype))
        
        self.lora_A = nn.Parameter(torch.empty(rank, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank, device=device, dtype=dtype))
        
        # Initialize lora_A with kaiming uniform (smaller scale)
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., in_features]
        # LoRA: x @ (A^T @ B^T) = x @ A^T @ B^T
        lora_out = self.dropout(x) @ self.lora_A.T @ self.lora_B.T
        return lora_out * self.scale


class LoRALinear(nn.Module):
    """Wraps a Linear layer with LoRA."""
    def __init__(self, linear: nn.Linear, rank: int = 4, alpha: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.linear = linear
        device = linear.weight.device
        dtype = linear.weight.dtype
        
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha, dropout, device = device, dtype = dtype)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x) + self.lora(x)


def apply_lora_to_linear(parent: nn.Module, attr_name: str, rank: int = 32, alpha: float = 1.0, dropout: float = 0.0) -> bool:
    """Replace a Linear layer with LoRALinear. Returns True if replaced."""
    if not hasattr(parent, attr_name):
        return False
    
    child = getattr(parent, attr_name)
    if isinstance(child, nn.Linear):
        # Replace with LoRA version
        lora_linear = LoRALinear(child, rank, alpha, dropout)
        setattr(parent, attr_name, lora_linear)
        return True
    return False


def apply_lora_to_model(
    model: nn.Module,
    target_modules: Optional[List[str]] = None,
    rank: int = 32,
    alpha: float = 1.0,
    dropout: float = 0.0,
    prefix: str = "",
) -> List[str]:
    """
    Apply LoRA to Linear layers in the model.
    
    Args:
        model: The model to apply LoRA to
        target_modules: List of module name patterns to target (e.g., ['qkv', 'proj', 'mlp'])
                       If None, applies to all Linear layers
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        dropout: LoRA dropout
        prefix: Prefix for logging
        
    Returns:
        List of names of modules that were converted to LoRA
    """
    if target_modules is None:
        # Default: target attention and MLP layers
        target_modules = ['qkv', 'proj', 'w1', 'w2', 'w3', 'adaLN_modulation', 'linear', 'context_embedder']
    
    converted = []
    
    def _apply_recursive(module, name_prefix=""):
        for name, child in list(module.named_children()):
            full_name = f"{name_prefix}.{name}" if name_prefix else name
            full_name = f"{prefix}.{full_name}" if prefix else full_name
            
            # Check if this module name matches any target pattern
            should_apply = any(pattern in name for pattern in target_modules)
            
            if should_apply and isinstance(child, nn.Linear):
                if apply_lora_to_linear(module, name, rank, alpha, dropout):
                    converted.append(full_name)
            else:
                _apply_recursive(child, full_name)
    
    _apply_recursive(model)
    return converted


###################################################################################################
### Custom CLIP Embedding
###################################################################################################


class CustomCLIPEmbedding(nn.Module):
    """
    Custom CLIP embedding that can learn a different embedding space.
    Can be initialized from original CLIP embeddings or learned from scratch.
    """
    def __init__(
        self,
        original_embed_dim: int = 4096,  # CLIP-L + CLIP-G concatenated + padded
        custom_embed_dim: Optional[int] = None,
        projection: bool = True,
        learnable: bool = True,
    ):
        super().__init__()
        self.original_embed_dim = original_embed_dim
        self.custom_embed_dim = custom_embed_dim or original_embed_dim
        self.projection = projection
        
        if projection:
            # Learn a projection from original CLIP space to custom space
            self.proj = nn.Linear(original_embed_dim, self.custom_embed_dim, bias=True)
            if learnable:
                # Initialize to identity-like transformation
                nn.init.eye_(self.proj.weight)
                nn.init.zeros_(self.proj.bias)
        else:
            # Direct embedding table (for token-based embeddings)
            # This would require knowing the vocabulary size, so we'll use projection by default
            self.proj = None
    
    def forward(self, original_embedding: torch.Tensor) -> torch.Tensor:
        """
        Transform original CLIP embedding to custom embedding.
        
        Args:
            original_embedding: [B, L, D] or [B, D] tensor from original CLIP
            
        Returns:
            [B, L, D'] or [B, D'] tensor in custom embedding space
        """
        if self.projection and self.proj is not None:
            return self.proj(original_embedding)
        return original_embedding


###################################################################################################
### Shared loading utilities (mirrors sd3_infer.py)
###################################################################################################


def load_into(f, model, prefix, device, dtype=None):
    """Apply weights from a safetensors file to a torch module (debug‑friendly, same as sd3_infer)."""
    for key in f.keys():
        if key.startswith(prefix) and not key.startswith("loss."):
            path = key[len(prefix):].split(".")
            obj = model
            for p in path:
                if obj is list:
                    obj = obj[int(p)]
                else:
                    obj = getattr(obj, p, None)
                    if obj is None:
                        print(f"Skipping key '{key}' in safetensors file as '{p}' does not exist in python model")
                        break
            if obj is None:
                continue
            try:
                tensor = f.get_tensor(key).to(device=device)
                if dtype is not None:
                    tensor = tensor.to(dtype=dtype)
                obj.requires_grad_(False)
                obj.set_(tensor)
            except Exception as e:
                print(f"Failed to load key '{key}' in safetensors file: {e}")
                raise e


CLIPG_CONFIG = {
    "hidden_act": "gelu",
    "hidden_size": 1280,
    "intermediate_size": 5120,
    "num_attention_heads": 20,
    "num_hidden_layers": 32,
}

CLIPL_CONFIG = {
    "hidden_act": "quick_gelu",
    "hidden_size": 768,
    "intermediate_size": 3072,
    "num_attention_heads": 12,
    "num_hidden_layers": 12,
}

T5_CONFIG = {
    "d_ff": 10240,
    "d_model": 4096,
    "num_heads": 64,
    "num_layers": 24,
    "vocab_size": 32128,
}


class ClipG:
    def __init__(self, device: str = "cuda", dtype=torch.float32):
        with safe_open("models/clip_g.safetensors", framework="pt", device="cpu") as f:
            self.model = SDXLClipG(CLIPG_CONFIG, device=device, dtype=dtype)
            load_into(f, self.model.transformer, "", device, dtype)
        for p in self.model.parameters():
            p.requires_grad_(False)


class ClipL:
    def __init__(self, device: str = "cuda", dtype=torch.float32):
        with safe_open("models/clip_l.safetensors", framework="pt", device="cpu") as f:
            self.model = SDClipModel(
                layer="hidden",
                layer_idx=-2,
                device=device,
                dtype=dtype,
                layer_norm_hidden_state=False,
                return_projected_pooled=False,
                textmodel_json_config=CLIPL_CONFIG,
            )
            load_into(f, self.model.transformer, "", device, dtype)
        for p in self.model.parameters():
            p.requires_grad_(False)


class T5XXL:
    def __init__(self, device: str = "cuda", dtype=torch.float32):
        with safe_open("models/t5xxl_fp16.safetensors", framework="pt", device="cpu") as f:
            self.model = T5XXLModel(T5_CONFIG, device=device, dtype=dtype)
            load_into(f, self.model.transformer, "", device, dtype)
        for p in self.model.parameters():
            p.requires_grad_(False)


class SD3:
    def __init__(self, model_path: str, shift: float, device: str = "cuda", dtype: torch.dtype = torch.float32):
        with safe_open(model_path, framework="pt", device="cpu") as f:
            # BaseModel will construct diffusion_model + sampling utilities
            self.model = BaseModel(shift=shift, file=f, prefix="model.diffusion_model.", device=device, dtype=dtype)
            load_into(f, self.model, "model.", device, dtype)


class VAE:
    def __init__(self, model_path: str, device: str = "cuda", dtype: torch.dtype = torch.float16):
        with safe_open(model_path, framework="pt", device="cpu") as f:
            self.model = SDVAE(device=device, dtype=dtype).eval()
            prefix = ""
            if any(k.startswith("first_stage_model.") for k in f.keys()):
                prefix = "first_stage_model."
            load_into(f, self.model, prefix, device, dtype)
        for p in self.model.parameters():
            p.requires_grad_(False)


###################################################################################################
### Simple image+caption dataset
###################################################################################################


class ImageCaptionDataset(Dataset):
    """
    Very small reference dataset:
      - image_root/: image files
      - captions_file: text file of "filename<TAB>caption" per line
    """

    def __init__(self, image_root: str, captions_file: str, lq_image_root: str, imgperprompt = 10):
        self.image_root = image_root
        self.items: List[str] = []
        self.lq_image_root = lq_image_root
        self.imgperprompt = imgperprompt
            
            
        self.img_items: List[str] = []
#         with open(captions_file, "r", encoding="utf-8") as f:
#             for line in f:
#                 line = line.strip()
#                 self.items.append(line)
            
        img_files = glob(f'{image_root}/*.png')
        img_files.sort()
        
        
        lq_img_files = glob(f'{lq_image_root}/*.png')
        lq_img_files.sort()
        
        caption_files = glob(f'{lq_image_root}/*.txt')
        caption_files.sort()
        
        self.img_items = img_files
        self.items = caption_files
        self.lqimg_items = lq_img_files
        
        
        
#                 if not line:
#                     continue
#                 parts = line.split("\t", 1)
#                 if len(parts) != 2:
#                     continue
#                 self.items.append((parts[0], parts[1]))

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


###################################################################################################
### Training logic
###################################################################################################


@dataclass
class TrainConfig:
#     model: str = "models/sd3_medium.safetensors"
    model: str = "models/sd3.5_medium.safetensors"
    vae: Optional[str] = None
    shift: float = 3.0
    width: int = 1024
    height: int = 1024
    batch_size: int = 2
    epochs: int = 10
    lr: float = 1e-4
    ema_decay: float = 0.0  # 0 disables EMA
    grad_accum_steps: int = 1
    max_steps: Optional[int] = None
    log_every: int = 50
    save_every: int = 1000
    out_dir: str = "checkpoints"
    # LoRA settings
    use_lora: bool = True
    lora_rank: int = 32
    lora_alpha: float = 1.0
    lora_dropout: float = 0.0
    lora_target_modules: Optional[List[str]] = None
    # Custom CLIP embedding settings
    use_custom_clip: bool = False
    custom_clip_dim: Optional[int] = None
    custom_clip_projection: bool = True
    # Resume training
    resume_from: Optional[str] = None
    sample_every: int = 50  # Generate sample images every N steps
    samples_dir: str = "samples"  # Directory to save sample images
    sample_prompt: str = "a photo of a cat"  # Prompt for sampling
    sample_steps: int = 28  # Sampling steps (fewer for faster sampling)
    sample_cfg_scale: float = 5.0  # CFG scale for sampling
    sample_seed: int = 42  # Seed for sampling


class SD3Trainer:
    def __init__(self, config: TrainConfig, device: str = "cuda"):
        self.config = config
        self.device = device if torch.cuda.is_available() else "cpu"
        self.custom_clip_embedding = None
        self.lora_modules = []

    def load_models(self):
        print("Loading tokenizers...")
        self.tokenizer = SD3Tokenizer()

        print("Loading OpenCLIP bigG...")
        self.clip_g = ClipG(device=self.device)

        print("Loading OpenAI CLIP L...")
        self.clip_l = ClipL(device=self.device)

        print("Loading Google T5-v1-XXL...")
        self.t5xxl = T5XXL(device=self.device)

        print("Loading SD3 diffusion model...")
        self.sd3 = SD3(self.config.model, self.config.shift, device=self.device)
        self.sd3.model.to(self.device)
        
        
        self.dtype = torch.float32  # or torch.float16 / torch.bfloat16
        print("Loading CLIP Image models")
        self.clip_l_repo = "openai/clip-vit-large-patch14"
        self.clip_g_repo = "laion/CLIP-ViT-bigG-14-laion2B-39B-b160k"
        self.clip_l_image_processor = CLIPImageProcessor.from_pretrained(self.clip_l_repo)
        self.clip_g_image_processor = CLIPImageProcessor.from_pretrained(self.clip_g_repo)

        self.clip_l_image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.clip_l_repo, torch_dtype=self.dtype).to(self.device).eval()
        self.clip_g_image_encoder = CLIPVisionModelWithProjection.from_pretrained(self.clip_g_repo,torch_dtype=self.dtype).to(self.device).eval()
        print("Finish Loading CLIP image models")
        
        for p in self.clip_l_image_encoder.parameters():
            p.requires_grad_(False)
        for p in self.clip_g_image_encoder.parameters():
            p.requires_grad_(False)

        # Trainable adaptor (dims come from the vision models)
        dim_l = self.clip_l_image_encoder.config.hidden_size
        dim_g = self.clip_g_image_encoder.config.hidden_size

        self.clip_vision_adaptor = ClipVisionAdaptor(dim_l=dim_l, dim_g=dim_g, out_dim=4096).to(self.device)
        for p in self.clip_vision_adaptor.parameters():
            p.requires_grad_(True)

        # Freeze base model parameters
        for p in self.sd3.model.parameters():
            p.requires_grad_(False)

        # Apply LoRA if enabled
        if self.config.use_lora:
            print(f"Applying LoRA (rank={self.config.lora_rank}, alpha={self.config.lora_alpha})...")
            self.lora_modules = apply_lora_to_model(
                self.sd3.model.diffusion_model,
                target_modules=self.config.lora_target_modules,
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
                dropout=self.config.lora_dropout,
            )
            print(f"Applied LoRA to {len(self.lora_modules)} modules: {self.lora_modules[:5]}...")
            
            # Enable gradients for LoRA parameters
            for name, param in self.sd3.model.named_parameters():
                if 'lora' in name.lower():
                    param.requires_grad_(True)

        # Setup custom CLIP embedding if enabled
        if self.config.use_custom_clip:
            print("Setting up custom CLIP embedding...")
            # Original CLIP embedding dimension: CLIP-L (768) + CLIP-G (1280) = 2048, padded to 4096, then T5 added
            # For cross-attention, we use the concatenated CLIP-L+G (padded to 4096) + T5
            # The context_embedder in MMDiT projects this to hidden_size
            # Note: If custom_clip_dim != 4096, you may need to adjust the context_embedder in MMDiT
            # For now, we keep the same dimension by default to avoid this complexity
            original_dim = 4096  # CLIP-L+G padded dimension
            custom_dim = self.config.custom_clip_dim or original_dim
            self.custom_clip_embedding = CustomCLIPEmbedding(
                original_embed_dim=original_dim,
                custom_embed_dim=custom_dim,
                projection=self.config.custom_clip_projection,
                learnable=True,
            ).to(self.device)
            print(f"Custom CLIP embedding: {original_dim} -> {self.custom_clip_embedding.custom_embed_dim}")

        print("Loading VAE...")
        vae_path = self.config.vae or self.config.model
        self.vae = VAE(vae_path, device=self.device)
        self.latent_format = SD3LatentFormat()

        self.sd3.model.train()

    
    def get_cond_image(self, image_paths, prompts, device="cuda"):
        """
        image_paths: list[str] or str
        prompts: list[str] or str

        Returns:
          prompt_embeds: [B, 77+256, 4096]
          pooled_embeds: [B, pooled_dim]  (l_pooled||g_pooled)
        """
        
                # ---- normalize inputs to lists ----
        if isinstance(image_paths, str):
            image_paths = [image_paths]
        if isinstance(prompts, str):
            prompts = [prompts]

        n_img = len(image_paths)
        n_txt = len(prompts)

        # ---- broadcast rules ----
        if n_img != n_txt:
            if n_img == 1 and n_txt > 1:
                image_paths = image_paths * n_txt
            elif n_txt == 1 and n_img > 1:
                prompts = prompts * n_img
            else:
                raise ValueError(f"Length mismatch: {n_img} image_paths vs {n_txt} prompts")

        B = len(image_paths)
        device = torch.device(device)

        # ============================================================
        # 1) IMAGE EMBEDDINGS (batched)
        # ============================================================
        images = []
        for p in image_paths:
            im = Image.open(p).convert("RGB")
            images.append(im)

        # preprocess in batch -> pixel_values: [B,3,H,W]
        clip_img_l = self.clip_l_image_processor(images=images, return_tensors="pt").pixel_values.to(device)
        clip_img_g = self.clip_g_image_processor(images=images, return_tensors="pt").pixel_values.to(device)
        
        
        with torch.no_grad():
            out_l = self.clip_l_image_encoder(pixel_values=clip_img_l)
            out_g = self.clip_g_image_encoder(pixel_values=clip_img_g)

        l_tokens = out_l.last_hidden_state   # [B, seq_l, dim_l]
        g_tokens = out_g.last_hidden_state   # [B, seq_g, dim_g]
        l_pooled = out_l.image_embeds        # [B, pooled_l]
        g_pooled = out_g.image_embeds        # [B, pooled_g]

        # TRAINABLE adaptor (do NOT no_grad this)
        lg_out = self.clip_vision_adaptor(l_tokens, g_tokens)  # [B,77,4096]

        pooled_embeds = torch.cat([l_pooled, g_pooled], dim=-1)  # keep frozen

        # t5 part (frozen)
        t5_out_list = []
        for prompt in prompts:
            tokens = self.tokenizer.tokenize_with_weights(prompt)
            with torch.no_grad():
                t5_out, _ = self.t5xxl.model.encode_token_weights(tokens["t5xxl"])
            t5_out_list.append(t5_out)
        t5_out = torch.cat(t5_out_list, dim=0).to(device)  # [B,256,4096]

        prompt_embeds = torch.cat([lg_out, t5_out], dim=-2)  # [B,333,4096]

        # IMPORTANT: do NOT detach prompt_embeds, or adaptor won’t learn
        return prompt_embeds.to(self.device), pooled_embeds.to(self.device).detach()
        
        
    
    
    def encode_prompts(self, prompts: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Batch encode prompts -> (c_crossattn, pooled_y)
        Mirrors sd3_infer.get_cond but batched.
        If custom_clip_embedding is enabled, applies transformation to CLIP embeddings.
        """
        cond_list = []
        pooled_list = []
        for text in prompts:
            tokens = self.tokenizer.tokenize_with_weights(text)
            l_out, l_pooled = self.clip_l.model.encode_token_weights(tokens["l"])
            g_out, g_pooled = self.clip_g.model.encode_token_weights(tokens["g"])
            t5_out, _ = self.t5xxl.model.encode_token_weights(tokens["t5xxl"])
            lg_out = torch.cat([l_out, g_out], dim=-1)
            lg_out = torch.nn.functional.pad(lg_out, (0, 4096 - lg_out.shape[-1]))
            
            # Apply custom CLIP embedding transformation if enabled
            if self.custom_clip_embedding is not None:
                # Transform the CLIP-L+G part (first 4096 dims)
                lg_out_transformed = self.custom_clip_embedding(lg_out)
                # Keep T5 as is, or you could also transform it
                # For now, we'll keep T5 unchanged and only transform CLIP
                # Note: This assumes the context_embedder can handle the new dimension
                # If custom_clip_dim != 4096, you may need to adjust the context_embedder
                cond = torch.cat([lg_out_transformed, t5_out], dim=-2)
            else:
                cond = torch.cat([lg_out, t5_out], dim=-2)
            
            pooled = torch.cat((l_pooled, g_pooled), dim=-1)
            cond, pooled = cond.detach(), pooled.detach()
            cond_list.append(cond)
            pooled_list.append(pooled)
        cond = torch.cat(cond_list, dim=0).to(self.device)
        pooled = torch.cat(pooled_list, dim=0).to(self.device)
        return cond, pooled

    
    
    def sample_sigmas(self, batch_size: int) -> torch.Tensor:
        
        """
        from diffusers.training_utils import compute_density_for_timestep_sampling, compute_loss_weighting_for_sd3
        """
        
#         #######uniform samples
#         sigmas_buffer = self.sd3.model.model_sampling.sigmas.to(self.device)
#         idx = torch.randint(0, sigmas_buffer.shape[0], (batch_size,), device=self.device)
#         sigmas = sigmas_buffer[idx]
#         return sigmas
    
    
        #########lognormal samples
        sigmas_buffer = self.sd3.model.model_sampling.sigmas.to(self.device)
        u = compute_density_for_timestep_sampling( 
                weighting_scheme='logit_normal', 
                batch_size=batch_size, 
                logit_mean= 0.0, 
                logit_std= 1.0, mode_scale=1.29,)
        indices = torch.clip((u * 1000).long(), 0, 999)
        sigmas = sigmas_buffer[indices]
        return sigmas

    def vae_encode(self, image) -> torch.Tensor:
        print("Encoding image to latent...")
        image = image.convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        image_np = np.moveaxis(image_np, 2, 0)
        batch_images = np.expand_dims(image_np, axis=0).repeat(1, axis=0)
        image_torch = torch.from_numpy(batch_images)
        image_torch = 2.0 * image_torch - 1.0
        image_torch = image_torch.cuda()
        self.vae.model = self.vae.model.cuda()
        latent = self.vae.model.encode(image_torch).cpu()
#         self.vae.model = self.vae.model.cpu()
        print("Encoded")
        return latent, image_torch
    
    
    def forward_loss(self, images: torch.Tensor, captions: List[str], lq_images = None) -> torch.Tensor:
        # images: [B, 3, H, W] in [-1, 1]
        images = images.to(self.device)

        with torch.no_grad():
            self.vae.model.eval()
            self.vae.model = self.vae.model.cuda()
            latents = self.vae.model.encode(images)  # [B, 16, H/8, W/8]
            latents = self.latent_format.process_in(latents)
#             self.vae.model = self.vae.model.cpu()
            

        batch_size = latents.shape[0]
        sigmas = self.sample_sigmas(batch_size)  # [B]
        b = batch_size
        
#         print(sigmas, "sampled sigmas")

        noise = torch.randn_like(latents)
#         print(latents.shape, images.shape, noise.shape, "latents images noise")
#         x_t = sigma * noise + (1-sigma)*x0, same as ModelSamplingDiscreteFlow.noise_scaling
#         x_t = self.sd3.model.model_sampling.noise_scaling(sigmas, noise, latents)

        weighting = compute_loss_weighting_for_sd3(weighting_scheme="logit_normal", sigmas=sigmas)
    
        print(sigmas, "sigmas", weighting,"weighting")
    
        x_t = sigmas.view(b,1, 1, 1) * noise + (1.0 - sigmas).view(b,1,1,1) * latents
        if lq_images is None:
            cond_c, cond_y = self.encode_prompts(captions)
        else:
            print(lq_images, "lq images")
            cond_c, cond_y = self.get_cond_image(lq_images, captions)
            

        # Ensure sigmas has right shape for apply_model
        sigmas_input = sigmas.view(batch_size)

        pred = self.sd3.model.apply_model(
            x_t,
            sigmas_input,
            c_crossattn=cond_c,
            y=cond_y,
        )
        # Train in x0‑space: predict clean latent
#         loss = nn.functional.mse_loss(pred.float(), latents.float())
        
        loss = torch.mean(
            (weighting.float().view(b,1,1,1) * (pred.float() - latents.float()) ** 2).reshape(latents.shape[0], -1),1,)
        loss = loss.mean()
        return loss

    
    
    

    
    def make_optimizer(self):
        # Collect all trainable parameters (LoRA + custom CLIP embedding)
        params = [p for p in self.sd3.model.parameters() if p.requires_grad] ####LORA only
        
#         params = [] ####no LORA
        
        if getattr(self, "clip_vision_adaptor", None) is not None:
            params += [p for p in self.clip_vision_adaptor.parameters() if p.requires_grad] ### + adaptor
            
        if self.custom_clip_embedding is not None:
            params.extend([p for p in self.custom_clip_embedding.parameters() if p.requires_grad]) ### + custom clip
        
        if len(params) == 0:
            raise ValueError("No trainable parameters found! Check LoRA and custom CLIP settings.")
        
        print(f"Training {sum(p.numel() for p in params)} parameters")
        print(f"Using Learning Rate {self.config.lr}")
        optimizer = torch.optim.AdamW(params, lr=self.config.lr, betas=(0.9, 0.999), weight_decay=0.00)
        return optimizer

    def train(
        self,
        image_root: str,
        captions_file: str,
        lq_image_root: str,
        
    ):
        self.load_models()
        os.makedirs(self.config.out_dir, exist_ok=True)

        dataset = ImageCaptionDataset(image_root, captions_file, lq_image_root)
        print(len(dataset), "length of dataset")
        dataloader = DataLoader(
            dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )

        optimizer = self.make_optimizer()
        global_step = 0
        start_epoch = 0
        
        
        first_prompt = None
        if len(dataset) > 0:
            _, _, first_prompt = dataset[0]
            print(f"Using first training prompt for sampling: '{first_prompt}'")
        else:
            print("Warning: Dataset is empty, using default sample prompt")
            _,_, first_prompt = self.config.sample_prompt
            
        
        # Resume from checkpoint if specified
        if self.config.resume_from:
            print(f"Resuming from checkpoint: {self.config.resume_from}")
            checkpoint = torch.load(self.config.resume_from, map_location=self.device)
            global_step = checkpoint.get("step", 0)
            start_epoch = checkpoint.get("epoch", 0)
            
            # Load model state
            if "model" in checkpoint:
                self.sd3.model.load_state_dict(checkpoint["model"], strict=False)
            if "custom_clip" in checkpoint and self.custom_clip_embedding is not None:
                self.custom_clip_embedding.load_state_dict(checkpoint["custom_clip"])
            if "optimizer" in checkpoint:
                optimizer.load_state_dict(checkpoint["optimizer"])
                
            if "clip_vision_adaptor" in checkpoint and getattr(self, "clip_vision_adaptor", None) is not None:
                self.clip_vision_adaptor.load_state_dict(checkpoint["clip_vision_adaptor"])
            
            print(f"Resumed from step {global_step}, epoch {start_epoch}")

        print("training for ", self.config.epochs - start_epoch, " epochs")
        for epoch in range(start_epoch, self.config.epochs):
            for batch_idx, (images, lq_images, captions) in enumerate(dataloader):
                global_step += 1
                
                captions = list(captions)
                for k in range(len(captions)):
                    drop_prob = random.random()
                    if drop_prob < 0.3:
                        captions[k] = ""
                        
                    elif drop_prob >= 0.3 and drop_prob < 0.5:
                        captions[k] = ""
                        lq_images = None
                    else:
                        captions = captions
                        lq_images = lq_images
                            
                loss = self.forward_loss(images, captions, lq_images = lq_images)
                
                print(loss.item(), global_step, "global step")
                wandb.log({"train/loss": float(loss.item())}, step=global_step)
                loss = loss / self.config.grad_accum_steps
                loss.backward()

                if global_step % self.config.grad_accum_steps == 0:
                    # Clip gradients for all trainable parameters
                    all_params = [p for p in self.sd3.model.parameters() if p.requires_grad]
                    if self.custom_clip_embedding is not None:
                        all_params.extend([p for p in self.custom_clip_embedding.parameters() if p.requires_grad])
#                     torch.nn.utils.clip_grad_norm_(all_params, 1.0)
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)

                if global_step % self.config.log_every == 0:
                    print(f"[epoch {epoch} step {global_step}] loss={loss.item() * self.config.grad_accum_steps:.6f}")

                
                
                if global_step % self.config.sample_every == 0:
                    # Generate two images with the first prompt from training data
                    
#                 if self.config.sample_every > 0:
                    gt, lq,first_prompt = dataset[0]
                    gt2, lq2,second_prompt = dataset[1]
                    contrast, lq3, _ = dataset[100]
                    self.generate_sample(global_step, first_prompt, seed_offset=0, im_num = 0, cond_im = lq)
                    self.generate_sample(global_step, first_prompt, seed_offset=0, im_num = 1, cond_im = lq2)
#                     self.generate_sample(global_step, first_prompt, seed_offset=0, im_num = 1, cond_im = lq3)
                    self.generate_sample(global_step, "", seed_offset=1, im_num = 91, cond_im = lq)
                    self.generate_sample(global_step, "", seed_offset=1, im_num = 92, cond_im = lq2)
                    
                if self.config.max_steps is not None and global_step >= self.config.max_steps:
                    break

                if global_step % self.config.save_every == 0:
                    ckpt_path = os.path.join(self.config.out_dir, f"sd3_step_{global_step}_lqmedium_rank32_lognormalsigma.pt")
                    print(f"Saving checkpoint to {ckpt_path}")
                    checkpoint_dict = {
                        "step": global_step,
                        "epoch": epoch,
                        "model": self.sd3.model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": {
                            "lora_rank": self.config.lora_rank,
                            "lora_alpha": self.config.lora_alpha,
                            "use_custom_clip": self.config.use_custom_clip,
                        },
                    }
                    if self.custom_clip_embedding is not None:
                        checkpoint_dict["custom_clip"] = self.custom_clip_embedding.state_dict()
                    
                    if getattr(self, "clip_vision_adaptor", None) is not None:
                        checkpoint_dict["clip_vision_adaptor"] = self.clip_vision_adaptor.state_dict()
                    
                    torch.save(checkpoint_dict, ckpt_path)

            if self.config.max_steps is not None and global_step >= self.config.max_steps:
                break

        # Final save
        final_ckpt = os.path.join(self.config.out_dir, f"sd3_final_step_{global_step}_lqmedium_rank32_lognormalsigma_nullembed.pt")
        print(f"Saving final checkpoint to {final_ckpt}")
        checkpoint_dict = {
            "step": global_step,
            "epoch": self.config.epochs - 1,
            "model": self.sd3.model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": {
                "lora_rank": self.config.lora_rank,
                "lora_alpha": self.config.lora_alpha,
                "use_custom_clip": self.config.use_custom_clip,
            },
        }
        if self.custom_clip_embedding is not None:
            checkpoint_dict["custom_clip"] = self.custom_clip_embedding.state_dict()
            
        if getattr(self, "clip_vision_adaptor", None) is not None:
            checkpoint_dict["clip_vision_adaptor"] = self.clip_vision_adaptor.state_dict()
            
        torch.save(checkpoint_dict, final_ckpt)
        
        
        
    def load_from_ckpt(
        self,
        ckpt_path: str,
        optimizer: torch.optim.Optimizer = None,
        strict_model: bool = False,
        strict_custom_clip: bool = True,
        load_optimizer: bool = False,
    ):
        """
        Load training state from a checkpoint saved by this trainer.

        Expected checkpoint format (as in your save code):
          ckpt["step"], ckpt["epoch"], ckpt["model"], ckpt["optimizer"], ckpt["config"], optional ckpt["custom_clip"]

        IMPORTANT for LoRA:
          - The model *must already have LoRA modules injected* (i.e., call self.load_models() first),
            otherwise LoRA params in state_dict won't match anything.
          - This function will also (optionally) sync self.config.lora_rank/alpha/use_custom_clip to ckpt["config"]
            so your config stays consistent.

        Returns:
          (global_step, start_epoch)
        """
        print(f"Loading checkpoint from: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=self.device)

        # ---- sync config (optional but recommended) ----
        cfg = ckpt.get("config", {}) or {}
        if "lora_rank" in cfg:
            self.config.lora_rank = int(cfg["lora_rank"])
        if "lora_alpha" in cfg:
            self.config.lora_alpha = float(cfg["lora_alpha"])
        if "use_custom_clip" in cfg:
            self.config.use_custom_clip = bool(cfg["use_custom_clip"])

        global_step = int(ckpt.get("step", 0))
        start_epoch = int(ckpt.get("epoch", 0))

        # ---- load model ----
        if "model" not in ckpt:
            raise KeyError("Checkpoint missing key 'model'")

        missing, unexpected = self.sd3.model.load_state_dict(ckpt["model"], strict=strict_model)
        print(f"[ckpt] model loaded. missing={len(missing)} unexpected={len(unexpected)}")
        if len(missing) > 0:
            print("  missing (first 20):", missing[:20])
        if len(unexpected) > 0:
            print("  unexpected (first 20):", unexpected[:20])

        
        # ---- load clip adaptor -----
        if "clip_vision_adaptor" in ckpt and getattr(self, "clip_vision_adaptor", None) is not None:
            self.clip_vision_adaptor.load_state_dict(ckpt["clip_vision_adaptor"], strict=True)
            
        if getattr(self, "clip_vision_adaptor", None) is not None:
            for p in self.clip_vision_adaptor.parameters():
                p.requires_grad_(True)
        
        # ---- load custom clip embedding (if present and initialized) ----
        
        if "custom_clip" in ckpt:
            if self.custom_clip_embedding is None:
                # If checkpoint has custom_clip but current run didn't create it, create it now.
                # We can infer dims from saved proj.weight if available, else default 4096->4096.
                print("[ckpt] Found custom_clip in checkpoint but self.custom_clip_embedding is None; creating it.")
                state = ckpt["custom_clip"]
                if "proj.weight" in state:
                    w = state["proj.weight"]
                    # w shape: [out_dim, in_dim]
                    out_dim = int(w.shape[0])
                    in_dim = int(w.shape[1])
                else:
                    in_dim, out_dim = 4096, 4096

                # Keep your same constructor signature
                self.custom_clip_embedding = CustomCLIPEmbedding(
                    original_embed_dim=in_dim,
                    custom_embed_dim=out_dim,
                    projection=self.config.custom_clip_projection,
                    learnable=True,
                ).to(self.device)

            missing_cc, unexpected_cc = self.custom_clip_embedding.load_state_dict(
                ckpt["custom_clip"], strict=strict_custom_clip
            )
            print(f"[ckpt] custom_clip loaded. missing={len(missing_cc)} unexpected={len(unexpected_cc)}")
            if len(missing_cc) > 0:
                print("  custom_clip missing (first 20):", missing_cc[:20])
            if len(unexpected_cc) > 0:
                print("  custom_clip unexpected (first 20):", unexpected_cc[:20])
        else:
            if self.custom_clip_embedding is not None:
                print("[ckpt] No custom_clip in checkpoint; keeping current self.custom_clip_embedding as-is.")

        # ---- (re)apply requires_grad logic for LoRA + custom clip ----
        # base is frozen in load_models(); enforce LoRA grads again in case load_state_dict changed anything
        if self.config.use_lora:
            lora_cnt = 0
            for name, p in self.sd3.model.named_parameters():
                if "lora" in name.lower():
                    p.requires_grad_(True)
                    lora_cnt += 1
            print(f"[ckpt] Re-enabled gradients for {lora_cnt} LoRA params")

        if self.custom_clip_embedding is not None:
            for p in self.custom_clip_embedding.parameters():
                p.requires_grad_(True)

        # ---- load optimizer (optional) ----
        if optimizer is not None and load_optimizer and ("optimizer" in ckpt):
            try:
                optimizer.load_state_dict(ckpt["optimizer"])
                print("[ckpt] optimizer loaded.")
            except Exception as e:
                print(f"[ckpt] optimizer load failed (continuing without): {e}")

        return global_step, start_epoch
    
    @torch.no_grad()
    def generate_sample(
        self,
        step: int,
        prompt: str,
        seed_offset: int = 0,
        im_num: int = 0,
        cond_im=None,
        denoise: float = 1.0,
        lq_noisy_latent: torch.Tensor = None,
        save_name = None# <- NEW: [1,16,H/8,W/8] in *model latent space* (same as x in sampling loop)
    ):
        """
        Generate a sample image to check training progress.

        NEW behavior:
          - If denoise < 1.0 and lq_noisy_latent is provided, use it as the initial "noise" state
            (instead of pure Gaussian), and only run the tail portion of the sigma schedule.

        Notes:
          - lq_noisy_latent should already be in the same latent space as the sampler state `x`
            (i.e., after SD3LatentFormat.process_in if it came from VAE encode).
          - We still apply sampling.noise_scaling(...) to match the model's expected scaling at sigmas[0].
        """
        print(f"Generating sample at step {step} with prompt: '{prompt}' (seed_offset={seed_offset})...")

        self.sd3.model.eval()
        if self.custom_clip_embedding is not None:
            self.custom_clip_embedding.eval()

        try:
            # Base latent (only used for noise_scaling reference; SD3 uses this constant init in sd3_infer too)
            latent = torch.ones(
                1, 16, self.config.height // 8, self.config.width // 8,
                device=self.device, dtype=torch.float32
            ) * 0.0609

            # Build sigma schedule
            sampling = self.sd3.model.model_sampling
            start = sampling.timestep(sampling.sigma_max)
            end = sampling.timestep(sampling.sigma_min)
            timesteps = torch.linspace(start, end, self.config.sample_steps, device=self.device)
            sigmas = [sampling.sigma(ts) for ts in timesteps] + [0.0]
            sigmas = torch.tensor(sigmas, device=self.device, dtype=torch.float32)

            # If doing partial denoise, drop the first part of the schedule (like sd3_infer)
            if denoise < 1.0:
                cut = int(self.config.sample_steps * (1.0 - float(denoise)))
                sigmas = sigmas[cut:]

            # Prepare initial state: Gaussian noise OR provided lq noisy latent (only when denoise < 1.0)
            use_lq_as_init = (denoise < 1.0) and (lq_noisy_latent is not None)

            gen = torch.Generator(device=self.device)
            gen.manual_seed(self.config.sample_seed + seed_offset)
            
            noise = torch.randn(latent.size(), generator = gen, device=self.device, dtype=torch.float32) ###this is the sampled noise
            if use_lq_as_init:
                latent = lq_noisy_latent.to(device=self.device, dtype=torch.float32)
            else:
                latent = noise      
#             if use_lq_as_init:
#                 # ensure correct device/dtype/shape
#                 noise = lq_noisy_latent.to(device=self.device, dtype=torch.float32)
#                 if noise.shape != latent.shape:
#                     raise ValueError(f"lq_noisy_latent shape {tuple(noise.shape)} != expected {tuple(latent.shape)}")
#             else:
#                 gen = torch.Generator(device=self.device)
#                 gen.manual_seed(self.config.sample_seed + seed_offset)
#                 noise = torch.randn(latent.size(), generator=gen, device=self.device, dtype=torch.float32)

            # Encode prompt
            if cond_im is None:
                cond_c, cond_y = self.encode_prompts([prompt])
                neg_cond_c, neg_cond_y = self.encode_prompts([""])
            else:
                cond_c, cond_y = self.get_cond_image([cond_im], [prompt])
                neg_cond_c, neg_cond_y = self.encode_prompts([""]) #####original uncond
#                 neg_cond_c, neg_cond_y = self.get_cond_image([cond_im], [""]) ####uncond with image embedding

            conditioning = {"c_crossattn": cond_c, "y": cond_y}
            neg_conditioning = {"c_crossattn": neg_cond_c, "y": neg_cond_y}

            # Initial scaling at sigmas[0]
            max_denoise = (
                math.isclose(float(sampling.sigma_max), float(sigmas[0]), rel_tol=1e-05)
                or sigmas[0] > sampling.sigma_max
            )
            # even when using lq_noisy_latent, treat it as the "noise input" to noise_scaling
            x = sampling.noise_scaling(sigmas[0], noise, latent, max_denoise)

            # Sampling loop with CFG
            s_in = x.new_ones([x.shape[0]])
            for i in range(len(sigmas) - 1):
                sigma_hat = sigmas[i]

                x_batch = torch.cat([x, x])
                sigma_batch = torch.cat([sigma_hat * s_in, sigma_hat * s_in])
                cond_c_batch = torch.cat([conditioning["c_crossattn"], neg_conditioning["c_crossattn"]])
                cond_y_batch = torch.cat([conditioning["y"], neg_conditioning["y"]])

                denoised_batch = self.sd3.model.apply_model(
                    x_batch,
                    sigma_batch,
                    c_crossattn=cond_c_batch,
                    y=cond_y_batch,
                )

                pos_out, neg_out = denoised_batch.chunk(2)
                denoised = neg_out + (pos_out - neg_out) * self.config.sample_cfg_scale

                d = (x - denoised) / append_dims(sigma_hat, x.ndim)
                dt = sigmas[i + 1] - sigma_hat
                x = x + d * dt

            # Process output latent
            latent_out = self.latent_format.process_out(x)

            # Decode with VAE
            self.vae.model.eval()
            latent_out = latent_out.to(self.device)
            image = self.vae.model.decode(latent_out).float()
            image = torch.clamp((image + 1.0) / 2.0, 0.0, 1.0)[0]
            decoded_np = (255.0 * np.moveaxis(image.cpu().numpy(), 0, 2)).astype(np.uint8)
            out_image = Image.fromarray(decoded_np)

            os.makedirs(self.config.samples_dir, exist_ok=True)
            
            if save_name is None:
                sample_path = os.path.join(
                    self.config.samples_dir,
                    f"sample_step_{step:06d}_{seed_offset}_recon_{im_num}_new_lora32_cfglrvanilla.png",
                )
            else:
                sample_path = save_name
            out_image.save(sample_path)
            print(f"Sample saved to {sample_path}")

        except Exception as e:
            print(f"Error generating sample: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.sd3.model.train()
            if self.custom_clip_embedding is not None:
                self.custom_clip_embedding.train()



###################################################################################################
### CLI entrypoint (Fire)
###################################################################################################


def main(
    image_root: str,
    captions_file: str,
    model: str = "models/sd3.5_medium.safetensors",
    vae: Optional[str] = None,
    shift: float = 3.0,
    batch_size: int = 2,
    epochs: int = 10,
    lr: float = 1e-4,
    grad_accum_steps: int = 1,
    max_steps: Optional[int] = None,
    log_every: int = 2,
    save_every: int = 1000,
    out_dir: str = "checkpoints",
    # LoRA parameters
    use_lora: bool = True,
    lora_rank: int = 32,
    lora_alpha: float = 1.0,
    lora_dropout: float = 0.0,
    lora_target_modules: Optional[str] = None,  # Comma-separated string, e.g., "qkv,proj,mlp"
    # Custom CLIP embedding parameters
    use_custom_clip: bool = False,
    custom_clip_dim: Optional[int] = None,
    custom_clip_projection: bool = True,
    # Resume training
    resume_from: Optional[str] = None,
    samples_dir: str = "samples",  # Directory to save sample images
    sample_prompt: str = "a photo of a cat",  # Prompt for sampling
    sample_steps: int = 28,  # Sampling steps (fewer for faster sampling)
    sample_cfg_scale: float = 5.0,  # CFG scale for sampling
    sample_seed: int = 42,  # Seed for sampling
    sample_every: int = 50,
    
):
    """
    SD3 training with LoRA and custom CLIP embedding support.

    Example:
      # Basic LoRA fine-tuning with custom CLIP embedding
      python3 -s sd3_train.py \\
        --image_root data/images \\
        --captions_file data/captions.txt \\
        --batch_size 1 --epochs 1 --max_steps 1000 \\
        --use_lora True --lora_rank 8 --lora_alpha 16.0 \\
        --use_custom_clip True --custom_clip_dim 4096

      # Resume training
      python3 -s sd3_train.py \\
        --image_root data/images \\
        --captions_file data/captions.txt \\
        --resume_from checkpoints/sd3_step_1000.pt
    """
    # Parse lora_target_modules if provided as string
    target_modules_list = None
    if lora_target_modules:
        target_modules_list = [m.strip() for m in lora_target_modules.split(",")]
    
    cfg = TrainConfig(
        model=model,
        vae=vae,
        shift=shift,
        batch_size=batch_size,
        epochs=epochs,
        lr=lr,
        grad_accum_steps=grad_accum_steps,
        max_steps=max_steps,
        log_every=log_every,
        save_every=save_every,
        out_dir=out_dir,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        lora_target_modules=target_modules_list,
        use_custom_clip=use_custom_clip,
        custom_clip_dim=custom_clip_dim,
        custom_clip_projection=custom_clip_projection,
        resume_from=resume_from,
        samples_dir=samples_dir,
        sample_prompt=sample_prompt,
        sample_steps=sample_steps,
        sample_cfg_scale=sample_cfg_scale,
        sample_seed=sample_seed,
        sample_every=sample_every,
    )
    trainer = SD3Trainer(cfg)
    
    #####################For inferencing##########################
    
#     trainer.load_models()
#     os.makedirs(trainer.config.out_dir, exist_ok=True)
#     trainer.load_from_ckpt("/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/checkpoints/sd3_final_step_250.pt")
#     init_image = Image.open("bowen_im/IMG_6369.jpeg")
    
#     init_image = Image.open("/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images/000_lq.png")
#     latent_code, image_torch = trainer.vae_encode(init_image)
#     latent_code = trainer.latent_format.process_in(latent_code)
# #     prompt = "A man [[280,000,633,924]] in a white shirt [[300,071,647,400]] and blue shorts [[334,372,589,593]] is running on a track [[000,800,995,995]]."
#     prompt = "A couch is in the foreground, with a white plate on a table in the background."
    
#     for i in range(10):
#         name = f"samples/sofa_{i}_denoise_0.7_cfg5.0.png"
# #         name = f"/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images/000_lq.png"
#         trainer.generate_sample(
#             step=766,
#             prompt=prompt,
#             seed_offset=0,
#             im_num=0,
# #             cond_im="bowen_im/IMG_6369.jpeg",
#             cond_im = "/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_images/000_lq.png",
#             denoise= 0.7,
#             lq_noisy_latent=latent_code,  # <- NEW: [1,16,H/8,W/8] in *model latent space* (same as x in sampling loop)
#             save_name = name
#         )
#########################################################################################
    
    
                            
    
    trainer.train(image_root="/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/images_medium", captions_file=captions_file, lq_image_root="/scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/lq_medium_images")
    wandb.finish()


if __name__ == "__main__":
    fire.Fire(main)

