import os
import cv2
import torch
import torchvision.transforms.functional as TF
import numpy as np
import lpips
import pyiqa
import torch.nn.functional as F
from torchmetrics.multimodal.clip_iqa import CLIPImageQualityAssessment
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import argparse

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}


def is_image(p: Path):
    return p.is_file() and p.suffix.lower() in IMG_EXTS


def sorted_image_list(folder: str):
    folder = Path(folder)
    files = [p for p in folder.rglob("*") if is_image(p)]
    # sort by relative path string for stable ordering across nested dirs
    files = sorted(files, key=lambda p: str(p.relative_to(folder)))
    return files


def load_image_tensor(path: Path, device="cuda"):
    # [1,3,H,W] float in [0,1]
    img_bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise RuntimeError(f"Failed to read image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    x = TF.to_tensor(img_rgb).unsqueeze(0).to(device)
    return x


def compute_psnr(img1, img2):
    # img1,img2: [1,3,H,W] in [0,1]
    mse = torch.mean((img1 - img2) ** 2)
    if mse.item() == 0:
        return float("inf")
    return float((20 * torch.log10(1.0 / torch.sqrt(mse))).item())


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", type=str, required=True)
    parser.add_argument("--recon_dir", type=str, required=True)
    parser.add_argument("--out_csv", type=str, default="metrics.csv")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resize_recon_to_gt", action="store_true",
                        help="If set, resize recon to GT size when mismatch (bilinear).")
    args = parser.parse_args()

    device = args.device

    gt_files = sorted_image_list(args.gt_dir)
    recon_files = sorted_image_list(args.recon_dir)

    print(f"GT images   : {len(gt_files)}")
    print(f"Recon images: {len(recon_files)}")

    N = min(len(gt_files), len(recon_files))
    
    N = min(N, 10)
    print(f"Evaluating N = min(len(gt), len(recon)) = {N}")

    if N == 0:
        raise RuntimeError("No images found in one or both directories.")

    # --- metrics (similar to your script) ---
    clip_metric = CLIPImageQualityAssessment().to(device)
    musiq_fn = pyiqa.create_metric("musiq", device=device)      # NR on recon
    ssim_fn = pyiqa.create_metric("ssim", device=device)        # FR needs recon+gt
    lpips_model = lpips.LPIPS(net="vgg").eval().to(device)      # FR needs recon+gt

    rows = []
    psnrs_, ssims_, lpips_, musiqs_, clips_ = [], [], [], [], []

    for i in tqdm(range(N), desc="Computing metrics"):
        gt_path = gt_files[i]
        recon_path = recon_files[i]
        print(gt_path, recon_path)

        gt = load_image_tensor(gt_path, device=device)
        recon = load_image_tensor(recon_path, device=device)

        if gt.shape[-2:] != recon.shape[-2:]:
            if args.resize_recon_to_gt:
                recon = F.interpolate(
                    recon, size=(gt.shape[-2], gt.shape[-1]),
                    mode="bilinear", align_corners=False
                )
            else:
                raise RuntimeError(
                    f"Size mismatch at i={i}\n"
                    f"  gt   : {gt_path}  {gt.shape[-2:]}\n"
                    f"  recon: {recon_path}  {recon.shape[-2:]}\n"
                    f"Use --resize_recon_to_gt to auto-resize recon."
                )

        gt = gt.clamp(0, 1)
        recon = recon.clamp(0, 1)

        # FR metrics
        psnr_val = compute_psnr(recon, gt)
        ssim_val = float(ssim_fn(recon, gt).item())
        
        recon_lpips, gt_lpips = F.interpolate(recon, size = (512, 512), mode="bilinear", align_corners=False), F.interpolate(gt, size = (512, 512), mode="bilinear", align_corners=False)
        
        lpips_val = float(lpips_model(recon * 2 - 1, gt * 2 - 1).item())

        # NR metrics (on recon)
        musiq_val = float(musiq_fn(recon).item())
        clip_val = float(clip_metric(recon).item())
        clip_metric.reset()  # like your video code

        psnrs_.append(psnr_val)
        ssims_.append(ssim_val)
        lpips_.append(lpips_val)
        musiqs_.append(musiq_val)
        clips_.append(clip_val)

        rows.append({
            "i": i,
            "gt_path": str(gt_path),
            "recon_path": str(recon_path),
            "psnr": psnr_val,
            "ssim": ssim_val,
            "lpips": lpips_val,
            "musiq": musiq_val,
            "clipiqa": clip_val,
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved per-pair metrics -> {args.out_csv}")

    print("\n=== Summary (mean) ===")
    print(f"PSNR    : {float(np.mean(psnrs_))}")
    print(f"SSIM    : {float(np.mean(ssims_))}")
    print(f"LPIPS   : {float(np.mean(lpips_))}")
    print(f"MUSIQ   : {float(np.mean(musiqs_))}")
    print(f"CLIP-IQA: {float(np.mean(clips_))}")


if __name__ == "__main__":
    """
    python eval_performance.py --gt_dir /scratch/liyues_root/liyues/shared_data/bowenbw/sd3-ref/data/images --recon_dir /scratch/liyues_root/liyues/shared_data/bowenbw/sd3.5/recon_all --out_csv metrics.csv --device cuda --resize_recon_to_gt
    
    """
    main()

    
    