from huggingface_hub import hf_hub_download




hf_hub_download("stabilityai/stable-diffusion-3.5-large", "sd3.5_large.safetensors", local_dir="models")
# hf_hub_download("stabilityai/stable-diffusion-3.5-controlnets", "sd3.5_large_controlnet_blur.safetensors", local_dir="models")
# hf_hub_download("stabilityai/stable-diffusion-3.5-medium", "sd3.5_medium.safetensors", local_dir = "models")
