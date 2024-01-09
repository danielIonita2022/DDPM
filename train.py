import numpy as np

from denoising_diffusion_pytorch import Unet, GaussianDiffusion, Trainer
import pandas as pd
from PIL import Image
import os
import io
import torch

def tensor_to_pil(tensor):
    tensor = tensor.permute(1, 2, 0)
    tensor = (tensor * 255).clamp(0, 255).byte()
    return Image.fromarray(tensor.cpu().numpy())

model = Unet(
    dim=64,
    dim_mults=(1, 2, 4, 8),
    flash_attn=True
)

print(model)

diffusion = GaussianDiffusion(
    model,
    image_size=136,
    timesteps=500,  # number of steps
    sampling_timesteps=100
).cuda()

trainer = Trainer(
    diffusion,
    '/mnt/hdd1/home/danielionita/diffusion_project/images_folder',
    train_batch_size=16,
    train_lr=8e-5,
    train_num_steps=6000,  # total training steps # modificat de la 700000
    gradient_accumulate_every=2,
    ema_decay=0.995,
    amp=True,  # turn on mixed precision
    calculate_fid=True,  # whether to calculate fid during training
    num_fid_samples=5000
)

trainer.load(5)
trainer.train()
trained_model = trainer.model
print(trained_model)
sampled_images = trained_model.sample(batch_size=16)
for i, img_tensor in enumerate(sampled_images):
    img = tensor_to_pil(img_tensor)
    img.save(f'/mnt/hdd1/home/danielionita/diffusion_project/denoising-diffusion-pytorch/results/testing/sampled_image_{i}.png')  # Save each image

print(trainer.fid_scorer.fid_score())


