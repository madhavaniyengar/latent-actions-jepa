import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
import tensorflow as tf
import wandb
import os
import yaml
from diffusers import AutoencoderKL, UNet2DConditionModel, DDPMScheduler, DDIMScheduler, StableDiffusionPipeline
import torch
from xformers.ops import MemoryEfficientAttentionFlashAttentionOp
from torch.optim.lr_scheduler import LambdaLR

class StableDiffusion(torch.nn.Module):
    def __init__(self, model_id="stabilityai/stable-diffusion-2-1", device="cuda", freeze_unet=False):
        super().__init__()
        self.device = device

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        pipeline = StableDiffusionPipeline.from_pretrained(model_id)
        pipeline = pipeline.to(self.device)
        
        self.unet = pipeline.unet
        self.vae = pipeline.vae
        self.text_encoder = pipeline.text_encoder
        self.tokenizer = pipeline.tokenizer
        self.scheduler = pipeline.scheduler
        self.image_processor = pipeline.feature_extractor
        
        self.encode_empty_text()
        self.empty_text_embed = self.empty_text_embed.detach().clone().to(device)
        
        self.unet.requires_grad_(True)
        
        self.vae.requires_grad_(False)
        self.text_encoder.requires_grad_(False)
        
        self.unet.enable_xformers_memory_efficient_attention(attention_op=MemoryEfficientAttentionFlashAttentionOp)
        self.unet.set_attention_slice("auto")

    def encode_image(self, image):
        image = torch.from_numpy(np.array(self.image_processor(image)['pixel_values'])).to(self.device)
        latents = self.vae.encode(image).latent_dist.sample() * 0.18215 # get the latent sample from VAE
        return latents

    def decode_latents(self, latents):
        latents = latents / 0.18215
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def forward(self, I0, masked_embedding=None):
        batch_size, _, height, width = I0.shape

        # move images to latent space
        I0_latent = self.encode_image(I0)
        
        # get random timesteps for entire batch
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (batch_size,), device=self.device
        ).long()
        
        # gaussian noise
        noise = torch.randn_like(I0_latent)
        noisy_latents = self.scheduler.add_noise(I0_latent, noise, timesteps)

        # predict noise with unet
        noise_pred = self.unet(
            noisy_latents, 
            timesteps, 
            encoder_hidden_states=masked_embedding
        ).sample

        # return noise prediction and actual noise added
        return noise_pred, noise
    
    def generate(self, I0, masked_embedding=None, num_inference_steps=50):
        device = self.device

        with torch.no_grad():
            I0_latent = self.encode_image(I0)

            noisy_latents = torch.randn_like(I0_latent).to(device)

            self.scheduler.set_timesteps(num_inference_steps, device=self.device)
            timesteps = self.scheduler.timesteps
            
            iterable = tqdm(
                enumerate(timesteps),
                total=len(timesteps),
                leave=False,
                desc=" " * 4 + "Diffusion denoising",
            )
            
            for i, t in iterable:
                noise_pred = self.unet(noisy_latents, t, encoder_hidden_states=masked_embedding).sample
                noisy_latents = self.scheduler.step(noise_pred, t, noisy_latents).prev_sample

        denoised_latents = self.decode_latents(noisy_latents)

        return denoised_latents
