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
import logging
import pprint

# Import V-JEPA model and necessary utilities
import src.models.vision_transformer as vit
from src.datasets.data_manager import init_data
from src.utils.distributed import init_distributed, AllReduce
from src.utils.schedulers import WarmupCosineSchedule, CosineWDSchedule
from src.utils.logging import AverageMeter, CSVLogger
from src.masks.utils import apply_masks

from visualization.diffusion import StableDiffusion

# Set up logging
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Global seed for reproducibility
_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)

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

    def encode_empty_text(self):
        # encoding empty text
        text_input = self.tokenizer(
            "",
            padding="max_length",
            max_length=self.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt",
        )
        self.empty_text_embed = self.text_encoder(text_input.input_ids.to(self.device))[0]

    def encode_image(self, image):
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).to(self.device)
        elif isinstance(image, list):
            image = torch.stack([torch.from_numpy(np.array(self.image_processor(img)['pixel_values'])).squeeze(0) for img in image]).to(self.device)
        else:
            image = image.to(self.device)
        latents = self.vae.encode(image).latent_dist.sample() * 0.18215 # get the latent sample from VAE
        return latents

    def decode_latents(self, latents):
        latents = latents / 0.18215
        image = self.vae.decode(latents).sample
        image = (image / 2 + 0.5).clamp(0, 1)
        return image

    def forward(self, I0, masked_embedding=None):
        batch_size = I0.shape[0]

        # move images to latent space
        I0_latent = self.encode_image(I0)
        
        # get random timesteps for entire batch
        timesteps = torch.randint(
            0, self.scheduler.config.num_train_timesteps, (batch_size,), device=self.device
        ).long()
        
        # gaussian noise
        noise = torch.randn_like(I0_latent)
        noisy_latents = self.scheduler.add_noise(I0_latent, noise, timesteps)

        # Prepare encoder_hidden_states
        if masked_embedding is not None:
            sequence_length = self.empty_text_embed.shape[1]
            masked_embedding = masked_embedding.unsqueeze(1).repeat(1, sequence_length, 1)
            encoder_hidden_states = masked_embedding
        else:
            encoder_hidden_states = self.empty_text_embed

        # predict noise with unet
        noise_pred = self.unet(
            noisy_latents, 
            timesteps, 
            encoder_hidden_states=encoder_hidden_states
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
                if masked_embedding is not None:
                    sequence_length = self.empty_text_embed.shape[1]
                    masked_embedding = masked_embedding.unsqueeze(1).repeat(1, sequence_length, 1)
                    encoder_hidden_states = masked_embedding
                else:
                    encoder_hidden_states = self.empty_text_embed

                noise_pred = self.unet(noisy_latents, t, encoder_hidden_states=encoder_hidden_states).sample
                noisy_latents = self.scheduler.step(noise_pred, t, noisy_latents).prev_sample

        denoised_latents = self.decode_latents(noisy_latents)

        return denoised_latents

def main(args_eval, resume_preempt=False):

    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- PRETRAIN
    args_pretrain = args_eval.get('pretrain')
    checkpoint_key = args_pretrain.get('checkpoint_key', 'target_encoder')
    model_name = args_pretrain.get('model_name', None)
    patch_size = args_pretrain.get('patch_size', None)
    pretrain_folder = args_pretrain.get('folder', None)
    ckp_fname = args_pretrain.get('checkpoint', None)
    tag = args_pretrain.get('write_tag', None)
    use_sdpa = args_pretrain.get('use_sdpa', True)
    use_SiLU = args_pretrain.get('use_silu', False)
    tight_SiLU = args_pretrain.get('tight_silu', True)
    uniform_power = args_pretrain.get('uniform_power', False)
    pretrained_path = os.path.join(pretrain_folder, ckp_fname)
    # Optional [for Video model]:
    tubelet_size = args_pretrain.get('tubelet_size', 2)
    pretrain_frames_per_clip = args_pretrain.get('frames_per_clip', 1)

    # -- DATA
    args_data = args_eval.get('data')
    train_data_path = [args_data.get('dataset_train')]
    val_data_path = [args_data.get('dataset_val')]
    dataset_type = args_data.get('dataset_type', 'VideoDataset')
    num_classes = args_data.get('num_classes')
    eval_num_segments = args_data.get('num_segments', 1)
    eval_frames_per_clip = args_data.get('frames_per_clip', 16)
    eval_frame_step = args_data.get('frame_step', 4)
    eval_duration = args_data.get('clip_duration', None)
    eval_num_views_per_segment = args_data.get('num_views_per_segment', 1)

    # -- OPTIMIZATION
    args_opt = args_eval.get('optimization')
    resolution = args_opt.get('resolution', 224)
    batch_size = args_opt.get('batch_size')
    attend_across_segments = args_opt.get('attend_across_segments', False)
    num_epochs = args_opt.get('num_epochs')
    wd = args_opt.get('weight_decay')
    start_lr = args_opt.get('start_lr')
    lr = args_opt.get('lr')
    final_lr = args_opt.get('final_lr')
    warmup = args_opt.get('warmup')
    use_bfloat16 = args_opt.get('use_bfloat16')

    # -- EXPERIMENT-ID/TAG (optional)
    resume_checkpoint = args_eval.get('resume_checkpoint', False) or resume_preempt
    eval_tag = args_eval.get('tag', None)

    # ----------------------------------------------------------------------- #

    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    # -- log/checkpointing paths
    folder = os.path.join(pretrain_folder, 'video_classification_frozen/')
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    latest_path = os.path.join(folder, f'{tag}-latest.pth.tar')

    # -- make csv_logger
    if rank == 0:
        csv_logger = CSVLogger(log_file,
                               ('%d', 'epoch'),
                               ('%.5f', 'loss'))

    # Initialize encoder (frozen)
    encoder = init_model(
        crop_size=resolution,
        device=device,
        pretrained=pretrained_path,
        model_name=model_name,
        patch_size=patch_size,
        tubelet_size=tubelet_size,
        frames_per_clip=pretrain_frames_per_clip,
        uniform_power=uniform_power,
        checkpoint_key=checkpoint_key,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
        use_sdpa=use_sdpa)
    encoder.eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Initialize diffusion model
    diffusion_model = StableDiffusion(device=device)
    diffusion_model.to(device)

    train_loader = make_dataloader(
        dataset_type=dataset_type,
        root_path=train_data_path,
        resolution=resolution,
        frames_per_clip=eval_frames_per_clip,
        frame_step=eval_frame_step,
        eval_duration=eval_duration,
        num_segments=eval_num_segments,
        num_views_per_segment=1,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=True)
    val_loader = make_dataloader(
        dataset_type=dataset_type,
        root_path=val_data_path,
        resolution=resolution,
        frames_per_clip=eval_frames_per_clip,
        frame_step=eval_frame_step,
        num_segments=eval_num_segments,
        eval_duration=eval_duration,
        num_views_per_segment=eval_num_views_per_segment,
        allow_segment_overlap=True,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        training=False)
    ipe = len(train_loader)
    logger.info(f'Dataloader created... iterations per epoch: {ipe}')

    # -- optimizer and scheduler for diffusion model
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        model=diffusion_model,
        wd=wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        use_bfloat16=use_bfloat16)
    diffusion_model = DistributedDataParallel(diffusion_model, static_graph=True)

    # -- load training checkpoint
    start_epoch = 0
    if resume_checkpoint:
        diffusion_model, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=latest_path,
            model=diffusion_model,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch*ipe):
            scheduler.step()
            wd_scheduler.step()
            
    def save_checkpoint(epoch):
        save_dict = {
            'diffusion_model': diffusion_model.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr
        }
        if rank == 0:
            torch.save(save_dict, latest_path)

    # TRAIN LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))
        train_loss = run_one_epoch(
            device=device,
            training=True,
            encoder=encoder,
            diffusion_model=diffusion_model,
            diffusion_optimizer=diffusion_optimizer,
            diffusion_scheduler=diffusion_scheduler,
            diffusion_wd_scheduler=diffusion_wd_scheduler,
            diffusion_scaler=diffusion_scaler,
            data_loader=train_loader,
            use_bfloat16=use_bfloat16)

        val_loss = run_one_epoch(
            device=device,
            training=False,
            encoder=encoder,
            diffusion_model=diffusion_model,
            diffusion_optimizer=diffusion_optimizer,
            diffusion_scheduler=diffusion_scheduler,
            diffusion_wd_scheduler=diffusion_wd_scheduler,
            diffusion_scaler=diffusion_scaler,
            data_loader=val_loader,
            use_bfloat16=use_bfloat16)

        logger.info('[%5d] train loss: %.3f val loss: %.3f' % (epoch + 1, train_loss, val_loss))
        if rank == 0:
            csv_logger.log(epoch + 1, train_loss, val_loss)
        save_checkpoint(epoch + 1)

def run_one_epoch(
    device,
    training,
    encoder,
    diffusion_model,
    scaler,
    optimizer,
    scheduler,
    wd_scheduler,
    data_loader,
    use_bfloat16,
):
    diffusion_model.train(mode=training)
    criterion = torch.nn.MSELoss()
    loss_meter = AverageMeter()
    for itr, data in enumerate(data_loader):

        if training:
            scheduler.step()
            wd_scheduler.step()

        with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):

            # Load data and put on GPU
            # Assume data[0] contains the videos
            videos = [di[0] for di in data[0]]  # di[0] because we have only one spatial view
            videos = torch.stack(videos).to(device)  # Shape: (B, C, T, H, W)
            batch_size = videos.shape[0]

            # Apply masking to videos (excluding last frame)
            masked_videos = apply_masking(videos[:, :, :-1, :, :])  # Shape: (B, C, T-1, H, W)

            # Get target frames (last frame)
            target_frames = videos[:, :, -1, :, :]  # Shape: (B, C, H, W)

            # Process masked_videos through encoder to get embeddings
            with torch.no_grad():
                embeddings = encoder(masked_videos)  # Shape: (B, D)
                embeddings = embeddings.unsqueeze(1)  # Shape: (B, 1, D)

            # Forward pass through diffusion model
            noise_pred, noise = diffusion_model.forward(target_frames, masked_embedding=embeddings)

            # Compute loss
            loss = criterion(noise_pred, noise)
            loss_value = loss.item()
            loss_meter.update(loss_value)

        if training:
            optimizer.zero_grad()
            if use_bfloat16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(diffusion_model.parameters(), 1.0)
                optimizer.step()

        if itr % 20 == 0:
            logger.info('[%5d] loss: %.3f [mem: %.2e]'
                        % (itr, loss_meter.avg,
                           torch.cuda.max_memory_allocated() / 1024.**2))

    return loss_meter.avg


def load_checkpoint(
    device,
    r_path,
    model,
    opt,
    scaler
):
    try:
        checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
        epoch = checkpoint['epoch']

        # -- loading model
        pretrained_dict = checkpoint['diffusion_model']
        msg = model.load_state_dict(pretrained_dict)
        logger.info(f'loaded pretrained diffusion model from epoch {epoch} with msg: {msg}')

        # -- loading optimizer
        opt.load_state_dict(checkpoint['opt'])
        if scaler is not None:
            scaler.load_state_dict(checkpoint['scaler'])
        logger.info(f'loaded optimizers from epoch {epoch}')
        logger.info(f'read-path: {r_path}')
        del checkpoint

    except Exception as e:
        logger.info(f'Encountered exception when loading checkpoint {e}')
        epoch = 0

    return model, opt, scaler, epoch

def load_pretrained(
    encoder,
    pretrained,
    checkpoint_key='target_encoder'
):
    logger.info(f'Loading pretrained model from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu')
    try:
        pretrained_dict = checkpoint[checkpoint_key]
    except Exception:
        pretrained_dict = checkpoint['encoder']

    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f'loaded pretrained model with msg: {msg}')
    logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {pretrained}')
    del checkpoint
    return encoder

def make_dataloader(
    root_path,
    batch_size,
    world_size,
    rank,
    dataset_type='VideoDataset',
    resolution=224,
    frames_per_clip=16,
    frame_step=4,
    num_segments=8,
    eval_duration=None,
    num_views_per_segment=1,
    allow_segment_overlap=True,
    training=False,
    num_workers=12,
    subset_file=None
):
    # Make Video Transforms
    transform = make_transforms(
        training=training,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4/3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.25,
        auto_augment=True,
        motion_shift=False,
        crop_size=resolution,
    )

    data_loader, _ = init_data(
        data=dataset_type,
        root_path=root_path,
        transform=transform,
        batch_size=batch_size,
        world_size=world_size,
        rank=rank,
        clip_len=frames_per_clip,
        frame_sample_rate=frame_step,
        duration=eval_duration,
        num_clips=num_segments,
        allow_clip_overlap=allow_segment_overlap,
        num_workers=num_workers,
        copy_data=False,
        drop_last=False,
        subset_file=subset_file)
    return data_loader

def init_model(
    device,
    pretrained,
    model_name,
    patch_size=16,
    crop_size=224,
    # Video specific parameters
    frames_per_clip=16,
    tubelet_size=2,
    use_sdpa=False,
    use_SiLU=False,
    tight_SiLU=True,
    uniform_power=False,
    checkpoint_key='target_encoder'
):
    encoder = vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=frames_per_clip,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_SiLU=use_SiLU,
        tight_SiLU=tight_SiLU,
    )

    encoder.to(device)
    encoder = load_pretrained(encoder=encoder, pretrained=pretrained, checkpoint_key=checkpoint_key)
    return encoder

def init_opt(
    model,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    use_bfloat16=False
):
    param_groups = [
        {
            'params': (p for n, p in model.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in model.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]

    logger.info('Using AdamW')
    optimizer = torch.optim.AdamW(param_groups)
    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs*iterations_per_epoch))
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs*iterations_per_epoch))
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None
    return optimizer, scaler, scheduler, wd_scheduler