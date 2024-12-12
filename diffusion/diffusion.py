import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    # -- WARNING: IF DOING DISTRIBUTED TRAINING ON A NON-SLURM CLUSTER, MAKE
    # --          SURE TO UPDATE THIS TO GET LOCAL-RANK ON NODE, OR ENSURE
    # --          THAT YOUR JOBS ARE LAUNCHED WITH ONLY 1 DEVICE VISIBLE
    # --          TO EACH PROCESS
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import time
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from src.datasets.data_manager import init_data
from src.masks.random_tube import MaskCollator as TubeMaskCollator
from src.masks.multiblock3d import MaskCollator as MB3DMaskCollator
from src.masks.random_p import MaskCollator as RandomPMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed, AllReduce
from src.utils.logging import (
    CSVLogger,
    gpu_timer,
    get_logger,
    grad_logger,
    adamw_logger,
    AverageMeter)
from src.utils.tensors import repeat_interleave_batch

from app.vjepa.utils import (
    load_checkpoint,
    init_video_model,
    init_opt,
)
from app.vjepa.transforms import make_transforms
from transformers import CLIPTokenizer, CLIPTextModel
log_timings = True
log_freq = 10
checkpoint_freq = 1
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__)
# Define the conditional diffusion model (U-Net architecture)
class ConditionalUNet(nn.Module):
    def __init__(self, image_channels=3, cond_channels=768, time_embed_dim=256):
        super(ConditionalUNet, self).__init__()
        self.image_channels = image_channels
        self.cond_channels = cond_channels
        self.time_embed_dim = time_embed_dim

        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Embedding(1000, time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # Conditioning embedding
        self.cond_embed = nn.Sequential(
            nn.Linear(cond_channels, time_embed_dim),
            nn.ReLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # U-Net architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 64, kernel_size=3, stride=1, padding=1),  # 64 x H x W
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 128 x H/2 x W/2
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 256 x H/4 x W/4
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256 + time_embed_dim + time_embed_dim, 128, kernel_size=4, stride=2, padding=1),  # 128 x H/2 x W/2
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 64 x H x W
            nn.ReLU(),
            nn.Conv2d(64, image_channels, kernel_size=3, stride=1, padding=1),  # image_channels x H x W
        )

    def forward(self, x, t, cond):
        # x: [B, C, H, W]
        # t: [B]
        # cond: [B, cond_channels]
        B = x.size(0)

        # Embed time step t
        t_embed = self.time_embed(t)  # [B, time_embed_dim]

        # Embed conditioning information
        cond_embed = self.cond_embed(cond)  # [B, time_embed_dim]

        # Encode image
        h = self.encoder(x)  # [B, 256, H/4, W/4]

        # Expand embeddings to match spatial dimensions
        t_embed = t_embed.view(B, -1, 1, 1).repeat(1, 1, h.size(2), h.size(3))  # [B, time_embed_dim, H/4, W/4]
        cond_embed = cond_embed.view(B, -1, 1, 1).repeat(1, 1, h.size(2), h.size(3))  # [B, time_embed_dim, H/4, W/4]

        # Concatenate embeddings with encoded image
        h = torch.cat([h, t_embed, cond_embed], dim=1)  # [B, 256 + 2 * time_embed_dim, H/4, W/4]

        # Decode to get output image
        x_recon = self.decoder(h)  # [B, image_channels, H, W]

        return x_recon

# Define the noise schedule and diffusion process
def get_beta_schedule(timesteps, beta_start=0.0001, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, timesteps)

def get_alphas(betas):
    alphas = 1.0 - betas
    alpha_bar = torch.cumprod(alphas, dim=0)
    return alphas, alpha_bar

def forward_diffusion_sample(x_0, t, alpha_bar_sqrt, one_minus_alpha_bar_sqrt):
    noise = torch.randn_like(x_0)
    x_t = alpha_bar_sqrt[t].view(-1, 1, 1, 1) * x_0 + one_minus_alpha_bar_sqrt[t].view(-1, 1, 1, 1) * noise
    return x_t, noise

# Training loop
def train_diffusion_model(model, dataloader, optimizer, epochs, device, timesteps):
    betas = get_beta_schedule(timesteps)
    alphas, alpha_bars = get_alphas(betas)
    alpha_bar_sqrt = torch.sqrt(alpha_bars).to(device)
    one_minus_alpha_bar_sqrt = torch.sqrt(1 - alpha_bars).to(device)

    for epoch in range(epochs):
        model.train()
        for batch in dataloader:
            x_0 = batch['images'].to(device)  # Original images
            cond = batch['video_embeddings'].to(device)  # Video embeddings from frozen V-JEPA

            B = x_0.size(0)

            # Sample random time steps for each image in the batch
            t = torch.randint(0, timesteps, (B,), device=device).long()

            # Generate noisy images and the noise used
            x_t, noise = forward_diffusion_sample(x_0, t, alpha_bar_sqrt, one_minus_alpha_bar_sqrt)

            # Predict the noise using the model
            noise_pred = model(x_t, t, cond)

            # Compute loss between the predicted noise and the actual noise
            loss = F.mse_loss(noise_pred, noise)

            # Backpropagation and optimization step
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    cfgs_meta = args.get('meta')
    load_model = cfgs_meta.get('load_checkpoint') or resume_preempt
    r_file = cfgs_meta.get('read_checkpoint', None)
    seed = cfgs_meta.get('seed', _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get('save_every_freq', -1)
    skip_batches = cfgs_meta.get('skip_batches', -1)
    use_sdpa = cfgs_meta.get('use_sdpa', False)
    which_dtype = cfgs_meta.get('dtype')
    logger.info(f'{which_dtype=}')
    if which_dtype.lower() == 'bfloat16':
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == 'float16':
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MASK
    cfgs_mask = args.get('mask')

    # -- MODEL
    cfgs_model = args.get('model')
    model_name = cfgs_model.get('model_name')
    pred_depth = cfgs_model.get('pred_depth')
    pred_embed_dim = cfgs_model.get('pred_embed_dim')
    uniform_power = cfgs_model.get('uniform_power', True)
    use_mask_tokens = cfgs_model.get('use_mask_tokens', True)
    zero_init_mask_tokens = cfgs_model.get('zero_init_mask_tokens', True)
    action_dim = cfgs_model.get('action_dim', pred_embed_dim)  # Set action_dim
    delta_frames = cfgs_model.get('delta_frames', 4)  # Set action_dim

    # -- DATA
    cfgs_data = args.get('data')
    dataset_type = cfgs_data.get('dataset_type', 'videodataset')
    mask_type = cfgs_data.get('mask_type', 'multiblock3d')
    dataset_paths = cfgs_data.get('datasets', [])
    datasets_weights = cfgs_data.get('datasets_weights', None)
    if datasets_weights is not None:
        assert len(datasets_weights) == len(dataset_paths), 'Must have one sampling weight specified for each dataset'
    batch_size = cfgs_data.get('batch_size')
    num_clips = cfgs_data.get('num_clips')
    num_frames = cfgs_data.get('num_frames')
    tubelet_size = cfgs_data.get('tubelet_size')
    sampling_rate = cfgs_data.get('sampling_rate')
    duration = cfgs_data.get('clip_duration', None)
    crop_size = cfgs_data.get('crop_size', 224)
    patch_size = cfgs_data.get('patch_size')
    pin_mem = cfgs_data.get('pin_mem', False)
    num_workers = cfgs_data.get('num_workers', 1)
    filter_short_videos = cfgs_data.get('filter_short_videos', False)
    decode_one_clip = cfgs_data.get('decode_one_clip', True)
    log_resource_util_data = cfgs_data.get('log_resource_utilization', False)

    # -- DATA AUGS
    cfgs_data_aug = args.get('data_aug')
    ar_range = cfgs_data_aug.get('random_resize_aspect_ratio', [3/4, 4/3])
    rr_scale = cfgs_data_aug.get('random_resize_scale', [0.3, 1.0])
    motion_shift = cfgs_data_aug.get('motion_shift', False)
    reprob = cfgs_data_aug.get('reprob', 0.)
    use_aa = cfgs_data_aug.get('auto_augment', False)

    # -- LOSS
    cfgs_loss = args.get('loss')
    loss_exp = cfgs_loss.get('loss_exp')
    reg_coeff = cfgs_loss.get('reg_coeff')

    # -- OPTIMIZATION
    cfgs_opt = args.get('optimization')
    ipe = cfgs_opt.get('ipe', None)
    ipe_scale = cfgs_opt.get('ipe_scale', 1.0)
    clip_grad = cfgs_opt.get('clip_grad', None)
    wd = float(cfgs_opt.get('weight_decay'))
    final_wd = float(cfgs_opt.get('final_weight_decay'))
    num_epochs = cfgs_opt.get('epochs')
    warmup = cfgs_opt.get('warmup')
    start_lr = cfgs_opt.get('start_lr')
    lr = cfgs_opt.get('lr')
    final_lr = cfgs_opt.get('final_lr')
    ema = cfgs_opt.get('ema')
    betas = cfgs_opt.get('betas', (0.9, 0.999))
    eps = cfgs_opt.get('eps', 1.e-8)

    # -- LOGGING
    cfgs_logging = args.get('logging')
    folder = cfgs_logging.get('folder')
    tag = cfgs_logging.get('write_tag')

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #    # Initialize the diffusion model
    h_dim = pred_embed_dim  # Assuming pred_embed_dim is the dimension of h
    diffusion_model = ConditionalUNet(image_channels=3, cond_channels=h_dim).to(device)
    diffusion_model = DistributedDataParallel(diffusion_model, device_ids=[device.index])
    diffusion_optimizer = optim.Adam(diffusion_model.parameters(), lr=1e-4)
    diffusion_scaler = torch.cuda.amp.GradScaler(enabled=mixed_precision)

    # Precompute betas, alphas for the diffusion process
    timesteps = 1000
    betas = get_beta_schedule(timesteps)
    alphas, alpha_bars = get_alphas(betas)
    alpha_bar_sqrt = torch.sqrt(alpha_bars).to(device)
    one_minus_alpha_bar_sqrt = torch.sqrt(1 - alpha_bars).to(device)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method('spawn')
    except Exception:
        pass

    # -- init torch distributed backend
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank/world-size) {rank}/{world_size}')

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    log_file = os.path.join(folder, f'{tag}_r{rank}.csv')
    latest_file = f'{tag}-latest.pth.tar'
    latest_path = os.path.join(folder, latest_file)
    load_path = None
    if load_model:
        load_path = os.path.join(folder, r_file) if r_file is not None else latest_path
        if not os.path.exists(load_path):
            load_path = None
            load_model = False

    # -- make csv_logger
    csv_logger = CSVLogger(
        log_file,
        ('%d', 'epoch'),
        ('%d', 'itr'),
        ('%.5f', 'loss'),
        ('%.5f', 'loss-jepa'),
        ('%.5f', 'reg-loss'),
        ('%.5f', 'enc-grad-norm'),
        ('%.5f', 'pred-grad-norm'),
        ('%d', 'gpu-time(ms)'),
        ('%d', 'wall-time(ms)'),
    )

    # -- init model
    encoder, predictor, action_model = init_video_model(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=len(cfgs_mask),
        zero_init_mask_tokens=zero_init_mask_tokens,
        device=device,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_embed_dim=pred_embed_dim,
        action_dim=action_dim,
        use_sdpa=use_sdpa,
        delta_frames=delta_frames,
        clip_embed_dim=512,
        action_transformer=True,
    )


    # -- make data transforms
    if mask_type == 'multiblock3d':
        logger.info('Initializing basic multi-block mask')
        mask_collator = MB3DMaskCollator(
            crop_size=crop_size,
            num_frames=num_frames,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            cfgs_mask=cfgs_mask)
    elif mask_type == 'random_p':
        logger.info('Initializing random p multi-block mask')
        mask_collator = RandomPMaskCollator(
            crop_size=crop_size,
            num_frames=num_frames,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            cfgs_mask=cfgs_mask)
    else:
        logger.info('Initializing random tube mask')
        mask_collator = TubeMaskCollator(
            crop_size=crop_size,
            num_frames=num_frames,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            cfgs_mask=cfgs_mask)
    transform = make_transforms(
        random_horizontal_flip=True,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size)

    # -- init data-loaders/samplers
    (unsupervised_loader,
     unsupervised_sampler) = init_data(
         data=dataset_type,
         root_path=dataset_paths,
         batch_size=batch_size,
         training=True,
         clip_len=num_frames,
         frame_sample_rate=sampling_rate,
         filter_short_videos=filter_short_videos,
         decode_one_clip=decode_one_clip,
         duration=duration,
         num_clips=num_clips,
         transform=transform,
         datasets_weights=datasets_weights,
         collator=mask_collator,
         num_workers=num_workers,
         world_size=world_size,
         pin_mem=pin_mem,
         rank=rank,
         log_dir=folder if log_resource_util_data else None)
    try:
        _dlen = len(unsupervised_loader)
    except Exception:  # Different interface for webdataset
        _dlen = unsupervised_loader.num_batches
    if ipe is None:
        ipe = _dlen
    logger.info(f'iterations per epoch/dataest length: {ipe}/{_dlen}')

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        action_model=action_model,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        iterations_per_epoch=ipe,
        warmup=warmup,
        num_epochs=num_epochs,
        ipe_scale=ipe_scale,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps)
    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    action_model = DistributedDataParallel(action_model, static_graph=True)
    for p in encoder.parameters():
        p.requires_grad = False
    for p in predictor.parameters():
        p.requires_grad = False
    for p in action_model.parameters():
        p.requires_grad = False
        
        
    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_embedding_dim = clip_model.config.hidden_size
        
    # -- momentum schedule
    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
                          for i in range(int(ipe*num_epochs*ipe_scale)+1))

    start_epoch = 0
    # -- load training checkpoint
    if load_model or os.path.exists(latest_path):
        (
            encoder,
            predictor,
            target_encoder,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler)
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            mask_collator.step()

    logger.info('Initializing loader...')
    loader = iter(unsupervised_loader)

    if skip_batches > 0:
        logger.info(f'Skip {skip_batches} batches')
        unsupervised_sampler.set_epoch(start_epoch)
        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f'Skip {itr}/{skip_batches} batches')
            try:
                udata = next(loader)
            except Exception:
                loader = iter(unsupervised_loader)
                udata = next(loader)

    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        diffusion_loss_meter = AverageMeter()
        mask_meters = [AverageMeter() for _ in range(len(cfgs_mask))]

        for itr in range(ipe):
            itr_start_time = time.time()

            try:
                udata, masks_enc, masks_pred, p = next(loader)
            except Exception:
                logger.info('Exhausted data loaders. Refreshing...')
                loader = iter(unsupervised_loader)
                udata, masks_enc, masks_pred, p = next(loader)
            assert len(masks_enc) == len(masks_pred), \
                'Currently require num encoder masks = num predictor masks'

            def load_clips():
                # -- unsupervised video clips
                # Put each clip on the GPU and concatenate along batch
                # dimension
                clips = torch.cat([u.to(device, non_blocking=True) for u in udata[0]], dim=0)
                # get labels from udata
                labels = udata[1]
                labels = [' '.join(l.split('_')) for l in labels]
                
                input_ids = clip_tokenizer(labels, padding=True, return_tensors='pt').input_ids.to(device)

                # Get text embeddings
                with torch.no_grad():
                    text_embeddings = clip_model(input_ids).last_hidden_state  # [B, seq_len, D]
                text_embeddings = text_embeddings.mean(dim=1, keepdim=True)  # [B, 1, D]


                # Put each mask-enc/mask-pred pair on the GPU and reuse the
                # same mask pair for each clip
                _masks_enc, _masks_pred = [], []
                for _me, _mp in zip(masks_enc, masks_pred):
                    _me = _me.to(device, non_blocking=True)
                    _mp = _mp.to(device, non_blocking=True)
                    _me = repeat_interleave_batch(_me, batch_size, repeat=num_clips)
                    _mp = repeat_interleave_batch(_mp, batch_size, repeat=num_clips)
                    _masks_enc.append(_me)
                    _masks_pred.append(_mp)

                return (clips, _masks_enc, _masks_pred, text_embeddings)
            clips, masks_enc, masks_pred, text_embeddings = load_clips()

            for _i, m in enumerate(mask_meters):
                m.update(masks_enc[_i][0].size(-1))

            def forward_pass():
                def forward_target(c):
                    """
                    Returns tensor h of shape [B, N, D], where B is batch size,
                    N is number of tokens, D is embedding dimension.
                    """
                    with torch.no_grad():
                        h = target_encoder(c)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim  [B, N, D]
                        # -- create targets (masked regions of h)
                        h = apply_masks(h, masks_pred, concat=False)
                        h = h[0]  # Assuming single mask for simplicity
                        return h

                # Step 1. Forward
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    h = forward_target(clips)  # h: [B, N, D]

                    # Extract frames from clips
                    B, C, num_frames, H, W = clips.shape
                    x_0 = clips.permute(0, 2, 1, 3, 4).reshape(B * num_frames, C, H, W)  # [B*num_frames, C, H, W]

                    # Prepare conditioning embeddings
                    h_pooled = h.mean(dim=1)  # [B, D]
                    cond = h_pooled.unsqueeze(1).repeat(1, num_frames, 1).reshape(B * num_frames, -1)  # [B*num_frames, D]

                    # Sample random time steps for each image in the batch
                    t = torch.randint(0, timesteps, (B * num_frames,), device=device).long()

                    # Generate noisy images and the noise used
                    x_t, noise = forward_diffusion_sample(x_0, t, alpha_bar_sqrt, one_minus_alpha_bar_sqrt)

                    # Predict the noise using the diffusion model
                    noise_pred = diffusion_model(x_t, t, cond)

                    # Compute loss between the predicted noise and the actual noise
                    diffusion_loss = F.mse_loss(noise_pred, noise)

                # Backpropagation and optimization step
                diffusion_optimizer.zero_grad()
                diffusion_scaler.scale(diffusion_loss).backward()
                diffusion_scaler.step(diffusion_optimizer)
                diffusion_scaler.update()

                return diffusion_loss.item()
        
            diffusion_loss = forward_pass()
            diffusion_loss_meter.update(diffusion_loss)
            
            if itr % log_freq == 0 and rank == 0:
                            logger.info(f'Epoch [{epoch + 1}/{num_epochs}], Iteration [{itr + 1}/{ipe}], '
                                        f'Diffusion Loss: {diffusion_loss_meter.avg:.5f}')

        # Save checkpoint
        if rank == 0 and (epoch + 1) % checkpoint_freq == 0:
            checkpoint = {
                'epoch': epoch + 1,
                'diffusion_model_state_dict': diffusion_model.state_dict(),
                'diffusion_optimizer_state_dict': diffusion_optimizer.state_dict(),
                'diffusion_scaler_state_dict': diffusion_scaler.state_dict(),
            }
            torch.save(checkpoint, os.path.join(folder, f'diffusion_model_epoch_{epoch + 1}.pth.tar'))




# Usage example:
# Assume you have a DataLoader 'dataloader' that provides batches with 'images' and 'video_embeddings'
# 'vjepa_model' is your frozen V-JEPA model used to generate 'video_embeddings'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timesteps = 1000  # Number of diffusion steps

# Initialize the conditional diffusion model
model = ConditionalUNet(image_channels=3, cond_channels=768).to(device)

# Set up the optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Train the diffusion model
train_diffusion_model(model, dataloader, optimizer, epochs=10, device=device, timesteps=timesteps)