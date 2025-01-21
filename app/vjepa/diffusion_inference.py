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
import matplotlib.pyplot as plt
import random
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
        
def p_sample(
    model: torch.nn.Module,
    x: torch.Tensor,
    t: torch.Tensor,
    cond: torch.Tensor,
    alphas: torch.Tensor,
    alphas_bar: torch.Tensor,
    betas: torch.Tensor
) -> torch.Tensor:
    """
    Single reverse diffusion step: x_t -> x_{t-1}
    
    model:        the ConditionalUNet that predicts noise
    x:            current noisy sample x_t  of shape [B, C, H, W]
    t:            time indices for each sample in the batch (shape [B])
    cond:         condition embeddings, shape [B, cond_dim]
    alphas:       1 - betas
    alphas_bar:   cumulative product of alphas
    betas:        noise schedule
    """

    # Predict the noise \epsilon using the model
    eps_pred = model(x, t, cond)  # same shape as x

    # Gather parameters for this timestep
    alpha_t     = alphas[t].view(-1, 1, 1, 1)       # [B, 1, 1, 1]
    alpha_bar_t = alphas_bar[t].view(-1, 1, 1, 1)   # [B, 1, 1, 1]
    beta_t      = betas[t].view(-1, 1, 1, 1)        # [B, 1, 1, 1]

    # Equation from DDPM paper:
    # x_{t-1} = 1/sqrt(alpha_t) * ( x_t - ( (1 - alpha_t)/sqrt(1 - alpha_bar_t) ) * eps_pred )  +  sigma_t * z
    # where sigma_t = sqrt(beta_t) (original DDPM choice).

    # Remove predicted noise
    x_no_noise = (
        1.0 / torch.sqrt(alpha_t)
    ) * (
        x - ( (1.0 - alpha_t) / torch.sqrt(1.0 - alpha_bar_t) ) * eps_pred
    )

    # Random noise z ~ N(0, I), except at t=0 (no noise to add)
    z = torch.randn_like(x) if t.min() > 0 else torch.zeros_like(x)
    
    # Scale by the variance
    sigma_t = torch.sqrt(beta_t)
    
    # Final x_{t-1}
    x_prev = x_no_noise + sigma_t * z
    return x_prev


@torch.no_grad()
def generate_images(
    model: torch.nn.Module,
    cond: torch.Tensor,
    device: torch.device,
    image_shape: tuple = (3, 64, 64),
    num_samples: int = 8,
    timesteps: int = 1000,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
) -> torch.Tensor:
    """
    Runs the reverse diffusion process to generate images conditioned on `cond`.

    model:         your ConditionalUNet that predicts the noise
    cond:          condition embeddings, shape [B, cond_dim]. B must match num_samples below.
    device:        CUDA or CPU device
    image_shape:   (C, H, W) for the output images
    num_samples:   how many images to generate
    timesteps:     total number of diffusion steps (T)
    beta_start:    schedule start for betas
    beta_end:      schedule end for betas

    Returns a tensor of generated images, shape [num_samples, C, H, W].
    """
    # ---------------------------------------------------------------------
    # 1) Prepare the noise schedule and relevant alpha terms
    # ---------------------------------------------------------------------
    betas = torch.linspace(beta_start, beta_end, timesteps, device=device)  # [T]
    alphas = 1.0 - betas                                                    # [T]
    alphas_bar = torch.cumprod(alphas, dim=0)                               # [T]

    # ---------------------------------------------------------------------
    # 2) Sample from Gaussian noise at t = T
    #    shape = [num_samples, C, H, W]
    # ---------------------------------------------------------------------
    x_t = torch.randn((num_samples,) + image_shape, device=device)

    # ---------------------------------------------------------------------
    # 3) Iteratively sample x_{t-1} from x_t (reverse diffusion)
    # ---------------------------------------------------------------------
    # Create a list of time indices in descending order, shape [T].
    # We'll broadcast them as needed inside `p_sample`.
    for i in reversed(range(timesteps)):
        # For each sample in the batch, create a time index t_i
        # with shape [B]; all equal to i for this step
        t_step = torch.full((num_samples,), i, device=device, dtype=torch.long)
        x_t = p_sample(model, x_t, t_step, cond, alphas, alphas_bar, betas)

    # By the end, x_t is x_0
    return x_t


"""
Usage example:

# Suppose you have:
#  1) diffusion_model: your ConditionalUNet instance (on GPU)
#  2) cond: a tensor [B, cond_dim] of condition embeddings, e.g. from text or video features
#  3) device: torch.device("cuda")

# Generate 8 images of size 64x64
generated = generate_images(
    model=diffusion_model,
    cond=cond,                      # shape [8, cond_dim] if you're generating 8 samples
    device=device,
    image_shape=(3, 64, 64),
    num_samples=8,
    timesteps=1000,
    beta_start=0.0001,
    beta_end=0.02,
)
# generated: [8, 3, 64, 64], range ~ [-1,1] or so. You can clamp/convert to a viewable format.
"""

def eval_diffusion_model(diffusion_model, 
                         encoder,
                         target_encoder,
                         device,
                         dataloader,
                         timesteps=1000,
                         num_samples=1,
                         image_shape=(3, 64, 64)):
    """
    Picks a random batch from `dataloader`, extracts the target embedding 
    from the (frozen) target_encoder, and uses `generate_images` to produce 
    a reconstruction. Displays a random sample from that batch side-by-side 
    with its ground truth image.
    """
    diffusion_model.eval()
    encoder.eval()
    target_encoder.eval()

    # 1) Retrieve one batch from the dataloader
    data_iter = iter(dataloader)
    try:
        udata, masks_enc, masks_pred, p = next(data_iter)
    except StopIteration:
        # If the dataloader is exhausted, re-init
        data_iter = iter(dataloader)
        udata, masks_enc, masks_pred, p = next(data_iter)

    # 2) Move the data to GPU if available
    clips = torch.cat([u.to(device, non_blocking=True) for u in udata[0]], dim=0)   # shape [B, C, T, H, W]
    B, C, T, H, W = clips.shape

    # If you have labels in udata[1], and possibly want them:
    # labels = udata[1]
    # For now we ignore them unless you have text-based conditioning.

    # 3) Extract embeddings from the target encoder (or from the predictor, if needed).
    #    Suppose we just want to get a single embedding vector [B, D] by averaging tokens.
    with torch.no_grad():
        h = target_encoder(clips)             # h shape: [B, N, D]
        h = F.layer_norm(h, (h.size(-1),))    # LN over feature dim
        h_pooled = h.mean(dim=1)             # [B, D]

    # 4) Use that embedding to generate images with the diffusion model.
    #    We can replicate each embedding for `T` frames if you want to compare per-frame,
    #    or just generate a single image per sample. Here, let's do one image per sample:
    cond = h_pooled  # shape [B, D]
    generated = generate_images(
        model=diffusion_model,
        cond=cond,                 # shape [B, D]
        device=device,
        image_shape=(C, H, W),     # match your ground-truth shapes
        num_samples=B,             # generate 1 image per example in the batch
        timesteps=timesteps,
        beta_start=0.0001,
        beta_end=0.02,
    ) 
    # `generated` shape: [B, C, H, W], typically in [-1, +1] or so

    # 5) Pick a random index in [0, B-1] to visualize
    idx = random.randint(0, B-1)
    gen_img = generated[idx].detach().cpu()
    gt_img  = clips[idx, :, 0, :, :].detach().cpu()  # for example, show the first frame T=0
                                                     # or you can show an average or something else

    # 6) Convert them to [0,1] range for display (assuming they're roughly in [-1,1])
    gen_img_disp = (gen_img - gen_img.min()) / (gen_img.max() - gen_img.min() + 1e-5)
    gt_img_disp  = (gt_img  - gt_img.min())  / (gt_img.max()  - gt_img.min()  + 1e-5)

    # 7) Plot side-by-side
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(gt_img_disp.permute(1, 2, 0).numpy())
    axes[0].set_title("Ground Truth (frame 0)")
    axes[0].axis("off")

    axes[1].imshow(gen_img_disp.permute(1, 2, 0).numpy())
    axes[1].set_title("Generated")
    axes[1].axis("off")

    plt.tight_layout()
    plt.savefig("figures/diffusion_sample.png")


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
    eval_dataset_paths = cfgs_data.get('eval_datasets', None)
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
    
    device = torch.device('cuda:0')

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #    # Initialize the diffusion model
    h_dim = action_dim  # Assuming pred_embed_dim is the dimension of h
    diffusion_model = ConditionalUNet(image_channels=3, cond_channels=h_dim).to(device)
    diffusion_model = DistributedDataParallel(diffusion_model, device_ids=[device.index])
    
    diffusion_ckpt_path = cfgs_meta.get('diffusion_checkpoint_path', None)
    if diffusion_ckpt_path and os.path.isfile(diffusion_ckpt_path):
        logger.info(f"Loading diffusion checkpoint from {diffusion_ckpt_path}")
        checkpoint = torch.load(diffusion_ckpt_path)
        # Load state_dict for the model
        diffusion_model.load_state_dict(checkpoint['diffusion_model_state_dict'])
        logger.info(f"Successfully loaded diffusion checkpoint from: {diffusion_ckpt_path}")
    else:
        logger.info("No valid diffusion checkpoint path provided or file does not exist. Skipping load.")

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
    target_encoder = copy.deepcopy(encoder)


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
        token_learner=None,
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
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False
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

    eval_diffusion_model(
            diffusion_model=diffusion_model,
            encoder=encoder,
            target_encoder=target_encoder,
            device=device,
            dataloader=unsupervised_loader,  # your chosen DataLoader
            timesteps=1000,
            num_samples=1,
            image_shape=(3, 64, 64)
        )