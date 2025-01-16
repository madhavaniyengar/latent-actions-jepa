# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

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

# --
log_timings = True
log_freq = 10
checkpoint_freq = 1
# --

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True


logger = get_logger(__name__)


class TokenLearner(nn.Module):
    """
    A minimal Token Learner module:
      - input shape: B x N x D   (B=batch, N=#tokens, D=dim)
      - output shape: B x K x D  (K < N, i.e., fewer learned tokens)
    """
    def __init__(self, embed_dim, num_output_tokens=8):
        super(TokenLearner, self).__init__()
        self.num_output_tokens = num_output_tokens
        
        # Example: 1D conv-based projection to get 'attention' over tokens
        # (You can replace this with MLP or other parametric forms as you wish)
        self.attn_conv = nn.Sequential(
            nn.Conv1d(in_channels=embed_dim,
                      out_channels=embed_dim // 4,
                      kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels=embed_dim // 4,
                      out_channels=num_output_tokens,
                      kernel_size=1)
        )

    def forward(self, x):
        """
        Args:
          x: B x N x D
        Returns:
          out: B x K x D
        """
        B, N, D = x.shape
        # We apply the conv along the token dimension, so transpose to B x D x N
        x_t = x.transpose(1, 2)  # now B x D x N

        # attn_map: B x K x N (K = num_output_tokens)
        attn_map = self.attn_conv(x_t)       # shape (B, K, N)
        attn_map = attn_map.softmax(dim=-1)  # softmax over the token dimension

        # Weighted sum over tokens:
        #   - x_t is (B, D, N)
        #   - attn_map is (B, K, N)
        # We want out to be (B, K, D).
        # "einops" or manual matmul:
        out = torch.einsum("bdn,bkn->bkd", x_t, attn_map)
        
        return out


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

    # ----------------------------------------------------------------------- #
    # ----------------------------------------------------------------------- #

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
    
    eval_log_file = os.path.join(folder, f'{tag}_eval_r{rank}.csv')
    eval_csv_logger = CSVLogger(
        eval_log_file,
        ('%d', 'epoch'),
        ('%.5f', 'eval_loss'),
        ('%.5f', 'eval_loss_jepa'),
        ('%.5f', 'eval_loss_reg'),
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
    
    # right after you initialize predictor, action_model, etc.

    # Suppose we want 8 tokens out from each frame's set of tokens
    num_output_tokens = 8
    token_learner = TokenLearner(embed_dim=action_dim,  # or whatever dimension your tokens have
                                num_output_tokens=num_output_tokens).to(device)

    # Optionally wrap it in DistributedDataParallel if you want
    token_learner = DistributedDataParallel(token_learner)
    

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
    
    eval_batch_size = 4  # for instance, use a smaller batch
    (eval_loader,
     eval_sampler) = init_data(
         data=dataset_type,
         root_path=eval_dataset_paths,
         batch_size=eval_batch_size,
         training=False,  # disable random sampling, shuffling, etc.
         clip_len=num_frames,
         frame_sample_rate=sampling_rate,
         filter_short_videos=filter_short_videos,
         decode_one_clip=decode_one_clip,
         duration=duration,
         num_clips=num_clips,
         transform=transform,
         datasets_weights=None,  # usually not needed for evaluation
         collator=mask_collator,
         num_workers=0,
         world_size=1,    # often just evaluate on a single GPU
         pin_mem=False,
         rank=0,
         log_dir=None
    )

    # -- init optimizer and scheduler
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        action_model=action_model,
        token_learner=token_learner,
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
    token_learner = DistributedDataParallel(token_learner, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
    for p in target_encoder.parameters():
        p.requires_grad = False
    for p in encoder.parameters():
        p.requires_grad = True
    for p in predictor.parameters():
        p.requires_grad = True
    for p in action_model.parameters():
        p.requires_grad = True
    for p in token_learner.parameters():
        p.requires_grad = True
        
        
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

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
            'encoder': encoder.state_dict(),
            'predictor': predictor.state_dict(),
            'action_model': action_model.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'target_encoder': target_encoder.state_dict(),
            'epoch': epoch,
            'loss': loss_meter.avg,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': lr,
        }
        try:
            torch.save(save_dict, path)
        except Exception as e:
            logger.info(f'Encountered exception when saving checkpoint: {e}')
            
    

    logger.info('Initializing loader...')
    
    logger.info(f'len unsupervised loader, {len(unsupervised_loader)}')

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

    def evaluate(
        encoder, predictor, action_model, target_encoder,
        data_loader, device, clip_model, clip_tokenizer, dtype, mixed_precision
    ):
        """Run evaluation (forward pass only) on a small subset."""
        encoder.eval()
        predictor.eval()
        action_model.eval()
        target_encoder.eval()

        eval_loss_meter = AverageMeter()
        eval_jepa_meter = AverageMeter()
        eval_reg_meter = AverageMeter()

        with torch.no_grad():
            for batch_idx, (udata, masks_enc, masks_pred, p) in enumerate(data_loader):
                # Move data to GPU
                clips = torch.cat([u.to(device, non_blocking=True) for u in udata[0]], dim=0)
                labels = udata[1]
                labels = [' '.join(l.split('_')) for l in labels]
                input_ids = clip_tokenizer(labels, padding=True, return_tensors='pt').input_ids.to(device)

                with torch.no_grad():
                    text_embeddings = clip_model(input_ids).last_hidden_state
                text_embeddings = text_embeddings.mean(dim=1, keepdim=True)

                # Prepare masks
                _masks_enc, _masks_pred = [], []
                for _me, _mp in zip(masks_enc, masks_pred):
                    _me = _me.to(device, non_blocking=True)
                    _mp = _mp.to(device, non_blocking=True)
                    _me = repeat_interleave_batch(_me, B=len(udata[0]), repeat=num_clips)
                    _mp = repeat_interleave_batch(_mp, B=len(udata[0]), repeat=num_clips)
                    _masks_enc.append(_me)
                    _masks_pred.append(_mp)

                # forward passes
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    # target encoding (no gradient)
                    h = target_encoder(clips)
                    h = F.layer_norm(h, (h.size(-1),)) 
                    h_splits = apply_masks(h, _masks_pred, concat=False)

                    # context encoding
                    z_splits = encoder(clips, _masks_enc)

                    # apply action model
                    per_frame_embeddings = [
                        seq.reshape(seq.size(0), pi, seq.size(1)//pi, seq.size(-1)) 
                        for seq, pi in zip(z_splits, p)
                    ]
                    per_frame_embeddings = [pfe.mean(dim=2) for pfe in per_frame_embeddings]
                    action_model_input = [
                        torch.cat([text_embeddings.repeat(1, 1, 2), pfe], dim=1) 
                        for pfe in per_frame_embeddings
                    ]
                    action_cls_tokens = [action_model(ami) for ami in action_model_input]

                    # predictor
                    z_pred_splits = predictor(z_splits, h_splits, _masks_enc, _masks_pred, action_cls_tokens)

                    # compute jepa loss
                    loss_jepa = 0.
                    for zi, hi in zip(z_pred_splits, h_splits):
                        loss_jepa += torch.mean(torch.abs(zi - hi)**loss_exp) / loss_exp
                    loss_jepa /= len(_masks_pred)

                    # compute regularization
                    pstd_z = sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z_pred_splits]) / len(z_pred_splits)
                    loss_reg = torch.mean(F.relu(1. - pstd_z))

                    loss_eval = loss_jepa + reg_coeff * loss_reg

                eval_loss_meter.update(loss_eval.item())
                eval_jepa_meter.update(loss_jepa.item())
                eval_reg_meter.update(loss_reg.item())

        # restore train mode
        encoder.train()
        predictor.train()
        action_model.train()
        target_encoder.train()

        return eval_loss_meter.avg, eval_jepa_meter.avg, eval_reg_meter.avg
    
    # -- TRAINING LOOP
    for epoch in range(start_epoch, num_epochs):
        logger.info('Epoch %d' % (epoch + 1))

        # -- update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)

        loss_meter = AverageMeter()
        input_var_meter = AverageMeter()
        input_var_min_meter = AverageMeter()
        jepa_loss_meter = AverageMeter()
        reg_loss_meter = AverageMeter()
        mask_meters = [AverageMeter() for _ in range(len(cfgs_mask))]
        gpu_time_meter = AverageMeter()
        wall_time_meter = AverageMeter()

        for itr in range(ipe):
            print('epoch', epoch, 'itr', itr)
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

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
                # --

                def forward_target(c):
                    """
                    Returns list of tensors of shape [B, N, D], one for each
                    mask-pred.
                    """
                    with torch.no_grad():
                        h = target_encoder(c)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim  [B, N, D]
                        # -- create targets (masked regions of h)
                        h = apply_masks(h, masks_pred, concat=False)
                        return h


                def forward_context(c, text_embeddings, p):
                    """
                    Returns list of tensors of shape [B, N, D], one for each mask-pred.
                    """
                    z = encoder(c, masks_enc)   # [B, N_z, D]

                    per_frame_embeddings = [
                        seq.reshape(seq.size(0), pi, seq.size(1)//pi, seq.size(-1)) 
                        for seq, pi in zip(z, p)
                    ]
                    # Instead of pfe.mean(dim=2), apply Token Learner to each [B, (N_z/frames), D].
                    # But note that each pfe is shape: B x #frames x #patches_per_frame x D
                    # or B x (#tokens) x D, depending on how you chunk it.

                    per_frame_embeddings = []
                    for seq, pi in zip(z, p):
                        # shape is [B, N, D] if each frame is flattened in time dimension
                        # or [B, pi, (N/pi), D] if you truly separated frames
                        # For example, if we do the same reshape approach:
                        reshaped = seq.reshape(
                            seq.size(0),  # B
                            pi,           # frames
                            seq.size(1) // pi,  
                            seq.size(-1)
                        )  # => B x frames x (tokens_per_frame) x D
                        
                        # Suppose you just want to flatten frames x tokens_per_frame => total tokens
                        # shape => B x (frames * tokens_per_frame) x D
                        reshaped = reshaped.view(reshaped.size(0),
                                                reshaped.size(1) * reshaped.size(2),
                                                reshaped.size(3))
                        
                        # Now pass through TokenLearner to get B x K x D
                        learned_tokens = token_learner(reshaped)
                        
                        # You can keep it as is and feed into the action model,
                        # or you might do something else, e.g. optionally add a final pooling
                        # or classification token. For now, let's just store:
                        per_frame_embeddings.append(learned_tokens)

                    # concat text embeddings with per_frame_embeddings
                    action_model_input = [
                        torch.cat([text_embeddings.repeat(1, 1, 2), pfe], dim=1) 
                        for pfe in per_frame_embeddings
                    ]
                    action_cls_tokens = [action_model(ami) for ami in action_model_input]

                    z = predictor(z, h, masks_enc, masks_pred, action_cls_tokens)
                    return z

                def loss_fn(z, h):
                    loss = 0.
                    # Compute loss and accumulate for each mask-enc/mask-pred pair
                    for zi, hi in zip(z, h):
                        loss += torch.mean(torch.abs(zi - hi)**loss_exp) / loss_exp
                    loss /= len(masks_pred)
                    return loss

                def reg_fn(z):
                    return sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z]) / len(z)

                # Step 1. Forward
                loss_jepa, loss_reg = 0., 0.
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    h = forward_target(clips)
                    z = forward_context(clips, text_embeddings, p)
                    loss_jepa = loss_fn(z, h)  # jepa prediction loss
                    pstd_z = reg_fn(z)  # predictor variance across patches
                    loss_reg += torch.mean(F.relu(1.-pstd_z))
                loss = loss_jepa + reg_coeff * loss_reg

                # Step 2. Backward & step
                _enc_norm, _pred_norm = 0., 0.
                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if (epoch > warmup) and (clip_grad is not None):
                    _enc_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
                    _pred_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                grad_stats = grad_logger(encoder.named_parameters())
                grad_stats.global_norm = float(_enc_norm)
                grad_stats_pred = grad_logger(predictor.named_parameters())
                grad_stats_pred.global_norm = float(_pred_norm)
                optimizer.zero_grad()
                optim_stats = adamw_logger(optimizer)

                # Step 3. momentum update of target encoder
                m = next(momentum_scheduler)
                with torch.no_grad():
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)

                return (
                    float(loss),
                    float(loss_jepa),
                    float(loss_reg),
                    _new_lr,
                    _new_wd,
                    grad_stats,
                    grad_stats_pred,
                    optim_stats,
                )
            (loss, loss_jepa, loss_reg, _new_lr, _new_wd, grad_stats, grad_stats_pred, optim_stats,), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.
            loss_meter.update(loss)
            input_var = float(AllReduce.apply(clips.view(clips.shape[0], -1).var(dim=1).mean(dim=0)))
            input_var_min = float(AllReduce.apply(torch.min(clips.view(clips.shape[0], -1).var(dim=1))))
            input_var_meter.update(input_var)
            input_var_min_meter.update(input_var_min)
            jepa_loss_meter.update(loss_jepa)
            reg_loss_meter.update(loss_reg)
            gpu_time_meter.update(gpu_etime_ms)
            wall_time_meter.update(iter_elapsed_time_ms)

            # -- Logging
            def log_stats():
                csv_logger.log(
                    epoch + 1,
                    itr,
                    loss,
                    loss_jepa,
                    loss_reg,
                    grad_stats.global_norm,
                    grad_stats_pred.global_norm,
                    gpu_etime_ms,
                    iter_elapsed_time_ms)
                if (itr % log_freq == 0) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        '[%d, %5d] loss: %.3f | p%.3f r%.3f | '
                        'input_var: %.3f %.3f | '
                        'masks: %s '
                        '[wd: %.2e] [lr: %.2e] '
                        '[mem: %.2e] '
                        '[gpu: %.1f ms]'
                        '[wall: %.1f ms]'
                        % (epoch + 1, itr,
                           loss_meter.avg,
                           jepa_loss_meter.avg,
                           reg_loss_meter.avg,
                           input_var_meter.avg,
                           input_var_min_meter.avg,
                           '[' + ', '.join(['%.1f' % m.avg for m in mask_meters]) + ']',
                           _new_wd,
                           _new_lr,
                           torch.cuda.max_memory_allocated() / 1024.0**2,
                           gpu_time_meter.avg,
                           wall_time_meter.avg))

                    if optim_stats is not None:
                        logger.info(
                            '[%d, %5d] first moment: %.2e [%.2e %.2e] second moment: %.2e [%.2e %.2e]'
                            % (epoch + 1, itr,
                               optim_stats.get('exp_avg').avg,
                               optim_stats.get('exp_avg').min,
                               optim_stats.get('exp_avg').max,
                               optim_stats.get('exp_avg_sq').avg,
                               optim_stats.get('exp_avg_sq').min,
                               optim_stats.get('exp_avg_sq').max))

                    if grad_stats is not None:
                        logger.info(
                            '[%d, %5d] enc_grad_stats: f/l[%.2e %.2e] mn/mx(%.2e, %.2e) %.2e'
                            % (epoch + 1, itr,
                               grad_stats.first_layer,
                               grad_stats.last_layer,
                               grad_stats.min,
                               grad_stats.max,
                               grad_stats.global_norm))

                    if grad_stats_pred is not None:
                        logger.info(
                            '[%d, %5d] pred_grad_stats: f/l[%.2e %.2e] mn/mx(%.2e, %.2e) %.2e'
                            % (epoch + 1, itr,
                               grad_stats_pred.first_layer,
                               grad_stats_pred.last_layer,
                               grad_stats_pred.min,
                               grad_stats_pred.max,
                               grad_stats_pred.global_norm))
            log_stats()
            assert not np.isnan(loss), 'loss is nan'

        # -- Save Checkpoint
        logger.info('avg. loss %.3f' % loss_meter.avg)
        
        
        if rank == 0:  # typically only do full eval on one process
            eval_loss_avg, eval_loss_jepa_avg, eval_loss_reg_avg = evaluate(
                encoder, predictor, action_model, target_encoder,
                eval_loader, device, clip_model, clip_tokenizer,
                dtype, mixed_precision
            )
            # Log to console
            logger.info(
                f"[Eval @ epoch {epoch+1}] "
                f"eval_loss: {eval_loss_avg:.3f}, "
                f"eval_jepa: {eval_loss_jepa_avg:.3f}, "
                f"eval_reg: {eval_loss_reg_avg:.3f}"
            )
            # Log to eval CSV
            eval_csv_logger.log(
                epoch + 1,
                eval_loss_avg,
                eval_loss_jepa_avg,
                eval_loss_reg_avg
            )
        
        # -- Save Last
        if epoch % checkpoint_freq == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_every_file = f'{tag}-e{epoch}.pth.tar'
                save_every_path = os.path.join(folder, save_every_file)
                save_checkpoint(epoch + 1, save_every_path)
