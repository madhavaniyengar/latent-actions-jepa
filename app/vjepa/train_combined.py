# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os
import copy
import time
import numpy as np

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

import wandb  # <-- New import for Weights & Biases

from src.datasets.data_manager import init_data
from src.masks.random_tube import MaskCollator as TubeMaskCollator
from src.masks.multiblock3d import MaskCollator as MB3DMaskCollator
from src.masks.random_p import MaskCollator as RandomPMaskCollator
from src.masks.utils import apply_masks
from src.utils.distributed import init_distributed, AllReduce
from src.utils.logging import (
    gpu_timer,
    grad_logger,
    adamw_logger,
    AverageMeter
)
from src.utils.tensors import repeat_interleave_batch

from app.vjepa.utils import (
    load_checkpoint,
    init_video_model,
    init_opt,
)
from app.vjepa.transforms import make_transforms
from transformers import CLIPTokenizer, CLIPTextModel
from models.TokenLearner import TokenLearner

# --------------------------------------------------------------------
# Original script constants
log_timings = True
log_freq = 10
checkpoint_freq = 1
# --------------------------------------------------------------------

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

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
    which_dtype = cfgs_meta.get('dtype', 'float32')

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
    delta_frames = cfgs_model.get('delta_frames', 4)  
    token_learning_model = cfgs_model.get('token_learning_model', 'tokenlearner')

    # -- DATA
    cfgs_data = args.get('data')
    dataset_type = cfgs_data.get('dataset_type', 'videodataset')
    mask_type = cfgs_data.get('mask_type', 'multiblock3d')
    dataset_paths = cfgs_data.get('datasets', [])
    eval_dataset_paths = cfgs_data.get('eval_datasets', None)
    datasets_weights = cfgs_data.get('datasets_weights', None)
    if datasets_weights is not None:
        assert len(datasets_weights) == len(dataset_paths), \
            'Must have one sampling weight specified for each dataset'
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
    print(f'[INFO] Initialized (rank/world-size) {rank}/{world_size}')

    # -- set device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)

    # -- log/checkpointing paths
    latest_file = f'{tag}-latest.pth.tar'
    latest_path = os.path.join(folder, latest_file)
    load_path = None
    if load_model:
        # If user specifically provided a checkpoint to read
        if r_file is not None:
            load_path = os.path.join(folder, r_file)
        else:
            load_path = latest_path

        if not os.path.exists(load_path):
            load_path = None
            load_model = False

    # -----------------------------------------------------------------------
    #  Initialize Weights & Biases
    # -----------------------------------------------------------------------
    if rank == 0:  # Only init W&B on rank=0
        wandb.init(
            project="jepa-latent-actions",
            name=tag,
            config=args,
            entity="miyen",  # if needed
        )
    # -----------------------------------------------------------------------

    print(f'[INFO] Using dtype={dtype} (mixed_precision={mixed_precision})')

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
    
    if token_learning_model == 'tokenlearner':
        num_output_tokens = 8
        token_learner = TokenLearner(
            embed_dim=action_dim,
            num_output_tokens=num_output_tokens
        ).to(device)
        token_learner = DistributedDataParallel(token_learner)
    else:

        token_learner = nn.ModuleList([
            
        ])
        
        token_learner = nn.AvgPool1d(kernel_size=action_dim, stride=action_dim).to(device)

    # -- mask collator
    if mask_type == 'multiblock3d':
        print('[INFO] Initializing basic multi-block mask')
        mask_collator = MB3DMaskCollator(
            crop_size=crop_size,
            num_frames=num_frames,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            cfgs_mask=cfgs_mask)
    elif mask_type == 'random_p':
        print('[INFO] Initializing random p multi-block mask')
        mask_collator = RandomPMaskCollator(
            crop_size=crop_size,
            num_frames=num_frames,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            cfgs_mask=cfgs_mask)
    else:
        print('[INFO] Initializing random tube mask')
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
         log_dir=folder if log_resource_util_data else None
    )

    # If ipe not set, default to len(loader)
    try:
        _dlen = len(unsupervised_loader)
    except Exception:
        _dlen = unsupervised_loader.num_batches  # WebDataset scenario
    if ipe is None:
        ipe = _dlen
    print(f'[INFO] Iterations per epoch: ipe={ipe}, dataset length={_dlen}')

    eval_batch_size = 4
    (eval_loader,
     eval_sampler) = init_data(
         data=dataset_type,
         root_path=eval_dataset_paths,
         batch_size=eval_batch_size,
         training=False,
         clip_len=num_frames,
         frame_sample_rate=sampling_rate,
         filter_short_videos=filter_short_videos,
         decode_one_clip=decode_one_clip,
         duration=duration,
         num_clips=num_clips,
         transform=transform,
         datasets_weights=None,
         collator=mask_collator,
         num_workers=0,
         world_size=1,
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
        eps=eps
    )

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
    momentum_scheduler = (
        ema[0] + i*(ema[1]-ema[0])/(ipe*num_epochs*ipe_scale)
        for i in range(int(ipe*num_epochs*ipe_scale)+1)
    )

    start_epoch = 0
    # -- load training checkpoint (if available)
    if load_model and (load_path is not None):
        (encoder,
         predictor,
         target_encoder,
         optimizer,
         scaler,
         start_epoch) = load_checkpoint(
             r_path=load_path,
             encoder=encoder,
             predictor=predictor,
             target_encoder=target_encoder,
             opt=optimizer,
             scaler=scaler
        )
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
            print(f"[INFO] Checkpoint saved to {path}")
        except Exception as e:
            print(f"[WARN] Exception when saving checkpoint: {e}")

    print('[INFO] Initializing loader...')
    try:
        print(f'[INFO] Unsupervised loader length = {len(unsupervised_loader)}')
    except Exception:
        pass

    # Optionally skip certain number of batches
    if skip_batches > 0:
        print(f'[INFO] Skipping {skip_batches} batches')
        unsupervised_sampler.set_epoch(start_epoch)
        loader_iter = iter(unsupervised_loader)
        for itr in range(skip_batches):
            if itr % 10 == 0:
                print(f'[INFO] Skipped {itr}/{skip_batches} batches so far')
            try:
                udata = next(loader_iter)
            except StopIteration:
                loader_iter = iter(unsupervised_loader)
                udata = next(loader_iter)

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
            for batch_idx, batch_datapoint in enumerate(data_loader):
                if mask_type == 'multiblock3d':
                    udata, masks_enc, masks_pred = batch_datapoint
                    p = [num_frames // 2]  * batch_size
                elif mask_type == 'random_p':
                    udata, masks_enc, masks_pred, p = batch_datapoint
                    
                clips = torch.cat([u.to(device, non_blocking=True) for u in udata[0]], dim=0)
                labels = udata[1]
                labels = [' '.join(l.split('_')) for l in labels]
                input_ids = clip_tokenizer(labels, padding=True, return_tensors='pt').input_ids.to(device)

                # get text embeddings
                with torch.no_grad():
                    text_embeddings = clip_model(input_ids).last_hidden_state  # [B, seq_len, D]
                text_embeddings = text_embeddings.mean(dim=1, keepdim=True)   # [B, 1, D]

                # Prepare the masks
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
                    per_frame_embeddings = []
                    for seq, pi in zip(z_splits, p):
                        reshaped = seq.reshape(
                            seq.size(0),
                            pi,
                            seq.size(1)//pi,
                            seq.size(-1)
                        )  # B x frames x (tokens_per_frame) x D
                        reshaped = reshaped.view(reshaped.size(0), -1, reshaped.size(-1))  # flatten
                        learned_tokens = token_learner(reshaped)
                        per_frame_embeddings.append(learned_tokens)

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

    # --------------------------------------------------------------------
    # Training loop
    # --------------------------------------------------------------------
    loader_iter = iter(unsupervised_loader)
    global_step = start_epoch * ipe

    for epoch in range(start_epoch, num_epochs):
        print(f'[INFO] Starting epoch {epoch + 1}/{num_epochs}')

        # update distributed-data-loader epoch
        unsupervised_sampler.set_epoch(epoch)

        # Meters
        loss_meter = AverageMeter()
        input_var_meter = AverageMeter()
        input_var_min_meter = AverageMeter()
        jepa_loss_meter = AverageMeter()
        reg_loss_meter = AverageMeter()
        mask_meters = [AverageMeter() for _ in range(len(cfgs_mask))]
        gpu_time_meter = AverageMeter()
        wall_time_meter = AverageMeter()

        for itr in range(ipe):
            itr_start_time = time.time()
            global_step = epoch * ipe + itr

            # Refresh data if needed
            try:
                loader_data = next(loader_iter)
            except Exception:
                print('[INFO] Data loader exhausted. Re-initializing iterator...')
                loader_iter = iter(unsupervised_loader)
                loader_data = next(loader_iter)
                
            if mask_type == 'multiblock3d':
                udata, masks_enc, masks_pred = loader_data
                p = None
            elif mask_type == 'random_p':
                udata, masks_enc, masks_pred, p = loader_data
            # Quick assertion
            assert len(masks_enc) == len(masks_pred), \
                'Need num encoder masks == num predictor masks'

            def load_clips():
                # Combine the multiple video clips in `udata[0]`
                clips = torch.cat([u.to(device, non_blocking=True) for u in udata[0]], dim=0)
                labels = udata[1]
                labels = [' '.join(l.split('_')) for l in labels]
                input_ids = clip_tokenizer(labels, padding=True, return_tensors='pt').input_ids.to(device)

                with torch.no_grad():
                    text_embeddings = clip_model(input_ids).last_hidden_state
                text_embeddings = text_embeddings.mean(dim=1, keepdim=True)

                _masks_enc, _masks_pred = [], []
                for _me, _mp in zip(masks_enc, masks_pred):
                    _me = _me.to(device, non_blocking=True)
                    _mp = _mp.to(device, non_blocking=True)
                    _me = repeat_interleave_batch(_me, batch_size, repeat=num_clips)
                    _mp = repeat_interleave_batch(_mp, batch_size, repeat=num_clips)
                    _masks_enc.append(_me)
                    _masks_pred.append(_mp)

                return clips, _masks_enc, _masks_pred, text_embeddings

            clips, masks_enc, masks_pred, text_embeddings = load_clips()

            for i_m, m_meter in enumerate(mask_meters):
                # Each masks_enc[i_m][0] has shape [B, <mask_dim>], so you can track a dimension or patch count
                m_meter.update(masks_enc[i_m][0].size(-1))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                def forward_target(c):
                    with torch.no_grad():
                        h = target_encoder(c)
                        h = F.layer_norm(h, (h.size(-1),))
                        # apply mask to create targets
                        h = apply_masks(h, masks_pred, concat=False)
                        return h

                def forward_context(c, h_splits, text_emb, p_splits):
                    z = encoder(c, masks_enc)
                    per_frame_embeddings = []
                    for seq, pi_ in zip(z, p_splits):
                        reshaped = seq.reshape(
                            seq.size(0),   # B
                            pi_,
                            seq.size(1)//pi_,
                            seq.size(-1)
                        )
                        reshaped = reshaped.view(reshaped.size(0), -1, reshaped.size(-1))
                        learned_tokens = token_learner(reshaped)
                        per_frame_embeddings.append(learned_tokens)

                    action_model_input = [
                        torch.cat([text_emb.repeat(1, 1, 2), pfe], dim=1)
                        for pfe in per_frame_embeddings
                    ]
                    action_cls_tokens = [action_model(ami) for ami in action_model_input]

                    z = predictor(z, h_splits, masks_enc, masks_pred, action_cls_tokens)
                    return z

                def loss_fn(z_splits, h_splits):
                    loss_val = 0.
                    for zi, hi in zip(z_splits, h_splits):
                        loss_val += torch.mean(torch.abs(zi - hi)**loss_exp) / loss_exp
                    loss_val /= len(masks_pred)
                    return loss_val

                def reg_fn(z_splits):
                    return sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z_splits]) / len(z_splits)

                # Forward
                loss_jepa_val, loss_reg_val = 0., 0.
                with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                    h_ = forward_target(clips)
                    if mask_type == 'multiblock3d': p = [num_frames // 2]  * batch_size
                    z_ = forward_context(clips, h_, text_embeddings, p)
                    loss_jepa_val = loss_fn(z_, h_)
                    pstd_z = reg_fn(z_)
                    loss_reg_val = torch.mean(F.relu(1. - pstd_z))

                loss_total = loss_jepa_val + reg_coeff * loss_reg_val

                # Backward
                enc_norm, pred_norm = 0., 0.
                if mixed_precision:
                    scaler.scale(loss_total).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss_total.backward()

                # Clip grads if needed
                if (epoch > warmup) and (clip_grad is not None):
                    enc_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
                    pred_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)

                # Optim step
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

                # Log param stats
                grad_stats_enc = grad_logger(encoder.named_parameters())
                grad_stats_enc.global_norm = float(enc_norm)
                grad_stats_pred = grad_logger(predictor.named_parameters())
                grad_stats_pred.global_norm = float(pred_norm)
                optim_stats = adamw_logger(optimizer)

                # EMA update of target encoder
                m = next(momentum_scheduler)
                with torch.no_grad():
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1. - m) * param_q.detach().data)

                return (
                    float(loss_total),
                    float(loss_jepa_val),
                    float(loss_reg_val),
                    _new_lr,
                    _new_wd,
                    grad_stats_enc,
                    grad_stats_pred,
                    optim_stats
                )

            (loss_val,
             loss_jepa_val,
             loss_reg_val,
             new_lr,
             new_wd,
             grad_stats_enc,
             grad_stats_pred,
             optim_stats), gpu_etime_ms = gpu_timer(train_step)

            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.
            loss_meter.update(loss_val)
            input_var = float(AllReduce.apply(clips.view(clips.shape[0], -1).var(dim=1).mean(dim=0)))
            input_var_min = float(AllReduce.apply(torch.min(clips.view(clips.shape[0], -1).var(dim=1))))
            input_var_meter.update(input_var)
            input_var_min_meter.update(input_var_min)
            jepa_loss_meter.update(loss_jepa_val)
            reg_loss_meter.update(loss_reg_val)
            gpu_time_meter.update(gpu_etime_ms)
            wall_time_meter.update(iter_elapsed_time_ms)

            # ---------------------------------------------------------
            # W&B logging for every iteration
            # ---------------------------------------------------------
            if rank == 0:
                wandb_dict = {
                    "epoch": epoch + 1,
                    "iteration": itr,
                    "global_step": global_step,
                    "loss": loss_val,
                    "loss_jepa": loss_jepa_val,
                    "loss_reg": loss_reg_val,
                    "enc_grad_norm": grad_stats_enc.global_norm,
                    "pred_grad_norm": grad_stats_pred.global_norm,
                    "gpu_time_ms": gpu_etime_ms,
                    "wall_time_ms": iter_elapsed_time_ms,
                    "input_var": input_var,
                    "input_var_min": input_var_min,
                    "lr": new_lr,
                    "weight_decay": new_wd,
                }
                # Optionally, log the first/last layer grads, etc.
                wandb_dict.update({
                    "enc_grad_first_layer": grad_stats_enc.first_layer,
                    "enc_grad_last_layer": grad_stats_enc.last_layer,
                    "pred_grad_first_layer": grad_stats_pred.first_layer,
                    "pred_grad_last_layer": grad_stats_pred.last_layer,
                })
                # Log
                wandb.log(wandb_dict, step=global_step)

            # Print to stdout only if needed
            if (itr % log_freq == 0) or np.isnan(loss_val) or np.isinf(loss_val):
                print((
                    f"[Epoch {epoch+1}, Iter {itr}/{ipe}] "
                    f"loss: {loss_meter.avg:.3f} | jepa: {jepa_loss_meter.avg:.3f}, reg: {reg_loss_meter.avg:.3f} | "
                    f"input_var: {input_var_meter.avg:.3f}, min: {input_var_min_meter.avg:.3f} | "
                    f"[wd: {new_wd:.2e}, lr: {new_lr:.2e}] | "
                    f"GPU mem: {torch.cuda.max_memory_allocated() / 1024.0**2:.2f}MB | "
                    f"gpu_time: {gpu_time_meter.avg:.2f}ms, wall_time: {wall_time_meter.avg:.2f}ms"
                ))

            if np.isnan(loss_val):
                print("[WARN] Loss is NaN! Stopping training.")
                return

        # -- End of epoch
        print(f"[INFO] Epoch {epoch+1} finished. Avg loss = {loss_meter.avg:.3f}")

        # Evaluation on rank=0
        if rank == 0:
            eval_loss_avg, eval_loss_jepa_avg, eval_loss_reg_avg = evaluate(
                encoder, predictor, action_model, target_encoder,
                eval_loader, device, clip_model, clip_tokenizer,
                dtype, mixed_precision
            )
            print((
                f"[Eval @ epoch {epoch+1}] eval_loss: {eval_loss_avg:.3f}, "
                f"eval_jepa: {eval_loss_jepa_avg:.3f}, eval_reg: {eval_loss_reg_avg:.3f}"
            ))
            # Log evaluation metrics in wandb
            wandb.log({
                "eval/epoch": epoch + 1,
                "eval/loss": eval_loss_avg,
                "eval/loss_jepa": eval_loss_jepa_avg,
                "eval/loss_reg": eval_loss_reg_avg
            }, step=global_step)

        # Save checkpoint
        if epoch % checkpoint_freq == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_every_path = os.path.join(folder, f'{tag}-e{epoch}.pth.tar')
                save_checkpoint(epoch + 1, save_every_path)
