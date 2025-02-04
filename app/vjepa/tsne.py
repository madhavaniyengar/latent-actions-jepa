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

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


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
        token_learner = None

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

    eval_batch_size = 25
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
    if token_learner is not None:
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
    if token_learner is not None:
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
    # Put all models in eval mode
    encoder.eval()
    predictor.eval()
    action_model.eval()
    target_encoder.eval()
    if token_learner is not None:
        token_learner.eval()

    # We will collect latent actions for the first TWO episodes/batches
    num_episodes_to_collect = 2
    latent_actions_list = []
    labels_list = []

    print("[INFO] Collecting latent actions for the first two episodes...")
    with torch.no_grad():
        loader_iter = iter(eval_loader)
        for batch_idx in range(num_episodes_to_collect):
            try:
                batch_data = next(loader_iter)
            except StopIteration:
                print("[WARN] eval_loader exhausted before 2 episodes.")
                break

            # Depending on your mask type:
            #  - random_p => (udata, masks_enc, masks_pred, p)
            #  - multiblock3d => (udata, masks_enc, masks_pred)
            # Adapt as needed:
            if mask_type == 'random_p':
                udata, masks_enc, masks_pred, p = batch_data
            else:
                udata, masks_enc, masks_pred = batch_data
                p = [num_frames // 2] * batch_size  # or [num_frames//2] * B, if your code requires it

            # ----------------------------------------------------------------
            #  1) Move video clips, text, masks to device
            # ----------------------------------------------------------------
            clips = torch.cat([u.to(device, non_blocking=True) for u in udata[0]], dim=0)
            batch_size = udata[0][0].shape[0]

            # Get text embeddings (similar to training)
            labels = udata[1]
            labels = [' '.join(l.split('_')) for l in labels]
            input_ids = clip_tokenizer(labels, padding=True, return_tensors='pt').input_ids.to(device)
            text_embeddings = clip_model(input_ids).last_hidden_state  # [B, seq_len, D]
            text_embeddings = text_embeddings.mean(dim=1, keepdim=True)  # [B,1,D]

            # Move masks to device, repeat them if multiple clips
            _masks_enc, _masks_pred = [], []
            for _me, _mp in zip(masks_enc, masks_pred):
                _me = _me.to(device, non_blocking=True)
                _mp = _mp.to(device, non_blocking=True)
                _me = repeat_interleave_batch(_me, batch_size, repeat=num_clips)
                _mp = repeat_interleave_batch(_mp, batch_size, repeat=num_clips)
                _masks_enc.append(_me)
                _masks_pred.append(_mp)

            # ----------------------------------------------------------------
            #  2) Forward pass to get latent actions
            # ----------------------------------------------------------------
            # As an example, we do something similar to "forward_target_with_actions" 
            # plus "forward_context".
            # The key is we want the outputs of `action_model(...)`.

            # === target encoder + action_model (like "h_actions") ===
            h = target_encoder(clips)
            h = F.layer_norm(h, (h.size(-1),))
            h_splits = apply_masks(h, _masks_pred, concat=False)

            if p is not None:
                # Example: gather frames, pass them through token_learner if used
                # You might need to adapt the below to your code’s shape
                # or if p is a single integer repeated B times, do that logic:
                per_frame_embeddings_h = []
                for seq in [h]*len(p):
                    # Reshape each to [B, p_i, tokens_per_frame, D]
                    # If your code uses (B, T, D) directly, adapt accordingly:
                    frames = seq.reshape(
                        seq.size(0),
                        p[0], 
                        seq.size(1)//p[0],
                        seq.size(-1)
                    )
                    if token_learner is not None:
                        frames = frames.view(frames.size(0), -1, frames.size(-1))
                        frames = token_learner(frames)
                    else:
                        frames = frames.mean(dim=2)
                    per_frame_embeddings_h.append(frames)

                # Pass text + frames to action_model
                # We store the result in `latent_actions`.
                for frames_h in per_frame_embeddings_h:
                    action_input = torch.cat([text_embeddings.repeat(1, 1, 2), frames_h], dim=1)
                    latent_actions = action_model(action_input)  # shape [B, D] or something
                    latent_actions = latent_actions.mean(dim=1)  # shape [B, D]
                    latent_actions_list.append(latent_actions.cpu().numpy())
            else:
                # If `p` is not used or the code is simpler,
                # you can adapt your approach for how you gather
                # frames to feed action_model. Possibly your code
                # uses a different approach for "action tokens."
                # Example:
                action_input = torch.cat([text_embeddings, h], dim=1)
                latent_actions = action_model(action_input)  # shape [B, D]
                latent_actions = latent_actions.mean(dim=1)  # shape [B, D]
                latent_actions_list.append(latent_actions.cpu().numpy())

            # Keep track of batch labels for coloring if you want:
            labels_list.extend(labels)

    print("[INFO] Collected latent actions. Now running t-SNE...")

    # ----------------------------------------------------------------
    #  3) Concatenate all latent actions and run t-SNE
    # ----------------------------------------------------------------
    all_latent_actions = np.concatenate(latent_actions_list, axis=0)  # shape [N1+N2, action_dim]
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(all_latent_actions)  # shape [N1+N2, 2]

    
    episode1_length = latent_actions_list[0].shape[0]  # N1
    episode2_length = latent_actions_list[1].shape[0]  # N2

    # Indices in the big array
    ep1_indices = np.arange(0, episode1_length)
    ep2_indices = np.arange(episode1_length, episode1_length + episode2_length)

    embeddings_ep1 = embeddings_2d[ep1_indices]  # shape [N1, 2]
    embeddings_ep2 = embeddings_2d[ep2_indices]  # shape [N2, 2]


    # ----------------------------------------------------------------
    #  4) Plot t-SNE
    # ----------------------------------------------------------------
    plt.figure(figsize=(6, 6))
    plt.scatter(
        embeddings_ep1[:, 0],
        embeddings_ep1[:, 1],
        s=20,
        alpha=0.8,
        c=np.arange(episode1_length),  # color by index
        cmap='rainbow'                   # 'Blues' goes from light (low idx) to dark (high idx)
    )
    plt.title("t-SNE episode 1")
    # plt.xlabel("Dimension 1")
    # plt.ylabel("Dimension 2")
    # plt.colorbar(label="Action Index (Episode 1)")
    plt.savefig("tsne_latent_actions_ep1.png")

    # Episode 2
    plt.figure(figsize=(6, 6))
    plt.scatter(
        embeddings_ep2[:, 0],
        embeddings_ep2[:, 1],
        s=20,
        alpha=0.8,
        c=np.arange(episode2_length),  # color by index
        cmap='rainbow'
    )
    plt.title("t-SNE episode 2")
    # plt.xlabel("Dimension 1")
    # plt.ylabel("Dimension 2")
    # plt.colorbar(label="Action Index (Episode 2)")
    plt.savefig("tsne_latent_actions_ep2.png")

    print("[INFO] Finished plotting Episode 1 and Episode 2.")