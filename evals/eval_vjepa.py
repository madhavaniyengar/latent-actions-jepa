# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import os

# -- FOR DISTRIBUTED TRAINING ENSURE ONLY 1 DEVICE VISIBLE PER PROCESS
try:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import copy
import time
import numpy as np
import matplotlib.pyplot as plt

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
from src.utils.distributed import init_distributed
from src.utils.logging import get_logger
from src.utils.tensors import repeat_interleave_batch

from app.vjepa.utils import (
    load_checkpoint,
    init_video_model,
)
from app.vjepa.transforms import make_transforms
from transformers import CLIPTokenizer, CLIPTextModel

logger = get_logger(__name__)

def main_inference(args):
    """
    Example script that runs inference on a held-out set and plots the loss.
    """
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #
    # -- META
    cfgs_meta = args.get('meta')
    load_model = cfgs_meta.get('load_checkpoint', True)
    r_file = cfgs_meta.get('read_checkpoint', None)
    seed = cfgs_meta.get('seed', 0)
    use_sdpa = cfgs_meta.get('use_sdpa', False)
    which_dtype = cfgs_meta.get('dtype', 'float32')
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
    action_dim = cfgs_model.get('action_dim', pred_embed_dim)  
    delta_frames = cfgs_model.get('delta_frames', 4)

    # -- DATA
    cfgs_data = args.get('data')
    dataset_type = cfgs_data.get('dataset_type', 'videodataset')
    mask_type = cfgs_data.get('mask_type', 'multiblock3d')
    dataset_paths = cfgs_data.get('datasets', [])
    batch_size = cfgs_data.get('batch_size', 1)
    num_clips = cfgs_data.get('num_clips', 1)
    num_frames = cfgs_data.get('num_frames', 16)
    tubelet_size = cfgs_data.get('tubelet_size', 2)
    sampling_rate = cfgs_data.get('sampling_rate', 4)
    duration = cfgs_data.get('clip_duration', None)
    crop_size = cfgs_data.get('crop_size', 224)
    patch_size = cfgs_data.get('patch_size', 16)
    pin_mem = cfgs_data.get('pin_mem', False)
    num_workers = cfgs_data.get('num_workers', 1)
    filter_short_videos = cfgs_data.get('filter_short_videos', False)
    decode_one_clip = cfgs_data.get('decode_one_clip', True)
    log_resource_util_data = cfgs_data.get('log_resource_utilization', False)

    # -- DATA AUGS
    cfgs_data_aug = args.get('data_aug', {})
    ar_range = cfgs_data_aug.get('random_resize_aspect_ratio', [3/4, 4/3])
    rr_scale = cfgs_data_aug.get('random_resize_scale', [0.3, 1.0])
    motion_shift = cfgs_data_aug.get('motion_shift', False)
    reprob = cfgs_data_aug.get('reprob', 0.)
    use_aa = cfgs_data_aug.get('auto_augment', False)

    # -- LOGGING
    cfgs_logging = args.get('logging', {})
    folder = cfgs_logging.get('folder', './')
    tag = cfgs_logging.get('write_tag', 'vjepa_inference')

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

    # -- checkpoint path
    latest_file = f'{tag}-latest.pth.tar'
    latest_path = os.path.join(folder, latest_file)
    load_path = None
    if load_model:
        if r_file is not None:
            load_path = os.path.join(folder, r_file)
        else:
            load_path = latest_path
        if not os.path.exists(load_path):
            logger.info(f"No valid checkpoint found at {load_path}, skipping load.")
            load_path = None
            load_model = False

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

    # encoder = DistributedDataParallel(encoder, static_graph=True)
    # predictor = DistributedDataParallel(predictor, static_graph=True)
    # action_model = DistributedDataParallel(action_model, static_graph=True)
    # target_encoder = DistributedDataParallel(target_encoder, static_graph=True)
    for p in target_encoder.parameters():
        p.requires_grad = False

    # -- mask collator
    if mask_type == 'multiblock3d':
        mask_collator = MB3DMaskCollator(
            crop_size=crop_size,
            num_frames=num_frames,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            cfgs_mask=cfgs_mask)
    elif mask_type == 'random_p':
        mask_collator = RandomPMaskCollator(
            crop_size=crop_size,
            num_frames=num_frames,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            cfgs_mask=cfgs_mask)
    else:
        mask_collator = TubeMaskCollator(
            crop_size=crop_size,
            num_frames=num_frames,
            patch_size=patch_size,
            tubelet_size=tubelet_size,
            cfgs_mask=cfgs_mask)

    # -- transforms
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=0.,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size)

    # -- data loader
    (eval_loader,
     eval_sampler) = init_data(
         data=dataset_type,
         root_path=dataset_paths,
         batch_size=batch_size,
         training=False,  # <--- For evaluation
         clip_len=num_frames,
         frame_sample_rate=sampling_rate,
         filter_short_videos=filter_short_videos,
         decode_one_clip=decode_one_clip,
         duration=duration,
         num_clips=num_clips,
         transform=transform,
         collator=mask_collator,
         num_workers=num_workers,
         world_size=world_size,
         pin_mem=pin_mem,
         rank=rank,
         log_dir=folder if log_resource_util_data else None
     )
    logger.info(f"Eval loader length: {len(eval_loader)}")

    # -- load checkpoint
    if load_path is not None:
        logger.info(f"Loading checkpoint from {load_path}")
        (
            encoder,
            predictor,
            target_encoder,
            _opt,   # not used in eval
            _scaler, # not used in eval
            start_epoch,
        ) = load_checkpoint(
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=None,
            scaler=None,
        )
        logger.info(f"Checkpoint loaded. Starting from epoch {start_epoch}")

    # -- CLIP for text embeddings (if needed)
    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    clip_model.eval()

    # Put models in eval mode
    encoder.eval()
    predictor.eval()
    action_model.eval()
    target_encoder.eval()

    # We'll store losses to plot at the end
    eval_losses = []

    # Example loss exponent from your training script (or pick any)
    loss_exp = 2.0  

    logger.info("Starting evaluation on the held-out set...")
    with torch.no_grad(), torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
        for batch_idx, (udata, masks_enc, masks_pred, p) in enumerate(eval_loader):
            # udata[0] = list of video clips
            # udata[1] = list of labels (strings)
            # masks_enc, masks_pred = context & target masks
            # p = additional info from collator (patches per frame, etc.)

            # Move data to device
            clips = torch.cat([clip_.to(device) for clip_ in udata[0]], dim=0)
            labels = [' '.join(l.split('_')) for l in udata[1]]
            input_ids = clip_tokenizer(labels, padding=True, return_tensors='pt').input_ids.to(device)
            text_embeddings = clip_model(input_ids).last_hidden_state.mean(dim=1, keepdim=True)

            # Prepare the mask-tensors for each clip in the batch
            _masks_enc, _masks_pred = [], []
            for me, mpred in zip(masks_enc, masks_pred):
                me = me.to(device)
                mpred = mpred.to(device)
                me = repeat_interleave_batch(me, batch_size, repeat=num_clips)
                mpred = repeat_interleave_batch(mpred, batch_size, repeat=num_clips)
                _masks_enc.append(me)
                _masks_pred.append(mpred)

            # Teacher (target) embeddings
            h = target_encoder(clips)  
            h = F.layer_norm(h, (h.size(-1),))
            h_masked_list = apply_masks(h, _masks_pred, concat=False)

            # Context embeddings
            z_enc_list = encoder(clips, _masks_enc)

            # Summarize frames and pass through action_model
            per_frame_embeddings = [
                seq.reshape(seq.size(0), pi, seq.size(1)//pi, seq.size(-1)) 
                for seq, pi in zip(z_enc_list, p)
            ]
            per_frame_embeddings = [pfe.mean(dim=2) for pfe in per_frame_embeddings]
            action_model_input = [
                torch.cat([text_embeddings.repeat(1, 1, 2), pfe], dim=1)
                for pfe in per_frame_embeddings
            ]
            action_cls_tokens = [action_model(ami) for ami in action_model_input]

            # Predictor output
            z_pred_list = predictor(
                z_enc_list,
                h_masked_list,
                _masks_enc,
                _masks_pred,
                action_cls_tokens
            )

            # Compute a simple L1 or L2 error as "loss" 
            #   (adapt to your actual objective as needed)
            batch_loss = 0.
            for z_pred, h_ref in zip(z_pred_list, h_masked_list):
                # E.g., L2 with exponent=2
                # shape: [B, #masked_tokens, D]
                batch_loss += torch.mean(
                    torch.abs(z_pred - h_ref)**loss_exp
                ) / loss_exp
            batch_loss /= len(z_pred_list)

            # Collect this batch's loss
            eval_losses.append(batch_loss.item())

            if (batch_idx % 10) == 0:
                logger.info(f"Eval batch {batch_idx}/{len(eval_loader)} | loss={batch_loss.item():.4f}")

    # -- Done with eval; now plot the loss
    logger.info("Evaluation done. Plotting the loss curve...")

    plt.figure(figsize=(7, 5))
    plt.plot(eval_losses, label='Evaluation Loss')
    plt.xlabel("Batch Index")
    plt.ylabel("Loss")
    plt.title("Held-out Evaluation Loss")
    plt.legend()
    plot_path = os.path.join(folder, f"{tag}_eval_loss.png")
    plt.savefig(plot_path)
    logger.info(f"Saved evaluation loss plot to {plot_path}")

    # Optionally, return the raw data as well
    return eval_losses


if __name__ == "__main__":
    """
    Example usage:
      python inference_script.py --config configs/inference_config.yaml
    """
    import yaml
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    args_cli = parser.parse_args()

    with open(args_cli.config, "r") as f:
        args = yaml.safe_load(f)

    # Run inference (evaluation) + plot the held-out losses
    eval_losses = main_inference(args)
    # Do something else with eval_losses if needed
