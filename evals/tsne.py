import os
import argparse
import yaml
import pprint
import logging
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.nn.parallel import DistributedDataParallel

from sklearn.manifold import TSNE

# -- These come from your updated training script modules
from src.utils.distributed import init_distributed
from src.datasets.data_manager import init_data
from app.vjepa.transforms import make_transforms
from app.vjepa.utils import init_video_model
from models.TokenLearner import TokenLearner

# CLIP
from transformers import CLIPTokenizer, CLIPTextModel

# Logging setup
logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

pp = pprint.PrettyPrinter(indent=4)

# ----------------------------------------------------------------------- #
#  COLLECT ACTION CLS TOKENS
# ----------------------------------------------------------------------- #

def collect_action_cls_tokens(
    device,
    target_encoder,
    action_model,
    token_learner,
    clip_model,
    clip_tokenizer,
    data_loader,
    num_frames,
    delta_frames=4,
    use_mixed_precision=False,
    dtype=torch.float32,
    max_episodes=10
):
    """
    Collect "action CLS" tokens from target_encoder + action_model
    (optionally token_learner) for t-SNE visualization.

    Args:
        device: torch device
        target_encoder: the frozen target encoder (e.g. from training code)
        action_model: the action model (transformer or MLP) that produces action CLS tokens
        token_learner: optional TokenLearner model (or None if not used)
        clip_model: CLIPTextModel for text embeddings (if your dataset has text labels)
        clip_tokenizer: CLIPTokenizer for turning labels into tokens
        data_loader: an iterable of (udata, masks, etc.) 
                     or simply (clips, labels) if you are not using masks
        num_frames: how many frames per clip
        delta_frames: how many frames go into each "action token" grouping
        use_mixed_precision: whether to run in amp autocast
        dtype: torch dtype to cast
        max_episodes: how many episodes/batches to collect (for clarity)
    """
    # Switch everything to eval mode
    target_encoder.eval()
    action_model.eval()
    if token_learner is not None:
        token_learner.eval()
    clip_model.eval()

    # Disable grad
    action_cls_tokens_per_episode = []
    labels_per_episode = []

    # We only collect from a subset for clarity
    data_loader_iter = iter(data_loader)

    with torch.no_grad():
        for ep_idx in range(max_episodes):
            try:
                batch_datapoint = next(data_loader_iter)
            except StopIteration:
                break

            # ------------------------------------------------------------------
            # The new training script organizes data differently:
            #   - Typically: batch_datapoint = (udata, masks_enc, masks_pred, ...)
            #   - For t-SNE, you might not need masks, or set them aside.
            #
            # Here we assume your dataset returns (udata, ...) 
            # where  udata[0] is a list of clips,  udata[1] are text labels.
            # Adjust if your dataset structure is different.
            # ------------------------------------------------------------------
            udata = batch_datapoint[0]
            # If you do have masks in your pipeline, adapt accordingly:
            # masks_enc, masks_pred = batch_datapoint[1], batch_datapoint[2], ...
            # We won't use them here for the sake of direct action embedding.

            # Combine the multiple video clips in udata[0]
            clips = torch.cat([u.to(device, non_blocking=True) for u in udata[0]], dim=0)
            labels = udata[1]
            # If your dataset uses string labels like 'something_something' 
            # (and you want CLIP embeddings):
            labels = [' '.join(l.split('_')) for l in labels]

            # B = batch size
            B = len(udata[0][0]) 

            # Compute text embeddings
            input_ids = clip_tokenizer(labels, padding=True, return_tensors='pt').input_ids.to(device)
            with autocast(enabled=use_mixed_precision, dtype=dtype):
                text_embeddings = clip_model(input_ids).last_hidden_state
            # For simplicity, pool them
            text_embeddings = text_embeddings.mean(dim=1, keepdim=True)  # shape [B,1,D]

            # ------------------------------------------------------------------
            #   1) Forward pass target_encoder
            # ------------------------------------------------------------------
            with autocast(enabled=use_mixed_precision, dtype=dtype):
                h = target_encoder(clips)  # shape [B, N, D], N ~ # patches/tokens
                h = F.layer_norm(h, (h.size(-1),))

            # ------------------------------------------------------------------
            #   2) Reshape to group frames (like in the training code)
            #      If you're doing tubelets or some other grouping, adapt this.
            # ------------------------------------------------------------------
            # Suppose N = num_tokens == (num_frames * tokens_per_frame).
            # We want to group delta_frames worth of frames into each action. 
            # For example, if num_frames=16 and delta_frames=4, we get 4 action tokens.
            B_, N, D_ = h.size()
            tokens_per_frame = N // num_frames
            group_size = tokens_per_frame * delta_frames
            num_actions = N // group_size  # e.g., 4 if delta_frames=4

            # Reshape to [B, num_actions, group_size, D]
            h_grouped = h.view(B_, num_actions, group_size, D_)
            # Average over the group dimension => shape [B, num_actions, D]
            h_pooled = h_grouped.mean(dim=2)

            # If using a TokenLearner, apply it here (each group = "frame"?)
            # The new training script does something similar:
            if token_learner is not None:
                # token_learner expects [B, #tokens, D]
                # so flatten (num_actions) for the token learner
                h_pooled = h_pooled.view(B_, num_actions, D_)
                # Suppose token_learner returns [B, new_num_tokens, D]
                h_pooled = token_learner(h_pooled)  # shape [B, new_num_tokens, D]
                # For a single "action CLS token" you might average or select the first
                # The training code may keep them separate, so adapt as needed
                h_pooled = h_pooled.mean(dim=1, keepdim=True)  # shape [B,1,D] 
                # or keep them all, depends on how you want to visualize

            # ------------------------------------------------------------------
            #   3) Pass text + visual embeddings into action_model
            # ------------------------------------------------------------------
            # The new code often concatenates text embeddings + frame embeddings 
            # something like:
            #
            #   action_input = torch.cat([text_embeddings.repeat(1,1,2), h_pooled], dim=1)
            #
            # If the training code uses 2 text tokens, ensure you do so consistently:
            text_rep = text_embeddings.repeat(1, 1, 2)  # shape [B,2,D]
            # Now cat with h_pooled along dim=1
            # Make sure h_pooled is shape [B, X, D]
            if len(h_pooled.shape) == 3:
                action_input = torch.cat([text_rep, h_pooled], dim=1)  # [B, 2+X, D]
            else:
                # If for some reason h_pooled is [B,D], unsqueeze
                action_input = torch.cat([text_rep, h_pooled.unsqueeze(1)], dim=1)  # [B,3,D]

            with autocast(enabled=use_mixed_precision, dtype=dtype):
                action_cls_tokens = action_model(action_input)  # shape [B, #tokens, action_dim]

            # ------------------------------------------------------------------
            #   4) Collect only a few action tokens per "episode"
            # ------------------------------------------------------------------
            # For example, if we want to store the first 4
            # or if we have a single token, we store them all
            # shape => [B, #action_tokens, action_dim]
            if action_cls_tokens.dim() == 3:
                # If the second dimension is your "action tokens"
                # pick how many you want to keep
                keep_count = min(action_cls_tokens.size(1), 4)
                action_tokens_batch = action_cls_tokens[:, :keep_count, :].cpu().numpy()
            else:
                # If 2D => [B, action_dim]
                action_tokens_batch = action_cls_tokens.cpu().numpy()
                action_tokens_batch = action_tokens_batch[:, None, :]  # shape [B,1,action_dim]

            # We can treat each sample in the batch as a separate "episode"
            # or treat the entire batch as one "episode". 
            # Usually, you'd do 1 sample = 1 episode. But adapt as needed.
            for i in range(B):
                # you could store labels if you want
                # label_i = labels[i]  # if your labels are strings
                # logger.info(f"Episode {ep_idx} - label: {label_i}")

                action_tokens_i = action_tokens_batch[i]  # shape [<=4, action_dim]
                action_cls_tokens_per_episode.append(action_tokens_i)
                labels_per_episode.append(labels[i])  # store label if you like

    # Return your collected tokens 
    # (list of length = #episodes, each element is [<=4, action_dim])
    return action_cls_tokens_per_episode, labels_per_episode


# ----------------------------------------------------------------------- #
#  MAIN
# ----------------------------------------------------------------------- #
def main(args_eval):
    """
    Example main function to:
      1) Parse config
      2) Init distributed
      3) Build model + load checkpoint
      4) Build dataloader
      5) Collect action CLS tokens
      6) Run t-SNE and plot
    """

    # ------------------------------------------------------------------- #
    #  1) LOAD CONFIG
    # ------------------------------------------------------------------- #
    # e.g. --fname configs.yaml
    parser = argparse.ArgumentParser()
    parser.add_argument('--fname', type=str, default='configs.yaml')
    cli_args = parser.parse_args()

    config_path = cli_args.fname
    with open(config_path, 'r') as y_file:
        params = yaml.safe_load(y_file)
    pp.pprint(params)

    # For clarity, some typical top-level fields:
    #   params['meta'], params['mask'], params['model'], params['data'], ...
    # Adjust usage as needed.

    # Short-hands
    cfg_meta = params.get('meta', {})
    cfg_model = params.get('model', {})
    cfg_data = params.get('data', {})
    cfg_data_aug = params.get('data_aug', {})
    cfg_logging = params.get('logging', {})
    folder = cfg_logging.get('folder', './checkpoints')
    tag = cfg_logging.get('write_tag', 'tsne-exp')
    
    # Data related
    dataset_paths = cfg_data.get('datasets', [])
    batch_size = cfg_data.get('batch_size', 4)
    num_frames = cfg_data.get('num_frames', 16)
    sampling_rate = cfg_data.get('sampling_rate', 4)
    duration = cfg_data.get('clip_duration', None)
    num_clips = cfg_data.get('num_clips', 1)
    crop_size = cfg_data.get('crop_size', 224)
    patch_size = cfg_data.get('patch_size', 16)

    # If you need tubelet_size, or other fields:
    tubelet_size = cfg_data.get('tubelet_size', 2)

    # Mixed precision / dtype
    which_dtype = cfg_meta.get('dtype', 'float32').lower()
    if which_dtype == 'bfloat16':
        dtype = torch.bfloat16
        use_mixed_precision = True
    elif which_dtype == 'float16':
        dtype = torch.float16
        use_mixed_precision = True
    else:
        dtype = torch.float32
        use_mixed_precision = False

    # Action dimension
    action_dim = cfg_model.get('action_dim', 1024)
    delta_frames = cfg_model.get('delta_frames', 4)

    # Where to load model from
    load_checkpoint = cfg_meta.get('load_checkpoint', True)
    read_ckpt_file = cfg_meta.get('read_checkpoint', f'{tag}-latest.pth.tar')

    # ------------------------------------------------------------------- #
    #  2) INIT DISTRIBUTED + DEVICE
    # ------------------------------------------------------------------- #
    world_size, rank = init_distributed()
    logger.info(f'Initialized (rank={rank}, world_size={world_size})')

    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda', rank)
        torch.cuda.set_device(device)

    # ------------------------------------------------------------------- #
    #  3) BUILD MODEL (encoder, predictor, action_model, target_encoder, ...)
    # ------------------------------------------------------------------- #
    # This is the new function from the updated training script:
    # app.vjepa.utils.init_video_model(...)
    # For t-SNE, we only *need* target_encoder & action_model, but 
    # we can create the full set to maintain the same structure.
    uniform_power = cfg_model.get('uniform_power', True)
    use_mask_tokens = cfg_model.get('use_mask_tokens', True)
    zero_init_mask_tokens = cfg_model.get('zero_init_mask_tokens', True)
    model_name = cfg_model.get('model_name', 'vit_small_patch16_224')
    pred_depth = cfg_model.get('pred_depth', 1)
    pred_embed_dim = cfg_model.get('pred_embed_dim', 384)
    use_sdpa = cfg_meta.get('use_sdpa', False)

    # Initialize
    encoder, predictor, action_model = init_video_model(
        uniform_power=uniform_power,
        use_mask_tokens=use_mask_tokens,
        num_mask_tokens=1,  # in training you might set >1 if you have multiple masks
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
        clip_embed_dim=512,    # from new code
        action_transformer=True
    )

    # Make a copy for target encoder
    target_encoder = torch.clone(encoder.module if isinstance(encoder, DistributedDataParallel) else encoder)
    if isinstance(target_encoder, nn.DataParallel) or isinstance(target_encoder, DistributedDataParallel):
        # in case you get an error about .clone() on DDP, remove the wrapper or load from checkpoint
        pass

    # Optionally, if you use TokenLearner
    token_learner = None
    if cfg_model.get('token_learning_model', 'tokenlearner') == 'tokenlearner':
        num_output_tokens = 8
        token_learner = TokenLearner(embed_dim=action_dim, num_output_tokens=num_output_tokens).to(device)

    # Build CLIP model for text
    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")

    # Wrap in DDP if you like (but for inference only, you can skip)
    target_encoder = DistributedDataParallel(target_encoder, device_ids=[rank])
    action_model = DistributedDataParallel(action_model, device_ids=[rank])
    if token_learner is not None:
        token_learner = DistributedDataParallel(token_learner, device_ids=[rank])

    # Load checkpoint if needed
    if load_checkpoint:
        ckpt_path = os.path.join(folder, read_ckpt_file)
        if os.path.exists(ckpt_path):
            logger.info(f"Loading checkpoint from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location='cpu')
            # adapt these loads to your checkpoint structure
            target_encoder.load_state_dict(ckpt['target_encoder'], strict=False)
            action_model.load_state_dict(ckpt['action_model'], strict=False)
            if 'encoder' in ckpt:
                encoder.load_state_dict(ckpt['encoder'], strict=False)
            if token_learner is not None and 'token_learner' in ckpt:
                token_learner.load_state_dict(ckpt['token_learner'], strict=False)
            logger.info(f"Checkpoint loaded (epoch={ckpt.get('epoch', 0)}).")
        else:
            logger.warning(f"No checkpoint found at {ckpt_path}, proceeding without load.")

    # Freeze target encoder (as done in training)
    for p in target_encoder.parameters():
        p.requires_grad = False
    # same for action_model if you want it frozen, or keep it trainable if your usage differs
    for p in action_model.parameters():
        p.requires_grad = False
    if token_learner is not None:
        for p in token_learner.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------- #
    #  4) BUILD DATALOADER
    # ------------------------------------------------------------------- #
    # e.g. something like your training script for eval
    transform = make_transforms(
        random_horizontal_flip=False,
        random_resize_aspect_ratio=(0.75, 4/3),
        random_resize_scale=(0.08, 1.0),
        reprob=0.0,
        auto_augment=False,
        motion_shift=False,
        crop_size=crop_size,
    )

    # This is a simplified usage; if you have a custom collator for masks, remove it if not needed
    data_loader, _ = init_data(
        data=cfg_data.get('dataset_type', 'VideoDataset'),
        root_path=dataset_paths,
        transform=transform,
        batch_size=batch_size,
        world_size=1,  # for local debugging, or adapt for DDP
        rank=0,
        clip_len=num_frames,
        frame_sample_rate=sampling_rate,
        duration=duration,
        num_clips=num_clips,
        training=False,
        num_workers=cfg_data.get('num_workers', 4),
        pin_mem=cfg_data.get('pin_mem', False),
    )

    # ------------------------------------------------------------------- #
    #  5) COLLECT ACTION CLS TOKENS
    # ------------------------------------------------------------------- #
    action_cls_tokens_per_episode, labels_per_episode = collect_action_cls_tokens(
        device=device,
        target_encoder=target_encoder,
        action_model=action_model,
        token_learner=token_learner,
        clip_model=clip_model,
        clip_tokenizer=clip_tokenizer,
        data_loader=data_loader,
        num_frames=num_frames,
        delta_frames=delta_frames,
        use_mixed_precision=use_mixed_precision,
        dtype=dtype,
        max_episodes=10
    )

    # Prepare data for t-SNE
    all_action_tokens = []
    action_indices = []
    episode_indices = []

    idx_counter = 0
    for episode_idx, tokens_np in enumerate(action_cls_tokens_per_episode):
        # tokens_np is [#actions, action_dim] for that episode
        count_actions = tokens_np.shape[0]
        all_action_tokens.append(tokens_np)
        action_indices.extend(range(count_actions))
        episode_indices.extend([episode_idx]*count_actions)

    all_action_tokens_array = np.concatenate(all_action_tokens, axis=0)  # shape [total_actions, action_dim]
    action_indices_array = np.array(action_indices)
    episode_indices_array = np.array(episode_indices)

    # ------------------------------------------------------------------- #
    #  6) RUN TSNE + PLOT
    # ------------------------------------------------------------------- #
    logger.info("Running t-SNE...")
    tsne = TSNE(n_components=2, random_state=42)
    action_cls_tokens_tsne = tsne.fit_transform(all_action_tokens_array)

    plt.figure(figsize=(12, 12))
    scatter = plt.scatter(action_cls_tokens_tsne[:, 0],
                          action_cls_tokens_tsne[:, 1],
                          c=action_indices_array,
                          cmap='tab10',
                          alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Action Index")
    plt.title('t-SNE of action_cls_tokens (New Training Script)')

    # Annotate the first action token of each "episode"
    for episode_idx in range(len(action_cls_tokens_per_episode)):
        indices = (episode_indices_array == episode_idx) & (action_indices_array == 0)
        if np.any(indices):
            x = action_cls_tokens_tsne[indices, 0][0]
            y = action_cls_tokens_tsne[indices, 1][0]
            plt.annotate(str(episode_idx), (x, y), textcoords="offset points",
                         xytext=(0,10), ha='center', fontsize=8)

    # Optionally connect the action tokens of each episode with a line
    for episode_idx in range(len(action_cls_tokens_per_episode)):
        indices = (episode_indices_array == episode_idx)
        xs = action_cls_tokens_tsne[indices, 0]
        ys = action_cls_tokens_tsne[indices, 1]
        plt.plot(xs, ys, linestyle='-', linewidth=0.5, alpha=0.5)

    # Save and show
    out_dir = os.path.join(folder, 'tsne_visualization', tag)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'tsne_action_cls_tokens.png')
    plt.savefig(out_path)
    logger.info(f"Saved t-SNE plot to {out_path}")
    plt.show()

    # Optionally compute distances between sequential action tokens
    sequential_distances = []
    for episode_idx in range(len(action_cls_tokens_per_episode)):
        indices = (episode_indices_array == episode_idx)
        episode_points = action_cls_tokens_tsne[indices]
        if episode_points.shape[0] > 1:
            diffs = np.diff(episode_points, axis=0)
            dists = np.linalg.norm(diffs, axis=1)
            sequential_distances.extend(dists)
    if sequential_distances:
        mean_dist = np.mean(sequential_distances)
        std_dist = np.std(sequential_distances)
        logger.info(f"Mean sequential distance: {mean_dist:.4f}")
        logger.info(f"Std dev of sequential distances: {std_dist:.4f}")

        # Plot histogram
        plt.figure(figsize=(8, 6))
        plt.hist(sequential_distances, bins=20, alpha=0.7)
        plt.xlabel('Sequential Distance')
        plt.ylabel('Frequency')
        plt.title('Histogram of Sequential Distances')
        hist_path = os.path.join(out_dir, 'sequential_distances_histogram.png')
        plt.savefig(hist_path)
        logger.info(f"Saved distance histogram to {hist_path}")
        plt.show()
    else:
        logger.info("Not enough data for sequential distances.")


if __name__ == '__main__':
    main(None)
