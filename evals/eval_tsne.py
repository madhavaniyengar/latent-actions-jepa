# Modified Classification Script for t-SNE Visualization of Action CLS Tokens
import os

import argparse
import yaml

# Ensure only 1 device visible per process for distributed training
try:
    os.environ['CUDA_VISIBLE_DEVICES'] = os.environ['SLURM_LOCALID']
except Exception:
    pass

import logging
import pprint
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F

from torch.nn.parallel import DistributedDataParallel

import src.models.vision_transformer as vit
from src.datasets.data_manager import init_data
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter

from app.vjepa.transforms import make_transforms

logging.basicConfig()
logger = logging.getLogger()
logger.setLevel(logging.INFO)

_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

pp = pprint.PrettyPrinter(indent=4)


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
    delta_frames = args_pretrain.get('delta_frames', 4)
    action_dim = args_pretrain.get('action_dim', None)

    # -- DATA
    args_data = args_eval.get('data')
    val_data_path = [args_data.get('dataset_val')]
    dataset_type = args_data.get('dataset_type', 'VideoDataset')
    num_classes = args_data.get('num_classes')
    eval_num_segments = args_data.get('num_segments', 1)
    eval_frames_per_clip = args_data.get('frames_per_clip', 16)
    eval_frame_step = args_pretrain.get('frame_step', 4)
    eval_duration = args_pretrain.get('clip_duration', None)
    eval_num_views_per_segment = args_data.get('num_views_per_segment', 1)

    # -- OPTIMIZATION
    args_opt = args_eval.get('optimization')
    resolution = args_opt.get('resolution', 224)
    batch_size = args_opt.get('batch_size')
    use_bfloat16 = args_opt.get('use_bfloat16')

    # -- EXPERIMENT-ID/TAG (optional)
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
    folder = os.path.join(pretrain_folder, 'tsne_visualization/')
    if eval_tag is not None:
        folder = os.path.join(folder, eval_tag)
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)

    # Initialize model

    # -- pretrained encoder (frozen)
    encoder, action_mlp = init_model(
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

    # Load data
    data_loader = make_dataloader(
        dataset_type=dataset_type,
        root_path=val_data_path,
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
        training=False)
    
    # Take only a subset of episodes for clarity (e.g., 10 episodes)
    data_loader_iter = iter(data_loader)
    data_loader_subset = [next(data_loader_iter) for _ in range(10)]

    delta_frames = args_pretrain.get('delta_frames', 4)
    num_frames = eval_frames_per_clip

    # Collect action_cls_tokens and labels
    action_cls_tokens_per_episode, labels_per_episode = collect_action_cls_tokens(
        device=device,
        encoder=encoder,
        action_mlp=action_mlp,
        data_loader=data_loader_subset,
        delta_frames=delta_frames,
        num_frames=num_frames,
        use_bfloat16=use_bfloat16
    )

    # Prepare data for t-SNE
    all_action_tokens = []
    action_indices = []
    episode_indices = []

    for episode_idx, action_tokens in enumerate(action_cls_tokens_per_episode):
        num_actions = action_tokens.shape[0]
        all_action_tokens.append(action_tokens)  # shape [num_actions, action_dim]
        action_indices.extend(range(num_actions))
        episode_indices.extend([episode_idx] * num_actions)

    all_action_tokens_array = np.concatenate(all_action_tokens, axis=0)  # shape [total_actions, action_dim]
    action_indices_array = np.array(action_indices)
    episode_indices_array = np.array(episode_indices)

    # Run t-SNE
    tsne = TSNE(n_components=2, random_state=42)
    action_cls_tokens_tsne = tsne.fit_transform(all_action_tokens_array)

    # Plot
    plt.figure(figsize=(12, 12))
    scatter = plt.scatter(action_cls_tokens_tsne[:, 0], action_cls_tokens_tsne[:, 1],
                          c=action_indices_array, cmap='tab10', alpha=0.7)
    plt.legend(*scatter.legend_elements(), title="Action Index")
    plt.title('t-SNE of action_cls_tokens')

    # Annotate the first action token of each episode with the episode number
    for episode_idx in range(len(action_cls_tokens_per_episode)):
        indices = (episode_indices_array == episode_idx) & (action_indices_array == 0)
        if np.any(indices):
            x = action_cls_tokens_tsne[indices, 0][0]
            y = action_cls_tokens_tsne[indices, 1][0]
            plt.annotate(str(episode_idx), (x, y), textcoords="offset points",
                         xytext=(0,10), ha='center', fontsize=8)

    # Connect the action tokens of each episode with lines
    for episode_idx in range(len(action_cls_tokens_per_episode)):
        indices = (episode_indices_array == episode_idx)
        xs = action_cls_tokens_tsne[indices, 0]
        ys = action_cls_tokens_tsne[indices, 1]
        plt.plot(xs, ys, linestyle='-', linewidth=0.5, alpha=0.5)

    plt.savefig(os.path.join(folder, 'tsne_action_cls_tokens.png'))
    plt.show()

    # Examine changes between actions
    # Compute distances between sequential action tokens within each episode
    sequential_distances = []
    for episode_idx in range(len(action_cls_tokens_per_episode)):
        indices = (episode_indices_array == episode_idx)
        episode_points = action_cls_tokens_tsne[indices]
        # Compute pairwise distances between sequential points
        if episode_points.shape[0] > 1:
            diffs = np.diff(episode_points, axis=0)
            dists = np.linalg.norm(diffs, axis=1)
            sequential_distances.extend(dists)

    # Output statistics
    if sequential_distances:
        mean_dist = np.mean(sequential_distances)
        std_dist = np.std(sequential_distances)
        logger.info(f'Mean sequential distance: {mean_dist:.4f}')
        logger.info(f'Standard deviation of sequential distances: {std_dist:.4f}')

        # Optionally, plot a histogram of the distances
        plt.figure(figsize=(8, 6))
        plt.hist(sequential_distances, bins=20, alpha=0.7)
        plt.xlabel('Sequential Distance')
        plt.ylabel('Frequency')
        plt.title('Histogram of Sequential Distances Between Actions')
        plt.savefig(os.path.join(folder, 'sequential_distances_histogram.png'))
        plt.show()
    else:
        logger.info('Not enough data to compute sequential distances.')


def collect_action_cls_tokens(
    device,
    encoder,
    action_mlp,
    data_loader,
    delta_frames,
    num_frames,
    use_bfloat16=False
):
    encoder.eval()
    action_cls_tokens_per_episode = []
    labels_per_episode = []

    with torch.no_grad():
        for itr, data in enumerate(data_loader):
            # Load data and put on GPU
            clips = torch.cat([u.to(device, non_blocking=True) for u in data[0]], dim=0)
            # labels = data[1].to(device)
            
            # batch_size = len(labels)

            with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_bfloat16):
                h = encoder(clips)  # h is [B, N, D]

                h = F.layer_norm(h, (h.size(-1),))  # [B, N, D]

                # Compute tokens per frame and delta tokens
                B, N, D = h.size()
                tokens_per_frame = N // num_frames
                delta_tokens = tokens_per_frame * delta_frames
                num_groups = N // delta_tokens

                # Reshape and average pool over delta_tokens
                h_grouped = h.view(B, num_groups, delta_tokens, D)
                h_pooled = h_grouped.mean(dim=2)  # [B, num_groups, D]

                # Pass through action_mlp to get Action CLS tokens
                action_cls_tokens = action_mlp(h_pooled)  # [B, num_groups, action_dim]

                # For each sample, collect action_cls_tokens
                for i in range(B):
                    # labels_per_episode.append(labels[i].cpu().item())
                    action_tokens = action_cls_tokens[i]  # shape [num_groups, action_dim]
                    # Collect first 4 action tokens
                    num_actions = min(4, action_tokens.shape[0])
                    action_tokens = action_tokens[:num_actions].cpu().numpy()  # shape [num_actions, action_dim]
                    action_cls_tokens_per_episode.append(action_tokens)

    # Now, action_cls_tokens_per_episode is a list of arrays of shape [num_actions, action_dim], one per episode
    # labels_per_episode is a list of labels per episode

    return action_cls_tokens_per_episode, labels_per_episode


def load_pretrained(
    encoder,
    action_mlp,
    pretrained,
    checkpoint_key='target_encoder'
):
    logger.info(f'Loading pretrained model from {pretrained}')
    checkpoint = torch.load(pretrained, map_location='cpu')
    try:
        pretrained_dict = checkpoint[checkpoint_key]
        pretrained_action_dict = checkpoint['action_mlp']
    except Exception:
        pretrained_dict = checkpoint['encoder']
        pretrained_action_dict = checkpoint['action_mlp']

    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace('backbone.', ''): v for k, v in pretrained_dict.items()}
    for k, v in encoder.state_dict().items():
        if k not in pretrained_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_dict[k] = v
    msg = encoder.load_state_dict(pretrained_dict, strict=False)
    logger.info(f'loaded pretrained encoder model with msg: {msg}')
    logger.info(f'loaded pretrained encoder from epoch: {checkpoint["epoch"]}\n path: {pretrained}')
    
    
    pretrained_action_dict = {k.replace('module.', ''): v for k, v in pretrained_action_dict.items()}
    pretrained_action_dict = {k.replace('backbone.', ''): v for k, v in pretrained_action_dict.items()}
    for k, v in action_mlp.state_dict().items():
        if k not in pretrained_action_dict:
            logger.info(f'key "{k}" could not be found in loaded state dict')
        elif pretrained_action_dict[k].shape != v.shape:
            logger.info(f'key "{k}" is of different shape in model and loaded state dict')
            pretrained_action_dict[k] = v
    msg = action_mlp.load_state_dict(pretrained_action_dict, strict=False)
    logger.info(f'loaded pretrained action model with msg: {msg}')
    logger.info(f'loaded pretrained action mlp from epoch: {checkpoint["epoch"]}\n path: {pretrained}')
    
    
    del checkpoint
    return encoder, action_mlp


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
    checkpoint_key='target_encoder',
    action_dim=1024
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
    
    action_mlp = nn.Sequential(
        nn.Linear(action_dim, action_dim),
        nn.ReLU(),
        nn.Linear(action_dim, action_dim)
    )
    action_mlp.to(device)
    
    
    encoder, action_mlp = load_pretrained(encoder=encoder, action_mlp=action_mlp, pretrained=pretrained, checkpoint_key=checkpoint_key)
                  
    return encoder, action_mlp


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--fname', type=str,
        help='name of config file to load',
        default='configs.yaml')
    args = parser.parse_args()
    fname = args.fname
    config_base = '/home/madhavan/jepa/configs'  # Update this path as needed
    with open(os.path.join(config_base, 'evals', fname), 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')
    # Convert args to dict
    main(params)
