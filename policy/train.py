import argparse
import os
import copy
import time
import numpy as np
import yaml

import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

import wandb 

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

from policy.dataset import Dataset
from torch.utils.data import DataLoader
from policy.MLP import MLP
from policy.embedding_video_cnn import EmbeddingVideoCNN
from policy.baseline_video_cnn import VideoCNN


_GLOBAL_SEED = 0
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)

def main(config):
    # load config yaml file
    with open(config, 'r') as f:
        args = yaml.load(f, Loader=yaml.FullLoader)
    
    cfgs_meta = args.get('meta')
    load_model = cfgs_meta.get('load_checkpoint')
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
    policy_learning_model = cfgs_model.get('policy_learning_model', 'mlp')

    # -- DATA
    cfgs_data = args.get('data')
    dataset_type = cfgs_data.get('dataset_type', 'videodataset')
    dataset_name = cfgs_data.get('dataset', None)
    action_representation = cfgs_data.get('action_representation', 'pose')
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


    # -- DATA AUGS
    cfgs_data_aug = args.get('data_aug')
    ar_range = cfgs_data_aug.get('random_resize_aspect_ratio', [3/4, 4/3])
    rr_scale = cfgs_data_aug.get('random_resize_scale', [0.3, 1.0])
    motion_shift = cfgs_data_aug.get('motion_shift', False)
    reprob = cfgs_data_aug.get('reprob', 0.)
    use_aa = cfgs_data_aug.get('auto_augment', False)
    
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method('spawn')
    except Exception:
        pass
    
    # -- set device
    if not torch.cuda.is_available():
        device = torch.device('cpu')
    else:
        device = torch.device('cuda:0')
        torch.cuda.set_device(device)
        
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
    
    transform = make_transforms(
        random_horizontal_flip=True,
        random_resize_aspect_ratio=ar_range,
        random_resize_scale=rr_scale,
        reprob=reprob,
        auto_augment=use_aa,
        motion_shift=motion_shift,
        crop_size=crop_size)
    
    clip_model = CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
    clip_tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    text_embedding_dim = clip_model.config.hidden_size
    
    load_path = cfgs_meta.get('read_checkpoint', None)
    
    if load_model and (load_path is not None):
        (encoder,
         predictor,
         target_encoder,
         action_model,
         optimizer,
         scaler,
         start_epoch) = load_checkpoint(
             r_path=load_path,
             encoder=encoder,
             predictor=predictor,
             target_encoder=None,
             action_model=action_model,
             opt=None,
             scaler=None
        )
         
    if policy_learning_model == 'action_mlp':
        print('Using MLP as policy model')
        policy_model = MLP(1024, 7).to(device)
    elif policy_learning_model == 'video_jepa_cnn':
        print('Using V-JEPA + CNN as policy model')
        policy_model = EmbeddingVideoCNN().to(device)
    elif policy_learning_model == 'video_cnn':
        print('Using default CNN policy model')
        policy_model = VideoCNN(num_frames=16).to(device)
    elif policy_learning_model == 'cnn':
        pass
    else:
        print('Using MLP as policy model')
        policy_model = MLP(1024, 7).to(device)

    optimizer = torch.optim.Adam(policy_model.parameters(), lr=1e-4)
    policy_loss = nn.MSELoss()
    
    text_description = ["pick up the mug"] * batch_size
    input_ids = clip_tokenizer(text_description, padding=True, return_tensors='pt').input_ids.to(device)
    with torch.no_grad():
        text_embeddings = clip_model(input_ids).last_hidden_state
    text_embeddings = text_embeddings.mean(dim=1, keepdim=True)
    
    
    wandb.init(project='vjepa', 
               entity='miyen',
               name=f'{policy_learning_model}-{action_representation}',)
        
    dataset = Dataset(dataset_name, action_representation=action_representation)
    
    # create a train and eval split
    train_size = int(0.8 * len(dataset))
    eval_size = len(dataset) - train_size
    train_dataset, eval_dataset = torch.utils.data.random_split(dataset, [train_size, eval_size])
    
    # create dataloaders
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
    
    
    def vjepa_forward_pass(clips, p, text_emb, token_learner=None):
        h = encoder(clips)
        h = F.layer_norm(h, (h.size(-1),))

        per_frame_embeddings = []
        reshaped = h.reshape(
            h.size(0),   # B
            p,
            h.size(1)//p,
            h.size(-1)
        )
        if token_learner is not None:
            reshaped = reshaped.view(reshaped.size(0), -1, reshaped.size(-1))
            learned_tokens = token_learner(reshaped)
        else:
            learned_tokens = reshaped.mean(dim=2)
        per_frame_embeddings.append(learned_tokens)
            
        action_model_input = [
            torch.cat([text_emb.repeat(1, 1, 2), pfe], dim=1)
            for pfe in per_frame_embeddings
        ]
        action_cls_tokens = [action_model(ami) for ami in action_model_input]
        return h, action_cls_tokens
    
    
    encoder.eval()
    predictor.eval()
    action_model.eval()
    policy_model.train()
    
    print(len(train_dataloader))
    for epoch in range(1):
        for obs, actions, idx in train_dataloader:
            obs_reshaped = obs.permute(0, 4, 1, 2, 3).to(device) # B x 3 x 16 x 224 x 224
            latent_obs, latent_actions = vjepa_forward_pass(clips=obs_reshaped, p=16, text_emb=text_embeddings)
            latent_actions = latent_actions[0]
            
            if policy_learning_model == 'action_mlp':
                policy_output = policy_model(latent_actions[:, 0, :])    
            elif policy_learning_model == 'video_jepa_cnn':
                policy_output = policy_model(latent_obs.reshape(-1, 8, 1024, 14, 14))
            elif policy_learning_model == 'video_cnn':
                policy_output = policy_model(obs_reshaped.permute(0, 2, 1, 3, 4))
            elif policy_learning_model == 'cnn':
                pass
            else:
                random_input = torch.randn_like(latent_actions)
                policy_output = policy_model(random_input[:, 0, :])
            
            loss = policy_loss(policy_output, actions.to(device))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            wandb.log({'loss': loss.item()})

    
    # perform evaluation
    policy_model.eval()
    with torch.no_grad():
        for obs, actions, idx in eval_dataloader:
            obs_reshaped = obs.permute(0, 4, 1, 2, 3).to(device)
            latent_obs, latent_actions = vjepa_forward_pass(clips=obs_reshaped, p=16, text_emb=text_embeddings)
            latent_actions = latent_actions[0]
            
            if policy_learning_model == 'action_mlp':
                policy_output = policy_model(latent_actions[:, 0, :])    
            elif policy_learning_model == 'video_jepa_cnn':
                policy_output = policy_model(latent_obs.reshape(-1, 8, 1024, 14, 14))
            elif policy_learning_model == 'video_cnn':
                policy_output = policy_model(obs_reshaped.permute(0, 2, 1, 3, 4))
            elif policy_learning_model == 'cnn':
                pass
            else:
                random_input = torch.randn_like(latent_actions)
                policy_output = policy_model(random_input[:, 0, :])
            
            loss = policy_loss(policy_output, actions.to(device))
            wandb.log({'eval_loss': loss.item()})
    
    
    
if __name__ == '__main__':
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--config', type=str, required=True)
    args = argparser.parse_args()
    main(args.config)