# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#

import argparse
import multiprocessing as mp
import pprint
import yaml
import os
import logging
import torch.distributed as dist
import torch
from app.scaffold import main as app_main
from src.utils.logging import get_logger
from src.utils.distributed import init_distributed

parser = argparse.ArgumentParser()
parser.add_argument(
    '--fname', type=str,
    help='name of config file to load',
    default='configs.yaml')
parser.add_argument(
    '--devices', type=str, nargs='+', default=['cuda:0'],
    help='which devices to use on local machine')
parser.add_argument(
    '--train_script', type=str, default='train',
    help='which script to run (train or train_causal_mask)')

def process_main(rank, fname, world_size, devices, train_script):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'
    os.environ['WORLD_SIZE'] = str(world_size)
    os.environ['RANK'] = str(rank)

    local_gpu_id = devices[rank].split(":")[-1]
    os.environ["CUDA_VISIBLE_DEVICES"] = local_gpu_id

    import logging
    from src.utils.logging import get_logger
    logger = get_logger(force=True)
    if rank == 0:
        logger.setLevel(logging.INFO)
    else:
        logger.setLevel(logging.ERROR)

    logger.info(f'called-params {fname}')

    # Load config
    params = None
    with open(fname, 'r') as y_file:
        params = yaml.load(y_file, Loader=yaml.FullLoader)
        logger.info('loaded params...')

    # Log config
    if rank == 0:
        pprint.PrettyPrinter(indent=4).pprint(params)
        dump = os.path.join(params['logging']['folder'], 'params-pretrain.yaml')
        with open(dump, 'w') as f:
            yaml.dump(params, f)

    # Init distributed (access to comm between GPUS on same machine)
    world_size, rank = init_distributed(rank_and_world_size=(rank, world_size))
    logger.info(f'Running... (rank: {rank}/{world_size})')

    # Launch the app with loaded config
    app_main(params['app'], args=params, train_script=train_script)


if __name__ == '__main__':
    args = parser.parse_args()
    num_gpus = len(args.devices)
    mp.set_start_method('spawn')
    
    for rank in range(num_gpus):
        mp.Process(
            target=process_main,
            args=(rank, args.fname, num_gpus, args.devices, args.train_script)
        ).start()
