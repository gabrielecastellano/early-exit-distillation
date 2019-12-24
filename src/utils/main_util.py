import builtins as __builtin__
import json
import os

import torch
import torch.distributed as dist


def overwrite_dict(org_dict, sub_dict):
    for sub_key, sub_value in sub_dict.items():
        if sub_key in org_dict:
            if isinstance(sub_value, dict):
                overwrite_dict(org_dict[sub_key], sub_value)
            else:
                org_dict[sub_key] = sub_value
        else:
            org_dict[sub_key] = sub_value


def overwrite_config(config, json_str):
    overwrite_dict(config, json.loads(json_str))


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def init_distributed_mode(world_size=1, dist_url='env://'):
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        device_id = int(os.environ['LOCAL_RANK'])
    elif 'SLURM_PROCID' in os.environ:
        rank = int(os.environ['SLURM_PROCID'])
        device_id = rank % torch.cuda.device_count()
    else:
        print('Not using distributed mode')
        return False, None

    torch.cuda.set_device(device_id)
    dist_backend = 'nccl'
    print('| distributed init (rank {}): {}'.format(rank, dist_url), flush=True)
    torch.distributed.init_process_group(backend=dist_backend, init_method=dist_url,
                                         world_size=world_size, rank=rank)
    torch.distributed.barrier()
    setup_for_distributed(rank == 0)
    return True, [device_id]


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0
