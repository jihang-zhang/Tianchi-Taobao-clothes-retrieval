import torch
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SequentialSampler, RandomSampler

from dataset import TaobaoTrainSet
from sampler import RandomIdentitySampler, GroupedBatchSampler

def collate_fn(batch):

    imgs, pids, c = zip(*batch)
    pids = torch.tensor(pids, dtype=torch.int64)
    c = torch.tensor(c, dtype=torch.int64)
    return torch.stack(imgs, dim=0), pids, c

def make_dataloader(json_list, batch_size, scale, mode):
        
    dataset = TaobaoTrainSet(json_list, scale, mode)
    aspect_ids = [d['aspect_group'] for d in dataset.json_list]

    if mode == 'train':
        sampler = RandomSampler(dataset)
    elif mode == 'valid':
        sampler = SequentialSampler(dataset)

    gb_sampler = GroupedBatchSampler(
        sampler=sampler, 
        group_ids=aspect_ids,
        batch_size=batch_size,
        drop_uneven=(mode=='train'))

    return DataLoader(dataset, batch_sampler=gb_sampler, collate_fn=collate_fn)