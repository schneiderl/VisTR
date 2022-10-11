import torch
from torch.utils.data import DataLoader, DistributedSampler
from datasets import build_dataset, get_coco_api_from_dataset
import util.misc as utils

dataset_train = build_dataset(image_set='train')

sampler_train = torch.utils.data.RandomSampler(dataset_train)
batch_sampler_train = torch.utils.data.BatchSampler(sampler_train, 2, drop_last=True)

        

data_loader_train = DataLoader(dataset_train, batch_sampler=batch_sampler_train,
                               collate_fn=utils.collate_fn, num_workers=1)
