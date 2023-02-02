from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import Sampler as Sampler
import torch
import torch.nn as nn
import numpy as np
import json
import time
import copy
import random
import utils

restypes = [
    "A",
    "R",
    "N",
    "D",
    "C",
    "Q",
    "E",
    "G",
    "H",
    "I",
    "L",
    "K",
    "M",
    "F",
    "P",
    "S",
    "T",
    "W",
    "Y",
    "V",
]
restype_order = {restype: i for i, restype in enumerate(restypes)}

class StructureDataset(Dataset):
    def __init__(self,jsonl_file,max_length=500,low_fraction=0.7,high_fraction=0.9):
        dataset = utils.load_jsonl(jsonl_file)
        for i in dataset:
            if i['name'].startswith("AF"):
                i['name'] += "_A"
            i['cctop'] = i['cctop'].replace("T","S") #替换减少一类
        cctop_code = 'IMOULS'
        self.data = []
        self.discard = {"bad_chars":0,"too_long":0}
        for entry in dataset:
            name = entry['name']
            seq = entry['seq']
            cctop = entry['cctop']
            length = torch.tensor([len(seq)],dtype=torch.long)
            # Check if in alphabet
            bad_chars = set([s for s in seq]).difference(utils.restypes)
            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length:
                    pass
                else:
                    self.discard['too_long'] += 1
                    continue
            else:
                # print(entry['name'], bad_chars, entry['seq'])
                self.discard['bad_chars'] += 1
                continue
            seq = torch.tensor([restype_order[i] for i in seq],dtype=torch.long)
            cctop = torch.tensor([cctop_code.index(i) for i in cctop],dtype=torch.long)
            coord = torch.from_numpy(np.stack(list(entry['coords'].values()),axis=-2))
            coord = coord.to(torch.float32)
            coord = utils.nan_to_num(coord) # remove the nan value
            seq_mask_fraction = torch.tensor([np.random.uniform(low=low_fraction, high=high_fraction),],dtype=torch.float32)
            seq_mask = []
            for _ in range(len(seq)):
                if np.random.random() < seq_mask_fraction:
                    seq_mask.append(False)
                else:
                    seq_mask.append(True)
            seq_mask = torch.tensor(seq_mask,dtype=torch.bool) # 0.0, mask; 1 unmask
            mask_seq = copy.deepcopy(seq)
            mask_seq[~seq_mask] = 20

            self.data.append({
                "name":name,
                "coord":coord,
                "seq":seq,
                "cctop":cctop,
                "seq_mask":seq_mask,
                "mask_seq":mask_seq,
                "seq_mask_fraction":seq_mask_fraction,
                "length":length
            })
            # X, S, C, mask, lengths, S_mask
        print(f"UNK token:{self.discard['bad_chars']},too long:{self.discard['too_long']}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self,idx):
        return self.data[idx]


def batch_collate_function(batch):
    """
    A customized wrap up collate function
    Args:
        batch: a list of structure objects
    Shape:
        Output:
            coord_batch [B, 5, 4, 3] dtype=float32
            seq_batch   [B, L]       dtype=int64
            bert_mask_fraction_batch [B,] dtype=float32
            bert_mask_batch          [B, L]  dtype=torch.bool   0 represents mask, 1 no mask
            padding_mask_batch       [B, L]  dtype=torch.float32   0 represents mask, 1 no mask
    """
    coord_batch = utils.CoordBatchConverter.collate_dense_tensors([i['coord'] for i in batch],0.0)
    seq_batch = utils.CoordBatchConverter.collate_dense_tensors([i['seq'] for i in batch],21)
    cctop_batch = utils.CoordBatchConverter.collate_dense_tensors([i['cctop'] for i in batch],0)
    mask_seq_batch = utils.CoordBatchConverter.collate_dense_tensors([i['mask_seq'] for i in batch],0)
    padding_mask_batch = seq_batch!=21 # True not mask, False represents mask
    seq_batch[~padding_mask_batch] = 0 # padding to 0
    padding_mask_batch = padding_mask_batch.to(torch.float32)
    length_batch = utils.CoordBatchConverter.collate_dense_tensors([i['length'] for i in batch],0)
    output = {
        "coord":coord_batch,
        "seq":seq_batch,
        "mask_seq":mask_seq_batch,
        "mask":padding_mask_batch,
        "cctop":cctop_batch,
        "length":length_batch
    }
    return output
    
def StructureDataloader(dataset,batch_size,num_workers=0,shuffle=True):
    """
    A wrap up dataloader,the batch_size is the number of sequences
    """
    return DataLoader(dataset,batch_size=batch_size,num_workers=num_workers,shuffle=shuffle,collate_fn=batch_collate_function)

def StructureTokenloader(dataset,batch_size,num_workers=0,shuffle=True):
    """
    A wrap up batch token dataloader,the batch_size is the number of tokens
    """
    lengths = [len(i['seq']) for i in dataset]
    return DataLoader(dataset,num_workers=num_workers,batch_sampler=StructureBatchSampler(lengths,batch_size=batch_size,shuffle=shuffle),collate_fn=batch_collate_function)

class StructureBatchSampler(Sampler):
    def __init__(self,lengths,batch_size,shuffle=True):
        self.lengths = lengths
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths)
        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = self.lengths[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)
    
    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            yield b_idx


class DistributedStructureBatchSampler(torch.utils.data.distributed.DistributedSampler):
    def __init__(self,dataset,batch_size,num_replicas,rank,shuffle=True):
        super().__init__(dataset,num_replicas,rank,shuffle)
        """
        the dataset here is just the length
        """
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.dataset)
        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix:
            size = self.dataset[ix]
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0:
            clusters.append(batch)
        self.clusters = clusters

    def __len__(self):
        return len(self.clusters)
    
    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)
        #print('here is pytorch code and you can delete it in the /home/lzk/anaconda3/lib/python3.7/site-packages/torch/utils/data')
        if self.shuffle:
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += indices[:(self.total_size - len(indices))]
        assert len(indices) == self.total_size

        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        # 然后我也要拿到每个数据的长度 (每个rank不同)
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters:
            yield b_idx

class StructureLoader:
    def __init__(self, dataset, batch_size=10000, shuffle=True,
                 collate_fn=lambda x: x, drop_last=False):
        self.dataset = dataset
        self.size = len(dataset)
        self.lengths = [dataset[i]['length'] for i in range(self.size)]
        self.batch_size = batch_size
        sorted_ix = np.argsort(self.lengths) # 默认原来的dataset有长有短，现在排序之后更方便batch

        # Cluster into batches of similar sizes
        clusters, batch = [], []
        batch_max = 0
        for ix in sorted_ix: #遍历dataset的所有样本得到许多小的minibatch
            size = self.lengths[ix] #第ix个样本的长度,也是这个batch里最大的长度(本身sorted_ix就是从小往大排序)
            if size * (len(batch) + 1) <= self.batch_size:
                batch.append(ix)
                batch_max = size
            else:
                clusters.append(batch)
                batch, batch_max = [], 0
        if len(batch) > 0: #最后一个mini batch可能还剩下一点东西
            clusters.append(batch)
        self.clusters = clusters #不同minibatch 组成的token，但最大只能有6000个氨基酸

    def __len__(self):
        return len(self.clusters)

    def __iter__(self):
        np.random.shuffle(self.clusters)
        for b_idx in self.clusters: # 根据这个batch里面的idx再取出原来的样本
            batch = [self.dataset[i] for i in b_idx]
            yield batch