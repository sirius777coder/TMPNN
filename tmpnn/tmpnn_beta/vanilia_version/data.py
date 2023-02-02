from torch.utils.data import Dataset
import numpy as np
import json
import time
import copy
import random
import utils

# Modified
# 1026:max_length = 1300(>1280)

class StructureDataset(Dataset):
    def __init__(self, jsonl_file, verbose=True, truncate=None, max_length=1300,alphabet='ACDEFGHIKLMNPQRSTVWYX-'):
        alphabet_set = set([a for a in alphabet])
        discard_count = {
            'bad_chars': 0,
            'too_long': 0,
            'bad_seq_length': 0
        }
        pdb_dict_list = utils.load_jsonl(jsonl_file)
        self.data = []
        start = time.time()
        for i, entry in enumerate(pdb_dict_list):
            seq = entry['seq']
            name = entry['name']

            bad_chars = set([s for s in seq]).difference(alphabet_set)
            if len(bad_chars) == 0:
                if len(entry['seq']) <= max_length:
                    self.data.append(entry)
                else:
                    discard_count['too_long'] += 1
            else:
                discard_count['bad_chars'] += 1

            # Truncate early
            if truncate is not None and len(self.data) == truncate:
                return

            if verbose and (i + 1) % 1000 == 0:
                elapsed = time.time() - start

            #print('Discarded', discard_count)
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


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
