# File adapted from https://github.com/NeuromorphicComputationResearchProgram/MalConv2/blob/main/binaryLoader.py
import os
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils import data



class BinaryDataset(data.Dataset):
    def __init__(self, mal_dir, md5_labels, num_labels, max_len=4000000):
        """Class implementing a dataset for directories of malicious files.
        
        Arguments:
        mal_dir -- Directory containing malicious files, or subdirectories of
                   malicious files (to traverse recursively)
        md5_labels -- Dict mapping md5 hashes to labels
        num_labels -- Total number of labels
        max_len -- The maximum number of bytes to read from a file
        """

        self.all_files = []
        self.max_len = max_len
        self.num_labels = num_labels
        for root, dirs, files in os.walk(mal_dir):
            for file_name in files:
                md5 = file_name[:-32]
                if md5_labels.get(md5) is None:
                    continue
                file_path = os.path.join(root, file_name)
                labels = md5_labels[md5]
                self.all_files.append((file_path, labels, None))

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, index):
        to_load, labels, _ = self.all_files[index]
        with open(to_load, 'rb') as f:
            x = f.read(self.max_len)
            x = np.frombuffer(x, dtype=np.uint8).astype(np.int16)+1
        x = torch.tensor(x)
        y = torch.zeros(self.num_labels)
        for label in labels:
            y[label] = 1
        return x, y
    

class RandomChunkSampler(torch.utils.data.sampler.Sampler):
    def __init__(self, data_source, batch_size):
        """
        Samples random "chunks" of a dataset, so that items within a chunk
        are always loaded together. Useful to keep chunks in similar size
        groups to reduce runtime. 

        data_source - The souce pytorch dataset object
        batch_size - The size of the chunks to keep together. Should 
                     generally be set to the desired batch size during
                     training to minimize runtime. 
        """

        self.data_source = data_source
        self.batch_size = batch_size
        
    def __iter__(self):
        n = len(self.data_source)
        data = [x for x in range(n)]
        blocks = [data[i:i+self.batch_size] for i in range(0,len(data),
                                                           self.batch_size)]
        random.shuffle(blocks)
        data[:] = [b for bs in blocks for b in bs]
        return iter(data)
        
    def __len__(self):
        return len(self.data_source)


def pad_collate_func(batch):
    """
    This should be used as the collate_fn=pad_collate_func for a pytorch
    DataLoader object in order to pad out files in a batch to the length of
    the longest item in the batch. 
    """

    vecs = [x[0] for x in batch]
    y = torch.stack([x[1] for x in batch])
    x = torch.nn.utils.rnn.pad_sequence(vecs, batch_first=True)
    return x, y
