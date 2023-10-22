import numpy as np
from scipy import sparse
from torch.utils.data.dataset import Dataset
import os
from tqdm import tqdm
import pandas as pd
import torch


class SparseTensorDataset(Dataset):
    r"""Sparse Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Args:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors):
        assert all(tensors[0].shape[0] == tensor.shape[0] for tensor in tensors), "Size mismatch between tensors"
        self.tensors = tensors

    def __getitem__(self, index):
        return tuple(tensor[index] for tensor in self.tensors)

    def __len__(self):
        return self.tensors[0].shape[0]


def collate_fn_for_sparse_data_loader(batch):
    length = len(batch[0])
    tensors = []
    for index in range(length):
        tensor = [x[index] for x in batch]
        if type(tensor[0]) == sparse.csr_matrix:
            tensor = sparse.vstack(tensor)
        elif type(tensor[0]) == np.ndarray:
            tensor = np.vstack(tensor)
        # else:
        #     raise ValueError("Unsupported type of data!")
        tensors.append(tensor)
    if len(tensors) == 1: return tensors[0]
    return tensors


def createDataCSV(path):
    labels = []
    texts = []
    dataType = []
    label_map = {}
    with open(os.path.join(path, 'x_train.txt')) as f:
        for i in tqdm(f):
            texts.append(i.replace('\n', ''))
            dataType.append('train')
    with open(os.path.join(path, 'x_test.txt')) as f:
        for i in tqdm(f):
            texts.append(i.replace('\n', ''))
            dataType.append('test')
    with open(os.path.join(path, 'y_train.txt')) as f:
        for i in tqdm(f):
            for l in i.replace('\n', '').split():
                label_map[l] = 0
            labels.append(i.replace('\n', ''))
    with open(os.path.join(path, 'y_test.txt')) as f:
        for i in tqdm(f):
            for l in i.replace('\n', '').split():
                label_map[l] = 0
            labels.append(i.replace('\n', ''))
    if os.path.exists(os.path.join(path, 'x_val.txt')):
        with open(os.path.join(path, 'x_val.txt')) as f:
            for i in tqdm(f):
                texts.append(i.replace('\n', ''))
                dataType.append('val')
        with open(os.path.join(path, 'y_val.txt')) as f:
            for i in tqdm(f):
                for l in i.replace('\n', '').split():
                    label_map[l] = 0
                labels.append(i.replace('\n', ''))

    assert len(texts) == len(labels) == len(dataType)
    df_row = {'text': texts, 'label': labels, 'dataType': dataType}
    for i, k in enumerate(sorted(label_map.keys())):
        label_map[k] = i
    df = pd.DataFrame(df_row)
    print('label map', len(label_map))
    return df, label_map


class MDataset(Dataset):
    def __init__(self, df, mode, tokenizer, label_map, max_length,
                 token_type_ids=None):
        assert mode in ["train", "val", "test"]
        self.mode = mode
        self.df, self.n_labels, self.label_map = df[df.dataType == self.mode], len(label_map), label_map
        self.len = len(self.df)
        self.tokenizer, self.max_length = tokenizer, max_length
        self.multi_group = False
        self.token_type_ids = token_type_ids

    def __getitem__(self, idx):
        max_len = self.max_length
        review = self.df.text.values[idx].lower()
        labels = [self.label_map[i] for i in self.df.label.values[idx].split() if i in self.label_map]

        review = ' '.join(review.split()[:max_len])

        text = review
        if self.token_type_ids is not None:
            input_ids = self.token_type_ids[idx]
            if input_ids[-1] == 0:
                input_ids = input_ids[input_ids != 0]
            input_ids = input_ids.tolist()
        elif hasattr(self.tokenizer, 'encode_plus'):
            input_ids = self.tokenizer.encode(
                'filling empty' if len(text) == 0 else text,
                add_special_tokens=True,
                max_length=max_len,
                truncation=True
            )
        else:
            # fast
            input_ids = self.tokenizer.encode(
                'filling empty' if len(text) == 0 else text,
                add_special_tokens=True
            ).ids

        if len(input_ids) == 0:
            print('zero string')
            assert 0
        if len(input_ids) > self.max_length:
            input_ids[self.max_length - 1] = input_ids[-1]
            input_ids = input_ids[:self.max_length]

        attention_mask = [1] * len(input_ids)
        token_type_ids = [0] * len(input_ids)

        padding_length = self.max_length - len(input_ids)
        input_ids = input_ids + ([0] * padding_length)
        attention_mask = attention_mask + ([0] * padding_length)
        token_type_ids = token_type_ids + ([0] * padding_length)

        input_ids = torch.tensor(input_ids)
        attention_mask = torch.tensor(attention_mask)
        token_type_ids = torch.tensor(token_type_ids)

        label_ids = torch.zeros(self.n_labels)
        label_ids = label_ids.scatter(0, torch.tensor(labels),
                                      torch.tensor([1.0 for i in labels]))
        return input_ids, attention_mask, token_type_ids, label_ids

    def __len__(self):
        return self.len
