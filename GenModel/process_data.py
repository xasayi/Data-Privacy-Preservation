import sys
sys.path.insert(0, '/Users/sarinaxi/Desktop/Thesis')

import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

parent = '/Users/sarinaxi/Desktop/Thesis/GenModel/data/'
paths = {'train': 'train_dataset.npy', 'valid': 'valid_dataset.npy', 'test': 'test_dataset.npy'}

def get_data(filename, index):
    with open(f'/Users/sarinaxi/Desktop/Thesis/{filename}') as f:
        content = f.read()
    data = content.split('\n')[:index]
    return data

def split_data(data, valtest_size):
    train_data, temp_data = train_test_split(data, random_state=2023, test_size=valtest_size)
    val_data, test_data = train_test_split(temp_data, random_state=2023, test_size=0.5)
    return train_data, val_data, test_data

def tokenize_data(tokenizer, data, max_seq, batch_size, type='train'):
    tokenizer.pad_token = '[PAD]'   
    tokenized_data = tokenizer.batch_encode_plus(data, max_length=max_seq, padding=True, truncation=True)

    data_seq = torch.tensor(tokenized_data['input_ids'])
    data_mask = torch.tensor(tokenized_data['attention_mask'])

    if type in ['train', 'valid']:
        datas = TensorDataset(data_seq, data_mask)
        sampler = RandomSampler(datas) if type == 'train' else SequentialSampler(datas)
        dataloader = DataLoader(datas, sampler=sampler, batch_size=batch_size)
        return dataloader
    return (data_seq, data_mask)

def process_data(tokenizer, splits, batch_size, file_name, index):
    data = get_data(file_name, index)
    train, val, test = split_data(data, splits)

    seq_len = [len(i.split()) for i in train]
    max_seq_len = min(int(np.ceil((pd.Series(seq_len).describe()['75%']) / 5) * 5), 100)

    train_dataloader = tokenize_data(tokenizer, train, max_seq_len, batch_size, type='train')
    valid_dataloader= tokenize_data(tokenizer, val, max_seq_len, batch_size, type='valid')
    test_data = tokenize_data(tokenizer, test, max_seq_len, batch_size, type='test')

    return train_dataloader, valid_dataloader, test_data
