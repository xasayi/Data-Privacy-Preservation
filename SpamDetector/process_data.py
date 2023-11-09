from SpamDetector.process_email import get_body_dic, combine_data
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.utils.class_weight import compute_class_weight
import os
import sys

def get_df(filename, index=None):
    with open(filename) as f:
        content = f.read()
    lines = content.split('\n')
    lines = [i.split('\t') for i in lines][:index]
    df = {'data': [i[1] for i in lines], 'label': [0 if i[0] == 'ham' else 1 for i in lines]}
    return df

def split_data(df, valtest_size):
    train_data, temp_data, train_labels, temp_labels = train_test_split(df['data'], df['label'], 
                                                                      random_state=44, test_size=valtest_size, 
                                                                      stratify=df['label'])
    val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels, 
                                                                  random_state=44, test_size=0.5, 
                                                                  stratify=temp_labels)
    return (train_data, train_labels), (val_data, val_labels), (test_data, test_labels)
 
def tokenize_data_and_get_weights(tokenizer, data, max_seq, batch_size, type='train'):
    tokenizer.pad_token = '[PAD]'
    tokenized_data = tokenizer.batch_encode_plus(data[0], max_length=max_seq,
                                               padding=True, truncation=True, return_token_type_ids=False)
    data_seq = torch.tensor(tokenized_data['input_ids'])
    data_mask = torch.tensor(tokenized_data['attention_mask'])
    data_y = torch.tensor(data[1])

    if type in ['train', 'valid']:
        datas = TensorDataset(data_seq, data_mask, data_y)
        sampler = RandomSampler(datas) if type == 'train' else SequentialSampler(datas)
        dataloader = DataLoader(datas, sampler=sampler, batch_size=batch_size)
        weight = get_weights(data[1]) if type == 'train' else None
        return dataloader, weight
    return (data_seq, data_mask, data_y)

def get_weights(labels):
    class_wts = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    weights= torch.tensor(class_wts,dtype=torch.float)
    return weights

def process_data(tokenizer, splits, batch_size, file_name, index, sms, easy):
    if sms:
        df = get_df(file_name, index)
    else:
        ham = get_body_dic("data/emailSpam/easy_ham/", 0) if easy else get_body_dic("data/emailSpam/hard_ham/", 0)
        spam = get_body_dic("data/emailSpam/spam/", 1)
        df = combine_data(ham, spam)

    train, valid, test = split_data(df, splits)
    seq_len = [len(i.split()) for i in train[0]]
    max_seq_len = min(int(np.ceil((pd.Series(seq_len).describe()['75%']) / 5) * 5), 100)

    train_dataloader, train_weights = tokenize_data_and_get_weights(tokenizer, train, max_seq_len, batch_size, type='train')
    valid_dataloader= tokenize_data_and_get_weights(tokenizer, valid, max_seq_len, batch_size, type='valid')[0]
    test_data = tokenize_data_and_get_weights(tokenizer, test, max_seq_len, batch_size, type='test')

    return train_dataloader, valid_dataloader, test_data, train_weights