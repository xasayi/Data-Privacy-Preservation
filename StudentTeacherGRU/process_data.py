import torch
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/sarinaxi/Desktop/Thesis')
from SpamDetector.process_data import split_data, get_weights
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

def get_data(filename, downsample):
    with open(filename) as f:
        content = f.read()
    lines = content.split('\n')
    lines = np.array([i.split('\t') for i in lines][:-2])
    
    df = pd.DataFrame(lines, columns=['type', 'data']).drop_duplicates()
    ham = df[df['type']=='ham']
    spam = df[df['type']=='spam']
    if downsample:
        ham = ham.sample(n = len(spam), random_state = 44)
    df = pd.concat([ham, spam]).reset_index(drop=True)
    ret = df.reindex(np.random.permutation(df.index))
    ret['label']= ret['type'].map({'ham': 0, 'spam': 1})
    return ret

def tokenize(tokenizer, data, max_len, padding_type, trunc_type, bs, type):
    training_sequences = tokenizer.texts_to_sequences(data[0])
    training_padded = pad_sequences(training_sequences, maxlen = max_len, padding = padding_type, truncating = trunc_type)
    if type in ['train', 'valid']:
        datas = TensorDataset(torch.tensor(training_padded), torch.tensor(list(data[1])))
        sampler = SequentialSampler(datas)
        dataloader = DataLoader(datas, sampler=sampler, batch_size=bs)
        weight = get_weights(data[1]) if type == 'train' else None
        return dataloader, weight
    return (training_padded, data[1])

def process_data(filename, vocab_size, splits, bs, max_seq, padding_type, trunc_type):
    dic1 = get_data(filename, True)
    tokenizer = Tokenizer(num_words = vocab_size, char_level=False, oov_token = "<OOV>")
    tokenizer.fit_on_texts(dic1['data'])
    
    train, valid, test = split_data(dic1, splits)
    train_dataloader, train_weight = tokenize(tokenizer, train, max_seq, padding_type, trunc_type, bs, 'train')
    valid_dataloader = tokenize(tokenizer, valid, max_seq, padding_type, trunc_type, bs, 'valid')[0]
    test_data = tokenize(tokenizer, test, max_seq, padding_type, trunc_type, bs, 'test')

    return train_dataloader, train_weight, valid_dataloader, test_data