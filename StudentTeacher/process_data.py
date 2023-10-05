import torch
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/sarinaxi/Desktop/Thesis')
from SpamDetector.process_data import split_data, get_weights
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler, RandomSampler
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf

def read_data(filename):
    with open(filename) as f:
        content = f.read()
    lines = content.split('\n')
    lines = np.array([i.split('\t') for i in lines][:-2])
    df = pd.DataFrame(lines, columns=['type', 'data']).drop_duplicates()
    return df

def get_data(df, downsample):
    if downsample:
        ham = df[df['type']=='ham']
        spam = df[df['type']=='spam']
        ham = ham.sample(n = len(spam), random_state = 44)
        df = pd.concat([ham, spam]).reset_index(drop=True)
    ret = df.reindex(np.random.permutation(df.index))
    ret['label']= ret['type'].map({'ham': 0, 'spam': 1})
    return ret

def tokenize(tokenizer, data, max_len, padding_type, trunc_type, bs, type, sampler):
    training_sequences = tokenizer.texts_to_sequences(data[0])
    training_padded = pad_sequences(training_sequences, maxlen = max_len, padding = padding_type, truncating = trunc_type)
    if type in ['train', 'valid']:
        datas = TensorDataset(torch.tensor(training_padded), torch.tensor(list(data[1])))
        sampler = sampler(datas)
        dataloader = DataLoader(datas, sampler=sampler, batch_size=bs)
        weight = get_weights(data[1]) if type == 'train' else None
        return dataloader, weight
    return (training_padded, data[1])

def process_data(df, vocab_size, splits, bs, max_seq, padding_type, trunc_type, downsample, sampler=SequentialSampler):
    dic1 = get_data(df, downsample)

    tokenizer = Tokenizer(num_words = vocab_size, char_level=False, oov_token = "<OOV>")
    tokenizer.fit_on_texts(dic1['data'])
    
    train, valid, test = split_data(dic1, splits)
    train_dataloader, train_weight = tokenize(tokenizer, train, max_seq, padding_type, trunc_type, bs, 'train', sampler)
    valid_dataloader = tokenize(tokenizer, valid, max_seq, padding_type, trunc_type, bs, 'valid', sampler)[0]
    test_data = tokenize(tokenizer, test, max_seq, padding_type, trunc_type, bs, 'test', sampler)

    return train_dataloader, valid_dataloader, test_data, train_weight

if __name__ == '__main__':
    filename = '/Users/sarinaxi/Desktop/Thesis/SpamDetector/data/smsSpam/SMSSpamCollection.txt'

    df = read_data(filename)
    sampler = RandomSampler
    a, b, c, d = process_data(df, 40, 0.3, 16, 40, 'post', 'post', True, sampler)
    print(d)

    #mask = df['type'] == 'ham'
    #ham, spam = df[mask], df[~mask]
    #ham_len, spam_len = len(ham), len(spam)
    #private = pd.concat([ham[:int(ham_len//2)] , spam[:int(spam_len//2)]]).reset_index(drop=True)
    #public = pd.concat([ham[int(ham_len//2):] , spam[int(spam_len//2):]]).reset_index(drop=True)
    #print(df[df['type'] == ham])