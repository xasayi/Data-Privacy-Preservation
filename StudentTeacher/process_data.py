import pickle
import torch
import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '/Users/sarinaxi/Desktop/Thesis')
from SpamDetector.process_data import split_data, get_weights
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import BertTokenizerFast

def load_data(file_name):
    open_file = open(file_name, "rb")
    dic = pickle.load(open_file)
    open_file.close()

    private = np.array(dic['private'])
    public = np.array(dic['public'])
    similar = np.array(dic['similar'])
    index = similar[:, 0, 0].astype(int)
    indexs = similar[:, 0, :].astype(int)
    print(private)
    print(public[indexs])
    pair_public = public[index]

    private_dic = {'data': list(private[:, 0]), 'label': list(private[:, 1].astype(int))}
    public_dic = {'data': list(pair_public[:, 0]), 'label': list(pair_public[:, 1].astype(int))}
    return private_dic, public_dic

def tokenize_data_and_get_weights(tokenizer, data, max_seq, batch_size, type='train'):
    tokenizer.pad_token = '[PAD]'
    tokenized_data = tokenizer.batch_encode_plus(data[0], max_length=max_seq,
                                               padding=True, truncation=True, return_token_type_ids=False)
    data_seq = torch.tensor(tokenized_data['input_ids'])
    data_mask = torch.tensor(tokenized_data['attention_mask'])
    data_y = torch.tensor(data[1])

    if type in ['train', 'valid']:
        datas = TensorDataset(data_seq, data_mask, data_y)
        sampler = SequentialSampler(datas)
        dataloader = DataLoader(datas, sampler=sampler, batch_size=batch_size)
        weight = get_weights(data[1]) if type == 'train' else None
        return dataloader, weight
    return (data_seq, data_mask, data_y)

def process_data(tokenizer, splits, batch_size, data):
    train, valid, test = split_data(data, splits)
    seq_len = [len(i.split()) for i in train[0]]
    max_seq_len = min(int(np.ceil((pd.Series(seq_len).describe()['75%']) / 5) * 5), 100)

    train_dataloader, train_weights = tokenize_data_and_get_weights(tokenizer, train, max_seq_len, batch_size, type='train')
    valid_dataloader = tokenize_data_and_get_weights(tokenizer, valid, max_seq_len, batch_size, type='valid')[0]
    test_data = tokenize_data_and_get_weights(tokenizer, test, max_seq_len, batch_size, type='test')

    return train_dataloader, valid_dataloader, test_data, train_weights

if __name__ == '__main__':
    file_name = "/Users/sarinaxi/Desktop/Thesis/StudentTeacher/sim.pkl"
    private, public = load_data(file_name)
    
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
    pri_train_loader, pri_val_loader, pri_test, pri_weight = process_data(tokenizer, 0.4, 32, private)
    pub_train_loader, pub_val_loader, pub_test, pub_weight = process_data(tokenizer, 0.4, 32, public)