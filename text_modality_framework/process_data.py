'''
Process and token the data fed into the model
'''
import random
import torch
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer, BertForSequenceClassification

def get_data(filename, map_, downsample=False):
    '''Get the data from the file'''
    df = pd.read_csv(filename)
    df = df.drop_duplicates(subset=['data']).reset_index(drop=True)

    if downsample:
        maps = []
        for i in map_:
            maps.append(df[df['type']==i])
        min_len = min([len(maps[j]) for j in range(len(maps))])

        for i in range(len(maps)):
            maps[i] = maps[i].sample(n=min_len)
        df = pd.concat(maps).reset_index(drop=True)

    ret = df.reindex(np.random.permutation(df.index))
    ret['label']= ret['type'].map(map_)
    print(f"Dataset contains {len(ret['label'])} samples.")
    return ret[['data', 'label', 'type']].reset_index(drop=True)

def split_data(df, pre_train):
    '''Split the data into train, valid, and test depending on whether it is used for
    pre-training or fine-tuning'''
    size = len(df['label'])
    if pre_train:
        print('Prepare Data for Pre-training')
        train_data, temp_data, train_labels, temp_labels = train_test_split(df['data'], df['label'],
                                                                          test_size=0.1,
                                                                          stratify=df['label'])
        val_data, test_data, val_labels, test_labels = train_test_split(temp_data, temp_labels,
                                                                        test_size=0.5,
                                                                        stratify=temp_labels)
    else:
        print('Prepare Data for Fine-tuning')
        first_80 = pd.concat((df[:-int(size*0.6)], df[-int(size*0.4):]))
        last_20 = df[-int(size*0.6):-int(size*0.4)]
        class_nums = len(df['label'].unique())
        if class_nums > 10:
            classes = random.sample(list(np.linspace(0, class_nums-1, class_nums)), 2)
        else:
            classes = list(df['label'].unique())
        print(f'Target classes are {classes}')
        sensitive = last_20[last_20['label'].isin(classes)]
        val_data, test_data, val_labels, test_labels = train_test_split(sensitive['data'],
                                                                        sensitive['label'],
                                                                        test_size=0.4,
                                                                        stratify=sensitive['label'])
        sensitive_ind = last_20.index.isin(sensitive.index)
        remaining_last_20 = last_20[~sensitive_ind]
        insensitive = pd.concat([first_80, remaining_last_20])
        train_data, train_labels = insensitive['data'], insensitive['label']

        for i in classes:
            print(len(test_labels[test_labels==i]))
    print(f'Train: {len(train_data)} | Valid: {len(val_data)} | Test: {len(test_data)}')
    train = (train_data.reset_index(drop=True), train_labels.reset_index(drop=True))
    test = (test_data.reset_index(drop=True), test_labels.reset_index(drop=True))
    valid = (val_data.reset_index(drop=True), val_labels.reset_index(drop=True))
    return train, valid, test

def get_weights(labels):
    '''Get the class weights for the data'''
    class_wts = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    weights= torch.tensor(class_wts,dtype=torch.float)
    return weights

def tokenize(tokenizer, data, max_len, bs, type_, sampler, pre_train):
    '''tokenize the data and return the data loaders'''
    tokenizer.pad_token = '[PAD]'
    tokenized_data = tokenizer.batch_encode_plus(data[0], max_length=max_len,
                                                 padding=True, truncation=True,
                                                 return_token_type_ids=False)
    data_seq = torch.tensor(tokenized_data['input_ids'])
    data_y = torch.tensor(data[1])
    weight = get_weights(data[1]) if type_ == 'train' else None
    if not pre_train:
        return (data_seq, data_y), weight
    if type_ in ['train', 'valid']:
        datas = TensorDataset(data_seq, data_y)
        sampler = sampler(datas)
        dataloader = DataLoader(datas, sampler=sampler, batch_size=bs)
        return dataloader, weight
    return (data_seq, data_y), None

def euclidean(a, b):
    '''Calculate euclidean distance between a and b'''
    return np.linalg.norm(a-b)

def cos_sim(a, b):
    '''calculate cosine similarity between a and b'''
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def custom(a, b):
    '''Custom similarity function between a and b '''
    const = 4
    x = cos_sim(a, b)
    return const*((1-np.exp(x-1))/np.exp(x-1))

def custom_dissimilar(a, b):
    '''Custom dissimilarity function between a and b'''
    const = 4
    x = cos_sim(a, b)
    return const*(np.exp(x)-1)

def inverse_cos_sim(a, b):
    '''Inverse cosine similarity between a and b'''
    return 1/np.exp(np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b))

def function_dissim(emb_x, emb_y, eps):
    '''Function to calculate dissimilarity score between emb_x and emb_y defined by epsilon'''
    return np.exp(-eps*abs(custom_dissimilar(emb_x,emb_y)))

def function(emb_x, emb_y, eps):
    '''Function to calculate similarity score between emb_x and emb_y defined by epsilon'''
    return np.exp(-eps*abs(custom(emb_x,emb_y)))

def sanitize_prob_dissim(eps, emb_x, emb_y, insensitive_embs):
    '''Function to calculate the prob of replacing emb_x with emb_y based on dissimilarity score'''
    cx = 1/np.sum([function_dissim(emb_x, i, eps) for i in insensitive_embs])
    prob = cx*function_dissim(emb_x, emb_y, eps)
    return prob

def sanitize_prob(eps, emb_x, emb_y, insensitive_embs):
    '''Function to calculate the prob of replacing emb_x with emb_y based on similarity score'''
    cx = 1/np.sum([function(emb_x, i, eps) for i in insensitive_embs])
    prob = cx*function(emb_x, emb_y, eps)
    return prob

def sanitize_data(tokenizer, dataloader, sens_ratio, eps):
    '''Function to sanitize the data based on the sensitivity ratio and epsilon value'''
    voc = dict(tokenizer.vocab)
    voc = {v: k for k, v in voc.items()}

    tok, label = dataloader
    flattened = tok.reshape(tok.shape[0]*tok.shape[1])

    bincount = np.bincount(flattened)/len(flattened)
    order = np.argsort(bincount)[::-1]
    non_zero_len = len(bincount[bincount!=0])
    sensitive_toks = order[int(non_zero_len*(1-sens_ratio)):non_zero_len]
    insensitive_toks = order[:int(non_zero_len*(1-sens_ratio))]
    model=BertForSequenceClassification.from_pretrained('bert-base-uncased')
    embedding_matrix = model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()
    sensitive = dict(zip(sensitive_toks, embedding_matrix[sensitive_toks]))
    insensitive = dict(zip(insensitive_toks, embedding_matrix[insensitive_toks]))
    tot_prob = 0
    count = 0
    for i in range(len(tok)):
        for j in range(len(tok[i])):
            sens_tok = tok[i][j].tolist()
            if sens_tok in sensitive_toks:
                emb_x = sensitive[sens_tok]
                insens_tok = random.sample(list(insensitive_toks), 1)[0]
                emb_y = insensitive[insens_tok]
                if eps != 0:
                    prob = sanitize_prob(eps, emb_x, emb_y, embedding_matrix[insensitive_toks])
                else:
                    prob = 0
                tot_prob += prob
                p = random.random()
                if p < prob*2500:
                    count += 1
                    tok[i][j] = insens_tok

    return tot_prob, count/len(tok)/len(tok[0]), (tok, label)

def sanitize_data_dissim(tokenizer, dataloader, sens_ratio, eps):
    '''Function to sanitize the data based on the sensitivity ratio and epsilon value'''
    voc = dict(tokenizer.vocab)
    voc = {v: k for k, v in voc.items()}

    tok, label = dataloader
    flattened = tok.reshape(tok.shape[0]*tok.shape[1])

    bincount = np.bincount(flattened)/len(flattened)
    order = np.argsort(bincount)[::-1]
    non_zero_len = len(bincount[bincount!=0])
    sensitive_toks = order[int(non_zero_len*(1-sens_ratio)):non_zero_len]
    insensitive_toks = order[:int(non_zero_len*(1-sens_ratio))]
    model=BertForSequenceClassification.from_pretrained('bert-base-uncased')
    embedding_matrix = model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()

    sensitive = dict(zip(sensitive_toks, embedding_matrix[sensitive_toks]))
    insensitive = dict(zip(order, embedding_matrix[order]))
    tot_prob = 0
    count = 0
    for i in range(len(tok)):
        for j in range(len(tok[i])):
            sens_tok = tok[i][j].tolist()
            if sens_tok in sensitive_toks:
                emb_x = sensitive[sens_tok]
                insens_tok = random.sample(list(insensitive_toks), 1)[0]
                emb_y = insensitive[insens_tok]
                if eps != 0:
                    prob = sanitize_prob_dissim(eps, emb_x, emb_y, embedding_matrix)
                else:
                    prob = 0
                tot_prob += prob
                p = random.random()
                if p < 2500*prob:
                    count += 1
                    tok[i][j] = insens_tok

    return tot_prob, count/len(tok)/len(tok[0]), (tok, label)

def process_data(name, map, pre_train, sequence_len, bs, sampler, bert, data=None):
    '''Function that uses helper functions to process the data and return the data loaders'''
    if data is None:
        data = get_data(filename=name, map_=map)

    train, valid, test = split_data(df=data, pre_train=pre_train)
    tokenizer = BertTokenizer.from_pretrained(bert)

    seq_len = [len(i.split()) for i in train[0]]
    seq_len = int(np.ceil((pd.Series(seq_len).describe()['75%']) / 5) * 5)
    if not sequence_len:
        sequence_len = seq_len
    print(f'Set sequence length is: {sequence_len} | 75% data sequence length is: {seq_len}')
    train_dataloader, weight = tokenize(tokenizer, train, sequence_len, bs, 'train',
                                        sampler, pre_train)
    valid_dataloader = tokenize(tokenizer, valid, sequence_len, bs, 'valid', sampler, pre_train)[0]
    test_data = tokenize(tokenizer, test, sequence_len, bs, 'test', sampler, pre_train)[0]
    return train_dataloader, valid_dataloader, test_data, weight
