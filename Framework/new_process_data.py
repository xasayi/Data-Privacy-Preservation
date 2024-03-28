import json
import torch
import pickle
import random
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from transformers import BertTokenizer, BertForMaskedLM, BertForSequenceClassification
from sklearn.metrics.pairwise import cosine_similarity

def get_data(filename, map_, downsample):
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
        val_data, test_data, val_labels, test_labels = train_test_split(sensitive['data'], sensitive['label'],
                                                                        test_size=0.4,
                                                                        stratify=sensitive['label'])
        sensitive_ind = last_20.index.isin(sensitive.index)
        remaining_last_20 = last_20[~sensitive_ind]
        insensitive = pd.concat([first_80, remaining_last_20])
        train_data, train_labels = insensitive['data'], insensitive['label']
        
        for i in classes:
            print(len(test_labels[test_labels==i]))
    print(f'Train size: {len(train_data)} | Valid size: {len(val_data)} | Test size: {len(test_data)}')
    
    return (train_data.reset_index(drop=True), train_labels.reset_index(drop=True)), (val_data.reset_index(drop=True), val_labels.reset_index(drop=True)), (test_data.reset_index(drop=True), test_labels.reset_index(drop=True))

def get_weights(labels):
    class_wts = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels)
    weights= torch.tensor(class_wts,dtype=torch.float)
    return weights

def tokenize(tokenizer, data, max_len, bs, type, sampler, att, pre_train):
    tokenizer.pad_token = '[PAD]'
    tokenized_data = tokenizer.batch_encode_plus(data[0], max_length=max_len,
                                                 padding=True, truncation=True, 
                                                 return_token_type_ids=False)
    data_seq = torch.tensor(tokenized_data['input_ids'])
    data_mask = torch.tensor(tokenized_data['attention_mask'])
    data_y = torch.tensor(data[1])
    weight = get_weights(data[1]) if type == 'train' else None
    if not pre_train:
        if att: 
            return (data_seq, data_mask, data_y), weight
        else:
            return (data_seq, data_y), weight
    if type in ['train', 'valid']:
        if att:
            datas = TensorDataset(data_seq, data_mask, data_y)
        else:
            datas = TensorDataset(data_seq, data_y)
        sampler = sampler(datas)
        dataloader = DataLoader(datas, sampler=sampler, batch_size=bs)
        return dataloader, weight
    if att:
        return (data_seq, data_mask, data_y), None
    else:
        return (data_seq, data_y), None
    
def euclidean(a, b):
    return np.linalg.norm(a-b)

def inverse_cos_sim(a, b):
    return 1/np.dot(a, b)/np.linalg.norm(a)/np.linalg.norm(b)

def function(emb_x, emb_y, eps):
    return np.exp(-eps*abs(euclidean(emb_x,emb_y)-1)/10)

def sanitize_prob(eps, emb_x, emb_y, insensitive_embs):
    cx = 1/np.sum([function(emb_x, i, eps) for i in insensitive_embs])
    prob = cx*function(emb_x, emb_y, eps)
    return prob

def sanitize_data(tokenizer, dataloader, sens_ratio, eps):
    voc = dict(tokenizer.vocab)
    voc = {v: k for k, v in voc.items()}

    print(dataloader[0])
    tok, label = dataloader
    flattened = tok.reshape(tok.shape[0]*tok.shape[1])
    
    bincount = np.bincount(flattened)/len(flattened)
    order = np.argsort(bincount)[::-1]
    non_zero_len = len(bincount[bincount!=0])
    sensitive_toks = order[int(non_zero_len*(1-sens_ratio)):non_zero_len]
    insensitive_toks = order[:int(non_zero_len*(1-sens_ratio))]
    model=BertForSequenceClassification.from_pretrained('bert-base-uncased')
    embedding_matrix = model.bert.embeddings.word_embeddings.weight.data.cpu().numpy()
    sensitive_embs, insensitive_embs = [], []
    
    for token in sensitive_toks:
        sensitive_embs.append(embedding_matrix[token])
    for token in insensitive_toks:
        insensitive_embs.append(embedding_matrix[token])
    
    sensitive = {k:v for k, v in zip(sensitive_toks, sensitive_embs)}
    insensitive = {k:v for k, v in zip(insensitive_toks, insensitive_embs)}
    #print([voc[tok] for tok in sensitive_toks[-20:]])
    #print([voc[tok] for tok in insensitive_toks[:20]])
    tot_prob = 0
    count = 0
    flag = False
    for i in range(len(tok)):
        sentence = [voc[tok[i][k].tolist()]for k in range(len(tok[i]))]
        if 'adam' in sentence:
            print('Original')
            print([voc[tok[i][k].tolist()]for k in range(len(tok[i]))])
            flag = True
        for j in range(len(tok[i])):
            sens_tok = tok[i][j].tolist()
            if sens_tok in sensitive_toks:
                emb_x = sensitive[sens_tok]
                insens_tok = random.sample(list(insensitive_toks), 1)[0]
                emb_y = insensitive[insens_tok]
                if eps != 0:
                    prob = sanitize_prob(eps, emb_x, emb_y, insensitive_embs)
                else:
                    prob = 0
                tot_prob += prob
                p = random.random()
                if p < prob*2500:
                    if flag == True:
                        print(f'\t{voc[sens_tok]} replaced by {voc[insens_tok]} {prob*2500}')
                    count += 1
                    tok[i][j] = insens_tok
        if flag == True:
            print('Swapped')
            print([voc[tok[i][k].tolist()]for k in range(len(tok[i]))])
            flag = False
    return tot_prob, count/len(tok)/len(tok[0]), (tok, label)

def process_data(filename, map, pre_train, sequence_len, batch_size, sampler, bert_model, downsample, att, data=None):
    
    if data is not None:
        data = data
    else:
        data = get_data(filename=filename, map_=map, downsample=downsample)
    
    train, valid, test = split_data(df=data, pre_train=pre_train)
    tokenizer = BertTokenizer.from_pretrained(bert_model)
    
    #toks = tokenize(tokenizer, [list(data['data']), list(data['label'])], sequence_len, batch_size, 'test', sampler, att, pre_train)[0][0]
    #toks_flat = toks.reshape(toks.shape[0]*toks.shape[1])
    
    # get sequency length
    seq_len = [len(i.split()) for i in train[0]]
    seq_len = int(np.ceil((pd.Series(seq_len).describe()['75%']) / 5) * 5)
    if not sequence_len:
        sequence_len = seq_len
    print(f'Set sequence length is: {sequence_len} | 75% data sequence length is: {seq_len}')
    train_dataloader, train_weight = tokenize(tokenizer, train, sequence_len, batch_size, 'train', sampler, att, pre_train)
    valid_dataloader, no_weight = tokenize(tokenizer, valid, sequence_len, batch_size, 'valid', sampler, att, pre_train)
    test_data, no_w = tokenize(tokenizer, test, sequence_len, batch_size, 'test', sampler, att, pre_train)
    return train_dataloader, valid_dataloader, test_data, train_weight

if __name__ == '__main__':

    
    #filename = '/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/data.jsonl'   
    #jsonObj = pd.read_json(path_or_buf=filename, lines=True)
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    sensitive = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_private69092.csv')
    local = sensitive[sensitive['label'].isin([0, 1, 2])]
    val_data, test_data, val_labels, test_labels = train_test_split(local['data'], local['label'],
                                                                        test_size=0.95,
                                                                        stratify=local['label'])
    test = (test_data.reset_index(drop=True), test_labels.reset_index(drop=True))
    test, no_w = tokenize(tokenizer, test, 50, 32, 'test', RandomSampler, False, False)
    tot_prob, replace_perc, test_data = sanitize_data(tokenizer, test, 0.85, 1000)
    print(f'Total Probability is {tot_prob}')
    print(f'Replaced percentage is {replace_perc}')
    
    '''
    data = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_remaining345456.csv')
    public, private = train_test_split(data, test_size=0.2,stratify=data['type'])
    print(len(public), len(private))
    public = public.drop_duplicates().reset_index(drop=True)
    seen, unseen = train_test_split(public, test_size=0.25,stratify=public['type'])
    private = private.drop_duplicates().reset_index(drop=True)
    seen = seen.drop_duplicates().reset_index(drop=True)
    unseen = unseen.drop_duplicates().reset_index(drop=True)
    seen.to_csv(f'/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_seen{len(seen)}.csv')
    unseen.to_csv(f'/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_unseen{len(unseen)}.csv')
    private.to_csv(f'/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_private{len(private)}.csv')


    
    def k():
    
    map = {'sadness': 0, 'joy': 1, 'surprise': 2, 'anger': 3, 'fear': 4, 'love': 5}
    #map = {'ham': 0, 'spam':1}
    df = get_data(filename, map, False)
    
    train_dataloader, valid_dataloader, test_data, train_weight = process_data(filename=filename, 
                                                                               map=map, pre_train=True, 
                                                                               sequence_len=40, batch_size=32, 
                                                                               sampler=RandomSampler, 
                                                                               bert_model='bert-base-uncased',
                                                                               downsample=False, att=False)
    

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    new_test_data = sanitize_data(tokenizer, test_data, 0.1, 100)
    '''
    '''
    model=BertForMaskedLM.from_pretrained('bert-base-uncased')
    embedding_matrix = model.bert.embeddings.word_embeddings.weight.data.numpy()
    voc = dict(tokenizer.vocab)
    str1 = 'banana'
    str2 = 'orange'
    str3 = 'war'
    str4 = 'fruit'
    token1 = tokenizer.tokenize(str1)
    token2 = tokenizer.tokenize(str2)
    token3 = tokenizer.tokenize(str3)
    token4 = tokenizer.tokenize(str4)

    embs1 = embedding_matrix[tokenizer.convert_tokens_to_ids(token1)].flatten()
    embs2 = embedding_matrix[tokenizer.convert_tokens_to_ids(token2)].flatten()
    embs3 = embedding_matrix[tokenizer.convert_tokens_to_ids(token3)].flatten()
    embs4 = embedding_matrix[tokenizer.convert_tokens_to_ids(token4)].flatten()
    
    #print(embs1)
    #print(embs2)
    #print(embs3)
    print(f'{str1} and {str2}')
    print(np.linalg.norm(embs1-embs2))
    print(np.dot(embs1, embs2)/np.linalg.norm(embs1)/np.linalg.norm(embs2))
    print(f'{str1} and {str4}')
    print(np.linalg.norm(embs1-embs4))
    print(np.dot(embs1, embs4)/np.linalg.norm(embs1)/np.linalg.norm(embs4))
    print(f'{str1} and {str3}')
    print(np.linalg.norm(embs1-embs3))
    print(np.dot(embs1, embs3)/np.linalg.norm(embs1)/np.linalg.norm(embs3))
    print(f'{str1} and {str1}')
    print(np.linalg.norm(embs1-embs1))
    print(np.dot(embs1, embs1)/np.linalg.norm(embs1)/np.linalg.norm(embs1))
    '''
    
