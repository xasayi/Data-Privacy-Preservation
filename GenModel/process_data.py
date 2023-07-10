import sys
sys.path.insert(0, '/Users/sarinaxi/Desktop/Thesis')

import torch
import re
from datasets import load_dataset
from sklearn.model_selection import train_test_split

parent = '/Users/sarinaxi/Desktop/Thesis/GenModel/data/'
paths = {'train': 'train_dataset.txt', 'valid': 'valid_dataset.txt', 'test': 'test_dataset.txt'}

def get_data(filename, index):
    with open(f'/Users/sarinaxi/Desktop/Thesis/{filename}') as f:
        content = f.read()
    lines = content.split('\n')[:index]
    return lines

def build_text_file(datas, path):
    f = open(path, 'w')
    data = ''
    for i in datas:
        all = str(i.strip())
        all = re.sub(r"\s", ' ', all)
        data += all + '  '
    f.write(data)
    f.close()

def split_data(data, valtest_size):
    train_data, temp_data = train_test_split(data, random_state=2023, test_size=valtest_size)
    val_data, test_data = train_test_split(temp_data, random_state=2023, test_size=0.5)
    return train_data, val_data, test_data

def process_data(filename, index, split, tokenizer):
    data = get_data(filename, index)
    train, val, test = split_data(data, split)

    build_text_file(train, parent + paths['train'])
    build_text_file(val, parent + paths['valid'])
    build_text_file(test, parent + paths['test'])
    datasets = load_dataset('data', paths)
    
    tokenizer.pad_token = '[PAD]'
    datasets = datasets.map(lambda x: tokenizer(x['text'], truncation=True, padding='max_length'), batched=True)
    train_dataloader = torch.utils.data.DataLoader(datasets['train'], batch_size=32)
    valid_dataloader = torch.utils.data.DataLoader(datasets['validation'], batch_size=32)
    test_dataloader = torch.utils.data.DataLoader(datasets['test'], batch_size=32)
    return train_dataloader, valid_dataloader, test_dataloader
