import numpy as np
import pickle
import os
import sys
sys.path.append('.')

def format_data(folder):
    files = os.listdir(folder)
    for i, file in enumerate(files):
        os.rename(folder + file, folder + str(i+1))

def get_data(folder):
    files = os.listdir(folder)
    all_data = []
    err = 0
    for file in files:
        try:
            f = open(folder + file, "r")
            data = f.read().split('\n\n', 1)
            f.close()
            if len(data) < 2:
                continue
            body = data[1]
            attr = [i.split(': ') for i in data[0].split('\n')]
            all_data.append((attr, body))
        except UnicodeDecodeError:
            err += 1
            print(f'Decode Error {err}')
    return all_data

def process_data(all_data, lab):
    body = [i[1].replace('\t', '').replace('\n', ' ') for i in all_data]
    label = [lab] * len(body)
    dic_ = {'data': body, 'label':label}
    return dic_

def save_data(path):
    data = get_data(path)
    with open (path[:-1] + '.txt', "wb") as f:
        pickle.dump(data, f)

def get_body_dic(path, type_):
    with open(path[:-1] + '.txt', "rb") as f: 
        data = pickle.load(f)
    body_dic = process_data(data, type_)
    return body_dic

def combine_data(ham, spam):
    indices = np.arange(len(ham['label']) + len(spam['label']))
    np.random.shuffle(indices)

    dic_ = {}
    for i in ham.keys():
        combined = ham[i] + spam[i]
        dic_[i] = list(np.array(combined)[indices])
    return dic_