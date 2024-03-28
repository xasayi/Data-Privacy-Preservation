import numpy as np
import torch
import pandas as pd
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
import os
from sklearn.metrics.pairwise import cosine_similarity
print(os.getcwd())
from model import LSTMModelMulti, LSTMModel, LSTMModelMulti2
from new_process_data import process_data, sanitize_data, get_data, split_data, tokenize
from torch.utils.data import RandomSampler, TensorDataset, DataLoader
device = torch.device("cpu")

def l2_distance(a, b):
    a_file = torch.load(a)
    b_file = torch.load(b)
    keys = list(a_file.keys())
    distance = 0
    weights = ['embed.weight', 'lstm1.weight_ih_l0', 'lstm1.weight_hh_l0', 'lstm1.weight_ih_l0_reverse', 'lstm1.weight_hh_l0_reverse', 'lstm2.weight_ih_l0', 'lstm2.weight_hh_l0', 'lstm2.weight_ih_l0_reverse', 'lstm2.weight_hh_l0_reverse', 'lstm3.weight_ih_l0', 'lstm3.weight_hh_l0', 'lstm3.weight_ih_l0_reverse', 'lstm3.weight_hh_l0_reverse',  'lstm4.weight_ih_l0', 'lstm4.weight_hh_l0', 'lstm4.weight_ih_l0_reverse', 'lstm4.weight_hh_l0_reverse',  'lstm5.weight_ih_l0', 'lstm5.weight_hh_l0', 'lstm5.weight_ih_l0_reverse', 'lstm5.weight_hh_l0_reverse', 'fc1.weight']
    bias = ['lstm1.bias_ih_l0', 'lstm1.bias_hh_l0', 'lstm1.bias_hh_l0_reverse', 'lstm2.bias_ih_l0', 'lstm2.bias_hh_l0',  'lstm2.bias_ih_l0_reverse', 'lstm2.bias_hh_l0_reverse', 'lstm3.bias_ih_l0', 'lstm3.bias_hh_l0',  'lstm3.bias_ih_l0_reverse', 'lstm3.bias_hh_l0_reverse',  'lstm4.bias_ih_l0', 'lstm4.bias_hh_l0',  'lstm4.bias_ih_l0_reverse', 'lstm4.bias_hh_l0_reverse',  'lstm5.bias_ih_l0', 'lstm5.bias_hh_l0', 'lstm5.bias_ih_l0_reverse', 'lstm5.bias_hh_l0_reverse',  'fc1.bias']
    
    
    for key in weights:
        distance += np.linalg.norm(b_file[key]-a_file[key])
    print(distance)

def cosine_softmax(a, b):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    train = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_unseen69091.csv')
    sensitive = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_private69092.csv')
    #train = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_private69092.csv')
    local = sensitive[sensitive['label'].isin([0, 1, 2])]
    
    print(f'Have sensitive dataset of size {len(local["label"])}')
    val_data, test_data, val_labels, test_labels = train_test_split(local['data'], local['label'],
                                                                        test_size=0.95,
                                                                        stratify=local['label'])
    valid = (val_data.reset_index(drop=True), val_labels.reset_index(drop=True))
    
    test_data, no_w = tokenize(tokenizer, valid, 50, 128, 'test', RandomSampler, False, False)
    data = test_data[0]
    
    model1 = LSTMModelMulti2(6, 30522, [50, 128, 64, 32, 16, 32], 0.1).to(device)
    model1.load_state_dict(torch.load(a))
    with torch.no_grad():
        preds1 = model1(data)[-1]

    model2 = LSTMModelMulti2(6, 30522, [50, 128, 64, 32, 16, 32], 0.1).to(device)
    model2.load_state_dict(torch.load(b))
    with torch.no_grad():
        preds2 = model2(data)[-1]
    #print(preds1)
    #print(preds2)
    
    sims = []
    for i in range(len(preds1)):
        sims.append(np.dot(preds1[i, :],preds2[i, :])/np.linalg.norm(preds1[i, :])/np.linalg.norm(preds2[i, :]))
    print(np.mean(sims))

if __name__ == '__main__':
    teacher = '/Users/sarinaxi/Desktop/Thesis/Framework_old/results_2024_before_feb/Emotion_PT_T_shrinked/teacher.pt'

    students = ['/Users/sarinaxi/Desktop/Thesis/Framework_old/results_2024_before_feb/Emotions_PT_S_5_new/student.pt',
                '/Users/sarinaxi/Desktop/Thesis/Framework_old/results_2024_before_feb/emotion_5_active1:4_20k_2/student_teacher.pt',
                '/Users/sarinaxi/Desktop/Thesis/Framework_old/results_2024_before_feb/emotion_5_active4:1_20k_2/student_teacher.pt',
                '/Users/sarinaxi/Desktop/Thesis/Framework_old/results_2024_before_feb/emotion_5_active1:1_20k_2/student_teacher.pt',
                '/Users/sarinaxi/Desktop/Thesis/Framework_old/results_2024_before_feb/emotion_5_dp_1000/student_teacher.pt',
                '/Users/sarinaxi/Desktop/Thesis/Framework_old/results_2024_before_feb/emotion_5_dp_100/student_teacher.pt',
                '/Users/sarinaxi/Desktop/Thesis/Framework_old/results_2024_before_feb/emotion_5_dp_1/student_teacher.pt',
                '/Users/sarinaxi/Desktop/Thesis/Framework_old/results_2024_before_feb/emotion_5_dp_0.1/student_teacher.pt']

    for student in students:
        print('ls')
        l2_distance(teacher, student)
        print('cosine')
        cosine_softmax(teacher, student)
        print('\n')