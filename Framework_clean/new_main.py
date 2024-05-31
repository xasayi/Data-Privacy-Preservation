'''
main file to run the student teacher model
'''
import os
import sys
import time
import pickle
import random
import warnings
import yaml
import torch

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
from torch.utils.data import RandomSampler, TensorDataset, DataLoader

sys.path.insert(0, '/Users/sarinaxi/Desktop/Thesis')
from transformer import MyTransformer
from Framework_clean.new_process_data import sanitize_data, split_data, tokenize
from Framework_clean.classifier import Classifier, model_performance
from Framework_clean.new_student_teacher import StudentTeacher
from SpamDetector.plotting_analytics import plot_loss_acc

SEED = 24
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
warnings.filterwarnings("ignore")
device = torch.device("cpu")

def args_and_init(config_file):
    '''get configuration arguments for initialization'''
    # check GPU, don't use it since there's a bug with GRU
    print(torch.backends.mps.is_available())
    print(torch.backends.mps.is_built())

    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)
    st_args, te_args = None, None
    student = config['student_bool']
    teacher = config['teacher_bool']

    if student:
        st_args = config['StudentPT']
        st_folder = st_args['folder']
        if not os.path.exists(st_folder):
            os.makedirs(st_folder)

    if teacher:
        te_args = config['TeacherPT']
        te_folder = te_args['folder']
        if not os.path.exists(te_folder):
            os.makedirs(te_folder)

    if not os.path.exists(config['StudentTeacher']['folder']):
        os.makedirs(config['StudentTeacher']['folder'])

    data_filename = config['filename_pretrain'] if config['pt'] else config['filename_train']
    return data_filename, st_args, te_args, config

def pre_train(model, args, train_loader, valid_loader, test_data, train_weight, active):
    '''Pre-train the student or teacher model'''
    agent = Classifier(model=model, train_dataloader=train_loader, device=device, lr=args['lr'],
                        batch_size=args['batch_size'], valid_dataloader=valid_loader,
                        epochs=args['epochs'],test_data=test_data, weights=train_weight,
                        folder=args['folder'], weight_path=args['name'], active=active)
    train_losses, train_acc, valid_losses, valid_accs = agent.run()
    dic = {'train_losses': train_losses,
            'student_train_accs': train_acc,
            'valid_losses': valid_losses,
            'student_valid_accs': valid_accs}

    file = open(f"{args['folder']}/train_data.pkl","wb")
    pickle.dump(dic,file)
    file.close()
    plot_loss_acc(train_losses, valid_losses, 'Loss', args['folder'])
    plot_loss_acc(train_acc, valid_accs, 'Acc', args['folder'])

def check_ptmodel(model, args, train_loader, valid_loader, test_data, train_weight, active):
    '''Check the pre-trained model and load the pre-train model'''
    agent = Classifier(model=model, train_dataloader=train_loader, device=device, lr=args['lr'],
                       batch_size=args['batch_size'], valid_dataloader=valid_loader,
                       epochs=args['epochs'], test_data=test_data, weights=train_weight,
                       folder=args['folder'], weight_path=args['name'], active=active)
    agent.model.load_state_dict(torch.load(f"{args['folder']}/{args['name']}"))
    model_performance(args, agent.model, agent.test_data[0], agent.test_data[1],
                      device, args['folder'])

def plot(train, folder, type):
    '''Plotting configurations'''
    plt.figure(figsize=(10, 7))
    color = ['r', 'b', 'g', 'm', 'c', 'y', 'cyan', 'pink', 'brown', 'teal']
    format = ['-', '-', '-', '-', '-', '-', '--', '--', '--','--', '--', '--']
    colors = color[:len(train.items())]+color[:len(train.items())]
    for ind, (i, j) in enumerate(train.items()):
        print(format[ind], f'{i}', colors[ind])
        plt.plot(j, format[ind], label = f'{i}', color=colors[ind])
    plt.xlabel('# of Iterations')
    plt.ylabel('Percentage %')
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.title(type)
    plt.grid()

    os.makedirs(folder, exist_ok=True)
    plt.savefig(f'{folder}/{type}.png', bbox_inches='tight')

def plot_stuff(filenames, labels, name):
    '''Plot results '''
    datas = []
    for i in filenames:
        file = open(f"Framework_clean/new_results/{i}/train_data.pkl", 'rb')
        datas.append(pickle.load(file))
        file.close()

    student_acc = {}
    print(labels)
    for i in range(len(labels)):
        student_acc[f'Student Valid {labels[i]}'] = np.array(datas[i]['student_valid_accs'])*100
    indes = np.array([0, 19, 39, 59, 79, 99, 119, 139, 159, 179, 199])
    print('Student Validation Accuracies')

    items = ['student_valid_accs', 'student_noisy_train_accs', 'student_raw_train_accs'
             'teacher_valid_accs', 'teacher_train_accs']
    for k in items:
        for j in np.array(datas[i][k])[indes]:
            print(j)

    train_loss = {}
    for i in range(len(labels)):
        length = 1+len(np.array(datas[i]['student_valid_accs']))
        point = datas[i]['teacher_valid_accs'][-1]
        student_acc[f'Teacher Valid {labels[i]}'] = np.array([point]*length)*100
        train_loss[f'Student {labels[i]}'] = np.array(datas[i]['valid_losses'])
    plot(train_loss, "Framework_clean/new_results/", f'{name} Student Validation Loss Curve')
    plot(student_acc, "Framework_clean/new_results/", f'{name} Student Validation Accuracy Curve')

def save_parameters(arg, folder):
    '''Save parameters and variables for run'''
    with open(f'{folder}/parameters.txt', 'w') as file:
        file.write(f'factor: {arg["factor"]}\n')
        file.write(f'dropout: {arg["dropout"]}\n')
        file.write(f'lr: {arg["lr"]}\n')
        file.write(f'wd: {arg["wd"]}\n\n')
        file.write(f'eps: {arg["eps"]}\n')
        file.write(f'sens_ratio: {arg["sens_ratio"]}\n')
        file.write(f'queries: {arg["queries"]}\n')
        file.write(f'epochs: {arg["epochs"]}\n')
        file.write(f'iters: {arg["iters"]}\n')
        file.write(f'dp: {arg["dp"]}\n\n')
        file.write(f'model: {arg["model"]}\n')
        file.write(f'batch_size {arg["batch_size"]}\n')
        file.write(f'input_size: {arg["input_size"]}\n')
        file.write(f'hidden: {arg["hidden"]}\n')
        file.write(f'downsample: {arg["downsample"]}\n')
        file.write(f'similarity: {arg["similarity"]}\n\n')

if __name__ == '__main__':
    file, st_args, te_args, config = args_and_init(config_file="Framework_clean/new_config.yaml")

    args = config['StudentTeacher']
    pt = config['pt']
    student_bool = config['student_bool']
    teacher_bool = config['teacher_bool']
    teacher_student_bool = config['teacher_student_bool']

    active = config['active']
    plot_ = config['plot_graphs']
    # define mappings here
    mapping = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    # {'sadness': 0, 'joy': 1, 'surprise': 2, 'anger': 3, 'fear': 4, 'love': 5}
    # {'ham': 0, 'spam':1}
    if plot_: ## NEED TO CHANGE VARIABLES THIS PART FOR PLOTTING
        lab = ['95', '95_dissim', '97', '97_dissim', '99', '99_dissim']
        k = [f'0new_emotions_dp1_{i}' for i in lab]
        plot_stuff(k, lab, 'dissim_test')

    elif teacher_student_bool:
        tokenizer = BertTokenizer.from_pretrained(te_args['model'])
        train = pd.read_csv('Framework_clean/data/sentiment_data/huggingface_unseen69091.csv')
        local = pd.read_csv('Framework_clean/data/sentiment_data/huggingface_private69092.csv')
        data = pd.concat((local, train)).reset_index(drop=True)
        bs = args['batch_size']
        val_data, test_data, val_labels, test_labels = train_test_split(local['data'],
                                                                        local['label'],
                                                                        test_size=0.95,
                                                                        stratify=local['label'])
        test_labels = data['label'][:80000]
        test_data = data['data'][:80000]
        print(f'Have sensitive dataset of size {len(test_data)}')
        new_train = data[80000:].sample(n=10000).reset_index(drop=True)
        print(f'Train is size {len(new_train["label"])}')
        train = (new_train['data'], new_train['label'])

        train_dataloader, train_weight = tokenize(tokenizer, train, args['hidden'][0],
                                                  bs, 'train', RandomSampler, False)
        valid = (val_data.reset_index(drop=True), val_labels.reset_index(drop=True))
        test = (test_data.reset_index(drop=True), test_labels.reset_index(drop=True))
        print(f'Train: {len(train[0])} | Valid: {len(valid[0])} | Test: {len(test[0])}')
        valid_dataloader = tokenize(tokenizer, valid, args['hidden'][0], bs,
                                    'valid', RandomSampler, False)[0]
        test_data = tokenize(tokenizer, test, args['hidden'][0], bs, 'test', RandomSampler, False)[0]

        student = MyTransformer(block_size=50, vocab_size=30522, embeds_size=32,
                        drop_prob=st_args['dropout'], num_classes=len(mapping), num_heads=8,
                        n_layers=1, device=torch.device("cpu"))
        # we can load pre-trained models here
        check_ptmodel(model=student, args=st_args, train_loader=train_dataloader,
                      valid_loader=valid_dataloader, test_data=valid_dataloader,
                      train_weight=train_weight, active=active)
        teacher = MyTransformer(block_size=50, vocab_size=30522, embeds_size=32,
                        drop_prob=st_args['dropout'], num_classes=len(mapping),
                        num_heads=8, n_layers=2, device=torch.device("cpu"))
        # load pre-train model as well as check pre-trained teacher performance on sanitized data
        tot_prob, replace_perc, validation = sanitize_data(tokenizer,
                                                           (train_dataloader[0].clone(),
                                                            train_dataloader[1].clone()),
                                                           args['sens_ratio'], args['eps'])
        check_ptmodel(model=teacher, args=te_args, train_loader=train_dataloader,
                      valid_loader=valid_dataloader, test_data=validation,
                      train_weight=train_weight, active=active)

        #-------------------------Student Teacher Finetuning------------------------
        agent = StudentTeacher(teacher, student, args, device, mapping, args['similarity'],
                               active, train_dataloader, valid_dataloader, test_data, train_weight)
        start = time.time()
        train_losses, student_train_accs, student_train_accs_raw, \
        valid_losses, student_valid_accs, teacher_train_accs, teacher_valid_accs, \
        acc, train_label, valid_label = agent.run()
        end = time.time()

        print(f'Took {round((end-start)/60, 3)} minutes')

        plot_loss_acc(train_losses[::args['epochs']], valid_losses[::args['epochs']],
                      'Loss', args['folder'])
        plot_loss_acc(student_train_accs[::args['epochs']],
                      student_valid_accs[::args['epochs']], 'Student Acc', args['folder'])
        plot_loss_acc(teacher_train_accs[::args['epochs']],
                      teacher_valid_accs[::args['epochs']], 'Teacher Acc', args['folder'])
        plot_loss_acc(train_losses, valid_losses, 'Loss_all',
                      args['folder'])
        plot_loss_acc(student_train_accs, student_valid_accs,
                      'Student Noisy Train Raw Valid', args['folder'])
        plot_loss_acc(student_train_accs_raw, student_valid_accs,
                      'Student Raw Train Raw Valid', args['folder'])
        plot_loss_acc(student_train_accs_raw, student_train_accs,
                      'Student Raw Train Noisy Train', args['folder'])
        plot_loss_acc(teacher_train_accs, teacher_valid_accs,
                      'Teacher Acc_all', args['folder'])
        plot_loss_acc(train_label, valid_label, 'Label Correctness',
                      args['folder'])
        save_parameters(args, args['folder'])

    elif student_bool:
        data = pd.read_csv('Framework_clean/data/sentiment_data/huggingface_seen_pub252467.csv')
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        train, valid, test = split_data(df=data, pre_train=pre_train)
        bs = st_args['batch_size']
        train_data, train_weight = tokenize(tokenizer, train, 50, bs, 'train', RandomSampler, False)
        valid_data = tokenize(tokenizer, valid, 50, bs, 'valid', RandomSampler, False)[0]
        test_data = tokenize(tokenizer, test, 50, bs, 'test', RandomSampler, False)[0]

        trains = TensorDataset(train_data[0], train_data[1])
        sampler = RandomSampler(trains)
        train_data = DataLoader(trains, sampler=sampler, batch_size=st_args['batch_size'])

        valids = TensorDataset(valid_data[0], valid_data[1])
        sampler = RandomSampler(valids)
        valid_data = DataLoader(valids, sampler=sampler, batch_size=st_args['batch_size'])

        student = MyTransformer(block_size=50, vocab_size=30522, embeds_size=32,
                      drop_prob=st_args['dropout'], num_classes=len(mapping), num_heads=8,
                      n_layers=1, device=torch.device("cpu"))

        print(train_weight)
        pre_train(model=student, args=st_args, train_loader=train_data,
                  valid_loader=valid_data,test_data=test_data,
                  train_weight=train_weight, active=active)
        check_ptmodel(model=student, args=st_args, train_loader=train_data,
                      valid_loader=valid_data, test_data=test_data,
                      train_weight=train_weight, active=active)

    elif teacher_bool:
        tokenizer = BertTokenizer.from_pretrained(te_args['model'])
        train = pd.read_csv('Framework_clean/data/sentiment_data/huggingface_seen_pub252467.csv')
        local = pd.read_csv('Framework_clean/data/sentiment_data/huggingface_private69092.csv')

        print(f'Have sensitive dataset of size {len(local["label"])}')
        val_data, test_data, val_labels, test_labels = train_test_split(local['data'],
                                                                        local['label'],
                                                                        test_size=0.95,
                                                                        stratify=local['label'])

        data = (pd.concat((test_data, train['data'])).reset_index(drop=True),
                pd.concat((test_labels, train['label'])).reset_index(drop=True))
        train_data, temp_data, train_labels, temp_labels = train_test_split(data[0], data[1],
                                                                            test_size=0.10,
                                                                            stratify=data[1])
        valid_data, test_data, valid_labels, temp_labels = train_test_split(temp_data, temp_labels,
                                                                            test_size=0.50,
                                                                            stratify=temp_labels)

        train = (train_data.reset_index(drop=True), train_labels.reset_index(drop=True))
        valid = (valid_data.reset_index(drop=True), valid_labels.reset_index(drop=True))
        test = (test_data.reset_index(drop=True), temp_labels.reset_index(drop=True))
        leng = te_args['hidden'][0]
        bs = te_args['batch_size']
        tokenizer = BertTokenizer.from_pretrained(te_args['model'])

        # get sequency length
        seq_len = [len(i.split()) for i in train[0]]
        seq_len = int(np.ceil((pd.Series(seq_len).describe()['75%']) / 5) * 5)
        if not te_args['hidden'][0]:
            seq_len = leng
        print(f'Set seq length is: {leng} | 75% data seq length is: {seq_len}')
        train_data, weight = tokenize(tokenizer, train, seq_len, bs, 'train', RandomSampler, True)
        valid_data = tokenize(tokenizer, valid, seq_len, bs, 'valid', RandomSampler, True)[0]
        test_data = tokenize(tokenizer, test, seq_len, bs, 'test', RandomSampler, True)[0]
        print(f'Train: {len(train[0])} | Valid: {len(valid[0])} | Test: {len(test[0])}')
        teacher = MyTransformer(block_size=50, vocab_size=30522, embeds_size=32,
                                drop_prob=te_args['dropout'], num_classes=len(mapping),
                                num_heads=8, n_layers=1, device=torch.device("cpu"))

        # pre-train teacher
        start = time.time()
        pre_train(model=teacher, args=te_args, train_loader=train_data,
                  valid_loader=valid_data, test_data=test_data,
                  train_weight=weight, active=active)
        end = time.time()
        print(F'TOOK {round(end-start, 2)} SECONDS')
        check_ptmodel(model=teacher, args=te_args, train_loader=train_data,
                      valid_loader=valid_data,test_data=test_data,
                      train_weight=weight,  active=True)
