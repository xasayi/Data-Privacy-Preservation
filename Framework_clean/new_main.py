import os
import sys
import time
import yaml
import torch
import pickle
import numpy as np
import pandas as pd
import random
import warnings
sys.path.insert(0, '/Users/sarinaxi/Desktop/Thesis')
import matplotlib.pyplot as plt
from Framework_clean.new_process_data import process_data, sanitize_data, get_data, split_data, tokenize
from torch.utils.data import RandomSampler, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split 
from Framework_clean.classifier import Classifier, model_performance
from SpamDetector.plotting_analytics import plot_loss_acc
from Framework_clean.new_student_teacher import StudentTeacher
from transformer import MyTransformer

device = torch.device("cpu")
from transformers import BertTokenizer

SEED = 24
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
warnings.filterwarnings("ignore")

def args_and_init(config_file):
    # check GPU, don't use it since there's a bug with GRU
    print(torch.backends.mps.is_available())
    print(torch.backends.mps.is_built())

    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
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
    
    agent = Classifier(model=model, train_dataloader=train_loader, device=device, lr=args['lr'], 
                                        batch_size=args['batch_size'], valid_dataloader=valid_loader, epochs=args['epochs'], 
                                        test_data=test_data, weights=train_weight, folder=args['folder'], weight_path=args['name'], 
                                        active=active)
    train_losses, train_acc, valid_losses, valid_accs = agent.run()
    dic = {'train_losses': train_losses, 
            'student_train_accs': train_acc, 
               'valid_losses': valid_losses, 
               'student_valid_accs': valid_accs, 
               }

    f = open(f"{args['folder']}/train_data.pkl","wb")
    pickle.dump(dic,f)
    f.close()
    plot_loss_acc(train_losses, valid_losses, 'Loss', args['folder'])
    plot_loss_acc(train_acc, valid_accs, 'Acc', args['folder'])

def check_ptmodel(model, args, train_loader, valid_loader, test_data, train_weight, active):
    agent = Classifier(model=model, train_dataloader=train_loader, device=device, lr=args['lr'], 
                                        batch_size=args['batch_size'], valid_dataloader=valid_loader, epochs=args['epochs'], 
                                        test_data=test_data, weights=train_weight, folder=args['folder'], weight_path=args['name'], 
                                        active=active)
    agent.model.load_state_dict(torch.load(f"{args['folder']}/{args['name']}"))
    acc = model_performance(args, agent.model, agent.test_data[0], agent.test_data[1], device, args['folder'])

def plot(train, folder, type):
    plt.figure(figsize=(10, 7))
    color = ['r', 'b', 'g', 'm', 'c', 'y']#, 'cyan', 'pink']#, 'brown', 'teal']
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

def plot_stuff(config, filenames, labels, name):
    args = config['StudentTeacher']
    datas = []
    for i in filenames:
        file = open(f"/Users/sarinaxi/Desktop/Thesis/Framework/new_results/{i}/train_data.pkl", 'rb')
        datas.append(pickle.load(file))
        file.close()

    student_train_acc, student_noisy = {}, {}
    accs = []
    print(labels)
    for i in range(len(labels)):
        student_train_acc[f'Student Valid {labels[i]}'] = np.array(list(np.array(datas[i]['student_valid_accs'])))*100
        accs.append(np.array(datas[i]['student_valid_accs'])[-1])
        #student_train_acc[f'Student Train Noisy {labels[i]}'] = np.array(list(np.array(datas[i]['student_noisy_train_accs'])[indices]))*100
        #student_train_acc[f'Student Train Raw {labels[i]}'] = np.array(list(np.array(datas[i]['student_raw_train_accs'])[indices]))*100
        #student_train_acc[f'Teacher Train Noisy {labels[i]}'] = np.array(list(np.array(datas[i]['teacher_train_accs'])[indices]))*100

    indes = np.array([0, 19, 39, 59, 79, 99, 119, 139, 159, 179, 199])
    print('Student Validation Accuracies')
    for j in np.array(datas[i]['student_valid_accs'])[indes]:
        print(j)
    
    print('Student Noisy Training Accuracies')
    for j in np.array(datas[i]['student_noisy_train_accs'])[indes]:
        print(j)

    print('Student Raq Training Accuracies')
    for j in np.array(datas[i]['student_raw_train_accs'])[indes]:
        print(j)
    
    print('Teacher Validation Accuracies')
    for j in np.array(datas[i]['teacher_valid_accs'])[indes]:
        print(j)
    
    print('Teacher RawTraining Accuracies')
    for j in np.array(datas[i]['teacher_train_accs'])[indes]:
        print(j)

    #print(labels)
    train_loss = {}
    tea = []
    for i in range(len(labels)):
        student_train_acc[f'Teacher Valid {labels[i]}'] = np.array([datas[i]['teacher_valid_accs'][-1]]*(1+len(np.array(datas[i]['student_valid_accs']))))*100
        tea.append(datas[i]['teacher_valid_accs'][-1])
        train_loss[f'Student {labels[i]}'] = np.array(datas[i]['valid_losses'])
    plot(train_loss, "/Users/sarinaxi/Desktop/Thesis/Framework/new_results/", f'{name} Student Validation Loss Curve')
    plot(student_train_acc, "/Users/sarinaxi/Desktop/Thesis/Framework/new_results/", f'{name} Student Validation Accuracy Curve')

def save_parameters(args, folder):
    with open(f'{folder}/parameters.txt', 'w') as f:
        f.write(f'factor: {args["factor"]}\n')
        f.write(f'dropout: {args["dropout"]}\n')
        f.write(f'lr: {args["lr"]}\n')
        f.write(f'wd: {args["wd"]}\n\n')
        f.write(f'eps: {args["eps"]}\n')
        f.write(f'sens_ratio: {args["sens_ratio"]}\n')
        f.write(f'queries: {args["queries"]}\n')
        f.write(f'epochs: {args["epochs"]}\n')
        f.write(f'iters: {args["iters"]}\n')
        f.write(f'dp: {args["dp"]}\n\n')
        f.write(f'model: {args["model"]}\n')
        f.write(f'batch_size {args["batch_size"]}\n')
        f.write(f'input_size: {args["input_size"]}\n')
        f.write(f'hidden: {args["hidden"]}\n')
        f.write(f'downsample: {args["downsample"]}\n')
        f.write(f'similarity: {args["similarity"]}\n\n')
        
if __name__ == '__main__':
    data_filename, st_args, te_args, config = args_and_init(config_file="Framework/new_config.yaml")
    
    args = config['StudentTeacher']
    pt = config['pt']
    student_bool = config['student_bool']
    teacher_bool = config['teacher_bool']
    teacher_student_bool = config['teacher_student_bool']
    
    transformer = config['transformer']
    active = config['active']
    plot_ = config['plot_graphs']
    normal = config['public_noise']
    half = config['half']
    map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    
    data_filename, st_args, te_args, config = args_and_init(config_file="Framework_clean/new_config.yaml")
    
    args = config['StudentTeacher']
    pt = config['pt']
    student_bool = config['student_bool']
    teacher_bool = config['teacher_bool']
    teacher_student_bool = config['teacher_student_bool']
    
    active = config['active']
    plot_ = config['plot_graphs']
    normal = config['public_noise']
    half = config['half']
    # define mappings here
    map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5} # 
    # {'sadness': 0, 'joy': 1, 'surprise': 2, 'anger': 3, 'fear': 4, 'love': 5}
    # {'ham': 0, 'spam':1}
    if plot_: ## NEED TO CHANGE VARIABLES THIS PART FOR PLOTTING
        k = []
        lab = ['95', '95_dissim', '97', '97_dissim', '99', '99_dissim',]
        for i in lab:
            k.append(f'0new_emotions_dp1_{i}')
        plot_stuff(config, k, lab, f'dissim_test')
        
    elif teacher_student_bool:
        tokenizer = BertTokenizer.from_pretrained(te_args['model'])
        train = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_unseen69091.csv')
        local = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_private69092.csv')
        data = pd.concat((local, train)).reset_index(drop=True)
        
        val_data, test_data, val_labels, test_labels = train_test_split(local['data'], local['label'],
                                                                        test_size=0.95,
                                                                        stratify=local['label'])
        test_labels = data['label'][:80000]
        test_data = data['data'][:80000]
        print(f'Have sensitive dataset of size {len(test_data)}')
        new_train = data[80000:].sample(n=10000).reset_index(drop=True)
        print(f'Train is size {len(new_train["label"])}')
        train = (new_train['data'], new_train['label'])
        
        # if we want to add in a mix of data to see how major and minor composition affects performance
        '''
        #spam = get_data(filename='/Users/sarinaxi/Desktop/Thesis/Framework/data/spam_data/df_full.csv', map_={'ham': 0, 'spam':1}, downsample=False)
        if half:
            print('half')
            spam = pd.concat((spam, spam, spam, spam, spam))
            spam = spam.sample(n = int(len(test_data)/2))
            new_train = new_train.sample(n = int(len(test_data)-len(test_data)/2))
            train_data = pd.concat((spam, new_train))
            train_data = train_data.sample(frac=1).reset_index(drop=True)
            train  = (train_data['data'], train_data['label'])
        else:
            print('minor')
            spam = pd.concat((spam, spam, spam, spam, spam, spam, spam, spam, spam, spam))
            spam = spam.sample(n=int(4*len(test_data)/5))
            new_train = new_train.sample(n=int(len(test_data)- int(4*len(test_data)/5)))
            train_data = pd.concat((spam, new_train))
            train_data = pd.concat((spam, new_train))
            train_data = train_data.sample(frac=1).reset_index(drop=True)
            train  = (train_data['data'], train_data['label'])
        print(f'Train is size {len(train[0])}')
        '''

        train_dataloader, train_weight = tokenize(tokenizer, train, args['hidden'][0], args['batch_size'], 'train', RandomSampler, False)
        valid = (val_data.reset_index(drop=True), val_labels.reset_index(drop=True))
        test = (test_data.reset_index(drop=True), test_labels.reset_index(drop=True))
        print(f'Train Size: {len(train[0])} | Valid Size: {len(valid[0])} | Test Size: {len(test[0])}')
        valid_dataloader, no_weight = tokenize(tokenizer, valid, args['hidden'][0], args['batch_size'], 'valid', RandomSampler, False)
        test_data, no_w = tokenize(tokenizer, test, args['hidden'][0], args['batch_size'], 'test', RandomSampler, False)
    
        student = MyTransformer(block_size=50, vocab_size=30522, embeds_size=32,
                        drop_prob=st_args['dropout'], num_classes=len(map), num_heads=8, n_layers=1, device=torch.device("cpu"))
        
        # we can load pre-trained models here 
        # check_ptmodel(model=student, args=st_args, train_loader=train_dataloader, valid_loader=valid_dataloader, 
        #              test_data=valid_dataloader, train_weight=train_weight, attention=transformer, active=active)
        teacher = MyTransformer(block_size=50, vocab_size=30522, embeds_size=32,
                        drop_prob=st_args['dropout'], num_classes=len(map), num_heads=8, n_layers=2, device=torch.device("cpu"))
        # load pre-train model as well as check pre-trained teacher performance on sanitized data 
        tot_prob, replace_perc, validation = sanitize_data(tokenizer, (train_dataloader[0].clone(), train_dataloader[1].clone()), args['sens_ratio'], args['eps'])
        check_ptmodel(model=teacher, args=te_args, train_loader=train_dataloader, valid_loader=valid_dataloader, 
                      test_data=validation, train_weight=train_weight,active=active)
        
        #-------------------------Student Teacher Finetuning------------------------
        
        agent = StudentTeacher(teacher, student, args, device, map, args['similarity'], 
                            transformer, active, train_dataloader, valid_dataloader, test_data, train_weight)
        start = time.time()
        train_losses, student_train_accs, student_train_accs_raw, valid_losses, student_valid_accs, teacher_train_accs, teacher_valid_accs, acc, train_label, valid_label = agent.run()
        end = time.time()
        print(f'Took {round((end-start)/60, 3)} minutes')
        
        plot_loss_acc(train_losses[::args['epochs']], valid_losses[::args['epochs']], 'Loss', args['folder'])
        plot_loss_acc(student_train_accs[::args['epochs']], student_valid_accs[::args['epochs']], 'Student Acc', args['folder'])
        plot_loss_acc(teacher_train_accs[::args['epochs']], teacher_valid_accs[::args['epochs']], 'Teacher Acc', args['folder'])
        plot_loss_acc(train_losses, valid_losses, 'Loss_all', args['folder'])
        plot_loss_acc(student_train_accs, student_valid_accs, 'Student Noisy Train Raw Valid', args['folder'])
        plot_loss_acc(student_train_accs_raw, student_valid_accs, 'Student Raw Train Raw Valid', args['folder'])
        plot_loss_acc(student_train_accs_raw, student_train_accs, 'Student Raw Train Noisy Train', args['folder'])
        plot_loss_acc(teacher_train_accs, teacher_valid_accs, 'Teacher Acc_all', args['folder'])
        plot_loss_acc(train_label, valid_label, 'Label Correctness', args['folder'])
        save_parameters(args, args['folder'])
        
    elif student_bool:
        
        data_filename = '/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_seen_pub252467.csv'
        data = pd.read_csv(data_filename)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

        train, valid, test = split_data(df=data, pre_train=pre_train)
    
        train_dataloader, train_weight = tokenize(tokenizer, train, 50, st_args['batch_size'], 'train', RandomSampler, False)
        valid_dataloader, no_weight = tokenize(tokenizer, valid, 50, st_args['batch_size'], 'valid', RandomSampler, False)
        test_data, no_w = tokenize(tokenizer, test, 50, st_args['batch_size'], 'test', RandomSampler, False)

        trains = TensorDataset(train_dataloader[0], train_dataloader[1])
        sampler = RandomSampler(trains)
        train_dataloader = DataLoader(trains, sampler=sampler, batch_size=st_args['batch_size'])
        
        valids = TensorDataset(valid_dataloader[0], valid_dataloader[1])
        sampler = RandomSampler(valids)
        valid_dataloader = DataLoader(valids, sampler=sampler, batch_size=st_args['batch_size'])
        
        student = MyTransformer(block_size=50, vocab_size=30522, embeds_size=32,
                      drop_prob=st_args['dropout'], num_classes=len(map), num_heads=8, n_layers=1, device=torch.device("cpu"))
        
        print(train_weight)
        pre_train(model=student, args=st_args, train_loader=train_dataloader, valid_loader=valid_dataloader,
                  test_data=test_data, train_weight=train_weight, active=active)
        check_ptmodel(model=student, args=st_args, train_loader=train_dataloader, valid_loader=valid_dataloader, 
                      test_data=test_data, train_weight=train_weight, active=active)
        
    elif teacher_bool:
        tokenizer = BertTokenizer.from_pretrained(te_args['model'])
        train = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_seen_pub252467.csv')
        local = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_private69092.csv')
       
        print(f'Have sensitive dataset of size {len(local["label"])}')
        val_data, test_data, val_labels, test_labels = train_test_split(local['data'], local['label'],
                                                                        test_size=0.95,
                                                                        stratify=local['label'])

        data = (pd.concat((test_data, train['data'])).reset_index(drop=True), pd.concat((test_labels, train['label'])).reset_index(drop=True))
        
        train_data, temp_data, train_labels, temp_labels = train_test_split(data[0], data[1],test_size=0.10,stratify=data[1])
        valid_data, test_data, valid_labels, temp_labels = train_test_split(temp_data, temp_labels, test_size=0.50,stratify=temp_labels)
        
        train = (train_data.reset_index(drop=True), train_labels.reset_index(drop=True))
        valid = (valid_data.reset_index(drop=True), valid_labels.reset_index(drop=True))
        test = (test_data.reset_index(drop=True), temp_labels.reset_index(drop=True))
        sequence_len = te_args['hidden'][0]
        batch_size = te_args['batch_size']
        tokenizer = BertTokenizer.from_pretrained(te_args['model'])
    
        # get sequency length
        seq_len = [len(i.split()) for i in train[0]]
        seq_len = int(np.ceil((pd.Series(seq_len).describe()['75%']) / 5) * 5)
        if not sequence_len:
            sequence_len = seq_len
        print(f'Set sequence length is: {sequence_len} | 75% data sequence length is: {seq_len}')
        train_dataloader, train_weight = tokenize(tokenizer, train, sequence_len, batch_size, 'train', RandomSampler, True)
        valid_dataloader, no_weight = tokenize(tokenizer, valid, sequence_len, batch_size, 'valid', RandomSampler, True)
        test_data, no_w = tokenize(tokenizer, test, sequence_len, batch_size, 'test', RandomSampler, True)
        print(f'Train size: {len(train[0])} | Valid size: {len(valid[0])} | Test size: {len(test[0])}')
        teacher = MyTransformer(block_size=50, vocab_size=30522, embeds_size=32,
                        drop_prob=te_args['dropout'], num_classes=len(map), num_heads=8, n_layers=1, device=torch.device("cpu"))

        # pre-train teacher
        start = time.time()
        pre_train(model=teacher, args=te_args, train_loader=train_dataloader, valid_loader=valid_dataloader,
                  test_data=test_data, train_weight=train_weight, active=active)
        end = time.time()
        print(F'TOOK {round(end-start, 2)} SECONDS')
        check_ptmodel(model=teacher, args=te_args, train_loader=train_dataloader, valid_loader=valid_dataloader, 
                      test_data=test_data, train_weight=train_weight,  active=True)
        
        