import os
import sys
import yaml
import torch
import pickle
import numpy as np
import pandas as pd
import random
import warnings
sys.path.insert(0, '/Users/sarinaxi/Desktop/Thesis')
import matplotlib.pyplot as plt
from Framework.new_process_data import process_data, sanitize_data, get_data, split_data, tokenize
from torch.utils.data import RandomSampler
from sklearn.model_selection import train_test_split
from Framework.model import LSTMModelMulti, LSTMModel, LSTMModelMulti2
from Framework.classifier import Classifier, model_performance
from SpamDetector.plotting_analytics import plot_loss_acc
from Framework.new_student_teacher import StudentTeacher

device = torch.device("cpu")
from transformers import BertTokenizer

SEED = 55
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

def pre_train(model, args, train_loader, valid_loader, test_data, train_weight, attention, active):
    
    agent = Classifier(model=model, train_dataloader=train_loader, device=device, lr=args['lr'], 
                                        batch_size=args['batch_size'], valid_dataloader=valid_loader, epochs=args['epochs'], 
                                        test_data=test_data, weights=train_weight, folder=args['folder'], weight_path=args['name'], 
                                        attention=attention,active=active)
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

def check_ptmodel(model, args, train_loader, valid_loader, test_data, train_weight, attention, active):
    agent = Classifier(model=model, train_dataloader=train_loader, device=device, lr=args['lr'], 
                                        batch_size=args['batch_size'], valid_dataloader=valid_loader, epochs=args['epochs'], 
                                        test_data=test_data, weights=train_weight, folder=args['folder'], weight_path=args['name'], 
                                        attention=attention, active=active)
    agent.model.load_state_dict(torch.load(f"{args['folder']}/{args['name']}"))
    if attention:
        acc = model_performance(args, agent.model, agent.test_data[0], agent.test_data[2], 
                          device, args['folder'], mask=agent.test_data[1])
    else:
        acc = model_performance(args, agent.model, agent.test_data[0], agent.test_data[1], 
                          device, args['folder'])

def plot(train, folder, type):
    plt.figure(figsize=(10, 7))
    color = ['r', 'b', 'g', 'm', 'c', 'y', 'cyan']
    format = ['-', '-', '-', '-', '-', '-', '-', '--', '--', '--', '--', '--', '--', '--']
    colors = color[:len(train)]+color[:len(train)]
    for ind, (i, j) in enumerate(train.items()):
        print(format[ind], f'{i}', colors[ind])
        plt.plot(j, format[ind], label = f'{i}', color=colors[ind])
    plt.xlabel('# of Iterations')
    plt.ylabel('Percentage %')
    plt.legend(loc='lower left')
    plt.title(type)
    plt.grid()
    
    os.makedirs(folder, exist_ok=True)
    plt.savefig(f'{folder}/{type}.png')

def plot_stuff(filenames, labels, name):
    data_filename, st_args, te_args, config = args_and_init(config_file="Framework/new_config.yaml")
    
    args = config['StudentTeacher']
    datas = []
    for i in filenames:
        file = open(f"/Users/sarinaxi/Desktop/Thesis/Framework/results/{i}/train_data.pkl", 'rb')
        datas.append(pickle.load(file))
        file.close()
        
    #train_loss = {f'Training {labels[i]}':np.mean(np.array(datas[i]['train_losses']).reshape(-1, 1), axis=1) for i in range(len(labels))}
    train_loss = {f'Student {labels[i]}':datas[i]['valid_losses'][args['epochs']::args['epochs']] for i in range(len(labels))}
    #valid_loss = {labels[i]:datas[i]['valid_losses'][::args['epochs']] for i in len(labels)}
   
    #student_train_acc = {f'Training {labels[i]}':np.mean(np.array(datas[i]['student_train_accs']).reshape(-1, 1), axis=1) for i in range(len(labels))}
    student_train_acc = {f'Student {labels[i]}':datas[i]['student_valid_accs'][args['epochs']::args['epochs']] for i in range(len(labels))}
    for i in range(len(labels)):
        student_train_acc[f'Teacher {labels[i]}'] = [datas[i]['teacher_train_accs']]*len(datas[i]['student_valid_accs'][args['epochs']::args['epochs']])
        #student_train_acc[f'Validation {labels[i]}'] = np.mean(np.array(datas[i]['student_valid_accs']).reshape(-1, 10), axis=1)
        #train_loss[f'Validation {labels[i]}'] = np.mean(np.array(datas[i]['valid_losses']).reshape(-1, 10), axis=1)
    plot(train_loss, "/Users/sarinaxi/Desktop/Thesis/Framework/results/", f'{name} Student Validation Loss Curve')
    plot(student_train_acc, "/Users/sarinaxi/Desktop/Thesis/Framework/results/", f'{name} Student Validation Accuracy Curve')

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

def run_stuff(dp, active_, folder, args, teacher, student, device, map, 
              train_dataloader, valid_dataloader, test_data, train_weight, 
              transformer=False):
    args['dp'] = dp
    active = active_
    args['folder'] = folder
    if not os.path.exists(args['folder']):
        os.makedirs(args['folder'])
    
    agent = StudentTeacher(teacher, student, args, device, map, args['similarity'], 
                            transformer, active, train_dataloader, valid_dataloader, test_data, train_weight)
    train_losses, student_train_accs, valid_losses, student_valid_accs, teacher_train_accs, teacher_valid_accs, acc, train_label, valid_label = agent.run()
    dic = {'train_losses': train_losses, 
            'student_train_accs': student_train_accs, 
            'valid_losses': valid_losses, 
            'student_valid_accs': student_valid_accs, 
            'teacher_train_accs': acc, 
            'teacher_valid_accs': teacher_valid_accs,
            'train_label': train_label, 
            'valid_label': valid_label}

    f = open(f"{args['folder']}/train_data.pkl","wb")
    pickle.dump(dic,f)
    f.close()
    plot_loss_acc(train_losses[::args['epochs']], valid_losses[::args['epochs']], 'Loss', args['folder'])
    plot_loss_acc(student_train_accs[::args['epochs']], student_valid_accs[::args['epochs']], 'Student Acc', args['folder'])
    plot_loss_acc(teacher_train_accs[::args['epochs']], teacher_valid_accs[::args['epochs']], 'Teacher Acc', args['folder'])
    plot_loss_acc(train_losses, valid_losses, 'Loss_all', args['folder'])
    plot_loss_acc(student_train_accs, student_valid_accs, 'Student Acc_all', args['folder'])
    plot_loss_acc(teacher_train_accs, teacher_valid_accs, 'Teacher Acc_all', args['folder'])
    plot_loss_acc(train_label, valid_label, 'Label Correctness', args['folder'])
    save_parameters(args, args['folder'])
        
if __name__ == '__main__':
    '''
    for i in ['d']:
        plot_stuff([f'SENTMENT_test_0.1_{i}_allclasses_diffarch', 
                    f'SENTMENT_test_1_{i}_allclasses_diffarch', f'SENTMENT_test_10_{i}_allclasses_diffarch', 
                f'SENTMENT_test_100_{i}_allclasses_diffarch', f'SENTMENT_test_1000_{i}_allclasses_diffarch',
                f'SENTMENT_test_0_allclasses_diffarch', f'SENTMENT_test_active_allclasses_diffarch'], [0.1, 1, 10, 100, 1000, r'$\infty$', 'active'], f'sens_active_allclasses')
    
    
    '''
    k = []
    labs = ['dp_0.1', 'dp_1', 'dp_10', 'dp_100', 'dp_1000','dp_10000', 'dp_inf']
    #labs = ['active_unbalanced_1', 'active_unbalanced_2', 'active_unbalanced_3', 'active_balanced_1', 'active_balanced_2', 'active_balanced_3']
    for i in labs:
        k.append(f'emotion_2_{i}')
    plot_stuff(k, labs, f'Emotions_Comparison')
    
    #def k():
    data_filename, st_args, te_args, config = args_and_init(config_file="Framework/new_config.yaml")
    
    args = config['StudentTeacher']
    pt = config['pt']
    student_bool = config['student_bool']
    teacher_bool = config['teacher_bool']
    teacher_student_bool = config['teacher_student_bool']
    teacher_student_bool2 = False

    transformer = config['transformer']
    active = config['active']
    #map = {'sadness': 0, 'joy': 1, 'surprise': 2, 'anger': 3, 'fear': 4, 'love': 5}
    map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    half = False
    normal = True
    #map = {'ham': 0, 'spam':1}
    if teacher_student_bool:
        tokenizer = BertTokenizer.from_pretrained(te_args['model'])
        train = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_unseen_pub44554.csv')
        sensitive = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_sensitive33003.csv')
        local = sensitive[sensitive['label'].isin([0, 1, 2])]
    
        print(f'Have sensitive dataset of size {len(local["label"])}')
        val_data, test_data, val_labels, test_labels = train_test_split(local['data'], local['label'],
                                                                        test_size=0.9,
                                                                        stratify=local['label'])
        new_train = train.sample(n=len(test_labels)).reset_index(drop=True)
        print(f'Train is size {len(new_train["label"])}')
        
        spam = get_data(filename='/Users/sarinaxi/Desktop/Thesis/Framework/data/spam_data/df_full.csv', map_={'ham': 0, 'spam':1}, downsample=False)
        if normal:
            train = (new_train['data'], new_train['label'])
        elif not half:
            spam = spam.sample(n=4263)
            new_train = new_train.sample(n=17052)
            train_data = pd.concat((spam, new_train))
            train_data = pd.concat((spam, new_train))
            train_data = train_data.sample(frac=1).reset_index(drop=True)
            train  = (train_data['data'], train_data['label'])
        else:
            spam = pd.concat((spam, spam))
            spam = spam.samoke(n = 10657)
            new_train = new_train.sample(n=10657)
            train_data = pd.concat((spam, new_train))
            train_data = train_data.sample(frac=1).reset_index(drop=True)
            train  = (train_data['data'], train_data['label'])
        

        train_dataloader, train_weight = tokenize(tokenizer, train, args['hidden'][0], args['batch_size'], 'train', RandomSampler, transformer, False)
        valid = (val_data.reset_index(drop=True), val_labels.reset_index(drop=True))
        test = (test_data.reset_index(drop=True), test_labels.reset_index(drop=True))
        print(f'Train Size: {len(train[0])} | Valid Size: {len(valid[0])} | Test Size: {len(test[0])}')
        valid_dataloader, no_weight = tokenize(tokenizer, valid, args['hidden'][0], args['batch_size'], 'valid', RandomSampler, transformer, False)
        test_data, no_w = tokenize(tokenizer, test, args['hidden'][0], args['batch_size'], 'test', RandomSampler, transformer, False)
    
        student = LSTMModelMulti2(len(map), 30522, st_args['hidden'], st_args['dropout']).to(device)
        check_ptmodel(model=student, args=st_args, train_loader=train_dataloader, valid_loader=valid_dataloader, 
                      test_data=test_data, train_weight=train_weight, attention=transformer, active=False)
        teacher = LSTMModelMulti2(len(map), 30522, te_args['hidden'], te_args['dropout']).to(device)
        check_ptmodel(model=teacher, args=te_args, train_loader=train_dataloader, valid_loader=valid_dataloader, 
                      test_data=test_data, train_weight=train_weight, attention=transformer, active=False)
        
        #-------------------------Student Teacher Finetuning------------------------
        agent = StudentTeacher(teacher, student, args, device, map, args['similarity'], 
                            transformer, active, train_dataloader, valid_dataloader, test_data, train_weight)
        train_losses, student_train_accs, valid_losses, student_valid_accs, teacher_train_accs, teacher_valid_accs, acc, train_label, valid_label = agent.run()
        dic = {'train_losses': train_losses, 
            'student_train_accs': student_train_accs, 
            'valid_losses': valid_losses, 
            'student_valid_accs': student_valid_accs, 
            'teacher_train_accs': acc, 
            'teacher_valid_accs': teacher_valid_accs,
            'train_label': train_label, 
            'valid_label': valid_label}

        f = open(f"{args['folder']}/train_data.pkl","wb")
        pickle.dump(dic,f)
        f.close()
        plot_loss_acc(train_losses[::args['epochs']], valid_losses[::args['epochs']], 'Loss', args['folder'])
        plot_loss_acc(student_train_accs[::args['epochs']], student_valid_accs[::args['epochs']], 'Student Acc', args['folder'])
        plot_loss_acc(teacher_train_accs[::args['epochs']], teacher_valid_accs[::args['epochs']], 'Teacher Acc', args['folder'])
        plot_loss_acc(train_losses, valid_losses, 'Loss_all', args['folder'])
        plot_loss_acc(student_train_accs, student_valid_accs, 'Student Acc_all', args['folder'])
        plot_loss_acc(teacher_train_accs, teacher_valid_accs, 'Teacher Acc_all', args['folder'])
        plot_loss_acc(train_label, valid_label, 'Label Correctness', args['folder'])
        save_parameters(args, args['folder'])
        
    elif teacher_student_bool2:
        
        #-------------------------Load Models--------------------
        train_dataloader, valid_dataloader, test_data, train_weight = process_data(filename=data_filename, 
                                                                               map=map, pre_train=True, 
                                                                               sequence_len=st_args['hidden'][0], 
                                                                               batch_size=st_args['batch_size'], 
                                                                               sampler=RandomSampler, 
                                                                               bert_model='bert-base-uncased',
                                                                               downsample=True,
                                                                               att=transformer)
    
        
        student = LSTMModelMulti2(len(map), 30522, st_args['hidden'], st_args['dropout']).to(device)
        #check_ptmodel(model=student, args=st_args, train_loader=train_dataloader, valid_loader=valid_dataloader, 
        #              test_data=test_data, train_weight=train_weight, attention=transformer, active=active)
        teacher = LSTMModelMulti2(len(map), 30522, te_args['hidden'], te_args['dropout']).to(device)
        check_ptmodel(model=teacher, args=te_args, train_loader=train_dataloader, valid_loader=valid_dataloader, 
                      test_data=test_data, train_weight=train_weight, attention=transformer, active=active)
        

        #---------------------START FINETUNING--------------------
        tokenizer = BertTokenizer.from_pretrained(te_args['model'])
        # ----------------Get Public Balanced-----------------
        file = open(f"/Users/sarinaxi/Desktop/Thesis/Framework/results/data/test16502.pkl", 'rb')
        test = pickle.load(file)
        file.close()
        file = open(f"/Users/sarinaxi/Desktop/Thesis/Framework/results/data/valid16501.pkl", 'rb')
        valid = pickle.load(file)
        file.close()
        
        public_data = pd.concat((test[0], valid[0]), ignore_index=True)
        public_labels = pd.concat((test[1], valid[1]), ignore_index=True)
        public = (public_data, public_labels)
        '''
        #downsampling
        maps = []
        for i in map:
            maps.append(len(public_labels[public_labels==i]))
        min_len = min(maps)
        
        for i in range(len(maps)):
            maps[i] = public_labels[public_labels==i].sample(n=min_len)
        inds = list(pd.concat(maps).index)
        np.random.shuffle(inds)
        balanced_public = (public_data.iloc[inds].reset_index(drop=True), public_labels.iloc[inds].reset_index(drop=True))
        print(f'We have a balanced Public Dataset of {len(balanced_public[0])} points')
        '''
        #print(f'We have a Public Dataset of {len(public[0])} points')
        train_dataloader, train_weight = tokenize(tokenizer, public, args['hidden'][0], args['batch_size'], 'train', RandomSampler, transformer, pt)
        
        # ------------------Get Private-------------------
        f = open(f"/Users/sarinaxi/Desktop/Thesis/Framework/results/data/sensitive_set.pkl", 'rb')
        sensitive = pickle.load(f)
        f.close()
        sensitive = sensitive.reset_index(drop=True)
        local = sensitive[sensitive['label'].isin([0, 1, 2])]
        print(f'We have a Partial Private Dataset of {len(local)} points')
        
        val_data, test_data, val_labels, test_labels = train_test_split(local['data'], local['label'],
                                                                        test_size=0.8,
                                                                        stratify=local['label'])
        valid = (val_data.reset_index(drop=True), val_labels.reset_index(drop=True))
        test = (test_data.reset_index(drop=True), test_labels.reset_index(drop=True))

        valid_dataloader, no_weight = tokenize(tokenizer, valid, args['hidden'][0], args['batch_size'], 'valid', RandomSampler, transformer, pt)
        test_data, no_w = tokenize(tokenizer, test, args['hidden'][0], args['batch_size'], 'test', RandomSampler, transformer, pt)
        
        #-------------------------Student Teacher Finetuning------------------------
        agent = StudentTeacher(teacher, student, args, device, map, args['similarity'], 
                            transformer, active, train_dataloader, valid_dataloader, test_data, train_weight)
        train_losses, student_train_accs, valid_losses, student_valid_accs, teacher_train_accs, teacher_valid_accs, acc, train_label, valid_label = agent.run()
        dic = {'train_losses': train_losses, 
            'student_train_accs': student_train_accs, 
            'valid_losses': valid_losses, 
            'student_valid_accs': student_valid_accs, 
            'teacher_train_accs': acc, 
            'teacher_valid_accs': teacher_valid_accs,
            'train_label': train_label, 
            'valid_label': valid_label}

        f = open(f"{args['folder']}/train_data.pkl","wb")
        pickle.dump(dic,f)
        f.close()
        plot_loss_acc(train_losses[::args['epochs']], valid_losses[::args['epochs']], 'Loss', args['folder'])
        plot_loss_acc(student_train_accs[::args['epochs']], student_valid_accs[::args['epochs']], 'Student Acc', args['folder'])
        plot_loss_acc(teacher_train_accs[::args['epochs']], teacher_valid_accs[::args['epochs']], 'Teacher Acc', args['folder'])
        plot_loss_acc(train_losses, valid_losses, 'Loss_all', args['folder'])
        plot_loss_acc(student_train_accs, student_valid_accs, 'Student Acc_all', args['folder'])
        plot_loss_acc(teacher_train_accs, teacher_valid_accs, 'Teacher Acc_all', args['folder'])
        plot_loss_acc(train_label, valid_label, 'Label Correctness', args['folder'])
        save_parameters(args, args['folder'])
        
    elif student_bool:
        
        data_filename = '/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_pretrain70831.csv'
        train_dataloader, valid_dataloader, test_data, train_weight = process_data(filename=data_filename, 
                                                                               map=map, pre_train=True, 
                                                                               sequence_len=st_args['hidden'][0], 
                                                                               batch_size=st_args['batch_size'], 
                                                                               sampler=RandomSampler, 
                                                                               bert_model='bert-base-uncased',
                                                                               downsample=False,
                                                                               att=transformer)
        student = LSTMModelMulti2(len(map), 30522, st_args['hidden'], st_args['dropout']).to(device)
        #student = LSTMModel(30522, st_args['embed_size'], st_args['hidden_size'], st_args['dropout']).to(device)
        pre_train(model=student, args=st_args, train_loader=train_dataloader, valid_loader=valid_dataloader,
                  test_data=test_data, train_weight=train_weight, attention=transformer, active=active)
        check_ptmodel(model=student, args=st_args, train_loader=train_dataloader, valid_loader=valid_dataloader, 
                      test_data=test_data, train_weight=train_weight, attention=transformer, active=active)
        
    elif teacher_bool:
        
        #data_filename = '/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/sentiment_data2.csv'
        data_filename = '/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_seen_pub252467.csv'
        train = get_data(filename=data_filename, map_=map, downsample=False)
        valid = get_data(filename='/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_unseen_pub44554.csv', map_=map, downsample=False)
        test = get_data(filename='/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_sensitive33003.csv', map_=map, downsample=False)

        train = (train['data'], train['label'])
        valid = (valid['data'], valid['label'])
        test = (test['data'], test['label'])
        sequence_len = te_args['hidden'][0]
        batch_size = te_args['batch_size']
        sampler = RandomSampler
        att = transformer
        tokenizer = BertTokenizer.from_pretrained(te_args['model'])
    
        # get sequency length
        seq_len = [len(i.split()) for i in train[0]]
        seq_len = int(np.ceil((pd.Series(seq_len).describe()['75%']) / 5) * 5)
        if not sequence_len:
            sequence_len = seq_len
        print(f'Set sequence length is: {sequence_len} | 75% data sequence length is: {seq_len}')
        train_dataloader, train_weight = tokenize(tokenizer, train, sequence_len, batch_size, 'train', sampler, att, True)
        valid_dataloader, no_weight = tokenize(tokenizer, valid, sequence_len, batch_size, 'valid', sampler, att, True)
        test_data, no_w = tokenize(tokenizer, test, sequence_len, batch_size, 'test', sampler, att, True)
        print(f'Train size: {len(train[0])} | Valid size: {len(valid[0])} | Test size: {len(test[0])}')
        teacher = LSTMModelMulti2(len(map), 30522, te_args['hidden'], te_args['dropout']).to(device)
        #teacher = LSTMModel(30522, te_args['embed_size'], te_args['hidden_size'], te_args['dropout']).to(device)
        pre_train(model=teacher, args=te_args, train_loader=train_dataloader, valid_loader=valid_dataloader,
                  test_data=test_data, train_weight=train_weight, attention=transformer, active=active)
        check_ptmodel(model=teacher, args=te_args, train_loader=train_dataloader, valid_loader=valid_dataloader, 
                      test_data=test_data, train_weight=train_weight, attention=transformer, active=active)
        