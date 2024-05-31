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
from Framework.new_process_data import process_data, sanitize_data, get_data, split_data, tokenize
from torch.utils.data import RandomSampler, TensorDataset, DataLoader, RandomSampler, SequentialSampler
from sklearn.model_selection import train_test_split
from Framework.model import LSTMModelMulti, LSTMModel, LSTMModelMulti2
from Framework.classifier import Classifier, model_performance
from SpamDetector.plotting_analytics import plot_loss_acc
from Framework.new_student_teacher import StudentTeacher
from transformer import MyTransformer
from sklearn.metrics import classification_report

device = torch.device("cpu")
from transformers import BertTokenizer

SEED = 1024
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
    #print(datas)
    indices = np.array([7, 15, 23, 31, 39, 47, 55, 63, 71, 79])#, 87, 95, 103, 111, 119, 127, 135, 143, 151, 159])
    indices = np.linspace(0,199,200).astype(int)
    #print(datas)

    #train_loss = {f'Training {labels[i]}':np.mean(np.array(datas[i]['train_losses']).reshape(-1, 1), axis=1) for i in range(len(labels))}
    #train_loss = {f'Student {labels[i]}':datas[i]['valid_losses'][:25*args['epochs']:] for i in range(len(labels))}
    #train_loss = {f'Student {labels[i]}':np.array(datas[i]['valid_losses'][:10*args['epochs']])[indices] for i in range(len(labels))}
    #valid_loss = {labels[i]:datas[i]['valid_losses'][::args['epochs']] for i in len(labels)}
   

    #student_train_acc = {f'Training {labels[i]}':np.mean(np.array(datas[i]['student_train_accs']).reshape(-1, 1), axis=1) for i in range(len(labels))}
    student_train_acc, student_noisy = {}, {}
    accs = []
    #starts = [0.49]#, 0.49]#0.65, 0.84]
    print(labels)
    for i in range(len(labels)):
        #print(i)
        #print(len(datas[i]['student_valid_accs']))
        #print(datas[i]['student_valid_accs'])
        student_train_acc[f'Student Valid {labels[i]}'] = np.array(list(np.array(datas[i]['student_valid_accs'])[indices]))*100
        accs.append(np.array(datas[i]['student_valid_accs'])[-1])
        #student_train_acc[f'Student Train Noisy {labels[i]}'] = np.array(list(np.array(datas[i]['student_noisy_train_accs'])[indices]))*100
        #student_train_acc[f'Student Train Raw {labels[i]}'] = np.array(list(np.array(datas[i]['student_raw_train_accs'])[indices]))*100
        #student_train_acc[f'Teacher Train Noisy {labels[i]}'] = np.array(list(np.array(datas[i]['teacher_train_accs'])[indices]))*100
    #print(accs)
    indes = np.array([19, 39, 59, 79, 99, 119, 139, 159, 179, 199])
    for j in np.array(datas[i]['student_valid_accs'])[indes]:
        print(j)
    print('k')
    for j in np.array(datas[i]['student_noisy_train_accs'])[indes]:
        print(j)
    print('k')
    for j in np.array(datas[i]['student_raw_train_accs'])[indes]:
        print(j)
    print('k')
    for j in np.array(datas[i]['teacher_valid_accs'])[indes]:
        print(j)
    print('k')
    for j in np.array(datas[i]['teacher_train_accs'])[indes]:
        print(j)
    print('k')

    #print(labels)
    train_loss = {}
    tea = []
    for i in range(len(labels)):
        #print([datas[i]['teacher_valid_accs'][-1]]*(len(np.array(datas[i]['student_valid_accs'])[indices])+1))
        student_train_acc[f'Teacher Valid {labels[i]}'] = np.array([datas[i]['teacher_valid_accs'][-1]]*(1+len(np.array(datas[i]['student_valid_accs'])[indices])))*100
        tea.append(datas[i]['teacher_valid_accs'][-1])
        #student_train_acc[f'Teacher Train {labels[i]}'] = np.array([datas[i]['teacher_train_accs']]*(1+len(np.array(datas[i]['student_valid_accs'])[indices])))*100
        train_loss[f'Student {labels[i]}'] = np.array(datas[i]['valid_losses'])[indices]
    #print(tea)
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
    
    data_filename, st_args, te_args, config = args_and_init(config_file="Framework/new_config.yaml")
    
    args = config['StudentTeacher']
    pt = config['pt']
    student_bool = config['student_bool']
    teacher_bool = config['teacher_bool']
    teacher_student_bool = config['teacher_student_bool']
    
    active = config['active']
    plot_ = config['plot_graphs']
    normal = config['public_noise']
    half = config['half']
    #map = {'sadness': 0, 'joy': 1, 'surprise': 2, 'anger': 3, 'fear': 4, 'love': 5}
    map = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}
    #map = {'ham': 0, 'spam':1}
    if plot_: ## NEED TO CHANGE VARIABLES THIS PART FOR PLOTTING
        '''
        tokenizer = BertTokenizer.from_pretrained(te_args['model'])
        train = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_unseen69091.csv')
        local = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_private69092.csv')
        data = pd.concat((local, train)).reset_index(drop=True)
        
        #train = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/Framework/data/spam_data/df_remaining.csv')
        #train, sensitive = train_test_split(train,test_size=0.5)
        #train = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_private69092.csv')
        
        #local = sensitive[sensitive['label'].isin([0, 1, 2])
        val_data, test_data, val_labels, test_labels = train_test_split(local['data'], local['label'],
                                                                        test_size=0.95,
                                                                        stratify=local['label'])
        test_labels = data['label'][:80000]
        test_data = data['data'][:80000]
        print(f'Have sensitive dataset of size {len(test_data)}')
        new_train = data[80000:].sample(n=10000).reset_index(drop=True)
        #new_train = train.sample(n=len(test_labels)).reset_index(drop=True)
        print(f'Train is size {len(new_train["label"])}')
        train = (new_train['data'], new_train['label'])

        teacher = MyTransformer(block_size=50, vocab_size=30522, embeds_size=32,
                        drop_prob=st_args['dropout'], num_classes=len(map), num_heads=8, n_layers=2, device=torch.device("cpu"))
        train_dataloader, train_weight = tokenize(tokenizer, train, args['hidden'][0], args['batch_size'], 'train', RandomSampler, transformer, False)
        valid = (val_data.reset_index(drop=True), val_labels.reset_index(drop=True))
        test = (test_data.reset_index(drop=True), test_labels.reset_index(drop=True))
        print(f'Train Size: {len(train[0])} | Valid Size: {len(valid[0])} | Test Size: {len(test[0])}')
        valid_dataloader, no_weight = tokenize(tokenizer, valid, args['hidden'][0], args['batch_size'], 'valid', RandomSampler, transformer, False)
        test_data, no_w = tokenize(tokenizer, test, args['hidden'][0], args['batch_size'], 'test', RandomSampler, transformer, False)
        
        print('TEST DATA')
        tot_prob, replace_perc, test_data = sanitize_data(tokenizer, (test_data[0].clone(), test_data[1].clone()), 0.95, 0.00001)
        print('Replcaed:', replace_perc)
        check_ptmodel(model=teacher, args=te_args, train_loader=train_dataloader, valid_loader=valid_dataloader, 
                      test_data=test_data, train_weight=train_weight, attention=transformer, active=active)
        '''
        k = []
        #lab = ['1', '10', '100', '200', '500', '1000']
        #lab = ['0', '17708', '35416', '70831']#, '500', '1000']
        #lab = ['diffclass', 'sameclass']
        lab = ['dp0.1', 'dp1', 'dp10', 'dp100', 'dp250', 'dp500', 'dpinf']#, 'pub']
        lab = ['dp0.1', 'dp1', 'dp100', 'dp1000']#, 'pub']
        lab = ['20240326_ST_dp100','20240326_ST_dp100_private75', '20240326_ST_dp100_half_half',  '20240326_ST_dp100_private25', '20240326_ST_pub']
        
        lab = ['20240326_ST_dp100','20240326_ST_dp100_private20']
        j = ['0%', '80%']#, 'pub']
        
        #lab = ['1_unnoised', '1_noised', '100_unnoised', '100_noised', '1000_unnoised', '1000_noised']#lab = ['minor', 'balance', 'major']
        lab = ['20240328_ST_dp1_unnoised', '20240328_ST_dp100_unnoised', '20240328_ST_dp1000_unnoised']#, 'Emotion_TS_pub']#lab = ['3', '5', '8']
        j = ['dp:1', 'dp:100', 'dp:1000']#, 'pub']#lab = ['70831', '35416', '17708']
        lab = ['1', '10', '100', '1000']
        lab = ['20240329_ST_dp100_unnoised_private25']
        
        #lab = ['100', '100_noised', '100_newseed', '100_noised_newseed']
        #lab = ['_minor', '_balance', '_major', '']
        #j = ['25%', '50%', '75%', '100%']
        # lab = ['20240328_ST_dp1_noised', '20240328_ST_dp1_unnoised', '20240328_ST_dp1000_noised', 'Emotion_TS_pub']#lab = ['3', '5', '8']
        #j = ['dp:1', 'dp:100', 'dp:1000', 'pub']#lab = ['70831', '35416', '17708']
        lab = ['17708', '35416', '70831']
        j = ['18k', '35k', '70k']
        lab = ['3', '5', '8']
        lab = ['1', '10', '20', '30']
        lab = ['1_95_b', '1_95_dissimilar_testa', '1_97', '1_97_dissimilar_testa', '1_99', '1_99_dissimilar_testa']
        for i in lab:
            k.append(f'emotions_dp{i}')
            #k.append(i)
        #k = ['emotions_dp1_95']
        #lab = ['1']
        #file = open("/Users/sarinaxi/Desktop/Thesis/Framework/results/20240326_ST_pub/train_data.pkl", 'rb')
        #data = pickle.load(file)
        #file.close()    
        #print(data)
        plot_stuff(config, k, lab, f'new_95')
        
        '''

        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        sensitive = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_private69092.csv')
        local = sensitive#[sensitive['label'].isin([0, 1, 2])]
    
        val_data, test_data, val_labels, test_labels = train_test_split(local['data'], local['label'],
                                                                        test_size=0.95,
                                                                        stratify=local['label'])
        valid = (val_data.reset_index(drop=True), val_labels.reset_index(drop=True))
    
        test_data, no_w = tokenize(tokenizer, valid, 50, 128, 'test', RandomSampler, False, False)
        data, label = test_data[0], test_data[1]

        filename_transformer = "/Users/sarinaxi/Desktop/Thesis/Framework/results/Emotions_PT_S_70831_3_1layer_all/student.pt"
        trans = MyTransformer(block_size=50, vocab_size=30522, embeds_size=32,
                        drop_prob=0.1, num_classes=6, num_heads=8, n_layers=2, device=device)
        trans.load_state_dict(torch.load(filename_transformer))
        
        filename_lstm = "Framework_old/results_2024_before_feb/Emotion_PT_T_train/teacher.pt"
        lstm = LSTMModelMulti2(6, 30522, [50, 128, 64, 32, 16, 32], 0.1).to(device)
        lstm.load_state_dict(torch.load(filename_lstm))

        with torch.no_grad():
            trans_res = trans(data)[-1]
            lstm_res = lstm(data)[-1]
        
        trans_res = trans_res.argmax(dim=1).detach().cpu().numpy()
        lstm_res = lstm_res.argmax(dim=1).detach().cpu().numpy()
        report_tran = classification_report(label, trans_res)
        report_lstm = classification_report(label, lstm_res)
        confusion_matrix_tran = pd.crosstab(label, trans_res)
        confusion_matrix_lstm = pd.crosstab(label, lstm_res)
        print('TRANSFORMER')
        print(report_tran)
        print('LSTM')
        print(report_lstm)
        ''' 
        
    elif teacher_student_bool:
        #map = {'ham': 0, 'spam':1}

        tokenizer = BertTokenizer.from_pretrained(te_args['model'])
        train = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_unseen69091.csv')
        local = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_private69092.csv')
        data = pd.concat((local, train)).reset_index(drop=True)
        
        #train = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/Framework/data/spam_data/df_remaining.csv')
        #train, sensitive = train_test_split(train,test_size=0.5)
        #train = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_private69092.csv')
        
        #local = sensitive[sensitive['label'].isin([0, 1, 2])
        val_data, test_data, val_labels, test_labels = train_test_split(local['data'], local['label'],
                                                                        test_size=0.95,
                                                                        stratify=local['label'])
        test_labels = data['label'][:80000]
        test_data = data['data'][:80000]
        print(f'Have sensitive dataset of size {len(test_data)}')
        new_train = data[80000:].sample(n=10000).reset_index(drop=True)
        #new_train = train.sample(n=len(test_labels)).reset_index(drop=True)
        print(f'Train is size {len(new_train["label"])}')
        train = (new_train['data'], new_train['label'])
        
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
        
        

        train_dataloader, train_weight = tokenize(tokenizer, train, args['hidden'][0], args['batch_size'], 'train', RandomSampler, transformer, False)
        valid = (val_data.reset_index(drop=True), val_labels.reset_index(drop=True))
        test = (test_data.reset_index(drop=True), test_labels.reset_index(drop=True))
        print(f'Train Size: {len(train[0])} | Valid Size: {len(valid[0])} | Test Size: {len(test[0])}')
        valid_dataloader, no_weight = tokenize(tokenizer, valid, args['hidden'][0], args['batch_size'], 'valid', RandomSampler, transformer, False)
        test_data, no_w = tokenize(tokenizer, test, args['hidden'][0], args['batch_size'], 'test', RandomSampler, transformer, False)
    
        #student = LSTMModelMulti2(len(map), 30522, st_args['hidden'], st_args['dropout']).to(device)
        student = MyTransformer(block_size=50, vocab_size=30522, embeds_size=32,
                        drop_prob=st_args['dropout'], num_classes=len(map), num_heads=8, n_layers=1, device=torch.device("cpu"))
        #print('Student')
        #check_ptmodel(model=student, args=st_args, train_loader=train_dataloader, valid_loader=valid_dataloader, 
        #              test_data=valid_dataloader, train_weight=train_weight, attention=transformer, active=active)
        #check_ptmodel(model=student, args=st_args, train_loader=train_dataloader, valid_loader=valid_dataloader, 
        #              test_data=test_data, train_weight=train_weight, attention=transformer, active=active)

        #teacher = LSTMModelMulti2(len(map), 30522, te_args['hidden'], te_args['dropout']).to(device)
        teacher = MyTransformer(block_size=50, vocab_size=30522, embeds_size=32,
                        drop_prob=st_args['dropout'], num_classes=len(map), num_heads=8, n_layers=2, device=torch.device("cpu"))
        print('Teacher on test')
        check_ptmodel(model=teacher, args=te_args, train_loader=train_dataloader, valid_loader=valid_dataloader, 
                      test_data=test_data, train_weight=train_weight, attention=transformer, active=active)
        print('Teacher on valid')
        check_ptmodel(model=teacher, args=te_args, train_loader=train_dataloader, valid_loader=valid_dataloader, 
                      test_data=valid_dataloader, train_weight=train_weight, attention=transformer, active=active)
        
        #-------------------------Student Teacher Finetuning------------------------
        ''' 
        agent = StudentTeacher(teacher, student, args, device, map, args['similarity'], 
                            transformer, active, train_dataloader, valid_dataloader, test_data, train_weight)
        start = time.time()
        train_losses, student_train_accs, student_train_accs_raw, valid_losses, student_valid_accs, teacher_train_accs, teacher_valid_accs, acc, train_label, valid_label = agent.run()
        end = time.time()
        print(f'Took {round((end-start)/60, 3)} minutes')
        
        dic = {'train_losses': train_losses, 
            'student_train_accs': student_train_accs, 
            'student_raw_train_accs': student_train_accs_raw,
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
        plot_loss_acc(student_train_accs, student_valid_accs, 'Student Noisy Train Raw Valid', args['folder'])
        plot_loss_acc(student_train_accs_raw, student_valid_accs, 'Student Raw Train Raw Valid', args['folder'])
        plot_loss_acc(student_train_accs_raw, student_train_accs, 'Student Raw Train Noisy Train', args['folder'])
        plot_loss_acc(teacher_train_accs, teacher_valid_accs, 'Teacher Acc_all', args['folder'])
        plot_loss_acc(train_label, valid_label, 'Label Correctness', args['folder'])
        save_parameters(args, args['folder'])
        '''
    elif student_bool:
        
        #data_filename = '/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_pretrain70831.csv'
        #train = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_seen_pub252467.csv')
        #sensitive = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_private69092.csv')
        
        data_filename = '/Users/sarinaxi/Desktop/Thesis/Framework/data/spam_data/df_pre_train.csv'
        data_filename = '/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_seen_pub252467.csv'
        data = pd.read_csv(data_filename)
        #data = data[data['label'].isin([0, 1, 2])]
        #map = {'ham': 0, 'spam':1}
        train_dataloader, valid_dataloader, test_data, train_weight = process_data(filename=data_filename, 
                                                                               map=map, pre_train=True, 
                                                                               sequence_len=st_args['hidden'][0], 
                                                                               batch_size=st_args['batch_size'], 
                                                                               sampler=RandomSampler, 
                                                                               bert_model='bert-base-uncased',
                                                                               downsample=False,
                                                                               att=transformer,
                                                                               data=data)
        tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        #tot_prob, replace_perc, train_dataloader = sanitize_data(tokenizer, train_dataloader, 0.8, 250)
        #tot_prob, replace_perc, valid_dataloader = sanitize_data(tokenizer, valid_dataloader, 0.8, 250)


        train, valid, test = split_data(df=data, pre_train=pre_train)
    
        #toks = tokenize(tokenizer, [list(data['data']), list(data['label'])], sequence_len, batch_size, 'test', sampler, att, pre_train)[0][0]
        #toks_flat = toks.reshape(toks.shape[0]*toks.shape[1])
    
        train_dataloader, train_weight = tokenize(tokenizer, train, 50, st_args['batch_size'], 'train', RandomSampler, False, False)
        valid_dataloader, no_weight = tokenize(tokenizer, valid, 50, st_args['batch_size'], 'valid', RandomSampler, False, False)
        test_data, no_w = tokenize(tokenizer, test, 50, st_args['batch_size'], 'test', RandomSampler, False, False)

        #tot_prob, replace_perc, train_dataloader = sanitize_data(tokenizer, train_dataloader, 0.85, 0.1)
        #tot_prob, replace_perc, valid_dataloader = sanitize_data(tokenizer, valid_dataloader, 0.85, 0.1)

        trains = TensorDataset(train_dataloader[0], train_dataloader[1])
        sampler = RandomSampler(trains)
        train_dataloader = DataLoader(trains, sampler=sampler, batch_size=st_args['batch_size'])
        
        valids = TensorDataset(valid_dataloader[0], valid_dataloader[1])
        sampler = RandomSampler(valids)
        valid_dataloader = DataLoader(valids, sampler=sampler, batch_size=st_args['batch_size'])
        
        student = MyTransformer(block_size=50, vocab_size=30522, embeds_size=32,
                      drop_prob=st_args['dropout'], num_classes=len(map), num_heads=8, n_layers=1, device=torch.device("cpu"))
        #student = LSTMModelMulti2(len(map), 30522, st_args['hidden'], st_args['dropout']).to(device)
        #student = LSTMModel(30522, st_args['embed_size'], st_args['hidden_size'], st_args['dropout']).to(device)
        #train_weight = torch.concat((train_weight, torch.tensor([0, 0, 0])))
        print(train_weight)
        pre_train(model=student, args=st_args, train_loader=train_dataloader, valid_loader=valid_dataloader,
                  test_data=test_data, train_weight=train_weight, attention=transformer, active=active)
        check_ptmodel(model=student, args=st_args, train_loader=train_dataloader, valid_loader=valid_dataloader, 
                      test_data=test_data, train_weight=train_weight, attention=transformer, active=active)
        
    elif teacher_bool:
        tokenizer = BertTokenizer.from_pretrained(te_args['model'])
        #train = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_pretrain70831.csv')
        train = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_seen_pub252467.csv')
        sensitive = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_private69092.csv')
       
        #local = sensitive[sensitive['label'].isin([0, 1, 2])]
        #train = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/Framework/data/spam_data/df_remaining.csv')
        #sensitive = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/Framework/data/spam_data/df_pre_train.csv')
        #map = {'ham': 0, 'spam':1}
        local = sensitive
        print(f'Have sensitive dataset of size {len(local["label"])}')
        val_data, test_data, val_labels, test_labels = train_test_split(local['data'], local['label'],
                                                                        test_size=0.95,
                                                                        stratify=local['label'])
        
        #test_labels = test_labels[:20000]
        #test_data = test_data[:20000]

        data = (pd.concat((test_data, train['data'])).reset_index(drop=True), pd.concat((test_labels, train['label'])).reset_index(drop=True))
        
        train_data, temp_data, train_labels, temp_labels = train_test_split(data[0], data[1],test_size=0.10,stratify=data[1])
        valid_data, test_data, valid_labels, temp_labels = train_test_split(temp_data, temp_labels, test_size=0.50,stratify=temp_labels)
        
        train = (train_data.reset_index(drop=True), train_labels.reset_index(drop=True))
        valid = (valid_data.reset_index(drop=True), valid_labels.reset_index(drop=True))
        test = (test_data.reset_index(drop=True), temp_labels.reset_index(drop=True))
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
        teacher = MyTransformer(block_size=50, vocab_size=30522, embeds_size=32,
                        drop_prob=te_args['dropout'], num_classes=len(map), num_heads=8, n_layers=1, device=torch.device("cpu"))
        teacher = LSTMModelMulti2(len(map), 30522, te_args['hidden'], te_args['dropout']).to(device)
        #teacher = LSTMModel(30522, te_args['embed_size'], te_args['hidden_size'], te_args['dropout']).to(device)
        start = time.time()
        pre_train(model=teacher, args=te_args, train_loader=train_dataloader, valid_loader=valid_dataloader,
                  test_data=test_data, train_weight=train_weight, attention=transformer, active=active)
        end = time.time()
        print(F'TOOK {round(end-start, 2)} SECONDS')
        #i=1000
        #used_data = (val_data.reset_index(drop=True), val_labels.reset_index(drop=True))
        #used_data, no_w = tokenize(tokenizer, used_data, sequence_len, batch_size, 'test', sampler, att, True)
        #tot_prob, replace_perc, test_data = sanitize_data(tokenizer, used_data, 0.85, i)
        
        #print(f'Used DP is {i}')
        check_ptmodel(model=teacher, args=te_args, train_loader=train_dataloader, valid_loader=valid_dataloader, 
                      test_data=test_data, train_weight=train_weight, attention=transformer, active=True)
        
        