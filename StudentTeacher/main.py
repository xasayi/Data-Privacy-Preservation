import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np
sys.path.insert(0, '/Users/sarinaxi/Desktop/Thesis')

from SpamDetector.plotting_analytics import plot_loss_acc
from StudentTeacher.model import EmbedModel, LSTMModel
from StudentTeacher.student_teacher import StudentTeacher
from StudentTeacher.process_data import process_data, process_data_bert
from StudentTeacher.spam_detector import SpamDetector, model_performance
device = torch.device("cpu")
def args_and_init(student, teacher, config_file):
    # check GPU, don't use it since there's a bug with GRU
    print(torch.backends.mps.is_available())
    print(torch.backends.mps.is_built())
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)
    st_args, te_args = None, None
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
    return config['file_name'], st_args, te_args, config['StudentTeacher']

def split_df(df, ratio):
    spam = df[df['type']=='spam']
    ham = df[df['type']=='ham']
    ham_ratio = ham.sample(n = int(len(ham)*ratio), random_state = 67)
    spam_ratio = spam.sample(n = int(len(spam)*ratio), random_state = 67)
    ham_other = ham[~ham.index.isin(ham_ratio.index)]
    spam_other = spam[~spam.index.isin(spam_ratio.index)]
    ret_ratio = pd.concat([ham_ratio, spam_ratio])
    ret_ratio = ret_ratio.reindex(np.random.permutation(ret_ratio.index)).reset_index(drop=True)
    ret_other = pd.concat([ham_other, spam_other])
    ret_other = ret_other.reindex(np.random.permutation(ret_other.index)).reset_index(drop=True)
    return ret_ratio, ret_other

def pre_train(model, args, df, map, device=device):
    train_loader, valid_loader, test_data, weight = process_data_bert(df=df, splits=args['splits'], 
                                                                 bs=args['batch_size'], max_seq=args['input_size'], 
                                                                 map=map, downsample=args['downsample'])
    print(weight)
    agent = SpamDetector(model=model, train_dataloader=train_loader, device=device, lr=args['lr'], 
                                        batch_size=args['batch_size'], valid_dataloader=valid_loader, epochs=args['epochs'], 
                                        test_data=test_data, weights=weight, folder=args['folder'], weight_path=args['name'])
    train_losses, train_acc, valid_losses, valid_accs = agent.run()
    plot_loss_acc(train_losses, valid_losses, 'Loss', args['folder'])
    plot_loss_acc(train_acc, valid_accs, 'Acc', args['folder'])

def check_ptmodel(model, args, df, downsample, map):
    train_loader, valid_loader, test_data, weight = process_data_bert(df=df, splits=args['splits'], 
                                                                 bs=args['batch_size'], max_seq=args['input_size'], 
                                                                 downsample=downsample, map=map)
    agent = SpamDetector(model=model, train_dataloader=train_loader, device=device, lr=args['lr'], 
                                        batch_size=args['batch_size'], valid_dataloader=valid_loader, epochs=args['epochs'], 
                                        test_data=test_data, weights=weight, folder=args['folder'], weight_path=args['name'])
    agent.model.load_state_dict(torch.load(f"{args['folder']}/{args['name']}"))
    model_performance(args, agent.model, agent.test_data[0], agent.test_data[1], device, args['folder'])

if __name__ == '__main__':
    device = torch.device("cpu")
    # define variables 
    filename, st_args, te_args, args = args_and_init(True, True, "StudentTeacher/config.yaml")
    other_df = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/StudentTeacher/data/df8.csv')
    df = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/StudentTeacher/data/df_full.csv')
    df2 = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/StudentTeacher/data/df2.csv')
    train_loader, valid_loader, test_data, weight = process_data_bert(df=df, splits=args['splits'], 
                                                                 bs=args['batch_size'], max_seq=args['input_size'], 
                                                                 downsample=args['downsample'])
    
    student = LSTMModel(30522, st_args['embed_size'], st_args['hidden_size'], st_args['dropout']).to(device)
    teacher = LSTMModel(30522, te_args['embed_size'], te_args['hidden_size'], te_args['dropout']).to(device)

    #pre_train(student, st_args, df2)
    #pre_train(teacher, te_args, df)
    check_ptmodel(teacher, te_args, df, True)
    check_ptmodel(student, st_args, df, True)
    agent = StudentTeacher(df, teacher, student, device, args)
    train_losses, student_train_accs, valid_losses, student_valid_accs, teacher_train_accs, teacher_valid_accs = agent.run('cosine')
    '''for i in train_loader:
        pred_s = student(i[0])
        pred_t = teacher(i[0])
        print(i[1].flatten().tolist())
        print([1 if i > 0.5 else 0 for i in pred_s[-1].flatten()])
        print([1 if i > 0.5 else 0 for i in pred_t[-1].flatten()])
        break

    for i in train_loader:
        pred_s = student(i[0])
        pred_t = teacher(i[0])
        print(i[1].flatten().tolist())
        print([1 if i > 0.5 else 0 for i in pred_s[-1].flatten()])
        print([1 if i > 0.5 else 0 for i in pred_t[-1].flatten()])
        break
    for i in train_loader:
        pred_s = agent.student(i[0])
        pred_t = agent.teacher(i[0])
        print(i[1].flatten().tolist())
        print([1 if i > 0.5 else 0 for i in pred_s[-1].flatten()])
        print([1 if i > 0.5 else 0 for i in pred_t[-1].flatten()])
        break
    '''
    
    plot_loss_acc(train_losses, valid_losses, 'Loss', args['folder'])
    plot_loss_acc(student_train_accs, student_valid_accs, 'Student Acc', args['folder'])
    plot_loss_acc(teacher_train_accs, teacher_valid_accs, 'Teacher Acc', args['folder'])
    #pre_train(teacher, te_args, df)
    #student_agent = pre_train(student, st_args, student_df)
    #student_agent.model.load_state_dict(torch.load(f"{st_args['folder']}/{st_args['name']}"))
    #model_performance(st_args, student_agent.model, student_agent.test_data[0], student_agent.test_data[1], device, st_args['folder'])
    #check_ptmodel(teacher, te_args, df)
    #check_ptmodel(student, st_args, df)
    '''
    args = te_args
    train_loss, valid_loss, train_acc, valid_acc = [], [], [], []
    for i in range(3):

        train_loader, valid_loader, test_data, weight = process_data(df=df, splits=args['splits'], 
                                                                 bs=args['batch_size'], max_seq=args['input_size'], downsample=False)
        agent = SpamDetector(model=teacher, train_dataloader=train_loader, device=device, lr=args['lr'], 
                                        batch_size=args['batch_size'], valid_dataloader=valid_loader, epochs=12, 
                                        test_data=test_data, weights=weight, folder=args['folder'], weight_path=args['name'])
        train_losses1, train_acc1, valid_losses1, valid_accs1 = agent.run()
        
        agent.train_dataloader, agent.valid_dataloader, agent.test_data, weight = process_data_bert(df=df, splits=args['splits'], 
                                                                 bs=args['batch_size'], max_seq=args['input_size'], downsample=True)
        train_losses2, train_acc2, valid_losses2, valid_accs2 = agent.run()
    
        train_loss += train_losses1 + train_losses2
        valid_loss += valid_losses1 + valid_losses2
        train_acc += train_acc1 + train_acc2
        valid_acc += valid_accs1 + valid_accs2
    plot_loss_acc(train_loss, valid_loss, 'Loss', args['folder'])
    plot_loss_acc(train_acc, valid_acc, 'Acc', args['folder'])
    '''