import os
import sys
import yaml
import torch
import pandas as pd
import numpy as np
sys.path.insert(0, '/Users/sarinaxi/Desktop/Thesis')

from SpamDetector.plotting_analytics import plot_loss_acc
from StudentTeacherGRU.model import EmbedModel, LSTMModel
from StudentTeacherGRU.student_teacher import StudentTeacher
from StudentTeacherGRU.process_data import process_data, read_data
from StudentTeacherGRU.spam_detector import SpamDetector, model_performance

def args_and_init(student, teacher):
    # check GPU, don't use it since there's a bug with GRU
    print(torch.backends.mps.is_available())
    print(torch.backends.mps.is_built())
    with open("StudentTeacherGRU/config.yaml", 'r') as f:
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

def pre_train(model, args, df):
    train_loader, weight, valid_loader, test_data = process_data(df, args['vocab_size'], args['splits'], args['batch_size'], 
                                                                 args['input_size'], 'post', 'post', args['downsample'])
    agent = SpamDetector(model=model, train_dataloader=train_loader, device=device, lr=args['lr'], 
                                        batch_size=args['batch_size'], valid_dataloader=valid_loader, epochs=args['epochs'], 
                                        test_data=test_data, weights=weight, folder=args['folder'], weight_path=args['name'])
    train_losses, train_acc, valid_losses, valid_accs = agent.run()
    plot_loss_acc(train_losses, valid_losses, 'Loss', args['folder'])
    plot_loss_acc(train_acc, valid_accs, 'Acc', args['folder'])
    return agent

def check_ptmodel(model, args, df):
    train_loader, valid_loader, test_data, weight = process_data(df, args['vocab_size'], args['splits'], args['batch_size'], 
                                                                 args['input_size'], 'post', 'post', args['downsample'])
    agent = SpamDetector(model=model, train_dataloader=train_loader, device=device, lr=args['lr'], 
                                        batch_size=args['batch_size'], valid_dataloader=valid_loader, epochs=args['epochs'], 
                                        test_data=test_data, weights=weight, folder=args['folder'], weight_path=args['name'])
    agent.model.load_state_dict(torch.load(f"{args['folder']}/{args['name']}"))
    model_performance(args, agent.model, agent.test_data[0], agent.test_data[1], device, args['folder'])

if __name__ == '__main__':
    device = torch.device("cpu")
    # define variables 
    filename, st_args, te_args, args = args_and_init(True, True)
    other_df = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/StudentTeacherGRU/data/df8.csv')
    df = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/StudentTeacherGRU/data/df_full.csv')
    
    student = LSTMModel(st_args['vocab_size'], st_args['embed_size'], st_args['hidden_size'], st_args['dropout']).to(device)
    teacher = LSTMModel(te_args['vocab_size'], te_args['embed_size'], te_args['hidden_size'], te_args['dropout']).to(device)
    
    check_ptmodel(teacher, te_args, df)

    agent = StudentTeacher(df, teacher, student, device, args)

    train_losses, student_train_accs, valid_losses, student_valid_accs, teacher_train_accs, teacher_valid_accs = agent.run('cosine')
    plot_loss_acc(train_losses, valid_losses, 'Loss', args['folder'])
    plot_loss_acc(student_train_accs, student_valid_accs, 'Student Acc', args['folder'])
    plot_loss_acc(teacher_train_accs, teacher_valid_accs, 'Teacher Acc', args['folder'])
    #teacher_agent = pre_train(teacher, te_args, df)
    #student_agent = pre_train(student, st_args, student_df)
    #student_agent.model.load_state_dict(torch.load(f"{st_args['folder']}/{st_args['name']}"))
    #model_performance(st_args, student_agent.model, student_agent.test_data[0], student_agent.test_data[1], device, st_args['folder'])
    #check_ptmodel(teacher, te_args, df)
    #check_ptmodel(student, st_args, df)