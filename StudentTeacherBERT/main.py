import os
import sys
import yaml
import torch
sys.path.insert(0, '/Users/sarinaxi/Desktop/Thesis')

import numpy as np
import pandas as pd
from transformers import BertModel, BertTokenizerFast
from SpamDetector.plotting_analytics import plot_loss_acc
from StudentTeacherBERT.base_model import PTModel, Model
from StudentTeacherBERT.process_data import load_data, process_data
from StudentTeacherBERT.spam_detector import SpamDetector, model_performance
from StudentTeacherBERT.student_teacher_bert_gru import StudentTeacher
from StudentTeacherBERT.student_teacher_gru_gru import StudentTeacherGRU

def test(args, spamDetector, folder, name, student_bool):
    train_losses, train_acc, valid_losses, valid_accs = spamDetector.run(student_bool)
    spamDetector.model.load_state_dict(torch.load(f'{folder}/{name}'))
    
    # plot curves and evaluate model on test set 
    plot_loss_acc(train_losses, valid_losses, 'Loss', folder)
    plot_loss_acc(train_acc, valid_accs, 'Acc', folder)
    model_performance(args, spamDetector.model, spamDetector.test_data[0], spamDetector.test_data[1], spamDetector.test_data[2], device, folder, student_bool)

def run_teacher(args, teacher, pub_train_loader, device, pub_val_loader, pub_test, weights, folder, name):
    teacher_spamDetector = SpamDetector(model=teacher, train_dataloader=pub_train_loader, device=device, lr=args['lr'], 
                                        batch_size=args['batch_size'], valid_dataloader=pub_val_loader, epochs=args['epochs'], 
                                        test_data=pub_test, weights=weights, folder=folder, weight_path=name)
    test(args, teacher_spamDetector, folder, name, False)

def run_student(args, student, pri_train_loader, device, pri_val_loader, pri_test, weights, folder, name):
    student_spamDetector = SpamDetector(model=student, train_dataloader=pri_train_loader, device=device, lr=args['lr'], 
                                        batch_size=args['batch_size'], valid_dataloader=pri_val_loader, epochs=args['epochs'], 
                                        test_data=pri_test, weights=weights, folder=folder, weight_path=name)
    test(args, student_spamDetector, folder, name, True)

def run_student_teacher(teacher, student, device, lr, bs, split, epoch, private, public, folder, name, te_folder, te_name):
    student_teacher = StudentTeacherGRU(teacher=teacher, student=student, device=device, lr=lr, batch_size=bs, splits=split, 
                                    epochs=epoch, private=private, public=public, folder=folder, weight_path=name)
    student_teacher.teacher.load_state_dict(torch.load(f'{te_folder}/{te_name}'))
    train_losses, student_train_accs, valid_losses, student_valid_accs, teacher_train_accs, teacher_valid_accs = student_teacher.run()
    student_teacher.student.load_state_dict(torch.load(f'{folder}/{name}'))
    
    # plot curves and evaluate model on test set 
    plot_loss_acc(train_losses, valid_losses, 'Loss', folder)
    plot_loss_acc(student_train_accs, student_valid_accs, 'Student_Acc', folder)
    plot_loss_acc(teacher_train_accs, teacher_valid_accs, 'Teacher_Acc', folder)

def get_data_downsamples(filename):
    with open(filename) as f:
        content = f.read()
    lines = content.split('\n')
    lines = np.array([i.split('\t') for i in lines][:-2])
    
    df = pd.DataFrame(lines, columns=['type', 'sms'])
    ham = df[df['type']=='ham'].drop_duplicates()
    spam = df[df['type']=='spam'].drop_duplicates()
    ham_less = ham.sample(n = len(spam), random_state = 44)
    df = pd.concat([ham_less, spam]).reset_index().drop(columns=['index'])
    idx = np.random.permutation(df.index)
    ret = df.reindex(idx)
    dic = {'data':list(ret['sms']), 'label': [1 if i == 'spam' else 0 for i in ret['type']]}
    return dic

if __name__ == '__main__':
    # check GPU, don't use it since there's a bug with GRU
    print(torch.backends.mps.is_available())
    print(torch.backends.mps.is_built())
    device = torch.device("cpu")

    # define variables 
    with open("StudentTeacher/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    
    # Teacher Config
    te_args = config['Teacher']
    te_arch = te_args['architecture']
    te_folder = te_args['folder']
    te_name = te_args['name']
    #pt_te_model = BertModel.from_pretrained(te_arch)
    tokenizer = BertTokenizerFast.from_pretrained(te_arch)
    teacher = Model(te_args['input_size'], te_args['hidden_size'], te_args['dropout']).to(device)
    #PTModel(pt_te_model, te_args['dropout']).to(device)
    if not os.path.exists(te_folder):
        os.makedirs(te_folder)
    
    # Student Config
    st_args = config['Student']
    st_folder = st_args['folder']
    st_name = st_args['name']
    student = Model(st_args['input_size'], st_args['hidden_size'], st_args['dropout']).to(device)
    if not os.path.exists(st_folder):
        os.makedirs(st_folder)

    # Student Teacher Config
    st_te_args = config['StudentTeacher']
    st_te_folder = st_te_args['folder']
    st_te_name = st_te_args['name']
    if not os.path.exists(st_te_folder):
        os.makedirs(st_te_folder)
    
    # load data
    #private, public = load_data(config['file_name'])
    #pri_train_loader, pri_val_loader, pri_test, pri_weight = process_data(tokenizer, st_args['splits'], 
    #                                                                      st_args['batch_size'], private, st_args['input_size'])
    #pub_train_loader, pub_val_loader, pub_test, pub_weight = process_data(tokenizer, te_args['splits'], 
    #                                                                      te_args['batch_size'], public, te_args['input_size'])
    filename = '/Users/sarinaxi/Desktop/Thesis/SpamDetector/data/smsSpam/SMSSpamCollection.txt'
    dic = get_data_downsamples(filename)
    train_loader, val_loader, test_data, weight = process_data(tokenizer, st_args['splits'], 
                                                                          st_args['batch_size'], dic, st_args['input_size'])
    #run_teacher(te_args, teacher, pub_train_loader, device, pub_val_loader, pub_test, pub_weight, te_folder, te_name)
    run_student(st_args, student, train_loader, device, val_loader, test_data, weight, st_folder, st_name)
    #run_student_teacher(teacher, student, device, st_te_args['lr'], st_te_args['batch_size'], st_te_args['splits'], 
    #                    st_te_args['epochs'], private, public, st_te_folder, st_te_name, te_folder, te_name)
    #student = SpamDetector(model=student, train_dataloader=pri_train_loader, device=device, lr=st_args['lr'], 
    #                        batch_size=st_args['batch_size'], valid_dataloader=pri_val_loader, epochs=st_args['epochs'], 
    #                       test_data=pri_test, weights=pri_weight, folder=st_folder, weight_path=st_name)
    
    #student.model.load_state_dict(torch.load(f'StudentTeacher/results/test_student/test_student.pt'))
    # plot curves and evaluate model on test set 
    #model_performance(st_args, student, student.test_data[0], student.test_data[1], student.test_data[2], device, st_folder, True)