import os
import sys
import yaml
import torch
sys.path.insert(0, '/Users/sarinaxi/Desktop/Thesis')

from SpamDetector.plotting_analytics import plot_loss_acc
from StudentTeacherGRU.model import EmbedModel
from StudentTeacherGRU.process_data import process_data
from StudentTeacherGRU.spam_detector import SpamDetector, model_performance

def test(args, spamDetector, folder, name, student_bool):
    train_losses, train_acc, valid_losses, valid_accs = spamDetector.run()
    spamDetector.model.load_state_dict(torch.load(f'{folder}/{name}'))
    
    # plot curves and evaluate model on test set 
    plot_loss_acc(train_losses, valid_losses, 'Loss', folder)
    plot_loss_acc(train_acc, valid_accs, 'Acc', folder)
    model_performance(args, spamDetector.model, spamDetector.test_data[0], spamDetector.test_data[1], device, folder)

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

if __name__ == '__main__':
    # check GPU, don't use it since there's a bug with GRU
    print(torch.backends.mps.is_available())
    print(torch.backends.mps.is_built())
    device = torch.device("cpu")
    # define variables 
    with open("StudentTeacherGRU/config.yaml", 'r') as f:
        config = yaml.safe_load(f)
    # Student Config
    st_args = config['Student']
    st_folder = st_args['folder']
    st_name = st_args['name']
    student = EmbedModel(1000, st_args['embed_size'], st_args['hidden_size'], st_args['dropout']).to(device)
    if not os.path.exists(st_folder):
        os.makedirs(st_folder)

    filename = '/Users/sarinaxi/Desktop/Thesis/SpamDetector/data/smsSpam/SMSSpamCollection.txt'
    train_loader, weight, valid_loader, test_data = process_data(filename, 1000, st_args['splits'], st_args['batch_size'], 
                                                                 st_args['input_size'], 'post', 'post')
    run_student(st_args, student, train_loader, device, valid_loader, test_data, weight, st_folder, st_name)
   