import torch
import torch.nn as nn
import numpy as np
from transformers import AdamW
from StudentTeacherGRU.process_data import tokenize, get_data, split_data
from StudentTeacherGRU.similarity import find_similar
from tensorflow.keras.preprocessing.text import Tokenizer
from torch.utils.data import RandomSampler

class StudentTeacher(nn.Module):
    def __init__(self, df, teacher, student, device, args):
        super(StudentTeacher, self).__init__()
        self.teacher = teacher
        self.student = student
        self.args = args
        self.df = df
        self.epochs = args['epochs']
        self.batch_size = args['batch_size']
        self.lr = args['lr']
        self.weight_path = args['name']
        self.folder = args['folder']
        self.optimizer = AdamW(self.student.parameters(), lr = self.lr)
        self.get_dataloaders(args, RandomSampler)
        self.loss  = nn.BCELoss() 
        self.device = device
    
    def get_dataloaders(self, args, sampler):
        dic1 = get_data(self.df, args['downsample'])
        tokenizer = Tokenizer(num_words = args['vocab_size'], char_level=False, oov_token = "<OOV>")
        tokenizer.fit_on_texts(dic1['data'])
    
        train, valid, test = split_data(dic1, args['splits'])
        self.trainloader = tokenize(tokenizer, train, args['input_size'], 'post', 'post', args['batch_size']*args['factor'], 'train', sampler)[0]
        self.validloader = tokenize(tokenizer, valid, args['input_size'], 'post', 'post', args['batch_size']*args['factor'], 'train', sampler)[0]
        self.testloader = tokenize(tokenizer, test, args['input_size'], 'post', 'post', args['batch_size'], 'train', sampler)[0]

    def get_acc(self, preds, labels):
        pred_y = np.argmax(preds.detach().cpu().numpy(), axis=1)
        accuracy = np.sum([1 if pred_y[j] == labels[j] else 0 for j in range(len(labels))]) / len(labels)
        return accuracy

    def train(self, sim):
        self.student.train()
        total_loss, teacher_total_accuracy, student_total_accuracy = 0, 0, 0
        for step, (test_batch, train_pool) in enumerate(zip(self.testloader, self.trainloader)):

            pri_batch = [r.to(self.device) for r in test_batch]
            pub_pool = [r.to(self.device) for r in train_pool]
            pri_tok, pri_labels = pri_batch
            pub_tok, pub_labels = pub_pool
            
            self.student.zero_grad()
            # get the data representation and predictions from the student model
            student_pri_lstm2, student_pri_fc1, student_pri_pred = self.student(pri_tok)
            student_pub_lstm2, student_pub_fc1, student_pub_pred = self.student(pub_tok)

            # find index of public data that is similar 
            lstm = True
            if lstm:
                sim_pub_index = find_similar(student_pri_lstm2, student_pub_lstm2, sim)
            else:
                sim_pub_index = find_similar(student_pri_fc1, student_pub_fc1, sim)
            
            # get the similar public inputs and labels
            sim_pub_tok, sim_pub_labels = pub_tok[sim_pub_index], pub_labels[sim_pub_index]
            
            # get teacher prediction on similar data
            teacher_pub_lstm2, teacher_pub_fc1, teacher_pub_pred = self.teacher(sim_pub_tok)
            
            # get loss between student private prediction and teacher public prediction
            loss = self.loss(student_pri_pred, teacher_pub_pred)
            incre_loss = loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()

            # print outputs
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(self.testloader)))
                output = np.argmax(student_pri_pred.detach().cpu().numpy(), axis=1)
                print(f'Student Pred: {output}')
                print(f'Student Targ: {pri_labels.detach().cpu().numpy()}')

                output = np.argmax(teacher_pub_pred.detach().cpu().numpy(), axis=1)
                print(f'Teacher Pred: {output}')
                print(f'Teacher Targ: {sim_pub_labels.detach().cpu().numpy()}')

            total_loss += incre_loss
            teacher_total_accuracy += self.get_acc(teacher_pub_pred, sim_pub_labels)
            student_total_accuracy += self.get_acc(student_pri_pred, pri_labels)
        avg_loss = total_loss / step
        student_avg_acc = student_total_accuracy / step
        teacher_avg_acc = teacher_total_accuracy / step
        return avg_loss, student_avg_acc, teacher_avg_acc

    def eval(self, sim):
        print("\nEvaluating...")
        self.student.eval()
        total_loss, teacher_total_accuracy, student_total_accuracy = 0, 0, 0

        for step, (test_batch, valid_pool) in enumerate(zip(self.testloader, self.validloader)):
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(self.testloader)))
            pri_batch = [t.to(self.device) for t in test_batch]
            pub_pool = [t.to(self.device) for t in valid_pool]
            pri_tok, pri_labels = pri_batch
            pub_tok, pub_labels = pub_pool

            with torch.no_grad():
                # get the data representation and predictions from the student model
                student_pri_lstm2, student_pri_fc1, student_pri_pred = self.student(pri_tok)
                student_pub_lstm2, student_pub_fc1, student_pub_pred = self.student(pub_tok)

                # find index of public data that is similar 
                lstm = True
                if lstm:
                    sim_pub_index = find_similar(student_pri_lstm2, student_pub_lstm2, sim)
                else:
                    sim_pub_index = find_similar(student_pri_fc1, student_pub_fc1, sim)
            
                # get the similar public inputs and labels
                sim_pub_tok, sim_pub_labels = pub_tok[sim_pub_index], pub_labels[sim_pub_index]
            
                # get teacher prediction on similar data
                teacher_pub_lstm2, teacher_pub_fc1, teacher_pub_pred = self.teacher(sim_pub_tok)
            
                # get loss between student private prediction and teacher public prediction
                loss = self.loss(student_pri_pred, teacher_pub_pred)
                loss = loss.item()
                total_loss += loss
                teacher_total_accuracy += self.get_acc(teacher_pub_pred, sim_pub_labels)
                student_total_accuracy += self.get_acc(student_pri_pred, pri_labels)
        avg_loss = total_loss / step
        student_avg_acc = student_total_accuracy / step
        teacher_avg_acc = teacher_total_accuracy / step
        return avg_loss, student_avg_acc, teacher_avg_acc

    def run(self, sim):
        best_valid_loss = float('inf')
        train_losses, valid_losses = [], []
        student_train_accs, student_valid_accs = [], []
        teacher_train_accs, teacher_valid_accs = [], []
        for epoch in range(self.epochs):
            print('\n Epoch {:} / {:}'.format(epoch + 1, self.epochs))
            train_loss, student_train_acc, teacher_train_acc = self.train(sim)
            valid_loss, student_valid_acc, teacher_train_acc = self.eval(sim)

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.student.state_dict(), f'{self.folder}/{self.weight_path}')

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            student_train_accs.append(student_train_acc)
            student_valid_accs.append(student_valid_acc)
            teacher_train_accs.append(teacher_train_acc)
            teacher_valid_accs.append(teacher_train_acc)

            print(f'\nTraining Loss: {train_loss:.3f}')
            print(f'Validation Loss: {valid_loss:.3f}')
            print(f'\nSTUDENT Training Acc: {student_train_acc:.3f}')
            print(f'Validation Acc: {student_valid_acc:.3f}')
            print(f'\nTEACHER Training Acc: {teacher_train_acc:.3f}')
            print(f'Validation Acc: {teacher_train_acc:.3f}')
        return train_losses, student_train_accs, valid_losses, student_valid_accs, teacher_train_accs, teacher_valid_accs