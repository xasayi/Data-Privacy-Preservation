import torch
import torch.nn as nn
import pandas as pd
import numpy as np
from transformers import AdamW, BertTokenizerFast
from StudentTeacher.process_data import tokenize_bert, get_data, split_data
from StudentTeacher.similarity import find_similar
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
        self.loss  = nn.NLLLoss(self.weight) 
        self.device = device
    
    def get_dataloaders(self, args, sampler):
        dic1 = get_data(self.df, args['downsample'])
        dic1 = {i: dic1[i].tolist() for i in list(dic1)}
        tokenizer = BertTokenizerFast.from_pretrained('bert-base-uncased')
        if not args['input_size']:
            seq_len = [len(i.split()) for i in train[0]]
            args['input_size'] = min(int(np.ceil((pd.Series(seq_len).describe()['75%']) / 5) * 5), 100)
    
        train, valid, test = split_data(dic1, args['splits'])
        self.trainloader, self.weight = tokenize_bert(tokenizer, train, args['input_size'], args['batch_size']*args['factor'], 'train', sampler)
        self.validloader = tokenize_bert(tokenizer, valid, args['input_size'], args['batch_size']*args['factor'], 'train', sampler)[0]
        self.testloader = tokenize_bert(tokenizer, test, args['input_size'], args['batch_size'], 'train', sampler)[0]

    def get_acc(self, preds, labels):
        preds = preds.argmax(dim=1)
        accuracy = np.sum([1 if preds[i] == labels[i] else 0 for i in range(len(labels))]) / len(labels)
        return accuracy

    def spam_ham_acc(self, pri_label, pub_label):
        spam_count, ham_count = 0, 0
        spam_acc, ham_acc = 0, 0
        for i in range(len(pri_label)):
            if pri_label[i] == 1:
                spam_count += 1
                if pub_label[i] == 1:
                    spam_acc += 1
            else:
                ham_count += 1
                if pub_label[i] == 0:
                    ham_acc += 1
        return spam_acc/spam_count if spam_count else 0, ham_acc/ham_count if ham_count else 0

    def acc(self, a, b):
        return np.sum([1 if a[i] == b[i] else 0 for i in range(len(a))])/len(a)

    def get_loss(self, pri_labels, pri_tok, pub_tok, pub_labels, sim):
        # get private pool representations
        student_pri_lstm2, student_pri_pred = self.student(pri_tok)
        teacher_pri_lstm2, teacher_pri_pred = self.teacher(pri_tok)
        # get public pool embeddings
        student_pub_lstm2, student_pub_pred = self.student(pub_tok)
        

        # find index of public data that is similar to private data
        sim_pub_index = find_similar(student_pri_lstm2, student_pub_lstm2, sim)

        # get the similar public inputs and labels
        sim_pub_tok, sim_pub_labels = pub_tok[sim_pub_index], pub_labels[sim_pub_index]
        
        #print(f'Private Lab: {pri_labels}')
        #print(f'Public  Lab: {sim_pub_labels}')
        #print(f'Label   Acc: {self.acc(pri_labels, sim_pub_labels)}\n')
        
        #print(f'Student Pri: {student_pri_pred.argmax(dim=1)}, {self.acc(pri_labels, student_pri_pred.argmax(dim=1))}')
        #print(f'Student Pub: {student_pub_pred[sim_pub_index].argmax(dim=1)}, {self.acc(sim_pub_labels, student_pub_pred[sim_pub_index].argmax(dim=1))}\n')
        # get teacher prediction on similar data
        _, teacher_pub_pred = self.teacher(sim_pub_tok)
        #print(f'Teacher Pri: {teacher_pri_pred.argmax(dim=1)}, {self.acc(pri_labels, teacher_pri_pred.argmax(dim=1))}')
        #print(f'Teacher Pub: {teacher_pub_pred.argmax(dim=1)}, {self.acc(sim_pub_labels, teacher_pub_pred.argmax(dim=1))}\n\n')
        # get loss between student private prediction and teacher public prediction
        teacher_pub_pred_10 = teacher_pub_pred.argmax(dim=1)
        loss = self.loss(student_pri_pred, teacher_pub_pred_10)
        #student_pri_pred_10 = student_pri_pred.argmax(dim=1)
        return loss, student_pri_pred, teacher_pub_pred, sim_pub_labels
        
    def train(self, sim):
        self.student.train()
        total_loss, teacher_total_accuracy, student_total_accuracy = 0, 0, 0
        for step, (test_batch, train_pool) in enumerate(zip(self.testloader, self.trainloader)):

            pri_batch = [r.to(self.device) for r in test_batch]
            pub_pool = [r.to(self.device) for r in train_pool]
            pri_tok, pri_labels = pri_batch
            pub_tok, pub_labels = pub_pool
            
            self.student.zero_grad()
            
            loss, student_pri_pred, teacher_pub_pred, sim_pub_labels = self.get_loss(pri_labels, pri_tok, 
                                                                                     pub_tok, pub_labels, sim)
            incre_loss = loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()

            # print outputs
            if step % 5 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(self.testloader)))
                s_output = np.argmax(student_pri_pred.detach().cpu().numpy(), axis=1)
                t_output = np.argmax(teacher_pub_pred.detach().cpu().numpy(), axis=1)
                print(f'Pri Labels: {pri_labels.detach().cpu().numpy()}')
                print(f'Pub Labels: {sim_pub_labels.detach().cpu().numpy()}')
                print(f'Teach Pred: {t_output}')
                print(f'Stude Pred: {s_output}')
                acc = np.sum([1 if sim_pub_labels[i] == pri_labels[i] else 0 for i in range(len(pri_labels))])/len(pri_labels)
                spam_acc, ham_acc = self.spam_ham_acc(pri_labels, sim_pub_labels)
                print(f'Ham Acc: {ham_acc}; Spam Acc: {spam_acc}; Total Acc: {acc}\n')

            total_loss += incre_loss
            teacher_total_accuracy += self.get_acc(teacher_pub_pred, sim_pub_labels)
            # sanity check
            #teacher_total_accuracy += self.get_acc(teacher_pri_pred, pri_labels)
            student_total_accuracy += self.get_acc(student_pri_pred, pri_labels)
        step += 1
        avg_loss = total_loss / step
        student_avg_acc = student_total_accuracy / step
        teacher_avg_acc = teacher_total_accuracy / step
        return avg_loss, student_avg_acc, teacher_avg_acc

    def eval(self, sim):
        print("\nEvaluating...")
        self.student.eval()
        total_loss, teacher_total_accuracy, student_total_accuracy = 0, 0, 0

        for step, (test_batch, valid_pool) in enumerate(zip(self.testloader, self.validloader)):
            if step % 10 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(self.testloader)))
            pri_batch = [t.to(self.device) for t in test_batch]
            pub_pool = [t.to(self.device) for t in valid_pool]
            pri_tok, pri_labels = pri_batch
            pub_tok, pub_labels = pub_pool

            with torch.no_grad():
                # get the data representation and predictions from the student model
                loss, student_pri_pred, teacher_pub_pred, sim_pub_labels = self.get_loss(pri_labels, pri_tok,
                                                                                         pub_tok, pub_labels, sim)
                loss = loss.item()
                total_loss += loss
                teacher_total_accuracy += self.get_acc(teacher_pub_pred, sim_pub_labels)
                student_total_accuracy += self.get_acc(student_pri_pred, pri_labels)
        step += 1
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