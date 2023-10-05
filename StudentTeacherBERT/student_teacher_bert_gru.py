from transformers import BertModel, BertTokenizerFast, AutoTokenizer

import torch
import torch.nn as nn
import numpy as np
from transformers import AdamW
from StudentTeacherBERT.process_data import process_data

teacher = BertModel.from_pretrained('bert-large-cased-whole-word-masking')
tokenizer = BertTokenizerFast.from_pretrained('bert-large-cased-whole-word-masking')

class StudentTeacher(nn.Module):
    def __init__(self, teacher, student, device, lr, batch_size, splits, epochs, private, public, folder, weight_path):
        super(StudentTeacher, self).__init__()
        self.student = student
        self.teacher = teacher
        self.optimizer = AdamW(self.student.parameters(), lr = lr)
        self.pri_train_loader, self.pri_val_loader, self.pri_test, pri_weight = process_data(tokenizer, splits, 32, private, 25)  
        self.pub_train_loader, self.pub_val_loader, self.pub_test, pub_weight = process_data(tokenizer, splits, 32, public, 25)  
        weights = (pri_weight + pub_weight)/2
        self.cross_entropy  = nn.CrossEntropyLoss(weight=weights.to(device)) 
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_path = weight_path
        self.folder = folder
        self.device = device

    def get_loss(self, pub_sent_id, pub_mask, pri_sent_id, train=True):
        teacher_preds = self.teacher(pub_sent_id, pub_mask)
        student_preds = self.student(pri_sent_id)
        loss = self.cross_entropy(student_preds, teacher_preds)
        total_loss = loss.item()
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.student.parameters(), 1.0)
            self.optimizer.step()
        return total_loss, student_preds, teacher_preds
        
 
    def get_acc(self, preds, labels):
        pred_y = np.argmax(preds.detach().cpu().numpy(), axis=1)
        accuracy = np.sum([1 if pred_y[j] == labels[j] else 0 for j in range(len(labels))]) / len(labels)
        return accuracy

    # change so it's compatible with rnns output, within only sent_id and no mask
    def train(self):
        self.student.train()
        total_loss, teacher_total_accuracy, student_total_accuracy = 0, 0, 0
        for step, (pri_batch, pub_batch) in enumerate(zip(self.pri_train_loader, self.pub_train_loader)):

            pri_batch = [r.to(self.device) for r in pri_batch]
            pub_batch = [r.to(self.device) for r in pub_batch]
            pri_sent_id, pri_mask, pri_labels = pri_batch
            pub_sent_id, pub_mask, pub_labels = pub_batch
            self.student.zero_grad()
            
            loss, student_preds, teacher_preds = self.get_loss(pub_sent_id, pub_mask, pri_sent_id)
            
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(self.pri_train_loader)))
                output = np.argmax(student_preds.detach().cpu().numpy(), axis=1)
                print(f'Student Pred: {output}')
                print(f'Student Targ: {pri_labels.detach().cpu().numpy()}')

                output = np.argmax(teacher_preds.detach().cpu().numpy(), axis=1)
                print(f'Teacher Pred: {output}')
                print(f'Teacher Targ: {pub_labels.detach().cpu().numpy()}')

            total_loss += loss
            teacher_total_accuracy += self.get_acc(teacher_preds, pub_labels)
            student_total_accuracy += self.get_acc(student_preds, pri_labels)
        avg_loss = total_loss / len(self.pri_train_loader)
        student_avg_acc = student_total_accuracy / len(self.pri_train_loader) 
        teacher_avg_acc = teacher_total_accuracy / len(self.pub_train_loader)
        return avg_loss, student_avg_acc, teacher_avg_acc

    def eval(self):
        print("\nEvaluating...")
        self.student.eval()
        total_loss, teacher_total_accuracy, student_total_accuracy = 0, 0, 0

        for step, (pri_batch, pub_batch) in enumerate(zip(self.pri_val_loader, self.pub_val_loader)):
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(self.pri_val_loader)))
            pri_batch = [t.to(self.device) for t in pri_batch]
            pub_batch = [t.to(self.device) for t in pub_batch]
            pri_sent_id, pri_mask, pri_labels = pri_batch
            pub_sent_id, pub_mask, pub_labels = pub_batch
            with torch.no_grad():
                loss, student_preds, teacher_preds = self.get_loss(pub_sent_id, pub_mask, pri_sent_id, train=False)
                total_loss += loss
                teacher_total_accuracy += self.get_acc(teacher_preds, pub_labels)
                student_total_accuracy += self.get_acc(student_preds, pri_labels)
        avg_loss = total_loss / len(self.pri_val_loader)
        student_avg_acc = student_total_accuracy / len(self.pri_val_loader) 
        teacher_avg_acc = teacher_total_accuracy / len(self.pub_val_loader) 
        return avg_loss, student_avg_acc, teacher_avg_acc

    def run(self):
        best_valid_loss = float('inf')
        train_losses, valid_losses = [], []
        student_train_accs, student_valid_accs = [], []
        teacher_train_accs, teacher_valid_accs = [], []
        for epoch in range(self.epochs):
            print('\n Epoch {:} / {:}'.format(epoch + 1, self.epochs))
            train_loss, student_train_acc, teacher_train_acc = self.train()
            valid_loss, student_valid_acc, teacher_train_acc = self.eval()

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