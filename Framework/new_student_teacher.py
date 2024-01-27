import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from transformers import AdamW
from Framework.similarity import find_similar
from scipy.stats import entropy
from Framework.new_process_data import sanitize_data
from transformers import BertTokenizer
from Framework.classifier import model_performance

class StudentTeacher(nn.Module):
    def __init__(self, teacher, student, args, device, map, sim, att, active, train_dataloader, valid_dataloader, test_data, train_weight):
        super(StudentTeacher, self).__init__()
        
        self.teacher = teacher
        self.student = student
        self.device = device
        self.map = map
        self.sim = sim
        self.att = att
        self.active = active
        self.factor = args['factor']

        self.queries = args['queries'] # amount of data added each round
        self.epochs = args['epochs'] # times we run the data
        self.iters = args['iters'] # how many times we add data 

        self.bs = args['batch_size']
        self.lr = args['lr']
        self.wd = args['wd']
        self.dp = args['dp']
        self.eps = args['eps']
        self.sens = args['sens_ratio']
        
        self.weight_path = args['name']
        self.folder = args['folder']
        self.train_tok = None
        self.train_label = None
        self.train_mask = None
        
        pub_data = [r.to(self.device) for r in train_dataloader]
        if self.att:
            self.pub_toks, self.pub_masks, self.pub_labels = pub_data
        else:
            self.pub_toks, self.pub_labels = pub_data
        self.valid_loader = valid_dataloader
        print(f'Differential Privacy: {self.dp}')
        if self.dp:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            tot_prob, replace_perc, self.test_data = sanitize_data(tokenizer, test_data, self.sens, self.eps)
            print(f'Total Probability is {tot_prob}')
            print(f'Replaced percentage is {replace_perc}')
            tot_prob, replace_perc, self.valid_loader = sanitize_data(tokenizer, valid_dataloader, self.sens, self.eps)
            print(f'Total Probability is {tot_prob}')
            print(f'Replaced percentage is {replace_perc}')
        else:
            self.test_data = test_data
            self.valid_loader = valid_dataloader
        acc = model_performance(args, self.teacher, self.test_data[0], self.test_data[1], device, args['folder'], mask=None)
        self.teacher_acc = acc
        self.test_data_raw = test_data
        self.valid_loader_raw = valid_dataloader
        self.weight = train_weight
        self.optimizer = AdamW(self.student.parameters(), lr=self.lr, weight_decay=self.wd)
        self.loss = nn.NLLLoss(self.weight) 
        self.target_classes = torch.unique(self.test_data[2] if self.att else self.test_data[1])

    def get_acc(self, preds, labels):
        preds = preds.argmax(dim=1)
        return self.acc(preds, labels)

    def acc(self, a, b):
        return np.sum([1 if a[i] == b[i] else 0 for i in range(len(a))])/len(a)

    def get_loss_sim(self, pri_labels, pri_mask, pri_tok, pub_tok, pub_mask, pub_labels):
        # query teacher with insensitive data using cosine similarity
        if self.att:
            student_pri_pred = self.student(pri_tok, pub_mask)
            with torch.no_grad():
                student_pub_pred = self.student(pub_tok, pub_mask)
        else:
            student_pri_pred = self.student(pri_tok)[-1]
            with torch.no_grad():
                student_pub_pred = self.student(pub_tok)[-1]

        sim_pub_index = find_similar(student_pri_pred, student_pub_pred, self.sim)
        sim_pub_tok, sim_pub_labels = pub_tok[sim_pub_index], pub_labels[sim_pub_index]
        sim_pub_mask = pub_mask[sim_pub_index] if self.att else None

        if self.att:
            teacher_pub_pred = self.teacher(sim_pub_tok, sim_pub_mask)
        else:
            teacher_pub_pred = self.teacher(sim_pub_tok)[-1]
       
        teacher_pub_pred_argmax = teacher_pub_pred.argmax(dim=1)
        loss = self.loss(student_pri_pred, sim_pub_labels)

        return loss, student_pri_pred, teacher_pub_pred_argmax, sim_pub_labels

    def get_loss(self, pri_labels, pri_mask, pri_tok, pub_tok, pub_mask, pub_labels):
        # use DP on text data and train
        if self.att:
            student_pri_pred = self.student(pri_tok, pri_mask)
        else:
            student_pri_pred = self.student(pri_tok)[-1]
        
        sim_pub_tok, sim_pub_labels = pri_tok, pri_labels
        sim_pub_mask = pri_mask if self.att else None

        if self.att:
            teacher_pub_pred = self.teacher(sim_pub_tok, sim_pub_mask)
        else:
            teacher_pub_pred = self.teacher(sim_pub_tok)[-1]

        teacher_pub_pred_argmax = teacher_pub_pred.argmax(dim=1)
        
        loss = self.loss(student_pri_pred, teacher_pub_pred_argmax)
        return loss, student_pri_pred, teacher_pub_pred_argmax, sim_pub_labels

    def get_loss_active(self, pri_labels, pri_mask, pri_tok, pub_tok, pub_mask, pub_labels):
        # random query teacher with sensitive data
        if self.att:
            student_pub_pred = self.student(pub_tok, pub_mask)
        else:
            student_pub_pred = self.student(pub_tok)[-1]
        
        order = self.sample_entropy(student_pub_pred)
        order = np.array(order)
        sim_pub_tok, sim_pub_labels = pub_tok[order], pub_labels[order]
        sim_pub_mask = pub_mask[order] if self.att else None

        if self.att:
            teacher_pub_pred = self.teacher(sim_pub_tok, sim_pub_mask)
        else:
            teacher_pub_pred = self.teacher(sim_pub_tok)[-1]
        teacher_pub_pred_argmax = teacher_pub_pred.argmax(dim=1)
        
        loss = self.loss(student_pub_pred[order], teacher_pub_pred_argmax)
        return loss, student_pub_pred[order], teacher_pub_pred_argmax, sim_pub_labels
    
    def get_pub_data(self, pub_tok, pub_mask, pub_label, pri_data_len):
        if self.train_tok is None:
            #target_class_ind = torch.isin(pub_label,self.target_classes)
            #train_tok = pub_tok[target_class_ind]
            #train_label = pub_label[target_class_ind]
            #if self.att: train_mask = pub_mask[target_class_ind]
            train_tok = pub_tok
            train_label = pub_label
            if self.att: train_mask = pub_mask
            print(pri_data_len)
            print(len(train_tok))
            indices = random.sample(list(np.linspace(0, len(train_tok)-1, len(train_tok))), pri_data_len)
            self.train_tok = train_tok[indices]
            self.train_label = train_label[indices]
            if self.att: self.train_mask = train_mask[indices]
        else:
            while len(self.train_tok) < pri_data_len:
                ind = random.sample(list(np.linspace(0, len(pub_tok)-1, len(pub_tok))), 1)
                #if int(pub_label[ind]) in self.target_classes:
                self.train_tok = torch.cat((self.train_tok, pub_tok[ind]), 0)
                self.train_label = torch.cat((self.train_label, pub_label[ind]), 0)
                if self.att: self.train_mask = torch.cat((self.train_mask, pub_mask[ind]), 0)
                ind = random.sample(list(np.linspace(0, len(pub_tok)-1, len(pub_tok))), 1)

    def get_pub_data_actie(self, pub_tok, pub_mask, pub_label, pri_data_len):
        if self.att:
            student_pub_pred = self.student(pub_tok, pub_mask)
        else:
            student_pub_pred = self.student(pub_tok)[-1]
        
        ind = self.sample_entropy(student_pub_pred)
        if self.train_tok is None:
            order = []
            i = 0
            while len(order) < pri_data_len:
                #if int(pub_label[ind[i]]) in self.target_classes:
                order.append(ind[i])
                i += 1
            self.train_tok = pub_tok[order]
            self.train_label = pub_label[order]
            if self.att: self.train_mask = pub_mask[order]
        else:
            order = []
            i = 0
            while len(order) < pri_data_len-len(self.train_tok):
                #if int(pub_label[ind[i]]) in self.target_classes:# and pub_tok[ind[i]] not in self.train_tok:
                order.append(ind[i])
                i += 1
            
            self.train_tok = torch.cat((self.train_tok, pub_tok[order]), 0)
            self.train_label = torch.cat((self.train_label, pub_label[order]), 0)
            if self.att: self.train_mask = torch.cat((self.train_mask, pub_mask[order]), 0)

    def train(self):
        self.student.train()
        total_loss, teacher_total_accuracy, student_total_accuracy, label_correctness = 0, 0, 0, 0

        step = 0
        pub_pool_ind = [random.sample(list(np.linspace(0, len(self.pub_toks)-1, len(self.pub_toks))), len(self.used_sens_data[0])*self.factor)]
        if self.att:
            pub_pool_tok, pub_pool_mask, pub_pool_label = self.pub_toks[pub_pool_ind], self.pub_masks[pub_pool_ind], self.pub_labels[pub_pool_ind]
        else:
            pub_pool_tok, pub_pool_label = self.pub_toks[pub_pool_ind], self.pub_labels[pub_pool_ind]
            pub_pool_mask = None
            
        if self.active:
            self.get_pub_data(pub_pool_tok, pub_pool_mask, pub_pool_label, len(self.used_sens_data[0]))
            #self.get_pub_data_actie(pub_pool_tok, pub_pool_mask, pub_pool_label, len(self.used_sens_data[0]))
        else:
            self.get_pub_data(pub_pool_tok, pub_pool_mask, pub_pool_label, len(self.used_sens_data[0]))
        
        for i in range(0, len(self.used_sens_data[0]), self.bs):
            if i+self.bs < len(self.used_sens_data[0]):
                last_ind = i+self.bs
            else:
                last_ind = -1
            
            if self.att:
                pri_tok = self.used_sens_data[0][i:last_ind]
                pri_mask = self.used_sens_data[1][i:last_ind]
                pri_labels = self.used_sens_data[2][i:last_ind]
                pub_tok = self.train_tok[i:last_ind]
                pub_mask = self.train_mask[i:last_ind]
                pub_label = self.train_label[i:last_ind]
            else:
                pri_tok = self.used_sens_data[0][i:last_ind]
                pri_labels = self.used_sens_data[1][i:last_ind]
                pub_tok = self.train_tok[i:last_ind]
                pub_label = self.train_label[i:last_ind]
                pri_mask, pub_mask = None, None
            
            self.student.zero_grad()                                                                            
            if self.sim is not None:
                print(self.sim)
                inds = torch.isin(pub_pool_label,self.target_classes)
                pub_pool_mask = pub_pool_mask[inds] if self.att else None
                loss, student_pri_pred, teacher_pub_pred, sim_pub_labels = self.get_loss_sim(pri_labels, pri_mask, pri_tok,
                                                                                             pub_pool_tok[inds], pub_pool_mask, pub_pool_label[inds])
            elif self.active:
                loss, student_pri_pred, teacher_pub_pred, sim_pub_labels = self.get_loss_active(pri_labels, pri_mask, pri_tok,
                                                                                                pub_tok, pub_mask, pub_label)
            else:
                mask = self.used_sens_data_raw[1][i:last_ind] if self.att else None
                loss, student_pri_pred, teacher_pub_pred, sim_pub_labels = self.get_loss(self.used_sens_data_raw[1][i:last_ind], 
                                                                                         mask, self.used_sens_data_raw[0][i:last_ind],
                                                                                         pri_tok, pri_mask, pri_labels)
            incre_loss = loss.item()
            loss.backward()
            self.optimizer.step()
            step += 1

            # print outputs
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(self.used_sens_data[0])//self.bs))
                s_output = np.argmax(student_pri_pred.detach().cpu().numpy(), axis=1)
                #t_output = np.argmax(teacher_pub_pred.detach().cpu().numpy(), axis=1)
                print(f'Pri Labels: {pri_labels.detach().cpu().numpy()}')
                print(f'Pub Labels: {sim_pub_labels.detach().cpu().numpy()}')
                print(f'Teach Pred: {teacher_pub_pred.detach().cpu().numpy()}')
                print(f'Stude Pred: {s_output}')
            total_loss += incre_loss
            teacher_total_accuracy += self.acc(teacher_pub_pred, sim_pub_labels)
            student_total_accuracy += self.get_acc(student_pri_pred, sim_pub_labels)
            label_correctness += self.acc(sim_pub_labels, pri_labels)
        avg_loss = total_loss / step
        student_avg_acc = student_total_accuracy / step
        teacher_avg_acc = teacher_total_accuracy / step
        label_avg_corr = label_correctness / step
        return avg_loss, student_avg_acc, teacher_avg_acc, label_avg_corr

    def sample_entropy(self, input):
        probabilities = np.exp(input.detach().cpu().numpy())
        entropies = entropy(probabilities.T)
        order = np.argsort(entropies)[::-1]
        return order

    def eval(self):
        print("\nEvaluating...")
        self.student.eval()
        total_loss, teacher_total_accuracy, student_total_accuracy, label_correctness = 0, 0, 0, 0
        
        step = 0
        pub_pool_ind = [random.sample(list(np.linspace(0, len(self.pub_toks)-1, len(self.pub_toks))), len(self.used_valid_data[0])*self.factor)]
        if self.att:
            pub_pool_tok, pub_pool_mask, pub_pool_label = self.pub_toks[pub_pool_ind], self.pub_masks[pub_pool_ind], self.pub_labels[pub_pool_ind]
        else:
            pub_pool_tok, pub_pool_label = self.pub_toks[pub_pool_ind], self.pub_labels[pub_pool_ind]
            pub_pool_mask = None
            
        if self.active:
            self.get_pub_data(pub_pool_tok, pub_pool_mask, pub_pool_label, len(self.used_valid_data[0]))
            #self.get_pub_data_actie(pub_pool_tok, pub_pool_mask, pub_pool_label, len(self.used_valid_data[0]))
        else:
            self.get_pub_data(pub_pool_tok, pub_pool_mask, pub_pool_label, len(self.used_valid_data[0]))
        
        for i in range(0, len(self.used_valid_data[0]), self.bs):
            
            if i+self.bs < len(self.used_valid_data[0]):
                last_ind = i+self.bs
            else:
                last_ind = -1
            
            if self.att:
                pri_tok = self.used_valid_data[0][i:last_ind]
                pri_mask = self.used_valid_data[1][i:last_ind]
                pri_labels = self.used_valid_data[2][i:last_ind]
                pub_tok = self.train_tok[i:last_ind]
                pub_mask = self.train_mask[i:last_ind]
                pub_label = self.train_label[i:last_ind]
            else:
                pri_tok = self.used_valid_data[0][i:last_ind]
                pri_labels = self.used_valid_data[1][i:last_ind]
                pub_tok = self.train_tok[i:last_ind]
                pub_label = self.train_label[i:last_ind]
                pri_mask, pub_mask = None, None
            
            step += 1
            with torch.no_grad():
                # get the data representation and predictions from the student model
                
                if self.sim is not None:
                    inds = torch.isin(pub_pool_label,self.target_classes)
                    pub_pool_mask = pub_pool_mask[inds] if self.att else None
                    loss, student_pri_pred, teacher_pub_pred, sim_pub_labels = self.get_loss_sim(pri_labels, pri_mask, pri_tok,
                                                                                             pub_pool_tok[inds], pub_pool_mask, pub_pool_label[inds])
                elif self.active:
                    loss, student_pri_pred, teacher_pub_pred, sim_pub_labels = self.get_loss_active(pri_labels, pri_mask, pri_tok,
                                                                                                pri_tok, pri_mask, pri_labels)
                else:
                    #print(self.used_valid_data_raw[0][i:last_ind])
                    #print(pri_tok)
                    mask = self.used_valid_data_raw[1][i:last_ind] if self.att else None
                    loss, student_pri_pred, teacher_pub_pred, sim_pub_labels = self.get_loss(self.used_valid_data_raw[1][i:last_ind], 
                                                                                         mask, self.used_valid_data_raw[0][i:last_ind],
                                                                                         pri_tok, pri_mask, pri_labels)
                loss = loss.item()
                total_loss += loss
                teacher_total_accuracy += self.acc(teacher_pub_pred, sim_pub_labels)
                student_total_accuracy += self.get_acc(student_pri_pred, sim_pub_labels)
                label_correctness += self.acc(sim_pub_labels, pri_labels)

                if step % 50 == 0 and not step == 0:
                    print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(self.used_valid_data[0])//self.bs))
                    s_output = np.argmax(student_pri_pred.detach().cpu().numpy(), axis=1)
                    #t_output = np.argmax(teacher_pub_pred.detach().cpu().numpy(), axis=1)
                    print(f'Pri Labels: {pri_labels.detach().cpu().numpy()}')
                    print(f'Pub Labels: {sim_pub_labels.detach().cpu().numpy()}')
                    print(f'Teach Pred: {teacher_pub_pred.detach().cpu().numpy()}')
                    print(f'Stude Pred: {s_output}')
        avg_loss = total_loss / step
        student_avg_acc = student_total_accuracy / step
        teacher_avg_acc = teacher_total_accuracy / step
        label_avg_corr = label_correctness / step
        return avg_loss, student_avg_acc, teacher_avg_acc, label_avg_corr
        
    def run(self):
        best_valid_loss = float('inf')
        train_losses, valid_losses = [], []
        student_train_accs, student_valid_accs = [], []
        teacher_train_accs, teacher_valid_accs = [], []
        valid_label, train_label = [], []
        print(f'\nStart fine-tuning with {len(self.test_data[0])} sensitive points.')
        for iter in range(self.iters):
            if (iter+1)*self.queries >= len(self.test_data[0]):
                self.used_sens_data = [self.test_data[0], self.test_data[1]]
                self.used_sens_data_raw = [self.test_data_raw[0], self.test_data_raw[1]]
                print('\nAlready have queried all sensitive train data.')
            else:
                self.used_sens_data = [self.test_data[0][:(iter+1)*self.queries], self.test_data[1][:(iter+1)*self.queries]]
                self.used_sens_data_raw = [self.test_data_raw[0][:(iter+1)*self.queries], self.test_data_raw[1][:(iter+1)*self.queries]]
            if (iter+1)*self.queries >= len(self.valid_loader[0]):
                self.used_valid_data = [self.valid_loader[0], self.valid_loader[1]]
                self.used_valid_data_raw = [self.valid_loader_raw[0], self.valid_loader_raw[1]]
                print('\nAlready have queried all sensitive valid data.')
            else:
                self.used_valid_data = [self.valid_loader[0][:(iter+1)*self.queries], self.valid_loader[1][:(iter+1)*self.queries]]
                self.used_valid_data_raw = [self.valid_loader_raw[0][:(iter+1)*self.queries], self.valid_loader_raw[1][:(iter+1)*self.queries]]
            print(f'\n Iter {iter+1} / {self.iters} with {len(self.used_sens_data[0])} points')
            for epoch in range(self.epochs):
                print(f'\nActive Learning: {self.active}')
                print('\n Epoch {:} / {:}'.format(epoch + 1, self.epochs))
                print(self.target_classes)

                train_loss, student_train_acc, teacher_train_acc, train_label_acc = self.train()
                valid_loss, student_valid_acc, teacher_valid_acc, valid_label_acc = self.eval()
                print(f'Losses: current is {valid_loss}, best is {best_valid_loss}')
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(self.student.state_dict(), f'{self.folder}/{self.weight_path}')
                
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                student_train_accs.append(student_train_acc)
                student_valid_accs.append(student_valid_acc)
                teacher_train_accs.append(teacher_train_acc)
                teacher_valid_accs.append(teacher_valid_acc)
                train_label.append(train_label_acc)
                valid_label.append(valid_label_acc)

                print(f'\nTraining Loss: {train_loss:.3f}')
                print(f'Validation Loss: {valid_loss:.3f}')
                print(f'\nSTUDENT Training Acc: {student_train_acc:.3f}')
                print(f'Validation Acc: {student_valid_acc:.3f}')
                print(f'\nTEACHER Training Acc: {teacher_train_acc:.3f}')
                print(f'Validation Acc: {teacher_valid_acc:.3f}')

        return train_losses, student_train_accs, valid_losses, student_valid_accs, teacher_train_accs, teacher_valid_accs, self.teacher_acc, train_label, valid_label
    