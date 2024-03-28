import torch
import pickle
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from transformers import AdamW
#from Framework.similarity import find_similar
from scipy.stats import entropy
from Framework.new_process_data import sanitize_data, tokenize
from transformers import BertTokenizer
from Framework.classifier import model_performance
from torch.utils.data import RandomSampler

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
        self.public = args['public']
        self.train_tok = None
        self.train_label = None
        self.train_mask = None
        
        pub_data = [r.to(self.device) for r in train_dataloader]
        self.pub_toks, self.pub_labels = pub_data
        self.valid_loader = valid_dataloader
        print(f'Differential Privacy: {self.dp}')

        # GET INDICES
        list_ = np.linspace(0, len(test_data[0])-1, len(test_data[0]))
        pri_indices = np.array(random.sample(list(list_), int(3*len(test_data[0])/4)))
        pub_indices = np.array(list(set(list_) - set(pri_indices)))
        print(len(pri_indices))
        print(len(pub_indices))
        if self.dp:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            if self.public:
                print('PUBLIC, shouldn"t be here')
                tot_prob, replace_perc, self.test_data = sanitize_data(tokenizer, pub_data, self.sens, self.eps)
            else:
                print('PRIVATE')
                #new_test_data = (torch.concat((test_data[0][pri_indices], pub_data[0][pub_indices])), torch.concat((test_data[1][pri_indices], pub_data[1][pub_indices])))
                #print(len(new_test_data[0]))
                tot_prob, replace_perc, new_test_data = sanitize_data(tokenizer, test_data, self.sens, self.eps)
                self.test_data = (torch.concat((new_test_data[0][pri_indices], pub_data[0][pub_indices])), torch.concat((new_test_data[1][pri_indices], pub_data[1][pub_indices])))
                print(len(self.test_data[0]))
            # previous code
            
            #else:
            #    print('PRIVATE')
            #    tot_prob, replace_perc, self.test_data = sanitize_data(tokenizer, test_data, self.sens, self.eps)
            
            # new code
            
            print(f'Total Probability is {tot_prob}')
            print(f'Replaced percentage is {replace_perc}')
            tot_prob, replace_perc, self.valid_loader = sanitize_data(tokenizer, valid_dataloader, self.sens, self.eps)
            print(f'Total Probability is {tot_prob}')
            print(f'Replaced percentage is {replace_perc}')
        else:
            if self.public:
                print('PUBLIC')
                self.test_data = pub_data
            else:
                print('PRIVATE, never happen')
                self.test_data = test_data
            self.valid_loader = valid_dataloader
        print('Teacher Acc')
        acc = model_performance(args, self.teacher, self.test_data[0], self.test_data[1], device, args['folder'], mask=None)
        # new code
        self.test_data_raw = (torch.concat((test_data[0][pri_indices], pub_data[0][pub_indices])), torch.concat((test_data[1][pri_indices], pub_data[1][pub_indices])))if not self.public else pub_data

        # previous code
        #self.test_data_raw = test_data if not self.public else pub_data
        
        self.remaining_test_tok = self.test_data[0].clone()
        self.remaining_test_labels = self.test_data[1].clone()
        self.current_test_tok = None
        self.current_test_labels = None
        self.teacher_acc = acc

        self.remaining_test_tok_raw = self.test_data_raw[0].clone()
        self.remaining_test_labels_raw = self.test_data_raw[1].clone()
        self.current_test_tok_raw = None
        self.current_test_labels_raw = None

        self.valid_loader_raw = valid_dataloader
        self.weight = train_weight
        self.optimizer = AdamW(self.student.parameters(), lr=self.lr, weight_decay=self.wd)
        self.loss = nn.NLLLoss(self.weight) 
        self.target_classes = torch.unique(self.test_data[1])
        self.get_pre_train_data(args['pre_train_file'])

    def get_pre_train_data(self, pre_train_file):
        if pre_train_file is not None:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            train = pd.read_csv(pre_train_file)
            #train = train[train['label'].isin([0, 1, 2])].reset_index(drop=True)
            train = (train['data'], train['label'])
            train_dataloader, train_weight = tokenize(tokenizer, train, 50, self.bs, 'train', RandomSampler, False, False)
            self.pre_train = train_dataloader
            print(len(self.pre_train[0]))
        else:
            self.pre_train = [torch.tensor([]).long(), torch.tensor([]).long()]
    
    def get_acc(self, preds, labels):
        preds = preds.argmax(dim=1)
        return self.acc(preds, labels)

    def acc(self, a, b):
        return np.sum([1 if a[i] == b[i] else 0 for i in range(len(a))])/len(a)

    def get_loss(self, st_tok, te_labels, te_tok):
        student_pri_pred = self.student(st_tok)[-1]
        #print('\n', st_tok == te_tok)
        #print(st_tok)
        #print(te_tok)
        teacher_pred = self.teacher(te_tok)[-1]
        teacher_pred_argmax = teacher_pred.argmax(dim=1)
        loss = self.loss(student_pri_pred, teacher_pred_argmax)
        return loss, student_pri_pred, teacher_pred_argmax, te_labels

    def train(self):
        self.student.train()
        total_loss, teacher_total_accuracy, student_total_accuracy, label_correctness = 0, 0, 0, 0

        step = 0
        for i in range(0, len(self.used_sens_data[0]), self.bs):
            if i+self.bs < len(self.used_sens_data[0]):
                last_ind = i+self.bs
            else:
                last_ind = -1
            st_tok = self.used_sens_data_raw[0][i:last_ind]
            st_labels = self.used_sens_data_raw[1][i:last_ind]
            te_tok = self.used_sens_data[0][i:last_ind]
            te_labels = self.used_sens_data[1][i:last_ind]
            
            self.student.zero_grad()
            loss, student_pri_pred, teacher_pub_pred, sim_pub_labels = self.get_loss(st_tok, te_labels, te_tok)                                                                            
            incre_loss = loss.item()
            loss.backward()
            self.optimizer.step()
            step += 1

            # print outputs
            if step % 200 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(self.used_sens_data[0])//self.bs))
                s_output = np.argmax(student_pri_pred.detach().cpu().numpy(), axis=1)
                #t_output = np.argmax(teacher_pub_pred.detach().cpu().numpy(), axis=1)
                print(f'Pri Labels: {st_labels.detach().cpu().numpy()}')
                print(f'Pub Labels: {sim_pub_labels.detach().cpu().numpy()}')
                print(f'Teach Pred: {teacher_pub_pred.detach().cpu().numpy()}')
                print(f'Stude Pred: {s_output}')
            total_loss += incre_loss
            teacher_total_accuracy += self.acc(teacher_pub_pred, sim_pub_labels)
            student_total_accuracy += self.get_acc(student_pri_pred, sim_pub_labels)
            label_correctness += self.acc(sim_pub_labels, st_labels)
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
        
        for i in range(0, len(self.used_valid_data[0]), self.bs):
            
            if i+self.bs < len(self.used_valid_data[0]):
                last_ind = i+self.bs
            else:
                last_ind = -1
            
            st_tok = self.used_valid_data_raw[0][i:last_ind]
            st_labels = self.used_valid_data_raw[1][i:last_ind]
            te_tok = self.used_valid_data[0][i:last_ind]
            te_labels = self.used_valid_data[1][i:last_ind]
        
            step += 1
            with torch.no_grad():
                # get the data representation and predictions from the student model
                loss, student_pri_pred, teacher_pub_pred, sim_pub_labels = self.get_loss(st_tok, te_labels, te_tok)   
                loss = loss.item()
                total_loss += loss
                teacher_total_accuracy += self.acc(teacher_pub_pred, sim_pub_labels)
                student_total_accuracy += self.get_acc(student_pri_pred, sim_pub_labels)
                label_correctness += self.acc(sim_pub_labels, st_labels)

                if step % 50 == 0 and not step == 0:
                    print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(self.used_valid_data[0])//self.bs))
                    s_output = np.argmax(student_pri_pred.detach().cpu().numpy(), axis=1)
                    #t_output = np.argmax(teacher_pub_pred.detach().cpu().numpy(), axis=1)
                    print(f'Pri Labels: {st_labels.detach().cpu().numpy()}')
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
        self.used_sens_data = None
        print(f'\nStart fine-tuning with {len(self.test_data[0])} sensitive points.')
        for iter in range(self.iters):
            if (iter+1)*self.queries >= len(self.test_data[0]):
                
                self.used_sens_data = self.used_sens_data 
                self.used_sens_data_raw = self.used_sens_data_raw
                print('\nAlready have queried all sensitive train data.')
            else:
                # active learning for private labels
                student_pred = self.student(self.remaining_test_tok_raw)[-1]
                print(f'My training set size: {self.remaining_test_tok_raw.shape}')
                inds = np.array(self.sample_entropy(student_pred))
                mask = np.zeros(inds.shape,dtype=bool)
                inds = inds[:self.queries]
                mask[inds] = True
                
                #print(self.remaining_test_tok[inds])
                if self.used_sens_data is None:
                    self.used_sens_data = [torch.concat((self.pre_train[0], self.remaining_test_tok[inds])), 
                                             torch.concat((self.pre_train[1], self.remaining_test_labels[inds]))]
                    self.used_sens_data_raw = [torch.concat((self.pre_train[0], self.remaining_test_tok_raw[inds])),
                                               torch.concat((self.pre_train[1], self.remaining_test_labels_raw[inds]))]
                else:
                    self.used_sens_data = [torch.concat((self.used_sens_data[0], self.remaining_test_tok[inds])),
                                             torch.concat((self.used_sens_data[1], self.remaining_test_labels[inds]))]
                    self.used_sens_data_raw = [torch.concat((self.used_sens_data_raw[0], self.remaining_test_tok_raw[inds])),
                                               torch.concat((self.used_sens_data_raw[1], self.remaining_test_labels_raw[inds]))]
                print(f'Originally {len(self.remaining_test_labels_raw)} points')
                self.remaining_test_tok = self.remaining_test_tok[~mask]
                self.remaining_test_labels = self.remaining_test_labels[~mask]
                self.remaining_test_tok_raw = self.remaining_test_tok_raw[~mask]
                self.remaining_test_labels_raw = self.remaining_test_labels_raw[~mask]
                print(f'Selected {len(inds)} points, remaining {len(self.remaining_test_labels_raw)} points')
                print(f'Training data has {len(self.used_sens_data[0])} points.')
                
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
        
            dic = {'train_losses': train_losses, 
            'student_train_accs': student_train_accs, 
            'valid_losses': valid_losses, 
            'student_valid_accs': student_valid_accs, 
            'teacher_train_accs': self.teacher_acc, 
            'teacher_valid_accs': teacher_valid_accs,
            'train_label': train_label, 
            'valid_label': valid_label}

            f = open(f"{self.folder}/train_data.pkl","wb")
            pickle.dump(dic,f)
            f.close()

        return train_losses, student_train_accs, valid_losses, student_valid_accs, teacher_train_accs, teacher_valid_accs, self.teacher_acc, train_label, valid_label
    