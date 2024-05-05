import torch
import pickle
import torch.nn as nn
import pandas as pd
import numpy as np
import random
from transformers import AdamW
from Framework.similarity import find_similar
from scipy.stats import entropy
from Framework.new_process_data import sanitize_data, tokenize, sanitize_data_dissim
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
        print(f'Differential Privacy: {self.dp}')

        # GET INDICES
        #list_ = np.linspace(0, len(test_data[0])-1, len(test_data[0]))
        # change here to determine public and private ratio 
        #pri_indices = np.array(random.sample(list(list_), int(len(test_data[0])/6)))
        #pub_indices = np.array(list(set(list_) - set(pri_indices)))
        #print(len(pri_indices))
        #print(len(pub_indices))
        #print(pri_indices)
        #print(pub_indices)
        if self.dp:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.valid_loader_raw = train_dataloader
            self.valid_loader = train_dataloader
            #tot_prob, replace_perc, self.valid_loader = sanitize_data(tokenizer, (train_dataloader[0].clone(), train_dataloader[1].clone()), self.sens, self.eps)
            #print(f'Total Probability is {tot_prob}')
            #print(f'Replaced percentage is {replace_perc}')
            if self.public:
                print('PUBLIC, shouldn"t be here')
                tot_prob, replace_perc, self.test_data = sanitize_data(tokenizer, pub_data, self.sens, self.eps)
            else:
                #self.remaining_test_tok_raw = torch.concat((test_data[0][pri_indices], pub_data[0][pub_indices]))
                #self.remaining_test_labels_raw = torch.concat((test_data[1][pri_indices], pub_data[1][pub_indices]))
                #self.test_data_raw = (self.remaining_test_tok_raw, self.remaining_test_labels_raw)

                #tot_prob, replace_perc, sanitize_test_data = sanitize_data(tokenizer, (test_data[0].clone(), test_data[1].clone()), self.sens, self.eps)
                #self.remaining_test_tok = torch.concat((sanitize_test_data[0][pri_indices], pub_data[0][pub_indices]))
                #self.remaining_test_labels = torch.concat((sanitize_test_data[1][pri_indices], pub_data[1][pub_indices]))
                #self.test_data = (self.remaining_test_tok, self.remaining_test_labels)
                #print(test_data[0][0])
                #print(sanitize_test_data[0][0])
                #tot_prob, replace_perc, new_test_data = sanitize_data(tokenizer, test_data, self.sens, self.eps)
                #self.test_data = (torch.concat((new_test_data[0][pri_indices], pub_data[0][pub_indices])), torch.concat((new_test_data[1][pri_indices], pub_data[1][pub_indices])))
                #print(len(self.test_data[0]))
            # previous code
            
            #else:
                print('PRIVATE')
                self.test_data_raw = test_data
                tot_prob, replace_perc, self.test_data = sanitize_data_dissim(tokenizer, (test_data[0].clone(), test_data[1].clone()), self.sens, self.eps)
            
            # new code
            print(f'Total Probability is {tot_prob}')
            print(f'Replaced percentage is {replace_perc}')
        else:
            if self.public:
                print('PUBLIC')
                self.test_data = pub_data
                self.test_data_raw = test_data
            else:
                print('PRIVATE, never happen')
                self.test_data = test_data
                
            self.valid_loader = valid_dataloader
            self.valid_loader_raw = valid_dataloader
        print('Teacher Acc')
        acc = model_performance(args, self.teacher, self.test_data[0], self.test_data[1], device, args['folder'], mask=None)
        # new code
        #self.test_data_raw = (torch.concat((test_data[0][pri_indices], pub_data[0][pub_indices])), torch.concat((test_data[1][pri_indices], pub_data[1][pub_indices])))if not self.public else pub_data
        #arr_test = self.test_data_raw[0] == self.test_data[0]
        #arr_valid = self.valid_loader_raw[0] == self.valid_loader[0]
        #print('Is all test the same?: ', arr_test.all()==True)
        #print('Is all valid the same?: ', arr_valid.all()==True)
        # previous code
        #self.test_data_raw = test_data if not self.public else pub_data
        
        self.remaining_test_tok = self.test_data[0]
        self.remaining_test_labels = self.test_data[1]
        self.teacher_acc = acc

        self.remaining_test_tok_raw = self.test_data_raw[0].clone()
        self.remaining_test_labels_raw = self.test_data_raw[1].clone()

        self.weight = train_weight
        self.optimizer = AdamW(self.student.parameters(), lr=self.lr, weight_decay=self.wd)
        self.loss = nn.NLLLoss(self.weight) 
        self.target_classes = torch.unique(self.test_data[1])
        self.get_pre_train_data(args['pre_train_file'])

    def get_pre_train_data(self, pre_train_file):
        if pre_train_file is not None:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            train = pd.read_csv(pre_train_file)
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

        teacher_pred = self.teacher(te_tok)[-1]
        teacher_pred_argmax = teacher_pred.argmax(dim=1)
        loss = self.loss(student_pri_pred, teacher_pred_argmax)
        return loss, student_pri_pred, teacher_pred_argmax, te_labels
    
    def get_loss_sim(self, st_tok, te_labels, te_tok):
        student_pri_pred = self.student(te_tok)[-1]
        with torch.no_grad():
            student_pub_pred = self.student(te_tok)[-1]

        sim_te_index = find_similar(student_pri_pred, student_pub_pred, 'cosine')
        new_te_tok, new_te_labels = te_tok[sim_te_index], te_labels[sim_te_index]

        teacher_pred = self.teacher(new_te_tok)[-1]
       
        teacher_pred_argmax = teacher_pred.argmax(dim=1)
        loss = self.loss(student_pri_pred, teacher_pred_argmax)

        return loss, student_pri_pred, teacher_pred_argmax, new_te_labels

    def train(self):
        self.student.train()
        total_loss, teacher_total_accuracy, student_avg_noisy_acc, student_avg_raw_acc, label_correctness = 0, 0, 0, 0, 0
        student_train_noise_accuracy, student_train_raw_accuracy = 0, 0
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
            #if self.sim:
            #    loss, student_pri_pred, teacher_pub_pred, sim_pub_labels = self.get_loss_sim(te_tok, te_labels, te_tok)  
            #else:
            loss, student_pri_pred, teacher_pub_pred, sim_pub_labels = self.get_loss(te_tok, te_labels, te_tok)                                                                          
            incre_loss = loss.item()
            loss.backward()
            self.optimizer.step()
            step += 1

            # print outputs
            if step % 200 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(self.used_sens_data[0])//self.bs))
                student_pri_pred_raw = self.student(st_tok)[-1]
                s_output = np.argmax(student_pri_pred.detach().cpu().numpy(), axis=1)
                #t_output = np.argmax(teacher_pub_pred.detach().cpu().numpy(), axis=1)
                print(f'Pri Labels: {st_labels.detach().cpu().numpy()}')
                print(f'Pub Labels: {sim_pub_labels.detach().cpu().numpy()}')
                print(f'Teach Pred: {teacher_pub_pred.detach().cpu().numpy()}')
                print(f'Stude Pred: {s_output}')
            with torch.no_grad():
                student_pri_pred_raw = self.student(st_tok)[-1]
            total_loss += incre_loss
            teacher_total_accuracy += self.acc(teacher_pub_pred, sim_pub_labels)
            student_train_noise_accuracy += self.get_acc(student_pri_pred, sim_pub_labels)
            student_train_raw_accuracy += self.get_acc(student_pri_pred_raw, sim_pub_labels)
            label_correctness += self.acc(sim_pub_labels, st_labels)
        avg_loss = total_loss / step
        student_avg_noisy_acc = student_train_noise_accuracy / step
        student_avg_raw_acc = student_train_raw_accuracy / step
        teacher_avg_acc = teacher_total_accuracy / step
        label_avg_corr = label_correctness / step
        return avg_loss, student_avg_noisy_acc, student_avg_raw_acc, teacher_avg_acc, label_avg_corr

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
                if self.sim:
                    loss, student_pri_pred, teacher_pub_pred, sim_pub_labels = self.get_loss_sim(st_tok, te_labels, te_tok)  
                else:
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
        student_train_accs_raw = []
        valid_label, train_label = [], []
        self.used_sens_data = None
        print(f'\nStart fine-tuning with {len(self.remaining_test_tok)} sensitive points.')
        for iter in range(self.iters):
            print((iter+1)*self.queries, len(self.remaining_test_tok))
            if (iter+1)*self.queries >= len(self.test_data[0]):
                #self.used_sens_data = self.used_sens_data 
                #self.used_sens_data_raw = self.used_sens_data_raw
                print('\nAlready have queried all sensitive train data.')
            else:
                # active learning for private labels
                student_pred = self.student(self.remaining_test_tok_raw)[-1]
                
                print(f'My training set size: {self.remaining_test_tok_raw.shape}')
                # uncertainty sampling on raw data
                if self.active:
                    inds = np.array(self.sample_entropy(student_pred))
                    mask = np.zeros(inds.shape,dtype=bool)
                    inds = inds[:self.queries]
                else:
                    inds = random.sample(range(len(student_pred)), self.queries)
                mask = np.zeros(len(student_pred), dtype=bool)
                #print(inds)
                mask[inds] = True
                if self.sim:
                    inds = np.linspace(0, self.queries-1, self.queries).astype(int)
                    mask = np.zeros(len(student_pred), dtype=bool)
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
                #arr_remain = self.remaining_test_tok == self.remaining_test_tok_raw
                #arr_selected = self.used_sens_data[0] == self.used_sens_data_raw[0]
                #print('Selected points are similar?', arr_selected.all() == True)
                #print('Remaining points are similar?', arr_remain.all() == True)
                
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

                train_loss, student_train_acc, student_avg_acc_raw, teacher_train_acc, train_label_acc = self.train()
                valid_loss, student_valid_acc, teacher_valid_acc, valid_label_acc = self.eval()
                print(f'Losses: current is {valid_loss}, best is {best_valid_loss}')
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(self.student.state_dict(), f'{self.folder}/{self.weight_path}')
                
                train_losses.append(train_loss)
                valid_losses.append(valid_loss)
                student_train_accs.append(student_train_acc)
                student_train_accs_raw.append(student_avg_acc_raw)
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
            'student_noisy_train_accs': student_train_accs, 
            'student_raw_train_accs': student_train_accs_raw, 
            'valid_losses': valid_losses, 
            'student_valid_accs': student_valid_accs, 
            'teacher_train_accs': teacher_train_accs, 
            'teacher_valid_accs': teacher_valid_accs,
            'train_label': train_label, 
            'valid_label': valid_label}

            f = open(f"{self.folder}/train_data.pkl","wb")
            pickle.dump(dic,f)
            f.close()

        return train_losses, student_train_accs, student_train_accs_raw, valid_losses, student_valid_accs, teacher_train_accs, teacher_valid_accs, self.teacher_acc, train_label, valid_label
    