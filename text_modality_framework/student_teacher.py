
'''
Student teacher module
'''
import pickle
import random
import torch
import pandas as pd
import numpy as np

from torch import nn
from scipy.stats import entropy
from transformers import AdamW, BertTokenizer
from torch.utils.data import RandomSampler

from text_modality_framework.similarity import find_similar
from text_modality_framework.process_data import sanitize_data, tokenize, sanitize_data_dissim
from text_modality_framework.classifier import model_performance

class StudentTeacher(nn.Module):
    '''define student teacher model'''
    def __init__(self, teacher, student, args, device, map_,
                 sim, active, train_dataloader, valid_dataloader, test_data, train_weight):
        super(StudentTeacher, self).__init__()
        self.teacher = teacher
        self.student = student
        self.device = device
        self.map = map_
        self.sim = sim
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

        pub_data = [r.to(self.device) for r in train_dataloader]
        self.pub_toks, self.pub_labels = pub_data
        print(f'Differential Privacy: {self.dp}')
        if self.dp:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            self.valid_loader_raw = train_dataloader
            self.valid_loader = train_dataloader
            if self.public:
                print('PUBLIC, shouldn"t be here')
                tot_prob, replace_perc, self.test_data = sanitize_data(tokenizer, pub_data,
                                                                       self.sens, self.eps)
                #tot_prob, replace_perc, self.test_data = sanitize_data_dissim(tokenizer,
                # pub_data, self.sens, self.eps)
            else:
                print('PRIVATE')
                self.test_data_raw = test_data
                tot_prob, replace_perc, self.test_data = sanitize_data(tokenizer,
                                                                       (test_data[0].clone(),
                                                                        test_data[1].clone()),
                                                                        self.sens, self.eps)
                #tot_prob, replace_perc, self.test_data = sanitize_data_dissim(tokenizer,
                # (test_data[0].clone(), test_data[1].clone()), self.sens, self.eps)
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
        acc = model_performance(args, self.teacher, self.test_data[0], self.test_data[1],
                                device, args['folder'])

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
        '''retrieve the pre-training data for the student model'''
        if pre_train_file is not None:
            tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
            train = pd.read_csv(pre_train_file)
            train = (train['data'], train['label'])
            train_dataloader = tokenize(tokenizer, train, 50, self.bs, 'train',
                                        RandomSampler, False, False)[0]
            self.pre_train = train_dataloader
            print(len(self.pre_train[0]))
        else:
            self.pre_train = [torch.tensor([]).long(), torch.tensor([]).long()]

    def get_acc(self, preds, labels):
        '''get accuracy'''
        preds = preds.argmax(dim=1)
        return self.acc(preds, labels)

    def acc(self, a, b):
        '''get accuracy for predictions b and labels a'''
        return np.sum([1 if a[i] == b[i] else 0 for i in range(len(a))])/len(a)

    def get_loss(self, st_tok, te_labels, te_tok):
        '''get the loss for the student teacher'''
        student_pri_pred = self.student(st_tok)[-1]
        teacher_pred = self.teacher(te_tok)[-1]
        teacher_pred_argmax = teacher_pred.argmax(dim=1)
        loss = self.loss(student_pri_pred, teacher_pred_argmax)
        return loss, student_pri_pred, teacher_pred_argmax, te_labels

    def get_loss_sim(self, st_tok, te_labels, te_tok):
        '''get the loss for the student teacher using similarity scores'''
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
        '''fine-tune the student model with the teacher'''
        self.student.train()
        total_loss, t_total_acc, label_correctness = 0, 0, 0
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
            loss, s_pri_pred, t_pub_pred, sim_pub_labels = self.get_loss(te_tok,
                                                                         te_labels,
                                                                         te_tok)
            incre_loss = loss.item()
            loss.backward()
            self.optimizer.step()
            step += 1
            # print outputs
            if step % 200 == 0 and step != 0:
                print(f'Batch {step} of {len(self.used_sens_data[0])//self.bs}.')
                student_pri_pred_raw = self.student(st_tok)[-1]
                s_output = np.argmax(s_pri_pred.detach().cpu().numpy(), axis=1)
                #t_output = np.argmax(teacher_pub_pred.detach().cpu().numpy(), axis=1)
                print(f'Pri Labels: {st_labels.detach().cpu().numpy()}')
                print(f'Pub Labels: {sim_pub_labels.detach().cpu().numpy()}')
                print(f'Teach Pred: {t_pub_pred.detach().cpu().numpy()}')
                print(f'Stude Pred: {s_output}')
            with torch.no_grad():
                student_pri_pred_raw = self.student(st_tok)[-1]
            total_loss += incre_loss
            t_total_acc += self.acc(t_pub_pred, sim_pub_labels)
            student_train_noise_accuracy += self.get_acc(s_pri_pred, sim_pub_labels)
            student_train_raw_accuracy += self.get_acc(student_pri_pred_raw, sim_pub_labels)
            label_correctness += self.acc(sim_pub_labels, st_labels)
        avg_loss = total_loss / step
        s_avg_noisy_acc = student_train_noise_accuracy / step
        s_avg_raw_acc = student_train_raw_accuracy / step
        teacher_avg_acc = t_total_acc / step
        label_avg_corr = label_correctness / step
        return avg_loss, s_avg_noisy_acc, s_avg_raw_acc, teacher_avg_acc, label_avg_corr

    def sample_entropy(self, input):
        '''use entropy sampling to select the most uncertain points'''
        probabilities = np.exp(input.detach().cpu().numpy())
        entropies = entropy(probabilities.T)
        order = np.argsort(entropies)[::-1]
        return order

    def eval(self):
        '''evaluate the student fine-tuning using the validation set'''
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
                    loss, s_pri_pred, t_pub_pred, sim_pub_labels = self.get_loss_sim(st_tok,
                                                                                     te_labels,
                                                                                     te_tok)
                else:
                    loss, s_pri_pred, t_pub_pred, sim_pub_labels = self.get_loss(st_tok,
                                                                                 te_labels,
                                                                                 te_tok)
                loss = loss.item()
                total_loss += loss
                teacher_total_accuracy += self.acc(t_pub_pred, sim_pub_labels)
                student_total_accuracy += self.get_acc(s_pri_pred, sim_pub_labels)
                label_correctness += self.acc(sim_pub_labels, st_labels)

                if step % 50 == 0 and not step == 0:
                    print(f'Batch {step} of {len(self.used_valid_data[0])//self.bs}.')
                    s_output = np.argmax(s_pri_pred.detach().cpu().numpy(), axis=1)
                    #t_output = np.argmax(teacher_pub_pred.detach().cpu().numpy(), axis=1)
                    print(f'Pri Labels: {st_labels.detach().cpu().numpy()}')
                    print(f'Pub Labels: {sim_pub_labels.detach().cpu().numpy()}')
                    print(f'Teach Pred: {t_pub_pred.detach().cpu().numpy()}')
                    print(f'Stude Pred: {s_output}')
        avg_loss = total_loss / step
        student_avg_acc = student_total_accuracy / step
        teacher_avg_acc = teacher_total_accuracy / step
        label_avg_corr = label_correctness / step
        return avg_loss, student_avg_acc, teacher_avg_acc, label_avg_corr

    def run(self):
        '''fine tune and evaluate the student teacher framework'''
        best_valid_loss = float('inf')
        t_loss, v_loss = [], []
        s_t_accs, s_v_accs = [], []
        t_t_accs, t_v_accs = [], []
        s_t_accs_raw = []
        v_label, t_label = [], []
        self.used_sens_data = None
        print(f'\nStart fine-tuning with {len(self.remaining_test_tok)} sensitive points.')
        for iters in range(self.iters):
            print((iters+1)*self.queries, len(self.remaining_test_tok))
            if (iters+1)*self.queries >= len(self.test_data[0]):
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
                mask[inds] = True
                if self.sim:
                    inds = np.linspace(0, self.queries-1, self.queries).astype(int)
                    mask = np.zeros(len(student_pred), dtype=bool)
                    mask[inds] = True

                if self.used_sens_data is None:
                    self.used_sens_data = [torch.concat((self.pre_train[0],
                                                         self.remaining_test_tok[inds])),
                                             torch.concat((self.pre_train[1],
                                                           self.remaining_test_labels[inds]))]
                    self.used_sens_data_raw = [torch.concat((self.pre_train[0],
                                                             self.remaining_test_tok_raw[inds])),
                                               torch.concat((self.pre_train[1],
                                                             self.remaining_test_labels_raw[inds]))]
                else:
                    self.used_sens_data = [torch.concat((self.used_sens_data[0],
                                                         self.remaining_test_tok[inds])),
                                             torch.concat((self.used_sens_data[1],
                                                           self.remaining_test_labels[inds]))]
                    self.used_sens_data_raw = [torch.concat((self.used_sens_data_raw[0],
                                                             self.remaining_test_tok_raw[inds])),
                                               torch.concat((self.used_sens_data_raw[1],
                                                             self.remaining_test_labels_raw[inds]))]
                print(f'Originally {len(self.remaining_test_labels_raw)} points')
                self.remaining_test_tok = self.remaining_test_tok[~mask]
                self.remaining_test_labels = self.remaining_test_labels[~mask]
                self.remaining_test_tok_raw = self.remaining_test_tok_raw[~mask]
                self.remaining_test_labels_raw = self.remaining_test_labels_raw[~mask]
                print(f'Selected {len(inds)}, remaining {len(self.remaining_test_labels_raw)}')
                print(f'Training data has {len(self.used_sens_data[0])} points.')

            if (iters+1)*self.queries >= len(self.valid_loader[0]):
                self.used_valid_data = [self.valid_loader[0], self.valid_loader[1]]
                self.used_valid_data_raw = [self.valid_loader_raw[0], self.valid_loader_raw[1]]
                print('\nAlready have queried all sensitive valid data.')
            else:
                self.used_valid_data = [self.valid_loader[0][:(iters+1)*self.queries],
                                        self.valid_loader[1][:(iters+1)*self.queries]]
                self.used_valid_data_raw = [self.valid_loader_raw[0][:(iters+1)*self.queries],
                                            self.valid_loader_raw[1][:(iters+1)*self.queries]]
            print(f'\n Iter {iters+1} / {self.iters} with {len(self.used_sens_data[0])} points')

            for epoch in range(self.epochs):
                print(f'\nActive Learning: {self.active}')
                print(f'\n Epoch {epoch + 1} / {self.epochs}')
                print(self.target_classes)

                train_loss, s_t_acc, s_avg_acc_raw, t_train_acc, train_label_acc = self.train()
                valid_loss, s_v_acc, t_v_acc, valid_label_acc = self.eval()
                print(f'Losses: current is {valid_loss}, best is {best_valid_loss}')
                if valid_loss < best_valid_loss:
                    best_valid_loss = valid_loss
                    torch.save(self.student.state_dict(), f'{self.folder}/{self.weight_path}')

                t_loss.append(train_loss)
                v_loss.append(valid_loss)
                s_t_accs.append(s_t_acc)
                s_t_accs_raw.append(s_avg_acc_raw)
                s_v_accs.append(s_v_acc)
                t_t_accs.append(t_train_acc)
                t_v_accs.append(t_v_acc)
                t_label.append(train_label_acc)
                v_label.append(valid_label_acc)

                print(f'\nTraining Loss: {train_loss:.3f}')
                print(f'Validation Loss: {valid_loss:.3f}')
                print(f'\nSTUDENT Training Acc: {s_t_acc:.3f}')
                print(f'Validation Acc: {s_v_acc:.3f}')
                print(f'\nTEACHER Training Acc: {t_train_acc:.3f}')
                print(f'Validation Acc: {t_v_acc:.3f}')

            dic = {'train_losses': t_loss,
            'student_noisy_train_accs': s_t_accs,
            'student_raw_train_accs': s_t_accs_raw,
            'valid_losses': v_loss,
            'student_valid_accs': s_v_accs,
            'teacher_train_accs': t_t_accs,
            'teacher_valid_accs': t_v_accs,
            'train_label': t_label,
            'valid_label': v_label}

            f = open(f"{self.folder}/train_data.pkl","wb")
            pickle.dump(dic,f)
            f.close()

        return t_loss, s_t_accs, s_t_accs_raw, v_loss, s_v_accs, t_t_accs, t_v_accs, self.teacher_acc, t_label, v_label
