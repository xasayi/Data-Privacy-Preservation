'''
Classifier class 
'''
import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import Subset

import pandas as pd
import numpy as np
from sklearn.metrics import classification_report
from scipy.stats import entropy

class Classifier(nn.Module):
    '''
    Classifier 
    '''
    def __init__(self, model,train_dataloader, device, lr, batch_size, valid_dataloader,
                 epochs, test_data, weights, folder, weight_path, active,
                 weight_decay=0.0005):
        super(Classifier, self).__init__()

        self.model = model
        self.optimizer = AdamW(self.model.parameters(), lr = lr, weight_decay=weight_decay)
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.test_data = test_data
        self.loss = nn.NLLLoss(weights)

        self.active = active
        self.epochs = epochs
        self.batch_size = batch_size

        self.weight_path = weight_path
        self.folder = folder
        self.device = device
        if self.active:
            # None if not using active learning 
            self.inds = [np.linspace(0, len(i[0])-1, len(i[0])) for i in self.train_dataloader]
        else:
            self.inds = None

    def get_loss(self, sent_id, labels, train=True):
        preds = self.model(sent_id)[-1]
        loss = self.loss(preds, labels)
        total_loss = loss.item()
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        return total_loss, preds

    def get_loss_mask(self, sent_id, labels, mask, train=True):
        preds = self.model(sent_id, mask)
        loss = self.loss(preds, labels)
        total_loss = loss.item()
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        return total_loss, preds

    def get_acc(self, preds, labels):
        preds = preds.argmax(dim=1)
        accuracy = np.sum([1 if preds[i]==labels[i] else 0 for i in range(len(labels))])/len(labels)
        return accuracy

    def train(self):
        self.model.train()
        total_loss, total_accuracy = 0, 0
        for step, batch in enumerate(self.train_dataloader):

            batch = [r.to(self.device) for r in batch]
            sent_id, labels = batch
            self.model.zero_grad()
            loss, preds = self.get_loss(sent_id, labels)
            if step % 1000 == 0 and not step == 0:
                print(f'Batch {step} of {len(self.train_dataloader)}')
                output = preds.argmax(dim=1)
                print(f'Pred: {list(output.detach().cpu().numpy())}')
                print(f'Targ: {list(labels.detach().cpu().numpy())}')

            total_loss += loss
            total_accuracy += self.get_acc(preds, labels)
        avg_loss = total_loss / len(self.train_dataloader)
        avg_acc = total_accuracy / len(self.train_dataloader)
        return avg_loss, avg_acc

    def sample_entropy(self, input_):
        probabilities = np.exp(input_.detach().cpu().numpy())
        entropies = entropy(probabilities.T)
        order = np.argsort(entropies)[::-1]
        return order

    def train_active(self):
        '''Change so it has active learning
        select the worse performing samples from previous 
        iteration to train on instead of the whole set'''
        self.model.train()
        total_loss, total_accuracy = 0, 0

        track_acc, inds = [], []
        for step, batch in enumerate(self.train_dataloader):
            order = np.array(self.inds[step])
            batch = [r.to(self.device) for r in batch]
            sent_id, labels = batch
            if len(order) == len(batch[0]):
                sent_id = sent_id[order]
                labels = labels[order]
            self.model.zero_grad()
            loss, preds = self.get_loss(sent_id, labels)
            ind = self.sample_entropy(preds)
            if step % 1000 == 0 and not step == 0:
                print(f'Batch {step} of {len(self.train_dataloader)}')
                output = preds.argmax(dim=1)
                print(f'Pred: {list(output.detach().cpu().numpy())}')
                print(f'Targ: {list(labels.detach().cpu().numpy())}')

            accuracy = self.get_acc(preds, labels)
            total_loss += loss
            total_accuracy += accuracy
            track_acc.append(accuracy)
            inds.append(ind)
        self.inds = inds

        indices = np.argsort(track_acc) # from low to high accuracy
        self.train_dataloader = Subset([i for i in self.train_dataloader], indices=indices)
        avg_loss = total_loss / len(self.train_dataloader)
        avg_acc = total_accuracy / len(self.train_dataloader)
        return avg_loss, avg_acc

    def eval(self):
        print("\nEvaluating...")
        self.model.eval()
        total_loss, total_accuracy = 0, 0

        for step,batch in enumerate(self.valid_dataloader):
            batch = [t.to(self.device) for t in batch]
            sent_id, labels = batch
            with torch.no_grad():
                loss, preds = self.get_loss(sent_id, labels, train=False)
                total_loss += loss
                total_accuracy += self.get_acc(preds, labels)
                if step % 1000 == 0 and not step == 0:
                    print(f'Batch {step} of {len(self.valid_dataloader)}')
                    output = preds.argmax(dim=1)
                    print(f'Pred: {list(output.detach().cpu().numpy())}')
                    print(f'Targ: {list(labels.detach().cpu().numpy())}')
        avg_loss = total_loss / len(self.valid_dataloader)
        avg_acc = total_accuracy / len(self.valid_dataloader)
        return avg_loss, avg_acc

    def run(self):
        best_valid_loss = float('inf')
        train_losses, valid_losses = [], []
        train_accs, valid_accs = [], []

        for epoch in range(self.epochs):
            print(f'\n Epoch {epoch + 1} / {self.epochs}')
            if not self.active:
                train_loss, train_acc = self.train()
            else:
                print(f'active: {self.active}')
                train_loss, train_acc = self.train_active()
            valid_loss, valid_acc = self.eval()

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), f'{self.folder}/{self.weight_path}')

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)
            train_accs.append(train_acc)
            valid_accs.append(valid_acc)

            print(f'\nTraining Loss: {train_loss:.3f}')
            print(f'Validation Loss: {valid_loss:.3f}')
            print(f'\nTraining Acc: {train_acc:.3f}')
            print(f'Validation Acc: {valid_acc:.3f}')
        return train_losses, train_accs, valid_losses, valid_accs

def model_performance(args, model, test_seq, test_y, device, folder, mask=None):
    with torch.no_grad():
        if mask:
            preds = model(torch.tensor(test_seq).to(device), torch.tensor(mask).to(device))
        else:
            preds = model(torch.tensor(test_seq).to(device))[-1]
    preds = preds.argmax(dim=1).detach().cpu().numpy()
    with open(f'{folder}/results.txt', 'w') as file:
        report = classification_report(test_y, preds)
        confusion_matrix = pd.crosstab(test_y, preds)
        file.write(report)
        file.write(str(confusion_matrix))
        file.write(f'\ndropout: {args["dropout"]}\n')
        file.write(f'lr: {args["lr"]}\n')
        file.write(f'batch_size: {args["batch_size"]}\n')
        file.write(f'epochs: {args["epochs"]}\n\n')
        file.write(f'downsample: {args["downsample"]}\n')
        file.write(f'hidden sizes: {args["hidden"]}\n')
        print(report)
        print(str(confusion_matrix))
    return np.sum([1 if test_y[i]==preds[i] else 0 for i in range(len(preds))])/len(preds)
