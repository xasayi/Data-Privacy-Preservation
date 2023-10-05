import numpy as np
import torch
import torch.nn as nn
from transformers import AdamW
import pandas as pd
from sklearn.metrics import classification_report

class SpamDetector(nn.Module):
    def __init__(self, model, train_dataloader, device, lr, batch_size, valid_dataloader, epochs, test_data, weights, folder, weight_path):
        super(SpamDetector, self).__init__()
        self.model = model
        self.optimizer = AdamW(self.model.parameters(), lr = lr)
        self.train_dataloader, self.valid_dataloader, self.test_data, weights = train_dataloader, valid_dataloader, test_data, weights
        self.cross_entropy  = nn.CrossEntropyLoss(weight=weights.to(device)) 
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_path = weight_path
        self.folder = folder
        self.device = device

    def get_loss(self, sent_id, mask, labels, train=True):
        preds = self.model(sent_id, mask)
        loss = self.cross_entropy(preds, labels)
        total_loss = loss.item()
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        return total_loss, preds
    
    def get_loss_student(self, sent_id, labels, train=True):
        preds = self.model(sent_id)
        #print(sent_id)
        #print(preds)
        #print(labels)
        loss = self.cross_entropy(preds, labels)
        #print(loss)
        total_loss = loss.item()
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        return total_loss, preds
 
    def get_acc(self, preds, labels):
        pred_y = np.argmax(preds.detach().cpu().numpy(), axis=1)
        accuracy = np.sum([1 if pred_y[j] == labels[j] else 0 for j in range(len(labels))])/len(labels)
        return accuracy

    # change so it's compatible with rnns output, within only sent_id and no mask
    def train(self, student):
        self.model.train()
        total_loss, total_accuracy = 0, 0
        for step, batch in enumerate(self.train_dataloader):

            batch = [r.to(self.device) for r in batch]
            sent_id, mask, labels = batch
            self.model.zero_grad()
            if student:
                loss, preds = self.get_loss_student(sent_id, labels)
            else:
                loss, preds = self.get_loss(sent_id, mask, labels)
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(self.train_dataloader)))
                output = np.argmax(preds.detach().cpu().numpy(), axis=1)
                print(f'Pred: {output}')
                print(f'Targ: {labels.detach().cpu().numpy()}')

            total_loss += loss
            total_accuracy += self.get_acc(preds, labels)
        avg_loss = total_loss / len(self.train_dataloader)
        avg_acc = total_accuracy / len(self.train_dataloader)
        return avg_loss, avg_acc

    def eval(self, student):
        print("\nEvaluating...")
        self.model.eval()
        total_loss, total_accuracy = 0, 0

        for step,batch in enumerate(self.valid_dataloader):
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(self.valid_dataloader)))
            batch = [t.to(self.device) for t in batch]
            sent_id, mask, labels = batch
            with torch.no_grad():
                if student:
                    loss, preds = self.get_loss_student(sent_id, labels, train=False)
                    #print(loss)
                else:
                    loss, preds = self.get_loss(sent_id, mask, labels, train=False)
                    #print(loss)
                total_loss += loss
                total_accuracy += self.get_acc(preds, labels)
        avg_loss = total_loss / len(self.valid_dataloader)
        avg_acc = total_accuracy / len(self.valid_dataloader)
        return avg_loss, avg_acc

    def run(self, student=False):
        best_valid_loss = float('inf')
        train_losses, valid_losses = [], []
        train_accs, valid_accs = [], []
        for epoch in range(self.epochs):
            print('\n Epoch {:} / {:}'.format(epoch + 1, self.epochs))
            train_loss, train_acc = self.train(student)
            valid_loss, valid_acc = self.eval(student)

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

def model_performance(args, model, test_seq, test_mask, test_y, device, folder, student=False):
    with torch.no_grad():
        if not student:
            preds = model(test_seq.to(device), test_mask.to(device))
        else:
            preds = model(test_seq.to(device))
        preds = preds.detach().cpu().numpy()

    preds = np.argmax(preds, axis = 1)
    with open(f'{folder}/results.txt', 'w') as f:
        report = classification_report(test_y, preds)
        confusion_matrix = pd.crosstab(test_y, preds)
        f.write(report)
        f.write(str(confusion_matrix))
        f.write(f'\ndropout: {args["dropout"]}\n')
        f.write(f'lr: {args["lr"]}\n')
        f.write(f'batch_size: {args["batch_size"]}\n')
        f.write(f'splits: {args["splits"]}\n')
        f.write(f'epochs: {args["epochs"]}\n')
        print(report)
        print(str(confusion_matrix))