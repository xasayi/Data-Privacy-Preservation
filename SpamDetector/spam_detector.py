import numpy as np
import torch
import torch.nn as nn
from transformers import AdamW
from process_data import process_data

class SpamDetector(nn.Module):
    def __init__(self, model, tokenizer, device, lr, batch_size, splits, epochs, data_filename, index, folder, weight_path, sms, easy):
        super(SpamDetector, self).__init__()
        self.model = model
        self.optimizer = AdamW(self.model.parameters(), lr = lr)
        self.train_dataloader, self.valid_dataloader, self.test_data, weights = process_data(tokenizer, splits, batch_size, 
                                                                                             data_filename, index, sms, easy)
        weights = torch.tensor([0.3, 1.5])
        print(weights)
        self.cross_entropy  = nn.CrossEntropyLoss(weight=weights.to(device)) 
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_path = weight_path
        self.folder = folder
        self.device = device

    def get_loss(self, sent_id, mask, labels, train=True):
        preds = self.model(sent_id, mask).squeeze()
        loss = self.cross_entropy(preds, labels)
        total_loss = loss.item()
        if train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
        return total_loss, preds
 
    def get_acc(self, preds, labels):
        pred_y = np.argmax(preds.detach().cpu().numpy(), axis=1)
        accuracy = np.sum([1 if pred_y[j] == labels[j] else 0 for j in range(len(labels))])
        return accuracy

    def train(self):
        self.model.train()
        total_loss, total_accuracy = 0, 0
        for step, batch in enumerate(self.train_dataloader):

            batch = [r.to(self.device) for r in batch]
            sent_id, mask, labels = batch
            self.model.zero_grad()
            loss, preds = self.get_loss(sent_id, mask, labels)
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(self.train_dataloader)))
                output = np.argmax(preds.detach().cpu().numpy(), axis=1)
                print(f'Pred: {output}')
                print(f'Targ: {labels.detach().cpu().numpy()}')

            total_loss += loss
            total_accuracy += self.get_acc(preds, labels)
        avg_loss = total_loss / len(self.train_dataloader)
        avg_acc = total_accuracy / len(self.train_dataloader) / self.batch_size
        return avg_loss, avg_acc

    def eval(self):
        print("\nEvaluating...")
        self.model.eval()
        total_loss, total_accuracy = 0, 0

        for step,batch in enumerate(self.valid_dataloader):
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(self.valid_dataloader)))
            batch = [t.to(self.device) for t in batch]
            sent_id, mask, labels = batch
            with torch.no_grad():
                loss, preds = self.get_loss(sent_id, mask, labels, train=False)
                total_loss += loss
                total_accuracy += self.get_acc(preds, labels)
        avg_loss = total_loss / len(self.valid_dataloader)
        avg_acc = total_accuracy / len(self.valid_dataloader) / self.batch_size
        return avg_loss, avg_acc

    def run(self):
        best_valid_loss = float('inf')
        train_losses, valid_losses = [], []
        train_accs, valid_accs = [], []
        for epoch in range(self.epochs):
            print('\n Epoch {:} / {:}'.format(epoch + 1, self.epochs))
            train_loss, train_acc = self.train()
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