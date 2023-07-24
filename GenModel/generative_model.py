import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AdamW, get_linear_schedule_with_warmup
from GenModel.process_data import process_data

class GenerativeModel(nn.Module):
    def __init__(self, model, lr, tokenizer, splits, batch_size, data_filename, index, epochs, weight_path, folder, device):
        super(GenerativeModel, self).__init__()
        self.model = model
        self.optimizer = AdamW(self.model.parameters(), lr=lr)
        self.tokenizer = tokenizer
        self.train_dataloader, self.valid_dataloader, self.test_data = process_data(tokenizer, splits, batch_size, 
                                                                                             data_filename, index)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 
                                                    num_warmup_steps=200, num_training_steps=-1)
        self.epochs = epochs
        self.batch_size = batch_size
        self.weight_path = weight_path
        self.folder = folder
        self.device = device
    
    def train(self):
        self.model.train()
        total_loss = 0
        for step, batch in enumerate(self.train_dataloader):
            batch = [r.to(self.device) for r in batch]
            sent_id = batch[0]
            outputs = self.model(sent_id, labels=sent_id)
            loss = outputs[0]
            loss.backward()
            self.model.zero_grad()
            self.optimizer.step()
            self.scheduler.step()
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(self.train_dataloader)))
            total_loss += loss
        avg_loss = total_loss / len(self.train_dataloader)  
        return avg_loss,
  
    def eval(self):
        print("\nEvaluating...")
        self.model.eval()
        total_loss = 0

        for step,batch in enumerate(self.valid_dataloader):
            if step % 50 == 0 and not step == 0:
                print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(self.valid_dataloader)))
            batch = [t.to(self.device) for t in batch]
            sent_id = batch[0]
            with torch.no_grad():
                output = self.model(sent_id, labels=sent_id)
                #input = self.tokenizer.encode('spam ', return_tensors='pt').to(self.device)
                #sample = self.model.generate(input.float(), do_sample=True,
                #                      max_length=50, top_k=50, top_p=0.95)
                #print(self.tokenizer.decode(sample, skip_special_tokens=True))
                loss = output[0]
                total_loss += loss
        avg_loss = total_loss / len(self.valid_dataloader)
        return avg_loss
    
    def run(self):
        best_valid_loss = float('inf')
        train_losses, valid_losses = [], []
        for epoch in range(self.epochs):
            print('\n Epoch {:} / {:}'.format(epoch + 1, self.epochs))
            train_loss = self.train()
            valid_loss = self.eval()

            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(self.model.state_dict(), f'{self.folder}/{self.weight_path}')

            train_losses.append(train_loss)
            valid_losses.append(valid_loss)

            print(f'\nTraining Loss: {train_loss[0]:.3f}')
            print(f'Validation Loss: {valid_loss:.3f}')
        return train_losses, valid_losses