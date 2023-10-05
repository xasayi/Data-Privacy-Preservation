import torch.nn as nn
import torch
    
class EmbedModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, linear_size, dropout):
        super(EmbedModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, max_norm=True)
        self.linear1 = nn.Linear(embedding_size, linear_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(linear_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_embed = self.embed(x)
        x_embed = torch.mean(x_embed, dim=1)
        fc1 = self.linear1(x_embed)
        fc1 = self.relu(fc1)
        fc1 = self.dropout(fc1)
        fc2 = self.linear2(fc1)
        pred = self.sigmoid(fc2)
        return fc1, fc2, pred
    
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, linear_size, dropout):
        super(LSTMModel, self).__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, max_norm=True)
        self.lstm1 = nn.LSTM(embedding_size, linear_size)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(linear_size, 10)
        self.dropout2 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(10, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_embed = self.embed(x)
        x_embed = torch.mean(x_embed, dim=1)
        lstm1, _ = self.lstm1(x_embed.view(len(x_embed), -1))
        lstm1 = self.dropout1(lstm1)
        lstm2, _ = self.lstm2(lstm1.view(len(lstm1), -1))
        lstm2 = self.dropout2(lstm2)
        fc1 = self.linear1(lstm2)
        pred = self.sigmoid(fc1)
        return lstm2, fc1, pred