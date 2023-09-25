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
        x = self.embed(x)
        x = torch.mean(x, dim=1)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.linear2(x)
        x = self.sigmoid(x)
        return x
    
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
        x = self.embed(x)
        x = torch.mean(x, dim=1)
        x, _ = self.lstm1(x.view(len(x), -1))
        x = self.dropout1(x)
        x, _ = self.lstm2(x.view(len(x), -1))
        x = self.dropout2(x)
        x = self.linear1(x)
        x = self.sigmoid(x)
        return x