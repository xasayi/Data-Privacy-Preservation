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