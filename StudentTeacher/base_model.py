import torch.nn as nn
import torch

class PTModel(nn.Module):
    def __init__(self, model, dropout):
        super(PTModel, self).__init__()
        self.model = model
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.model.config.hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sent_id, mask):
        x = self.model(sent_id, attention_mask=mask)[1]
        x = self.dropout(x)
        x = self.out(x)
        x = self.softmax(x)
        return x
    
class Model(nn.Module):
    def __init__(self, input_size, hidden_size, dropout):
        super(Model, self).__init__()
        self.expand_dim = nn.Conv1d(in_channels=1, out_channels=hidden_size, kernel_size=1)
        
        self.rnn = nn.GRU(input_size, hidden_size, 1, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_size, 2) 
        self.softmax = nn.Softmax(dim=1)

    def forward(self, sent_id):
        x = torch.unsqueeze(sent_id, 1).float()
        x = self.expand_dim(x)
        x, _ = self.rnn(x)
        x = self.dropout(x)
        x = torch.max(x, dim=1)[0]
        x = self.out(x)
        x = self.softmax(x)
        return x