import torch
import torch.nn as nn

class PT_Arch(nn.Module):
    def __init__(self, model, dropout, type):
        super(PT_Arch, self).__init__()
        self.model = model 
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.model.config.hidden_size, 2)
        self.softmax = nn.Softmax(dim=1)
        self.type_ = type

    def forward(self, sent_id, mask):
        if self.type_=='bert':
            x = self.model(sent_id, attention_mask=mask)[1]
        if self.type_ == 'gpt':
             x = self.model(sent_id, attention_mask=mask)[0]
             x = torch.mean(x, axis=1)
        x = self.dropout(x)
        x = self.out(x)
        x = self.softmax(x)
        return x