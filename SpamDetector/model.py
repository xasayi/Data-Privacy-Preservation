import torch
import torch.nn as nn

class BERT_Arch(nn.Module):
    def __init__(self, bert, dropout):
        super(BERT_Arch, self).__init__()
        self.bert = bert 
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.bert.config.hidden_size, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, sent_id, mask):
        x = self.bert(sent_id, attention_mask=mask)[1]
        x = self.dropout(x)
        x = self.out(x)
        x = self.softmax(x)
        return x

class GPT_Arch(nn.Module):
    def __init__(self, gpt, dropout):
        super(GPT_Arch, self).__init__()
        self.gpt = gpt 
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(self.gpt.config.hidden_size, 2)
        self.softmax = nn.LogSoftmax(dim=-1)
        
    def forward(self, sent_id, mask):
        x = self.gpt(sent_id, attention_mask=mask)[0]
        x = self.dropout(torch.mean(x, axis=1))
        x = self.out(x)
        x = self.softmax(x)
        return x