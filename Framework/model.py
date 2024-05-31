import torch
from torch import nn

class EmbedModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, linear_size, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, max_norm=True)
        self.linear1 = nn.Linear(embedding_size, linear_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(linear_size, 2)
        self.softmax = nn.LogSoftmax()

    def forward(self, input_):
        x_embed = self.embed(input_)
        x_embed = torch.mean(x_embed, dim=1)
        fc1 = self.linear1(x_embed)
        fc1 = self.relu(fc1)
        fc1 = self.dropout(fc1)
        fc2 = self.linear2(fc1)
        pred = self.softmax(fc2)
        return fc1, fc2, pred

class LSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_size, linear_size, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embedding_size, max_norm=True)
        self.lstm1 = nn.LSTM(embedding_size, linear_size)
        self.dropout1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(linear_size, 10)
        self.dropout2 = nn.Dropout(dropout)
        self.linear1 = nn.Linear(10, 2)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_):
        x_embed = self.embed(input_)
        x_embed = torch.mean(x_embed, dim=1)
        lstm1, _ = self.lstm1(x_embed.view(len(x_embed), -1))
        lstm1 = self.dropout1(lstm1)
        lstm2, _ = self.lstm2(lstm1.view(len(lstm1), -1))
        lstm2 = self.dropout2(lstm2)
        fc1 = self.linear1(lstm2)
        pred = self.softmax(fc1)
        return lstm2, pred

class LSTMModelMulti(nn.Module):
    def __init__(self, size, vocab_size, hidden, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden[0], max_norm=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm1 = nn.LSTM(hidden[0], hidden[1],
                             bidirectional=True, batch_first=True)
        self.lstm2 = nn.LSTM(hidden[1]*2, hidden[2],
                             bidirectional=True, batch_first=True)
        self.lstm3 = nn.LSTM(hidden[1]*2+hidden[2]*2, hidden[3],
                             bidirectional=True, batch_first=True)
        self.fc1 = nn.Linear(hidden[3]*2, size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_):
        x_embed = self.embed(input_)
        x_embed = torch.mean(x_embed, dim=1)
        lstm1, _ = self.lstm1(x_embed.view(len(x_embed), -1))
        lstm1 = self.dropout(lstm1)
        lstm2, _ = self.lstm2(lstm1.view(len(lstm1), -1))
        lstm2 = self.dropout(lstm2)
        input_ = torch.concat((lstm1, lstm2), axis=1)
        lstm3, _ = self.lstm3(input_.view(len(input_), -1))
        pred = self.fc1(lstm3)
        pred = self.softmax(pred)
        return lstm2, pred

class LSTMModelMulti2(nn.Module):
    def __init__(self, size, vocab_size, hidden, dropout):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden[0], max_norm=True)
        self.dropout = nn.Dropout(dropout)
        self.lstm1 = nn.LSTM(hidden[0], hidden[1],
                             bidirectional=True, batch_first = True)
        self.lstm2 = nn.LSTM(hidden[1]*2, hidden[2],
                             bidirectional=True, batch_first = True)
        self.lstm3 = nn.LSTM(hidden[1]*2+hidden[2]*2, hidden[3],
                             bidirectional=True, batch_first = True)
        self.lstm4 = nn.LSTM(hidden[3]*2, hidden[4],
                             bidirectional=True, batch_first = True)
        self.lstm5 = nn.LSTM(hidden[1]*2+hidden[3]*2+hidden[4]*2, hidden[5],
                             bidirectional=True, batch_first = True)
        self.fc1 = nn.Linear(hidden[5]*2, size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input_):
        x_embed = self.embed(input_)
        x_embed = torch.mean(x_embed, dim=1)
        lstm1, _ = self.lstm1(x_embed.view(len(x_embed), -1))
        lstm1 = self.dropout(lstm1)
        lstm2, _ = self.lstm2(lstm1.view(len(lstm1), -1))
        lstm2 = self.dropout(lstm2)
        res = torch.concat((lstm1, lstm2), axis=1)
        lstm3, _ = self.lstm3(res.view(len(res), -1))
        lstm4 = self.dropout(lstm3)
        lstm4, _ = self.lstm4(lstm3.view(len(lstm3), -1))
        lstm4 = self.dropout(lstm4)
        input2 = torch.concat((lstm1, lstm3, lstm4), axis=1)
        lstm5, _ = self.lstm5(input2.view(len(input2), -1))
        pred = self.fc1(lstm5)
        pred = self.softmax(pred)
        return lstm2, pred
