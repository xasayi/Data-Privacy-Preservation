'''Define custome transformer model'''
import torch
from torch import nn

class Block(nn.Module):
    '''Define a transformer block consisting of multi head attention, ff layer, and lns'''
    def __init__(self, embeds_size, num_heads, drop_prob):
        super(Block, self).__init__()
        self.attention = nn.MultiheadAttention(embeds_size, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embeds_size, 4 * embeds_size),
            nn.LeakyReLU(),
            nn.Linear(4 * embeds_size, embeds_size),
            )
        self.drop1 = nn.Dropout(drop_prob)
        self.drop2 = nn.Dropout(drop_prob)
        self.ln1 = nn.LayerNorm(embeds_size, eps=1e-6)
        self.ln2 = nn.LayerNorm(embeds_size, eps=1e-6)

    def forward(self, hidden_state):
        '''forward pass'''
        attn, _ = self.attention(hidden_state, hidden_state, hidden_state, need_weights=False)
        attn = self.drop1(attn)
        out = self.ln1(hidden_state + attn)
        observed = self.ffn(out)
        observed = self.drop2(observed)
        return self.ln2(out + observed)

class MyTransformer(nn.Module):
    '''Define my transformer model'''
    def __init__(self, block_size, vocab_size, embeds_size, drop_prob,
                 num_classes, num_heads, n_layers, device):
        super(MyTransformer, self).__init__()
        self.device = device
        self.n_layers = n_layers
        self.tok_embs = nn.Embedding(vocab_size, embeds_size)
        self.pos_embs = nn.Embedding(block_size, embeds_size)
        self.block = Block(embeds_size, num_heads, drop_prob)
        self.classifier_head = nn.Sequential(
            nn.Linear(embeds_size, embeds_size),
            nn.LeakyReLU(),
            nn.Dropout(drop_prob),
            nn.Linear(embeds_size, embeds_size),
            nn.LeakyReLU(),
            nn.Linear(embeds_size, num_classes),
            nn.LogSoftmax(dim=1)
        )
        print("number of parameters: %.2fM" % (self.num_params()/1e6,))

    def num_params(self):
        '''Get the number of parameters'''
        n_params = sum(p.numel() for p in self.parameters())
        return n_params

    def forward(self, seq):
        '''forward pass'''
        # seq [batchsize, seq_len]
        embedded = self.tok_embs(seq)
        # embeded [batchsize, seq_len, embed]
        embedded = embedded + self.pos_embs(torch.arange(seq.shape[-1], device=self.device))
        for i in range(self.n_layers):
            embedded = self.block(embedded)
        # embeded [batchsize, seq_len, embed]
        output = embedded.mean(dim=1)
        # output [batchsize, embed]
        output = self.classifier_head(output)
        # output [batchsize, classes]
        return 0, output
