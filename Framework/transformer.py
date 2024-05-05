import torch
from torch import nn
import pandas as pd
import numpy as np
from transformers import BertTokenizer
from sklearn.model_selection import train_test_split
    
class Block(nn.Module):
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
       attn, _ = self.attention(hidden_state, hidden_state, hidden_state, need_weights=False)
       attn = self.drop1(attn)
       out = self.ln1(hidden_state + attn)
       observed = self.ffn(out)
       observed = self.drop2(observed)
       return self.ln2(out + observed)

class MyTransformer(nn.Module):
   def __init__(self, block_size, vocab_size, embeds_size, drop_prob, num_classes, num_heads, n_layers, device):
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
           nn.LogSoftmax(dim=1),
       )
       print("number of parameters: %.2fM" % (self.num_params()/1e6,))
   
   def num_params(self):
       n_params = sum(p.numel() for p in self.parameters())
       return n_params
   
   def forward(self, seq):
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

if __name__ == '__main__':
    model = MyTransformer(block_size=50, vocab_size=30522, embeds_size=32,
                        drop_prob=0.1, num_classes=6, num_heads=8, n_layers=1, device=torch.device("cpu"))
    
    print('model ok')
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    tokenizer.pad_token = '[PAD]'
    print('tokenizer ok')
    data = pd.read_csv('/Users/sarinaxi/Desktop/Thesis/Framework/data/sentiment_data/huggingface_private69092.csv').iloc[:100]
    val_data, test_data, val_labels, test_labels = train_test_split(data['data'], data['label'],
                                                                        test_size=0.5,
                                                                        stratify=data['label'])
    test_data, test_labels = test_data.reset_index(drop=True), test_labels.reset_index(drop=True)
    tokenized_data = tokenizer.batch_encode_plus(test_data, max_length=50,
                                                 padding=True, truncation=True, 
                                                 return_token_type_ids=False)
    data_seq = torch.tensor(tokenized_data['input_ids'])
    data_mask = torch.tensor(tokenized_data['attention_mask'])
    data_y = torch.tensor(test_labels)
    print('data ok')
    print(data_seq.shape)
    preds = model(data_seq[:10])
    print(np.argmax(preds.detach().numpy(), axis=-1))
    print(data_y[:10].detach().numpy())
    
    
