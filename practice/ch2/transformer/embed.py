import torch
import torch.nn as nn
import math
from torch.autograd import Variable


class PositionalEncoder(nn.Module):
    def __init__(self, d_model, max_seq_len = 80):
        super().__init__()
        self.d_model = d_model
        pe = torch.zeros(max_seq_len,d_model)
        for pos in range(max_seq_len):
            for i in range(0,d_model,2):
                pe[pos,i] = math.sin(pos/(10000**(i/d_model)))
                pe[pos,i+1] = math.cos(pos/(10000**(i/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer('pe',pe)
    def forward(self,x):
        #使得单词嵌入表示相对大一些
        device = x.device
        x = x*math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len],requires_grad=False).to(device)
        return x 
class Embedder(nn.Module):
    def __init__(self, vocab_size,d_model):
        super().__init__()
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size,d_model)
    def forwad(self,x):
        return self.emb(x)