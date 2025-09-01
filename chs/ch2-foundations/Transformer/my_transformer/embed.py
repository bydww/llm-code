import torch
import torch.nn as nn
import math
from torch.autograd import Variable

class Embedder(nn.Module):
    def __init__(self,vocab_size,d_model):
        super().__init__()
        self.d_model=d_model
        self.embed = nn.Embedding(vocab_size,d_model)
    
    def forward(self,x):
        return self.embed(x)

class PositionalEncoder(nn.Module):
    def __init__(self,d_model,max_seq_len=80,dropout =0.1):
        super().__init__()
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_seq_len,d_model)
        for pos in range(max_seq_len):
            for i in range(0,d_model,2):
                pe[pos,i] = math.sin(pos/(10000**(i/d_model)))
                pe[pos,i+1] = math.cos(pos/(10000**(i/d_model)))
        pe = pe.unsqueeze(0)
        self.register_buffer("pe",pe)

    def forward(self,x):
        x = x * math.sqrt(self.d_model)
        seq_len = x.size(1)
        x = x + Variable(self.pe[:,:seq_len,:],requires_grad=False).to(x.device)
        return self.dropout(x)

if __name__ == "__main__":
    embedder = Embedder(vocab_size=52, d_model=512)
    positional_encoder = PositionalEncoder(d_model=512)
    x = torch.randint(0,52,(5,5))
    x = embedder(x)
    print(x.size())
    x = positional_encoder(x)
    print(x.size())