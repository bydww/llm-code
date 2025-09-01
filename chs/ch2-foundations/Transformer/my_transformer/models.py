import torch
import torch.nn as nn 
from layers import EncoderLayer, DecoderLayer
from embed import Embedder, PositionalEncoder
from sublayers import Norm
import copy

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

'''编码器'''
class Encoder(nn.Module):
    def __init__(self,vocab_size,d_model,N,heads,dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size,d_model)
        self.pe = PositionalEncoder(d_model,dropout=dropout)
        self.layers = get_clones(EncoderLayer(d_model,heads,dropout),N)
        self.norm = Norm(d_model)
    
    def forward(self,src,mask):
        x = self.embed(src)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x,mask)
        return self.norm(x)

'''解码器'''
class Decoder(nn.Module):
    def __init__(self,vocab_size,d_model,N,heads,dropout):
        super().__init__()
        self.N = N
        self.embed = Embedder(vocab_size,d_model)
        self.pe = PositionalEncoder(d_model,dropout=dropout)
        self.layers = get_clones(DecoderLayer(d_model,heads,dropout),N)
        self.norm = Norm(d_model)
        
    def forward(self,trg,e_outputs,src_mask,trg_mask):
        x = self.embed(trg)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x,e_outputs,src_mask,trg_mask)
        return self.norm(x)

class Transformer(nn.Module):
    def __init__(self,src_vocab, trg_vocab,d_model,N,heads,dropout):
        super().__init__()
        self.encoder = Encoder(src_vocab,d_model,N,heads,dropout)
        self.decoder = Decoder(trg_vocab,d_model,N,heads,dropout)
        self.out = nn.Linear(d_model, trg_vocab)

    def forward(self,src,trg,src_mask,trg_mask):
        e_outputs = self.encoder(src,src_mask)
        d_output = self.decoder(trg,e_outputs,src_mask,trg_mask)
        return self.out(d_output)
    
if __name__ == "__main__":
    model = Transformer(52,52,512,6,8,0.1)
    src_input = torch.randint(0,52,(32,10))
    trg_input = torch.randint(0,52,(32,10))
    src_mask = torch.ones(10,10).bool().unsqueeze(0).unsqueeze(1)
    trg_mask = torch.ones(10,10).bool().unsqueeze(0).unsqueeze(1)
    output = model(src_input,trg_input,src_mask,trg_mask)
    print(output.shape)
