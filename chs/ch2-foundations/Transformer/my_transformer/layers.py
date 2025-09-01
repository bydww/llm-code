import torch
import torch.nn as nn
from sublayers import FeedForward, MultiHeadAttention, Norm

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads,dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        
        self.attn = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)
        
        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
    
    def forward(self, x, mask):
        attn_output = self.attn(x,x,x,mask)
        attn_output = self.dropout_1(attn_output)
        x = attn_output + x 
        x = self.norm_1(x)
        ff_output = self.ff(x)
        ff_output = self.dropout_2(ff_output)
        x = ff_output + x
        x = self.norm_2(x)
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, heads, dropout=0.1):
        super().__init__()
        self.norm_1 = Norm(d_model)
        self.norm_2 = Norm(d_model)
        self.norm_3 = Norm(d_model)

        self.attn1 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.attn2 = MultiHeadAttention(heads, d_model, dropout=dropout)
        self.ff = FeedForward(d_model, dropout=dropout)

        self.dropout_1 = nn.Dropout(dropout)
        self.dropout_2 = nn.Dropout(dropout)
        self.dropout_3 = nn.Dropout(dropout)

    def forward(self, x, enc_out, src_mask, tgt_mask):
        attn_output = self.attn1(x, x, x, tgt_mask)
        attn_output = self.dropout_1(attn_output)
        x = attn_output + x
        x = self.norm_1(x)
        attn_output = self.attn2(x, enc_out, enc_out, src_mask)
        attn_output = self.dropout_2(attn_output)
        x = attn_output + x
        x = self.norm_2(x)
        ff_output = self.ff(x)
        ff_output = self.dropout_3(ff_output)
        x = ff_output + x
        x = self.norm_3(x)
        return x