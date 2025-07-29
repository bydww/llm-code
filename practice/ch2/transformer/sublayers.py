import torch
import torch.nn as nn
import torch.nn.functional as F
import math


'''多头注意力机制'''
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model,heads,dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_model // heads
        self.h = heads
        
        self.q_linear = nn.Linear(d_model,d_model)
        self.k_linear = nn.Linear(d_model,d_model)
        self.v_linear = nn.Linear(d_model,d_model)
        
        self.out = nn.Linear(d_model,d_model)
        self.dropout = nn.Dropout(dropout)
    
    def attention(self,q,k,v,d_k,mask=None,dropout=None):
        scores = torch.matmul
        
        