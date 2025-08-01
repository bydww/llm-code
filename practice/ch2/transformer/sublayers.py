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
        scores = torch.matmul(q,k.transpose(-2,-1))/math.sqrt(d_k)
        
        if mask is not None:
            mask = mask.unsqueeze(1)
            scores = scores.masked_fill(mask==0,-1e9)
        
        scores = F.softmax(scores,dim=-1)

        if dropout is not None:
            scores = dropout(scores)
        
        output = torch.matmul(scores,v)

        return output
        
    def forward(self,q,k,v,mask=None):
        bs = q.size(0)
        q = self.q_linear(q).view(bs,-1,self.h,self.d_k)
        k = self.k_linear(k).view(bs,-1,self.h,self.d_k)
        v = self.q_linear(v).view(bs,-1,self.h,self.d_k)

        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        scores = self.attention(q,k,v,self.d_k,mask)

        concat = scores.transpose(1,2).contiguous().view(bs,-1,self.d_model)
        out = self.out(concat)

        return out
    

'''前馈层'''
class FeedForward(nn.Module):
    def __init__(self,d_model,d_ff = 2048,dropout = 0.1):
        super().__init__()
        self.linear_1 = nn.Linear(d_model,d_ff)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(d_ff,d_model)
    def forward(self,x):
        x = self.dropout(F.relu(self.linear_1(x)))
        x = self.linear_2(x)

        return x

'''层归一化'''

class Norm(nn.Module):
    def __init__(self,d_model,eps = 1e-6 ):
        super().__init__()
        self.size = d_model

        self.alpha = nn.Parameter(torch.ones(self.size))
        self.bias = nn.Parameter(torch.zeros(self.size))
        self.eps = eps
    def forward(self,x):
        norm = self.alpha*(x-x.mean(dim=-1,keepdim=True))/(x.std(dim=-1,keepdim=True)+self.eps)+self.bias
        return norm