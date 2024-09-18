import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LayerNorm(nn.Module):
    """
    Layer Normalization
    """

    def __init__(self, features, eps=1e-6):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta



class MyMultiHeadAttention(nn.Module):
    def __init__(self,args,n_head,d_model,d_kv,dropout=0.1):
        super(MyMultiHeadAttention,self).__init__()
        self.args=args
        self.slf_attn=MultiHeadAttention(args,n_head,d_model,d_kv,d_kv,dropout=dropout)
        self.pos_ffn=PositionwiseFeedForward(args,d_model,d_model,dropout=dropout)


    def forward(self,q,k,v,mask=None):
        output,p_attn=self.slf_attn(q,k,v,mask)
        output=self.pos_ffn(output)

        return  output,p_attn



class MultiHeadAttention(nn.Module):
    def __init__(self,args,n_head,d_model,d_k,d_v,dropout=0.1):
        super().__init__()
        self.args = args

        self.n_head=n_head
        self.d_k=d_k
        self.d_v=d_v

        self.w_q=nn.Linear(d_model,n_head*d_k)
        self.w_k=nn.Linear(d_model,n_head*d_k)
        self.w_v=nn.Linear(d_model,n_head*d_v)
        nn.init.normal_(self.w_q.weight,mean=0,std=np.sqrt(2.0/(d_model+d_k)))
        nn.init.normal_(self.w_k.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_k)))
        nn.init.normal_(self.w_v.weight, mean=0, std=np.sqrt(2.0 / (d_model + d_v)))

        self.attention=ScaledDotProductAttention(temperature=np.power(d_k,0.5))
        self.layer_norm=LayerNorm(d_model)


        self.fc=nn.Linear(n_head*d_v,d_model)
        nn.init.xavier_normal_(self.fc.weight)
        self.dropout=nn.Dropout(dropout)


    def forward(self,q,k,v,mask=None):

        d_k,d_v,n_head=self.d_k,self.d_v,self.n_head

        sz_b,len_q,_=q.size()
        sz_b,len_k,_=k.size()
        sz_b,len_v,_=v.size()

        residual=q

        q=self.w_q(q).view(sz_b,len_q,n_head,d_k)
        k=self.w_k(k).view(sz_b,len_k,n_head,d_k)
        v=self.w_v(v).view(sz_b,len_v,n_head,d_v)

        q=q.permute(2,0,1,3).contiguous().view(-1,len_q,d_k)  #n_head*batch_size,seq_len,model_dim
        k = k.permute(2, 0, 1, 3).contiguous().view(-1, len_k, d_k)  # n_head*batch_size,seq_len,model_dim
        v = v.permute(2, 0, 1, 3).contiguous().view(-1, len_v, d_v)  # n_head*batch_size,seq_len,model_dim

        output,attn=self.attention(q,k,v,mask=mask,n_head=n_head)

        output=output.view(n_head,sz_b,len_q,d_v)

        output=output.permute(1,2,0,3).contiguous()
        output=output.view(sz_b,len_q,-1)


        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output,attn

class ScaledDotProductAttention(nn.Module):
    def __init__(self,temperature,attn_dropout=0.1):
        super().__init__()
        self.temperature=temperature
        self.dropout=nn.Dropout(attn_dropout)
        self.softmax=nn.Softmax(dim=2)

    def forward(self,q,k,v,mask=None,n_head=None):
        attn=torch.bmm(q,k.transpose(1,2))
        attn=attn/self.temperature

        if mask is not None:
            attn_mask = mask.unsqueeze(-1).repeat(1, 1, q.shape[1])
            attn_mask = attn_mask.unsqueeze(1).expand(mask.size(0), n_head, q.size(1), q.size(1))
            attn_mask=attn_mask.reshape(mask.size(0)*n_head,q.size(1),q.size(1))

            attn = attn - attn_mask.masked_fill(attn_mask == -10000, 1e9)
            attn_mask_T = torch.transpose(attn_mask, -2, -1)
            attn_mask_T = attn_mask_T.masked_fill(attn_mask_T == -10000, 1e9)
            attn = attn - attn_mask_T

        attn=self.softmax(attn)
        attn=self.dropout(attn)
        output=torch.bmm(attn,v)

        return output,attn


class PositionwiseFeedForward(nn.Module):
    def __init__(self,args,d_in,d_hid,dropout=0.1):
        super().__init__()
        self.args=args
        self.w_1=nn.Linear(d_in,d_hid)
        self.w_2=nn.Linear(d_hid,d_in)
        self.layer_norm=LayerNorm(d_in)
        self.dropout=nn.Dropout(dropout)

    def forward(self,x):
        residual=x
        output=x
        output=self.w_2(F.relu(self.w_1(output)))
        #output=output.transpose(1,2)
        output=self.dropout(output)

        output = self.layer_norm(output + residual)


        return output
