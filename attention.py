import math
import copy
import  torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import BCELoss


def _getMatrixTree_multi(scores,root,is_multi_head):
    A=scores.exp()
    R=root.exp()

    if is_multi_head is True:
        L = torch.sum(A, 2)
    else:
        L = torch.sum(A, 1)

    L=torch.diag_embed(L)
    L=L-A                  #拉普拉斯矩阵
    if is_multi_head is True:
        R = R.squeeze(-1).unsqueeze(1).expand(A.size(0), A.size(1), A.size(2))
    else:
        R = R.squeeze(-1)

    LL=L+torch.diag_embed(R)   #加了根概率的拉普拉斯矩阵

    LL_inv=torch.inverse((LL))    #得到LL的逆矩阵
    if is_multi_head is True:
        LL_inv_diag = torch.diagonal(LL_inv, 0, 2, 3)
    else:
        LL_inv_diag = torch.diagonal(LL_inv, 0, 1, 2)

    d0=R*LL_inv_diag     #每个节点成为根的边际概率  B,h,L

    if is_multi_head is True:
        LL_inv_diag = torch.unsqueeze(LL_inv_diag, 3)
    else:
        LL_inv_diag = torch.unsqueeze(LL_inv_diag, 2)


    _A=torch.transpose(A,-2,-1)
    _A=_A*LL_inv_diag
    tmp1=torch.transpose(_A,-2,-1)
    tmp2=A*torch.transpose(LL_inv,-2,-1)

    d=tmp1-tmp2       #两个节点的边的边际概率 B,L,L

    return d,d0

class StructuredAttention(nn.Module):
    def __init__(self,args):
        self.model_dim=args.hidden_dim
        self.h=args.dynamic_tree_attn_head
        self.d_k=args.hidden_dim//args.dynamic_tree_attn_head
        self.device=args.device

        super(StructuredAttention,self).__init__()

        self.linear_keys=nn.Linear(args.hidden_dim,self.model_dim)
        self.linear_query=nn.Linear(args.hidden_dim,self.model_dim)
        self.linear_root=nn.Linear(args.hidden_dim,1)


    def forward(self,x,mask=None,roots_label=None,root_mask=None,dep_type_adj=None,is_multi_head=None):


        key=self.linear_keys(x)
        query=self.linear_query(x)
        root=self.linear_root(x)
        batches = key.size(0)
        len=key.size(1)

        query=query/math.sqrt(self.model_dim)
        if is_multi_head is True:
            key = key.view(batches, -1, self.h, self.d_k).transpose(1, 2)
            query = query.view(batches, -1, self.h, self.d_k).transpose(1, 2)


        scores=torch.matmul(query,key.transpose(-2,-1))

        if dep_type_adj is not None:
            scores=scores+dep_type_adj

        mask=mask/-10000
        root=root-mask.unsqueeze(-1)*50
        root=torch.clamp(root,min=-40)

        scores_mask=mask.unsqueeze(-1).repeat(1,1,x.shape[1])
        if is_multi_head is True:
            scores_mask = scores_mask.unsqueeze(1).expand(batches, self.h, len, len)


        scores = scores - scores_mask * 50
        scores = scores - torch.transpose(scores_mask, -2, -1) * 50
        scores = torch.clamp(scores, min=-40)


        d,d0=_getMatrixTree_multi(scores,root,is_multi_head)  #d->B,h,L,L  d0->B,L

        loss_root_all=[]
        if roots_label is not None:

            loss_fct=BCELoss(reduction='none')
            if root_mask is not None:
                active_labels=roots_label.view(-1)

                if is_multi_head is True:
                    d0_all = [d0_temp.squeeze(1) for d0_temp in torch.split(d0, 1, dim=1)]
                    for i in range(self.h):
                        d0_all[i] = d0_all[i].contiguous().view(-1)
                        d0_all[i] = torch.clamp(d0_all[i], 1e-5, 1 - 1e-5)
                        temp_loss = loss_fct(d0_all[i].to(torch.float32), active_labels.to(torch.float32))
                        loss_root_all.append(temp_loss)

                    loss_root = sum(loss_root_all)
                    loss_root = (loss_root * roots_label.view(-1).float()).mean()

                else:
                    active_logits = d0.view(-1)
                    active_logits=torch.clamp(active_logits,1e-5,1 - 1e-5)
                    loss_root=loss_fct(active_logits.to(torch.float32),active_labels.to(torch.float32))
                    loss_root = (loss_root * roots_label.view(-1).float()).mean()



        attn=torch.transpose(d,-2,-1)
        if mask is not None:
            scores_mask=scores_mask+torch.transpose(scores_mask,-2,-1)
            scores_mask=scores_mask.bool()

            attn=attn.masked_fill(scores_mask,0)

        if is_multi_head is True:
            x = x.unsqueeze(1).expand(batches, self.h, len, self.model_dim)



        context=torch.matmul(attn,x)


        return context,d, d0, loss_root




def attention(query,key,mask=None,dropout=None):
    d_k=query.size(-1)
    scores=torch.matmul(query,key.transpose(-2,-1))/math.sqrt(d_k)

    if mask is not None:
        scores_mask = mask.unsqueeze(-1).repeat(1, 1, query.shape[2])
        scores_mask = scores_mask.unsqueeze(1).expand(query.size(0), query.size(1), query.size(2), query.size(2))

        scores = scores - scores_mask.masked_fill(scores_mask == -10000, 1e9)
        scores_mask_T = torch.transpose(scores_mask, -2, -1)
        scores_mask_T = scores_mask_T.masked_fill(scores_mask_T == -10000, 1e9)
        scores = scores - scores_mask_T

    p_attn=F.softmax(scores,dim=-1)


    if dropout is not None:
        p_attn=dropout(p_attn)
    return p_attn


def clones(module,N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAttention(nn.Module):

    def __init__(self,args,dropout=0.1):
        super(MultiHeadAttention,self).__init__()
        self.model_dim = args.hidden_dim
        self.h = args.attention_heads
        self.d_k = args.hidden_dim // args.attention_heads
        self.dropout=nn.Dropout(p=dropout)
        self.linears=clones(nn.Linear(args.hidden_dim,args.hidden_dim),2)

    def forward(self,query,key,mask=None):
        nbatches=query.size(0)

        query,key=[l(x).view(nbatches,-1,self.h,self.d_k).transpose(1,2) for l,x in zip(self.linears,(query,key))]

        attn=attention(query,key,mask=mask,dropout=self.dropout)
        return attn



class LocalAttention(nn.Module):

    def __init__(self,args,dropout=0.1):
        super(LocalAttention,self).__init__()
        self.model_dim=args.hidden_dim
        self.h = args.attention_heads
        self.d_k = args.hidden_dim // args.attention_heads
        self.dropout = nn.Dropout(p=dropout)
        self.device = args.device


        #左掩盖线性层
        self.linear_keys_left = nn.Linear(args.hidden_dim, self.model_dim)
        self.linear_query_left = nn.Linear(args.hidden_dim, self.model_dim)

        #右掩盖线性层
        self.linear_keys_right = nn.Linear(args.hidden_dim, self.model_dim)
        self.linear_query_right = nn.Linear(args.hidden_dim, self.model_dim)

        #注意力分数线性层
        self.linear_keys = nn.Linear(args.hidden_dim, self.model_dim)
        self.linear_query = nn.Linear(args.hidden_dim, self.model_dim)


    def forward(self,query,key,mask=None,span_matrix=None,aspect_mask=None):
        batch_size=query.size(0)
        seq_len=query.size(1)
        hidden_dim=query.size(2)

        left_boundary=np.ones([batch_size,self.h,seq_len,seq_len])
        left_boundary=np.tril(left_boundary,k=0)
        left_boundary=np.where(left_boundary==0,1e9,left_boundary)
        left_boundary = np.where(left_boundary == 1, 0, left_boundary)
        left_boundary=torch.tensor(left_boundary)
        left_boundary=left_boundary.cuda(0)
        right_boundary=torch.transpose(left_boundary,-2,-1)

        key_left=key
        key_right=key
        query_left=query
        query_right=query
        #左边界
        key_left=self.linear_keys_left(key_left)
        query_left = self.linear_query_left(query_left)
        key_left = key_left.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        query_left = query_left.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        query_left = query_left / math.sqrt(self.model_dim)
        left_scores = torch.matmul(query_left, key_left.transpose(-2, -1))
        theta_l=F.softmax(left_scores-left_boundary,dim=-1)

        # 右边界
        key_right = self.linear_keys_right(key_right)
        query_right = self.linear_query_right(query_right)
        key_right = key_right.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        query_right = query_right.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        query_right = query_right / math.sqrt(self.model_dim)
        right_scores = torch.matmul(query_right, key_right.transpose(-2, -1))
        theta_r = F.softmax(right_scores - right_boundary, dim=-1)


        #软掩码
        downTri_M = np.ones([batch_size, self.h, seq_len, seq_len])
        downTri_M = np.tril(downTri_M, k=0)
        downTri_M = torch.tensor(downTri_M)
        downTri_M=downTri_M.cuda(0)
        upperTri_M=torch.transpose(downTri_M,-2,-1)
        soft_Mask_l=torch.matmul(theta_l,upperTri_M)
        soft_Mask_r = torch.matmul(theta_r, downTri_M)
        soft_Mask=soft_Mask_l*soft_Mask_r


        #注意力分数
        #self attention
        bert_out=query
        key = self.linear_keys(key)
        query = self.linear_query(query)
        key = key.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        query = query.view(batch_size, -1, self.h, self.d_k).transpose(1, 2)
        query = query / math.sqrt(self.model_dim)
        att_scores = torch.matmul(query, key.transpose(-2, -1))



        scores_mask = mask.unsqueeze(-1).repeat(1, 1, query.shape[2])
        scores_mask = scores_mask.unsqueeze(1).expand(query.size(0), query.size(1), query.size(2), query.size(2))
        att_scores = att_scores - scores_mask.masked_fill(scores_mask == -10000, 1e9)
        scores_mask_T = torch.transpose(scores_mask, -2, -1)
        scores_mask_T = scores_mask_T.masked_fill(scores_mask_T == -10000, 1e9)
        att_scores = att_scores - scores_mask_T


        p_local_attn = F.softmax(att_scores * soft_Mask, dim=-1)       #batch_size,h,seq_len,seq_len

        #范围约束
        span_matrix=span_matrix.transpose(0,1)
        p_attn_all = [p_attn_temp.squeeze(1) for p_attn_temp in torch.split(p_local_attn, 1, dim=1)]
        span_matrix_all=[span_matrix_temp.squeeze(1) for span_matrix_temp in torch.split(span_matrix, 1, dim=1)]
        span_loss_all=[]
        loss_fct = BCELoss(reduction='none')
        for i in range(self.h):
            p_attn_all[i]=torch.sigmoid(p_attn_all[i])
            temp_loss = loss_fct(p_attn_all[i].view(-1).to(torch.float32), span_matrix_all[i].view(-1).to(torch.float32))
            loss_mean=torch.mean(temp_loss,dim=-1)
            span_loss_all.append(loss_mean)

        span_loss=sum(span_loss_all)/len(span_loss_all)

        return p_local_attn,span_loss





















