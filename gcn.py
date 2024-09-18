import copy
import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class LayerNorm(nn.Module):

    def __init__(self,features,eps=1e-6):
        super(LayerNorm,self).__init__()
        self.a_2=nn.Parameter(torch.ones(features))
        self.b_2=nn.Parameter(torch.zeros(features))
        self.eps=eps

    def forward(self,x):
        mean=x.mean(-1,keepdim=True)
        std=x.std(-1,keepdim=True)
        return self.a_2*(x-mean)/(std+self.eps)+self.b_2


class GCN_for_segment_aware_graph(nn.Module):
    def __init__(self,args):
        super(GCN_for_segment_aware_graph,self).__init__()
        self.opt=args
        self.layers=args.gcn_layers
        self.attention_heads=args.attention_heads
        self.hidden_dim=args.hidden_dim
        self.bert_drop=nn.Dropout(args.bert_dropout)
        self.gcn_drop=nn.Dropout(args.gcn_dropout)
        self.layernorm=LayerNorm(args.hidden_dim)

        self.gcn_dim=300
        self.W=nn.Linear(self.gcn_dim,self.gcn_dim)
        self.Wxx=nn.Linear(self.hidden_dim,self.gcn_dim)


    def forward(self,bert_input,attn_score,layer_mask,is_multi_head):
        batch=bert_input.size(0)
        len=bert_input.size(1)

        bert_input=self.layernorm(bert_input)
        bert_input=self.bert_drop(bert_input)
        gcn_input=self.Wxx(bert_input)


        if layer_mask is not None:
            weight_adj = attn_score * layer_mask.transpose(0, 1)
        else:
            weight_adj=attn_score

        gcn_output=gcn_input
        layer_list=[gcn_input]

        for i in range(self.layers):
            if is_multi_head is True:
                gcn_output = gcn_output.unsqueeze(1).expand(batch, self.attention_heads, len, self.gcn_dim)
            else:
                gcn_output=gcn_output


            Ax=torch.matmul(weight_adj,gcn_output)

            if is_multi_head is True:
                Ax = Ax.mean(dim=1)
                Ax = self.W(Ax)
            else:
                Ax = self.W(Ax)


            weight_gcn_output=F.relu(Ax)

            gcn_output=weight_gcn_output
            layer_list.append(gcn_output)
            gcn_output=self.gcn_drop(gcn_output) if i<self.layers-1 else gcn_output


        node_outputs=layer_list[-1]
        return node_outputs


class GCN_for_latentTree(nn.Module):
    def __init__(self,args):
        super(GCN_for_latentTree,self).__init__()
        self.opt=args
        self.layers=args.gcn_layers
        self.attention_heads=args.dynamic_tree_attn_head
        self.hidden_dim=args.hidden_dim
        self.bert_drop=nn.Dropout(args.bert_dropout)
        self.gcn_drop=nn.Dropout(args.gcn_dropout)
        self.layernorm=LayerNorm(args.hidden_dim)

        self.gcn_dim=300
        self.W=nn.Linear(self.gcn_dim,self.gcn_dim)
        self.Wxx = nn.Linear(self.hidden_dim, self.gcn_dim)


    def forward(self,bert_input,attn_score,layer_mask,is_multi_head=None):
        batch=bert_input.size(0)
        len=bert_input.size(1)

        bert_input=self.layernorm(bert_input)
        bert_input=self.bert_drop(bert_input)
        gcn_input=self.Wxx(bert_input)


        if layer_mask is not None:
            weight_adj = attn_score * layer_mask.transpose(0, 1)
        else:
            weight_adj=attn_score


        if is_multi_head is True:
            weight_adj_all = [weight_adj_temp.squeeze(1) for weight_adj_temp in torch.split(weight_adj, 1, dim=1)]

        gcn_output=gcn_input
        layer_list=[gcn_input]


#-----------------------------------------------------------------------------------------------------------------------------------------
        for i in range(self.layers):
            if is_multi_head is True:
                input = gcn_output.unsqueeze(1).expand(batch, self.attention_heads, len, self.gcn_dim)
            else:
                input=gcn_output

            Ax=torch.matmul(weight_adj,input)

            if is_multi_head is True:
                Ax = Ax.mean(dim=1)
                Ax = self.W(Ax)
            else:
                Ax = self.W(Ax)


            weight_gcn_output=F.relu(Ax)

            gcn_output=weight_gcn_output
            layer_list.append(gcn_output)
            gcn_output=self.gcn_drop(gcn_output) if i<self.layers-1 else gcn_output


        node_outputs=layer_list[-1]
        return node_outputs

def _get_clones(module,N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

