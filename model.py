import os
import pickle

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel, BertConfig
from attention import StructuredAttention as StructuredAtt
from attention import LocalAttention as LocalAtt
from gcn import GCN_for_segment_aware_graph, GCN_for_latentTree
from module_interaction import MyMultiHeadAttention

class S2GSL(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()
        in_dim = args.hidden_dim + args.hidden_dim + args.hidden_dim
        self.args = args
        self.device = args.device

        self.dual_graph_learning_module = Dual_graph_learning(args, tokenizer)

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(in_dim, args.num_class)



    def forward(self, inputs):
        device = torch.device(self.device)
        length, bert_length, word_mapback, \
        adj,  \
        map_AS, map_AS_idx, \
        bert_sequence, bert_segments_ids, \
        aspect_indi, aspect_token,\
        con_spans, \
        dep_type_matrix = inputs

        # input
        initial_input = (
        length[map_AS], bert_length[map_AS], word_mapback[map_AS], aspect_token,aspect_indi, bert_sequence, bert_segments_ids,
        adj[map_AS], con_spans, dep_type_matrix[map_AS])


        final_feature, span_loss, loss_root, graph_balance = self.dual_graph_learning_module(initial_input)  # 完全模型
        hiddens = final_feature

        # aspect-level
        logits = self.classifier(self.dropout(hiddens))


        return logits, span_loss, loss_root, graph_balance



# Dual_graph_learning
class Dual_graph_learning(nn.Module):
    def __init__(self, args, tokenizer):
        super().__init__()

        self.args = args

        bert_config = BertConfig.from_pretrained('../pretrain_model/bert-base-uncased/config.json')
        bert_config.output_hidden_states = True
        bert_config.num_labels = 3

        self.layer_drop = nn.Dropout(args.layer_dropout)
        self.context_encoder = BertModel.from_pretrained('../pretrain_model/bert-base-uncased', config=bert_config)
        self.dense = nn.Linear(args.bert_hidden_dim, args.hidden_dim)
        self.tokenizer = tokenizer
        self.data_name = args.data_name


        # syntactic-based latent graph
        self.latent_tree_att = StructuredAtt(args)
        self.syn_dynamic_gcn_encoder = GCN_for_latentTree(args)
        self.dep_size = args.dep_type_size
        self.dep_embed_dim = args.dep_embed_dim
        self.edge_embeddings = nn.Embedding(num_embeddings=self.dep_size,
                                            embedding_dim=self.dep_embed_dim,
                                            padding_idx=0)
        self.dep_embed_linear = nn.Linear(args.dep_embed_dim, args.dynamic_tree_attn_head)

        # segment-aware semantic graph
        self.local_att = LocalAtt(args)
        self.sem_gcn_encoder = GCN_for_segment_aware_graph(args)


        # self-adaptive aggregation
        self.model_dim = args.hidden_dim
        self.h = args.fusion_attention_heads
        self.d_k = args.hidden_dim // args.attention_heads
        self.sem_fusion_syn = MyMultiHeadAttention(self.args, self.h, self.model_dim, self.d_k)
        self.syn_fusion_sem = MyMultiHeadAttention(self.args, self.h, self.model_dim, self.d_k)


        # semantic concatenate syntactic
        self.dropout = nn.Dropout(0.1)
        self.feedforword = nn.Sequential(
            nn.Linear(args.hidden_dim * 2, 2048),
            nn.ReLU(),
            self.dropout,
            nn.Linear(2048, args.hidden_dim),
            self.dropout
        )
        self.W_Chanel = nn.Linear(args.hidden_dim, 1)

        # graph training balance
        self.lossLinear = nn.Linear(args.hidden_dim, 1)
        self.temp_Parameter = nn.Parameter(torch.FloatTensor(args.hidden_dim))




    # -------------------------------------------------------------------------------------------

    def forward(self, inputs):

        length, bert_lengths, word_mapback, aspect_token,mask, bert_sequence, bert_segments_ids, adj, con_spans, dep_type_matrix = inputs


        ###############################################################
        # 1. contextual encoder
        bert_outputs = self.context_encoder(bert_sequence, token_type_ids=bert_segments_ids)
        bert_out, bert_pooler_out = bert_outputs.last_hidden_state, bert_outputs.pooler_output
        bert_out = self.layer_drop(bert_out)

        # rm [CLS]
        bert_seq_indi = ~sequence_mask(bert_lengths).unsqueeze(dim=-1)
        bert_out = bert_out[:, 1:max(bert_lengths) + 1, :] * bert_seq_indi.float()
        word_mapback_one_hot = (F.one_hot(word_mapback).float() * bert_seq_indi.float()).transpose(1, 2)
        bert_out = torch.bmm(word_mapback_one_hot.float(), self.dense(bert_out))

        # average
        wnt = word_mapback_one_hot.sum(dim=-1)
        wnt.masked_fill_(wnt == 0, 1)
        bert_out = bert_out / wnt.unsqueeze(dim=-1)


        ###############################################################
        # 2. graph encoder
        key_padding_mask_original = sequence_mask(length)  # [B, seq_len]

        key_padding_mask = key_padding_mask_original.long()
        key_padding_mask = key_padding_mask * -10000


        root_mask = ~key_padding_mask_original
        root_mask = root_mask.long()

        # from phrase(span) to form mask
        B, N, L = con_spans.shape
        span_matrix = get_span_matrix_4D(con_spans.transpose(0, 1))
        span_matrix = span_matrix.bool()


        # segment-aware semantic
        p_local_attn, span_loss = self.local_att(bert_out, bert_out, key_padding_mask, span_matrix, aspect_mask=None)
        sem_graph_out = self.sem_gcn_encoder(bert_out, p_local_attn.to(torch.float32), None, is_multi_head=True)



        # syntactic-based latent
        dep_type_adj = self.edge_embeddings(dep_type_matrix.long())
        dep_type_adj = self.dep_embed_linear(dep_type_adj).squeeze(-1)
        dep_type_adj = dep_type_adj.transpose(1, 3)
        context, d, d0, loss_root = self.latent_tree_att(bert_out, key_padding_mask, mask, root_mask,
                                                                dep_type_adj= dep_type_adj,
                                                                is_multi_head=True)
        syn_graph_out = self.syn_dynamic_gcn_encoder(bert_out, d, None, is_multi_head=True)


        # graph balance
        same_loss = torch.norm(p_local_attn-d).to(self.args.device)
        different_loss = (B/torch.norm(p_local_attn - d)).to(self.args.device)
        loss_gate = torch.sigmoid(self.lossLinear(self.temp_Parameter))
        graph_balance = loss_gate * different_loss + (1 - loss_gate) * same_loss
        #————————————————————————————————————————————————


        # self-adaptive aggregation
        # semantic concatenate syntactic
        sem_concat_syn_graph_out = self.feedforword(torch.cat((sem_graph_out, syn_graph_out), dim=-1))

        # semantic<--syntactic
        sem_fusion_syn_graph_out, _ = self.sem_fusion_syn(sem_graph_out, syn_graph_out, syn_graph_out,key_padding_mask)
        # syntactic<--semantic
        syn_fusion_sem_graph_out, _ = self.syn_fusion_sem(syn_graph_out, sem_graph_out, sem_graph_out,key_padding_mask)



        ###############################################################
        # 3. fusion
        asp_wn = mask.sum(dim=1).unsqueeze(-1)  # aspect words num

        mask_simple = mask.unsqueeze(-1).repeat(1, 1, self.args.hidden_dim)  # mask for h
        # h_t
        bert_enc_outputs = (bert_out * mask_simple).sum(dim=1) / asp_wn


        sem_fusion_syn_graph_out = (sem_fusion_syn_graph_out * mask_simple).sum(dim=1) / asp_wn  # mask h
        if self.data_name != 'laptop':
            sem_fusion_syn_graph_out = sem_fusion_syn_graph_out + bert_enc_outputs
        a1 = F.relu(self.W_Chanel(sem_fusion_syn_graph_out))

        syn_fusion_sem_graph_out = (syn_fusion_sem_graph_out * mask_simple).sum(dim=1) / asp_wn  # mask h
        if self.data_name != 'laptop':
            syn_fusion_sem_graph_out = syn_fusion_sem_graph_out + bert_enc_outputs
        a2 = F.relu(self.W_Chanel(syn_fusion_sem_graph_out))


        sem_concat_syn_graph_out = (sem_concat_syn_graph_out * mask_simple).sum(dim=1) / asp_wn  # mask h
        if self.data_name != 'laptop':
            sem_concat_syn_graph_out = sem_concat_syn_graph_out + bert_enc_outputs
        a3 = F.relu(self.W_Chanel(sem_concat_syn_graph_out))
        a_all = torch.cat((a1, a2, a3), dim=-1)
        a_all = F.softmax(a_all, dim=-1)
        a1_rate = a_all[:, 0].unsqueeze(-1)
        a2_rate = a_all[:, 1].unsqueeze(-1)
        a3_rate = a_all[:, 2].unsqueeze(-1)
        if self.data_name == 'twitter':
            representation = torch.cat(
                (a1_rate * (sem_fusion_syn_graph_out + bert_enc_outputs),
                 a2_rate * (syn_fusion_sem_graph_out + bert_enc_outputs),
                 a3_rate * (sem_concat_syn_graph_out + bert_enc_outputs)),
                dim=-1)
        else:
            representation = torch.cat(
                (a1_rate * (sem_fusion_syn_graph_out),
                 a2_rate * (syn_fusion_sem_graph_out),
                 a3_rate * (sem_concat_syn_graph_out)),
                dim=-1)
        as_features = representation


        return as_features, span_loss, loss_root, graph_balance


def sequence_mask(lengths, max_len=None):
    """
    create a boolean mask from sequence length `[batch_size,  seq_len]`
    """
    batch_size = lengths.numel()
    max_len = max_len or lengths.max()
    a = torch.arange(0, max_len, device=lengths.device).type_as(lengths).unsqueeze(0).expand(
        batch_size, max_len
    ) >= (lengths.unsqueeze(1))
    return a


def get_span_matrix_4D(span_list, rm_loop=False, max_len=None):
    '''
    span_list: [N,B,L]
    return span:[N,B,L,L]
    '''
    # [N,B,L]
    N, B, L = span_list.shape
    span = get_span_matrix_3D(span_list.contiguous().view(-1, L), rm_loop, max_len).contiguous().view(N, B, L, L)
    return span


def get_span_matrix_3D(span_list, rm_loop=False, max_len=None):
    # [N,L]
    origin_dim = len(span_list.shape)
    if origin_dim == 1:  # [L]
        span_list = span_list.unsqueeze(dim=0)
    N, L = span_list.shape
    if max_len is not None:
        L = min(L, max_len)
        span_list = span_list[:, :L]
    span = span_list.unsqueeze(dim=-1).repeat(1, 1, L)
    span = span * (span.transpose(-1, -2) == span)
    if rm_loop:
        span = span * (~torch.eye(L).bool()).unsqueeze(dim=0).repeat(N, 1, 1)
        span = span.squeeze(dim=0) if origin_dim == 1 else span  # [N,L,L]
    return span



class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2
