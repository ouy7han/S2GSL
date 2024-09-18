import os 
import json
import torch 
import numpy as np 
from transformers import BertTokenizer

import copy 
import random 
import itertools 
from itertools import chain

from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import _SingleProcessDataLoaderIter, _MultiProcessingDataLoaderIter
from spans import *
import pickle
from dep_parser import DepInstanceParser




class ABSA_Dataset(Dataset):
    def __init__(self, args, file_name, vocab, tokenizer,flag):
        super().__init__()

        # load raw data
        with open(file_name,'r',encoding='utf-8') as f:
            raw_data = json.load(f)

            if args.need_preprocess:
                raw_data = self.process_raw(raw_data)
                new_file_name = file_name.replace('.json','_con.json')
                with open(new_file_name, 'w', encoding='utf-8') as f:
                    json.dump(raw_data,f)
                print('Saving to:', new_file_name)

        # 加载依赖关系图信息
        all_dep_info = load_depfile(os.path.join(args.data_dir,'{}.txt.dep'.format(flag)))

        dependency_type_dict=prepare_type_dict(args.data_dir)


        self.data = self.process(raw_data, vocab, args, tokenizer,all_dep_info,dependency_type_dict)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

    
    def process_raw(self, data):
        # get parserd data
        # we already provide here
        pass
    

    def process(self, data, vocab, args, tokenizer,all_dep_info,dependency_type_dict):
        token_vocab = vocab['token']
        pol_vocab = vocab['polarity']

        processed = []
        max_len = args.max_len 
        CLS_id = tokenizer.convert_tokens_to_ids(["[CLS]"])
        SEP_id = tokenizer.convert_tokens_to_ids(["[SEP]"])
        sub_len = len(args.special_token)

        span_num = []

        i=0
        for d,dep_info in zip(data,all_dep_info):
            graph_id=i
            tok = list(d['token'])
            if args.lower:
                tok = [t.lower() for t in tok]
            
            text_raw_bert_indices, word_mapback, _ = text2bert_id(tok, tokenizer)

            text_raw_bert_indices = text_raw_bert_indices[:max_len]
            word_mapback = word_mapback[:max_len]

            length = word_mapback[-1] + 1

            # tok = tok[:length]
            bert_length = len(word_mapback)

            dep_head = list(d['dep_head'])[:length]

            # map2id 
            # tok = [token_vocab.stoi.get(t, token_vocab.unk_index) for t in tok]
            
            # con
            con_head = d['con_head']
            con_mapnode = d['con_mapnode']
            con_path_dict, con_children = get_path_and_children_dict(con_head)
            mapback = [ idx for idx ,word in enumerate(con_mapnode) if word[-sub_len: ]!= args.special_token]

            layers, influence_range, node2layerid = form_layers_and_influence_range(con_path_dict, mapback)

            spans = form_spans(layers, influence_range, length, con_mapnode)

            adj_i_oneshot = head_to_adj_oneshot(dep_head, length, d['aspects'])

            cd_adj = np.ones((length,length))
            if args.con_dep_conditional:
                father = 1
                if father in con_children and [con_mapnode[node] for node in con_children[father]].count('S[N]') > 1 and con_mapnode[father] == 'S[N]':
                    cd_span = spans[node2layerid[father]+1]
                    cd_adj = get_conditional_adj(father, length, cd_span, con_children, con_mapnode)

            adj_i_oneshot = adj_i_oneshot * cd_adj

            dep_instance_parser=DepInstanceParser(basicDependencies=dep_info,tokens=tok)
            dep_type_matrix=dep_instance_parser.get_first_order()

            
            # aspect-specific
            bert_sequence_list = []
            bert_segments_ids_list = []
            label_list = []
            aspect_indi_list = []
            aspect_token_list = []
            select_spans_list = []

            for aspect in d['aspects']:
                asp = list(aspect['term'])
                asp_bert_ids, _, _ = text2bert_id(asp, tokenizer)
                bert_sequence = CLS_id  + text_raw_bert_indices +  SEP_id + asp_bert_ids + SEP_id
                bert_segments_ids = [0] * (bert_length + 2) + [1] * (len(asp_bert_ids ) +1)

                bert_sequence = bert_sequence[:max_len+3]
                bert_segments_ids = bert_segments_ids[:max_len+3]

                label = aspect['polarity']

                aspect_indi = [0] * length 

                for pidx in range(aspect['from'], aspect['to']):
                    aspect_indi[pidx] = 1
                
                label = pol_vocab.stoi.get(label)

                aspect_range = list(range(mapback[aspect['from']], mapback[aspect['to']-1] + 1))

                con_lca = find_inner_LCA(con_path_dict, aspect_range)


                select_spans, span_indications = form_aspect_related_spans(con_lca, spans, con_mapnode, node2layerid, con_path_dict)

                # span_num.append(len(select_spans))
                #
                # averge_layers=np.array(span_num).sum()/len(span_num)

                select_spans = select_func(select_spans, args.max_num_spans, length)

                select_spans = [[ x+ 1 for x in span] for span in select_spans] 


                label_list.append(label)
                aspect_indi_list.append(aspect_indi)
                aspect_token_list.append(asp_bert_ids)
                bert_sequence_list.append(bert_sequence)
                bert_segments_ids_list.append(bert_segments_ids)

                select_spans_list.append(select_spans)





            processed += [
                (
                    length, bert_length, word_mapback,
                    adj_i_oneshot,
                    # aspect-specific
                    bert_sequence_list, bert_segments_ids_list, aspect_indi_list, aspect_token_list,select_spans_list,
                    # label
                    label_list,
                    #dep_type_matrix
                    dep_type_matrix,
                    #dependency_type_dict
                    dependency_type_dict
                )
            ]
            i=i+1

        
        return processed 
                    

def ABSA_collate_fn(batch):
    batch_size = len(batch)
    batch = list(zip(*batch))

    lens = batch[0]

    (length_, bert_length_, word_mapback_,
    adj_i_oneshot_,
    bert_sequence_list_, bert_segments_ids_list_,
    aspect_indi_list_, aspect_token_list_,select_spans_list_,
    label_list_,dep_type_matrix_,dependency_type_dict_) = batch

    max_lens = max(lens)
    dep_label_map=dependency_type_dict_[0]

    #str=dep_type_matrix_[0][0][3]

    
    length = torch.LongTensor(length_)
    bert_length = torch.LongTensor(bert_length_)
    word_mapback = get_long_tensor(word_mapback_, batch_size)

    adj_oneshot = np.zeros((batch_size, max_lens, max_lens), dtype=np.float32)

    for idx in range(batch_size):
        mlen = adj_i_oneshot_[idx].shape[0]
        adj_oneshot[idx,:mlen,:mlen] = adj_i_oneshot_[idx]


    adj_oneshot = torch.FloatTensor(adj_oneshot)

    map_AS = [[idx] * len(a_i) for idx, a_i in enumerate(bert_sequence_list_)]
    map_AS_idx = [range(len(a_i)) for a_i in bert_sequence_list_]

    # add_pre = np.array([0] + [len(m) for m in map_AS[:-1]]).cumsum()
    
    map_AS = torch.LongTensor([m for m_list in map_AS for m in m_list])
    map_AS_idx = torch.LongTensor([m for m_list in map_AS_idx for m in m_list])

    as_batch_size = len(map_AS)

    bert_sequence = [p for p_list in bert_sequence_list_ for p in p_list]
    bert_sequence = get_long_tensor(bert_sequence, as_batch_size)

    bert_segments_ids = [p for p_list in bert_segments_ids_list_ for p in p_list]
    bert_segments_ids = get_long_tensor(bert_segments_ids, as_batch_size)

    aspect_indi = [p for p_list in aspect_indi_list_ for p in p_list]
    aspect_indi = get_long_tensor(aspect_indi, as_batch_size)

    # aspect_token_list
    aspect_token_list = [p for p_list in aspect_token_list_ for p in p_list]
    aspect_token_list = get_long_tensor(aspect_token_list, as_batch_size)

    con_spans_list = [p for p_list in select_spans_list_ for p in p_list]
    max_num_spans = max([len(p) for p in con_spans_list])
    con_spans = np.zeros((as_batch_size, max_num_spans, max_lens), dtype=np.int64)
    for idx in range(as_batch_size):
        mlen = len(con_spans_list[idx][0])
        con_spans[idx,:,:mlen] = con_spans_list[idx]
    
    con_spans = torch.LongTensor(con_spans)

    # label
    label = torch.LongTensor([sl for sl_list in label_list_ for sl in sl_list if isinstance(sl, int)])




    def get_adj_with_value_matrix(dep_type_matrix):
        final_dep_type_matrix=np.zeros((batch_size, max_lens, max_lens), dtype=int)
        for idx in range(batch_size):
            mlen = len(dep_type_matrix[idx])
            for pi in range(mlen):
                for pj in range(mlen):
                    final_dep_type_matrix[idx][pi][pj] = dep_label_map[dep_type_matrix[idx][pi][pj]]

        return final_dep_type_matrix

    dep_type_matrix=get_adj_with_value_matrix(dep_type_matrix_)
    dep_type_matrix=torch.IntTensor(dep_type_matrix)




    
    return (
        length, bert_length, word_mapback, adj_oneshot,
        map_AS, map_AS_idx,
        bert_sequence, bert_segments_ids,
        aspect_indi, aspect_token_list,con_spans,
        dep_type_matrix,
        label
    )


def text2bert_id(token, tokenizer):
    re_token = []
    word_mapback = []
    word_split_len = []
    for idx, word in enumerate(token):
        temp = tokenizer.tokenize(word)
        re_token.extend(temp)
        word_mapback.extend([idx] * len(temp))
        word_split_len.append(len(temp))
    re_id = tokenizer.convert_tokens_to_ids(re_token)
    return re_id ,word_mapback, word_split_len

class ABSA_DataLoader(DataLoader):
    def __init__(self, dataset, sort_key, sort_bs_num=None, is_shuffle=True, **kwargs):
        '''
        :param dataset: Dataset object
        :param sort_idx: sort_function
        :param sort_bs_num: sort range; default is None(sort for all sequence)
        :param is_shuffle: shuffle chunk , default if True
        :return:
        '''
        assert isinstance(dataset.data, list)
        super().__init__(dataset,**kwargs)
        self.sort_key = sort_key
        self.sort_bs_num = sort_bs_num
        self.is_shuffle = is_shuffle

    def __iter__(self):
        if self.is_shuffle:
            self.dataset.data = self.block_shuffle(self.dataset.data, self.batch_size, self.sort_bs_num, self.sort_key, self.is_shuffle)

        if self.num_workers == 0:
            return _SingleProcessDataLoaderIter(self)
        else:
            return _MultiProcessingDataLoaderIter(self)

    @staticmethod
    def block_shuffle(data, batch_size, sort_bs_num, sort_key, is_shuffle):
        # sort
        random.shuffle(data)
        data = sorted(data, key = sort_key) # 先按照长度排序
        batch_data = [data[i : i + batch_size] for i in range(0,len(data),batch_size)]
        batch_data = [sorted(batch, key = sort_key) for batch in batch_data]
        if is_shuffle:
            random.shuffle(batch_data)
        batch_data = list(chain(*batch_data))
        return batch_data

def get_long_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded LongTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.LongTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, : len(s)] = torch.LongTensor(s)
    return tokens


def get_float_tensor(tokens_list, batch_size):
    """ Convert list of list of tokens to a padded FloatTensor. """
    token_len = max(len(x) for x in tokens_list)
    tokens = torch.FloatTensor(batch_size, token_len).fill_(0)
    for i, s in enumerate(tokens_list):
        tokens[i, : len(s)] = torch.FloatTensor(s)
    return tokens


def sort_all(batch, lens):
    """ Sort all fields by descending order of lens, and return the original indices. """
    unsorted_all = [lens] + [range(len(lens))] + list(batch)
    sorted_all = [list(t) for t in zip(*sorted(zip(*unsorted_all), reverse=True))]
    return sorted_all[2:], sorted_all[1]


def get_dep_labels(data_dir,direct=False):
    dep_labels=["self_loop"]
    dep_type_path=os.path.join(data_dir,"dep_type.json")
    with open(dep_type_path,'r') as f:
        dep_types=json.load(f)
        for label in dep_types:
            if direct:
                dep_labels.append("{}_in".format(label))
                dep_labels.append("{}_out".format(label))
            else:
                dep_labels.append(label)
    return dep_labels


def prepare_type_dict(data_dir):
    dep_type_list=get_dep_labels(data_dir)
    types_dict={"none":0}
    for dep_type in dep_type_list:
        types_dict[dep_type]=len(types_dict)

    return types_dict



def load_depfile(filename):
    data=[]
    with open(filename,'r') as f:
        dep_info=[]
        for line in f:
            line=line.strip()
            if len(line)>0:
                items=line.split("\t")
                dep_info.append({
                    "governor":int(items[0]),
                    "dependent":int(items[1]),
                    "dep":items[2],
                })
            else:
                if len(dep_info)>0:
                    data.append(dep_info)
                    dep_info=[]
        if len(dep_info)>0:
            data.append(dep_info)
            dep_info = []

    return data

