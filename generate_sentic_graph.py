import numpy as np
import pickle
import json

def load_sentic_word():

    path='./senticNet/senticnet_word_80.txt'
    senticNet={}
    fp=open(path,'r')
    for line in fp:
        line=line.strip()
        if not line:
            continue
        word,sentic=line.split('\t')
        senticNet[word]=sentic
    fp.close()
    return senticNet


def dependency_adj_matrix(text,aspect,senticNet):
    seq_len=len(text)
    matrix=np.zeros((seq_len,seq_len)).astype('float32')

    # for i in range(seq_len):
    #     word=text[i]
    #     if word in senticNet:
    #         #sentic=float(senticNet[word])+1.0
    #         #a=senticNet[word]
    #         sentic = abs(float(senticNet[word]))
    #     else:
    #         sentic=0
    #     for p in range(len(aspect)):
    #         if word in aspect[p]:
    #             sentic += 1.0
    #     # if word in aspect:
    #     #     sentic+=1.0
    #     for j in range(seq_len):
    #         matrix[i][j]+=sentic
    #         matrix[j][i]+=sentic
    # for i in range(seq_len):
    #     if matrix[i][i]==0:
    #         matrix[i][i]=1


    aspect_tok_list=[]
    for p in range(len(aspect)):
        for q in range(len(aspect[p])):
            aspect_tok_list.append(aspect[p][q])

    for i in range(seq_len):
        word=text[i]
        for p in range(len(aspect)):
            if word in aspect[p]:
                for j in range(seq_len):
                    context_word=text[j]
                    if context_word in senticNet:
                        #for q in range(len(aspect)):
                            if context_word not in aspect_tok_list:
                                sentic = float(senticNet[context_word])
                            else:
                                sentic=0
                    else:
                        sentic=0
                    matrix[i][j]=sentic
                    matrix[j][i] = sentic

    # for i in range(seq_len):
    #     if matrix[i][i]==0:
    #         matrix[i][i]=1



    return matrix

def generate_sentic_word_list(text,senticNet):
    seq_len=len(text)
    sentic_word_list=np.zeros(seq_len).astype('float32')

    for i in range(seq_len):
        word=text[i]
        if word in senticNet:
            sentic=float(senticNet[word])
        else:
            sentic=0
        sentic_word_list[i]=sentic

    return  sentic_word_list

def process(filename):
    senticNet=load_sentic_word()
    # fin=open(filename,'r',encoding='utf-8')
    # lines=fin.readlines()
    # fin.close()
    with open(filename,'r',encoding='utf-8') as f:
        raw_data=json.load(f)
    idx2graph={}
    fout=open(filename+'.sentic','wb')
    # for i in range(0,len(lines),3):
    #     text_left,_,text_right=[s.lower().strip() for s in lines[i].partition("$T$")]
    #     aspect=lines[i+1].lower().strip()
    #     adj_matrix=dependency_adj_matrix(text_left+' '+aspect+' '+text_right,aspect,senticNet)
    #     idx2graph[i]=adj_matrix
    i = 0
    for d in raw_data:
        aspects = []
        tok = list(d['token'])
        for aspect in d['aspects']:
            asp=list(aspect['term'])
            aspects.append(asp)
        adj_matrix=dependency_adj_matrix(tok,aspects,senticNet)
        idx2graph[i] = adj_matrix
        i=i+1
    pickle.dump(idx2graph,fout)
    print('done!!!',filename)
    fout.close()

    sentic_word_list={}
    fout_word_list=open(filename+'.sentic_word_list','wb')
    j=0
    for d in raw_data:
        tok=list(d['token'])
        word_list=generate_sentic_word_list(tok,senticNet)
        sentic_word_list[j]=word_list
        j=j+1
    pickle.dump(sentic_word_list,fout_word_list)
    print('word_list_done!!!',filename)
    fout_word_list.close()


if __name__=='__main__':
    process('./data/V2/Laptops/train_con_new.json')
    process('./data/V2/Laptops/valid_con_new.json')
    process('./data/V2/Laptops/test_con_new.json')
    process('./data/V2/MAMS/train_con_new.json')
    process('./data/V2/MAMS/valid_con_new.json')
    process('./data/V2/MAMS/test_con_new.json')
    process('./data/V2/Restaurants/train_con_new.json')
    process('./data/V2/Restaurants/valid_con_new.json')
    process('./data/V2/Restaurants/test_con_new.json')
    process('./data/V2/Tweets/train_con_new.json')
    process('./data/V2/Tweets/valid_con_new.json')
    process('./data/V2/Tweets/test_con_new.json')