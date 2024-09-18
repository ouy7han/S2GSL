import os
import argparse
import json
from corenlp import StanfordCoreNLP
from tqdm import tqdm

FULL_MODEL='./stanford-corenlp-full-2018-10-05'

def request_features_from_stanford(data_dir,flag):
    data_path=os.path.join(data_dir,flag+'_con_new.json')
    if not os.path.exists(data_path):
        print("{} not exist".format(data_path))
        return

    token_str=[]
    with open(data_path,'r',encoding='utf-8') as f:
        raw_data=json.load(f)
        for d in raw_data:
            tok=list(d['token'])
            tok=[t.lower() for t in tok]
            token_str.append(tok)

        all_data=[]
        with StanfordCoreNLP(FULL_MODEL,lang='en') as nlp:
            for sentence in tqdm(token_str):
                props = {'timeout': '5000000', 'annotators': 'pos, parse, depparse', 'tokenize.whitespace': 'true',
                         'ssplit.eolonly': 'true', 'pipelineLanguage': 'en', 'outputFormat': 'json'}
                results = nlp.annotate(' '.join(sentence), properties=props)
                all_data.append(results)

        with open(os.path.join(data_dir,flag+'.txt.dep'),'w',encoding='utf8') as fout_dep:
            for data in all_data:
                for dep_info in data["sentences"][0]["basicDependencies"]:
                    fout_dep.write("{}\t{}\t{}\n".format(dep_info["governor"],dep_info["dependent"],dep_info["dep"]))
                fout_dep.write("\n")


def get_dep_type_dict(data_dir):
    dep_type_set=set()
    for flag in ["train","valid","test"]:
        data_path=os.path.join(data_dir,flag+'.txt.dep')
        if not os.path.exists(data_path):
            continue
        with open(data_path,'r',encoding='utf-8') as f:
            for line in f:
                if line == "\n":
                    continue
                governor,dependent,dep_type=line.strip().split("\t")
                dep_type_set.add(dep_type)

    save_path=os.path.join(data_dir,"dep_type.json")
    with open(save_path,'w',encoding='utf-8') as f:
        json.dump(list(dep_type_set),f,ensure_ascii=False)


def get_args():
    parser=argparse.ArgumentParser()
    parser.add_argument("--data_path",default="./Tweets/",type=str,required=True)
    args=parser.parse_args()

    return args

if __name__=='__main__':
    args=get_args()
    for flag in ["train","valid","test"]:
        request_features_from_stanford(args.data_path,flag)
    get_dep_type_dict(args.data_path)
