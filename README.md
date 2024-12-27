# S2GSL


This repository contains Pytorch implementation for "S2GSL: Incorporating Segment to Syntactic Enhanced Graph Structure Learning for Aspect-based Sentiment Analysis" （ACL2024）



## Requirements
```
python = 3.8.10
torch == 2.0.0+cu118
transformers == 4.3.3 
packaging == 21.3
scikit-learn == 1.3.2
```

## Get Start
1. Prepare data
   
   We follow the dataset setting in https://github.com/CCIIPLab/BiSyn_GAT_plus, and provide the parsed data at directory **data/V2** (or, you can use  ***preprocess_file*** function in ```parse_tree.py``` to preprocess on your own)

2. Download pre-trained [BERT-Base English](https://drive.google.com/drive/folders/1sbwkL3NQ8c7I0vugAO-HuLmg5SiO2iPS?usp=sharing) to **pretrain_model/bert-base-uncased**
   
   Download [stanford corenlp](https://drive.google.com/drive/folders/12epkro2pU8ICURm9eWMjv7uMbWC0wQiK?usp=drive_link)  to **data/V2/stanford-corenlp**.


3. Train
   
   ```
   bash run_bash/start.sh
   ```
   
# Credit
   The code and datasets in this repository is based on [Bisyn-gat+](https://github.com/CCIIPLab/BiSyn_GAT_plus).

## Citation

**Please kindly cite our paper if this paper and the code are helpful.**

```
@inproceedings{chen-etal-2024-s2gsl,
    title = "{S}$^2${GSL}: Incorporating Segment to Syntactic Enhanced Graph Structure Learning for Aspect-based Sentiment Analysis",
    author = "Chen, Bingfeng  and
      Ouyang, Qihan  and
      Luo, Yongqi  and
      Xu, Boyan  and
      Cai, Ruichu  and
      Hao, Zhifeng",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.721",
    pages = "13366--13379",
    abstract = "Previous graph-based approaches in Aspect-based Sentiment Analysis(ABSA) have demonstrated impressive performance by utilizing graph neural networks and attention mechanisms to learn structures of static dependency trees and dynamic latent trees. However, incorporating both semantic and syntactic information simultaneously within complex global structures can introduce irrelevant contexts and syntactic dependencies during the process of graph structure learning, potentially resulting in inaccurate predictions. In order to address the issues above, we propose S$^2$GSL, incorporating Segment to Syntactic enhanced Graph Structure Learning for ABSA. Specifically, S$^2$GSL is featured with a segment-aware semantic graph learning and a syntax-based latent graph learning enabling the removal of irrelevant contexts and dependencies, respectively. We further propose a self-adaptive aggregation network that facilitates the fusion of two graph learning branches, thereby achieving complementarity across diverse structures. Experimental results on four benchmarks demonstrate the effectiveness of our framework.",
}
```
