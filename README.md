# S2GSL


[//]: # (This repository contains Pytorch implementation for "[S2GSL: Incorporating Segment to Syntactic Enhanced Graph Structure)

[//]: # (Learning for Aspect-based Sentiment Analysis]" )

## Requirements
```
python = 3.7
pytorch >= 1.7
transformers >= 4.3.3 
```

## Get Start
1. Prepare data
   
   We follow the dataset setting in https://github.com/muyeby/RGAT-ABSA, and provide the parsed data at directory **data/V2** (or, you can use  ***preprocess_file*** function in ```parse_tree.py``` to preprocess on your own)

2. Download pre-trained BERT-Base English from [HuggingFace](https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-chinese.tar.gz).
   Download [stanford corenlp](https://drive.google.com/drive/folders/12epkro2pU8ICURm9eWMjv7uMbWC0wQiK?usp=drive_link) and unzip to **data/V2**.

3. Training
   
   ```
   bash run_bash/start.sh
   ```

[//]: # (## Citation)

[//]: # (**Please kindly cite our paper if this paper and the code are helpful.**)

[//]: # (```)

[//]: # (@inproceedings{liang-etal-2022-bisyn,)

[//]: # (    title = "{B}i{S}yn-{GAT}+: Bi-Syntax Aware Graph Attention Network for Aspect-based Sentiment Analysis",)

[//]: # (    author = "Liang, Shuo  and)

[//]: # (      Wei, Wei  and)

[//]: # (      Mao, Xian-Ling  and)

[//]: # (      Wang, Fei  and)

[//]: # (      He, Zhiyong",)

[//]: # (    booktitle = "Findings of the Association for Computational Linguistics: ACL 2022",)

[//]: # (    month = may,)

[//]: # (    year = "2022",)

[//]: # (    address = "Dublin, Ireland",)

[//]: # (    publisher = "Association for Computational Linguistics",)

[//]: # (    url = "https://aclanthology.org/2022.findings-acl.144",)

[//]: # (    pages = "1835--1848",)

[//]: # (    abstract = "Aspect-based sentiment analysis &#40;ABSA&#41; is a fine-grained sentiment analysis task that aims to align aspects and corresponding sentiments for aspect-specific sentiment polarity inference. It is challenging because a sentence may contain multiple aspects or complicated &#40;e.g., conditional, coordinating, or adversative&#41; relations. Recently, exploiting dependency syntax information with graph neural networks has been the most popular trend. Despite its success, methods that heavily rely on the dependency tree pose challenges in accurately modeling the alignment of the aspects and their words indicative of sentiment, since the dependency tree may provide noisy signals of unrelated associations &#40;e.g., the {``}conj{''} relation between {``}great{''} and {``}dreadful{''} in Figure 2&#41;. In this paper, to alleviate this problem, we propose a Bi-Syntax aware Graph Attention Network &#40;BiSyn-GAT+&#41;. Specifically, BiSyn-GAT+ fully exploits the syntax information &#40;e.g., phrase segmentation and hierarchical structure&#41; of the constituent tree of a sentence to model the sentiment-aware context of every single aspect &#40;called intra-context&#41; and the sentiment relations across aspects &#40;called inter-context&#41; for learning. Experiments on four benchmark datasets demonstrate that BiSyn-GAT+ outperforms the state-of-the-art methods consistently.",)

[//]: # (})

[//]: # ()
[//]: # (```)
