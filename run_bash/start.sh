#!/bin/bash
exp_path=log/Result

if [ ! -d "$exp_path" ]; then
  echo "making new dir.."
  mkdir -p "$exp_path"
fi

DATE=$(date +%Y-%m-%d-%H_%M_%S)
CUDA_VISIBLE_DEVICES=0 python3 -u ../train.py \
	--data_dir ../data/V2/Laptops \
	--vocab_dir ../data/V2/Laptops \
	--data_name laptop \
	--batch_size 16 \
	--alpha 0.1 \
	--beta 0.5 \
	--input_dropout 0.2 \
	--layer_dropout 0.1 \
	--gcn_dropout 0.1 \
	--max_len 100 \
	--lr 2e-5 \
	--seed 1 \
	--attention_heads 4 \
	--max_num_spans 4 \
	--num_epoch 20 2>&1 | tee $exp_path/training_$DATE.log

DATE=$(date +%Y-%m-%d-%H_%M_%S)
CUDA_VISIBLE_DEVICES=0 python3 -u ../train.py \
	--data_dir ../data/V2/Restaurants \
	--vocab_dir ../data/V2/Restaurants \
	--data_name restaurant \
	--batch_size 16 \
	--alpha 0.1 \
	--beta 0.45 \
	--input_dropout 0.2 \
	--layer_dropout 0.1 \
	--gcn_dropout 0.1 \
	--max_len 100 \
	--lr 2e-5 \
	--seed 1 \
	--attention_heads 4 \
	--max_num_spans 4 \
	--num_epoch 20 2>&1 | tee $exp_path/training_$DATE.log



