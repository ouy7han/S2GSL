#!/bin/bash
exp_path=log/result

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
	--alpha 0.65 \
	--beta 0.6 \
	--input_dropout 0.2 \
	--layer_dropout 0.1 \
	--gcn_dropout 0.1 \
	--max_len 100 \
	--lr 2e-5 \
	--seed 1 \
	--attention_heads 4 \
	--max_num_spans 4 \
	--num_epoch 20 2>&1 | tee $exp_path/laptop_$DATE.log

DATE=$(date +%Y-%m-%d-%H_%M_%S)
CUDA_VISIBLE_DEVICES=0 python3 -u ../train.py \
	--data_dir ../data/V2/Restaurants \
	--vocab_dir ../data/V2/Restaurants \
	--data_name restaurant \
	--batch_size 16 \
	--alpha 0.3 \
	--beta 0.3 \
	--input_dropout 0.2 \
	--layer_dropout 0.1 \
	--gcn_dropout 0.1 \
	--max_len 100 \
	--lr 2e-5 \
	--seed 1 \
	--attention_heads 4 \
	--max_num_spans 4 \
	--num_epoch 20 2>&1 | tee $exp_path/restaurant_$DATE.log

DATE=$(date +%Y-%m-%d-%H_%M_%S)
CUDA_VISIBLE_DEVICES=0 python3 -u ../train.py \
	--data_dir ../data/V2/Tweets \
	--vocab_dir ../data/V2/Tweets \
	--data_name twitter \
	--batch_size 16 \
	--alpha 0.035 \
	--beta 0.95 \
	--input_dropout 0.2 \
	--layer_dropout 0.1 \
	--gcn_dropout 0.1 \
	--max_len 100 \
	--lr 2e-5 \
	--seed 1 \
	--attention_heads 4 \
	--max_num_spans 4 \
	--num_epoch 20 2>&1 | tee $exp_path/twitter_$DATE.log

DATE=$(date +%Y-%m-%d-%H_%M_%S)
CUDA_VISIBLE_DEVICES=0 python3 -u ../train.py \
	--data_dir ../data/V2/MAMS \
	--vocab_dir ../data/V2/MAMS \
	--data_name mams \
	--batch_size 16 \
	--alpha 0.7 \
	--beta 0.4 \
	--input_dropout 0.2 \
	--layer_dropout 0.1 \
	--gcn_dropout 0.1 \
	--max_len 100 \
	--lr 2e-5 \
	--seed 1 \
	--attention_heads 4 \
	--max_num_spans 4 \
	--num_epoch 20 2>&1 | tee $exp_path/mams_$DATE.log