#!/bin/bash

python3 ./code/train_s2s.py \
    --output_folder output/ \
    --epochs 300 --noise 0.10 --batch_size 6000 --mask 0.9 --max_length 500 --ipa_layer 3\
    --data_jsonl /pubhome/bozhang/data/tmpnn_v8.jsonl \
    --split_json /pubhome/bozhang/data/tmpnn_v8.json \
    --job_name tmpnn-ipa-v3 \
    --description "no noise to frame only to distance map + mix_data" \
    --num_tags 6