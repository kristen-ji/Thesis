#!/bin/bash

model_id="russwang/ThinkLite-VL-7B"
datasets=("mathvista" "mathvision" "mmmu" "mmstar" "mathverse" "ai2d")


for dataset in "${datasets[@]}"; do
    # Ensure nested directory exists if model_id contains '/'
    mkdir -p "./eval_files/${dataset}/${model_id%/*}"
    datapath="./eval_files/${dataset}/${model_id}"
    model_script="eval/${dataset}_score.py"

    if [ ! -f "${datapath}.jsonl" ]; then
        echo "WARN: Missing predictions file: ${datapath}.jsonl (skipping ${dataset})"
        continue
    fi

    python "$model_script" --data_path "${datapath}.jsonl"
    wait
done

