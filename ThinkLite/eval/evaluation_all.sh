#!/bin/bash

num_chunks=16
model_id="russwang/ThinkLite-VL-7B"


datasets=("mathvista" "mathvision" "mmmu" "mmstar" "mmbench" "mmstar" "mathverse" "ai2d")


for dataset in "${datasets[@]}"; do
   # ensure output directory exists for nested model_id (e.g., russwang/ThinkLite-VL-7B)
   mkdir -p "./eval_files/${dataset}/${model_id%/*}"
   output_prefix="./eval_files/${dataset}/${model_id}_"
   final_file_prefix="./eval_files/${dataset}/${model_id}"

   model_script="eval/model_${dataset}_qwen.py"

   for i in {0..15}; do
       python "$model_script" \
           --model_id "$model_id" \
           --answers-file "${output_prefix}$((i+1)).jsonl" \
           --num-chunks "$num_chunks" \
           --chunk-idx "$i" \
           --gpu-id "$i" &
   done

   wait

   cat ${output_prefix}*.jsonl > ${final_file_prefix}.jsonl

   wait
done

