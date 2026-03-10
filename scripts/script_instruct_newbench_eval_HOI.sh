#!/bin/bash
## Setting 1 
for method in CMD_SE LAIN
do
  for type in person
  do
    echo "Running with threshold = $thres and person_settings = $ps"
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 python script_eval_32B_newbench_HOI.py \
      --image_folder /mnt/disk1/qinqian/data \
      --dataset hicodet \
      --hoi_pred_json_file ./HOI_pred_hicodet/$method-test_default_B.json \
      --output ./outputs/gpt_bench_eval/$method/hicodet/$type-top5 \
      --prompt_box_type $type \
      --pred_thres 5 \
      --save_pred
  done
done
for method in  CMMP ADA_CM HOLa 
do
  for type in person
  do
    echo "Running with threshold = $thres and person_settings = $ps"
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 python script_eval_32B_newbench_HOI.py \
      --image_folder /mnt/disk1/qinqian/data \
      --dataset hicodet \
      --hoi_pred_json_file ./HOI_pred_hicodet/$method-test_default_L.json \
      --output ./outputs/gpt_bench_eval/$method/hicodet/$type-top5 \
      --prompt_box_type $type \
      --pred_thres 5 \
      --save_pred
  done
done

## Setting 3
for method in CMD_SE LAIN
do
  for type in none
  do
    echo "Running with threshold = $thres and person_settings = $ps"
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 python script_eval_32B_newbench_HOI.py \
      --image_folder /mnt/disk1/qinqian/data \
      --dataset hicodet \
      --hoi_pred_json_file ./HOI_pred_hicodet/$method-test_default_B.json \
      --output ./outputs/gpt_bench_eval/$method/hicodet/$type-top5 \
      --prompt_box_type $type \
      --pred_thres 5 \
      --save_pred
  done
done
for method in  CMMP ADA_CM HOLa 
do
  for type in none
  do
    echo "Running with threshold = $thres and person_settings = $ps"
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=2 python script_eval_32B_newbench_HOI.py \
      --image_folder /mnt/disk1/qinqian/data \
      --dataset hicodet \
      --hoi_pred_json_file ./HOI_pred_hicodet/$method-test_default_L.json \
      --output ./outputs/gpt_bench_eval/$method/hicodet/$type-top5 \
      --prompt_box_type $type \
      --pred_thres 5 \
      --save_pred
  done
done