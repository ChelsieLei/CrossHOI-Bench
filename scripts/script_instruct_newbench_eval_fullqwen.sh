export OMP_NUM_THREADS=48
export MKL_NUM_THREADS=48
export OPENBLAS_NUM_THREADS=48
export NUMEXPR_NUM_THREADS=48


### Setting 1: detection + MCQA.  With only person box detection (person) and with both person & object box detection (all)
for boxtype in  person all
do
CUDA_VISIBLE_DEVICES=3 python script_eval_32B_newbench_fullqwen.py \
  --dataset hicodet \
  --model Qwen/Qwen2.5-VL-32B-Instruct \
  --image_folder /mnt/disk1/qinqian/data \
  --output "./outputs/qwen_2.5_32b/hicodet/test-$boxtype-det" \
  --hf_pth /mnt/disk1/qinqian/hf_home \
  --prompt_box_type "$boxtype" \
  --batch_size 1 \
  --two_stage 
done

### Setting 2: Provide GT box for MCQA. With only person box detection (person) and with both person & object box detection (all)
for boxtype in  person all
do
CUDA_VISIBLE_DEVICES=3 python script_eval_32B_newbench_fullqwen.py \
  --dataset hicodet \
  --model Qwen/Qwen2.5-VL-32B-Instruct \
  --image_folder /mnt/disk1/qinqian/data \
  --output "./outputs/qwen_2.5_32b/hicodet/test-$boxtype" \
  --hf_pth /mnt/disk1/qinqian/hf_home \
  --prompt_box_type "$boxtype" \
  --batch_size 1 
done

### Setting 3: Identify all HOIs in the image.
for boxtype in  none
do
CUDA_VISIBLE_DEVICES=3 python script_eval_32B_newbench_fullqwen.py \
  --dataset hicodet \
  --model Qwen/Qwen2.5-VL-32B-Instruct \
  --image_folder /mnt/disk1/qinqian/data \
  --output "./outputs/qwen_2.5_32b/hicodet/test-$boxtype" \
  --hf_pth /mnt/disk1/qinqian/hf_home \
  --prompt_box_type "$boxtype" \
  --batch_size 1 
done

## arguments
# --hoi_pred_json_file [saved the prediction and only requires evaluation]
# --detection_pth      [leverage existing ]






