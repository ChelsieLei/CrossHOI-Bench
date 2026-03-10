import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
import argparse
import ast
from transformers import AutoProcessor, BitsAndBytesConfig
from PIL import Image
import torch
# from qwen_vl_utils import process_vision_info

import re 
import time
import json
from hico_text_label import MAP_AO_TO_HOI, OBJ_IDX_TO_OBJ_NAME, ACT_IDX_TO_ACT_NAME, RARE_HOI_IDX, HICO_INTERACTIONS, TIME_AMBIGUITY_HOI_INDEX, HOI_TO_AO
from vcoco_text_label import MAP_AO_TO_HOI_COCO, HOI_TO_AO_COCO, VCOCO_ACTS
from swighoi_categories import SWIG_CATEGORIES, SWIG_ACTIONS, SWIG_INTERACTIONS
import pickle
from torchvision.ops.boxes import batched_nms, box_iou
# from script_hico_evaluation_descrip import merge_json_files
import random 
import gc
import math
import numpy as np
import torchvision.transforms as T
# from decord import VideoReader, cpu
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer, AutoConfig
from huggingface_hub import snapshot_download
from newbench_question_func import newbench_interaction_question, mllm_instancef1_eval, mllm_macrof1_eval, generate_candidate_pairs, match_gtbox, parse_detection_answer, newbench_detection_question, newbench_pre_question_imgsize, parse_imgsize_answer
import string
from typing import List, Dict, Iterable

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

def build_transform(input_size):
    MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
    transform = T.Compose([
        T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(mean=MEAN, std=STD)
    ])
    return transform

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def dynamic_preprocess(image, min_num=1, max_num=12, image_size=448, use_thumbnail=False):
    orig_width, orig_height = image.size
    aspect_ratio = orig_width / orig_height

    # calculate the existing image aspect ratio
    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
        i * j <= max_num and i * j >= min_num)
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    # find the closest aspect ratio to the target
    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size)

    # calculate the target width and height
    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]
    blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

    # resize the image
    resized_img = image.resize((target_width, target_height))
    processed_images = []
    for i in range(blocks):
        box = (
            (i % (target_width // image_size)) * image_size,
            (i // (target_width // image_size)) * image_size,
            ((i % (target_width // image_size)) + 1) * image_size,
            ((i // (target_width // image_size)) + 1) * image_size
        )
        # split the image
        split_img = resized_img.crop(box)
        processed_images.append(split_img)

    assert len(processed_images) == blocks

    if use_thumbnail and len(processed_images) != 1:
        thumbnail_img = image.resize((image_size, image_size))
        processed_images.append(thumbnail_img)
    return processed_images, target_width, target_height

def load_image(image_file, input_size=448, max_num=12):
    image = Image.open(image_file).convert('RGB')
    transform = build_transform(input_size=input_size)
    images, target_width, target_height = dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
    pixel_values = [transform(image) for image in images]
    pixel_values = torch.stack(pixel_values)
    return pixel_values, target_width, target_height

def split_model(model_name):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f'language_model.model.layers.{layer_cnt}'] = i
            layer_cnt += 1
    device_map['vision_model'] = 0
    device_map['mlp1'] = 0
    device_map['language_model.model.tok_embeddings'] = 0
    device_map['language_model.model.embed_tokens'] = 0
    device_map['language_model.output'] = 0
    device_map['language_model.model.norm'] = 0
    device_map['language_model.model.rotary_emb'] = 0
    device_map['language_model.lm_head'] = 0
    device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

    return device_map


def extract_assistant_response(text):
    lines = text.strip().split("\n")
    if len(lines) < 2:
        return text
    if lines[-2].strip() == 'assistant':
        return lines[-1].strip()
    if 'assistant' in lines:
        idx = lines.index('assistant')
        return " ".join(l.strip() for l in lines[idx+1:]).strip()
    return text.strip()






def _label_to_index(label: str) -> int:
    """
    Excel 风格：A->1, B->2, ..., Z->26, AA->27, AB->28, ...
    仅接受大写 A-Z 的非空串，否则报错。
    """
    if not label or any(not ('A' <= ch <= 'Z') for ch in label):
        raise ValueError(f"无效的起始标签: {label!r}（仅支持大写 A-Z，例如 'A'、'C'、'AA'）")
    val = 0
    for ch in label:
        val = val * 26 + (ord(ch) - ord('A') + 1)
    return val

def _index_to_label(idx: int) -> str:
    """
    1 -> A, 2 -> B, ..., 26 -> Z, 27 -> AA, ...
    """
    if idx <= 0:
        raise ValueError(f"索引必须为正整数，收到 {idx}")
    s = []
    while idx > 0:
        idx -= 1
        s.append(chr(ord('A') + (idx % 26)))
        idx //= 26
    return ''.join(reversed(s))

def _labels(n: int, start: str = "A") -> List[str]:
    """
    生成长度为 n 的标签序列，从 start 开始（包含 start）。
    例：_labels(5, 'Y') -> ['Y','Z','AA','AB','AC']
    """
    if n < 0:
        raise ValueError("n 必须为非负整数")
    if n == 0:
        return []
    base = _label_to_index(start)
    return [_index_to_label(base + i) for i in range(n)]


def label_choices(
    choices: List[str],
    start: str = "A",
    az_only: bool = True,   # True 时只允许 A-Z；False 时用数字
) -> Dict[str, str]:
    """
    将任意长度的 choices 列表映射为 {'A': choice0, 'B': choice1, ...}
    - az_only=True：使用 A–Z / AA–AZ 形式；
    - az_only=False：改为使用数字（'1', '2', '3', ...）。
    """
    if not choices:
        return {}

    if az_only:
        # 规范化 start：允许传入小写，自动转大写
        start = start.upper()
        # 校验 start 合法性（仅大写 A-Z 串）
        _ = _label_to_index(start)
        if len(start) != 1 or not ('A' <= start <= 'Z'):
            raise ValueError("az_only=True 时，start 必须是单个大写字母（A-Z）。")
        end_idx = ord(start) - ord('A') + len(choices)
        if end_idx > 26:
            raise ValueError("选项超过 26 个，且 az_only=True。请关闭 az_only 或减少数量。")
        labels = _labels(len(choices), start=start)
    else:
        # 用数字编号，start 参数忽略
        labels = [str(i + 1) for i in range(len(choices))]

    return dict(zip(labels, choices))


def main(args):
    image_folder = args.image_folder
    if args.dataset == 'hicodet':
        image_folder_name = 'hico_20160224_det/images/test2015' 
    elif args.dataset == 'vcoco':
        image_folder_name = 'mscoco2014/val2014'
    elif args.dataset == 'swig':
        image_folder_name = 'swig_hoi/test_images_512'
    else:
        print("unsupported dataset, only for merged evaluation among hicodet, vcoco and swighoi")
        image_folder_name = ''
    image_folder = os.path.join(image_folder, image_folder_name)


    output_folder = args.output
    model_id = args.model

    os.makedirs(output_folder, exist_ok=True)

    if args.hoi_pred_json_file is None:
        print(f"🚀 Loading model: {model_id}")

        model_id = os.path.join(args.hf_home, model_id)
        device_map = split_model(model_id)

        model = AutoModel.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            load_in_8bit=False,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)


    hoi_det_question = {}
    if 'hicodet' in args.dataset:
        eval_folder = "./hicodet/our_test"
        if args.prompt_box_type == 'none':
            anno_file_name = 'crosshoi_bench_allscene_final.json'
        elif args.prompt_box_type == 'person':
            anno_file_name = 'crosshoi_bench_hum_box_final.json'
        if args.prompt_box_type == 'all':
            anno_file_name = 'crosshoi_bench_hum_obj_box_final.json'
        
        hoi_question_json_file = os.path.join(eval_folder, anno_file_name)
        with open(hoi_question_json_file, 'r') as f:
            hoi_det_question_temp = json.load(f)
        hoi_det_question = hoi_det_question_temp | hoi_det_question

    if 'vcoco' in args.dataset:
        eval_folder = "./vcoco/our_test"
        if args.prompt_box_type == 'none':
            filename_anno = "vcoco_newbench_none.json"
        elif args.prompt_box_type == 'person':
            filename_anno = "vcoco_newbench_person.json"
        else:
            filename_anno = "vcoco_newbench_ho.json"
        hoi_question_json_file = os.path.join(eval_folder, filename_anno)

        with open(hoi_question_json_file, 'r') as f:
            hoi_det_question_temp = json.load(f)
        hoi_det_question = hoi_det_question_temp | hoi_det_question

    if 'swig' in args.dataset:
        eval_folder = "./swighoi/our_test"
        if args.prompt_box_type == 'none':
            filename_anno = "swighoi_newbench_none.json"
        elif args.prompt_box_type == 'person':
            filename_anno = "swighoi_newbench_person.json"
        else:
            filename_anno = "swighoi_newbench_ho.json"
        hoi_question_json_file = os.path.join(eval_folder, filename_anno)

        with open(hoi_question_json_file, 'r') as f:
            hoi_det_question_temp = json.load(f)
        hoi_det_question = hoi_det_question_temp | hoi_det_question

    print("Loading the annotation from", hoi_question_json_file)



    detection_results = None
    if args.detection_pth is not None:
        print(f"🔎 Loading detection results from: {args.detection_pth}")
        with open(args.detection_pth, 'rb') as f:
            detection_results = pickle.load(f)


    if args.previous_preds_info is not None:
        previous_preds, cnt_num = args.previous_preds_info
        generate_flag = 0
        with open(os.path.join(output_folder, str(cnt_num)+"_hoi_eval.json"), 'r') as f:
            output_dict = json.load(f)
    else:
        generate_flag = 1
        previous_preds = None
        output_dict = {}

    cnt = 0
    all_ans_per_question = {'tp': [], 'fp': [], 'full_pred': 0, 'full_gt': 0, 'ood': []} ### tp, fp, full_pred, full_gt, ood
    f1_per_question = []
    macro_f1_dict = {}
    acc_top1 = 0
    acc_fullmatch = 0
    transfer_table = {i: letter for i, letter in enumerate(string.ascii_uppercase)} if args.number_choice is False else {i: f"({i + 1})" for i in range(300)}



    GT_freq = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    Pred_freq = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    if args.hoi_pred_json_file is None:
        for root, _, files in os.walk(image_folder):
            for file in sorted(files):
                if file not in hoi_det_question:
                    continue

                if previous_preds is not None and file == previous_preds:
                    generate_flag = 1 
                    cnt = int(cnt_num)
                    print(f"⚠️ Skipping {file} as it already exists in previous predictions.")
                    continue
                if previous_preds is not None and generate_flag == 0:
                    print(f"⚠️ Skipping {file} as it already exists in previous predictions.")
                    continue    

                start_time = time.time() 
                image_path = os.path.join(root, file)
                image = Image.open(image_path).convert("RGB")
                ## match with Qwen image processing
                width, height = image.size
                
                pixel_values, resized_width, resized_height = load_image(image_path, max_num=12)
                pixel_values = pixel_values.to(torch.bfloat16).cuda() 
         
                # box_scale_factor = torch.tensor([resized_width/width, resized_height/height, resized_width/width, resized_height/height])

                generation_config = dict(max_new_tokens=512, do_sample=True)

                ## assertion of getting questions
                if file not in hoi_det_question:
                    continue
                hoi_det_questioni = hoi_det_question[file]
                hoi_det_questioni_keys = [i for i in hoi_det_questioni]
                if "QA_0" not in hoi_det_questioni_keys:
                    hoi_det_questioni_qa = {"QA_0": hoi_det_questioni}
                else:
                    hoi_det_questioni_qa = hoi_det_questioni

                print("🔍 Processing file:", file)
                conversation_history_list =[]
                pixel_values_list = []
                num_patches_list = []
                response_list = []
                question_list = []
                answer_list = []
                choices_list = []

                if args.prompt_box_type != "none":
                    prompt = newbench_pre_question_imgsize()
                    responses = model.chat(tokenizer, pixel_values, prompt, generation_config)
                    print("MLLM processed image size: ")
                    print(responses)
                    img_size = parse_imgsize_answer(responses)

                    if len(img_size) == 0:
                        print("Fail to get MLLM processed image size.")
                        continue
                    try:
                        resized_width, resized_height = img_size
                        box_scale_factor = torch.tensor([resized_width/width, resized_height/height, resized_width/width, resized_height/height])
                    except:
                        print("Fail to get MLLM processed image size.")
                        continue
                else:
                    box_scale_factor = None


                if args.two_stage is True and args.detection_pth is None:
                    box_type = 'person' if args.prompt_box_type == 'person' or args.dataset == 'swig' else 'all'
                    prompt = newbench_detection_question(box_type)                  

                    responses = model.chat(tokenizer, pixel_values, prompt, generation_config)
                    det_result = parse_detection_answer(box_type, responses)
                    print("MLLM object detection results: ")
                    print(responses)

                    if len(det_result) == 0:
                        print("🙅 fail get detection results, skip")
                        continue
                    if args.dataset == 'swig' and args.prompt_box_type == 'all':
                        det_result['labels'] = [0] * len(det_result['boxes'])
                        det_result['scores'] = [1.0] * len(det_result['boxes'])
                                    

                for qi in hoi_det_questioni_qa:
                    content_i = hoi_det_questioni_qa[qi]
                    hboxes_gt = None
                    oboxes_gt = None
                    if qi != 'ALL':
                        prompt_box_type = args.prompt_box_type
                    else:
                        prompt_box_type = 'none'
                    if prompt_box_type != 'none':
                        hboxes_gt = content_i['boxes']["human"]
                    if prompt_box_type == 'all':
                        oboxes_gt = content_i['boxes']["object"]

                    gt_choices = content_i['gt_choices']
                    incorrect_choices = content_i['wrong_choices']

                    if len(gt_choices + incorrect_choices) > 26:
                        incorrect_choices = incorrect_choices[:26-len(gt_choices)]



                    choices = gt_choices + incorrect_choices
                    random.shuffle(choices)

                    ########## batch size processing
                    ### using GT boxes
                    while len(choices) < 4:
                        print("☹️ missing choices and add None as one")
                        choices.append('None')
                        print(f"choices: {choices}")
                        if len(choices) == 4: 
                            break
                    prompt1 = None
                    if args.two_stage is False:
                        prompt1 = newbench_interaction_question(choices, prompt_box_type = prompt_box_type, box_scale_factor = box_scale_factor, hbox = hboxes_gt, obox = oboxes_gt, box_resize=True, localization = args.localization, number_choice = args.number_choice, dataset = args.dataset)

                    else:
                        if args.detection_pth is not None:
                            det_result = detection_results[file][0]

                        try:
                            candidate_pairs, _ = generate_candidate_pairs(det_result, human_only = (prompt_box_type == 'person'), prompt_box_type = prompt_box_type)
                        except:
                            print(" detection has format issue, skip")
                            continue
                        for ho_det_box_i in candidate_pairs:
                            sub_box = ho_det_box_i[0][1] if prompt_box_type != 'none' else None
                            obj_box = ho_det_box_i[1][1] if prompt_box_type == 'all' else None
                            obj_label = ho_det_box_i[1][2] if prompt_box_type == 'all' else None
                            if args.detection_pth is not None:
                                gt_obj_label = gt_choices[0].split(' a/an ')[-1] if len(gt_choices) > 0 else None
                                if args.dataset == 'swig':
                                    gt_obj_label = 'person'  # SWIG only has person as object
                                gt_box_match, h_det_box_i, o_det_box_i = match_gtbox(sub_box, obj_box, obj_label, hboxes_gt, oboxes_gt, prompt_box_type, box_scale_factor = None, gt_obj_label = gt_obj_label)
                            else:
                                gt_obj_label = gt_choices[0].split(' a/an ')[-1] if len(gt_choices) > 0 else None
                                if args.dataset == 'swig':
                                    gt_obj_label = 'person' # SWIG only has person as object
                                gt_box_match, h_det_box_i, o_det_box_i = match_gtbox(sub_box, obj_box, obj_label, hboxes_gt, oboxes_gt, prompt_box_type, box_scale_factor = box_scale_factor, gt_obj_label = gt_obj_label)


                            if gt_box_match == False:
                                continue

                            if args.second_stage_GT is True:
                                prompt1 = newbench_interaction_question(choices, prompt_box_type = prompt_box_type, box_scale_factor = box_scale_factor, hbox = hboxes_gt, obox = oboxes_gt, box_resize=True, localization = args.localization, number_choice = args.number_choice, dataset = args.dataset)
                            elif args.detection_pth is not None:
                                prompt1 = newbench_interaction_question(choices, prompt_box_type = prompt_box_type, box_scale_factor = box_scale_factor, hbox = h_det_box_i, obox = o_det_box_i, box_resize=True, localization = args.localization, number_choice = args.number_choice, dataset = args.dataset)
                            else:
                                prompt1 = newbench_interaction_question(choices, prompt_box_type = prompt_box_type, hbox = h_det_box_i, obox = o_det_box_i, localization = args.localization, number_choice = args.number_choice, dataset = args.dataset)       
                            break


                    if prompt1 == None:
                        continue

                    pixel_values_list.append(pixel_values)
                    num_patches_list.append(pixel_values.size(0))
                    conversation_history_list.append(prompt1)
                    question_list.append(qi)

                    gt_ind_i = []
                    
     
                    for i in gt_choices:
                        num_i = choices.index(i)
                        gt_ind_i.append(transfer_table[num_i])
                        answer_list.append(transfer_table[num_i])
                    choices_list.append(label_choices(choices, az_only = (not args.number_choice)))



                response_list = []
                chunk_size = args.batch_size
                for i in range(0, len(conversation_history_list), chunk_size):
                    chunk = conversation_history_list[i:i + chunk_size]
                    img_chunk = pixel_values_list[i:i + chunk_size]
                    img_chunk_tensor = torch.cat(img_chunk, dim=0)  
                    num_patches_list_chunk = num_patches_list[i:i + chunk_size]

                    response_chunk = model.batch_chat(
                        tokenizer,
                        img_chunk_tensor,
                        num_patches_list=num_patches_list_chunk,
                        questions=chunk,
                        generation_config=generation_config
                    )
                    print("\n📢 Prediction for file : ")    
                    print(response_chunk)
                    response_list.extend(response_chunk)


                for answer_i in (answer_list):
                    if answer_i not in GT_freq.keys():
                        GT_freq[answer_i] = 0
                    GT_freq[answer_i] += 1
                for pi in (response_list):
                    pi_list = [p.strip() for p in pi.split(',')]
                    if len(pi_list) > 0:
                        for j in pi_list:
                            if j not in Pred_freq.keys():
                                Pred_freq[j] = 0
                            Pred_freq[j] += 1



                output_dict[file] = {}

                for response_i, qli, chsi in zip(response_list, question_list, choices_list):
                    ans_i = [x.strip().upper().rstrip('.') for x in response_i.split(',') if x.strip()]
                    
                    if qli not in output_dict[file]:
                        output_dict[file][qli] = [chsi[letter] for letter in ans_i if letter in chsi]
                    else:
                        output_dict[file][qli] += [chsi[letter] for letter in ans_i if letter in chsi]
                    print("Ans: ", output_dict[file][qli])

                del response_list
                gc.collect()
                torch.cuda.empty_cache()
     

                cnt += 1
                if cnt % 500 == 0 or cnt >= len(files):
                    out_file = os.path.join(output_folder, f"{cnt}_hoi_eval.json")
                    with open(out_file, "w") as f:
                        json.dump(output_dict, f, indent=2)
                    # output_dict = {}
                    print(f"✅ Saved predictions to {out_file}")


                end_time = time.time()  # end time counting
                elapsed_time = end_time - start_time
                print(f"🕒 Time for {file}: {elapsed_time:.2f} seconds\n")
        out_file = os.path.join(output_folder, f"merged_hoi_eval.json")
        with open(out_file, "w") as f:
            json.dump(output_dict, f, indent=2)

    else:
        if '_' not in args.dataset and "+" not in args.dataset:
            with open(args.hoi_pred_json_file, 'r') as f:
                output_dict = json.load(f)
        else:
            output_dict = {}
            if 'hicodet' in args.dataset:
                box_type = args.prompt_box_type if args.two_stage is False else args.prompt_box_type+"-det"
                pred1_path = os.path.join(args.hoi_pred_json_file, 'hicodet/final_v5', box_type, 'merged_hoi_eval.json')
                with open(pred1_path, 'r') as f:
                    output_dict1 = json.load(f)
                output_dict = {**output_dict, **output_dict1}
            if 'vcoco' in args.dataset:
                box_type = args.prompt_box_type if args.two_stage is False else args.prompt_box_type+"-det"
                pred2_path = os.path.join(args.hoi_pred_json_file, 'vcoco', box_type, 'merged_hoi_eval.json')
                with open(pred2_path, 'r') as f:
                    output_dict2 = json.load(f)
                output_dict = {**output_dict, **output_dict2}
            if 'swig' in args.dataset:
                box_type = args.prompt_box_type if args.two_stage is False else args.prompt_box_type+"-det"
                pred3_path = os.path.join(args.hoi_pred_json_file, 'swig', box_type, 'merged_hoi_eval.json')
                with open(pred3_path, 'r') as f:
                    output_dict3 = json.load(f)
                output_dict = {**output_dict, **output_dict3}

    print("GT_freq: ", GT_freq)
    print("Pred_freq: ", Pred_freq)
    ######### evaluation    
    macro_f1_dict_hoicls = {}
    if 'hicodet' in args.dataset or 'vcoco' in args.dataset:
        macro_f1_dict_hoicls = macro_f1_dict_hoicls | {(HICO_INTERACTIONS[i]['action'].replace(" ", "_"), HICO_INTERACTIONS[i]['object'].replace(" ", "_")): {'tp': 0, 'fp': 0, 'gt': 0, 'tn': 0} for i in range(600)}
    if 'vcoco' in args.dataset:
        coco_more = {(VCOCO_ACTS[i].replace(" ", "_"), OBJ_IDX_TO_OBJ_NAME[j-1].replace(" ", "_")): {'tp': 0, 'fp': 0, 'gt': 0, 'tn': 0} for i,j in (MAP_AO_TO_HOI_COCO)}
        macro_f1_dict_hoicls = {**macro_f1_dict_hoicls, **coco_more}
    if 'swig' in args.dataset:
        swig_more = {(i['name'], 'person'): {'tp': 0, 'fp': 0, 'gt': 0, 'tn': 0} for i in SWIG_ACTIONS}
        macro_f1_dict_hoicls = {**macro_f1_dict_hoicls, **swig_more}
    for file in hoi_det_question:
        hoi_det_questioni = hoi_det_question[file]
        if file not in output_dict:
            response_process_list = {'QA_0': []}
        else:
            response_process_list = output_dict[file]


        f1_per_question, macro_f1_dict, all_ans_per_question, acc_top1, acc_fullmatch = mllm_instancef1_eval(hoi_det_questioni, response_process_list, f1_per_question, macro_f1_dict, all_ans_per_question, file, acc_top1, acc_fullmatch)
    #### calculate prec and recall
    ### all_ans_per_question
    prec_all_ans_per_question = len(all_ans_per_question['tp']) / (len(all_ans_per_question['tp']) + len(all_ans_per_question['fp']))
    recall_all_ans_per_question = len(all_ans_per_question['tp']) / all_ans_per_question['full_gt']
    print(f"All answers per question - Precision: {prec_all_ans_per_question:.4f}, Recall: {recall_all_ans_per_question:.4f}")

    print(f"Instance F1: {sum(f1_per_question) / len(f1_per_question):.4f}")
    

    macro_f1_dict_hoicls, macro_f1_list, prec_list, rec_list = mllm_macrof1_eval(macro_f1_dict, macro_f1_dict_hoicls, dataset = args.dataset)  

    print(f"Macro F1: {sum(list(macro_f1_list.values())) / len(macro_f1_list):.4f}")


    micro_F1 = 2 * (prec_all_ans_per_question * recall_all_ans_per_question) / (prec_all_ans_per_question + recall_all_ans_per_question)
    print(f"Micro F1: {micro_F1:.4f}")
    
    all_question = len(hoi_det_question) 
    print("Full match prediction accuracy: ", acc_fullmatch/all_question)
    
    results_txt = os.path.join(output_folder, "evaluation_results.txt")
    with open(results_txt, "w") as ftxt:
        ftxt.write(f"Macro F1: {sum(list(macro_f1_list.values())) / len(macro_f1_list):.4f}\n")
        ftxt.write(f"Instance F1: {sum(f1_per_question) / len(f1_per_question):.4f}\n")
        ftxt.write(f"Micro F1 Score: {micro_F1:.4f}\n")
        ftxt.write(f"Full match prediction accuracy: {acc_fullmatch/all_question:.4f}\n")
        ftxt.write(f"All answers per question - Precision: {prec_all_ans_per_question:.4f}, Recall: {recall_all_ans_per_question:.4f}\n")

    
    print("Evaluation results saved to:", results_txt)

     

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run OpenGVLab/InternVL3-38B for HOI detection with image input.")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--dataset", type=str, required=True, choices=['hicodet', 'vcoco', 'swig', 'hicodet_vcoco_swig', 'vcoco_swig'], help="dataset images")
    parser.add_argument("--model", type=str, default="OpenGVLab/InternVL3-38B", help="Model name or path.")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt to use.")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max new tokens to generate.")
    parser.add_argument("--output", type=str, default="./output", help="Folder to save predictions.")
    parser.add_argument("--detection_pth", type=str, default=None, help="pth saved detected objects for HICO-DET.")
    parser.add_argument("--two_stage", action='store_true')
    parser.add_argument("--second_stage_GT", action='store_true')
    parser.add_argument('--previous_preds_info', nargs='+', type=str, default=None)
    parser.add_argument("--hoi_question_json_file", type=str, default=None, help="Folder to get gt.")
    parser.add_argument("--hoi_pred_json_file", type=str, default=None, help="Folder to get gt.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument('--prompt_box_type', type=str, default='person', choices=['person', 'all', 'none'], help="evaluation settings with provided boxes for human or human-object or None")
    parser.add_argument("--hf_home", type=str, default="/mnt/disk1/qinqian/hf_home/hub", help="Folder to get hungging face checkpoints.")
    parser.add_argument("--enhanced_setting1_anno", action='store_true', help="Changed the positive in setting 1 by only revising no_interaction labels.")
    parser.add_argument("--all_test", action='store_true', help='evaluate all test images or only challenging images')
    parser.add_argument('--localization', type=str, default='box', choices=['box', 'draw'], help="using box for localization or draw it on images")
    parser.add_argument("--number_choice", action='store_true', help='use number instead of A_Z for chocies')
    
    args = parser.parse_args()

    main(args)