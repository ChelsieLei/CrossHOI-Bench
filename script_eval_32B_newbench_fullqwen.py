import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"
os.environ["NUMEXPR_NUM_THREADS"] = "4"
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"
import argparse
from PIL import Image
import torch
from qwen_vl_utils import process_vision_info
import re 
import time
import json
from hico_text_label import MAP_AO_TO_HOI, OBJ_IDX_TO_OBJ_NAME, ACT_IDX_TO_ACT_NAME, HICO_INTERACTIONS, HOI_TO_AO
from vcoco_text_label import MAP_AO_TO_HOI_COCO, HOI_TO_AO_COCO, VCOCO_ACTS
from swighoi_categories import SWIG_CATEGORIES, SWIG_ACTIONS, SWIG_INTERACTIONS
import pickle
from torchvision.ops.boxes import batched_nms, box_iou
# from script_hico_evaluation_descrip import merge_json_files
import random 
from transformers import BitsAndBytesConfig
import gc
from newbench_question_func import newbench_interaction_question, mllm_instancef1_eval, mllm_macrof1_eval, generate_candidate_pairs, match_gtbox, newbench_detection_question, parse_detection_answer, newbench_pre_question_imgsize, parse_imgsize_answer
from peft import PeftModel
from typing import List, Dict, Iterable
import string
import matplotlib.pyplot as plt


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

### resize the image for Qwen processing
def qwen_img_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 3136, max_pixels: int = 12845056
) -> tuple[int, int]:
    """
    Rescales the image so that the following conditions are met:

    1. Both dimensions (height and width) are divisible by 'factor'.

    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].

    3. The aspect ratio of the image is maintained as closely as possible.
    """
    MAX_RATIO = 200
    if max(height, width) / min(height, width) > MAX_RATIO:
        raise ValueError(
            f"absolute aspect ratio must be smaller than {MAX_RATIO}, got {max(height, width) / min(height, width)}"
        )
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar

def round_by_factor(number: int, factor: int) -> int:
    """Returns the closest integer to 'number' that is divisible by 'factor'."""
    return round(number / factor) * factor

def ceil_by_factor(number: int, factor: int) -> int:
    """Returns the smallest integer greater than or equal to 'number' that is divisible by 'factor'."""
    return math.ceil(number / factor) * factor

def floor_by_factor(number: int, factor: int) -> int:
    """Returns the largest integer less than or equal to 'number' that is divisible by 'factor'."""
    return math.floor(number / factor) * factor



def overlay_heatmap_on_image(pil_image, heatmap_2d, save_path, cmap="jet", alpha=0.4):
    """
    pil_image: PIL.Image
    heatmap_2d: [H_grid, W_grid] 的 torch.Tensor / numpy array
    """
    import numpy as np
    if isinstance(heatmap_2d, torch.Tensor):
        heat = heatmap_2d.float().cpu().numpy()
    else:
        heat = np.array(heatmap_2d, dtype=np.float32)

    # 归一化到 [0,1]
    heat = (heat - heat.min()) / (heat.max() - heat.min() + 1e-6)

    W_img, H_img = pil_image.size  # PIL: (W, H)

    plt.figure(figsize=(6, 6))
    plt.imshow(pil_image)
    plt.imshow(
        heat,
        cmap=cmap,
        alpha=alpha,
        extent=(0, W_img, H_img, 0),   # align 到图像坐标
        interpolation="bilinear",
    )
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()
    
def visualize_attention_matrix(att_matrix, save_path, vmax=None):
    mat = att_matrix.float().cpu().numpy()
    plt.figure(figsize=(5, 5))
    plt.imshow(mat, cmap="viridis", aspect="auto", vmax=vmax)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()

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

def qwen_chatbox(processor, model, batch_conversation_history, image_patch_size, args):
    # Preprocess input

    text_input_list = []
    image_input_list = []
   
    
    for conv in batch_conversation_history:
        text_input_list.append(processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True))
        img_input, _ = process_vision_info(conv) # , image_patch_size=image_patch_size
        image_input_list.append(img_input)

    # 用 processor 批量处理
    inputs = processor(
        text=text_input_list,
        images=image_input_list,
        return_tensors="pt",
        padding=True
    )


    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Generate
    print("🔮 Generating...")
    # generated_ids = model.generate(**inputs, max_new_tokens=args.max_tokens)
    # responses = processor.batch_decode(generated_ids, skip_special_tokens=True)
    # responses = [extract_assistant_response(r) for r in responses]

    with torch.no_grad():
        gen_out = model.generate(
            **inputs,
            max_new_tokens=args.max_tokens,
            return_dict_in_generate=True,
            output_scores=True,   # 收集每一步 logits
            # do_sample=False,      # 建议先 greedy，方便分析
        )

    sequences = gen_out.sequences           # [batch, prompt_len + gen_len]
    scores_list = gen_out.scores            # 长度 = gen_len，每个 [batch, vocab]

    responses = processor.batch_decode(sequences, skip_special_tokens=True)
    responses = [extract_assistant_response(r) for r in responses]
    print("\n📢 Prediction for file : ", responses)


    return responses

def qwen_chatbox_with_probs(processor, model, batch_conversation_history, args):
    # ---- (same pre-processing as existing qwen_chatbox) ----
    text_input_list, image_input_list = [], []
    for conv in batch_conversation_history:
        text_input_list.append(
            processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True)
        )
        img_input, _ = process_vision_info(conv)
        image_input_list.append(img_input)

    inputs = processor(
        text=text_input_list,
        images=image_input_list,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # ---- generate while saving logits for each step ----
    gen_out = model.generate(
        **inputs,
        max_new_tokens=args.max_tokens,
        return_dict_in_generate=True,
        output_scores=True,          # collects logits
    )
    seqs = gen_out.sequences                 # [batch, prompt_len + gen_len]
    scores = gen_out.scores                  # list of [batch, vocab] tensors
    prompt_len = inputs["input_ids"].shape[-1]
    gen_ids = seqs[:, prompt_len:]           # [batch, gen_len]

    # ---- convert logits -> probability of each produced token ----
    token_probs = []
    for step, score in enumerate(scores):
        step_prob = score.softmax(dim=-1)    # [batch, vocab]
        tok_prob = step_prob[
            torch.arange(step_prob.size(0)),  # batch index
            gen_ids[:, step]                  # id generated at this step
        ]
        token_probs.append(tok_prob)
    token_probs = torch.stack(token_probs, dim=1)  # [batch, gen_len]

    # ---- decode to text and show per-token probability (example for batch 0) ----
    decoded = processor.batch_decode(seqs, skip_special_tokens=True)
    tok_texts = [processor.tokenizer.decode([tid]) for tid in gen_ids[0]]
    for t, p in zip(tok_texts, token_probs[0].tolist()):
        print(f"{repr(t)} : {p:.4f}")
    # if len(scores) > 2:
    #     import pdb
    #     pdb.set_trace()
    return decoded, gen_ids, token_probs



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

        if 'Qwen2.5' in args.model:
            from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
            base_model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                # attn_implementation="flash_attention_2",
                trust_remote_code=True
            )
        elif 'Qwen2' in args.model:
            from transformers import Qwen2VLForConditionalGeneration, AutoProcessor
            base_model = Qwen2VLForConditionalGeneration.from_pretrained(
                model_id,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                # attn_implementation="flash_attention_2",
            )
        elif 'Qwen3' in args.model:

            from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor
            from transformers import Qwen3VLForConditionalGeneration

            if 'A3B' in args.model:
                base_model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    # attn_implementation="flash_attention_2",
                    trust_remote_code=True
                )
            else:
                base_model = Qwen3VLForConditionalGeneration.from_pretrained(
                    model_id,
                    torch_dtype=torch.bfloat16,
                    device_map="auto",
                    # attn_implementation="flash_attention_2",
                    trust_remote_code=True,
                    attn_implementation="eager" if args.vis_prob_chat is True else 'sdpa',  
                )



        processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
        processor.tokenizer.padding_side = "left"


        if args.lora_dir is not None:        
            model = PeftModel.from_pretrained(base_model, args.lora_dir, device_map="auto")
            model.eval()
        else:
            model = base_model

        image_patch_size = 16 if 'Qwen3' in args.model else 14

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


    output_dict = {}
    cnt = 0
    all_ans_per_question = {'tp': [], 'fp': [], 'full_pred': 0, 'full_gt': 0, 'ood': []} ### tp, fp, full_pred, full_gt, ood
    f1_per_question = []
    macro_f1_dict = {}
    acc_top1 = 0
    acc_fullmatch = 0
    transfer_table = {i: letter for i, letter in enumerate(string.ascii_uppercase)} if args.number_choice is False else {i: f"({i + 1})" for i in range(300)}


    ### load the settings

    GT_freq = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    Pred_freq = {'A': 0, 'B': 0, 'C': 0, 'D': 0}
    if args.hoi_pred_json_file is None:
        for root, _, files in os.walk(image_folder):
            for file in sorted(files):
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
                resize_factor = 32 if 'Qwen3' in args.model else 28
                resized_height, resized_width = qwen_img_resize(height, width, factor = resize_factor)
                image = image.resize((resized_width, resized_height))
                box_scale_factor = torch.tensor([resized_width/width, resized_height/height, resized_width/width, resized_height/height])

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
                question_list = []
                answer_list = []
                choices_list = []

                if args.two_stage is True and args.detection_pth is None:
                    box_type = 'person' if args.prompt_box_type == 'person' or args.dataset == 'swig' else 'all'
                    prompt = newbench_detection_question(box_type)

                    conversation_history = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt}
                        ],
                    }]

                    responses = qwen_chatbox(processor, model, [conversation_history], image_patch_size, args)
                    det_result = parse_detection_answer(box_type, responses[0])
                    if len(det_result) == 0:
                        continue
                    if args.dataset == 'swig' and args.prompt_box_type == 'all':
                        det_result['labels'] = [0] * len(det_result['boxes'])
                        det_result['scores'] = [1.0] * len(det_result['boxes'])
                
                for qi in hoi_det_questioni_qa:
                    content_i = hoi_det_questioni_qa[qi]
                    hboxes_gt = None
                    oboxes_gt = None
                    prompt_box_type = args.prompt_box_type

                    if prompt_box_type != 'none':
                        hboxes_gt = content_i['boxes']["human"]
                    if prompt_box_type == 'all':
                        oboxes_gt = content_i['boxes']["object"]
                    gt_choices = content_i['gt_choices']
                    incorrect_choices = content_i['wrong_choices']


                    choices = gt_choices + incorrect_choices
                    random.shuffle(choices)

                    ########## batch size processing
                    ### using GT boxes
                    while len(choices) < 4:
                    # if len(choices) == 3:
                        print("☹️ missing choices and add None as one")
                        choices.append('None')
                        if len(choices) == 4: 
                            break

                    prompt1 = None
                    if args.two_stage is False:
                        prompt1 = newbench_interaction_question(choices, prompt_box_type = prompt_box_type, box_scale_factor = box_scale_factor, hbox = hboxes_gt, obox = oboxes_gt, box_resize=True, reasoning = args.reasoning, localization = args.localization, number_choice = args.number_choice, dataset = args.dataset)

                    ##### using the detected bounding boxes
                    else:
                        if args.detection_pth is not None:
                            det_result = detection_results[file][0]
                    
                        try:
                            candidate_pairs, _ = generate_candidate_pairs(det_result, human_only = (prompt_box_type == 'person'), prompt_box_type = prompt_box_type)
                        except:
                            print("❌ fail to extract detected bounding boxese with format issue")
                            continue

                        for ho_det_box_i in candidate_pairs:
                            sub_box = ho_det_box_i[0][1] if prompt_box_type != 'none' else None
                            obj_box = ho_det_box_i[1][1] if prompt_box_type == 'all' else None
                            obj_label = ho_det_box_i[1][2] if prompt_box_type == 'all' else None
                            
                            gt_obj_label = gt_choices[0].split(' a/an ')[-1] if len(gt_choices) > 0 else None
                            if args.dataset == 'swig':
                                gt_obj_label = 'person'
                            if args.detection_pth is not None:
                                gt_box_match, h_det_box_i, o_det_box_i = match_gtbox(sub_box, obj_box, obj_label, hboxes_gt, oboxes_gt, prompt_box_type, box_scale_factor = None, gt_obj_label = gt_obj_label)
                            else: 
                                gt_box_match, h_det_box_i, o_det_box_i = match_gtbox(sub_box, obj_box, obj_label, hboxes_gt, oboxes_gt, prompt_box_type, box_scale_factor = box_scale_factor, gt_obj_label = gt_obj_label)

                            if gt_box_match == False:
                                continue
                            
                            if args.second_stage_GT is True:
                                prompt1 = newbench_interaction_question(choices, prompt_box_type = prompt_box_type, box_scale_factor = box_scale_factor, hbox = hboxes_gt, obox = oboxes_gt, box_resize=True, reasoning = args.reasoning, localization = args.localization, number_choice = args.number_choice, dataset = args.dataset)
                            elif args.detection_pth is not None:
                                prompt1 = newbench_interaction_question(choices, prompt_box_type = prompt_box_type, box_scale_factor = box_scale_factor, hbox = h_det_box_i, obox = o_det_box_i, box_resize=True, reasoning = args.reasoning, localization = args.localization, number_choice = args.number_choice, dataset = args.dataset)
                            else:
                                prompt1 = newbench_interaction_question(choices, prompt_box_type = prompt_box_type, hbox = h_det_box_i, obox = o_det_box_i, reasoning = args.reasoning, localization = args.localization, number_choice = args.number_choice, dataset = args.dataset)       
                            break
                   
                    if prompt1 == None:
                        print("No matched detection boxes, skip")
                        continue
                    
                    conversation_history = [{
                        "role": "user",
                        "content": [
                            {"type": "image", "image": image},
                            {"type": "text", "text": prompt1}
                        ],
                    }]
                    conversation_history_list.append(conversation_history)
                    question_list.append(qi)
                    gt_ind_i = []
                    for i in gt_choices:
                        num_i = choices.index(i)
                        gt_ind_i.append(transfer_table[num_i])
                        answer_list.append(transfer_table[num_i])
                    choices_list.append(label_choices(choices, az_only = (not args.number_choice)))
                    # choices_list.append({'A': choices[0], 'B': choices[1], 'C': choices[2], 'D': choices[3]})


                # Batch the conversation_history_list into chunks of size 6 and process each chunk
                response_list = []
                chunk_size = args.batch_size
                for i in range(0, len(conversation_history_list), chunk_size):
                    chunk = conversation_history_list[i:i + chunk_size]
                    response_chunk = qwen_chatbox(processor, model, chunk, image_patch_size, args)
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
                    if args.reasoning == 'v1' or args.reasoning == 'v2':
                        answer_match = re.search(r"<answer>\s*([A-Za-z, ]+)\s*</answer>", response_i, re.IGNORECASE | re.DOTALL)
                        letters = answer_match.group(1).replace(" ", "") if answer_match else ""
                    elif args.reasoning == 'thinking':
                        letters = response_i.split("</think>")[-1].strip()
                    else:
                        letters = response_i
                    ans_i = [x.strip().upper().rstrip('.') for x in letters.split(',') if x.strip()]
                    if qli not in output_dict[file]:
                        output_dict[file][qli] = [chsi[letter] for letter in ans_i if letter in chsi]
                    else:
                        output_dict[file][qli] += [chsi[letter] for letter in ans_i if letter in chsi]
                    print("Ans: ", output_dict[file][qli])
                    
                    if args.reasoning != 'none':
                        output_dict[file][qli + "_reasoning"] = response_i
                    
                del response_list
                gc.collect()
                torch.cuda.empty_cache()



                cnt += 1
                num_written = 500 # if args.two_stage is False else 200
                if cnt % num_written == 0 or cnt >= len(files):
                    out_file = os.path.join(output_folder, f"{cnt}_hoi_eval.json")
                    with open(out_file, "w") as f:
                        json.dump(output_dict, f, indent=2)
                    print(f"✅ Saved predictions to {out_file}")


                end_time = time.time()  # end time counting
                elapsed_time = end_time - start_time
                print(f"🕒 Time for {file}: {elapsed_time:.2f} seconds\n")
    
        out_file_name =  f"merged_hoi_eval.json" # if sec_part == 'none' else f"merged_hoi_eval_{sec_part}anno.json"
        out_file = os.path.join(output_folder, out_file_name) 
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
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL-32B-Instruct for HOI detection with image input.")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--dataset", type=str, required=True, choices=['hicodet', 'vcoco', 'swig', 'hicodet_vcoco_swig', 'vcoco_swig', 'hicodet_swig'], help="dataset images")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-VL-32B-Instruct", help="Model name or path.")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max new tokens to generate.")
    parser.add_argument("--output", type=str, default="./output", help="Folder to save predictions.")
    parser.add_argument("--detection_pth", type=str, default=None, help="pth saved detected objects for HICO-DET.")
    parser.add_argument("--two_stage", action='store_true')
    parser.add_argument("--second_stage_GT", action='store_true')
    parser.add_argument('--previous_preds_info', nargs='+', type=str, default=None)
    parser.add_argument("--hoi_pred_json_file", type=str, default=None, help="Folder to get prediction")
    # parser.add_argument("--hf_pth", type=str, default="/mnt/disk1/qinqian/hf_home")
    parser.add_argument("--batch_size", type=int, default=3)
    parser.add_argument('--prompt_box_type', type=str, default='person', choices=['person', 'all', 'none'], help="evaluation settings with provided boxes for human or human-object or None")
    parser.add_argument("--lora_dir", type=str, default=None, help="Folder for pretrained lora qwen model.")
    parser.add_argument("--reasoning", default='none', choices=['none', 'v1', 'v2', 'thinking'], help='none means directly output answer without reasoning, v1 is reasoning + answer, v2 is answer + reasoning')
    parser.add_argument('--localization', type=str, default='box', choices=['box', 'draw'], help="using box for localization or draw it on images")
    parser.add_argument("--number_choice", action='store_true', help='use number instead of A_Z for chocies')
    parser.add_argument("--vis_prob_chat", action='store_true', help='visualize probability of output answer among choices')
    
    args = parser.parse_args()

    main(args)