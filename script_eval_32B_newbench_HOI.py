import argparse
from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor
from PIL import Image
import torch
from qwen_vl_utils import process_vision_info
import os
import re 
import time
import json
from hico_text_label import MAP_AO_TO_HOI, OBJ_IDX_TO_OBJ_NAME, ACT_IDX_TO_ACT_NAME, RARE_HOI_IDX, HICO_INTERACTIONS, TIME_AMBIGUITY_HOI_INDEX
from vcoco_text_label import MAP_AO_TO_HOI_COCO, HOI_TO_AO_COCO, VCOCO_ACTS
from swighoi_categories import SWIG_CATEGORIES, SWIG_ACTIONS, SWIG_INTERACTIONS
import pickle
from torchvision.ops.boxes import batched_nms, box_iou
# from script_hico_evaluation_descrip import merge_json_files
import random 
from transformers import BitsAndBytesConfig
import gc
from torchvision.ops import batched_nms
from newbench_question_func import mllm_instancef1_eval, mllm_macrof1_eval, match_gtbox

def format_bbox(bbox):
    """
    Format a bounding box list such that each number is represented with 2 decimal places.
    Example: [262.80762, 301.31366, 314.68677, 351.58646] -> "[262.81, 301.31, 314.69, 351.59]"
    """
    try:
        return "[" + ", ".join(f"{x:.2f}" for x in bbox) + "]"
    except Exception as e:
        # Fallback: if formatting fails, return the original bounding box.
        return str(bbox)

def get_thres(output_dict, args):

    HOI_cls_predscore = {i: [] for i in range(args.num_hoi_cls)}
    
    for fi in output_dict:
        response_process_list = output_dict[fi]
        ho_scores = torch.tensor(response_process_list["ho_scores"], dtype=torch.float32)
        ao_names = response_process_list['ao_names']
        for sci, lbi in zip(ho_scores, ao_names):
            act, obj = lbi.split(" a/an ")
            act = act.replace(" ", "_")
            obj = obj.replace(" ", "_")
            if "no_interaction" in act:
                act = "no_interaction"

            act_no = ACT_IDX_TO_ACT_NAME.index(act)
            obj_no = OBJ_IDX_TO_OBJ_NAME.index(obj)
            hoii = int(MAP_AO_TO_HOI[act_no, obj_no])
            HOI_cls_predscore[hoii].append(sci)

    return HOI_cls_predscore


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
  
    # model_id = args.model

    os.makedirs(output_folder, exist_ok=True)
    # raw_txt_folder = os.path.join(output_folder, "raw_predictions_txt")
    # os.makedirs(raw_txt_folder, exist_ok=True)



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

    if '_' not in args.dataset and "+" not in args.dataset:
        with open(args.hoi_pred_json_file, 'r') as f:
            output_dict = json.load(f)
    else:
        output_dict = {}
        if 'hicodet' in args.dataset:
            with open(args.hoi_pred_json_file, 'r') as f:
                output_dict1 = json.load(f)
            output_dict = {**output_dict, **output_dict1}
        if 'vcoco' in args.dataset:
            filename =  (args.hoi_pred_json_file.split("/")[-1]).split(".")[0]
            pred2_path = os.path.join(args.hoi_pred_json_file, filename +  '_vcoco.json')
            with open(pred2_path, 'r') as f:
                output_dict2 = json.load(f)
            output_dict = {**output_dict, **output_dict2}
        if 'swig' in args.dataset:
            filename =  (args.hoi_pred_json_file.split("/")[-1]).split(".")[0]
            pred3_path = os.path.join(args.hoi_pred_json_file, filename +  '_swig.json')
            with open(pred3_path, 'r') as f:
                output_dict3 = json.load(f)
            output_dict = {**output_dict, **output_dict3}

    all_ans_per_question = {'tp': [], 'fp': [], 'full_pred': 0, 'full_gt': 0, 'ood': []} ### tp, fp, full_pred, full_gt, ood
    f1_per_question = []
    macro_f1_dict = {}
    acc_top1 = 0
    acc_fullmatch = 0


    if args.pred_select == 'rank':
        ### iterate whole prediction in test set
        HOI_cls_predscore = get_thres(output_dict, args)
        all_preds = torch.tensor([item for sublist in HOI_cls_predscore.values() for item in sublist])
        rank_pred_num = min(int(args.pred_thres), len(all_preds))
        thres_calc, _ = all_preds.topk(rank_pred_num)
        thres_calc = thres_calc[-1].item()
    else:
        thres_calc = args.pred_thres


    fcnt = 0
    newbench_output_dict = {}
    for file in output_dict:
        fcnt += 1
        # if file != "HICO_test2015_00001379.jpg":
        #     continue
        if file not in hoi_det_question:
            continue
        # if args.person_settings != "all" and file not in evaluation_files:
        #     continue
        newbench_output_dict[file] = {}


        response_process_list = output_dict[file]
        print("🔍 Processing file:", file)

        
        hboxes = torch.tensor(response_process_list["h_boxes"], dtype=torch.float32)
        oboxes = torch.tensor(response_process_list["o_boxes"], dtype=torch.float32)
        ho_scores = torch.tensor(response_process_list["ho_scores"], dtype=torch.float32)
        obj_labels = [i.split(" a/an ")[-1] for i in response_process_list['ao_names']]

        
        hoi_det_questioni = hoi_det_question[file]
        hoi_det_questioni_keys = [i for i in hoi_det_questioni]
        if "QA_0" not in hoi_det_questioni_keys:
            hoi_det_questioni_qa = {"QA_0": hoi_det_questioni}
        else:
            hoi_det_questioni_qa = hoi_det_questioni
        
        ### process the HOI prediction for this question
        for qli in hoi_det_questioni_qa:
            newbench_output_dict[file][qli] = []
            if len(hboxes) == 0 or len(oboxes) == 0:
                continue

            content_i = hoi_det_questioni_qa[qli]
            if qli != 'ALL':
                prompt_box_type = args.prompt_box_type
            else:
                prompt_box_type = 'none'
            if prompt_box_type != 'none':
                gt_hboxes = content_i['boxes']['human']
                gt_oboxes = content_i['boxes']['object'] if prompt_box_type == 'all' else None
                if len(content_i['gt_choices']) > 0:
                    if ' a/an ' in content_i['gt_choices'][0]:
                        gt_obj_label = content_i['gt_choices'][0].split(' a/an ')[-1]  
                    else:
                        gt_obj_label = 'person'
                else:
                    gt_obj_label = None
                gt_pred_match, _, _ = match_gtbox(hboxes, oboxes, obj_labels, gt_hboxes, gt_oboxes, prompt_box_type, box_scale_factor=None, gt_obj_label = gt_obj_label)
                
                pred_ind = torch.nonzero(gt_pred_match, as_tuple=False)
                pred_ind = pred_ind[:, 1] if pred_ind.ndim == 2 else pred_ind[1]

                if 'swig' in args.dataset:
                    predi = [response_process_list['ao_names'][i].split(" ")[0] for i in pred_ind.tolist()]
                else:
                    predi = [response_process_list['ao_names'][i] for i in pred_ind.tolist()]
                scoresi = [ho_scores[i] for i in pred_ind.tolist()]
            else:
                if 'swig' in args.dataset:
                    predi = [i.split(" ")[0] for i in response_process_list['ao_names']]
                else:
                    predi = response_process_list['ao_names']
                scoresi = ho_scores

            if args.pred_select == 'question_rank':
                max_rank_qi = min(int(args.pred_thres), len(ho_scores))
                thres_calc = ho_scores.topk(max_rank_qi).values[-1]
            for idx_rpi, rpii in enumerate(predi):
                if scoresi[idx_rpi] < thres_calc:
                    continue
                # if rpii not in newbench_output_dict[file][qli]:
                if rpii in (hoi_det_questioni_qa['QA_0']['gt_choices'] + hoi_det_questioni_qa['QA_0']['wrong_choices']):
                    newbench_output_dict[file][qli].append(rpii)

        # import pdb
        # pdb.set_trace()
        f1_per_question, macro_f1_dict, all_ans_per_question, acc_top1, acc_fullmatch = mllm_instancef1_eval(hoi_det_questioni, newbench_output_dict[file], f1_per_question, macro_f1_dict, all_ans_per_question, file, acc_top1, acc_fullmatch)


         
    if args.save_pred is True:
        os.makedirs(output_folder, exist_ok=True)
        out_file = os.path.join(output_folder, f"{fcnt-1}_hoi_eval_{args.pred_thres}.json")
        with open(out_file, "w") as f:
            json.dump(newbench_output_dict, f, indent=2)

    #### calculate prec and recall
    ### all_ans_per_question
    prec_all_ans_per_question = len(all_ans_per_question['tp']) / (len(all_ans_per_question['tp']) + len(all_ans_per_question['fp']))
    recall_all_ans_per_question = len(all_ans_per_question['tp']) / all_ans_per_question['full_gt']
    print(f"All answers per question - Precision: {prec_all_ans_per_question:.4f}, Recall: {recall_all_ans_per_question:.4f}")

    print(f"Instance F1: {sum(f1_per_question) / len(f1_per_question):.4f}")



    macro_f1_dict_hoicls = {}
    if 'hicodet' in args.dataset or 'vcoco' in args.dataset:
        macro_f1_dict_hoicls = macro_f1_dict_hoicls | {(HICO_INTERACTIONS[i]['action'].replace(" ", "_"), HICO_INTERACTIONS[i]['object'].replace(" ", "_")): {'tp': 0, 'fp': 0, 'gt': 0, 'tn': 0} for i in range(600)}
    if 'vcoco' in args.dataset:
        coco_more = {(VCOCO_ACTS[i].replace(" ", "_"), OBJ_IDX_TO_OBJ_NAME[j-1].replace(" ", "_")): {'tp': 0, 'fp': 0, 'gt': 0, 'tn': 0} for i,j in (MAP_AO_TO_HOI_COCO)}
        macro_f1_dict_hoicls = {**macro_f1_dict_hoicls, **coco_more}
    if 'swig' in args.dataset:
        swig_more = {(i['name'], 'person'): {'tp': 0, 'fp': 0, 'gt': 0, 'tn': 0} for i in SWIG_ACTIONS}
        macro_f1_dict_hoicls = {**macro_f1_dict_hoicls, **swig_more}

    
    macro_f1_dict_hoicls, macro_f1_list, prec_list, rec_list = mllm_macrof1_eval(macro_f1_dict, macro_f1_dict_hoicls, dataset = args.dataset)  

    print(f"Macro F1: {sum(list(macro_f1_list.values())) / len(macro_f1_list):.4f}")

    micro_F1 = 2 * (prec_all_ans_per_question * recall_all_ans_per_question) / (prec_all_ans_per_question + recall_all_ans_per_question)
    print(f"Micro F1 Score: {micro_F1:.4f}")


    all_question = len(hoi_det_question) 
    print("Full match prediction accuracy: ", acc_fullmatch/all_question)
    
    results_txt = os.path.join(output_folder, prompt_box_type + "_evaluation_results.txt")
    with open(results_txt, "w") as ftxt:
        ftxt.write(f"Macro F1: {sum(list(macro_f1_list.values())) / len(macro_f1_list):.4f}\n")
        ftxt.write(f"Instance F1: {sum(f1_per_question) / len(f1_per_question):.4f}\n")
        ftxt.write(f"Micro F1 Score: {micro_F1:.4f}\n")
        ftxt.write(f"Full match prediction accuracy: {acc_fullmatch/all_question:.4f}\n")
        ftxt.write(f"All answers per question - Precision: {prec_all_ans_per_question:.4f}, Recall: {recall_all_ans_per_question:.4f}\n")



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Qwen2.5-VL-32B-Instruct for HOI detection with image input.")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--dataset", type=str, required=True, choices=['hicodet', 'vcoco', 'swig', 'hicodet_vcoco_swig', 'vcoco_swig'], help="dataset images")
    parser.add_argument("--prompt", type=str, default=None, help="Text prompt to use.")
    parser.add_argument("--max_tokens", type=int, default=2048, help="Max new tokens to generate.")
    parser.add_argument("--output", type=str, default="./output", help="Folder to save predictions.")
    parser.add_argument("--detection_pth", type=str, default=None, help="pth saved detected objects for HICO-DET.")
    parser.add_argument('--previous_preds_info', nargs='+', type=str, default=None)
    parser.add_argument("--hoi_pred_json_file", type=str, default=None, help="Folder to get gt.")
    parser.add_argument("--pred_thres", type=float, default=0.5, help="prediction threshold / ranking for HOI methods")
    parser.add_argument("--save_pred", action='store_true')
    parser.add_argument('--prompt_box_type', type=str, default='person', choices=['person', 'all', 'none'], help="evaluation settings with provided boxes for human or human-object or None")
    parser.add_argument('--pred_select', type=str, default='question_rank', choices=['rank', 'thres', 'question_rank'], help="selecting HOI prediction")
    parser.add_argument("--num_hoi_cls", type=int, default=600, help="HOI classes in the benchmark.")
    parser.add_argument('--localization', type=str, default='box', choices=['box', 'draw'], help="using box for localization or draw it on images")
    args = parser.parse_args()

    main(args)