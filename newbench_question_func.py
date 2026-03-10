

from hico_text_label import MAP_AO_TO_HOI, OBJ_IDX_TO_OBJ_NAME, ACT_IDX_TO_ACT_NAME, RARE_HOI_IDX, HICO_INTERACTIONS
from torchvision.ops.boxes import batched_nms, box_iou
import torch
import re
import json

#### new benchmark questions prompt
def newbench_interaction_question(choices, prompt_box_type, box_scale_factor= None, hbox = None, obox = None, box_resize = False, answer_tag = None, reasoning = False, localization = 'box', number_choice = False, dataset = 'hicodet'):
    if localization == 'box':
        if box_resize == True:
            if prompt_box_type != 'none':
                hbox_resize = (torch.tensor(hbox) * box_scale_factor).tolist()
                hbox_resized = [round(hbox_resize[0], 2), round(hbox_resize[1], 2), round(hbox_resize[2], 2), round(hbox_resize[3], 2)]
                
                if prompt_box_type == 'all':
                    obox_resize = (torch.tensor(obox) * box_scale_factor).tolist()
                    obox_resized = [round(obox_resize[0], 2), round(obox_resize[1], 2), round(obox_resize[2], 2), round(obox_resize[3], 2)]
                    
        else:
            hbox_resized = hbox
            obox_resized = obox

        if prompt_box_type == 'all':
            prompt1 = ( 
                f"Context: You are given an image  <image> and two bounding boxes:\n"
                f"- Person bbox: {hbox_resized}\n"
                f"- Object bbox: {obox_resized}\n"
                f"Question: Which of the following properly describes the interaction between the person and the object?\n"
            )
            if dataset == 'swig':
                prompt1 = ( 
                    f"Context: You are given an image  <image> and two human bounding boxes:\n"
                    f"- Subject bbox: {hbox_resized}\n"
                    f"- Object bbox: {obox_resized}\n"
                    f"Question: Which of the following best describes the human-human interaction between the subject and the object?\n"
                )
            # (
            #     f"Context: You are given an image <image> with two bounding boxes:\n"
            #     f"- Person bbox: {hbox_resized}\n"
            #     f"- Object bbox: {obox_resized}\n"
            #     f"Focus only on the interaction between this specific person and this specific object.\n"
            #     f"Ignore any other objects or people that may appear in the image.\n"
            #     f"Question: Which of the following best describes the interaction between the person and the object?\n"
            # )
            
        elif prompt_box_type == 'person':
            prompt1 = (
                f"Context: You are given an image  <image> and a target person with a bounding box {hbox_resized}.\n"
                "Question: Which of the following describes the interactions between the target person and any object in the image?\n"
            )
            if dataset == 'swig':
                prompt1 = (
                    f"Context: You are given an image  <image> and a target person with a bounding box {hbox_resized}.\n"
                    "Question: Which of the following best describes the interactions between the target person and any other persons in the image?\n"
                )
            # (
            #     f"Context: You are given an image <image> with one bounding box:\n"
            #     f"- Person bbox: {hbox_resized}\n"
            #     f"Focus only on this specific person and describe their interaction with surrounding objects in the image.\n"
            #     f"Ignore other people or objects that are not directly related to this person.\n"
            #     f"Question: Which of the following best describes what the person in the red box is doing?\n"
            # )
                        
            

        elif prompt_box_type == 'none':
            prompt1  = ( 
                f"Question: Which of the following properly describes the interactions in the image  <image>?\n"
            )
            if dataset == 'swig':
                prompt1  = ( 
                    f"Question: Which of the following best describes the human-human interactions in the image  <image>?\n"
                )

    elif localization == 'draw':
        if prompt_box_type == 'all':
            prompt1 = ( 
                    f"Context: You are given an image <image> with a target person (localized by a red box) and a target object (localized by a blue box).\n"
                    # f"Note: Each box indicates the main entity it encloses — focus on the primary person or object occupying the box, not smaller items or regions inside it.\n"
                    f"Question: Which of the following best describes the interaction between the person and the object?\n"
                )
            # (
            #     f"Context: You are given an image <image> with two bounding boxes:\n"
            #     f"- Red box: the target person\n"
            #     f"- Blue box: the target object\n"
            #     f"Focus only on the interaction between this specific person and this specific object.\n"
            #     f"Ignore any other objects or people that may appear in the image.\n"
            #     f"Question: Which of the following best describes the interaction between the person and the object?\n"
            # )
            

        elif prompt_box_type == 'person':
            prompt1 = (
                f"Context: You are given an image <image> with a target person (localized by a red box).\n"
                # f"Note: The red box indicates the main person it encloses — focus only on this individual, not other people or regions inside or outside the box.\n"
                f"Question: Which of the following best describes this person's interaction with surrounding objects in the image?\n"
            )
            # (
            #     f"Context: You are given an image <image> with one bounding box:\n"
            #     f"- Red box: the target person\n"
            #     f"Focus only on this specific person and describe their interaction with surrounding objects in the image.\n"
            #     f"Ignore other people or objects that are not directly related to this person.\n"
            #     f"Question: Which of the following best describes what the person in the red box is doing?\n"
            # )
            

        elif prompt_box_type == 'none':
                prompt1  = ( 
                    f"Question: Which of the following properly describes the interactions in the image  <image>?\n"
                )
    # prompt1 = prompt1 + (
    #     f"Choices:\n"
    #     + "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(choices)]) + "\n"
    #     f"Hint: Please reply with the letter(s) only (e.g. A, B, C), not the full \"(A)...\" or \"A. ...\" text.\n"
    #     "You may select multiple answers if applicable."
    # )
    renamed_choices = [i if 'no_interaction' not in i else 'no_interaction with' + i[i.index(" a/an ") + 5:] for i in choices]
    
    if number_choice is False:
        prompt1 = prompt1 + (
                f"Choices:\n"
                + "\n".join([f"({chr(65+i)}) {opt}" for i, opt in enumerate(renamed_choices)]) + "\n"
                
            )
    else:
        prompt1 = prompt1 + (
            f"Choices:\n"
            + "\n".join([f"({i+1}) {opt}" for i, opt in enumerate(renamed_choices)]) + "\n"
            "IMPORTANT: Reply with the number(s) ONLY, separated by commas if multiple (e.g. 1,2).\n  "
            "For example, if correct answers are (1) and (2), your output must be: 1,2  \n"
            "Do NOT include any brackets or other symbols. \n"
        )
        return prompt1

    if reasoning == 'v1':
        prompt1 = prompt1 + (
            "Important: First provide your reasoning under a section labeled \"Reasoning:\".\n"
            "Then, on a new line, give your final prediction wrapped inside <answer>...</answer> in the format: letter(s) only (e.g., A,B,C).\n"
            "Do not output the full option text. You may select multiple answers if applicable.\n"
            "For example:\n"
            "Reasoning: …\n"
            "<answer>A,B</answer>"
        )
    elif reasoning == 'v2':
        prompt1 = prompt1 + (
            "Important: First provide your final prediction wrapped inside <answer>...</answer> in the format: letter(s) only (e.g., A,B,C). "
            "Do not output the full option text. \n"
            "Then, on a new line, give your reasoning under a section labeled \"Reasoning:\".\n"
            "You may select multiple answers if applicable.\n"
            "For example:\n"
            "<answer>A,B</answer>\n"
            "Reasoning: \u2026"
        )
    elif answer_tag == None:
        prompt1 = prompt1 + (
            "IMPORTANT: Reply with the letter(s) ONLY, separated by commas if multiple (e.g. A,B).\n  "
            "For example, if correct answers are (A) and (B), your output must be: A,B  \n"
            "Do NOT include any brackets or other symbols. \n"
        )
        # (
        #     "\nHint: Please reply with the letter(s) only, separated by commas (e.g. A,B ), not the full text.\n"
        #     "You may select multiple answers if applicable."
        # )
    elif answer_tag == "answer":
        prompt1 = prompt1 + (
            "\nHint: Please reply with letter(s) only, separated by commas (e.g. A,B ).\n Wrap your answer like this:\n<answer>A,B</answer>\n"
            "You may select multiple answers if applicable."
        )


    return prompt1


#### new benchmark questions prompt for object detection
def newbench_detection_question(prompt_box_type):
    if prompt_box_type == 'person':
        prompt =  (
            "Provide the bounding box coordinates for every single person in the input image. \n"
            " The box coordinates represent as [x1, y1, x2, y2], \n"
            ' where x is the horizontal pixel coordinate from the left edge,\n'
            ' and y is the vertical pixel coordinate from the top edge.\n'
            "Return the detection results in JSON format strictly. \n\n"
            "For example:\n"
            # "{ \"boxes\": [box1, box2, ...]}\n"
            "{ \"boxes\": [[32, 109, 644, 418], [517, 0, 644, 23], [100, 50, 160, 200]]}\n"
        )
    elif prompt_box_type == 'all':
        prompt = (
                    "Provide the bounding box coordinates for **all visible objects and humans** in the input image based on the following object list:\n"
                    f"{OBJ_IDX_TO_OBJ_NAME}\n"
                    " The box coordinates represent as [x1, y1, x2, y2], \n"
                    ' where x is the horizontal pixel coordinate from the left edge,\n'
                    ' and y is the vertical pixel coordinate from the top edge.\n'
                    "Return the detection results in JSON format strictly. \n\n"
                    "For example:\n"
                    "{ \"boxes\": [[32, 109, 644, 418], [517, 0, 644, 23], [100, 50, 160, 200]],\n"
                    "  \"labels\": [\"person\", \"bench\", \"cup\"] }\n"
                    "Only include objects from the given list. Ensure the output is a valid JSON dictionary without additional comments.\n"
                    "Ensure the length of the boxes and labels arrays are equal.\n"
                )
    return prompt


#### new benchmark questions prompt for object detection
def newbench_pre_question_imgsize():

    # prompt =  (
    #     "Please provide the coordinates for the bottom right point of the input image. \n"
    #     " The coordinates represent as [width, height], \n"
    #     "Return the results in JSON format strictly. \n\n"
    #     "For example:\n"
    #     "```json\n[638, 415]\n```"
    # )

    prompt = (    
        "Please provide the coordinates for the bottom-right point of the input image. "
        "Assume the coordinate system origin is at the top-left of the image, "
        "with x increasing to the right and y increasing downward. "
        "Return the coordinates as [width, height] in JSON format strictly. "
        "For example:\n"
        "```json\n[638, 415]\n```"
    )

    return prompt


#### parsing new benchmark detection answers
def parse_detection_answer(prompt_box_type, responses):
    match = re.search(r'```json\s*(\{.*\})\s*```', responses, re.DOTALL)
    if match:
        json_str = match.group(1)
        try:
            det_result = json.loads(json_str)

            if prompt_box_type == 'all':
                if "boxes" not in det_result  or "labels" not in det_result:
                    print("❌failed detection generation")
                    return []
                if isinstance(det_result['boxes'], list) is False or isinstance(det_result['labels'], list) is False:
                    print("❌failed detection generation")
                    return []
            
                if len(det_result['boxes']) != len(det_result['labels']):
                    print("❌failed detection generation")
                    return []
                det_result['scores'] = [1.0] * len(det_result['boxes'])  ## set all scores to 1.0
                det_result['labels'] = [OBJ_IDX_TO_OBJ_NAME.index(lb) if lb in OBJ_IDX_TO_OBJ_NAME else -1 for lb in det_result['labels']]
            if prompt_box_type == 'person':
                if "boxes" not in det_result:
                    print("❌failed detection generation")
                    return []
                if isinstance(det_result['boxes'], list) is False:
                    print("❌failed detection generation")
                    return []
            
        except json.JSONDecodeError:
            return []
    else:
        return []
    return det_result


#### parsing new benchmark detection answers
def parse_imgsize_answer(responses):
    match = re.search(r'```json\s*(\[.*\])\s*```', responses, re.DOTALL)
    if match:
        try:
            json_str = match.group(1).strip()
            img_size = json.loads(json_str)
            if isinstance(img_size, list):
                img_size_final = torch.tensor(img_size).squeeze(0)
            else:
                return []
            if len(img_size_final) == 2 or len(img_size_final.squeeze()) == 2:
                resized_width, resized_height = img_size_final.squeeze()
            else:
                return []
        except:
            return []
    else:
        return []


    return resized_width, resized_height



#### match the detection with GT boxes
def generate_candidate_pairs(det, human_only, prompt_box_type, box_score_thresh=0.5, min_instances=1, max_instances=60, human_idx=0):
    """
    Generate candidate human-object pairs using detection results.
    
    The detection input is expected to be a list whose first element is a dictionary
    with keys 'boxes', 'scores', 'labels', etc.
    
    Parameters:
      - detection: the detection result for one image.
      - box_score_thresh: the minimum score required for a detection to be kept.
      - min_instances: the minimum number of instances to retain for each category (human or object).
      - max_instances: the maximum number of instances to retain for each category.
      - human_idx: the label index that corresponds to a human (subject).
      - faster_rcnn: a flag indicating whether the detection results are in the order (boxes, scores, labels).
                     If False, we expect the order to be (scores, labels, boxes).
    
    Returns:
      A string listing candidate pairs for interaction evaluation.
    """
    # Check input format: if detection is a list and its first element is a dict.
    bx = det.get("boxes")

    if not torch.is_tensor(bx):
        bx = torch.tensor(bx, dtype=torch.float32)
        if len(bx.shape) == 1:
            bx = bx.unsqueeze(0)

    if prompt_box_type == "all":
        sc = det.get("scores")
        lb = det.get("labels")
        if not torch.is_tensor(sc):
            sc = torch.tensor(sc, dtype=torch.float32)
        if not torch.is_tensor(lb):
            lb = torch.tensor(lb, dtype=torch.float32)
    else:
        sc = torch.ones((len(bx)), dtype=torch.float32)
        lb = torch.zeros((len(bx)), dtype=torch.float32)

    # Ensure that bx, sc, and lb are tensors; they will be used in NMS and filtering.
    # (Assumes they are already tensors; if not, you may need to convert them.)
    # Apply NMS with IoU threshold 0.5.

    keep = batched_nms(bx, sc, lb, 0.5)
    sc = sc[keep]
    lb = lb[keep]
    bx = bx[keep]
    
    # Filter detections by score threshold.
    keep_score = torch.nonzero(sc >= box_score_thresh).squeeze(1)
    sc = sc[keep_score]
    lb = lb[keep_score]
    bx = bx[keep_score]
    
    # Determine which detections are human (subject) based on label equality.
    is_human = lb == human_idx
    hum = torch.nonzero(is_human).squeeze(1)
    obj = torch.nonzero((lb >= 0)).squeeze(1)  # all non-human detections

    n_human = is_human.sum()
    n_object = len(sc) 

    # Select human detections.
    if n_human < min_instances:
        if len(hum) > 0:
            _, idx = sc[hum].sort(descending=True)
            keep_h = hum[idx[:min_instances]]
        else:
            keep_h = torch.tensor([], dtype=torch.long)
    elif n_human > max_instances:
        _, idx = sc[hum].sort(descending=True)
        keep_h = hum[idx[:max_instances]]
    else:
        keep_h = hum


    if human_only is True:
        candidate_pairs = []
        paired_id = []
        for h in keep_h:
            subj_bbox = bx[h].tolist()
            subj_score = sc[h].item()
            candidate_pairs.append([(subj_score, subj_bbox)])
            paired_id.append(h.item())

        return candidate_pairs, paired_id


    # Select object detections.
    if n_object < min_instances:
        if len(obj) > 0:
            _, idx = sc[obj].sort(descending=True)
            keep_o = obj[idx[:min_instances]]
        else:
            keep_o = torch.tensor([], dtype=torch.long)
    elif n_object > max_instances:
        _, idx = sc[obj].sort(descending=True)
        keep_o = obj[idx[:max_instances]]
    else:
        keep_o = obj

    # Generate candidate pairs by pairing each human (subject) with each object.
    candidate_pairs = []
    paired_id = []
    for h in keep_h:
        for o in keep_o:
            subj_bbox = bx[h].tolist()
            subj_score = sc[h].item()
            subj_label = "person"  # since human detection
            obj_bbox = bx[o].tolist()
            obj_score = sc[o].item()
            # Map object label (assumed integer) to a string via OBJ_IDX_TO_OBJ_NAME.
            label_obj = int(lb[o].item())
            try:
                obj_label = OBJ_IDX_TO_OBJ_NAME[label_obj]
            except KeyError:
                obj_label = f"Unknown object index: {label_obj}"
                continue
            candidate_pairs.append([(subj_score, subj_bbox), (obj_score, obj_bbox, obj_label)])
            paired_id.append((h.item(), o.item()))

    return candidate_pairs, paired_id


#### instance F1 evaluation for the prediction
def mllm_instancef1_eval(hoi_det_questioni, response_process_list, f1_per_question, macro_f1_dict, all_ans_per_question, file, acc_top1, acc_fullmatch):
    hoi_det_questioni_keys = [i for i in hoi_det_questioni]
    if "QA_0" not in hoi_det_questioni_keys:
        hoi_det_questioni_qa = {"QA_0": hoi_det_questioni}
    else:
        hoi_det_questioni_qa = hoi_det_questioni
    
    for idx, qli in enumerate(hoi_det_questioni_qa):
        gti = hoi_det_questioni_qa[qli]['gt_choices']
        incorrect_i = hoi_det_questioni_qa[qli]['wrong_choices']
    
        tp_qi = 0
        fp_qi = 0
        gt_qi = 0
        pred_qi = 0

        if qli not in response_process_list or len(response_process_list[qli]) == 0:
            all_ans_per_question['full_gt'] += len(gti)  ## full_gt
            f1 = 0 if len(gti) > 0 else 1
            f1_per_question.append(f1)
            if len(gti) > 0:
                for gtii in gti:
                    if gtii not in macro_f1_dict:
                        macro_f1_dict[gtii] = {'tp': 0, 'fp': 0, 'gt': 0, 'tn': 0}
                    macro_f1_dict[gtii]['gt'] = macro_f1_dict[gtii]['gt'] + 1
            continue

        rpi = response_process_list[qli]
        
        ###### each gt interaction answer considers separately
        gt_pool = gti.copy()
        flag_firstmatch = 0
        for idx_rpi, rpii in enumerate(rpi):
            if rpii not in macro_f1_dict:
                macro_f1_dict[rpii] = {'tp': 0, 'fp': 0, 'gt': 0, 'tn': 0}

            if rpii in gt_pool:
                all_ans_per_question['tp'].append(file +"_"+str(idx) + "_" + str(idx_rpi)) ## tp
                tp_qi += 1
                all_ans_per_question['full_pred'] += 1  ## full_pred 
                macro_f1_dict[rpii]['tp'] += 1 
                if flag_firstmatch == 0:
                    acc_top1 += 1
                    flag_firstmatch = 1
                gt_pool.remove(rpii)
            elif rpii in incorrect_i:
                flag_firstmatch = 1
                all_ans_per_question['fp'].append(file +"_"+str(idx) + "_" + str(idx_rpi)) ## fp
                fp_qi += 1
                all_ans_per_question['full_pred'] += 1  ## full_pred  
                macro_f1_dict[rpii]['fp'] += 1
            else:
                all_ans_per_question['ood'].append(file +"_"+str(idx) + "_" + str(idx_rpi)) ## ood
            
            pred_qi += 1
        all_ans_per_question['full_gt'] += len(gti)  ## full_gt
        gt_qi = len(gti)

        ## full match
        true_pred = [i for i in rpi if i in gti or i in incorrect_i]
        if len(true_pred)> 0 and len(true_pred) == len(gti) and set(true_pred) == set(gti):
            acc_fullmatch += 1


        for gtii in gti:
            if gtii not in macro_f1_dict:
                macro_f1_dict[gtii] = {'tp': 0, 'fp': 0, 'gt': 0, 'tn': 0}
            macro_f1_dict[gtii]['gt'] += 1
        
        for incorrect_ii in incorrect_i:
            if incorrect_ii in macro_f1_dict:
                macro_f1_dict[incorrect_ii]['tn'] += 1

        prec_qi = tp_qi / (tp_qi + fp_qi) if (tp_qi + fp_qi) > 0 else 0
        recall_qi = tp_qi / gt_qi if gt_qi > 0 else 0
        f1_qi = 2 * (prec_qi * recall_qi) / (prec_qi + recall_qi) if (prec_qi + recall_qi) > 0 else 0
        f1_per_question.append(f1_qi)

    return f1_per_question, macro_f1_dict, all_ans_per_question, acc_top1, acc_fullmatch


#### macro F1 evaluation for the prediction
def mllm_macrof1_eval(macro_f1_dict, macro_f1_dict_hoicls, dataset = 'hicodet'):
    # macro_f1_dict_hoicls = {i: {'tp': 0, 'fp': 0, 'gt': 0} for i in range(args.num_hoi_cls)}
    for lbi in macro_f1_dict:
        if lbi == "None":
            continue

        if " a/an " in lbi:
            act, obj = lbi.split(" a/an ")
            act = act.replace(" ", "_")
            obj = obj.replace(" ", "_")
            if "no_interaction" in act:
                act = "no_interaction"
        else:
            act = lbi
            obj = "person"
        if (act, obj) not in macro_f1_dict_hoicls:
            continue
       
        macro_f1_dict_hoicls[(act, obj)]['tp'] += macro_f1_dict[lbi]['tp']
        macro_f1_dict_hoicls[(act, obj)]['fp'] += macro_f1_dict[lbi]['fp']
        macro_f1_dict_hoicls[(act, obj)]['gt'] += macro_f1_dict[lbi]['gt']

  
    # if dataset != 'hicodet':   ## hicodet evaluation only considers 600 classes
    macro_f1_dict_hoicls = {
                k: v for k, v in macro_f1_dict_hoicls.items()
                if not (v['tp'] == 0 and v['fp'] == 0 and v['gt'] == 0 and v['tn'] == 0)
            }
    macro_f1_list = {}
    prec_list = []
    rec_list = []

    for lbi in macro_f1_dict_hoicls:       

        tpi = macro_f1_dict_hoicls[lbi]['tp']
        fpi = macro_f1_dict_hoicls[lbi]['fp']
        gti = macro_f1_dict_hoicls[lbi]['gt']
        if (tpi+fpi) == 0 and gti == 0:
            preci = 1
            reci = 1
            f1i = 1
        elif (tpi+fpi > 0) and gti == 0:
            preci = 0
            reci = 0
            f1i = 0
        elif (tpi+fpi) == 0 and gti > 0:
            preci = 0
            reci = 0
            f1i = 0
        elif tpi == 0 and gti > 0:
            preci = 0
            reci = 0
            f1i = 0
        else:
            preci = tpi / (tpi+fpi)
            reci = tpi / gti
            f1i = 2 * (preci * reci) / (preci + reci)
        macro_f1_list[lbi] = f1i
        prec_list.append(preci)
        rec_list.append(reci)
    return macro_f1_dict_hoicls, macro_f1_list, prec_list, rec_list


#### calculate the box IoU
def match_gtbox(sub_box, obj_box, obj_label, sub_box_gt, obj_box_gt, prompt_box_type, box_scale_factor, gt_obj_label = None):


    h_str_ho_det_box_i = None
    o_str_ho_det_box_i = None
    gt_box_match = True
    if prompt_box_type != 'none':
        if isinstance(sub_box, torch.Tensor) is False:
            sub_box = torch.tensor(sub_box, dtype=torch.float32).unsqueeze(0)
        elif len(sub_box.shape) == 1:
            sub_box = sub_box.unsqueeze(0)
        if isinstance(sub_box_gt, torch.Tensor) is False:
            sub_box_gt = torch.tensor(sub_box_gt, dtype=torch.float32)
            if box_scale_factor is not None:
                sub_box_gt = (sub_box_gt * box_scale_factor).unsqueeze(0)
            else:
                sub_box_gt = sub_box_gt.unsqueeze(0)
        elif len(sub_box_gt.shape) == 1:
            sub_box_gt = sub_box_gt.unsqueeze(0)

        h_str_ho_det_box_i = [round(sub_box[0][0].item(), 1), round(sub_box[0][1].item(), 1), round(sub_box[0][2].item(), 1), round(sub_box[0][3].item(), 1)]

        ## calculate the GT IoU
        gt_h_box_match = (box_iou(sub_box_gt, sub_box) > 0.5)
        gt_box_match = gt_h_box_match

    if prompt_box_type == 'all':
        if isinstance(obj_box, torch.Tensor) is False:
            obj_box = torch.tensor(obj_box, dtype=torch.float32).unsqueeze(0)
        elif len(obj_box.shape) == 1:
            obj_box = obj_box.unsqueeze(0)
        if isinstance(obj_box_gt, torch.Tensor) is False:
            obj_box_gt = torch.tensor(obj_box_gt, dtype=torch.float32)
            if box_scale_factor is not None:
                obj_box_gt = (obj_box_gt * box_scale_factor).unsqueeze(0)
            else:
                obj_box_gt = obj_box_gt.unsqueeze(0)
        elif len(obj_box_gt.shape) == 1:
            obj_box_gt = obj_box_gt.unsqueeze(0)
                        
        o_str_ho_det_box_i = [round(obj_box[0][0].item(), 1), round(obj_box[0][1].item(), 1), round(obj_box[0][2].item(), 1), round(obj_box[0][3].item(), 1)]

        ## calculate the GT IoU
        try:
            if isinstance(obj_label, list):
                label_bool = []
                for i in obj_label:
                    if i.replace(" ", "_") == gt_obj_label.replace(" ", "_"):
                        label_bool.append(True)
                    else:
                        label_bool.append(False)
                gt_o_box_match = (box_iou(obj_box_gt, obj_box) > 0.5) & torch.tensor(label_bool)
            else:
                gt_o_box_match = (box_iou(obj_box_gt, obj_box) > 0.5) & (obj_label.replace(" ", "_") == gt_obj_label.replace(" ", "_"))
        except:
            gt_o_box_match = (box_iou(obj_box_gt, obj_box) > 0.5) & (False)
        gt_box_match = gt_h_box_match & gt_o_box_match

    return gt_box_match, h_str_ho_det_box_i, o_str_ho_det_box_i


