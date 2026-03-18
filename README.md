# [CVPR 2026] CrossHOI-Bench: A Unified Benchmark for HOI Evaluation across Vision-Language Models and HOI-Specific Methods

## Paper Links

[arXiv version](https://arxiv.org/abs/2508.18753)

## Dataset 
Download dataset images: [HICO-DET](https://huggingface.co/datasets/zhimeng/hico_det), [V-COCO](https://github.com/fredzzhang/vcoco/tree/cb13e3d3cd74158b41acee09979e25e875c02053), and [SWiG-HOI](https://github.com/scwangdyd/large_vocabulary_hoi_detection)

The downloaded files should be placed as follows. Otherwise, please replace the default path to your custom locations.
```
|- CrossHOI-Bench
|   |- data
|   |   |- hico_20160224_det
|   |       |- annotations
|   |       |- images
|   |   |- mscoco2014
|   |       |- train2014
|   |       |- val2014
|   |   |- swig_hoi
|   |       |- annotations
|   |       |- images_512
|   |       |- test_images_512
:   :      
```


## Dependencies
1. Follow the environment setup in [Qwen](https://huggingface.co/models?search=Qwen/Qwen).

2. Follow the environment setup in [InternVL](https://huggingface.co/collections/OpenGVLab/internvl3).

## Our Benchmark New Annotations
HICO-DET-based main evaluation annotation is included in "hicodet" folder.

V-COCO-based sub-benchmark evaluation annotation is included in "vcoco" folder. 

SWiG-HOI-based sub-benchmark evaluation annotation is included in "swighoi" folder.

## HOI-specific methods predictions
We provide 5 HOI-specific methods' predictions [here](https://huggingface.co/chelsielei/CrossHOI_Bench/tree/main/HOI_pred_hicodet) (ADA-CM, CMMP, CMD-SE, LAIN, HOLa).
Our HICO-DET-based training dataset annotation is provided [here](https://huggingface.co/chelsielei/CrossHOI_Bench/tree/main/hicodet/mllmdata).

## Scripts
### Test Qwen model:
```
bash scripts/script_instruct_newbench_eval_fullqwen.sh
```
### Test InternVL model:
```
bash scripts/script_instruct_newbench_eval_internvl.sh
```
### Test HOI-specific models:
```
bash scripts/script_instruct_newbench_eval_HOI.sh
```

## Citation
If you find our paper and/or code helpful, please consider citing :
```
@inproceedings{
lei2026crosshoi_bench,
title={CrossHOI-Bench: A Unified Benchmark for HOI Evaluation across Vision-Language Models and HOI-Specific Methods},
author={Lei, Qinqian and Wang, Bo and Tan, Robby T.},
booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
year={2026}
}
```

## Acknowledgement
We gratefully thank the authors from [UPT](https://github.com/fredzzhang/upt), [ADA-CM](https://github.com/ltttpku/ADA-CM/tree/main),[CMMP](https://github.com/ltttpku/CMMP), [LAIN](https://github.com/OreoChocolate/LAIN), [HOLa](https://github.com/ChelsieLei/HOLa), [Qwen](https://huggingface.co/models?search=Qwen/Qwen) and [InternVL](https://huggingface.co/collections/OpenGVLab/internvl3) for open-sourcing their code.

