# [CVPR 2026] CrossHOI-Bench: A Unified Benchmark for HOI Evaluation across Vision-Language Models and HOI-Specific Methods

## Paper Links

[arXiv version](https://arxiv.org/abs/2508.18753)

[project page](https://chelsielei.github.io/crosshoibench_page/)

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

### Corrected specialist evaluation

We identified and corrected four inconsistencies in the original specialist evaluation: `no_interaction` label formatting is now normalized; top-5 ranking for target-specific questions is computed over target-matched candidates; final answers are deduplicated as label sets; and benchmark images absent from a prediction file are evaluated as empty answers. The last rule is relevant to the released CMD-SE file, which contains predictions for 1,182 of the 1,274 main-benchmark images.

The corrected results (%) for Setting 1 are:

| Method | Macro-F1 | Instance-F1 | Micro-F1 | EM | Avg. Prec. | Avg. Rec. |
|---|---:|---:|---:|---:|---:|---:|
| ADA-CM | 50.65 | 54.73 | 66.14 | 24.41 | 69.64 | 62.98 |
| CMMP | 50.87 | 54.19 | 65.41 | 23.63 | 68.40 | 62.68 |
| CMD-SE | 40.59 | 44.65 | 59.23 | 18.37 | 69.33 | 51.69 |
| LAIN | 48.02 | 51.95 | 63.27 | 20.02 | 65.76 | 60.96 |
| HOLa | 37.76 | 44.48 | 59.14 | 25.75 | 81.48 | 46.42 |

The corrected results (%) for Setting 3 are:

| Method | Macro-F1 | Instance-F1 | Micro-F1 | EM | Avg. Prec. | Avg. Rec. |
|---|---:|---:|---:|---:|---:|---:|
| ADA-CM | 51.51 | 59.55 | 69.64 | 27.55 | 84.26 | 59.34 |
| CMMP | 51.49 | 58.71 | 69.21 | 27.32 | 83.51 | 59.10 |
| CMD-SE | 42.52 | 52.94 | 64.78 | 21.74 | 82.49 | 53.33 |
| LAIN | 49.65 | 56.78 | 67.15 | 23.08 | 81.59 | 57.06 |
| HOLa | 39.46 | 49.54 | 62.43 | 24.02 | 90.95 | 47.53 |

We thank **Usman Karamat** for independently reproducing the evaluation and bringing the label-formatting, ranking, and duplicate-label issues to our attention.

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
