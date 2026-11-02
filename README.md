# Weight Gait Analysis: vision-language model strand

Code and results for the vision-language model (VLM) part of FIT2082 Project 45 (Monash University), by Arnob Saha. The joint project, with Nay Chi Moe Htet, compares three ways of classifying simulated gait (normal, limp, shuffle) in the WeightGait dataset: a 1D-CNN on raw 2D pose, XGBoost on 27 gait features with SHAP, and an off-the-shelf VLM prompted with video frames. This repository holds the VLM strand. The pose models are Nay Chi's work and are not included.

On the 35 held-out clips scored by all three methods:

| Method | Macro-F1 | Accuracy |
|---|---|---|
| 1D-CNN on raw pose | 0.857 | 85.7% |
| XGBoost on 27 features | 0.600 | 60.0% |
| Gemma-4-E4B, best prompt (few-shot) | 0.331 | 42.9% |
| Majority baseline | 0.170 | 34.3% |

## Layout

- `scripts/` holds the pipeline: clip selection and the participant split (`stratified_split.py`, `samples.py`), frame sampling, the four input formats (RGB, skeleton, overlay, montage), the prompts (`prompts_v2.py`), the Gemma and Qwen backends and run scripts, the 27-attribute rating code (`branch_d_*.py`) and the analysis scripts.
- `results/` has every saved model call as one JSON line, with its frames, raw answer and parsed fields, plus the gait attributes measured from pose.
- `report/` holds the tables behind the report's VLM numbers.
- `notebooks/` and `notebook40b2a24df4.ipynb` are the Kaggle notebooks used for GPU inference (two NVIDIA T4s).

## Reproducing the numbers

Every VLM number in the report is recomputed from the saved results without a GPU. From `scripts/`:

```
python macro_f1_table.py       # macro-F1, accuracy, balanced accuracy for every Gemma and Qwen setting
python report_stats.py         # development-set tests with Holm adjustment, held-out McNemar tests
python report_vlm_numbers.py   # bootstrap intervals, probes, confusion matrices, confidence
python -m pytest test_gemma_v2.py -q
```

`head_to_head.py` scores the pose models' clip-level predictions on the same held-out clips. It needs the pose predictions CSV, which is not in this repository.

Running new model calls needs the WeightGait videos and joints, which are licensed for academic use and are not redistributed here. They are available from the dataset authors at https://github.com/ChrisLochhead/GaitMonitor.

## Requirements

Python 3.13 with numpy, pandas, SciPy, scikit-learn, OpenCV and matplotlib. Inference used Hugging Face Transformers on Kaggle.
