# SuReFAR: Summarized Reviews Fusion for Adaptive Recommendation

Official repository for the paper: **"Enhancing Recommendation with Integration of Extractive and Abstractive Summarization"**.

## 1. Overview
To address the limitations of using full review texts that may contain redundant semantics or noise 
, we propose **SuReFAR** (Summarized Reviews Fusion for Adaptive Recommendation).
This model predicts ratings by summarizing reviews into key information using a multi-summarization strategy.

## 2. Framework Architecture
The proposed SuReFAR model consists of three main modules:
* **Review Summarization Module**: Utilizes TextRank and BART to summarize user and item review sets.
* **Summary Fusion Module**: Applies an attention mechanism to extractive and abstractive summary vectors and uses a Gated Multimodal Unit (GMU) to adaptively weight them.
* **Rating Prediction Module**: Predicts user ratings from the fused representation.


<img width="3194" height="1865" alt="framework" src="https://github.com/user-attachments/assets/3e8e0184-185d-4185-98a2-1e508818eb71" />
[SuReFAR Framework Architecture]

## 3. Datasets
The model was evaluated on the following open datasets:
* Amazon Books 
* Amazon Movie and TV 
* Yelp (restaurant reviews) 

## 4. Experimental Results
The proposed SuReFAR model consistently outperforms baseline models and captures user preferences more effectively.

| Model | Books (MAE) | Books (RMSE) | Movie & TV (MAE) | Movie & TV (RMSE) | Yelp (MAE) | Yelp (RMSE) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| NCF | 0.700 | 0.902 | 1.098 | 1.358 | 0.968 | 1.212 |
| DeepCoNN | 0.523 | 0.766 | 0.782 | 1.085 | 0.848 | 1.077 |
| NARRE | 0.530 | 0.791 | 0.780 | 1.083 | 0.842 | 1.080 |
| AENAR | 0.519 | 0.762 | 0.750 | 1.083 | 0.836 | 1.076 |
| MFNR | 0.507 | 0.776 | 0.738 | 1.093 | 0.837 | 1.082 |
| **SuReFAR (Ours)** | **0.476** | **0.757** | **0.696** | **1.070** | **0.820** | **1.068** |

*(Note: Data sourced from the comparative baseline experiments.)*

## 5. Citation
If you find this work helpful to your research, please consider citing our paper:
```bibtex
@article{
  title={Enhancing Recommendation with Integration of Extractive and Abstractive Summarization},
  author={Minkyung Park, Suji Kim, Xinzhe Li, Seonu Park, and Jaekyeong Kim},
  journal={Electronics},
  year={2026}
}
