# SuReFAR: Summarized Reviews Fusion for Adaptive Recommendation

[cite_start]Official repository for the paper: **"Enhancing Recommendation with Integration of Extractive and Abstractive Summarization"**[cite: 2].

## 1. Overview
[cite_start]To address the limitations of using full review texts that may contain redundant semantics or noise [cite: 12][cite_start], we propose **SuReFAR** (Summarized Reviews Fusion for Adaptive Recommendation)[cite: 13]. [cite_start]This model predicts ratings by summarizing reviews into key information using a multi-summarization strategy[cite: 13].

## 2. Framework Architecture
[cite_start]The proposed SuReFAR model consists of three main modules[cite: 104]:
* [cite_start]**Review Summarization Module**: Utilizes TextRank and BART to summarize user and item review sets[cite: 105].
* [cite_start]**Summary Fusion Module**: Applies an attention mechanism to extractive and abstractive summary vectors and uses a Gated Multimodal Unit (GMU) to adaptively weight them[cite: 106].
* [cite_start]**Rating Prediction Module**: Predicts user ratings from the fused representation[cite: 107].

[cite_start]*![SuReFAR Framework Architecture](/Users/minky/Desktop/대학원/paper1/framework.png)*

## 3. Datasets
[cite_start]The model was evaluated on the following open datasets[cite: 217]:
* [cite_start]Amazon Books [cite: 218]
* [cite_start]Amazon Movie and TV [cite: 218]
* [cite_start]Yelp (restaurant reviews) [cite: 218]

## 4. Experimental Results
[cite_start]The proposed SuReFAR model consistently outperforms baseline models and captures user preferences more effectively[cite: 16, 260].

| Model | Books (MAE) | Books (RMSE) | Movie & TV (MAE) | Movie & TV (RMSE) | Yelp (MAE) | Yelp (RMSE) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| NCF | 0.700 | 0.902 | 1.098 | 1.358 | 0.968 | 1.212 |
| DeepCoNN | 0.523 | 0.766 | 0.782 | 1.085 | 0.848 | 1.077 |
| NARRE | 0.530 | 0.791 | 0.780 | 1.083 | 0.842 | 1.080 |
| AENAR | 0.519 | 0.762 | 0.750 | 1.083 | 0.836 | 1.076 |
| MFNR | 0.507 | 0.776 | 0.738 | 1.093 | 0.837 | 1.082 |
| **SuReFAR (Ours)** | **0.476** | **0.757** | **0.696** | **1.070** | **0.820** | **1.068** |

[cite_start]*(Note: Data sourced from the comparative baseline experiments[cite: 264].)*

## 5. Citation
If you find this work helpful to your research, please consider citing our paper:
```bibtex
@article{
  title={Enhancing Recommendation with Integration of Extractive and Abstractive Summarization},
  author={Park, Minkyung and Kim, Suji and Li, Xinzhe and Park, Seonu and Kim, Jaekyeong},
  journal={Electronics},
  year={2026}
}