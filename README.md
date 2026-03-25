# SuReFAR

![Last Commit](https://img.shields.io/github/last-commit/kkkimsuji/SuReFAR?style=flat-square)

This repository contains the official implementation of the following paper:
> **Enhancing Recommendation with Integration of Extractive and Abstractive Summarization**  
> Minkyung Park, **Suji Kim**, Xinzhe Li, Seonu Park, Jaekyeong Kim  
> *Electronics*, under review

## Overview

SuReFAR (**Summarized Reviews Fusion for Adaptive Recommendation**) is a review-based recommender designed to improve rating prediction by reducing noise in raw review text and fusing complementary summary representations. Instead of relying on full reviews or an emotion/semantic dual-channel setup, SuReFAR summarizes user and item review sets using two different strategies: **TextRank** for extractive summarization and **BART** for abstractive summarization. These two summary types are then refined with an **attention mechanism** and adaptively integrated through a **Gated Multimodal Unit (GMU)** to produce a personalized fused representation for rating prediction.

The framework consists of three main modules. First, the **Review Summarization Module** generates extractive and abstractive summaries from user and item review sets. Second, the **Summary Fusion Module** applies attention to emphasize salient information and uses GMU to learn how much each summary type should contribute for a given prediction. Third, the **Rating Prediction Module** takes the fused representation and predicts the final user-item rating. In this way, SuReFAR leverages the complementary strengths of extractive and abstractive summarization to better capture user preferences and item characteristics while filtering out redundant or recommendation-irrelevant review content.

## Environment & Requirements

This project is implemented in **Python 3.8+**. The package versions below are aligned with the current `requirements.txt` in this repository.

### 1. Key Dependencies

| Category | Library | Version | Description |
| :--- | :--- | :--- | :--- |
| **Deep Learning** | `tensorflow` / `keras` | `2.21.0` / `3.13.2` | Implements the core recommendation model, including the fusion and rating prediction layers. |
| **NLP** | `transformers` | `5.3.0` | Provides the pretrained transformer stack used for abstractive summarization with BART. |
| **NLP Backend** | `torch` | `2.11.0` | Serves as the backend for Hugging Face transformer models. |
| **Analysis** | `pandas` | `3.0.1` | Handles dataset loading, preprocessing, filtering, and review aggregation. |
| **Matrix / Numerical** | `numpy` | `2.4.3` | Supports numerical operations for embeddings, tensors, and preprocessing. |
| **Graph / Algorithms** | `networkx` | `3.6.1` | Useful for graph-based extractive summarization such as TextRank. |
| **ML Tools** | `scikit-learn` | `1.8.0` | Supports data splitting and evaluation metrics such as MAE and RMSE. |

### 2. Utility Libraries

- **`PyYAML` (`6.0.3`)**: Parses `config.yaml` for experiment settings and file paths. :contentReference[oaicite:4]{index=4}
- **`pyarrow` (`23.0.1`)**: Supports efficient columnar data storage and loading. :contentReference[oaicite:5]{index=5}
- **`tqdm` (`4.67.3`)**: Displays progress bars during preprocessing, summarization, and training. :contentReference[oaicite:6]{index=6}
- **`h5py` (`3.14.0`)**: Handles model weight serialization for Keras/TensorFlow workflows. :contentReference[oaicite:7]{index=7}
- **`huggingface_hub` (`1.7.2`)**: Manages downloading and caching of pretrained transformer checkpoints. :contentReference[oaicite:8]{index=8}


## Repository Structure

The repository is organized as follows to ensure a clear workflow from data preprocessing to model evaluation:

```text
SuReFAR/
├── main.py                  # Main pipeline entry point
├── config.yaml              # Global configuration file
├── requirements.txt         # Python dependencies
├── README.md                # Project documentation
│
├── model/
│   ├── __init__.py
│   └── proposed.py          # SuReFAR model implementation
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py   # Data loading, filtering, split generation
│   ├── trainer.py           # Training and evaluation
│   ├── textrank.py          # Extractive summarization module
│   └── bart.py              # Abstractive summarization module
│
└── data/
    ├── raw/                 # Raw review datasets
    └── processed/           # Preprocessed outputs
```

## How to Run
### Installation & Environment Setup
We recommend using a virtual environment to manage dependencies.

```bash
python -m venv .venv

pip install -r requirements.txt
```

### Configuration
You can customize hyperparameters and file paths in the centralized config file.
- File Path: ```config.yaml```

### Train and Evaluate
Once the environment and data are ready, execute the following command to start the full workflow (Preprocessing → Training → Evaluation):
```
python main.py
```


## Model Description

**SuReFAR (Summarized Reviews Fusion for Adaptive Recommendation)** is a review-based recommender designed to improve rating prediction by reducing noise in raw review text and integrating complementary summary representations. Instead of directly using full reviews, SuReFAR summarizes user and item review sets into key information and learns adaptive fused representations for recommendation.

The framework consists of three modules.

### 1. Review Summarization Module
SuReFAR generates two types of summaries from user and item review sets:

- **Extractive summarization** using **TextRank**, which selects salient sentences while preserving the original review semantics.
- **Abstractive summarization** using **BART**, which generates coherent summaries that capture broader contextual meaning.

By combining both methods, the model reduces summarization bias and preserves complementary information from the original reviews.

### 2. Summary Fusion Module
The extractive and abstractive summary representations are first refined with an **attention mechanism** to emphasize recommendation-relevant information. Then, a **Gated Multimodal Unit (GMU)** adaptively controls the contribution of each summary type for each prediction, producing a fused user-item representation.

### 3. Rating Prediction Module
The fused representation is passed into a neural prediction layer to estimate the final user-item rating.

In summary, SuReFAR leverages the complementary strengths of **TextRank** and **BART**, while **attention** and **GMU** enable adaptive fusion of summarized review information for more accurate recommendation.

<img width="829" height="483" alt="image" src="https://github.com/user-attachments/assets/d37a9ae3-e136-4a4c-8933-f6096bc2e9f7" />

## Experimental Results

The following table summarizes the performance comparison reported in the paper. SuReFAR consistently achieves the best results across the **Books**, **Movie**, and **Yelp** datasets.

<table width="100%" style="border-collapse: collapse; text-align: center; font-family: sans-serif; border: 1px solid #ddd;">
  <thead>
    <tr style="background-color: #f8f9fa; border-bottom: 1px solid #dee2e6;">
      <th rowspan="2" style="padding: 12px; border: 1px solid #ddd;">Model</th>
      <th colspan="2" style="padding: 12px; border: 1px solid #ddd;">Books</th>
      <th colspan="2" style="padding: 12px; border: 1px solid #ddd;">Movie</th>
      <th colspan="2" style="padding: 12px; border: 1px solid #ddd;">Yelp</th>
    </tr>
    <tr style="background-color: #f8f9fa; border-bottom: 2px solid #dee2e6;">
      <th style="padding: 10px; border: 1px solid #ddd;">MAE ↓</th>
      <th style="padding: 10px; border: 1px solid #ddd;">RMSE ↓</th>
      <th style="padding: 10px; border: 1px solid #ddd;">MAE ↓</th>
      <th style="padding: 10px; border: 1px solid #ddd;">RMSE ↓</th>
      <th style="padding: 10px; border: 1px solid #ddd;">MAE ↓</th>
      <th style="padding: 10px; border: 1px solid #ddd;">RMSE ↓</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="padding: 10px; border: 1px solid #ddd;">NCF</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.700</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.902</td>
      <td style="padding: 10px; border: 1px solid #ddd;">1.098</td>
      <td style="padding: 10px; border: 1px solid #ddd;">1.358</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.968</td>
      <td style="padding: 10px; border: 1px solid #ddd;">1.212</td>
    </tr>
    <tr>
      <td style="padding: 10px; border: 1px solid #ddd;">DeepCoNN</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.523</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.766</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.782</td>
      <td style="padding: 10px; border: 1px solid #ddd;">1.085</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.848</td>
      <td style="padding: 10px; border: 1px solid #ddd;">1.077</td>
    </tr>
    <tr>
      <td style="padding: 10px; border: 1px solid #ddd;">NARRE</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.530</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.791</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.780</td>
      <td style="padding: 10px; border: 1px solid #ddd;">1.083</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.842</td>
      <td style="padding: 10px; border: 1px solid #ddd;">1.083</td>
    </tr>
    <tr>
      <td style="padding: 10px; border: 1px solid #ddd;">AENAR</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.519</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.762</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.750</td>
      <td style="padding: 10px; border: 1px solid #ddd;">1.083</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.836</td>
      <td style="padding: 10px; border: 1px solid #ddd;">1.076</td>
    </tr>
    <tr>
      <td style="padding: 10px; border: 1px solid #ddd;">MFNR</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.507</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.776</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.738</td>
      <td style="padding: 10px; border: 1px solid #ddd;">1.093</td>
      <td style="padding: 10px; border: 1px solid #ddd;">0.837</td>
      <td style="padding: 10px; border: 1px solid #ddd;">1.082</td>
    </tr>
    <tr style="background-color: #e6ffed; font-weight: bold;">
      <td style="padding: 10px; border: 1px solid #ddd; color: #1a7f37;">SuReFAR (Ours)</td>
      <td style="padding: 10px; border: 1px solid #ddd; color: #1a7f37;">0.476</td>
      <td style="padding: 10px; border: 1px solid #ddd; color: #1a7f37;">0.757</td>
      <td style="padding: 10px; border: 1px solid #ddd; color: #1a7f37;">0.702</td>
      <td style="padding: 10px; border: 1px solid #ddd; color: #1a7f37;">1.071</td>
      <td style="padding: 10px; border: 1px solid #ddd; color: #1a7f37;">0.825</td>
      <td style="padding: 10px; border: 1px solid #ddd; color: #1a7f37;">1.070</td>
    </tr>
  </tbody>
</table>
