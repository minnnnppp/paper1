# main.py

import yaml
import os
import sys
import pickle
import tensorflow as tf
import numpy as np
import pandas as pd

print("\n>>> Initializing SuReFAR (Multi-Summarization Fusion) Pipeline...")
# Add root path for module imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_processing import run_data_pipeline
from src.textrank import load_glove
from src.trainer import run_training_pipeline
from model.proposed import build_surefar_model

def load_config(config_path="src/config.yaml"):
    """Load configuration file (YAML)"""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at: {config_path}")
    with open(config_path, "r", encoding='utf-8') as f:
        return yaml.safe_load(f)

def main():
    print("="*60)
    print("SuReFAR: Summarized Reviews Fusion for Adaptive Recommendation")
    print("="*60)

    config = load_config("src/config.yaml")
    
    # Hardware check (GPU availability)
    gpus = tf.config.list_physical_devices('GPU')
    print(f"[SYSTEM] Hardware Check: Running on {'GPU' if gpus else 'CPU'}")

    # Phase 1: Data Pipeline (Preprocessing & Summary Extraction)
    output_data_path = os.path.join(config['data']['save_path'], config['data']['final_file'])
    tokenizer_path = os.path.join(config['data']['save_path'], "tokenizer.pkl")

    if not os.path.exists(output_data_path):
        print("\n[PHASE 1] Starting Data Pipeline (TextRank & BART)...")
        # Fetch dataframe along with tokenizer and calculated average length
        df, tokenizer, dynamic_max_len = run_data_pipeline(config)
        
        # Save metadata to skip preprocessing in future runs
        with open(tokenizer_path, "wb") as f:
            pickle.dump({'tokenizer': tokenizer, 'max_len': dynamic_max_len}, f)
    else:
        print("\n[PHASE 1] Loading existing processed data...")
        df = pd.read_pickle(output_data_path)
        
        # Load saved tokenizer and sequence length
        with open(tokenizer_path, "rb") as f:
            saved_meta = pickle.load(f)
            tokenizer = saved_meta['tokenizer']
            dynamic_max_len = saved_meta['max_len']
        print(f"[INFO] Loaded tokenizer and dynamic_max_len: {dynamic_max_len}")

    # Phase 2: Model Architecture (Construct GloVe Embedding Matrix)
    print("\n[PHASE 2] Building SuReFAR Model Architecture")
    
    # Load GloVe file and create matrix
    glove_dict = load_glove(config['data']['glove_path'])
    vocab_size = len(tokenizer.word_index) + 1
    embedding_matrix = np.zeros((vocab_size, 300))
    
    for word, i in tokenizer.word_index.items():
        if word in glove_dict:
            embedding_matrix[i] = glove_dict[word]
            
    # Build model (Matching arguments with the modified proposed.py)
    model = build_surefar_model(config, vocab_size, embedding_matrix, dynamic_max_len)
    
    # Check model architecture (Verify the sequence length of the glove_embedding layer!)
    model.summary()

    # Phase 3: Training & Evaluation (Start Training)
    print("\n[PHASE 3] Starting SuReFAR Training Engine")
    try:
        # Proceed with training using column names like 'user_ext_seq', 'user_abs_vec' in the dataframe
        run_training_pipeline(model, df, config)
    except Exception as e:
        print(f"[CRITICAL ERROR] Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n" + "="*60)
    print("SuReFAR Pipeline Completed Successfully!")
    print("="*60)

if __name__ == "__main__":
    main()