import os
import argparse
import pandas as pd
import numpy as np

# import module that we created for each step
from data_prep import process_dataset
from TextRank import ( 
    generate_review_summaries, load_glove_dictionary, build_tokenizer, generate_embedding_matrix, process_and_pad_sequences
)
from BART import generate_bart_embeddings
from model_train import ModelTrainer

def main():
    # ==========================================
    # 0. configurations & hyperparameters
    # ==========================================
    parser = argparse.ArgumentParser(description="SuReFAR Full Pipeline")
    parser.add_argument('--domain', type=str, default='amazon', help='amazon or yelp')
    parser.add_argument('--dataset', type=str, default='Books', help='Dataset name (e.g., Books, Yelp)')
    parser.add_argument('--seq_len', type=int, default=118, help='Input sequence length')
    
    args = parser.parse_args()


    DOMAIN = args.domain
    DATASET_NM = args.dataset
    RAW_DIR = 'data/raw'
    SAVE_DIR = 'data/processed'
    
    # 딥러닝 모델 파라미터 세팅
    base_params = {
        'input_dim': args.seq_len,         
        'embedding_dim': 300,     
        'mlp_depth': 2, 
        'mlp_hidden_dim': 128, 
        'dropout_rate': 0.2,
        'learning_rate': 0.001, 
        'batch_size': 256, 
        'epochs': 100,
        'patience': 5,            
        'verbose': True
    }

    # ==========================================
    # 1. data load & prep (Cleaning, 5-core)
    # ==========================================
    print("\n" + "="*50 + "\n[STEP 1] Data Preprocessing\n" + "="*50)
    prep_df = process_dataset(
        file_path=RAW_DIR, 
        domain=DOMAIN, 
        output_path=f"{SAVE_DIR}/{DOMAIN}_{DATASET_NM}_prep.pkl"
    )

    # ==========================================
    # 2. TextRank summary by user/item (ratio=0.6)
    # ==========================================
    print("\n" + "="*50 + "\n[STEP 2] TextRank Summarization\n" + "="*50)
    user_summary_df, item_summary_df = generate_review_summaries(
        df=prep_df, 
        user_col='user_id', item_col='item_id', review_col='clean_review', 
        ratio=0.6, save_path=SAVE_DIR
    )

    # ==========================================
    # 3. GloVe embedding & Tokenizing (for TextRank)
    # ==========================================
    print("\n" + "="*50 + "\n[STEP 3] Tokenizing & GloVe Embedding\n" + "="*50)
    MAX_WORDS = 50000
    MEAN_SEQ_LEN = base_params['input_dim']
    
    glove_dict = load_glove_dictionary(dim=base_params['embedding_dim'])
    tokenizer, total_words, word_index = build_tokenizer(user_summary_df, item_summary_df, max_words=MAX_WORDS)
    
    # embedding matrix
    glove_matrix = generate_embedding_matrix(word_index, glove_dict, total_words, dim=base_params['embedding_dim'])
    
    # padding & truncating (max_len=MEAN_SEQ_LEN)
    user_summary_df = process_and_pad_sequences(user_summary_df, 'user_summary', tokenizer, MEAN_SEQ_LEN, MAX_WORDS)
    item_summary_df = process_and_pad_sequences(item_summary_df, 'item_summary', tokenizer, MEAN_SEQ_LEN, MAX_WORDS)
    
    # assign new column names for clarity in later merging steps
    user_summary_df.rename(columns={'user_summary_padded_sequences': 'user_textrank'}, inplace=True)
    item_summary_df.rename(columns={'item_summary_padded_sequences': 'item_textrank'}, inplace=True)

    # ==========================================
    # 4. BART embedding (768차원)
    # ==========================================
    print("\n" + "="*50 + "\n[STEP 4] Generating BART Embeddings\n" + "="*50)
    # generate BART embeddings from userReviews / itemReviews
    user_bart_df = generate_bart_embeddings(user_summary_df, text_col='Reviews_origin', batch_size=8)
    item_bart_df = generate_bart_embeddings(item_summary_df, text_col='Reviews_origin', batch_size=8)
    
    user_bart_df.rename(columns={'bart_embedding': 'user_bart'}, inplace=True)
    item_bart_df.rename(columns={'bart_embedding': 'item_bart'}, inplace=True)

    # ==========================================
    # 5. merge dataframe for model training (full_df)
    # ==========================================
    print("\n" + "="*50 + "\n[STEP 5] Merging Features into Full Dataset\n" + "="*50)
    # merge user features (TextRank + BART) into prep_df
    full_df = pd.merge(prep_df, user_bart_df[['user_id', 'user_textrank', 'user_bart']], on='user_id', how='left')
    # merge item features (TextRank + BART) into full_df
    full_df = pd.merge(full_df, item_bart_df[['item_id', 'item_textrank', 'item_bart']], on='item_id', how='left')
    
    # leave only necessary columns and drop rows with any missing values (if any)
    full_df = full_df[['user_id', 'item_id', 'rating', 'user_textrank', 'item_textrank', 'user_bart', 'item_bart']].dropna()
    print(f"Final Dataset Shape: {full_df.shape}")

    # save the full dataset for model training 
    full_df.to_pickle(f"{SAVE_DIR}/{DOMAIN}_{DATASET_NM}_data.pkl")
    print(f"✅ Saved full dataset to {SAVE_DIR}/{DOMAIN}_{DATASET_NM}_data.pkl")

    # ==========================================
    # 6. Model Training & Evaluation (Ablation Study)
    # ==========================================
    print("\n" + "="*50 + "\n[STEP 6] Model Training & Evaluation\n" + "="*50)
    
    # 실험 세팅 정의 (논문에 들어갈 비교 실험들)
    experiments = [
        {'name': 'SuReFAR (GMU + Attn)', 'fusion_version': 'gmu', 'use_attention': True, 'textrank_bool': True, 'bart_bool': True},
        {'name': 'SuReFAR (Concat + Attn)', 'fusion_version': 'concat', 'use_attention': True, 'textrank_bool': True, 'bart_bool': True},
        {'name': 'Ablation (No Attn)', 'fusion_version': 'gmu', 'use_attention': False, 'textrank_bool': True, 'bart_bool': True},
        {'name': 'TextRank Only', 'fusion_version': 'single', 'use_attention': False, 'textrank_bool': True, 'bart_bool': False},
        {'name': 'BART Only', 'fusion_version': 'single', 'use_attention': False, 'textrank_bool': False, 'bart_bool': True}
    ]

    results_log = []

    for exp in experiments:
        print(f"\n▶ Running Experiment: {exp['name']}")
        current_params = {**base_params, **exp} # 공통 파라미터에 현재 실험 파라미터 덮어쓰기
        
        # generate instance of model trainer with current experiment settings
        trainer = ModelTrainer(
            full_df=full_df, 
            params=current_params,
            user_embedding_matrix=glove_matrix, 
            item_embedding_matrix=glove_matrix
        )
        
        # data split, model build, train, evaluate
        metrics, _ = trainer.run_pipeline()
        
        results_log.append({
            'Model': exp['name'],
            'MAE': metrics['MAE'],
            'RMSE': metrics['RMSE']
        })

    # ==========================================
    # 7. print and save results
    # ==========================================
    results_df = pd.DataFrame(results_log)
    print("\n" + "="*50 + "\n🏆 All Experiments Completed! Benchmark Summary:\n" + "="*50)
    print(results_df.to_markdown(index=False))
    
    results_df.to_csv(f'{SAVE_DIR}/benchmark_results.csv', index=False)
    print(f"\nResults saved to {SAVE_DIR}/benchmark_results.csv")

if __name__ == "__main__":
    main()