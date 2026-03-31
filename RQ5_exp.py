import os
import argparse
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import the ModelTrainer class we created
from model_train import ModelTrainer

def get_sparsity_group(count):
    """
    Classify users/items into groups (G1-G4) based on interaction frequency.
    Please adjust these thresholds according to your paper's criteria.
    """
    if count <= 10:
        return 'G1 (<10)'
    elif count <= 20:
        return 'G2 (10-20)'
    elif count <= 30:
        return 'G3 (20-30)'
    else:
        return 'G4 (>30)'

def evaluate_by_group(trainer, group_col='user_id'):
    """
    Evaluate the trained model by splitting the test set into sparsity groups.
    """
    # 1. Calculate interaction counts from the Training set
    train_counts = trainer.train_df[group_col].value_counts().reset_index()
    train_counts.columns = [group_col, 'interaction_count']
    
    # 2. Assign sparsity group labels
    train_counts['sparsity_group'] = train_counts['interaction_count'].apply(get_sparsity_group)
    
    # 3. Map group information to the Test set
    test_df_with_group = pd.merge(trainer.test_df, train_counts, on=group_col, how='inner')
    
    results = []
    
    # 4. Evaluate the model for each group
    groups = sorted(test_df_with_group['sparsity_group'].unique())
    print(f"\n📊 Evaluating Sparsity Groups based on [{group_col}]...")
    
    for group in groups:
        group_df = test_df_with_group[test_df_with_group['sparsity_group'] == group]
        
        if len(group_df) == 0:
            continue
            
        # Extract model inputs using the method defined in ModelTrainer
        test_x, test_y = trainer._get_model_inputs(group_df)
        
        # Predict and calculate performance metrics
        predictions = trainer.model.predict(test_x, verbose=0)
        mae = mean_absolute_error(test_y, predictions)
        rmse = np.sqrt(mean_squared_error(test_y, predictions))
        
        results.append({
            'Group': group,
            'Test_Samples': len(group_df),
            'MAE': mae,
            'RMSE': rmse
        })
        print(f" - {group}: MAE={mae:.4f}, RMSE={rmse:.4f} (Samples: {len(group_df)})")
        
    return pd.DataFrame(results)

def main():
    # ==========================================
    # 0. Argument Parsing
    # ==========================================
    parser = argparse.ArgumentParser(description="RQ5 Sparsity Experiment")
    parser.add_argument('--domain', type=str, required=True, help='amazon or yelp')
    parser.add_argument('--dataset', type=str, required=True, help='Dataset name (e.g., Books, Yelp)')
    parser.add_argument('--seq_len', type=int, default=118, help='Input sequence length')
    
    args = parser.parse_args()

    DOMAIN = args.domain
    DATASET_NM = args.dataset
    SAVE_DIR = 'data/processed'

    # ==========================================
    # 1. Data Loading & Environment Setup
    # ==========================================
    # Use dynamic paths based on input arguments
    full_data_path = f"{SAVE_DIR}/{DOMAIN}_{DATASET_NM}_data.pkl"
    glove_matrix_path = f"{SAVE_DIR}/glove_embedding_matrix.npy"
    
    print(f"\n🚀 Starting Sparsity Experiment for: {DOMAIN} - {DATASET_NM}")
    
    full_dataset = pd.read_pickle(full_data_path) 
    glove_matrix = np.load(glove_matrix_path)
    
    # Best parameter settings (SuReFAR)
    best_params = {
        'input_dim': args.seq_len, 
        'embedding_dim': 300,
        'mlp_depth': 2, 
        'mlp_hidden_dim': 128, 
        'dropout_rate': 0.2,
        'learning_rate': 0.001, 
        'batch_size': 256, 
        'epochs': 100,
        'patience': 5,
        'fusion_version': 'gmu', 
        'use_attention': True,
        'textrank_bool': True, 
        'bart_bool': True,
        'verbose': False,
        'random_state': 42
    }

    # ==========================================
    # 2. Model Training
    # ==========================================
    print("\n📦 Training the best model for sparsity analysis...")
    trainer = ModelTrainer(
        params=best_params,
        user_embedding_matrix=glove_matrix,
        item_embedding_matrix=glove_matrix
    )
    
    # Instead of load_dataset method, we directly assign the loaded dataset
    trainer.full_df = full_dataset
    
    trainer.split_data()
    trainer.build_and_compile()
    trainer.train()

    # ==========================================
    # 3. Sparsity Group Evaluation
    # ==========================================
    # RQ5-1: User Sparsity Analysis
    user_sparsity_df = evaluate_by_group(trainer, group_col='user_id')
    
    # RQ5-2: Item Sparsity Analysis
    item_sparsity_df = evaluate_by_group(trainer, group_col='item_id')

    # ==========================================
    # 4. Saving Results
    # ==========================================
    user_sparsity_df.to_csv(f'{SAVE_DIR}/{DOMAIN}_{DATASET_NM}_rq5_user.csv', index=False)
    item_sparsity_df.to_csv(f'{SAVE_DIR}/{DOMAIN}_{DATASET_NM}_rq5_item.csv', index=False)
    print(f"\n✅ Sparsity evaluation completed! Results saved in {SAVE_DIR}")

if __name__ == "__main__":
    main()