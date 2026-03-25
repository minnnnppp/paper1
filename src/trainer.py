# src/trainer.py
import pandas as pd
import numpy as np
import tensorflow as tf
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import EarlyStopping

def prepare_surefar_inputs(df):
    """
    Prepares the multi-summary inputs for the SuReFAR model:
    [User Extractive, Item Extractive, User Abstractive, Item Abstractive]
    """
    # [Update] Updated to use the newly defined column names (e.g., user_ext_seq).
    u_ext = np.stack(df['user_ext_seq'].values)
    i_ext = np.stack(df['item_ext_seq'].values)
    u_abs = np.stack(df['user_abs_vec'].values)
    i_abs = np.stack(df['item_abs_vec'].values)
    
    # Target ratings
    y = df['rating'].values.astype('float32')
    
    return [u_ext, i_ext, u_abs, i_abs], y

# [Update] Added 'df' argument to match the call in main.py (model, df, config).
def run_training_pipeline(model, df, config):
    """
    Executes the training and evaluation workflow.
    """
    # --- 1. Load Fused Data ---
    # Since df is passed directly from main.py, no manual file loading is required.
    print(f"\n[TRAIN] Data received. Total records: {len(df)}")

    # --- 2. 7:1:2 Data Splitting ---
    # 
    # First split: 80% (Train + Val), 20% (Test)
    train_val_df, test_df = train_test_split(df, test_size=0.2, random_state=42)
    # Second split: 12.5% of 80% is 10% of total (Val)
    train_df, val_df = train_test_split(train_val_df, test_size=0.125, random_state=42)
    
    print(f"[INFO] Dataset Statistics - Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # --- 3. Prepare Model Inputs ---
    X_train, y_train = prepare_surefar_inputs(train_df)
    X_val, y_val = prepare_surefar_inputs(val_df)
    X_test, y_test = prepare_surefar_inputs(test_df)

    # --- 4. Model Compilation ---
    lr = config['model_params'].get('learning_rate', 0.0003)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss='mse',
        metrics=['mae', tf.keras.metrics.RootMeanSquaredError(name='rmse')]
    )

    # --- 5. Training with EarlyStopping ---
    patience = config['model_params'].get('patience', 5)
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=patience,
        restore_best_weights=True,
        verbose=1
    )

    batch_size = config['model_params'].get('batch_size', 128)
    epochs = config['model_params'].get('epochs', 100)

    print("\n--- Starting SuReFAR Model Training ---")
    history = model.fit(
        x=X_train,
        y=y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early_stopping],
        verbose=1
    )

    # --- 6. Final Performance Evaluation ---
    print("\n--- Final Test Performance (RQ1 Verification) ---")
    y_pred = model.predict(X_test).flatten()
    
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    print(f"SuReFAR Test Results -> MAE: {mae:.4f}, RMSE: {rmse:.4f}")

    # --- 7. Save Final Model Weights ---
    # Check if the save directory exists before saving the model.
    os.makedirs(config['data']['save_path'], exist_ok=True)
    save_path = os.path.join(config['data']['save_path'], 'surefar_model_best.h5')
    model.save(save_path)
    print(f"[SUCCESS] Model saved at: {save_path}")
    
    return history