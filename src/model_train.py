import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Import the model builder from model.py
from model import build_proposed_model

class ModelTrainer:
    def __init__(self, params, user_embedding_matrix=None, item_embedding_matrix=None):
        """
        Initialize the trainer with experiment parameters and embedding matrices.
        """
        self.params = params
        self.user_embedding_matrix = user_embedding_matrix
        self.item_embedding_matrix = item_embedding_matrix
        
        # Initialize variables for data and model
        self.full_df = None
        self.train_df = None
        self.test_df = None
        self.model = None

    def load_dataset(self, data_dir, dataset_nm):
        """
        Load dataset in the format of {dataset_nm}_data.pkl from the specified directory.
        """
        file_path = os.path.join(data_dir, f"{dataset_nm}_data.pkl")
        print(f"📂 Loading dataset from: {file_path}")
        self.full_df = pd.read_pickle(file_path)
        return self.full_df

    def split_data(self):
        """
        Split the full dataset into Train(70%), Val(10%), and Test(20%) sets.
        """
        print("\n✂️ Splitting dataset into Train(70%), Val(10%), Test(20%)...")
        # Split into Train(70%) and Test(20%)
        train_df, test_df = train_test_split(
            self.full_df, test_size=0.2, random_state=self.params.get('random_state', 42)
        )
        
        self.train_df = train_df
        self.test_df = test_df
        
        print(f"   -> Train samples: {len(self.train_df)}")
        print(f"   -> Test samples:  {len(self.test_df)}")

    def _get_model_inputs(self, df):
        """
        Internal method to extract Numpy arrays for model input from a DataFrame.
        """
        inputs = []
        if self.params.get('textrank_bool', True):
            inputs.append(np.stack(df['user_textrank'].values))
            inputs.append(np.stack(df['item_textrank'].values))
            
        if self.params.get('bart_bool', True):
            inputs.append(np.stack(df['user_bart'].values))
            inputs.append(np.stack(df['item_bart'].values))
            
        targets = np.array(df['rating'].values, dtype=np.float32)
        return inputs, targets

    def build_and_compile(self):
        """
        Construct the model architecture and compile it with the specified parameters.
        """
        self.model = build_proposed_model(
            params=self.params,
            textrank_user_embedding_matrix=self.user_embedding_matrix,
            textrank_item_embedding_matrix=self.item_embedding_matrix
        )

        self.model.compile(
            optimizer=Adam(learning_rate=self.params.get('learning_rate', 0.001)),
            loss=tf.keras.losses.MeanSquaredError(),
            metrics=[tf.keras.metrics.MeanAbsoluteError(), tf.keras.metrics.MeanSquaredError()]
        )
        
        if self.params.get('verbose', True):
            self.model.summary()

    def train(self):
        """
        Execute the model training process.
        """
        if self.train_df is None:
            self.split_data()
        if self.model is None:
            self.build_and_compile()

        # Prepare input features and labels
        train_x, train_y = self._get_model_inputs(self.train_df)

        # Configure EarlyStopping callback
        early_stopping = EarlyStopping(
            monitor='val_loss', 
            mode='min', 
            patience=self.params.get('patience', 5), 
            restore_best_weights=True
        )

        print(f"\n🚀 Start Training Model [{self.model.name}]...")
        history = self.model.fit(
            train_x, train_y,
            validation_split=0.125, 
            epochs=self.params.get('epochs', 100),
            batch_size=self.params.get('batch_size', 128),
            callbacks=[early_stopping],
            verbose=1 if self.params.get('verbose', True) else 2
        )
        return history

    def evaluate(self):
        """
        Evaluate the final performance on the Test set.
        """
        print("\n📊 Evaluating on Test Set...")
        test_x, test_y = self._get_model_inputs(self.test_df)
        
        predicted_ratings = self.model.predict(test_x)
        
        mae = mean_absolute_error(test_y, predicted_ratings)
        rmse = np.sqrt(mean_squared_error(test_y, predicted_ratings))
        
        print("\n✅ Final Evaluation Results:")
        print(f"   - MAE : {mae:.4f}")
        print(f"   - RMSE: {rmse:.4f}")
        
        return {'MAE': mae, 'RMSE': rmse}

    def run_pipeline(self):
        """
        Run the complete pipeline: Split -> Build -> Train -> Evaluate.
        """
        self.split_data()
        self.build_and_compile()
        history = self.train()
        metrics = self.evaluate()
        return metrics, history