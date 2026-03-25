import pandas as pd
import re
import os
import numpy as np
from bs4 import BeautifulSoup
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from src.textrank import run_textrank, load_glove
from src.bart import bart_embeddings

def clean_and_normalize(text):
    """HTML removal, lowercasing, and preserving periods (.) for TextRank."""
    if not isinstance(text, str): return ""
    # Explicitly specify BeautifulSoup parser and handle text extraction
    text = BeautifulSoup(text, "html.parser").get_text()
    processed = re.sub(r'[^a-zA-Z0-9\s.]', '', text).lower()
    return processed

def has_actual_content(text):
    """Filters out reviews consisting only of dots or whitespace."""
    if not isinstance(text, str): return False
    return len(text.replace('.', '').strip()) > 0

def apply_user_core_filter(df, min_count=5):
    """Keep only users with at least min_count interactions."""
    print(f"\n[Step 3] Filtering users with >= {min_count} interactions...")
    start_rows = len(df)
    df = df[df['user'].map(df['user'].value_counts()) >= min_count]
    print(f" > Filtering Complete: {start_rows} -> {len(df)} rows")
    return df

def generate_review_sets(df):
    """Aggregates individual reviews into User (Su) and Item (Si) review sets."""
    user_reviews = df.groupby('user')['review'].apply(lambda x: " ".join(x)).reset_index()
    user_reviews.columns = ['user', 'user_review_set']
    item_reviews = df.groupby('item')['review'].apply(lambda x: " ".join(x)).reset_index()
    item_reviews.columns = ['item', 'item_review_set']
    return user_reviews, item_reviews

def run_data_pipeline(config):
    # 1. Loading
    print(f"\n[Step 1] Loading raw data: {config['data']['file_path']}")
    df = pd.read_json(config['data']['file_path'], compression='gzip', lines=True)
    df = df.rename(columns={'reviewerID': 'user', 'asin': 'item', 'reviewText': 'review', 'overall': 'rating'})
    
    # Calculate Dynamic Sequence Length
    seq_length_mean = int(df['review'].apply(lambda x: len(str(x).split())).mean())
    print(f"[INFO] Calculated reviewText seq_length_mean: {seq_length_mean}")

    # 2. Text Preprocessing
    print(f"\n[Step 2] Cleaning text & Filtering noise...")
    df['review'] = df['review'].apply(clean_and_normalize)
    df = df[df['review'].apply(has_actual_content)].copy()

    # 3. 5-Core Filtering
    min_core = config['model_params'].get('min_core', 1) 
    df = apply_user_core_filter(df, min_count=min_core)

    # 4. Label Encoding
    le_user, le_item = LabelEncoder(), LabelEncoder()
    df['user'] = le_user.fit_transform(df['user'])
    df['item'] = le_item.fit_transform(df['item'])

    # 5. Review Sets
    print(f"\n[Step 5] Creating aggregate review sets Su and Si...")
    user_set, item_set = generate_review_sets(df)

    # 6. Multi-Summarization
    print(f"\n[Step 6-A] Branch A: TextRank Summarization (Extractive)...")
    user_set['user_ext_summary'] = user_set['user_review_set'].apply(lambda x: run_textrank(x, ratio=config['textrank']['ratio']))
    item_set['item_ext_summary'] = item_set['item_review_set'].apply(lambda x: run_textrank(x, ratio=config['textrank']['ratio']))

    # Tokenization & Padding (Naming updated for better clarity)
    tokenizer = Tokenizer(lower=True)
    tokenizer.fit_on_texts(user_set['user_ext_summary'].tolist() + item_set['item_ext_summary'].tolist())
    
    # [Update] Use descriptive name 'user_ext_seq' instead of 'u_seq'
    user_set['user_ext_seq'] = list(pad_sequences(tokenizer.texts_to_sequences(user_set['user_ext_summary']), maxlen=seq_length_mean, padding='post'))
    item_set['item_ext_seq'] = list(pad_sequences(tokenizer.texts_to_sequences(item_set['item_ext_summary']), maxlen=seq_length_mean, padding='post'))

    print(f"\n[Step 6-B] Branch B: BART Encoding (Abstractive)...")
    # [Update] Use descriptive name 'user_abs_vec' instead of 'u_abs'
    user_set['user_abs_vec'] = list(bart_embeddings(user_set, 'user_review_set'))
    item_set['item_abs_vec'] = list(bart_embeddings(item_set, 'item_review_set'))

    # 7. Merging & Persistence
    print(f"\n[Step 7] Merging all descriptive features and saving...")
    # Apply updated naming convention during the merge
    df = df.merge(user_set[['user', 'user_ext_seq', 'user_abs_vec']], on='user', how='left')
    df = df.merge(item_set[['item', 'item_ext_seq', 'item_abs_vec']], on='item', how='left')
    
    save_path = os.path.join(config['data']['save_path'], config['data']['final_file'])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_pickle(save_path)
    
    print(f"\n[SUCCESS] SuReFAR Pipeline Completed. Total interactions: {len(df)}")
    return df, tokenizer, seq_length_mean