import os
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from transformers import BartTokenizer, BartModel

def generate_bart_embeddings(df, text_col, model_name='facebook/bart-base', batch_size=8):
    """
    사전 학습된 BART 모델을 사용하여 텍스트의 임베딩(768차원)을 추출합니다.
    """
    print(f"Loading {model_name}...")
    tokenizer = BartTokenizer.from_pretrained(model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    model = BartModel.from_pretrained(model_name).to(device)
    model.eval() # 추론 모드로 전환

    embeddings = []
    text_list = df[text_col].tolist()

    print(f"Generating BART embeddings for {len(text_list)} texts...")
    
    # 배치(Batch) 단위 처리
    for i in tqdm(range(0, len(text_list), batch_size)):
        batch_reviews = text_list[i:i + batch_size]
        inputs = tokenizer(batch_reviews, return_tensors='pt', padding=True, truncation=True, max_length=1024)
        
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            results = outputs.last_hidden_state
            
            # 형태: (batch_size, sequence_length, hidden_size) -> (batch_size, hidden_size)
            batch_embeddings = results[:, -1, :].cpu().numpy()
            
            embeddings.extend(batch_embeddings)

    # 추출된 임베딩을 데이터프레임에 새로운 컬럼으로 추가
    new_col_name = 'bart_embedding'
    df[new_col_name] = embeddings
    
    print(f"Completed! Extracted embedding dimension: {embeddings[0].shape}")
    return df

def process_and_save_bart(input_path, output_path, entity_type, text_col, batch_size=8):
    """
    데이터 로드, BART 임베딩 적용, 저장까지 수행하는 파이프라인 함수
    """
    print(f"\n--- Starting BART Embedding for {entity_type.upper()} ---")
    df = pd.read_pickle(input_path)
    
    df_embedded = generate_bart_embeddings(df, text_col=text_col, batch_size=batch_size)
    
    # 불필요한 원본 텍스트 컬럼을 제거하고 id와 임베딩만 남기기 (option)
    # df_embedded = df_embedded[[f'{entity_type}ID', 'bart_embedding']]
    
    df_embedded.to_pickle(output_path)
    print(f"Saved BART embeddings to {output_path}")
    return df_embedded

if __name__ == "__main__":
    # ==========================================
    # 실행 예시
    # ==========================================
    # GPU 메모리에 따라 batch_size를 조절 (OOM 발생 시 1이나 2로 낮춤)
    # BATCH_SIZE = 8 
    
    # 1. User BART Embedding 생성
    # user_embedded = process_and_save_bart(
    #     input_path='data/user_textrank_summary.pkl', 
    #     output_path='data/user_bart_768.pkl', 
    #     entity_type='user', 
    #     text_col='userReviews', # 요약된 텍스트가 아닌 원본 결합 텍스트 컬럼명
    #     batch_size=BATCH_SIZE
    # )
    
    # 2. Item BART Embedding 생성
    # item_embedded = process_and_save_bart(
    #     input_path='data/item_textrank_summary.pkl', 
    #     output_path='data/item_bart_768.pkl', 
    #     entity_type='item', 
    #     text_col='itemReviews',
    #     batch_size=BATCH_SIZE
    # )
    pass