import os
import zipfile
import numpy as np
from bs4 import BeautifulSoup
from summa.summarizer import summarize
from tqdm import tqdm
from urllib.request import urlretrieve
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# pandas progress_apply 활성화
tqdm.pandas()

def clean_text_empty(text):
    """
    텍스트의 마침표(.)를 제거하고 양옆 공백을 지웠을 때 빈 문자열('')이 되는지 확인합니다.
    """
    if not isinstance(text, str):
        return True
    cleaned = text.replace('.', '').strip()
    return cleaned == ''

def filter_empty_reviews(df, col_name):
    """
    텍스트가 완전히 비어있는(길이가 0인) 데이터를 필터링하여 제거합니다.
    """
    initial_shape = df.shape[0]
    
    # 의미 있는 텍스트가 있는 행만 남김
    df_filtered = df[~df[col_name].apply(clean_text_empty)].copy()
    
    empty_count = initial_shape - df_filtered.shape[0]
    if empty_count > 0:
        print(f"  -> Removed {empty_count} empty reviews.")
    else:
        print("  -> No empty reviews found.")
        
    return df_filtered

def aggregate_reviews(df, id_col, review_col, entity_type):
    """
    User 또는 Item 아이디를 기준으로 리뷰들을 하나의 긴 텍스트 문서로 결합합니다.
    """
    print(f"\n--- Aggregating reviews for {entity_type.upper()} ---")
    
    # 1. 아이디별로 그룹화하여 리뷰 텍스트 결합 (공백으로 이어 붙이기)
    # 기존의 .sum() 방식보다 .apply(' '.join)이 텍스트 결합에 더 안전하고 빠릅니다.
    grouped = df.groupby(id_col)[review_col].apply(lambda x: ' '.join(x.astype(str))).reset_index()
    
    # 2. 컬럼명 변경
    review_merged_col = f'{entity_type}Reviews'
    grouped.rename(columns={review_col: review_merged_col}, inplace=True)
    
    # 3. HTML 태그 파싱 및 제거
    print(f"Removing HTML tags for {entity_type}...")
    grouped[review_merged_col] = grouped[review_merged_col].apply(
        lambda txt: BeautifulSoup(txt, "html.parser").get_text()
    )
    
    # 4. 결합된 텍스트 중 비어있는 리뷰 셋 제거
    grouped = filter_empty_reviews(grouped, review_merged_col)
    
    # 5. 최종 컬럼명 표준화 (Reviews_origin)
    grouped.rename(columns={review_merged_col: 'Reviews_origin'}, inplace=True)
    print(f"Aggregated {entity_type} shape: {grouped.shape}")
    
    return grouped

def safe_summarize(text, ratio=0.6, max_len=250000):
    """
    TextRank를 사용해 텍스트를 요약합니다. 
    메모리 에러를 방지하기 위해 텍스트 길이가 너무 길면 자릅니다.
    """
    # 노트북에 작성하신 메모리 에러(뻑남) 방지 로직
    if len(str(text).split(' ')) > max_len:
        text = text[:max_len] 
        
    try:
        summary = summarize(text, ratio=ratio)
        # 요약 결과가 비어있지 않으면 요약본 반환, 아니면 원문 반환
        return summary if summary and summary.strip() else text
    except Exception:
        return text  # 예외 발생 시 원문 그대로 유지

def apply_textrank_summarization(df, entity_type, ratio=0.6):
    """
    Reviews_origin 컬럼에 TextRank 요약을 일괄 적용합니다.
    """
    print(f"\n--- Applying TextRank for {entity_type.upper()} (Ratio: {ratio}) ---")
    
    summary_col = f'{entity_type}_summary'
    
    # tqdm을 사용해 진행률 표시 (progress_apply)
    df[summary_col] = df['Reviews_origin'].progress_apply(lambda x: safe_summarize(x, ratio=ratio))
    
    # 요약 결과 내의 줄바꿈 처리
    df[summary_col] = df[summary_col].apply(lambda x: str(x).replace('.\\n', '. ').replace('\n', ' '))
    
    return df

def generate_review_summaries(df, user_col='user_id', item_col='item_id', review_col='clean_review', ratio=0.6, save_path=None):
    """
    전처리된 리뷰 데이터셋을 받아 유저와 아이템별 요약 데이터프레임을 생성하는 메인 함수.
    """
    # 1. User Summarization
    user_grouped = aggregate_reviews(df, id_col=user_col, review_col=review_col, entity_type='user')
    user_summary_df = apply_textrank_summarization(user_grouped, entity_type='user', ratio=ratio)
    
    if save_path:
        user_summary_df.to_pickle(f"{save_path}/user_textrank_summary.pkl")
        print(f"User summary saved to {save_path}/user_textrank_summary.pkl")

    # 2. Item Summarization
    item_grouped = aggregate_reviews(df, id_col=item_col, review_col=review_col, entity_type='item')
    item_summary_df = apply_textrank_summarization(item_grouped, entity_type='item', ratio=ratio)
    
    if save_path:
        item_summary_df.to_pickle(f"{save_path}/item_textrank_summary.pkl")
        print(f"Item summary saved to {save_path}/item_textrank_summary.pkl")
        
    print("\n--- All Summarization Completed! ---")
    return user_summary_df, item_summary_df

if __name__ == "__main__":
    # ==========================================
    # 실행 예시 
    # ==========================================
    # 전처리 완료된 데이터프레임이 있다고 가정합니다. (preprocess.py의 결과)
    # prep_df = pd.read_csv('data/amazon_prep.csv')
    
    # user_summary, item_summary = generate_review_summaries(
    #     df=prep_df, 
    #     user_col='user_id', 
    #     item_col='item_id', 
    #     review_col='clean_review', 
    #     ratio=0.6,
    #     save_path='data/output'
    # )
    pass


def load_glove_dictionary(glove_dir="glove", dim=300):
    """
    HuggingFace를 통해 GloVe 6B 데이터를 다운로드하고 딕셔너리 형태로 로드합니다.
    """
    os.makedirs(glove_dir, exist_ok=True)
    glove_zip_path = os.path.join(glove_dir, "glove.6B.zip")
    glove_txt_path = os.path.join(glove_dir, f"glove.6B.{dim}d.txt")
    
    # 1. GloVe 다운로드 및 압축 해제 (최초 1회)
    if not os.path.exists(glove_txt_path):
        print("Downloading GloVe embeddings (this may take a while)...")
        # 스탠포드 공식 링크는 인증서 만료 이슈가 있으므로 안전한 허깅페이스 링크 사용
        urlretrieve("https://huggingface.co/stanfordnlp/glove/resolve/main/glove.6B.zip", filename=glove_zip_path)
        
        print("Extracting GloVe zip file...")
        with zipfile.ZipFile(glove_zip_path, 'r') as zf:
            zf.extractall(glove_dir)
            
    # 2. GloVe 벡터를 딕셔너리로 로드
    print(f"Loading GloVe {dim}d vectors into memory...")
    glove_dict = {}
    with open(glove_txt_path, encoding="utf8") as f:
        for line in f:
            word_vector = line.split()
            word = word_vector[0]
            vector = np.asarray(word_vector[1:], dtype='float32')
            glove_dict[word] = vector
            
    print(f"Loaded {len(glove_dict)} word vectors.")
    return glove_dict

def build_tokenizer(user_df, item_df, max_words=50000):
    """
    유저와 아이템의 요약 텍스트를 모두 모아 통합 Tokenizer를 학습시킵니다.
    """
    print(f"Fitting Tokenizer (Max words: {max_words})...")
    tokenizer = Tokenizer()
    
    # 두 요약본을 결합하여 전체 어휘 사전 구축
    all_texts = user_df['user_summary'].tolist() + item_df['item_summary'].tolist()
    tokenizer.fit_on_texts(all_texts)
    
    word_index = tokenizer.word_index
    total_words = min(max_words, len(word_index))
    print(f"Total words in vocabulary: {total_words}")
    
    return tokenizer, total_words, word_index

def generate_embedding_matrix(word_index, glove_dict, total_words, dim=300):
    """
    학습된 Tokenizer의 단어 사전을 바탕으로 GloVe Embedding Matrix를 생성합니다.
    """
    print("Generating GloVe Embedding Matrix...")
    # 0번 인덱스는 패딩(Padding)을 위해 0으로 채워진 상태를 유지합니다.
    embedding_matrix = np.zeros((total_words + 1, dim))
    
    match_count = 0
    for word, i in word_index.items():
        if i >= total_words:
            continue
        if word in glove_dict:
            embedding_matrix[i] = glove_dict[word]
            match_count += 1
            
    print(f"Matched {match_count} words out of {total_words} with GloVe vectors.")
    return embedding_matrix

def filter_sequences(sequences, max_words):
    """
    max_words 범위를 벗어나는 인덱스를 제거하여 에러를 방지합니다.
    """
    return [[idx for idx in seq if idx < max_words] for seq in sequences]

def process_and_pad_sequences(df, col_nm, tokenizer, max_seq_len, max_words):
    """
    텍스트를 시퀀스로 변환하고 지정된 길이에 맞게 패딩(Padding)을 적용합니다.
    """
    print(f"Tokenizing and Padding column: '{col_nm}' (Max Length: {max_seq_len})...")
    
    # 1. 텍스트 -> 정수 인덱스 시퀀스 변환
    text_sequence = tokenizer.texts_to_sequences(df[col_nm])
    
    # 2. max_words를 초과하는 희귀 단어 인덱스 필터링
    text_sequence = filter_sequences(text_sequence, max_words)
    
    # 3. 시퀀스 패딩 (post: 문장 뒤에 0을 채움)
    padded_result = pad_sequences(text_sequence, maxlen=max_seq_len, padding='post')
    
    # 4. 데이터프레임에 리스트 형태로 저장
    new_col_name = f"{col_nm}_padded_sequences"
    df[new_col_name] = list(padded_result)
    df[new_col_name] = df[new_col_name].apply(lambda x: list(x))
    
    return df

if __name__ == "__main__":
    # ==========================================
    # 실행 예시 (파이프라인)
    # ==========================================
    # 1. 데이터 로드 (textrank_summarizer.py 의 결과물)
    # user_summary_df = pd.read_pickle('data/user_textrank_summary.pkl')
    # item_summary_df = pd.read_pickle('data/item_textrank_summary.pkl')
    
    # 2. 파라미터 세팅 (논문 기준)
    # MAX_WORDS = 50000
    # EMBEDDING_DIM = 300
    # MEAN_SEQ_LEN = 44  # 도메인에 맞게 설정 (e.g., Movie=44, Book=118)
    
    # 3. GloVe 로드
    # glove_dict = load_glove_dictionary(dim=EMBEDDING_DIM)
    
    # 4. 통합 Tokenizer 학습
    # tokenizer, total_words, word_index = build_tokenizer(user_summary_df, item_summary_df, max_words=MAX_WORDS)
    
    # 5. GloVe 임베딩 매트릭스 생성
    # embedding_matrix = generate_embedding_matrix(word_index, glove_dict, total_words, dim=EMBEDDING_DIM)
    # np.save('data/glove_embedding_matrix.npy', embedding_matrix)
    # print("Saved GloVe embedding matrix.")
    
    # 6. 유저 및 아이템 시퀀스 패딩
    # user_summary_df = process_and_pad_sequences(user_summary_df, 'user_summary', tokenizer, MEAN_SEQ_LEN, MAX_WORDS)
    # item_summary_df = process_and_pad_sequences(item_summary_df, 'item_summary', tokenizer, MEAN_SEQ_LEN, MAX_WORDS)
    
    # 7. 최종 결과 저장
    # user_summary_df.to_pickle('data/user_padded_summary.pkl')
    # item_summary_df.to_pickle('data/item_padded_summary.pkl')
    pass