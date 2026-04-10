import pandas as pd
import re
import emoji
import nltk
from nltk.corpus import stopwords
import gzip
import json

# 최초 1회 실행 필요
# nltk.download('stopwords')

def parse(path):
    """
    gzip으로 압축된 JSONL 파일을 한 줄씩 읽어오는 제너레이터 함수입니다.
    """
    g = gzip.open(path, 'rb')
    for l in g:
        yield json.loads(l)

def load_amazon_data(file_paths):
    """
    Amazon 리뷰 데이터(.json.gz)를 로드하고 표준 컬럼명으로 변경합니다.
    """
    print("Loading Amazon dataset...")
    
    # file_paths가 문자열로 들어온 경우 처리
    path = file_paths['review'] if isinstance(file_paths, dict) else file_paths
    
    i = 0
    df_dict = {}
    for d in parse(path):
        df_dict[i] = d
        i += 1
        
    df = pd.DataFrame.from_dict(df_dict, orient='index')
    
    # 필요한 컬럼만 추출 및 표준 이름으로 변경
    df = df[['reviewerID', 'asin', 'reviewText', 'overall']]
    df.columns = ['user_id', 'item_id', 'review', 'rating']
    
    return df

def load_yelp_data(file_paths):
    """
    Yelp의 User, Review, Business 데이터를 각각 로드하고 병합하여 표준화합니다.
    """
    print("Loading Yelp Review dataset...")
    with open(file_paths['review'], 'r', encoding='utf-8') as f:
        df_review = pd.DataFrame(json.loads(line) for line in f)
        
    print("Loading Yelp User dataset...")
    with open(file_paths['user'], 'r', encoding='utf-8') as f:
        df_user = pd.DataFrame(json.loads(line) for line in f)
        
    print("Loading Yelp Business dataset...")
    with open(file_paths['business'], 'r', encoding='utf-8') as f:
        df_business = pd.DataFrame(json.loads(line) for line in f)
        
    print("Preprocessing and merging Yelp datasets...")
    
    # [메모리 최적화] 병합(Merge) 전 사용할 컬럼만 미리 필터링
    # 논문에서 특별히 추가 메타데이터를 쓰지 않는다면 아래 컬럼만 남깁니다.
    df_review = df_review[['user_id', 'business_id', 'text', 'stars']]
    df_user = df_user[['user_id']] 
    df_business = df_business[['business_id']]
    
    # 데이터 결합 (Inner Join: 유저와 비즈니스 정보가 모두 있는 리뷰만 남김)
    df_merged = pd.merge(df_review, df_user, on='user_id', how='inner')
    df_merged = pd.merge(df_merged, df_business, on='business_id', how='inner')
    
    # 컬럼명 표준화
    df_merged.columns = ['user_id', 'item_id', 'review', 'rating']
    return df_merged

def clean_text(text):
    """
    리뷰 텍스트를 정제합니다. (논문에 명시된 전처리 조건 반영)
    """
    if not isinstance(text, str):
        return ""
        
    text = text.lower() # 소문자 변환
    text = re.sub(r'<[^>]+>', ' ', text) # HTML 태그 제거
    text = emoji.replace_emoji(text, replace='') # 이모지 제거
    text = re.sub(r'[^a-z0-9\s\.\?\!]', '', text) # 특수문자 제거 (구두점 보존)
    
    # 불용어 제거
    stop_words = set(stopwords.words('english'))
    words = text.split()
    words = [w for w in words if w not in stop_words]
    text = ' '.join(words)
    
    text = re.sub(r'\s+', ' ', text).strip() # 다중 공백 제거
    return text

def filter_k_core(df, user_col='user_id', item_col='item_id', k=5):
    """
    K-core 필터링 (기본값: 5). 
    상호작용이 K번 미만인 유저와 아이템을 반복적으로 제거합니다.
    """
    print(f"Applying {k}-core filtering...")
    print(f"  -> Initial interactions: {len(df)}")
    
    while True:
        start_len = len(df)
        
        # 유저 필터링
        user_counts = df[user_col].value_counts()
        valid_users = user_counts[user_counts >= k].index
        df = df[df[user_col].isin(valid_users)]
        
        # 아이템 필터링
        item_counts = df[item_col].value_counts()
        valid_items = item_counts[item_counts >= k].index
        df = df[df[item_col].isin(valid_items)]
        
        if len(df) == start_len:
            break
            
    print(f"  -> Filtered interactions: {len(df)}")
    return df

def process_dataset(file_paths, domain, output_path=None):
    """
    데이터 로드, 텍스트 정제, 필터링을 모두 수행하는 메인 파이프라인.
    """
    # 1. 도메인별 데이터 로드
    if domain.lower() == 'amazon':
        df = load_amazon_data(file_paths)
    elif domain.lower() == 'yelp':
        df = load_yelp_data(file_paths)
    else:
        raise ValueError("Domain must be 'amazon' or 'yelp'")
        
    # 2. 결측치 제거
    df = df.dropna(subset=['review', 'rating'])
    
    # 3. 텍스트 전처리
    print("Cleaning review texts...")
    df['clean_review'] = df['review'].apply(clean_text)
    
    # 4. 길이가 0인 리뷰 제거
    df = df[df['clean_review'].str.len() > 0]
    
    # 5. 5-Core 필터링 적용
    df = filter_k_core(df, k=5)
    
    # 6. 전처리 완료된 데이터 저장 (옵션)
    if output_path:
        df.to_pickle(output_path, index=False)
        print(f"Preprocessed data saved to {output_path}")
        
    print("--- Preprocessing Pipeline Completed! ---\n")
    return df

if __name__ == "__main__":
    # 실행 예시: Yelp 데이터는 3개의 파일 경로를 딕셔너리로 묶어서 전달합니다.
    # yelp_paths = {
    #     'review': 'data/yelp_academic_dataset_review.json',
    #     'user': 'data/yelp_academic_dataset_user.json',
    #     'business': 'data/yelp_academic_dataset_business.json'
    # }
    # yelp_df = process_dataset(yelp_paths, domain='yelp', output_path='data/yelp_prep.pkl')
    
    # Amazon 데이터는 단일 경로(또는 'review' 키를 가진 딕셔너리)를 전달합니다.
    # amz_df = process_dataset('data/amazon_books.json.gz', domain='amazon', output_path='data/amazon_prep.pkl')
    pass