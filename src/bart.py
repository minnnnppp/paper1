# src/bart.py
import torch
import numpy as np
from transformers import BartTokenizer, BartModel
from tqdm import tqdm

def bart_embeddings(df, text_col, batch_size=1, max_length=512):
    """
    Extracts abstractive representations using the BART-base encoder.
    """
    model_name = 'facebook/bart-base'
    tokenizer = BartTokenizer.from_pretrained(model_name)
    
    # Device setup: automatically detects CPU or GPU inside the function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BartModel.from_pretrained(model_name).to(device)
    model.eval()

    embeddings = []
    text_list = df[text_col].tolist()
    
    for i in tqdm(range(0, len(text_list), batch_size), desc=f"BART ({text_col})"):
        batch_reviews = text_list[i:i + batch_size]
        inputs = tokenizer(batch_reviews, return_tensors='pt', max_length=max_length, 
                          truncation=True, padding='max_length').to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Use the last hidden state of the encoder
            result = outputs.last_hidden_state[:, -1, :][0].cpu().numpy()
            embeddings.append(result)

    return np.vstack(embeddings)