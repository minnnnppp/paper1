# src/textrank.py
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize

def run_textrank(text, ratio=0.6):
    """
    Performs extractive summarization using the TextRank algorithm.
    This identifies and selects the most salient sentences from the review set.
    """
    if not text or len(str(text).split()) < 10:
        return text
    
    # 1. Sentence Segmentation
    # Splits the text by periods and filters out very short fragments.
    sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 5]
    if len(sentences) < 2:
        return text
    
    try:
        # 2. Calculate sentence similarity using TF-IDF and Cosine Similarity
        tfidf = TfidfVectorizer().fit_transform(sentences)
        sim_mat = cosine_similarity(tfidf)
        
        # 3. Build the sentence graph and execute PageRank (TextRank core)
        nx_graph = nx.from_numpy_array(sim_mat)
        scores = nx.pagerank(nx_graph, weight='weight')
        
        # 4. Sort sentences by importance scores and select top-N
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)
        num_to_pick = max(1, int(len(sentences) * ratio))
        
        # Select the top sentences to form the final extractive summary
        top_sentences = [s for score, s in ranked_sentences[:num_to_pick]]
        return ". ".join(top_sentences) + "."
    except Exception:
        # Fallback to original text if graph processing fails
        return text

def embed_with_glove(text, glove_dict, dim=300):
    """
    Converts the summarized text into an extractive representation vector (Eu, Ei).
    Uses mean pooling over pre-trained GloVe embeddings.
    """
    if not isinstance(text, str) or not text.strip():
        return np.zeros(dim)
    
    # Lowercase tokenization to match GloVe dictionary keys
    tokens = word_tokenize(text.lower())
    vectors = [glove_dict[token] for token in tokens if token in glove_dict]
    
    if vectors:
        # Calculate the centroid of the word vectors as the sentence representation
        return np.mean(vectors, axis=0)
    return np.zeros(dim)

def load_glove(path):
    """
    Loads pre-trained GloVe vectors into a dictionary for fast lookup.
    Used to initialize the extractive branch of SuReFAR.
    """
    glove_dict = {}
    with open(path, 'r', encoding="utf8") as f:
        for line in f:
            values = line.split()
            # The first element is the word, the rest are the vector dimensions
            glove_dict[values[0]] = np.asarray(values[1:], dtype='float32')
    return glove_dict