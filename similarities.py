from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("all-MiniLM-L6-v2")

text_embeddings = np.load("text_embeddings.npy")
cast_embeddings = np.load("cast_embeddings.npy")
director_embeddings = np.load("director_embeddings.npy")

def compute_text_similarity(idx=None, query=None):
    
    # 🎯 CASE 1: reference movie
    if idx is not None:
        return cosine_similarity(
            text_embeddings[idx].reshape(1, -1),
            text_embeddings
        )[0]
    
    # 🎯 CASE 2: user query
    elif query is not None:
        
        query_vec = model.encode(
            [query],
            normalize_embeddings=True
        )
        
        return cosine_similarity(
            query_vec,
            text_embeddings
        )[0]
    
    else:
        raise ValueError("Either idx or query must be provided")
    

def compute_cast_similarity(idx=None, cast=None):
    
    if idx is not None:
        return cosine_similarity(
            cast_embeddings[idx].reshape(1, -1),
            cast_embeddings
        )[0]
    
    elif cast is not None:
        query_vec = model.encode([cast], normalize_embeddings=True)
        
        return cosine_similarity(
            query_vec,
            cast_embeddings
        )[0]
    

def compute_director_similarity(idx=None, director=None):
    
    if idx is not None:
        return cosine_similarity(
            director_embeddings[idx].reshape(1, -1),
            director_embeddings
        )[0]
    
    elif director is not None:
        query_vec = model.encode([director], normalize_embeddings=True)
        
        return cosine_similarity(
            query_vec,
            director_embeddings
        )[0]
    

def compute_genre_similarity(idx, df):
    def jaccard(a, b):
        a, b = set(a), set(b)
        
        if len(a | b) == 0:
            return 0
        
        return len(a & b) / len(a | b)
    
    query_genres = df.iloc[idx]["genres_x"]

    genre_scores = df["genres_x"].apply(
        lambda g: jaccard(query_genres, g)
    ).values
    return genre_scores

