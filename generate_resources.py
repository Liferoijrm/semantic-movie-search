import pandas as pd
import numpy as np
import faiss
from gensim.models import Word2Vec
from sentence_transformers import SentenceTransformer

def generate_all_resources():
    """
    Carrega a base de dados limpa, treina e persiste o modelo Word2Vec, 
    gera as matrizes de embeddings e constrói o índice HNSW do FAISS em disco.
    """
    df = pd.read_csv('data/processed_movies.csv')
    sentences = [str(text).split() for text in df['Clean_summary']]
    
    w2v_model = Word2Vec(sentences, vector_size=100, window=5, min_count=1, workers=4)
    w2v_model.save('data/word2vec.model')
    
    def get_average_vector(text):
        """
        Calcula o vetor médio composto a partir das palavras de uma sinopse 
        utilizando o modelo Word2Vec treinado.
        """
        words = str(text).split()
        vectors = [w2v_model.wv[word] for word in words if word in w2v_model.wv]
        if len(vectors) == 0:
            return np.zeros(100)
        return np.mean(vectors, axis=0)
        
    movie_vectors = np.array([get_average_vector(text) for text in df['Clean_summary']])
    np.save('data/word2vec_embeddings.npy', movie_vectors)
    
    sentence_embeddings = np.load('data/sentence_embeddings.npy').astype('float32')
    dimension = sentence_embeddings.shape[1]
    
    # FIX: Normaliza os embeddings dos filmes na norma L2 antes de adicioná-los ao índice.
    # Isso garante que a distância euclidiana mapeie perfeitamente para a similaridade de cosseno,
    # correspondendo à query que também é normalizada em tempo de busca.
    faiss.normalize_L2(sentence_embeddings)
    
    index = faiss.IndexHNSWFlat(dimension, 32)
    index.add(sentence_embeddings)
    faiss.write_index(index, 'data/hnsw_index.faiss')

if __name__ == "__main__":
    generate_all_resources()