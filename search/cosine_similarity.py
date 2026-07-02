import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

class ClassicCosineSearch:
    def __init__(self, data_path='data/processed_movies.csv'):
        """
        Inicializa o mecanismo de busca clássico utilizando TF-IDF e matrizes esparsas.
        """
        self.df = pd.read_csv(data_path)
        self.vectorizer = TfidfVectorizer(stop_words='english')
        self.X_tfidf = self.vectorizer.fit_transform(self.df['Clean_summary'].astype(str))

    def search(self, query, top_k=5):
        """
        Realiza uma busca linear O(N) calculando a similaridade de cosseno sobre a matriz TF-IDF.
        Retorna um DataFrame com os top_k resultados e o tempo gasto.
        """
        start_time = time.time()
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.X_tfidf).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]
        search_time = time.time() - start_time
        
        return self.df.iloc[top_indices], search_time

if __name__ == "__main__":
    searcher = ClassicCosineSearch()
    results, duration = searcher.search("romantic comedy in New York")