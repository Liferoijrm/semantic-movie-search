import pandas as pd
import time
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from search.utils import build_result

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
        Retorna uma tupla: (lista_de_dicts padronizados, tempo gasto).
        """
        start_time = time.time()
        query_vec = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vec, self.X_tfidf).flatten()
        top_indices = similarities.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            row = self.df.iloc[idx]
            score = similarities[idx]
            results.append(build_result(row, score))
            
        search_time = time.time() - start_time
        
        return results, search_time

if __name__ == "__main__":
    searcher = ClassicCosineSearch()
    results, duration = searcher.search("romantic comedy in New York")
    
    # Validação visual do novo schema
    import json
    print(f"Tempo de busca: {duration:.4f}s\n")
    print("Top 2 Resultados:")
    print(json.dumps(results[:2], indent=2, ensure_ascii=False))