import pandas as pd

def build_result(row, score):
    """
    Constrói o dicionário padronizado de resultado de busca a partir de uma linha do DataFrame.
    """
    # Trata a coluna Genres_clean (que está como string separada por ';') para virar lista
    genres_raw = row.get('Genres_clean', '')
    if pd.isna(genres_raw) or not isinstance(genres_raw, str) or not genres_raw.strip():
        genres_list = []
    else:
        genres_list = genres_raw.split(';')

    return {
        "movie_id": row['Wikipedia_movie_ID'],
        "title": row['Movie_name'],
        "genres": genres_list,
        "summary": row['Plot_summary'], # Mantém o texto original para leitura legível
        "score": float(score)           # Garante que o score seja serializável (float em vez de numpy.float)
    }