import pandas as pd
import re
import json
import logging

# Configuração simples de log para monitorar a qualidade dos dados (falhas de parse)
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

def clean_text(text):
    """
    Limpeza básica do texto para melhorar a geração dos vetores.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().strip()

def parse_genres(raw_value, failure_counter):
    """
    Faz o parse da string de gêneros (formato Freebase/JSON) e 
    retorna uma string com os nomes separados por ponto e vírgula (;).
    """
    # Trata valores ausentes, NaN ou vazios
    if pd.isna(raw_value) or not isinstance(raw_value, str) or not raw_value.strip():
        return ""
    
    try:
        # O corpus utiliza aspas duplas, tornando-o um JSON válido
        genres_dict = json.loads(raw_value)
        # Pega apenas os valores do dicionário (nomes dos gêneros) e junta com ';'
        return ";".join(genres_dict.values())
    except json.JSONDecodeError:
        # Se falhar silenciosamente, incrementamos a lista contadora de erros mutável
        failure_counter[0] += 1
        return ""

def prepare_data():
    """
    Carrega os dados brutos de metadados e sinopses, realiza a unificação 
    dos arquivos, aplica a limpeza textual, parseia os gêneros e persiste o resultado.
    """
    meta_cols = [
        'Wikipedia_movie_ID', 'Freebase_movie_ID', 'Movie_name', 
        'Movie_release_date', 'Movie_box_office_revenue', 
        'Movie_runtime', 'Movie_languages', 'Movie_countries', 'Movie_genres'
    ]
    
    # Leitura dos arquivos brutos
    df_meta = pd.read_csv('data/movie.metadata.tsv', sep='\t', names=meta_cols)
    
    plot_cols = ['Wikipedia_movie_ID', 'Plot_summary']
    df_plot = pd.read_csv('data/plot_summaries.txt', sep='\t', names=plot_cols)
    
    # Junção dos DataFrames
    df_merged = pd.merge(
        df_plot, 
        df_meta[['Wikipedia_movie_ID', 'Movie_name', 'Movie_genres']], 
        on='Wikipedia_movie_ID', 
        how='inner'
    )
    
    # 1. Parsing da coluna Movie_genres
    # Usamos uma lista de um elemento como contador mutável para não precisar de variáveis globais
    parse_failures = [0]
    
    # Aplica o parser para criar a nova coluna, mantendo a original para rastreabilidade
    df_merged['Genres_clean'] = df_merged['Movie_genres'].apply(
        lambda x: parse_genres(x, parse_failures)
    )
    
    if parse_failures[0] > 0:
        logging.warning(f"Falha de parse em {parse_failures[0]} linhas na coluna Movie_genres.")
    else:
        logging.info("Parsing de gêneros concluído sem erros de JSON.")
    
    # 2. Aplicação da função de limpeza na coluna de sinopses (separada dos gêneros)
    df_merged['Clean_summary'] = df_merged['Plot_summary'].apply(clean_text)
    
    # Exportação
    output_path = 'data/processed_movies.csv'
    df_merged.to_csv(output_path, index=False)
    logging.info(f"Dados salvos com sucesso em {output_path}")

if __name__ == "__main__":
    prepare_data()