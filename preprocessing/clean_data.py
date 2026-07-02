import pandas as pd
import re

def clean_text(text):
    """
    Limpeza básica do texto para melhorar a geração dos vetores.
    """
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().strip()

def prepare_data():
    """
    Carrega os dados brutos de metadados e sinopses, realiza a unificação 
    dos arquivos, aplica a limpeza textual e persiste o resultado em um CSV.
    """
    meta_cols = [
        'Wikipedia_movie_ID', 'Freebase_movie_ID', 'Movie_name', 
        'Movie_release_date', 'Movie_box_office_revenue', 
        'Movie_runtime', 'Movie_languages', 'Movie_countries', 'Movie_genres'
    ]
    
    # Leitura dos arquivos brutos na pasta de dados
    df_meta = pd.read_csv('data/movie.metadata.tsv', sep='\t', names=meta_cols)
    
    plot_cols = ['Wikipedia_movie_ID', 'Plot_summary']
    df_plot = pd.read_csv('data/plot_summaries.txt', sep='\t', names=plot_cols)
    
    # Junção dos DataFrames usando o ID da Wikipedia como chave comum
    df_merged = pd.merge(
        df_plot, 
        df_meta[['Wikipedia_movie_ID', 'Movie_name', 'Movie_genres']], 
        on='Wikipedia_movie_ID', 
        how='inner'
    )
    
    # Aplicação da função de limpeza na coluna de sinopses
    df_merged['Clean_summary'] = df_merged['Plot_summary'].apply(clean_text)
    
    # Exportação e persistência dos dados processados
    output_path = 'data/processed_movies.csv'
    df_merged.to_csv(output_path, index=False)

if __name__ == "__main__":
    prepare_data()