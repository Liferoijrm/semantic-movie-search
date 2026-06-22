import pandas as pd
import re

def clean_text(text):
    """Limpeza básica do texto para melhorar a geração dos vetores."""
    if not isinstance(text, str):
        return ""
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().strip()

def prepare_data():
    print("Carregando os dados do CMU Movie Summary Corpus...")
    
    #definindo os nomes das colunas para os dados
    meta_cols = ['Wikipedia_movie_ID', 'Freebase_movie_ID', 'Movie_name', 
                 'Movie_release_date', 'Movie_box_office_revenue', 
                 'Movie_runtime', 'Movie_languages', 'Movie_countries', 'Movie_genres']
    
    #Lendo os arquivos da pasta data/
    df_meta = pd.read_csv('data/movie.metadata.tsv', sep='\t', names=meta_cols)
    
    plot_cols = ['Wikipedia_movie_ID', 'Plot_summary']
    df_plot = pd.read_csv('data/plot_summaries.txt', sep='\t', names=plot_cols)
    
    print("Fazendo o merge dos arquivos...")
    df_merged = pd.merge(df_plot, df_meta[['Wikipedia_movie_ID', 'Movie_name', 'Movie_genres']], 
                         on='Wikipedia_movie_ID', how='inner')
    
    print("Limpando os textos (isso pode levar alguns segundos)...")
    df_merged['Clean_summary'] = df_merged['Plot_summary'].apply(clean_text)
    
    #salvando o arquivo
    output_path = 'data/processed_movies.csv'
    df_merged.to_csv(output_path, index=False)
    
    print(f"Pronto! Dados unificados salvos em {output_path}")
    print(f"Total de filmes processados: {len(df_merged)}")

if __name__ == "__main__":
    prepare_data()