import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer

df = pd.read_csv('data/processed_movies.csv')

model = SentenceTransformer('all-MiniLM-L6-v2')

embeddings = model.encode(df['Clean_summary'].astype(str).tolist(), show_progress_bar=True)

np.save('data/sentence_embeddings.npy', embeddings)