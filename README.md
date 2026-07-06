# Semantic Movie Search

Sistema de busca semântica e recomendação de filmes utilizando processamento de linguagem natural, embeddings vetoriais e modelos de linguagem locais. 
Projeto desenvolvido para a disciplina de Projeto e Análise de Algoritmos (PAA) da Universidade de Brasília (UnB) - Semestre 2026.1.

## Equipe
* Daniel Hollenbach - Matrícula: 241020859
* Davi Bragança - Matrícula: 242001473
* Davi Galvão - Matrícula: 241038577
* Lucas Mendes - Matrícula: 241020750
* Pedro Marcinoni - Matrícula: 241002396

## Objetivo
O sistema permite que usuários façam perguntas em linguagem natural sobre filmes, como:
* "Qual é o filme em que um homem fica preso em uma ilha e tenta sobreviver?"
* "Me recomende filmes de ficção científica com inteligência artificial."

Utilizando a base de dados CMU Movie Summary Corpus (aprox. 42 mil filmes), o sistema realiza buscas semânticas comparando métodos clássicos e modernos, e em seguida formata a resposta final utilizando um LLM local (Phi-4 Mini).

## Tecnologias e Bibliotecas
* **Linguagem**: Python 3.11
* **Interface**: Streamlit
* **Modelos de Linguagem (LLM)**: Phi-4 Mini (via Ollama)
* **Vetorização**: Sentence Transformers (all-MiniLM-L6-v2), Word2Vec (Gensim), TF-IDF (Scikit-learn)
* **Busca e Indexação**: FAISS (Meta), Scikit-learn (Similaridade de Cosseno)
* **Manipulação de Dados**: Pandas, NumPy

*Nota: Todas as bibliotecas utilizadas possuem licenças de código aberto (MIT, BSD, Apache 2.0 ou LGPL).*

## Análise de Complexidade (PAA)
O coração deste projeto é a análise rigorosa do custo algorítmico das operações:

1. **Pré-processamento:** Limpeza de texto e união de bases. 
    * **Complexidade:** $O(N)$, onde $N$ é o número de filmes.
2. **Busca Exata (Similaridade de Cosseno):** Comparação matricial de força bruta.
    * **Complexidade:** $O(N \times D)$, onde $D$ é o número de dimensões do vetor (ex: 384). Alto custo computacional em tempo de inferência.
3. **Busca Aproximada (HNSW via FAISS)**: Navegação baseada em grafos hierárquicos. 
    * **Complexidade (busca):** $O(\log N)$. Permite buscas instantâneas, sacrificando precisão marginal por ganho drástico de desempenho.
    * **Complexidade (construção):** $O(N \log N)$.

**Trade-offs (Tempo vs Qualidade)**: Modelos densos (Sentence Embeddings) custam mais tempo/memória para treinar/indexar do que médias simples (Word2Vec), mas a qualidade de recuperação (Hit@5) é substancialmente superior. Passar mais contexto para o LLM aumenta a qualidade da resposta, mas eleva o custo temporal de inferência linearmente.

## Estrutura do Projeto
```text
semantic-movie-search/
│
├── data/                                 # Diretório para os dados brutos e processados
├── preprocessing/
│   └── clean_data.py                     # Script de limpeza e fusão (O(N))
├── search/
│   ├── cosine_similarity.py              # Busca exata (TF-IDF/Dense)
│   ├── hnsw_search.py                    # Busca aproximada com FAISS
│   ├── sentence_embeddings.py            # Vetorização com MiniLM
│   ├── word2vec_average.py               # Vetorização com Gensim
│   └── utils.py                          # Funções auxiliares de busca
├── llm/
│   ├── phi4_mini.py                      # Integração com servidor Ollama local
│   └── query_formatter.py                # Limpeza sintática do prompt do usuário
├── evaluation/
│   ├── analise_benchmark.ipynb           # Notebook com gráficos de tempo vs qualidade
│   ├── benchmark.py                      # Script de execução de testes (Hit@k)
│   ├── metrics.py                        # Fórmulas de cálculo de precisão
│   └── queries.py                        # Dataset de perguntas com gabarito
├── scripts/
│   └── sanity_check.py                   # Verificador de ambiente e dependências
├── app.py                                # Interface web (Streamlit) principal
├── requirements.txt
├── PLANO_DE_PROJETO.md
└── README.md
```

## Como Executar

### 1. Preparação do Ambiente
Recomenda-se estritamente o uso do Python 3.11 para evitar quebras de compatibilidade em C++ com o gensim e o numpy.

```bash
# Criação do ambiente virtual
python3.11 -m venv venv

# Ativação (Linux/macOS)
source venv/bin/activate
# Ativação (Windows)
venv\Scripts\activate

# Instalação das dependências
pip install -r requirements.txt
```

### 2. Preparação dos Dados
Baixe os arquivos movie.metadata.tsv e plot_summaries.txt do CMU Movie Summary Corpus e coloque-os dentro da pasta data/. 
Em seguida, execute a limpeza:

```bash
python preprocessing/clean_data.py
``` 

### 3. Servidor LLM (Ollama)
Certifique-se de ter o Ollama instalado em sua máquina e o modelo Phi-4 Mini baixado:

```bash
ollama pull phi4-mini
``` 

Mantenha o serviço do ollama rodando em background

### 4. Interface de Usuário
Com os dados limpos, os vetores gerados e o Ollama rodando, inicie o aplicativo:

```bash
streamlit run app.py
```
Acesse http://localhost:8501 no seu navegador.