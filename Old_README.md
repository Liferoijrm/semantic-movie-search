# Semantic Movie Search

Sistema de busca semântica e recomendação de filmes utilizando processamento de linguagem natural, embeddings vetoriais e modelos de linguagem locais.

## Objetivo

O projeto foi desenvolvido para a disciplina de Projeto e Análise de Algoritmos da Universidade de Brasília (UnB) e tem como objetivo permitir que usuários façam perguntas em linguagem natural sobre filmes, como:

* "Qual é o filme em que um homem fica preso em uma ilha e tenta sobreviver?"
* "Me recomende filmes sobre viagens no tempo."
* "Filmes parecidos com uma história de vingança e redenção."

O sistema realiza uma busca semântica sobre a base de dados **CMU Movie Summary Corpus**, recupera as sinopses mais relevantes e utiliza um modelo de linguagem local para gerar uma resposta formatada ao usuário.

---

## Tecnologias Utilizadas

### Linguagem

* Python 3.11

### Interface

* Streamlit

### Inteligência Artificial

* TinyLlama (via Ollama)
* Sentence Transformers
* Word2Vec (Gensim)

### Recuperação de Informação

* Similaridade do Cosseno
* Sentence Embeddings
* HNSW / Approximate Nearest Neighbors
* FAISS

### Manipulação e Processamento de Dados

* NumPy
* Pandas
* Scikit-learn

---

## Arquitetura do Sistema

```text
Usuário
   │
   ▼
Pergunta em Linguagem Natural
   │
   ▼
Pré-processamento
   │
   ▼
Geração de Vetores
 ├─ Word2Vec Average
 └─ Sentence Embeddings
   │
   ▼
Busca Semântica
 ├─ Similaridade do Cosseno
 └─ HNSW (FAISS)
   │
   ▼
Recuperação das Sinopses
   │
   ▼
TinyLlama (Ollama)
   │
   ▼
Resposta Formatada
```

---

## Dataset

### CMU Movie Summary Corpus

Base de dados pública contendo:

* Metadados de milhares de filmes
* Resumos (plot summaries)
* Informações de elenco
* Informações de gêneros

Fonte:

https://www.cs.cmu.edu/~ark/personas/

---

## Métodos Implementados

### 1. Word2Vec Average

Cada sinopse é representada pela média dos vetores das palavras presentes no texto.

Vantagens:

* Simples
* Baixo custo computacional

Limitações:

* Perde contexto semântico complexo

---

### 2. Sentence Embeddings

Utiliza modelos pré-treinados da biblioteca Sentence Transformers para gerar vetores representando frases e documentos completos.

Vantagens:

* Melhor compreensão semântica
* Melhor qualidade de recuperação

Limitações:

* Maior custo computacional

---

### 3. Similaridade do Cosseno

Mede o grau de semelhança entre dois vetores através do ângulo entre eles.

[
\cos(\theta) =
\frac{A \cdot B}{||A|| ||B||}
]

Utilizada como método principal para ranqueamento.

---

### 4. HNSW Search

Estrutura de busca aproximada baseada em grafos hierárquicos implementada através do FAISS.

Vantagens:

* Busca extremamente rápida
* Escala para grandes volumes de dados

Limitações:

* Pode perder alguns resultados exatos

---

## Instalação

### Clonar o Repositório

```bash
git clone https://github.com/seu-usuario/semantic-movie-search.git

cd semantic-movie-search
```

### Criar Ambiente Virtual

```bash
python -m venv venv
```

Linux/macOS:

```bash
source venv/bin/activate
```

Windows:

```bash
venv\Scripts\activate
```

### Instalar Dependências

```bash
pip install -r requirements.txt
```

---

## Instalação do Ollama

Instale o Ollama:

https://ollama.com

Baixe o modelo TinyLlama:

```bash
ollama pull tinyllama
```

Verifique se o servidor está ativo:

```bash
ollama serve
```

---

## Execução

```bash
streamlit run app.py
```

A aplicação ficará disponível em:

```text
http://localhost:8501
```

---

## Estrutura do Projeto

```text
semantic-movie-search/
│
├── data/
│   ├── movie.metadata.tsv
│   └── plot_summaries.txt
│
├── preprocessing/
│   ├── clean_data.py
│   └── embeddings.py
│
├── search/
│   ├── cosine_search.py
│   ├── hnsw_search.py
│   └── faiss_index.py
│
├── llm/
│   └── tinyllama.py
│
├── app.py
├── requirements.txt
└── README.md
```

---

## Comparação de Métodos

| Método                        | Qualidade | Velocidade |
| ----------------------------- | --------- | ---------- |
| Word2Vec Average + Cosseno    | Média     | Alta       |
| Sentence Embeddings + Cosseno | Alta      | Média      |
| Sentence Embeddings + HNSW    | Alta      | Muito Alta |

---

## Possíveis Perguntas

* Qual é o filme sobre um náufrago em uma ilha?
* Filmes parecidos com Matrix.
* Recomende filmes de ficção científica envolvendo inteligência artificial.
* Qual filme tem uma história sobre vingança familiar?

---

## Licenças

As bibliotecas utilizadas possuem licenças abertas compatíveis com uso acadêmico:

* Python — PSF License
* NumPy — BSD
* Pandas — BSD
* Scikit-learn — BSD
* Gensim — LGPL
* Sentence Transformers — Apache 2.0
* FAISS — MIT
* Streamlit — Apache 2.0
* Ollama — MIT
* TinyLlama — Apache 2.0

---

## Autores

Projeto desenvolvido para a disciplina Projeto e Análise de Algoritmos (PAA) da Universidade de Brasília (UnB), semestre 1/2026.
