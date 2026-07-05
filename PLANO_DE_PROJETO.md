# Guia Completo — Semantic Movie Search

> Documento de referência gerado a partir da estrutura atual do projeto + roteiro do professor.

---

## 1. Ambiente (venv) e execução do que já existe

### 1.1 Setup básico

```bash
# na raiz do projeto
python3.11 -m venv venv // usar PYTHON 3.11!!!

# Linux/macOS
source venv/bin/activate
# Windows
venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt
```

Antes de rodar qualquer coisa, falta uma peça que **não existe na estrutura atual**: a pasta `data/` com os arquivos brutos do dataset (`movie.metadata.tsv` e `plot_summaries.txt`). Baixem do CMU Movie Summary Corpus e coloquem em `data/` na raiz — sem isso nada do pipeline roda.

### 1.2 Ordem de execução do pipeline atual

A ordem importa e não é óbvia pelos nomes dos arquivos:

1. `preprocessing/clean_data.py` → gera `data/processed_movies.csv` (precisa dos dois arquivos brutos)
2. `generate_initial_embeddings.py` (no arquivo do repo está salvo como `generate_inital_embeddings.py`, veja seção 2.2) → gera `data/sentence_embeddings.npy` (precisa do CSV do passo 1)
3. `generate_resources.py` → treina Word2Vec, gera `data/word2vec_embeddings.npy` e constrói o índice `data/hnsw_index.faiss` **a partir do `.npy` gerado no passo 2** — ou seja, mesmo o nome sugerir "recursos gerais", ele depende do passo 2 ter rodado antes.
4. Só depois disso os arquivos em `search/` conseguem instanciar suas classes (todos leem algum artefato gerado nos passos 1–3).
5. instale o ollama e de ollama pull phi4-mini
6. rode streamlit run app.py

Isso merece ficar documentado no README (a versão atual, gerada por IA, não menciona essa ordem de dependência).

---

## 2. Estado atual da implementação — avaliação detalhada

### 2.1 O que está pronto e funcional (possivelmente desatualizado)

| Arquivo | O que faz | Nível de maturidade |
|---|---|---|
| `preprocessing/clean_data.py` | Lê metadados + sinopses, faz merge por `Wikipedia_movie_ID`, limpa texto (remove pontuação, lowercase), salva `processed_movies.csv` | Funcional, básico. Não normaliza a coluna `Movie_genres` (no dataset real ela vem como string tipo dicionário Freebase→gênero, não como texto legível) e não colapsa espaços múltiplos. |
| `search/cosine_similarity.py` (ver 2.2 sobre o nome do arquivo) | TF-IDF + similaridade de cosseno, busca linear O(N) | Funcional. É o baseline "não-semântico" (palavra-chave), distinto dos outros três métodos — faz sentido como ponto de comparação. |
| `search/word2vec_average.py` | Carrega Word2Vec treinado, faz média dos vetores das palavras da query, cosseno linear O(N) | Funcional. Fallback de vetor zero quando nenhuma palavra é conhecida está OK, mas o `100` do `np.zeros(100)` está hardcoded (deveria vir de uma constante compartilhada com `generate_resources.py`). |
| `search/sentence_embeddings.py` | SentenceTransformer (`all-MiniLM-L6-v2`) + cosseno exato O(N) | Funcional. É o único que já retorna um formato de saída "amigável" (lista de dicts com title/genres/score/summary) — os outros três retornam fatias de DataFrame. Isso vai gerar dor de cabeça na hora de padronizar (ver Fase 0). |
| `search/hnsw_search.py` | Índice FAISS `IndexHNSWFlat` pré-carregado do disco, busca sublinear | Funcional, mas precisa de atenção: `IndexHNSWFlat` por padrão usa distância L2, não similaridade de cosseno. Isso só dá resultado equivalente ao cosseno se os embeddings estiverem normalizados (o `all-MiniLM-L6-v2` normaliza por padrão na pipeline do sentence-transformers, então provavelmente está ok — mas isso precisa ser **verificado e documentado explicitamente** no relatório, não deixado implícito). |
| `generate_resources.py` | Treina Word2Vec, gera embeddings médios, constrói e salva índice HNSW | Funcional, mas os hiperparâmetros do HNSW (`M=32`) e do Word2Vec (`window=5`, `min_count=1`) estão fixos sem discussão/justificativa — o enunciado pede para documentar complexidade e decisões de projeto, então isso vai precisar de uma nota explicando o porquê dessas escolhas. |
| `requirements.txt` | Lista as libs | Existe, mas **sem nenhuma versão fixada** — dado o problema da seção 1.3, isso é um risco real de quebrar em máquinas diferentes do grupo. |

### 2.2 O que está completamente ausente

Comparando com o pipeline exigido pelo professor (6 passos) e os "aspectos que devem ser documentados":

- **Pasta `data/`** com o dataset bruto (cada desenvolvedor baixa o seu).
- **Integração com LLM local** (Ollama/TinyLlama) — não existe nenhum arquivo `llm/`, nenhuma chamada HTTP/SDK para o Ollama. Isso é o passo 5–6 do pipeline exigido e ainda não foi começado.
- **`app.py` / interface Streamlit** — o README descreve `streamlit run app.py`, mas esse arquivo não existe. É o ponto de entrada do usuário (passo 1 e 6 do pipeline) e está zerado.
- **Avaliação/benchmark comparativo** — o enunciado exige comparar os 4 métodos em qualidade, tempo, complexidade e consumo de recursos. Não existe nenhum script, conjunto de queries de teste ou métrica de qualidade implementada.
- **Relatório/documentação real** — o único texto existente é o README gerado por IA, que o próprio grupo já sinalizou que pode não bater com a realidade. Nenhuma das seções exigidas (arquitetura, pré-processamento, treinamento/indexação, inferência, avaliação) foi escrita de fato.

### 2.3 Estimativa de completude

Com as Fases 0 e 1 concluídas, o **núcleo algorítmico de busca está em torno de 100% pronto**. Olhando o projeto como um todo, a estimativa de completude está em **~50%**: a camada de dados + recuperação de informação está bem encaminhada, mas LLM, interface Streamlit, avaliação/benchmark comparativo e o relatório em si continuam em 0% — e juntos ainda representam mais da metade do que o enunciado exige.

---

## 3. Roteiro completo dos próximos passos

### Fase 0 — Fundação — ✅ CONCLUÍDA

### Fase 1 — Validação dos métodos já implementados (script de sanidade) ✅ CONCLUÍDA

### Fase 2 — Avaliação e comparação (requisito obrigatório do enunciado)

Micropassos:
- Montar um conjunto de ~15–20 queries de teste em linguagem natural com o(s) filme(s) esperado(s) como gabarito (pode ser manual, escolhendo filmes conhecidos que estão no dataset).
- Definir métrica de qualidade simples: hit@k / recall@k (o filme esperado apareceu entre os top-k retornados?).
- Criar `evaluation/benchmark.py` que roda os 4 métodos sobre o mesmo conjunto de queries e mede:
  - tempo médio de busca por query;
  - tempo de indexação/treinamento (uma vez só, por método);
  - uso de memória (opcional: `tracemalloc` ou `psutil`);
  - qualidade (hit@k).
- Gerar uma tabela final comparando os 4 métodos nesses quatro eixos (pode virar direto uma seção do relatório).
- Opcional: gráfico de barras (tempo) e gráfico de qualidade vs. velocidade (trade-off).

### Fase 3 — Integração com LLM local (Ollama)

Micropassos:
- Instalar Ollama, baixar o modelo (`ollama pull tinyllama`), confirmar `ollama serve` rodando.
- Criar `llm/tinyllama.py` com uma função tipo `gerar_resposta(query, filmes_recuperados) -> str`.
- Definir o prompt: instruir o modelo a responder **somente com base nas sinopses fornecidas** (para reduzir alucinação, já que é um modelo pequeno), formatando a saída como identificação/recomendação conforme o tipo de pergunta.
- Chamar a API local do Ollama (via lib `ollama` para Python, ou `requests` no endpoint `http://localhost:11434/api/generate`).
- Testar com resultados reais vindos da Fase 0/1 e ajustar o prompt conforme os erros observados (modelos pequenos tendem a inventar detalhes fora do contexto dado — vale reforçar no prompt "responda apenas com base nas sinopses abaixo").

### Fase 4 — Aplicação Streamlit

Micropassos:
- Criar `app.py`: campo de texto para a pergunta do usuário, seletor do método de busca (útil para fins de demonstração da comparação pedida no enunciado), botão de buscar.
- Exibir os resultados brutos recuperados (título, gênero, trecho da sinopse, score, tempo).
- Exibir a resposta formatada pelo LLM (Fase 3).
- Usar `st.cache_resource` para os modelos/índices pesados (evitar recarregar a cada interação).
- Tratar estados de erro (dataset/índice ausente, Ollama fora do ar).

### Fase 5 — Documentação/relatório final

Micropassos — cobrir cada item exigido no enunciado do professor:
- Tecnologias (linguagens, bibliotecas + licenças, programas usados).
- Arquitetura do sistema ponta a ponta (pode reaproveitar o diagrama do README, mas atualizado pra realidade).
- Pré-processamento (limpeza, geração de embeddings, estruturas de dados, complexidade de cada etapa).
- Treinamento/indexação (como os embeddings são gerados, como o HNSW é construído, complexidade).
- Inferência (como a busca ocorre, como o contexto vai pro LLM, como a resposta final é produzida, complexidade).
- Avaliação (recursos computacionais disponíveis, o que foi viável rodar, tempos medidos na Fase 2, como tempo de treino/inferência afeta qualidade).
- Reescrever o README para bater com a implementação real (o atual foi gerado por IA e pode conter descrições que não existem no código, como a coluna de gêneros já "limpa").

### Fase 6 — Polimento e entrega

- Revisão de código (remover dead code, checar os bugs da seção 2.2 foram todos corrigidos).
- Teste final end-to-end com todo mundo rodando em ambiente limpo (Python 3.11, `requirements.txt` fixado).
- Preparar demo/checklist final contra o enunciado item a item.

---

## 4. Divisão de tarefas — 4 dias, 5 pessoas

A Fase 0 (que era o Dia 1 do Davi Galvão na versão anterior desta tabela) já foi adiantada — dataset, nomes de arquivo, `requirements.txt`, `__init__.py`, parsing de gêneros e padronização de retorno estão feitos. Isso libera espaço no Dia 1 pra todo mundo já começar na frente principal de cada um, e sobra folga extra pro Dia 4 virar buffer/integração de verdade em vez de correria de última hora. Critério de distribuição continua o mesmo de antes: quem já contribuiu mais fica com carga mais leve / papel de apoio agora; quem ainda não tinha contribuído assume as frentes 0→1 maiores.

| Pessoa | Frente principal | Dia 1 | Dia 2 | Dia 3 | Dia 4 |
|---|---|---|---|---|---|
| **Davi Galvão** |  | Validação técnica + apoio geral | Livre pra destravar quem precisar (revisão de código, dúvidas pontuais) — ou começar a dar suporte técnico ao Daniel | Apoio de integração conforme necessário | Revisão final de código + testes de integração em grupo |
| **Davi Bragança** |  | Dataset + documentação de pré-processamento | Escrever a seção "Pré-processamento" do relatório (Fase 5) já com base no pipeline real e nas decisões tomadas | Escrever a parte de "Dataset" do relatório (como foi obtido, tamanho, campos usados) | Apoiar Lucas na consolidação do relatório final |
| **Daniel + Davi Galvão** | Integração LLM (Ollama) | Setup do Ollama + TinyLlama local, função básica de chamada à API | Prompt engineering: montar contexto a partir dos resultados padronizados (`build_result`) e testar com queries reais | Tratamento de erro/formatação da resposta do LLM | Escrever seção "Inferência" do relatório |
| **Pedro Marcinoni** | Interface Streamlit (`app.py`) | Esqueleto da UI (campo de busca, seletor de método) | Ligar `app.py` aos 4 métodos de busca — schema de retorno já está pronto (`search/utils.py`), então não depende mais de ninguém pra começar | Ligar `app.py` à resposta do LLM (depende do módulo do Daniel) | Cache (`st.cache_resource`), estados de erro, polimento visual |
| **Lucas Mendes** | Avaliação/benchmark + montagem final do relatório | Montar conjunto de 15–20 queries de teste com gabarito — pode reaproveitar/expandir as queries do script de sanidade do Galvão | Escrever `evaluation/benchmark.py` (tempo, hit@k) para os 4 métodos | Rodar benchmark completo, gerar tabela/gráfico comparativo | Consolidar todas as seções do relatório num documento único + revisar README |

**Dependências-chave para não travar o grupo:**
- O módulo do Daniel e do Davi Galvão (LLM) precisa estar minimamente funcional até o fim do Dia 2 para Pedro Marcinoni conseguir integrar no Dia 3.
- Com a folga ganha da Fase 0 já estar pronta, o Dia 4 pode ser dedicado quase inteiramente a testar app + busca + LLM + benchmark rodando juntos pela primeira vez, em vez de ainda estar terminando peças soltas.