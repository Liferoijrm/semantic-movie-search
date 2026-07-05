"""
Conjunto de queries de avaliação (gabarito) para o benchmark dos métodos de busca.

Cada query parafraseia o enredo de um ou mais filmes do CMU Movie Summary Corpus,
sem citar o título. Schema unificado (um dict por query):

    {
        "query":        <pergunta em linguagem natural>,
        "expected_ids": [<Wikipedia_movie_ID relevante>, ...],  # 1 ou vários
        "titles":       [<título correspondente>, ...],          # legibilidade/debug
        "category":     "iconic" | "obscure" | "multi" | "user",
    }

Categorias (usadas para analisar viés no relatório):
  - iconic  : blockbusters de enredo muito distintivo. Caso fácil / limite superior.
  - obscure : filmes cult/menos populares. Estressam o método; medem robustez real.
  - multi   : queries de recomendação com VÁRIOS filmes relevantes. São as únicas em
              que recall@k difere de hit@k (com 1 só relevante, recall@k == hit@k).
  - user    : queries adicionadas manualmente pelo grupo (reduzir viés de autoria).

Todos os IDs foram verificados contra `data/processed_movies.csv`. Para os obscuros,
as queries foram redigidas a partir do texto REAL da sinopse indexada, garantindo
que o gabarito seja fiel ao que o sistema realmente pode recuperar.
"""


def single(query, title, movie_id, category):
    """Query com um único filme relevante (recall@k == hit@k)."""
    return {
        "query": query,
        "expected_ids": [movie_id],
        "titles": [title],
        "category": category,
    }


def multi(query, members):
    """Query de recomendação com vários filmes relevantes. `members`: lista de (título, id)."""
    return {
        "query": query,
        "expected_ids": [mid for _title, mid in members],
        "titles": [t for t, _mid in members],
        "category": "multi",
    }


# ---------------------------------------------------------------------------
# ICONIC — blockbusters de enredo distintivo (caso fácil)
# ---------------------------------------------------------------------------
ICONIC = [
    single(
        "A computer hacker discovers that reality is a simulation controlled by machines",
        "The Matrix",
        30007,
        "iconic",
    ),
    single(
        "A man is stranded alone on a deserted island after a plane crash and befriends a volleyball",
        "Cast Away",
        44501,
        "iconic",
    ),
    single(
        "Scientists clone dinosaurs for a theme park that goes catastrophically wrong",
        "Jurassic Park",
        68485,
        "iconic",
    ),
    single(
        "A giant man-eating shark terrorizes a small beach town",
        "Jaws",
        74830,
        "iconic",
    ),
    single(
        "Toys secretly come to life whenever humans are not around",
        "Toy Story",
        53085,
        "iconic",
    ),
    single(
        "A young lion prince flees his kingdom after his father is murdered by his uncle",
        "The Lion King",
        88678,
        "iconic",
    ),
    single(
        "A teenager accidentally travels back in time in a modified sports car",
        "Back to the Future",
        42993,
        "iconic",
    ),
    single(
        "An aging mafia patriarch hands control of his crime empire to his reluctant son",
        "The Godfather",
        2466773,
        "iconic",
    ),
    single(
        "A slow-witted but kind-hearted man unwittingly witnesses major historical events",
        "Forrest Gump",
        41528,
        "iconic",
    ),
    single(
        "A clownfish father crosses the ocean to find his captured son",
        "Finding Nemo",
        239587,
        "iconic",
    ),
    single(
        "A lonely robot left to clean up an abandoned Earth falls in love with another robot",
        "WALL-E",
        8980330,
        "iconic",
    ),
    single(
        "A group of small hobbits set out to destroy a powerful magical ring",
        "The Lord of the Rings: The Fellowship of the Ring",
        173941,
        "iconic",
    ),
    single(
        "An ogre and a talking donkey journey to rescue a princess from a dragon",
        "Shrek",
        18717177,
        "iconic",
    ),
    single(
        "An underdog boxer from Philadelphia gets an unexpected shot at the world title",
        "Rocky",
        45772,
        "iconic",
    ),
    single(
        "An insomniac office worker starts a secret underground fight club",
        "Fight Club",
        1009041,
        "iconic",
    ),
    single(
        "A team of thieves enters people's dreams to plant an idea in someone's mind",
        "Inception",
        23270459,
        "iconic",
    ),
    single(
        "A wrongly convicted banker slowly plans his escape from prison over decades",
        "The Shawshank Redemption",
        30625,
        "iconic",
    ),
    single(
        "A young boy discovers he is a wizard and attends a school of magic",
        "Harry Potter and the Philosopher's Stone",
        667361,
        "iconic",
    ),
    single(
        "A cyborg assassin is sent back in time to kill a woman whose son will save humanity",
        "The Terminator",
        30327,
        "iconic",
    ),
    single(
        "Two star-crossed lovers meet aboard a doomed ocean liner that strikes an iceberg",
        "Titanic",
        52371,
        "iconic",
    ),
]

# ---------------------------------------------------------------------------
# OBSCURE — filmes cult/menos populares (queries redigidas a partir da sinopse real)
# ---------------------------------------------------------------------------
OBSCURE = [
    single(
        "Two engineers accidentally build a device in their garage that lets them travel a few hours into the past",
        "Primer",
        1134373,
        "obscure",
    ),
    single(
        "A lone astronaut nearing the end of a solitary three-year contract on a lunar mining base discovers a clone of himself",
        "Moon",
        20348262,
        "obscure",
    ),
    single(
        "A reclusive number theorist obsessed with finding patterns in nature spirals into paranoia over a mysterious number",
        "Pi",
        458573,
        "obscure",
    ),
    single(
        "Strangers wake up trapped in a maze of identical cube-shaped rooms rigged with deadly traps",
        "Cube",
        376485,
        "obscure",
    ),
    single(
        "In a future ruled by genetic selection, a man with inferior DNA assumes another person's identity to qualify for a space mission",
        "Gattaca",
        42886,
        "obscure",
    ),
    single(
        "At a farewell gathering, a departing professor claims to his colleagues that he is a fourteen-thousand-year-old man who never ages",
        "The Man from Earth",
        14238656,
        "obscure",
    ),
    single(
        "A troubled teenager is visited by a figure in a monstrous rabbit costume who predicts the end of the world",
        "Donnie Darko",
        57820,
        "obscure",
    ),
    single(
        "A man unable to form new memories hunts his wife's killer using tattoos and instant photographs",
        "Memento",
        2750041,
        "obscure",
    ),
    single(
        "Extraterrestrial refugees stranded above a South African city are confined to a slum, and a bureaucrat slowly mutates into one of them",
        "District 9",
        20777420,
        "obscure",
    ),
    single(
        "In a dystopian bureaucratic society, a low-level clerk escapes into daydreams while a paperwork error condemns an innocent man",
        "Brazil",
        74537,
        "obscure",
    ),
    single(
        "A sleazy television programmer discovers a broadcast of real torture and begins suffering violent hallucinations",
        "Videodrome",
        456587,
        "obscure",
    ),
    single(
        "A grieving man searches across different eras of time for a way to save his dying wife from cancer",
        "The Fountain",
        2450308,
        "obscure",
    ),
    single(
        "A rescue crew boards a starship that vanished years earlier and reappeared from another dimension",
        "Event Horizon",
        29320113,
        "obscure",
    ),
    single(
        "A man hiding in the woods steps into a strange machine and travels a short time into the past, where he meets himself",
        "Timecrimes",
        13490096,
        "obscure",
    ),
    single(
        "A guide leads two men through a forbidden zone toward a room said to grant a person's deepest wish",
        "Stalker",
        286266,
        "obscure",
    ),
]

# ---------------------------------------------------------------------------
# MULTI — recomendação com VÁRIOS relevantes (recall@k passa a diferir de hit@k).
#
# Usamos FRANQUIAS como gabarito: cada query descreve uma saga e os relevantes são
# TODOS os filmes daquela franquia presentes no dataset. Diferente de um tema aberto
# ("filmes sobre viagem no tempo"), a franquia tem um conjunto relevante objetivo e
# COMPLETO — evita o problema de gabarito incompleto, no qual dezenas de filmes
# válidos no corpus não são creditados e o recall colapsa artificialmente para zero.
# Os membros foram curados manualmente (o match por prefixo de título traz ruído,
# p.ex. "Rocky Mountain" não pertence à saga Rocky e fan films de Star Wars).
# As queries citam personagens/cenários da saga (Hogwarts, Jack Sparrow), mas nunca
# os títulos dos filmes — é assim que um usuário pediria uma recomendação.
# ---------------------------------------------------------------------------
MULTI = [
    multi(
        "A young wizard and his friends study at a school of witchcraft and wizardry "
        "while fighting the dark lord Voldemort over many years",
        [
            ("Harry Potter and the Philosopher's Stone", 667361),
            ("Harry Potter and the Chamber of Secrets", 667368),
            ("Harry Potter and the Prisoner of Azkaban", 667371),
            ("Harry Potter and the Goblet of Fire", 667372),
            ("Harry Potter and the Order of the Phoenix", 670407),
            ("Harry Potter and the Half-Blood Prince", 858575),
            ("Harry Potter and the Deathly Hallows: Part I", 9834441),
            ("Harry Potter and the Deathly Hallows – Part 2", 31941988),
        ],
    ),
    multi(
        "Rebels wielding laser swords and the Force battle a tyrannical galactic "
        "empire across outer space",
        [
            ("Star Wars Episode I: The Phantom Menace", 50793),
            ("Star Wars Episode II: Attack of the Clones", 50957),
            ("Star Wars Episode III: Revenge of the Sith", 55447),
            ("Star Wars Episode IV: A New Hope", 52549),
            ("Star Wars Episode V: The Empire Strikes Back", 53964),
            ("Star Wars Episode VI: Return of the Jedi", 50744),
        ],
    ),
    multi(
        "Hobbits, elves and men wage war across a fantasy world over a powerful "
        "ring that must be destroyed",
        [
            ("The Lord of the Rings: The Fellowship of the Ring", 173941),
            ("The Lord of the Rings: The Two Towers", 173944),
            ("The Lord of the Rings: The Return of the King", 174251),
            ("The Lord of the Rings", 396607),
        ],
    ),
    multi(
        "A witty pirate captain and his crew chase cursed treasure and battle "
        "supernatural enemies across the high seas",
        [
            ("Pirates of the Caribbean: The Curse of the Black Pearl", 321496),
            ("Pirates of the Caribbean: Dead Man's Chest", 999394),
            ("Pirates of the Caribbean: At World's End", 1689394),
            ("Pirates of the Caribbean: On Stranger Tides", 24314116),
        ],
    ),
]

# ---------------------------------------------------------------------------
# USER — adicione aqui as suas queries à mão (reduz viés de autoria do gabarito).
# Use single(...) para 1 filme ou multi(...) para recomendação. Ex.:
#   single("A description in your own words", "Movie Title", <movie_id>, "user"),
# Lembre de verificar o <movie_id> em data/processed_movies.csv antes.
# ---------------------------------------------------------------------------
# Preencha a string vazia com a SUA descrição do filme (com suas palavras, como um
# usuário pediria). Remova as linhas dos filmes que você não conhece/não viu.
USER = [
    single(
        "A bat-themed vigilante hero fights a clown-faced villain",
        "The Dark Knight",
        4276475,
        "user",
    ),
    single(
        "An extraterrestrial is stranded on Earth, where some kids find him and help him",
        "E.T. the Extra-Terrestrial",
        73441,
        "user",
    ),
    single(
        "A cop tries to save his wife and other people taken hostage by terrorists in a building",
        "Die Hard",
        97646,
        "user",
    ),
    single(
        "A kid is accidentally left behind when his parents leave for a family trip",
        "Home Alone",
        216072,
        "user",
    ),
    single(
        "Paranormal investigators battle ghosts in New York",
        "Ghostbusters",
        205013,
        "user",
    ),
    single(
        "A child psychologist treats a young boy who hides a disturbing secret",
        "The Sixth Sense",
        30582,
        "user",
    ),
    single(
        "An old man flies his house away with balloons, with a kid accidentally on board",
        "Up",
        11659396,
        "user",
    ),
    single(
        "A rat that loves cooking helps a man cook", "Ratatouille", 14941280, "user"
    ),
    single("Family with superpowers, animated", "The Incredibles", 565879, "user"),
    single("A panda learns kung fu", "Kung Fu Panda", 3248340, "user"),
    single("Man learns his life is a TV show", "The Truman Show", 232711, "user"),
    single("Man relives the same day multiple times", "Groundhog Day", 142327, "user"),
    single(
        "A prison guard on death row cares for a giant inmate with a mysterious healing gift",
        "The Green Mile",
        946164,
        "user",
    ),
    single(
        "After a wild bachelor party, three friends try to remember what happened and find their missing groom",
        "The Hangover",
        21918632,
        "user",
    ),
    single(
        "Kids play a board game that unleashes real-world dangers",
        "Jumanji",
        3700174,
        "user",
    ),
    single(
        "A kid wishes to be grown up and wakes up as an adult", "Big", 168597, "user"
    ),
]


ALL_QUERIES = ICONIC + OBSCURE + MULTI + USER
