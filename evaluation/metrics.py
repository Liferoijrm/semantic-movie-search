def hit_at_k(retrieved_ids, expected_ids, k):
    """1.0 se algum id esperado está entre os k primeiros recuperados, senão 0.0."""
    top = retrieved_ids[:k]
    return 1.0 if any(e in top for e in expected_ids) else 0.0


def recall_at_k(retrieved_ids, expected_ids, k):
    """Fração dos ids esperados que aparecem entre os k primeiros recuperados."""
    top = set(retrieved_ids[:k])
    return sum(1 for e in expected_ids if e in top) / len(expected_ids)


def reciprocal_rank(retrieved_ids, expected_ids):
    """1/posição do primeiro id esperado encontrado (0 se nenhum aparece)."""
    for position, mid in enumerate(retrieved_ids, start=1):
        if mid in expected_ids:
            return 1.0 / position
    return 0.0
