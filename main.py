def tokenize(text):
    """Tokenizuje tekst na słowa."""
    return [word.lower() for word in text.split()]


def calculate_models(documents):
    """Oblicza modele dla dokumentów i korpusu."""
    # Inicjalizacja liczników
    doc_models = []
    corpus_counts = {}
    corpus_total = 0

    # Oblicz model dla każdego dokumentu i zbierz statystyki korpusu
    for doc in documents:
        tokens = tokenize(doc)
        doc_counts = {}
        doc_len = len(tokens)

        for token in tokens:
            doc_counts[token] = doc_counts.get(token, 0) + 1
            corpus_counts[token] = corpus_counts.get(token, 0) + 1
            corpus_total += 1

        doc_models.append((doc_counts, doc_len))

    return doc_models, corpus_counts, corpus_total


def calculate_score(doc_model, doc_len, corpus_counts, corpus_total, query_terms, lambda_param=0.5):
    """Oblicza prawdopodobieństwo wygenerowania zapytania przez dokument."""
    score = 0

    for term in query_terms:
        # P(t|Md) - prawdopodobieństwo termu w dokumencie
        doc_prob = doc_model.get(term, 0) / doc_len if doc_len > 0 else 0

        # P(t|Mc) - prawdopodobieństwo termu w korpusie
        corpus_prob = corpus_counts.get(term, 0) / corpus_total if corpus_total > 0 else 0

        # Wygładzanie Jelineka-Mercera
        smoothed_prob = lambda_param * doc_prob + (1 - lambda_param) * corpus_prob

        # Dodaj do sumy logarytmicznej
        if smoothed_prob > 0:
            score += smoothed_prob * doc_model.get(term, 0)

    return score


def main():
    # Wczytaj dane wejściowe
    n = int(input().strip())
    documents = [input().strip() for _ in range(n)]
    query = input().strip()

    # Przygotuj modele
    doc_models, corpus_counts, corpus_total = calculate_models(documents)
    query_terms = tokenize(query)

    # Oblicz score dla każdego dokumentu
    scores = []
    for idx, (doc_model, doc_len) in enumerate(doc_models):
        score = calculate_score(doc_model, doc_len, corpus_counts, corpus_total, query_terms)
        scores.append((score, -idx))  # -idx dla zachowania stabilnego sortowania

    # Posortuj i przygotuj wynik
    # Sortujemy malejąco po score i rosnąco po indeksie (dlatego -idx)
    result = [-i for _, i in sorted(scores, reverse=True)]

    # Wyświetl wynik
    print(f"[{', '.join(map(str, result))}]")


if __name__ == "__main__":
    main()