def tokenize(text):
    """Tokenizuje tekst na słowa."""
    return [word.lower() for word in text.split()]


def calculate_corpus_model(documents):
    """Oblicza model języka dla całego korpusu."""
    word_counts = {}
    total_words = 0

    # Zlicz wszystkie słowa w korpusie
    for doc in documents:
        tokens = tokenize(doc)
        total_words += len(tokens)
        for token in tokens:
            word_counts[token] = word_counts.get(token, 0) + 1

    # Oblicz prawdopodobieństwa w korpusie
    corpus_probs = {}
    if total_words > 0:
        for word, count in word_counts.items():
            corpus_probs[word] = count / total_words

    return corpus_probs


def calculate_document_model(document):
    """Oblicza model języka dla pojedynczego dokumentu."""
    tokens = tokenize(document)
    word_counts = {}
    total_words = len(tokens)

    # Zlicz słowa w dokumencie
    for token in tokens:
        word_counts[token] = word_counts.get(token, 0) + 1

    # Oblicz prawdopodobieństwa w dokumencie
    doc_probs = {}
    if total_words > 0:
        for word, count in word_counts.items():
            doc_probs[word] = count / total_words

    return doc_probs


def calculate_query_likelihood(doc_model, corpus_model, query_terms, lambda_param=0.5):
    """Oblicza prawdopodobieństwo wygenerowania zapytania przez model dokumentu."""
    log_prob = 0

    for term in query_terms:
        # Prawdopodobieństwo w dokumencie
        doc_prob = doc_model.get(term, 0)
        # Prawdopodobieństwo w korpusie
        corpus_prob = corpus_model.get(term, 0)
        # Wygładzanie Jelineka-Mercera
        smoothed_prob = lambda_param * doc_prob + (1 - lambda_param) * corpus_prob

        if smoothed_prob > 0:
            log_prob += smoothed_prob

    return log_prob


def rank_documents(documents, query):
    """Szereguje dokumenty według prawdopodobieństwa wygenerowania zapytania."""
    # Oblicz model korpusu
    corpus_model = calculate_corpus_model(documents)
    query_terms = tokenize(query)
    doc_scores = []

    # Oblicz score dla każdego dokumentu
    for idx, doc in enumerate(documents):
        doc_model = calculate_document_model(doc)
        score = calculate_query_likelihood(doc_model, corpus_model, query_terms)
        doc_scores.append((idx, score))

    # Sortuj malejąco według score, przy równości zachowaj kolejność dokumentów
    ranked_docs = sorted(doc_scores, key=lambda x: (-x[1], x[0]))
    return [idx for idx, _ in ranked_docs]


def main():
    # Wczytaj liczbę dokumentów
    n = int(input())

    # Wczytaj dokumenty
    documents = []
    for _ in range(n):
        documents.append(input())

    # Wczytaj zapytanie
    query = input()

    # Oblicz ranking
    result = rank_documents(documents, query)

    # Wyświetl wynik
    print(f"[{', '.join(map(str, result))}]")


if __name__ == "__main__":
    main()