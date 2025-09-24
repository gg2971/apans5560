import spacy
from typing import List

# Load spaCy model with vectors once
_nlp = spacy.load("en_core_web_md")

def embed_texts(texts: List[str]) -> List[List[float]]:
    vecs = []
    for doc in _nlp.pipe(texts, batch_size=64):
        vecs.append(doc.vector.tolist())
    return vecs

def embed_word(token: str) -> List[float]:
    return _nlp(token).vector.tolist()
