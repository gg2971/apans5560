from typing import List, Union
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from app.embedder import embed_texts, embed_word

app = FastAPI(title="Embeddings API (Module 3 layout)")

class EmbedRequest(BaseModel):
    text: Union[str, List[str]]
    mode: str = "auto"

class EmbedResponse(BaseModel):
    vectors: List[List[float]]
    dim: int

@app.get("/")
def read_root():
    return {"message": "hello", "endpoints": ["/embed"]}

@app.post("/embed", response_model=EmbedResponse)
def embed(req: EmbedRequest):
    texts = [req.text] if isinstance(req.text, str) else req.text
    if not texts:
        raise HTTPException(status_code=400, detail="text cannot be empty")

    if req.mode == "word":
        vecs = [embed_word(t) for t in texts]
    elif req.mode == "text":
        vecs = embed_texts(texts)
    else:
        if len(texts) == 1 and len(texts[0].split()) == 1:
            vecs = [embed_word(texts[0])]
        else:
            vecs = embed_texts(texts)

    dim = len(vecs[0]) if vecs else 0
    return EmbedResponse(vectors=vecs, dim=dim)
