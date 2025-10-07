from typing import List, Union
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
import torch
from torchvision import transforms
from PIL import Image

CIFAR10_CLASSES = [
    "airplane","automobile","bird","cat","deer",
    "dog","frog","horse","ship","truck"
]

# OPTIONAL IMPORTS FOR EMBEDDINGS
try:
    from app.embedder import embed_texts, embed_word
    EMBEDDER_OK = True
except ModuleNotFoundError:
    EMBEDDER_OK = False
    print("spaCy or embedder not found — /embed endpoint disabled.")

# FASTAPI INITIALIZATION
app = FastAPI(title="Embeddings + CNN API")

# TEXT EMBEDDING ENDPOINTS
if EMBEDDER_OK:

    class EmbedRequest(BaseModel):
        text: Union[str, List[str]]
        mode: str = "auto"

    class EmbedResponse(BaseModel):
        vectors: List[List[float]]
        dim: int

    @app.get("/")
    def read_root():
        return {"message": "hello", "endpoints": ["/embed", "/classify"]}

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

else:
    @app.get("/")
    def read_root():
        return {"message": "hello", "endpoints": ["/classify"]}

# CNN CLASSIFIER ENDPOINT
from app.assignment2.cnn_model import CNN64

MODEL_PATH = "app/assignment2/cnn64_cifar10.pt"
model = CNN64()
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

transform = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.ToTensor(),
])

@app.post("/classify")
def classify_image(file: UploadFile = File(...)):
    image = Image.open(file.file).convert("RGB")
    x = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(x)
        pred_idx = int(logits.argmax(1).item())

    return {
        "predicted_class": pred_idx,
        "class_name": CIFAR10_CLASSES[pred_idx]
    }

