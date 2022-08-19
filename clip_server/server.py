from typing import Union
from unicodedata import name
from fastapi import FastAPI, Request
from clip_model import CLIP


app = FastAPI(name='CLIP Server')

# Load CLIP model
model_name = 'ViT-L/14' # Current light-weight state-of-the-art model
device = 'cuda'
model = CLIP(model_name, device)


@app.post('/query')
async def query_by_text(request: Request):
    body = await request.json()
    print(body)
    text_query = body['text_query']

    # Encode text
    text_features = model.encode_text(text_query).cpu().numpy().tolist()
    response = {
        'text_embedding': text_features
    }
    return response