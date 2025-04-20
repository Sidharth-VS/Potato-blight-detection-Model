from fastapi import FastAPI, File, UploadFile
import uvicorn
from io import BytesIO
import numpy as np
from PIL import Image
from model import Model
import torch

model = Model()

app = FastAPI()

def read_file_as_image(data):
    image = Image.open(BytesIO(data))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    res = model.getPrediction(image)
    return res

if __name__ == "__main__":
    uvicorn.run(app, host='localhost', port=8000)