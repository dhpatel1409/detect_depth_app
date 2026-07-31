from fastapi import FastAPI , UploadFile
from fastapi.responses import JSONResponse , StreamingResponse
import cv2
import numpy as np
from app.pipeline import run_pipeline
import io

app = FastAPI()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.post("/predict") #client sending data to server , where get is used to get data from server
async def predict(file: UploadFile): #async let server while waiting for i/o instead of freezing
    content = await file.read() # used with async to read the file content asynchronously , read as raw bytes
    img_array = np.frombuffer(content, np.uint8) # convert the bytes to a numpy array
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR) # decode the numpy array to an image using OpenCV

    annotated, detections = run_pipeline(img)

    return JSONResponse({'detection': detections})

@app.post("/predict_image")
async def predict_image(file: UploadFile):
    content = await file.read()
    img_array = np.frombuffer(content, np.uint8)
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)

    annotated, detections = run_pipeline(img)

    # Convert the annotated image to bytes
    _, buffer = cv2.imencode('.jpg', annotated)

    return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg") # buffer to byte and file like object to send as response