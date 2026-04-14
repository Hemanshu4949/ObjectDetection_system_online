from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Load your heavy, highly accurate model into server memory
model = YOLO("best.pt") 

@app.post("/analyze-food")
async def analyze_food(file: UploadFile = File(...)):
    # 1. Read the image sent by the Android app
    image_data = await file.read()
    image = Image.open(io.BytesIO(image_data))

    # 2. Run the YOLO prediction
    results = model(image)
    
    # 3. Parse the results
    best_confidence = 0.0
    detected_product = "unknown"

    for r in results:
        boxes = r.boxes
        for box in boxes:
            conf = float(box.conf[0])
            if conf > best_confidence:
                best_confidence = conf
                # Get the class name directly from the model!
                detected_product = model.names[int(box.cls[0])] 

    # 4. Send it back to the Android phone
    return {
        "status": "success",
        "product": detected_product,
        "confidence": best_confidence
    }