from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image

app = FastAPI()

# Load the pre-trained model
model = load_model("flower_classification_model.h5")  # Replace with your model file

# Class indices (this should match the output from your training script)
class_indices = {0: 'bougainvillea', 1: 'daisies', 2: 'garden_roses', 3: 'gardenias', 4: 'hibiscus',
                 5: 'hydrangeas', 6: 'lilies', 7: 'orchids', 8: 'peonies', 9: 'tulip'}  # Replace with actual class names

def preprocess_image(image_data):
    img = Image.open(image_data)
    img = img.resize((224, 224))  # Resize to match model's expected input size
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize if your model expects it
    return img_array

@app.get('/')
def index():
    return {'message': 'Flower Classification ML API'}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    try:
        # Preprocess the image
        img_array = preprocess_image(file.file)
        
        # Run prediction
        predictions = model.predict(img_array)
        predicted_class_index = np.argmax(predictions[0])  # Get the class with the highest probability
        predicted_class = class_indices[predicted_class_index]  # Map index to class label
        
        return {"prediction": predicted_class}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
