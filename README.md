# 🌼 Flower Classification FastAPI Application

This repository contains a FastAPI web application that allows users to classify images of flowers using a pre-trained deep learning model. The model has been trained on a dataset of 10 different flower categories. Users can upload an image, and the API will predict which flower type is in the image.

## 🌍 Live Application URL

The FastAPI application is deployed on AWS and can be accessed via the following link:
[http://13.211.82.83:8000](http://13.211.82.83:8000)

## 📖 About the Project

The FastAPI application provides an easy-to-use interface for flower classification. It utilizes a Convolutional Neural Network (CNN) model that was trained on images of flowers. Users can upload an image through a web interface or via an API request, and the application will return the predicted flower type.

### Flower Categories

The model can classify flowers into the following categories:

1. Bougainvillea
2. Daisies
3. Garden Roses
4. Gardenias
5. Hibiscus
6. Hydrangeas
7. Lilies
8. Orchids
9. Peonies
10. Tulip

### Technologies Used
- **FastAPI**: For building the API.
- **TensorFlow**: Used for loading and serving the pre-trained flower classification model.
- **Pillow**: For image processing.
- **Uvicorn**: ASGI server to run the FastAPI app.

## 🧠 Model Information

The model used in this project is a pre-trained TensorFlow model (`flower_classification_model.h5`). It is a Convolutional Neural Network (CNN) that was trained on flower images of 224x224 pixels. The model was designed to classify 10 different flower types. 

### Model Summary:
- **Input shape**: 224x224 pixels (resized for consistency)
- **Normalization**: Images are scaled to values between 0 and 1.
- **Prediction Output**: The model returns the probability of each flower type, and the class with the highest probability is selected as the prediction.

The model file (`flower_classification_model.h5`) is loaded at the start of the application.

### Root Endpoint

This is the default endpoint for the application. It returns a simple welcome message.

#### **Request**: `GET /`
#### **Response**:
```json
{
  "message": "Flower Classification ML API"
}
