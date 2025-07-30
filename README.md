# **HAM Fine-Tune: Skin Lesion Classification**

This project implements a deep learning solution for classifying skin lesions based on the HAM10000 dataset. It leverages a fine-tuned ResNet model, provides a RESTful API for inference, and includes a user-friendly web interface for interacting with the model. The entire application is containerized using Docker for easy deployment and reproducibility.

## **Table of Contents**

* [Project Overview](#project-overview)
* [Features](#features)
* [Project Structure](#project-structure)
* [Getting Started](#getting-started)
    * [Prerequisites](#prerequisites)
    * [Setup and Installation](#setup-and-installation)
* [Training the Model](#training-the-model)
* [API Usage](#api-usage)
* [UI Usage](#ui-usage)
* [Model Details](#model-details)
* [License](#license)

## **Project Overview**

The core objective of this project is to classify dermatoscopic images into one of seven diagnostic categories of common pigmented skin lesions. It utilizes the HAM10000 dataset, a large collection of multi-source dermatoscopic images of common pigmented skin lesions. The solution involves:

1. **Data Preparation:** Processing and splitting the HAM10000 dataset.  
2. **Model Training:** Fine-tuning a pre-trained ResNet model on the prepared dataset using PyTorch.  
3. **Inference API:** A FastAPI application to serve predictions from the trained model.  
4. **Web User Interface:** A Streamlit application for easy image upload and prediction visualization.

## **Features**

* **End-to-End Solution:** From data handling to model deployment.  
* **Deep Learning Model:** Utilizes a fine-tuned ResNet for high accuracy.  
* **Containerized Deployment:** Docker and Docker Compose for seamless setup and execution.  
* **RESTful API:** Programmatic access to model inference.  
* **Interactive UI:** User-friendly web interface for practical use.  
* **Pre-trained Model:** Includes a pre-trained model for immediate inference.  
* **Experiment Tracking:** Integrates with Weights & Biases (WandB) for experiment tracking, model versioning, and evaluation of multiple model iterations.

## **Project Structure**

The repository is organized as follows:
```
.  
├── .dockerignore  
├── .gitattributes  
├── .gitignore  
├── docker-compose.yml           \# Defines Docker services for API and UI  
└── project/  
    ├── LICENSE  
    ├── api/                     \# FastAPI application for model inference  
    │   ├── Dockerfile           \# Dockerfile for the API service  
    │   ├── main.py              \# Main API application  
    │   └── requirements.txt     \# Python dependencies for the API  
    ├── artifacts/               \# Stores trained models  
    │   └── my-model:v2/  
    │       └── model.pth        \# Pre-trained model weights  
    ├── data/                    \# Data processing scripts and datasets  
    │   ├── HAM10000\_metadata.csv \# Original dataset metadata  
    │   ├── data\_encoding.py     \# Script for encoding categorical data  
    │   ├── datamodule.py        \# PyTorch Lightning DataModule for HAM10000  
    │   ├── enc\_HAM10000\_metadata.csv \# Encoded metadata  
    │   ├── samplers.py          \# Custom data samplers  
    │   ├── split\_csv.py         \# Script to split metadata into train/val/test  
    │   ├── test\_set.csv         \# Test set metadata  
    │   ├── train\_set.csv        \# Training set metadata  
    │   ├── val\_set.csv          \# Validation set metadata  
    ├── models/                  \# Model definition and transformations  
    │   ├── .DS\_Store  
    │   ├── model.py             \# PyTorch model definition (ResNet)  
    │   ├── transforms.py        \# Image transformation pipelines  
    │   └── weights/  
    │       └── resnet50\_imagenet\_v2.pth \# Pre-trained ResNet weights  
    ├── pyproject.toml           \# Project configuration (e.g., poetry)  
    ├── train.py                 \# Main training script  
    └── trainer/                 \# Custom training utilities  
        ├── callbacks/  
        │   └── base.py          \# Base callback definitions  
        └── trainer.py           \# Custom training loop/logic  
└── ui/                      \# Streamlit web application  
    ├── Dockerfile               \# Dockerfile for the UI service  
    ├── app.py                   \# Main Streamlit application  
    └── requirements.txt         \# Python dependencies for the UI
```
## **Getting Started**

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

### **Prerequisites**

* [Docker](https://www.google.com/search?q=https://www.docker.com/get-started)  
* [Docker Compose](https://docs.docker.com/compose/install/)

### **Setup and Installation**

1. **Clone the repository (if you only have the files, skip this step):**  
   git clone https://github.com/BM-N/HAMfine-tune  
   cd hamfine-tune

2. Build and run the Docker containers:  
   Navigate to the root directory of the project (where docker-compose.yml is located) and run:  
   docker-compose up \--build

   This command will:  
   * Build the Docker images for the api and ui services based on their respective Dockerfiles.  
   * Start the api service, accessible at http://localhost:8000.  
   * Start the ui service, accessible at http://localhost:8501.

Wait for both services to start. You should see logs indicating that the FastAPI and Streamlit applications are running.

## **Training the Model**

To train your own model, you can run the **train.py** script. This script is designed to be run within the Docker environment or a compatible Python environment.

**Note:** The HAM10000 dataset images are not included in this repository due to their size. You will need to download them separately and place them in a project/data/images directory (or modify **datamodule.py** to point to your image location) if you wish to train from scratch with the full dataset. The HAM10000_metadata.csv file is provided for data splitting.

To run the training script:

1. **Ensure you have the dataset images available.**  
2. **Execute the training script within the api container (after docker-compose up is running):**  
   docker-compose exec api python project/train.py

   This command will start the training process. Trained models will be saved in the project/artifacts directory.

## **API Usage**

The API service runs on http://localhost:8000. You can interact with it using curl, Postman, or any HTTP client.

The API also provides interactive documentation at http://localhost:8000/docs (Swagger UI) and http://localhost:8000/redoc (ReDoc).

### **Endpoints**

* **GET /**:  
  * **Summary**: Health Check  
  * **Description**: A simple endpoint to check if the API is running and responsive. Returns a status message.  
* **POST /predict**:  
  * **Summary**: Classify a Skin Lesion Image  
  * **Description**: This is the primary inference endpoint. It accepts an image file (multipart/form-data), preprocesses it using the model's defined transformations, and returns the predicted skin lesion class along with certainty scores for all categories.  
  * **Request Body**: file (UploadFile) \- The image file to be classified.  
  * **Response**: JSON object containing prediction (the predicted class full name), certainty (the confidence score for the predicted class), and all\_certainties (a dictionary of confidence scores for all seven classes).  
* **GET /test-images**:  
  * **Summary**: Get list of test images and labels  
  * **Description**: This endpoint provides a list of image IDs, their corresponding URLs (for static serving), and their ground truth diagnostic labels from the test\_set.csv. This is particularly useful for the UI to display pre-selected test images for demonstration or evaluation.  
  * **Response**: JSON array of objects, where each object contains image\_id, image\_url, and dx\_full (the full diagnostic name).

**Example using curl (replace path/to/your/image.jpg with an actual image file):**

curl \-X POST "http://localhost:8000/predict/" \\  
     \-H "accept: application/json" \\  
     \-H "Content-Type: multipart/form-data" \\  
     \-F "file=@path/to/your/image.jpg;type=image/jpeg"

## **UI Usage**

The web UI service runs on http://localhost:8501.

1. Open your web browser and navigate to http://localhost:8501.  
2. You will see a simple interface where you can upload a skin lesion image for prediction.  
3. **Alternatively, you can select an image from the pre-defined test set.** The UI fetches a list of test images from the API, allowing you to choose one and immediately see the model's prediction for that specific image. This feature is useful for quickly evaluating the model's performance on known data.  
4. Upload an image, or select a test image, and the UI will send it to the API for prediction and display the results.

## **Model Details**

* **Architecture:** The project uses a ResNet-50 model, pre-trained on ImageNet, and then fine-tuned on the HAM10000 dataset.  
* **Dataset:** [Skin Cancer MNIST: HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000). This dataset consists of 10,015 dermatoscopic images of common pigmented skin lesions.  
* **Classes:** The model classifies images into seven categories:  
  * akiec (Actinic Keratoses and Intraepithelial Carcinoma)  
  * bcc (Basal Cell Carcinoma)  
  * bkl (Benign Keratosis-like lesions)  
  * df (Dermatofibroma)  
  * nv (Melanocytic nevi)  
  * vasc (Vascular lesions)  
  * mel (Melanoma)

## **License**

This project is licensed under the MIT License \- see the project/LICENSE file for details.