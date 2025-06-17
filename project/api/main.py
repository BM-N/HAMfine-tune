import io
import os
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager

from data.datamodule import get_loss_class_weights
from models.model import get_model
from models.transforms import get_transforms


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan. The model is loaded on startup
    and stored in the app's state.
    """
    print("Application startup: Loading model...")
    try:
        app.state.model = load_inference_model()
        print(f"Model loaded successfully and is running on {device}.")
        print(
            "Using validation transforms from models/transforms.py for preprocessing."
        )
    except Exception as e:
        print(f"Application startup failed: Could not load model. {e}")
        app.state.model = None

    yield

    print("Application shutdown: Clearing model from memory.")
    app.state.model = None


app = FastAPI(
    title="Skin Lesion Classifier API (Project-Aware)",
    description="API that uses the fine-tuned model to classify skin lesions.",
    version="1.0.0",
    lifespan=lifespan,
)
_, _, class_names = get_loss_class_weights(
    os.path.abspath("data/enc_HAM10000_metadata.csv")
)
class_names_full = {
    'akiec': 'Actinic Keratoses', 'bcc': 'Basal Cell Carcinoma',
    'bkl': 'Benign Keratosis-like Lesions', 'df': 'Dermatofibroma',
    'mel': 'Melanoma', 'nv': 'Melanocytic Nevi', 'vasc': 'Vascular Lesions'
}
model_path = "artifacts/my-model:v2/model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_inference_model(model_path=model_path, device=device):
    """
    Loads the model architecture, replaces the classifier head to match the
    exact architecture used during training, and then loads the fine-tuned
    weights from the specified artifact path.
    """
    
    new_head = nn.Sequential(
        nn.Linear(2048, 512),
        nn.ReLU(),
        nn.Dropout(p=0.36057091203514374),
        nn.Linear(512, 7),
    )
    model = get_model(name="resnet50", new_head=new_head)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Successfuly loaded model from artifact")
    except FileNotFoundError:
        print(
            f"ERROR: Model file not found at {model_path}. Please place your trained model artifact in the project root."
        )
        raise
    except Exception as e:
        print(f"ERROR: An error occurred while loading the model state_dict: {e}")
        raise
    model = model.to(device)
    model.eval()

    return model

# --- Static File Serving for Test Images ---
app.mount("/static", StaticFiles(directory="data"), name="static")

preprocess = get_transforms(train=False)


def transform_image(image_bytes: bytes) -> torch.Tensor:
    """
    Takes image bytes, applies the project's validation transforms, and returns a tensor.
    """
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return preprocess(image).unsqueeze(0)  # type: ignore
    except Exception as e:
        print(f"Error transforming image: {e}")
        raise HTTPException(
            status_code=400, detail="Invalid image file. Could not process."
        )


def get_prediction(model: nn.Module, tensor: torch.Tensor):
    """
    Performs inference on the input tensor using the loaded model.
    """
    tensor = tensor.to(device)
    with torch.no_grad():
        outputs = model(tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        predicted_index = torch.argmax(probabilities).item()
    return probabilities, predicted_index


# API endpoints
@app.get("/", summary="Health Check")
def read_root():
    return {"status": "ok", "message": "Welcome to the HAM10000 Classifier API!"}


@app.post("/predict", summary="Classify a Skin Lesion Image")
async def predict(request: Request, file: UploadFile = File(...)):
    """
    The main prediction endpoint. It uses the project's specific modules
    for model architecture and image preprocessing.
    """
    model = request.app.state.model
    if model is None:
        raise HTTPException(
            status_code=503, detail="Model is not available. Please check server logs."
        )

    image_bytes = await file.read()
    image_tensor = transform_image(image_bytes)
    probabilities, predicted_index = get_prediction(model, image_tensor)

    certainty_scores = {
        class_names[i]: prob.item() for i, prob in enumerate(probabilities)
    }
    predicted_class_name = class_names[predicted_index]  # type: ignore

    return JSONResponse(
        content={
            "prediction": predicted_class_name,
            "certainty": certainty_scores[predicted_class_name],
            "all_certainties": certainty_scores,
        }
    )
    
    
app.get("/test-images", summary="Get list of test images and labels")
def get_test_images(TEST_SET_CSV = "data/test_set.csv"):
    try:
        df = pd.read_csv(TEST_SET_CSV)
        def find_image_url(image_id):
            if os.path.exists(f"data/HAM10000_images_part_1/{image_id}.jpg"):
                return f"/static/HAM10000_images_part_1/{image_id}.jpg"
            if os.path.exists(f"data/HAM10000_images_part_2/{image_id}.jpg"):
                return f"/static/HAM10000_images_part_2/{image_id}.jpg"
            return None

        df['image_url'] = df['image_id'].apply(find_image_url)
        df = df.dropna(subset=['image_url'])
        df['dx_full'] = df['dx'].map(class_names_full)
        result = df[['image_id', 'image_url', 'dx_full']].to_dict(orient="records")
        return JSONResponse(content=result)
    except FileNotFoundError:
        raise HTTPException(status_code=500, detail="Test set CSV not found.")

# uvicorn project.api.main:app --reload