import io
import os
from contextlib import asynccontextmanager

import pandas as pd
import torch
import torch.nn as nn
from data.datamodule import get_loss_class_weights
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from models.model import get_model
from models.transforms import get_transforms
from PIL import Image

device = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
_, _, class_names = get_loss_class_weights(
    os.path.abspath("data/enc_HAM10000_metadata.csv")
)
class_names_full = {
    "akiec": "Actinic Keratoses",
    "bcc": "Basal Cell Carcinoma",
    "bkl": "Benign Keratosis-like Lesions",
    "df": "Dermatofibroma",
    "mel": "Melanoma",
    "nv": "Melanocytic Nevi",
    "vasc": "Vascular Lesions",
}

resnet_path = "models/weights/resnet50_imagenet_v2.pth"
model_path = "model.pth"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Manages the application's lifespan. The model is loaded on startup
    and stored in the app's state.
    """
    print("Application startup: Loading model...")
    try:
        app.state.class_names = class_names
        print("Class names loaded successfully.")
        print(f"Model loaded successfully and is running on {device}.")
        print(
            "Using validation transforms from models/transforms.py for preprocessing."
        )
        new_head = nn.Sequential(
            nn.Linear(2048, 512),
            nn.ReLU(),
            nn.Dropout(p=0.36057091203514374),
            nn.Linear(512, 7),
        )

        model = get_model(name="resnet50", new_head=new_head, weight_path=resnet_path)

        model.load_state_dict(torch.load(model_path, map_location=device, weights_only=False))
        print("Successfuly loaded model from artifact")
        model.to(device)
        model.eval()
        app.state.model = model
    except Exception as e:
        print(f"Application startup failed: Could not load model. {e}")
        app.state.model = None
        app.state.class_names = None

    yield

    print("Application shutdown: Clearing model from memory.")
    app.state.model = None
    app.state.class_names = None


app = FastAPI(
    title="Skin Lesion Classifier API (Project-Aware)",
    description="API that uses the fine-tuned model to classify skin lesions.",
    version="1.0.0",
    lifespan=lifespan,
)
# static file 'serving'
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
    class_names = request.app.state.class_names
    if not model or not class_names:
        raise HTTPException(
            status_code=503,
            detail="Model or class names not available. Check server logs.",
        )

    image_bytes = await file.read()
    image_tensor = transform_image(image_bytes)
    probabilities, predicted_index = get_prediction(model, image_tensor)

    predicted_class_abbrev = class_names[predicted_index]
    predicted_class_full_name = class_names_full[predicted_class_abbrev]

    certainty_scores = {
        class_names_full[class_names[i]]: prob.item()
        for i, prob in enumerate(probabilities)
    }

    return JSONResponse(
        content={
            "prediction": predicted_class_full_name,
            "certainty": certainty_scores[predicted_class_full_name],
            "all_certainties": certainty_scores,
        }
    )


@app.get("/test-images", summary="Get list of test images and labels")
def get_test_images(TEST_SET_CSV="data/test_set.csv"):
    try:
        df = pd.read_csv(TEST_SET_CSV)

        def find_image_url(image_id):
            path1 = f"data/HAM10000_images_part_1/{image_id}.jpg"
            path2 = f"data/HAM10000_images_part_2/{image_id}.jpg"

            if os.path.exists(path1):
                return f"/static/HAM10000_images_part_1/{image_id}.jpg"
            elif os.path.exists(path2):
                return f"/static/HAM10000_images_part_2/{image_id}.jpg"
            return None

        df["image_url"] = df["image_id"].apply(find_image_url)
        df = df.dropna(subset=["image_url"])

        df["dx"] = df["label"].apply(lambda label_int: class_names[label_int])
        df["dx_full"] = df["dx"].map(class_names_full)

        result = df[["image_id", "image_url", "dx_full"]].to_dict(orient="records")
        return JSONResponse(content=result)
    except FileNotFoundError:
        raise HTTPException(
            status_code=500, detail=f"Test set CSV not found at {TEST_SET_CSV}"
        )
    except (KeyError, IndexError) as e:
        raise HTTPException(
            status_code=500, detail=f"Error mapping labels in test set: {e}"
        )
