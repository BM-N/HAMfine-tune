FROM python:3.12-slim

RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY ./project/api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY ./project /app/project
COPY ./project/pyproject.toml /app/pyproject.toml
COPY ./project/artifacts/my-model:v2/model.pth /app/model.pth
COPY ./project/models/weights/resnet50_imagenet_v2.pth /app/models/weights/resnet50_imagenet_v2.pth

RUN pip install -e /app/project

CMD ["uvicorn", "project.api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "4"]