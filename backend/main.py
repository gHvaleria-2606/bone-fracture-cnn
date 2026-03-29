from fastapi import FastAPI, UploadFile, File
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import torch.nn as nn
import io
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = models.resnet18(pretrained=False)

num_classes = 2

model.fc = nn.Linear(model.fc.in_features, num_classes)

# load weights
model.load_state_dict(torch.load("best_resnet_model.pth", map_location="cpu"))
model.eval()

# transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)

    prediction = torch.argmax(output, dim=1).item()

    return {"prediction": int(prediction)}