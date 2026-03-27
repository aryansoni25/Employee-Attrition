from fastapi import FastAPI
import torch
from model.model import GaussianMLEstimatorNN

app = FastAPI()

num_features = 10

model = GaussianMLEstimatorNN(num_features, 2)
model.load_state_dict(torch.load("model/attrition_model.pth"))
model.eval()

from pydantic import BaseModel

class EmployeeData(BaseModel):
    features: list
@app.post("/predict")
def predict(data: EmployeeData):

    x = torch.tensor([data.features], dtype=torch.float32)

    with torch.no_grad():
        output = model(x)

    pred = torch.argmax(output, dim=1).item()
    prob = torch.softmax(output, dim=1)[0][1].item()

    return {
    "prediction": "Employee likely to leave" if pred == 1 else "Employee likely to stay",
    "confidence": round(prob * 100, 2)
}