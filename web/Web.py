from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from models.TimeLLM import Model
from utils.tools import EarlyStopping
import json
from azure.storage.blob import BlobServiceClient
import os
from dotenv import load_dotenv  # Add import for load_dotenv

class TimeSeriesRequest(BaseModel):
    data: list[float]
    features: int
    prediction_length: int

app = FastAPI()

class TimeLLMPredictor:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Load config
        with open('configs/TimeLLM.json', 'r') as f:
            self.config = json.load(f)
        
        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = Model(self.config).to(self.device)
        
        # Load checkpoint from Azure Blob Storage
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable not set")
        
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container="models", blob="13-25-03-0.checkpoint")
        checkpoint = torch.load(blob_client.download_blob().readall(), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.eval()

    def predict(self, data, features, pred_len):
        # Reshape input data
        x = torch.tensor(data).float()
        x = x.reshape(-1, features, -1)
        
        # Create mark tensors (if needed by your model)
        x_mark = torch.zeros_like(x)  # Modify as needed
        
        with torch.no_grad():
            prediction = self.model.forecast(
                x.to(self.device),
                x_mark.to(self.device),
                None,
                None
            )
        
        return prediction.cpu().numpy().tolist()

# Initialize predictor
predictor = TimeLLMPredictor()

@app.post("/predict")
async def predict(request: TimeSeriesRequest):
    try:
        prediction = predictor.predict(
            request.data,
            request.features,
            request.prediction_length
        )
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))