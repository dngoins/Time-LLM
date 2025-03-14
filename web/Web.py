from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import torch
import numpy as np
from models.TimeLLM import TimeLLM
from utils.tools import EarlyStopping
import json
from azure.storage.blob import BlobServiceClient
import os
from dotenv import load_dotenv
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from fastapi.responses import FileResponse  # Add import for FileResponse

class TimeSeriesRequest(BaseModel):
    data: list[float]
    features: int
    prediction_length: int

app = FastAPI()

class TimeLLMPredictor:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Load config from file
        with open('configs/TimeLLM.json', 'r') as f:
            self.config = json.load(f)
        
        # Deserialize config to attributes
        obj = self 
        #  
        for key, value in self.config.items():
            setattr(obj, key, value)

        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TimeLLM(obj).to(self.device)
        
        # Load checkpoint from local file or Azure Blob Storage
        local_checkpoint_path = 'checkpoints/model.pt'
        if not os.path.exists(local_checkpoint_path):
            connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
            if not connection_string:
                raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable not set")
            
            blob_service_client = BlobServiceClient.from_connection_string(connection_string)
            blob_client = blob_service_client.get_blob_client(container="models", blob="13-25-03-0.checkpoint")
            with open(local_checkpoint_path, "wb") as f:
                f.write(blob_client.download_blob().readall())
        
        checkpoint = torch.load(local_checkpoint_path, map_location=self.device)
        self.model.load_state_dict(checkpoint)
        self.model.eval()
        print("Model loaded successfully\nWeb server running...\n")

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
        
        return prediction.cpu().numpy()

    def plot_timeseries(self, data, prediction):
        sns.set(style="darkgrid")
        plt.figure(figsize=(10, 6))
        sns.lineplot(data=data, palette="tab10", linewidth=2.5, label="Actual")
        sns.lineplot(data=prediction, color="red", linewidth=2.5, label="Prediction")
        plt.title("Time Series Data")
        plt.xlabel("Time")
        plt.ylabel("Value")
        plt.legend()
        plot_path = "timeseries_plot.png"
        plt.savefig(plot_path)
        plt.close()
        return plot_path

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

@app.get("/plot")
async def plot():
    # Sample data for demonstration
    sample_data = np.random.randn(100).tolist()
    sample_prediction = np.random.randn(100).tolist()
    plot_path = predictor.plot_timeseries(sample_data, sample_prediction)
    return FileResponse(plot_path)

# if __name__ == "__main__":
#     import uvicorn
#     uvicorn.run("web.Web:app", host="0.0.0.0", port=8000, reload=True)
    
#     # Keep the application running
#     while True:
#         pass