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

class TimeSeriesRequest(BaseModel):
    data: list[float]
    features: int
    prediction_length: int

app = FastAPI()

class TimeLLMPredictor:
    def __init__(self):
        # Load environment variables
        load_dotenv()

        # Parse arguments
        parser = argparse.ArgumentParser(description="TimeLLM Configuration")
        parser.add_argument('--model_name', type=str, default="TimeLLM")
        parser.add_argument('--task_name', type=str, default="long_term_forecast")
        parser.add_argument('--input_size', type=int, default=10)
        parser.add_argument('--hidden_size', type=int, default=50)
        parser.add_argument('--num_layers', type=int, default=2)
        parser.add_argument('--output_size', type=int, default=1)
        parser.add_argument('--dropout', type=float, default=0.2)
        parser.add_argument('--learning_rate', type=float, default=0.001)
        parser.add_argument('--batch_size', type=int, default=64)
        parser.add_argument('--num_epochs', type=int, default=100)
        # forecasting task
        parser.add_argument('--seq_len', type=int, default=512, help='input sequence length')
        parser.add_argument('--label_len', type=int, default=48, help='start token length')
        parser.add_argument('--pred_len', type=int, default=192, help='prediction sequence length')
        parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')

        # model define
        parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
        parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
        parser.add_argument('--c_out', type=int, default=7, help='output size')
        parser.add_argument('--d_model', type=int, default=32, help='dimension of model')
        parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
        parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
        parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
        parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
        parser.add_argument('--moving_avg', type=int, default=25, help='window size of moving average')
        parser.add_argument('--factor', type=int, default=3, help='attn factor')
        parser.add_argument('--embed', type=str, default='timeF',
                            help='time features encoding, options:[timeF, fixed, learned]')
        parser.add_argument('--activation', type=str, default='gelu', help='activation')
        parser.add_argument('--output_attention', action='store_true', help='whether to output attention in encoder')
        parser.add_argument('--patch_len', type=int, default=16, help='patch length')
        parser.add_argument('--stride', type=int, default=8, help='stride')
        parser.add_argument('--prompt_domain', type=int, default=0, help='')
        parser.add_argument('--llm_model', type=str, default='LLAMA', help='LLM model') # LLAMA, GPT2, BERT
        parser.add_argument('--llm_dim', type=int, default='4096', help='LLM model dimension')# LLama7b:4096; GPT2-small:768; BERT-base:768


        # optimization
        parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
        parser.add_argument('--itr', type=int, default=1, help='experiments times')
        parser.add_argument('--train_epochs', type=int, default=5, help='train epochs')
        parser.add_argument('--align_epochs', type=int, default=10, help='alignment epochs')
        
        parser.add_argument('--des', type=str, default='Exp', help='exp description')
        parser.add_argument('--loss', type=str, default='MSE', help='loss function')
        parser.add_argument('--lradj', type=str, default='type1', help='adjust learning rate')
        parser.add_argument('--pct_start', type=float, default=0.2, help='pct_start')
        parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
        parser.add_argument('--llm_layers', type=int, default=32)
        parser.add_argument('--percent', type=int, default=100)
        parser.add_argument('--content', type=str, default='Description')

        args = parser.parse_args()
        self.config = args
        print(f'Configuration: {self.config}')

        # Initialize model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = TimeLLM(self.config).to(self.device)
        
        # Load checkpoint from Azure Blob Storage
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        if not connection_string:
            raise ValueError("AZURE_STORAGE_CONNECTION_STRING environment variable not set")
        
        blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        blob_client = blob_service_client.get_blob_client(container="models", blob="13-25-03-0.checkpoint")
                
        checkpoint = torch.load(blob_client.download_blob().readall(), map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
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
        plt.show()

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
    predictor.plot_timeseries(sample_data, sample_prediction)
    return {"message": "Plot displayed"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)