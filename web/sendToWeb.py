import requests
import numpy as np

# Create sample data
data = np.random.randn(7, 96).tolist()  # 7 features, 96 timesteps

# Prepare request
payload = {
    "data": data,
    "features": 7,
    "prediction_length": 24
}


# Make prediction
response = requests.post("http://localhost:8000/predict", json=data)
print(response.json())

# View plot
response = requests.get("http://localhost:8000/plot")
print(response.json())


# on command line run
# uvicorn app:app --reload --host 0.0.0.0 --port 8000