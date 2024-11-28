import pandas as pd
import pickle
from sklearn.metrics import accuracy_score
import yaml
import os
import mlflow

from urllib.parse import urlparse

os.environ['MLFLOW_TRACKING_URI'] = "https://dagshub.com/varshavt/ML-Pipeline.mlflow"
os.environ["MLFLOW_TRACKING_USERNAME"] = "varshavt"
os.environ["MLFLOW_TRACKING_PASSWORD"] = "93a1d31b0867bb5b411eed74a7a0bb6b60ffd065"

## Load the paraameters from params.yaml
params=yaml.safe_load(open("params.yaml"))["train"]

def evaluate(data_path, model_path):
    data= pd.read_csv(data_path)
    X = data.drop(columns=["Outcome"])
    y = data["Outcome"]

    mlflow.set_tracking_uri("https://dagshub.com/varshavt/ML-Pipeline.mlflow")

    ## Loading the model 
    model = pickle.load(open(model_path,'rb'))
    prediction = model.predict(X)
    accuracy = accuracy_score(y, prediction)

    ## Log metrix
    mlflow.log_metric("accuracy",accuracy)
    print(f"Model accuracy {accuracy}")

if __name__=="__main__":
    evaluate(params["data"], params["model"])