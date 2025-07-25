#FastAPI server
from fastapi import FastAPI
from pydantic import BaseModel
import joblib 
import pandas as pd 
from config import collection

app = FastAPI()
model = joblib.load("model.pkl")

class CustomerInput(BaseModel):
    age: int
    tenure: int
    services: list

@app.post("/predict/")
def predict_monthly_charge(data: CustomerInput):
    input_dict = data.dict()
    input_dict["num_services"]=len(input_dict["services"])
    input_dict.pop("services")

    df= pd.DataFrame([input_dict])
    prediction = model.predict(df)[0]

    #Save prediction to MongoDB
    record = data.dict()
    record["predicted_monthly_charge"] = round(float(prediction),2)
    collection.insert_one(record)

    return{"predicted_monthly_charged": round(float(prediction),2)}