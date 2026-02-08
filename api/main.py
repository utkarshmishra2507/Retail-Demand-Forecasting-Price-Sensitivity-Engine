from fastapi import FastAPI
import pickle
import pandas as pd

app = FastAPI(title="Retail Demand Forecasting API")

with open("models/random_forest.pkl", "rb") as f:
    model = pickle.load(f)

FEATURES = [
    'Inventory Level','Units Ordered','Price','Discount',
    'effective_price','price_vs_competitor',
    'Promotion','Competitor Pricing',
    'day','month','weekday','weekend'
]

@app.post("/predict")
def predict(data: dict):
    df = pd.DataFrame([data])
    prediction = model.predict(df[FEATURES])[0]
    return {"predicted_units_sold": float(prediction)}
