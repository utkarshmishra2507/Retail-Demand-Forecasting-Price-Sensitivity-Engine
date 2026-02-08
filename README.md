#  Retail Demand Forecasting & Price Sensitivity Engine

An end-to-end data science and machine learning project that enables retailers to
forecast product demand and evaluate price sensitivity using historical sales data.

The system integrates demand forecasting models, price elasticity analysis,
real-time price simulation, and an API-driven architecture to support
data-driven retail decision-making.

---

##  Problem Statement

Retailers often face challenges in:
- Accurately forecasting future demand
- Understanding how price changes affect sales
- Optimizing inventory levels and pricing strategies

This project addresses these challenges by combining machine learning models
with economic insights to deliver actionable forecasts and pricing intelligence.

---

##  Solution Overview

The solution includes:
- A demand forecasting model built using Random Forest
- Category-level price sensitivity and elasticity analysis
- Real-time price simulation through an interactive frontend
- A FastAPI backend for serving model predictions
- A Streamlit-based frontend for visualization and scenario testing

---

## ⚙️ Tech Stack

- Python
- Pandas, NumPy
- Scikit-learn
- Statsmodels
- FastAPI
- Streamlit
- Matplotlib

---

## 📊 Model Performance

| Model | MAE | RMSE |
|-----|-----|------|
| Baseline | 32.95 | 41.62 |
| Linear Regression | 28.73 | 36.24 |
| Random Forest | **25.58** | **33.48** |

---

##  How to Run

### Install Dependencies
```bash
pip install -r requirements.txt

Run Backend API
uvicorn api.main:app --reload

API Documentation:
http://127.0.0.1:8000/docs

Run Streamlit App
streamlit run app.py

------------------------------

👤 Author
Utkarsh

-------------------------------

📄 License
This project is intended for educational and portfolio use.
