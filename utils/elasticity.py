import pickle
import pandas as pd

def load_elasticity():
    with open("models/elasticity_results.pkl", "rb") as f:
        return pickle.load(f)

def elasticity_df(results):
    return pd.DataFrame(
        results.items(),
        columns=["Category", "Elasticity"]
    )
