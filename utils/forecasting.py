import pickle

FEATURES = [
    'Inventory Level','Units Ordered','Price','Discount',
    'effective_price','price_vs_competitor',
    'Promotion','Competitor Pricing',
    'day','month','weekday','weekend'
]

def load_model():
    with open("models/random_forest.pkl", "rb") as f:
        return pickle.load(f)

def forecast(model, df):
    return model.predict(df[FEATURES])


def simulate_price_change(model, df, category, new_price):
    subset = df[df['Category'] == category].copy()

    old_price = subset['Price'].mean()
    subset['Price'] = new_price
    subset['effective_price'] = new_price * (1 - subset['Discount']/100)
    subset['price_vs_competitor'] = new_price - subset['Competitor Pricing']

    old_demand = model.predict(df[df['Category'] == category][FEATURES]).mean()
    new_demand = model.predict(subset[FEATURES]).mean()

    pct_change = ((new_demand - old_demand) / old_demand) * 100

    return old_demand, new_demand, pct_change
