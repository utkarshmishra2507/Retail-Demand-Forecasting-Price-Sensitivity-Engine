import pandas as pd

def preprocess(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    df['day'] = df['Date'].dt.day
    df['month'] = df['Date'].dt.month
    df['weekday'] = df['Date'].dt.weekday
    df['weekend'] = df['weekday'].isin([5, 6]).astype(int)

    df['effective_price'] = df['Price'] * (1 - df['Discount']/100)
    df['price_vs_competitor'] = df['Price'] - df['Competitor Pricing']

    return df
