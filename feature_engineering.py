import numpy as np
import pandas as pd

def handling_NaN_valus(df):
    df = df.fillna(0)
    return df

def handling_date(df):
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date'] = df['Date'].values.astype('float64')
    return df

def create_moving_avg_centered(df, window, center):
    moving_avg_centered = df["Consumption"].rolling(window=window, center=center).mean()
    return moving_avg_centered

def create_rolling_avg(df):
    rolling_avg = [df.iloc[:i+1]['Consumption'].mean() for i in range(len(df['Consumption']))]
    return rolling_avg

def create_lag(df, lag):
    lag_data = [df.iloc[i-lag]["Consumption"] if i>lag-1 else np.nan for i in range(len(df["Consumption"]))]
    return lag_data