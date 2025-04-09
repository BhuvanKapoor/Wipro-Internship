import numpy as np

def create_moving_avg_centered(df, window, center):
    moving_avg_centered = df["Consumption"].rolling(window=window, center=center).mean()
    return moving_avg_centered

def create_rolling_avg(df):
    rolling_avg = [df.iloc[:i+1]['Consumption'].mean() for i in range(len(df['Consumption']))]
    return rolling_avg

def create_lag(df, lag):
    lag_data = [df.iloc[i-lag]["Consumption"] if i>lag-1 else np.nan for i in range(len(df["Consumption"]))]
    return lag_data