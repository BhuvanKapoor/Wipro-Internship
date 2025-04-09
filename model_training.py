import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.arima.model import ARIMA


def train_linear_regression(df):
    X = df['Date'].map(pd.Timestamp.toordinal).values.reshape(-1,1)
    y = df['moving_avg_centered'].fillna(0)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model_LR = LinearRegression()
    model_LR.fit(X_train, y_train)

    return model_LR, X_test, y_test

def train_ARIMA(df, order=(3,1,0), train_size=0.8):
    train, test = df.iloc[:int(len(df)*0.8)].dropna(), df.iloc[int(len(df)*0.8):].dropna()
    model_ARIMA = ARIMA(train['moving_avg_centered'])
    model_ARIMA_fit = model_ARIMA.fit()
    return model_ARIMA_fit, test['moving_avg_centered']
