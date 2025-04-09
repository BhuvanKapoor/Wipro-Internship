from sklearn.metrics import mean_squared_error

def model_predict(model, df_to_pred, model_name):
    if model_name.lower() == 'arima':
        if hasattr(model,'forecast'):
            result = model.forecast(len(df_to_pred))
        else:
            result = model.predict(start=df_to_pred[0], end=df_to_pred[-1])
    else:
        result = model.predict(df_to_pred)
    return result

def evaluate_model(y_test, y_pred): 
    mse = mean_squared_error(y_test, y_pred)
    return mse