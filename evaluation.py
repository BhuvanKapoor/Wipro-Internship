from sklearn.metrics import mean_squared_error

def eval_LR_model(X_test, y_test, model):
    y_pred = model.predict(X_test)
    mean_sq_error_LR = mean_squared_error(y_true=y_test, y_pred=y_pred)
    return mean_sq_error_LR

def eval_arima_model(test, model):
    y_pred = model.forecast(steps=len(test))
    mean_sq_error_arima = mean_squared_error(y_true=test, y_pred=y_pred)
    return mean_sq_error_arima