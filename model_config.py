model_config = {
    'linear_reg': {
        'model_class' : 'sklearn.linear_model.LinearRegression',
        'params':{}
    },
    'arima':{
        'model_class': 'statsmodels.tsa.arima.model.ARIMA',
        'params': {
            'order': (3,1,0)
        }
    }
}

def get_model_params(model_name):
    if model_name.lower() not in model_config:
        raise ValueError(f'{model_name} not available. List of available models {list(model_config.keys())}')
    return model_config[model_name.lower()]