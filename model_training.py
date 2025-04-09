import importlib
from model_config import get_model_params

def split_data(df,target_col, train_size=0.8):
    train_df= df.iloc[:int(len(df)*train_size)]
    test_df= df.iloc[int(len(df)*train_size):]

    features_cols = [feature for feature in df.columns if feature!=target_col]

    X_train = train_df[features_cols]
    y_train = train_df[target_col]
    X_test = test_df[features_cols]
    y_test = test_df[target_col]

    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train, model_name):
    model_config = get_model_params(model_name)
    model_class_path = model_config["model_class"]
    params = model_config['params']
    module_path, class_name = model_class_path.rsplit('.',1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)

    if model_name.lower() == 'arima':
        model = model_class(y_train, **params)
        model = model.fit()
    else:
        model = model_class(**params)
        model.fit(X_train, y_train)
    
    return model
    
