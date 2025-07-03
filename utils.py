import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def load_boston():
    url = "http://lib.stat.cmu.edu/datasets/boston"
    raw = pd.read_csv(url, sep=r"\s+", skiprows=22, header=None)
    data = np.hstack([raw.values[::2], raw.values[1::2, :2]])
    target = raw.values[1::2, 2]
    cols = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE',
            'DIS','RAD','TAX','PTRATIO','B','LSTAT']
    df = pd.DataFrame(data, columns=cols)
    df['MEDV'] = target
    return df

def eval_metrics(y_true, y_pred):
    return mean_squared_error(y_true, y_pred), r2_score(y_true, y_pred)
