import csv

import pandas as pd
import numpy as np


def get_data(file_path):
    df=pd.read_csv(file_path)
    columns=list(df.columns)
    dtypes={x: np.float16 for x in columns if x.startswith(('feature','target'))}
    df=df.astype(dtypes)
    return df


def get_feature_names(df):
    return [c for c in df.columns if 'feature' in c]



            


    
