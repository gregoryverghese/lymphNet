from scipy.stats import spearmanr
import numpy as np
import pandas as pd


#submitted predictions scored using spearman correlation
def spearman(y_true, y_pred):
    corrs, _ = spearmanr(y_true, y_pred, axis=0)
    return corrs


#get scores between predicitons andi target
def score(df,pred_name='prediction',target_name='target'):
    return spearman(df[pred_name],df[target_name])


#get correlation score per era
def per_era_score(df):
    return df.groupby('era').apply(score)


#return sharpe ration on dataset
#fix this to allow direct setup of correlations
def sharpe(df):
    corrs = score(df)
    print(corrs)
    return corrs.mean() / corrs.std()


#The payout is capped ato 25%
def payout(scores):
    return scores.clip(min=-0.25, max=0.25)


#check individual feature exposure to each score
def feature_exposures(df1,df2,feature_names,predict_name='prediction'):
    return df1[feature_name].apply(lambda x: spearman(df2[predict_name],x),axis=0)








