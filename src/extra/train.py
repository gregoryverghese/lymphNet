import sys
sys.path.append("..") 
import argparse
import pickle
import json

#import numerapi
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.model_selection import RandomizedSearchCV
from lightgbm import LGBMRegressor

import utils
import evaluation

params={'num_leaves': [30, 40, 50], 
        'max_depth': [4, 5, 6], 
        'learning_rate': [0.05, 0.01, 0.005],
        'bagging_freq':[7], 
        'bagging_fraction': [0.6, 0.7, 0.8], 
        'feature_fraction': [0.85, 0.75, 0.65]}

def train():
    print('getting files...')    
    try:
        train_data=utils.get_data(TRAIN_PATH)
        test_data=utils.get_data(TEST_PATH)
    except Exception as e:
        print(e)
        #num_api = numerapi.NumerAPI(PUBLIC_KEY, SECRET_GUY,verbosity="info")
        #num_api.download_current_dataset(dest_path='../data/')
        #train_data=utils.get_data(TRAIN_PATH)
        #test_data=utils.get_data(TEST_PATH)

    feature_names=utils.get_feature_names(train_data)
    print('performing pca dimensions reduction...',flush=True)
    #use pca for dimensionality reduction
    pca=PCA(n_components=N_COMPONENTS)
    pca.fit(train_data[feature_names])
    x_train_pca = pca.transform(train_data[feature_names])
    x_test_pca = pca.transform(test_data[feature_names])

    print('injecting noise into dataset...',flush=True)
    #corrupt dataset using gaussian noise
    mu,sigma=0,0.1
    noise=np.random.normal(mu,sigma,x_train_pca.shape)
    x_train_pca_noise=x_train_pca+noise

    print('training lgbm model...',flush=True)
    #train an LGBMRegressor model - use random search for parameter tuning
    #with cross validation
    lgb=LGBMRegressor()
    lgb_randomsearch=RandomizedSearchCV(estimator=lgb,cv=CV,param_distributions=params, n_iter=100)
    lgb_model=lgb_randomsearch.fit(x_train_pca_noise,train_data['target'])
    lgb_model_best=lgb_model.best_estimator_
    lgb_model_best=lgb_model_best.fit(x_train_pca_noise,train_data['target'])
    
    print("Generating all predictions...")
    train_data['prediction'] = lgb_model_best.predict(x_train_pca_noise)
    test_data['prediction'] = lgb_model_best.predict(x_test_pca)

    train_corrs = (evaluation.per_era_score(train_data))
    print('train correlations mean: {}, std: {}'.format(train_corrs.mean(), train_corrs.std(ddof=0)))
    #print('avg per-era payout: {}'.format(evaluation.payout(train_corrs).mean()))

    valid_data = test_data[test_data.data_type == 'validation']
    valid_corrs = evaluation.per_era_score(valid_data)
    #valid_sharpe = evaluation.sharpe(valid_data)
    print('valid correlations mean: {}, std: {}'.format(valid_corrs.mean(), valid_corrs.std(ddof=0)))
    #print('avg per-era payout {}'.format(evaluation.payout(valid_corrs.mean())))
    #print('valid sharpe: {}'.format(valid_sharpe))

    #live_data = test_data[test_data.data_type == "test"]
    #live_corrs = evaluation.per_era_score(test_data)
    #test_sharpe = evaluation.sharpe(test_data)
    #print('live correlations - mean: {}, std: {}'.format(live_corrs.mean(),live_corrs.std(ddof=0)))
    #print('avg per-era payout is {}'.format(evaluation.payout(live_corrs).mean()))
    #print('live Sharpe: {}'.format(test_sharpe))
    
    #pickle and save the model
    with open('lgbm_model_round_253.pkl', 'wb') as f:
        pickle.dump(lgb_model,f)

    #save down predictions
    valid_corrs.to_csv('valid_predictions.csv')

if __name__ == '__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('-cp','--configpath',required=True,help='config path')

    args = vars(ap.parse_args())
    config_path=args['configpath']
    with open(config_path) as json_file:
        config=json.load(json_file)
    
    PUBLIC_KEY=config['public_key']
    SECRET_KEY=config['secret_key']
    MODEL_ID=config['model_id']

    TRAIN_PATH=config['train_path']
    TEST_PATH=config['test_path']

    N_COMPONENTS=config['n_components']
    CV=config['cv_folds']

    train()
