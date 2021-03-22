import sys

#import numerapi
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
rom sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.svm import SVR

from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import VotingRegressor
from sklearn.decomposition import PCA

NUM_COMPONENTS=50

#public_key='Z3WSDWZNFSLTNXIJPUGWLXUEXNRGENV6'
#secret_key='LNA57ECIKXLOO73P2JS3CA56BNMAA6BFVFY2YTAQ7IEBS47HWVDFYM7PRHT5OSD2'
#model_id='cf1b66bc-0e73-44dd-95b4-e3021ea3b09a'

#num_api = numerapi.NumerAPI(public_key, secret_key,verbosity="info")    

#num_api.download_current_dataset(dest_path='/home/verghese/breastcancer_ln_deeplearning/scripts/python/data/')





print('loading data...')
#Upload data - note can download directly from python API
trainpath='/home/verghese/breastcancer_ln_deeplearning/scripts/python/data/train.csv'
testpath='/home/verghese/breastcancer_ln_deeplearning/scripts/python/data/test.csv'   
train_data=pd.read_csv(trainpath).set_index('id')
#test_data=pd.read_csv(testpath).set_index('id')


feature_names=[c for c in train_data.columns if 'feature' in c]
print(train_data.isnull().values.any())
#Use PCA to reduce feature space
print('Number of features: {}'.format(len(feature_names)))
#train_data=np.nan_to_num(train_data)
pca=PCA(n_components=NUM_COMPONENTS)
pca.fit(train_data[feature_names])

x_train_pca = pca.transform(train_data[feature_names])
#x_test_pca = pca.transform(test_data[feature_names])

import numpy as np

mu,sigma=0,0.1
noise=np.random.normal(mu,sigma,x_train_pca.shape)
x_train_pca=x_train_pca+noise
#First lets look at random forest
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 2000, num = 15)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print('trainin TreeRegressorrandom forest...')
rf = RandomForestRegressor()
#rf_randomsearch=RandomizedSearchCV(estimator=rf, param_distributions=random_grid,
                                   #cv=10, n_iter=100, n_jobs=-1)

#rf_model=rf_randomsearch.fit(x_train_pca,train_data['target'])
#rf_model_best=rf_model.best_estimator_

#print('best model: {}'.format(rf_model.best_estimator_))
#print('best score: {}'.format(rf_model.best_score_))
#print('best model_params: {}'.format(rf_model.best_params_))

rf_model_best=rf.fit(x_train_pca,train_data['target'])

#Second - Gradient Boosting

loss=['ls', 'lad', 'huber', 'quantile']
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 2000, num = 15)]
max_features = ['auto', 'sqrt']
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
learning_rate = [0.1,0.01,0.001]

random_grid = {'loss': loss,
               'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'learning_rate':learning_rate}

print('training gradient boosting...')
gb = GradientBoostingRegressor()
#gb_randomsearch=RandomizedSearchCV(estimator=gb, param_distributions=random_grid, cv=10, n_iter=100, n_jobs=-1)
#gb_model=gb_randomsearch.fit(x_train_pca,train_data['target'])
#gb_model_best=gb_model.best_estimator_

#print('best model: {}'.format(gb_model.best_estimator_))
#print('best score: {}'.format(gb_model.best_score_))
#print('best model_params: {}'.format(gb_model.best_params_))

gb_model_best=gb.fit(x_train_pca,train_data['target'])
#Extra trees regresor

criterion=['mse', 'mae']
n_estimators = [int(x) for x in np.linspace(start = 20, stop = 2000, num = 15)]
max_depth = [int(x) for x in np.linspace(10, 110, num = 11)]
max_depth.append(None)
min_samples_split = [2, 5, 10]
min_samples_leaf = [1, 2, 4]
bootstrap = [True, False]

random_grid = {'n_estimators': n_estimators,
               'criterion': criterion,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

print('training xtrees...')
xtrees = ExtraTreesRegressor()
#x_randomsearch=RandomizedSearchCV(estimator=xtrees,param_distributions=random_grid,cv=10,n_iter=100,n_jobs=-1)
#x_model=x_randomsearch.fit(x_train_pca,train_data['target'])

#x_model_best=x_model.best_estimator_
x_model_best=xtrees.fit(x_train_pca,train_data['target'])

ensemble = VotingRegressor(estimators=[('rf', rf_model_best), ('gb', gb_model_best),('x', x_model_best)])
ensemble = ensemble.fit(x_train_pca, train_data['target'])
#predictions=ensemble.predict(x_test_pca)


import pickle

# save
with open('model2.pkl','wb') as f:
    pickle.dump(ensemble,f)


#print(predictions)








