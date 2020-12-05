# enconding:utf8
import os
import math
import numpy as np
import pandas as pd
import lightgbm as lgb
import pickle
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold,KFold
from sklearn.metrics import mean_squared_error
import time
import warnings

warnings.filterwarnings("ignore")

train_file = './pre_data/training.pkl'
data_set = pickle.load(open(train_file,'rb'))
data_set.fillna(0.,inplace=True)

label = data_set['label'].values # ndarray

feature_list = list(data_set.columns)
feature_list.remove('id')
feature_list.remove('label')

training = data_set[feature_list].values

test_data = pickle.load(open('./pre_data/eval.pkl','rb'))
test_data.fillna(0.,inplace=True)
sub_df = test_data['id'].copy()

del test_data['id']
test_data = test_data.values

# kf = KFold(n_splits = 5,random_state=2017,shuffle=True)
kf = KFold(n_splits = 5,random_state=2020,shuffle=True)
rmse_list = []
sub_pred = []
for train_index, val_index in kf.split(training):
    X_train, y_train, X_val, y_val = training[train_index], label[train_index],training[val_index],label[val_index]

    # lgb model
    ''''max_depth在10的时候得分有所下降
        num_leaves大了得分下降
    '''
    params = {
            'task': 'train','boosting_type': 'gbdt','objective': 'regression',
            'metric': {'l2', 'rmse'},'max_depth':5,'num_leaves':31,
            'min_data_in_leaf':20,'learning_rate': 0.05,
            'feature_fraction': 0.8,'bagging_fraction': 0.8,'bagging_freq': 5,
            # 'num_boost_round':1500,
            'num_boost_round':5000,
            # 'verbose': -2
            'seed': 1,     #我加的
            'bagging_seed': 1,#我加的
            'verbose': -1
    }

    lgb_train = lgb.Dataset(X_train,label=y_train,feature_name=feature_list)
    lgb_eval = lgb.Dataset(X_val,label=y_val,feature_name=feature_list, reference=lgb_train)
    gbm = lgb.train(params,lgb_train,valid_sets=lgb_eval,early_stopping_rounds=100)
    y_pred = gbm.predict(X_val, num_iteration=gbm.best_iteration)
    rmse = mean_squared_error(y_val, y_pred) ** 0.5
    print("rmse:",rmse)
    rmse_list.append(rmse)

    # xgb model

    # sub
    test_pred = gbm.predict(test_data, num_iteration=gbm.best_iteration)
    sub_pred.append(test_pred)


print("kflod rmse: {}\n mean rmse : {}".format(rmse_list, np.mean(np.array(rmse_list))))


pred = np.mean(np.array(sub_pred),axis=0)
# sub_df.loc[:,'pred'] = pred
# sub_df.loc[:,'pred'] = pred
# sub_df.iloc[:, 'pred'] = pred
# sub_df.values[:,'pred'] = pred
sub_df.to_csv('submission.csv',sep=',',header=None,index=False,encoding='utf8')


fi=open('re.txt','w')
def n_pred():
    for i in pred:
        print(i)
        fi.write(str(i)+'\n')
n_pred()




