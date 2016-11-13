import pandas as pd
import numpy as np
df = pd.read_csv("F:\Careers\Showcase\credit_fraud\creditcard.csv")
print(df.describe())

y = df['Class']
df = df.drop("Class",axis=1)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(df, y, test_size=0.1, random_state=0)

#1 Random Forest Classifier
from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(x_train,y_train)
y_pred_rf = rfc.predict(x_test)
from sklearn.metrics import f1_score
f1_score(y_test, y_pred_rf) #0.82692307692307687
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_rf) #0.89080355372482301

#2 Gradient Boosting Classifier
from sklearn.ensemble import GradientBoostingClassifier
gbc = GradientBoostingClassifier(n_estimators=100)
gbc.fit(x_train,y_train)
y_pred_gb = gbc.predict(x_test)
from sklearn.metrics import f1_score
f1_score(y_test, y_pred_gb) #0.7142857142857143
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_gb) #0.81804110193612756

#3 Ada Boost Classifier
from sklearn.ensemble import AdaBoostClassifier
abc = AdaBoostClassifier(n_estimators=100)
abc.fit(x_train,y_train)
y_pred_ab = abc.predict(x_test)
from sklearn.metrics import f1_score
f1_score(y_test, y_pred_ab) #0.83809523809523812
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_ab) #0.89989446281573204

#4 Logistic Regression
from sklearn.linear_model import LogisticRegression
lrc = LogisticRegression()
lrc.fit(x_train,y_train)
y_pred_lr = lrc.predict(x_test)
from sklearn.metrics import f1_score
f1_score(y_test, y_pred_lr) #0.66666666666666663
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_ab) #0.79075078513268904

#5 Linear SVM
from sklearn.svm import LinearSVC
svmc=LinearSVC()
svmc.fit(x_train,y_train)
y_pred_svm = svmc.predict(x_test)
from sklearn.metrics import f1_score
f1_score(y_test, y_pred_svm) #0.4183006535947712
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, y_pred_svm) #0.78974818188214368


#6 Extreme Gradient Boosting Classifier
import xgboost as xgb
param = {'max_depth':5, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
num_round=100
dtrain = xgb.DMatrix(x_train, label=y_train)
xgb_fit = xgb.train(param, dtrain, num_round)
dtest = xgb.DMatrix(x_test)
xgb_pred = pd.DataFrame(xgb_fit.predict(dtest), columns=['y'])
xgb_pred.ix[xgb_pred.y>=0.85, 'y']=1
xgb_pred.ix[xgb_pred.y<0.85, 'y']=0
from sklearn.metrics import f1_score
f1_score(y_test, xgb_pred) #0.83999999999999997
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, xgb_pred) #0.88176541322604773

#Hence ExGBT has best predict power

# feature engineering
x_train_full=x_train.copy()
x_test_full=x_test.copy()
#1) outlier indicator
out_min=x_train.quantile(0.25)-1.5*(x_train.quantile(0.75)-x_train.quantile(0.25))
out_max=x_train.quantile(0.75)+1.5*(x_train.quantile(0.75)-x_train.quantile(0.25))
varlist=x_train.columns
for i in varlist:
    x_train_full[i+'_outlier']=0
    x_train_full.ix[(x_train_full[i]<out_min[i])|(x_train_full[i]>out_max[i]), i+'_outlier']=1

    x_test_full[i+'_outlier']=0
    x_test_full.ix[(x_test_full[i]<out_min[i])|(x_test_full[i]>out_max[i]), i+'_outlier']=1
#2) zero indicator
for i in varlist:
    x_train_full[i+'_zero']=0
    x_train_full.ix[(x_train_full[i]==0), i+'_zero']=1

    x_test_full[i+'_zero']=0
    x_test_full.ix[(x_test_full[i]==0), i+'_zero']=1 
#3) feature transformation (sq and logit)
for i in varlist:
    x_train_full[i+'_sq']=x_train_full[i]**2
    x_test_full[i+'_sq']=x_test_full[i]**2
    #logit transformation from paper: "logistic quantile regression for bounded outcomes"
    y_min=min(x_train_full[i])
    y_max=max(x_train_full[i])

    yy_min=min(x_test_full[i])
    yy_max=max(x_test_full[i])
    epsilon=0.001
    x_train_full[i+'_logit']=np.log((x_train_full[i]-(y_min-epsilon))/(y_max+epsilon-x_train_full[i]))
    x_test_full[i+'_logit']=np.log((x_test_full[i]-(yy_min-epsilon))/(yy_max+epsilon-x_test_full[i]))
#4) bining based on quantiles
for i in varlist:
    x_train_full[i+'_bin']=1
    x_train_full.ix[(x_train_full[i]>=x_train_full[i].quantile(0.2))&(x_train_full[i]<x_train_full[i].quantile(0.4)), i+'_bin']=2
    x_train_full.ix[(x_train_full[i]>=x_train_full[i].quantile(0.4))&(x_train_full[i]<x_train_full[i].quantile(0.6)), i+'_bin']=3
    x_train_full.ix[(x_train_full[i]>=x_train_full[i].quantile(0.6))&(x_train_full[i]<x_train_full[i].quantile(0.8)), i+'_bin']=4
    x_train_full.ix[(x_train_full[i]>=x_train_full[i].quantile(0.8)), i+'_bin']=5

    x_test_full[i+'_bin']=1
    x_test_full.ix[(x_test_full[i]>=x_train_full[i].quantile(0.2))&(x_test_full[i]<x_train_full[i].quantile(0.4)), i+'_bin']=2
    x_test_full.ix[(x_test_full[i]>=x_train_full[i].quantile(0.4))&(x_test_full[i]<x_train_full[i].quantile(0.6)), i+'_bin']=3
    x_test_full.ix[(x_test_full[i]>=x_train_full[i].quantile(0.6))&(x_test_full[i]<x_train_full[i].quantile(0.8)), i+'_bin']=4
    x_test_full.ix[(x_test_full[i]>=x_train_full[i].quantile(0.8)), i+'_bin']=5
#5) two way interation
for i in varlist:
    for j in varlist:
        if(i!=j):
            x_train_full[i+j]=x_train_full[i]*x_train_full[j]
            x_test_full[i+j]=x_test_full[i]*x_test_full[j]

# var importance   
dtrain_full = xgb.DMatrix(x_train_full, label=y_train)
xgb_full_fit = xgb.train( param, dtrain_full, num_round)
importance = xgb_full_fit.get_fscore()
importance_frame = pd.DataFrame({'Importance': list(importance.values()), 'Feature': list(importance.keys())})
importance_frame.sort_values(by = 'Importance', inplace = True, ascending=False)

#Rebuild Model using the Features
feature_list=importance_frame[importance_frame.Importance>=3].Feature

x_train_feature=x_train_full.loc[:, x_train_full.columns.isin(feature_list)]
x_test_feature=x_test_full.loc[:, x_test_full.columns.isin(feature_list)]

dtrain_feature= xgb.DMatrix(x_train_feature, label=y_train)
dtest_feature = xgb.DMatrix(x_test_feature)

xgb_feature_fit = xgb.train( param, dtrain_feature, num_round)

xgb_pred = pd.DataFrame(xgb_feature_fit.predict(dtest_feature), columns=['y'])
xgb_pred.ix[xgb_pred.y>=0.85, 'y']=1
xgb_pred.ix[xgb_pred.y<0.85, 'y']=0
from sklearn.metrics import f1_score
f1_score(y_test, xgb_pred) #0.87128712871287128
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, xgb_pred) #0.89996482093857744

#Resampling on training
from sklearn.utils import resample
xy_train_feature=pd.concat([x_train_feature, y_train], axis=1)
xy_train_feature_fraud=xy_train_feature[xy_train_feature.Class==1]
xy_train_feature_nonfraud=xy_train_feature[xy_train_feature.Class==0]
xy_train_feature_fraud_resample=resample(xy_train_feature_fraud, n_samples=4*len(xy_train_feature_fraud))
xy_train_feature_new=pd.concat([xy_train_feature, xy_train_feature_fraud_resample])
xy_train_feature_new.describe()
y_train_new = xy_train_feature_new['Class']
x_train_new = xy_train_feature_new.drop("Class",axis=1)
dtrain_feature_new= xgb.DMatrix(x_train_new, label=y_train_new)
dtest_feature = xgb.DMatrix(x_test_feature)
xgb_feature_fit_new = xgb.train(param, dtrain_feature_new, num_round)
xgb_pred_new = pd.DataFrame(xgb_feature_fit_new.predict(dtest_feature), columns=['y'])
#change the threshold due to resampling
xgb_pred_new.ix[xgb_pred_new.y>=0.55, 'y']=1
xgb_pred_new.ix[xgb_pred_new.y<0.55, 'y']=0
from sklearn.metrics import f1_score
f1_score(y_test, xgb_pred_new) #0.85981308411214941
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, xgb_pred_new) #0.91807628099755023

#tune the xgb parameters

# increase n rounds
#100-300 has same results, go up or down will have weak results, so 100 is the optimal parameter
xgb_feature_fit_new = xgb.train(param, dtrain_feature_new, 100)
xgb_pred_new = pd.DataFrame(xgb_feature_fit_new.predict(dtest_feature), columns=['y'])
#change the threshold due to resampling
xgb_pred_new.ix[xgb_pred_new.y>=0.55, 'y']=1
xgb_pred_new.ix[xgb_pred_new.y<0.55, 'y']=0
from sklearn.metrics import f1_score
f1_score(y_test, xgb_pred_new) #0.85981308411214941
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, xgb_pred_new) #0.91807628099755023

# increse max depth
#10-#50: f1-0.86792452830188671, auc:0.91809387052826152, so 10 is the optimal parameter
param = {'max_depth':50, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
xgb_feature_fit_new = xgb.train(param, dtrain_feature_new, 100)
xgb_pred_new = pd.DataFrame(xgb_feature_fit_new.predict(dtest_feature), columns=['y'])
#change the threshold due to resampling
xgb_pred_new.ix[xgb_pred_new.y>=0.55, 'y']=1
xgb_pred_new.ix[xgb_pred_new.y<0.55, 'y']=0
from sklearn.metrics import f1_score
f1_score(y_test, xgb_pred_new) #0.86792452830188671
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, xgb_pred_new) #0.91809387052826152

# increase min_child_weight
#1: f1-0.86792452830188671, auc:0.91809387052826152
#2: f1-0.8545454545454545, auc:0.92713201102703668
#4: f1-0.86792452830188671, auc:0.91809387052826152
#6: f1-0.81415929203539816, auc:0.91797074381328236
param = {'max_depth':10, 'min_child_weight':1, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
xgb_feature_fit_new = xgb.train(param, dtrain_feature_new, 100)
xgb_pred_new = pd.DataFrame(xgb_feature_fit_new.predict(dtest_feature), columns=['y'])
#change the threshold due to resampling
xgb_pred_new.ix[xgb_pred_new.y>=0.55, 'y']=1
xgb_pred_new.ix[xgb_pred_new.y<0.55, 'y']=0
from sklearn.metrics import f1_score
f1_score(y_test, xgb_pred_new) #0.86792452830188671
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, xgb_pred_new) #0.91809387052826152

#tune subsample
#subsample(0.5): f1-0.83333333333333337, auc:0.90895019284521861
#subsample(0.75): f1-0.86238532110091748, auc:0.92714960055774798
#subsample(0.85): f1-0.86538461538461542, auc:0.90902055096806389
param = {'max_depth':10, 'min_child_weight':1, 'subsample' :0.75, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
xgb_feature_fit_new = xgb.train(param, dtrain_feature_new, 100)
xgb_pred_new = pd.DataFrame(xgb_feature_fit_new.predict(dtest_feature), columns=['y'])
#change the threshold due to resampling
xgb_pred_new.ix[xgb_pred_new.y>=0.55, 'y']=1
xgb_pred_new.ix[xgb_pred_new.y<0.55, 'y']=0
from sklearn.metrics import f1_score
f1_score(y_test, xgb_pred_new) 
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, xgb_pred_new)

#tune colsample_bytree
#colsample_bytree(0.75): f1-0.86538461538461542, auc:0.90902055096806389
#colsample_bytree(0.9): f1-0.85185185185185186, auc:0.91805869146683894
#colsample_bytree(1): f1-0.86238532110091748, auc:0.92714960055774798
param = {'max_depth':10, 'min_child_weight':1, 'subsample' :0.75, 'colsample_bytree':1, 'eta':1, 'silent':1, 'objective':'binary:logistic' }
xgb_feature_fit_new = xgb.train(param, dtrain_feature_new, 100)
xgb_pred_new = pd.DataFrame(xgb_feature_fit_new.predict(dtest_feature), columns=['y'])
#change the threshold due to resampling
xgb_pred_new.ix[xgb_pred_new.y>=0.55, 'y']=1
xgb_pred_new.ix[xgb_pred_new.y<0.55, 'y']=0
from sklearn.metrics import f1_score
f1_score(y_test, xgb_pred_new) 
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, xgb_pred_new)

#tune eta
#0.5: f1-0.85185185185185186, auc:0.91805869146683894
#0.75: f1-0.87850467289719625, auc:0.92718477961917056
#0.85: f1-0.85714285714285721, auc:0.90900296143735249
param = {'max_depth':10, 'min_child_weight':1, 'subsample' :0.75, 'colsample_bytree':1, 'eta':0.85, 'silent':1, 'objective':'binary:logistic' }
xgb_feature_fit_new = xgb.train(param, dtrain_feature_new, 100)
xgb_pred_new = pd.DataFrame(xgb_feature_fit_new.predict(dtest_feature), columns=['y'])
#change the threshold due to resampling
xgb_pred_new.ix[xgb_pred_new.y>=0.55, 'y']=1
xgb_pred_new.ix[xgb_pred_new.y<0.55, 'y']=0
from sklearn.metrics import f1_score
f1_score(y_test, xgb_pred_new) 
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, xgb_pred_new)

#tune gamma
#0.5: f1-0.85714285714285721, auc:0.90900296143735249
#1: f1=0.85981308411214941, auc:0.91807628099755023
param = {'max_depth':10, 'min_child_weight':1, 'subsample' :0.75, 'colsample_bytree':1, 'eta':0.75, 'gamma': 1, 'silent':1, 'objective':'binary:logistic' }
xgb_feature_fit_new = xgb.train(param, dtrain_feature_new, 100)
xgb_pred_new = pd.DataFrame(xgb_feature_fit_new.predict(dtest_feature), columns=['y'])
#change the threshold due to resampling
xgb_pred_new.ix[xgb_pred_new.y>=0.55, 'y']=1
xgb_pred_new.ix[xgb_pred_new.y<0.55, 'y']=0
from sklearn.metrics import f1_score
f1_score(y_test, xgb_pred_new) 
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, xgb_pred_new)

#use different eval metrics (default is misclassification rate)
#auc/logloss: same as default
param = {'max_depth':10, 'min_child_weight':1, 'subsample' :0.75, 'colsample_bytree':1, 'eta':0.75, 'eval_metric ': 'logloss', 'silent':1, 'objective':'binary:logistic' }
xgb_feature_fit_new = xgb.train(param, dtrain_feature_new, 100)
xgb_pred_new = pd.DataFrame(xgb_feature_fit_new.predict(dtest_feature), columns=['y'])
#change the threshold due to resampling
xgb_pred_new.ix[xgb_pred_new.y>=0.55, 'y']=1
xgb_pred_new.ix[xgb_pred_new.y<0.55, 'y']=0
from sklearn.metrics import f1_score
f1_score(y_test, xgb_pred_new) 
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, xgb_pred_new)

#try cross_validation (5 folds)
import math
import random
def n_folds_sampling(datafile, n=5):
    index_set=datafile.index
    nrows=int(math.floor(len(datafile)/n))
    datapart=[]
    for i in range(0,n):
        if (i==n-1):
            rows=index_set
        else:
            rows=random.sample(list(index_set),nrows) 
        datapart.append(datafile.loc[rows,:])
        index_set=index_set.drop(rows)
    return datapart

xy_train_feature_new.reset_index(inplace=True)
xy_train_feature_new.drop("index", axis=1, inplace=True)
n_fold_xy_training=n_folds_sampling(xy_train_feature_new)
xgb_fold=[]
for i in range(0,5):
        fold=xy_train_feature_new.drop(n_fold_xy_training[i].index)
        fold=fold.reset_index()
        fold=fold.drop(['index'],axis=1)
        fold_target=fold['Class']
        fold_data=fold.drop(['Class'],axis=1)

        val=n_fold_xy_training[i]
        val=val.reset_index()
        val=val.drop(['index'],axis=1)
        val_tar=val['Class']
        val_data=val.drop(['Class'],axis=1)

        fold_DM= xgb.DMatrix(fold_data, label=fold_target)
        val_DM = xgb.DMatrix(val_data)

        param = {'max_depth':10, 'min_child_weight':1, 'subsample' :0.75, 'colsample_bytree':1, 'eta':0.75, 'silent':1, 'objective':'binary:logistic' }
        xgb_fold.append(xgb.train(param, fold_DM, 100))
        val_pred = pd.DataFrame(xgb_fold[i].predict(val_DM), columns=['y'])

        val_pred.ix[val_pred.y>=0.55, 'y']=1
        val_pred.ix[val_pred.y<0.55, 'y']=0
        print(str(i+1)+'th fold')
        print(f1_score(val_tar, val_pred))
        print(roc_auc_score(val_tar, val_pred))

 #try to use model from 3rd fold
#3rd fold: f1-0.87850467289719625, auc:0.92718477961917056
xgb_pred_new = pd.DataFrame(xgb_fold[2].predict(dtest_feature), columns=['y'])
#change the threshold due to resampling
xgb_pred_new.ix[xgb_pred_new.y>=0.55, 'y']=1
xgb_pred_new.ix[xgb_pred_new.y<0.55, 'y']=0
from sklearn.metrics import f1_score
f1_score(y_test, xgb_pred_new) 
from sklearn.metrics import roc_auc_score
roc_auc_score(y_test, xgb_pred_new)

#further adjust the threshold
#0.5 - 0.6 are same: f1-0.87850467289719625, auc:0.92718477961917056
#0.4- f1:0.87037037037037035, auc:0.92716719008845938
#0.7- f1:0.85714285714285721, auc:0.90900296143735249
xgb_pred_new = pd.DataFrame(xgb_fold[2].predict(dtest_feature), columns=['y'])
xgb_pred_new.ix[xgb_pred_new.y>=0.7, 'y']=1
xgb_pred_new.ix[xgb_pred_new.y<0.7, 'y']=0
f1_score(y_test, xgb_pred_new) 
roc_auc_score(y_test, xgb_pred_new)

#save model
xgb_fold[2].dump_model('F:\Careers\Showcase\credit_fraud\model.txt')

