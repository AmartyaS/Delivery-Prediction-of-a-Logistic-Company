# -*- coding: utf-8 -*-
"""
Created on Thu Aug 19 20:44:08 2021

@author: ASUS
"""
# Importing the Necessary Libraries
import pickle
import optuna
import xgboost
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf
from sklearn import metrics
from tensorflow import keras
import matplotlib.pyplot as plt
from tpot import TPOTClassifier
from sklearn.metrics import roc_curve
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import classification_report
from hyperopt import hp,fmin,tpe,STATUS_OK,Trials
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import RandomizedSearchCV

# Loading up the dataset
file=pd.read_csv(r"D:\Python Project\EXCELR Project - Warehouse\shipments.csv")
data=file.copy()
data.isna().sum()
data.describe()
data.columns
data.Customer_rating.value_counts()
sns.barplot(x=data.Customer_rating,y=data.Customer_rating.value_counts())
for column in data:
    print(f'{column} : {data[column].unique()}')
data.drop(['ID'],axis=1,inplace=True)
data.info()

"""
data['Customer_rating']=data['Customer_rating'].map({1:'Worst',
                                                     2:'Bad',3:'Medium',
                                                     4:'Good',5:'Best'})
"""

# Data Visualisation and Data Manipulation
plt.hist(data.Customer_rating)
sns.pairplot(data)
data['Warehouse_block'].unique()
data['Mode_of_Shipment'].unique()
data['Product_importance'].unique()
data['Prior_purchases'].unique()
sns.boxplot(y=data['Customer_care_calls'])
sns.boxplot(y=data['Cost_of_the_Product'])
sns.boxplot(y=data['Prior_purchases'])  # Outliers Detected
sns.boxplot(y=data['Discount_offered']) # Outliers Detected
sns.boxplot(y=data['Weight_in_gms'])
sns.distplot(data['Discount_offered'])
sns.distplot(data['Prior_purchases'])
data['Weight_in_gms'].describe()
data['Prior_purchases'].describe()
data['Discount_offered'].describe()
data["Gender"].replace({'F':1,'M':0},inplace=True)
#sample=data.sample(1000)
#data=data.drop([data.index[i] for i in data.index if i in sample.index])
data=pd.get_dummies(data,columns=["Warehouse_block","Mode_of_Shipment",
                                  "Product_importance"
                                  ],prefix=["WB","MS","PI"])
'''
sample=pd.get_dummies(sample,columns=["Warehouse_block","Mode_of_Shipment",
                                  "Product_importance"
                                  ],prefix=["WB","MS","PI"])

'''
data.info()

# Correlation Plot
corrmat=data.corr()
cor_feat=corrmat.index
plt.figure(figsize=(20,20))
sns.heatmap(data[cor_feat].corr(),annot=True,cmap="RdYlGn")

# data.drop(['ID','G_F','PI_low','CR_Worst','MS_Road','WB_A'],axis=1,inplace=True)
# Outlier Visualisation of Discount_offered
figure=data.Discount_offered.hist(bins=50)
figure.set_title('Discount_offered')
figure.set_xlabel('Discount Offered')
figure.set_ylabel('Shipments Range')

# Outlier Visualisation of Prior Purchases
figure=data.Prior_purchases.hist(bins=50)
figure.set_title('Prior Purchases')
figure.set_xlabel('Prior_purchases')
figure.set_ylabel('Shipments Range')

##############################################################################
# Outlier Management of Prior Purchases
upper_bound=data["Prior_purchases"].mean()+(3*data["Prior_purchases"].std())
lower_bound=data["Prior_purchases"].mean()-(3*data["Prior_purchases"].std())
data.loc[data["Prior_purchases"]>=8,"Prior_purchases"].value_counts()
data.loc[data["Prior_purchases"]>=8,"Prior_purchases"]=8   # 306 Values of Prior Purchases >=8
# Outlier Management of Discount_offered                   
IQR=data.Discount_offered.quantile(0.75)-data.Discount_offered.quantile(0.25)
lower_bridge=data["Discount_offered"].quantile(0.25)-(IQR*3)
upper_bridge=data["Discount_offered"].quantile(0.75)+(IQR*3)
data.loc[data["Discount_offered"]>=28,"Discount_offered"] # 1839 Values of Discount_offered >= 28
data.loc[data["Discount_offered"]>=28,"Discount_offered"]=28
##############################################################################

### Scale your Dataset only if you are using Logistic Regression or ANN ####
# Scaling the dataset
scaler=MinMaxScaler()
columns=["Customer_care_calls","Customer_rating","Cost_of_the_Product",
         "Prior_purchases","Discount_offered","Weight_in_gms"]
data[columns]=scaler.fit_transform(data[columns])

# Splitting dataset for train and test sample
x=data.drop("Reached.on.Time_Y.N",axis=1)
y=data["Reached.on.Time_Y.N"]
y.value_counts()


# Balancing through SMOTEENN
sm=SMOTEENN()
x_sm,y_sm=sm.fit_resample(x,y)
y_sm.value_counts()


'''
# Balancing the imbalanced data
smote=SMOTE(sampling_strategy='minority')
x_sm, y_sm=smote.fit_resample(x,y)
y_sm.value_counts()
'''
# Feature Importance
mod=ExtraTreesClassifier()
mod.fit(x_sm,y_sm)
print(mod.feature_importances_)

# Plot graph of Important Features
feat_imp=pd.Series(mod.feature_importances_,index=x.columns)
plt.figure(figsize=(20,20))
feat_imp.nlargest(23).plot(kind='barh')
plt.show()

'''
# When we require only the important columns    
# Selecting Important Features
columns=["Customer_rating","Cost_of_the_Product","Discount_offered",
         "Gender","Weight_in_gms","WB_F","MS_Ship","PI_low"]
x_sm=x_sm[columns]
'''

# Splitting the Dataset into training and testing data
x_train,x_test,y_train,y_test=train_test_split(x_sm, y_sm, random_state=33, test_size=0.2)
y_train.value_counts()


# Random Forest Model
rf_model1=RandomForestClassifier(n_estimators=100)
rf_model1.fit(x_train,y_train)
pred1=rf_model1.predict(x_test)
sns.distplot(y_test-pred1)
#Checking the accuracy of the model
acc1=(metrics.accuracy_score(y_test,pred1)*100)
print("Accuracy : ",acc1)

confusion_matrix(y_test,pred1)
pd.crosstab(y_test,pred1)
print(classification_report(y_test,pred1))
print(roc_auc_score(y_test,pred1))


################   Hyper-Parameter Tuning   ###################

# RandomizedSearchCV Model
n_estimators=[int(i) for i in np.linspace(start=100,stop=2000,num=10)]
max_features=['auto','sqrt','log2']
max_depth=[int(i) for i in np.linspace(start=5,stop=1000,num=10)]
min_samples_split=[int(i) for i in np.linspace(start=2, stop=100,num=5)]
min_samples_leaf=[int(i) for i in np.linspace(start=2, stop=100,num=5)]
# Random-Grid Parameter Dictionary
random_grid={
    'n_estimators':n_estimators,
    'max_features':max_features,
    'max_depth':max_depth,
    'min_samples_split':min_samples_split,
    'min_samples_leaf':min_samples_leaf,
    'criterion':['entropy','gini']
    }
# Randomized SearchCV Model
rf=RandomForestClassifier()
rf_model2=RandomizedSearchCV(estimator=rf,param_distributions=random_grid,
                             n_iter=200,cv=3,verbose=2,n_jobs=-1,random_state=42)
rf_model2.fit(x_train,y_train)
pred2=rf_model2.predict(x_test)
#Checking the accuracy of the model
acc2=(metrics.accuracy_score(y_test,pred2)*100)
print(confusion_matrix(y_test,pred2))
print("Accuracy : ",acc2)
print(classification_report(y_test,pred2))
rf_model2.best_estimator_

# GridSearchCV Parameter List Based on RandomSearchCV's Best Parameters
n_estimators1=[int(i) for i in np.linspace(start=rf_model2.best_params_['n_estimators']-10,
                                          stop=rf_model2.best_params_['n_estimators']+10, num=5)]
max_features1=[rf_model2.best_params_['max_features']]
max_depth1=[int(i) for i in np.linspace(start=rf_model2.best_params_['max_depth']-5,
                                        stop=rf_model2.best_params_['max_depth']+5,num=5)]
min_samples_split1=[int(i) for i in np.linspace(start=rf_model2.best_params_['min_samples_split']-5, 
                                                stop=rf_model2.best_params_['min_samples_split']+5,num=5)]
min_samples_leaf1=[int(i) for i in np.linspace(start=rf_model2.best_params_['min_samples_leaf'], 
                                               stop=rf_model2.best_params_['min_samples_leaf']+5,num=5)]
# GridSearch Parameter Dictionary
grid={'criterion' : [rf_model2.best_params_['criterion']],
      'n_estimators':n_estimators1,
      'max_features':max_features1,
      'max_depth': max_depth1,
      'min_samples_split':min_samples_split1,
      'min_samples_leaf':min_samples_leaf1
             }
# GridSearch Model
rf2=RandomForestClassifier()
rf_model3=GridSearchCV(estimator=rf2,param_grid=grid,cv=3,verbose=2,n_jobs=-1)
rf_model3.fit(x_train,y_train)
best_grid=rf_model3.best_estimator_
pred3=best_grid.predict(x_test)
#Checking the accuracy of the model
acc3=(metrics.accuracy_score(y_test,pred3)*100)
print(confusion_matrix(y_test,pred3))
print("Accuracy : ",acc3)
print(classification_report(y_test,pred3))

# Genetic Algorithm Based on RandomizedSearchCV's Parameter Dictionary
tpot_rf=TPOTClassifier(generations=30,population_size=100,offspring_size=50,
                       verbosity=2,early_stop=12,
                       config_dict={'sklearn.ensemble.RandomForestClassifier' : random_grid},
                       cv=5,n_jobs=-1,scoring='accuracy')
tpot_rf.fit(x_train,y_train)
pred4=tpot_rf.predict(x_test)
acc4=tpot_rf.score(x_test,y_test)
print(confusion_matrix(y_test,pred4))
print("Accuracy : ",acc4)
print(classification_report(y_test,pred4))

# Automated Hyperparameter Tuning
#      Bayesian Optimization
n_band=list(range(100,2000,35))
space={
       'criterion' : hp.choice('criterion',['entropy','gini']),
       'max_depth' : hp.quniform('max_depth',10,1500,35),
       'max_features' : hp.choice('max_features',['auto','sqrt','log2']),
       'min_samples_leaf' : hp.uniform('min_samples_leaf',0,0.5),
       'min_samples_split' : hp.uniform('min_samples_split',0,1),
       'n_estimators' : hp.choice('n_estimators', n_band)
       }
def objective(space):
    model=RandomForestClassifier(criterion=space['criterion'],
                                 max_depth=space['max_depth'],
                                 max_features=space['max_features'],
                                 min_samples_leaf=space['min_samples_leaf'],
                                 min_samples_split=space['min_samples_split'],
                                 n_estimators=space['n_estimators'])
    accuracy=cross_val_score(model,x_train,y_train,cv=5).mean()
    return {'loss': -accuracy,'status': STATUS_OK}
trials=Trials()
best=fmin(fn = objective, 
          space = space, 
          algo = tpe.suggest,
          max_evals = 80,
          trials = trials)
# Checking the best parameters
best 
# Indicating the positions
crit = {0:'entropy',1:'gini'}
feat = {0:'auto',1:'sqrt',2:'log2'}
# Final Model
trained_model=RandomForestClassifier(criterion=crit[best['criterion']],
                                     max_depth=best['max_depth'],
                                     max_features=feat[best['max_features']],
                                     min_samples_leaf=best['min_samples_leaf'],
                                     min_samples_split=best['min_samples_split'],
                                     n_estimators=n_band[best['n_estimators']])
trained_model.fit(x_train,y_train)
pred5=trained_model.predict(x_test)
acc5=metrics.accuracy_score(y_test,pred5)*100
print(confusion_matrix(y_test,pred5))
print("Accuracy : ",acc5)
print(classification_report(y_test,pred5))

# OPTUNA
def obj(trial):
    criterion =trial.suggest_categorical('criterion',['entropy','gini'])
    max_depth = int(trial.suggest_float('max_depth',10,1500,log=True))
    max_features = trial.suggest_categorical('max_features',['auto','sqrt','log2'])
    min_samples_leaf = trial.suggest_float('min_samples_leaf',0,0.5)
    min_samples_split = trial.suggest_float('min_samples_split',0,1)
    n_estimators = trial.suggest_int('n_estimators',10,2000)
    model1=RandomForestClassifier(n_estimators=n_estimators,
                                  min_samples_leaf=min_samples_leaf,
                                  min_samples_split=min_samples_split,
                                  max_features=max_features,
                                  max_depth=max_depth,
                                  criterion=criterion)
    accuracy=cross_val_score(model1,x_train,y_train,cv=3,n_jobs=-1).mean()
    return accuracy
study=optuna.create_study(direction='maximize')
study.optimize(obj,n_trials=300)
trial=study.best_trial
print("Accuracy : ", trial.value)
print("Best Hyperparameters : ",trial.params)
trial.params['criterion']
# Final Model
trained_model2=RandomForestClassifier(criterion=trial.params['criterion'],
                                     max_depth=trial.params['max_depth'],
                                     max_features=trial.params['max_features'],
                                     min_samples_leaf=trial.params['min_samples_leaf'],
                                     min_samples_split=trial.params['min_samples_split'],
                                     n_estimators=trial.params['n_estimators'])
trained_model2.fit(x_train,y_train)
pred6=trained_model2.predict(x_test)
acc6=metrics.accuracy_score(y_test,pred6)*100

print(confusion_matrix(y_test,pred6))
print("Accuracy : ",acc6)
print(classification_report(y_test,pred6))

###############################################################################
###############################################################################
#             XG-BOOST 
xmodel=xgboost.XGBClassifier()
xmodel.fit(x_train,y_train)
# Prediction
xpred=xmodel.predict(x_test)
xacc=metrics.accuracy_score(y_test,xpred)*100
print("Accuracy :",xacc)
print(classification_report(y_test,xpred))
print(confusion_matrix(y_test,xpred))
pd.crosstab(y_test,xpred)

# Hyper-Tuning XGBOOST
#   RandomizedSearchCV
n_estimators = [int(i) for i in np.linspace(10,1500,20)]
max_depth = [int(i) for i in np.linspace(10,1000,20)]
min_child_weight =[float(i) for i in np.linspace(0.01,0.5,10)]
gamma = [float(i) for i in np.linspace(0.0,0.4,10)]
learning_rate = [float(i) for i in np.linspace(0.005,0.3,20)]
subsample = [float(i) for i in np.linspace(0.01,1,10)]
colsample_bylevel = [float(i) for i in np.linspace(0.1,1.0,10)]
colsample_bytree = [float(i) for i in np.linspace(0.1,1.0,10)]
random_param={
    'n_estimators' : n_estimators,
    'max_depth' : max_depth,
    'min_child_weight' : min_child_weight,
    'gamma' : gamma,
    'learning_rate': learning_rate,
    'subsample' : subsample,
    'colsample_bylevel' : colsample_bylevel,
    'colsample_bytree' : colsample_bytree
    }
xm=xgboost.XGBClassifier()
xrandom=RandomizedSearchCV(estimator=xm,param_distributions=random_param,cv=5,
                           verbose=3,n_jobs=-1,n_iter=200,scoring='roc_auc')
xrandom.fit(x_train,y_train)
# Best Parameters and Estimators of RandomizedSearchCV
xrandom.best_estimator_
xrandom.best_params_

# Final Model
xranmod=xgboost.XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.2,
              colsample_bynode=1, colsample_bytree=0.2,
              gamma=0.08888888888888889, gpu_id=-1, importance_type='gain',
              interaction_constraints='', learning_rate=0.020526315789473684,
              max_delta_step=0, max_depth=322, min_child_weight=0.01,
               monotone_constraints='()', n_estimators=402,
              n_jobs=12, num_parallel_tree=1, random_state=0, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=1, subsample=0.56,
              tree_method='exact', validate_parameters=1, verbosity=None)
xranmod.fit(x_train,y_train)
score=cross_val_score(xranmod,x_train,y_train,cv=5).mean()
# Predictions
x_rs_pred=xranmod.predict(x_test)
print("Accuracy :",score)
print(confusion_matrix(y_test, x_rs_pred))
print(classification_report(y_test, x_rs_pred))
xrandom.best_params_['n_estimators']
# Grid Search
n_estimators = [int(i) for i in np.linspace(start=xrandom.best_params_['n_estimators']-50,stop=xrandom.best_params_['n_estimators']+50,num=30)]
max_depth = [int(i) for i in np.linspace(start=xrandom.best_params_['max_depth']-20,stop=xrandom.best_params_['max_depth']+20,num=10)]
min_child_weight =[float(i) for i in np.linspace(start=xrandom.best_params_['min_child_weight']-0.005,stop=xrandom.best_params_['min_child_weight']+0.005,num=5)]
gamma = [float(i) for i in np.linspace(start=xrandom.best_params_['gamma']-0.05,stop=xrandom.best_params_['gamma']+0.05,num=5)]
learning_rate = [float(i) for i in np.linspace(start=xrandom.best_params_['learning_rate']-0.005,stop=xrandom.best_params_['learning_rate']+0.005,num=5)]
subsample = [float(i) for i in np.linspace(start=xrandom.best_params_['subsample']-0.005,stop=xrandom.best_params_['subsample']+0.005,num=5)]
colsample_bylevel = [float(i) for i in np.linspace(start=xrandom.best_params_['colsample_bylevel']-0.05,stop=xrandom.best_params_['colsample_bylevel']+0.05,num=5)]
colsample_bytree = [float(i) for i in np.linspace(start=xrandom.best_params_['colsample_bytree']-0.05,stop=xrandom.best_params_['colsample_bytree']+0.05,num=5)]
grid={
    'n_estimators' : n_estimators,
    'max_depth' : max_depth,
    'min_child_weight' : min_child_weight,
    'gamma' : gamma,
    'learning_rate': learning_rate,
    'subsample' : subsample,
    'colsample_bylevel' : colsample_bylevel,
    'colsample_bytree' : colsample_bytree
    }
xgs_model=GridSearchCV(estimator=xm,param_grid=grid,cv=2,verbose=2,n_jobs=-1)
xgs_model.fit(x_train,y_train)
xgs_model.best_params_

# Final Model
xgs_mod=xgboost.XGBClassifier()
xgs_mod.fit(x_train,y_train)
# Predictions
g_pred=xgs_mod.predict(x_test)
gacc=metrics.accuracy_score(y_test,g_pred)*100
print("Accuracy : ",gacc)
print(confusion_matrix(y_test, g_pred))
print(classification_report(y_test,g_pred)) 

##############################################################################
##############################################################################

# Artificial Neural Network
model=tf.keras.Sequential([
    tf.keras.layers.Dense(18,input_shape=(18,),activation='relu'),
    tf.keras.layers.Dense(12,activation='relu'),
    tf.keras.layers.Dense(6,activation='relu'),
    tf.keras.layers.Dense(1,activation='sigmoid')])
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train,epochs=500)
model.evaluate(x_test, y_test)
yp = model.predict(x_test)
y_pred = []
for element in yp:
    if element > 0.5:
        y_pred.append(1)
    else:
        y_pred.append(0)
y_pred=pd.DataFrame(y_pred,columns=["Output"])
n_acc=metrics.accuracy_score(y_test,y_pred)*100
print("Accuracy : ",n_acc)
print(classification_report(y_test,y_pred))
confusion_matrix(y_test,y_pred)
y_pred.value_counts()
print(tf.__version__)


##############################################################################
##############################################################################

# Logistic Regression
lr=LogisticRegression()
lr_model=lr.fit(x_train,y_train)
lr_pred=lr_model.predict(x_test)
lr_acc=(metrics.accuracy_score(y_test,lr_pred))*100
fpr,tpr,thresholds=roc_curve(y_test,lr_model.predict_proba(x_test)[:,1])
print("Accuracy :",lr_acc)
print("ROC_AUC_Score :",(roc_auc_score(y_test,lr_pred)*100))
print("Classification Report :\n",classification_report(y_test,lr_pred))
print(" Cross-Tabulation Report :\n",pd.crosstab(y_test,lr_pred))
auc=roc_auc_score(y_test, lr_pred)
plt.plot(fpr,tpr,color='orange',label='ROC Curve (area= %0.2f)'%auc)
plt.plot([0,1],[0,1],color='navy',linestyle='--')
plt.title("ROC Curve")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.legend(loc='lower right')

##############################################################################
##############################################################################
# Testing the model
sample["Reached.on.Time_Y.N"].value_counts()

# x and y variable
check_x=sample.drop("Reached.on.Time_Y.N",axis=1)
check_y=sample["Reached.on.Time_Y.N"]
smote=SMOTEENN()
cx_sm, cy_sm=smote.fit_resample(check_x,check_y)

# Predict
c_pred=rf_model1.predict(check_x)
c_pred=rf_model1.predict(cx_sm)
''' FOR NEURAL Network
c_p = []
for element in c_pred:
    if element > 0.5:
        c_p.append(1)
    else:
        c_p.append(0)'''
# Validation
c_acc=metrics.accuracy_score(cy_sm,c_pred)*100
c_acc=metrics.accuracy_score(check_y,c_pred)*100
print("Accuracy :",c_acc)
print(classification_report(check_y,c_pred))
pd.crosstab(check_y,c_pred)


print(classification_report(cy_sm,c_pred))
pd.crosstab(cy_sm,c_pred)

'''
    Accuracy Names :
        Random Forest Basic                 : acc1    - Accuracy :  95.26864474739375
        Random Forest RandomizedSearchCV    : acc2    - Accuracy :  95.37902097902097                    
        Random Forest GridSearchCV          : acc3    - Accuracy :  95.27902097902097                        
        Random Forest TPOTClassifier        : acc4    - Accuracy :  95.32902097902097                   
        Random Forest Bayesian Optimization : acc5    - Accuracy :  94.30633520449078                 
        Random Forest OPTUNA                : acc6    - Accuracy :  95.34883720930233
        XGBoost                             : xacc    - Accuracy :  95.34883720930233
        Logistic Regression                 : lr_acc  - Accuracy :  94.23076923076923
        Artificial Neural Network           : n_acc   - Accuracy :  94.49300699300699
'''

##############################################################################
#   Final Model
# Random Forest Model
rf_model1=RandomForestClassifier(n_estimators=100)
rf_model1.fit(x_train,y_train)
pred1=rf_model1.predict(x_test)
sns.distplot(y_test-pred1)
#Checking the accuracy of the model
acc1=(metrics.accuracy_score(y_test,pred1)*100)
print("Accuracy : ",acc1)
##############################################################################

# Creating a pickle file 
pkl=open('Warehouse_rf_base.pkl','wb')
# Dump the RandomForest Base Model in it
pickle.dump(rf_model1,pkl)
x_train.columns 
