# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 20:12:54 2022

@author: juanm
"""

#%% Libraries 
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import GradientBoostingRegressor  #GBM algorithm
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
import scipy.io

#%% Functions
    # Importing thscsae libraries
def RMSE(y_pred_lr,y):
  return np.sqrt(np.sum(((y_pred_lr-y)**2/len(y_pred_lr))))
#%% Select parameters

sig_n='Radial' # Signal location : Radial , Brachial and Digital
wav='BP'       # Signal type : BP or PPG
SNR="no"        # Noise level : PPG: 65, 45 and 30  and BP: 20, 10, 5  
models='Energy'  # Feature method: Stat, SCSA, Energy
n_seed=0       # Number of seed used
num_feature=5  # Number of features to be used


#%% Load data

filen='./Results/ML/'+models+'_'+wav+'_'+sig_n+'_SNR='+SNR 

indx_fet=pd.read_csv(filen+'/best_feature_indx.csv',header=None)
rank_f=indx_fet.to_numpy()
features=rank_f[:,0:num_feature].reshape(-1,)

fet_train=scipy.io.loadmat(filen+'/features_train_all.mat')
y_train=scipy.io.loadmat(filen+'/y_train.mat')
y_train_t=y_train['y_train']

datos_train=fet_train['feature_train']
X_train=datos_train[:,:]
PWV_cf_train=y_train_t[:]


# Test data
fet_test=scipy.io.loadmat(filen+'/features_test_all.mat')
datos_test=fet_test['feature_test']
y_test=scipy.io.loadmat(filen+'/y_test.mat')
y_test_t=y_test['y_test']    
X_test=datos_test[:,:]
PWV_cf_test=y_test_t[:]


# Data standarization

sc = StandardScaler()
sc.fit(X_train)
X_train=sc.transform(X_train)
X_test=sc.transform(X_test)

y_test=PWV_cf_test.reshape(-1,)

#pickle.dump(sc, open("scaler_LR.pkl", "wb"))



#%% Random Forest Training and testing


print('Strart classification using RF')

# Number of desicion trees
n_estimators = [int(x) for x in np.linspace(start = 50, stop = 200, num = 100)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt','log2',1,2,5]
# Minimum number of samples required to split a node
min_samples_split = [2,5, 10,15,20]
# Minimum number of samples required at each leaf node
min_samples_leaf = [2, 5, 10,15,20]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid_rf = {'n_estimators': n_estimators,
            'max_features': max_features,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            'bootstrap': bootstrap}
rf = RandomForestRegressor()
rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid_rf, n_iter = 10, cv = 5, verbose=2, random_state=42, n_jobs = -1)

# Fit the model
rf_random.fit(X_train,PWV_cf_train.reshape(-1,))
param_rf=rf_random.best_params_

f = open(filen+"/RF_Hparams.txt","w")
f.write( str(param_rf) )
f.close()
# Test model noisy free
y_pred_rf =rf_random.predict(X_test)
RMSE_RF=RMSE(y_pred_rf,y_test)

# Estimated vs measured
m, b = np.polyfit(y_pred_rf, y_test,1)
X = sm.add_constant(y_pred_rf)
est = sm.OLS(y_test, X)
est2 = est.fit()
r_squared = est2.rsquared


RMSE_rf=[RMSE_RF]
R2_rf=[r_squared]
res_rf=[[RMSE_rf,R2_rf]]
RF_result= pd.DataFrame(res_rf, columns = ['RMSE','R2'])


savedata = [y_test, y_pred_rf]
df_savedata = pd.DataFrame(savedata)

y_pred_rf=list(y_pred_rf)
y_test=list(y_test)



#pickle.dump(rf_random, open(".\model_RF.pkl", "wb"))

#%% Gradient Boost Regression Training and testing

print('Strart classification using GB')

# Loss function
loss = ['absolute_error', 'squared_error', 'huber']
# Learning rate
learning_rate = [0.01, 0.02, 0.05, 0.1]
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(2, 10, num = 5)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [40, 60, 80]
# Minimum number of samples required at each leaf node
min_samples_leaf = [20, 30, 40]

# Create the random grid
random_grid = {'loss': loss, 
            'learning_rate': learning_rate,
            'n_estimators': n_estimators,
            'max_features': max_features,
            'max_depth': max_depth,
            'min_samples_split': min_samples_split,
            'min_samples_leaf': min_samples_leaf,
            }
# First create the base model to tune
gb = GradientBoostingRegressor()
GBR = RandomizedSearchCV(estimator = gb, param_distributions = random_grid, n_iter = 10, cv = 5, random_state=42, n_jobs = -1)
GBR.fit(X_train,PWV_cf_train.reshape(-1,))


param_gbr=GBR.best_params_

f = open(filen+"/GBR_Hparams.txt","w")
f.write( str(param_gbr) )
f.close()

# Test model
y_pred_gbr =GBR.predict(X_test)

RMSE_gbr=RMSE(y_pred_gbr,y_test)

#r_squared = r2_score(y_test, y_pred_gbr)

m, b = np.polyfit(y_pred_gbr, y_test,1)
X = sm.add_constant(y_pred_gbr)
est = sm.OLS(y_test, X)
est2 = est.fit()
p_value =  est2.pvalues[1]
r_squared = est2.rsquared

RMSE_gbr=[RMSE_gbr]
R2_gbr=[r_squared]
res_gbr=[[RMSE_gbr,R2_gbr]]
gbr_result= pd.DataFrame(res_gbr, columns = ['RMSE','R2'])


savedata = [y_test, y_pred_gbr]
df_savedata = pd.DataFrame(savedata)

y_pred_gbr=list(y_pred_gbr)
y_test=list(y_test)


#pickle.dump(GBR, open("model_GB.pkl", "wb"))

#%% MLP Training and testing

print('Strart classification using MLP')

parameter_space = {
    'hidden_layer_sizes': [(10,30,10),(20,), (15,11)],
    'activation':['tanh', 'relu'],
    'solver':['sgd', 'adam','lbfgs'],
    'alpha': [0.0001, 0.001 ,0.01,0.1],
    'learning_rate':['constant','adaptive'],
}

clf=MLPRegressor(max_iter=400)
model_mlp =RandomizedSearchCV(estimator = clf, param_distributions = parameter_space, n_iter = 10, cv = 5, random_state=42, n_jobs = -1)

model_mlp.fit(X_train,PWV_cf_train.reshape(-1,))


param_mlp=model_mlp.best_params_

f = open(filen+"/MLP_Hparams.txt","w")
f.write( str(param_mlp) )
f.close()

y_pred_mlp =model_mlp.predict(X_test)

RMSE_mlp=RMSE(y_pred_mlp,y_test)

m, b = np.polyfit(y_pred_mlp, y_test,1)
X = sm.add_constant(y_pred_mlp)
est = sm.OLS(y_test, X)
est2 = est.fit()
p_value =  est2.pvalues[1]
r_squared = est2.rsquared

RMSE_mlp=[RMSE_mlp]
R2_mlp=[r_squared]
res_mlp=[[RMSE_mlp,R2_mlp]]
mlp_result= pd.DataFrame(res_mlp, columns = ['RMSE','R2'])




#pickle.dump(model_mlp, open("model_MLP.pkl", "wb"))

#%% Linear regression Training and testing

print("Strart classification using LR")

# Set hyper-parameter space
hyper_params = [{"fit_intercept":[True,False]}]

# Create linear regression model 
lm = LinearRegression()
# Create RandomSearchCV() with 5-fold cross-validation
model_cv = RandomizedSearchCV(estimator = lm,param_distributions=hyper_params,n_iter=2,cv = 5,random_state=42)  


# Fit the model
model_cv.fit(X_train,PWV_cf_train.reshape(-1,))

param_lr=model_cv.best_params_

f = open(filen+"/LR_Hparams.txt","w")
f.write( str(param_lr) )
f.close()

# Test model
y_pred_lr =model_cv.predict(X_test)

RMSE_LR=RMSE(y_pred_lr,y_test)

#r_squared = r2_score(y_test, y_pred_lr)

m, b = np.polyfit(y_pred_lr, y_test,1)
X = sm.add_constant(y_pred_lr)
est = sm.OLS(y_test, X)
est2 = est.fit()
p_value =  est2.pvalues[1]
r_squared = est2.rsquared

RMSE_lr=[RMSE_LR]
R2_lr=[r_squared]
res_lr=[[RMSE_lr,R2_lr]]
lr_result= pd.DataFrame(res_lr, columns = ['RMSE','R2'])


#pickle.dump(model_cv, open("model_LR.pkl", "wb"))


#%%  SVR Training and testing

print('Strart classification using SVR')
#CV using random search
# C parameter
C= [int(x) for x in np.linspace(100, 400, num = 5)]
kernel=['linear', 'rbf', 'sigmoid']
gamma=['scale','auto']

# Create the random grid
random_grid = {'C':C, 
               'kernel':kernel
             
           }
print(random_grid)
# Use the random grid to search for best hyperparameters
# First create the base model to tune
svr= SVR()

SVR_rs = RandomizedSearchCV(estimator = svr, param_distributions = random_grid, n_iter = 10, cv = 5, random_state=42, n_jobs = -1)



# predict with the best parameters from random search

SVR_rs.fit(X_train,PWV_cf_train.reshape(-1,))


param_svr=SVR_rs.best_params_

f = open(filen+"/SVR_Hparams.txt","w")
f.write( str(param_svr) )
f.close()

y_pred_svr= SVR_rs.predict(X_test)


RMSE_svr=RMSE(y_pred_svr,y_test)

#r_squared = r2_score(y_test, y_pred_svr)

m, b = np.polyfit(y_pred_svr, y_test,1)
X = sm.add_constant(y_pred_svr)
est = sm.OLS(y_test, X)
est2 = est.fit()
p_value =  est2.pvalues[1]
r_squared = est2.rsquared

RMSE_svr=[RMSE_svr]
R2_svr=[r_squared]
res_svr=[[RMSE_svr,R2_svr]]
svr_result= pd.DataFrame(res_svr, columns = ['RMSE','R2'])


#pickle.dump(SVR_rs, open("model_SVR.pkl", "wb"))




print('Ending classification using SVR')


#%% Save Data

RMSE=np.asarray([RMSE_rf,RMSE_gbr,RMSE_mlp,RMSE_lr,RMSE_svr])
R2=np.asarray([R2_rf,R2_gbr,R2_mlp,R2_lr,R2_svr])

y_pred=[list(y_pred_rf),list(y_pred_gbr),list(y_pred_mlp),list(y_pred_lr),list(y_pred_svr)]
y_test=list(y_test)

data=np.concatenate((RMSE,R2),axis=1)
df=pd.DataFrame(data,columns=["RMSE","R2"],index=['RF','GBR','MLP','LR','SVR'])
df.to_csv(filen+'/Result_hypertunnig_SNR='+SNR+'.csv')
