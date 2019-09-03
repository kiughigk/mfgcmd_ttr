"""
Description : feature selection
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from scipy import stats

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

import xgboost as xgb

import copy

import sys
sys.path.append('C:/Users/GKiuc339340273/skunk/dstools')
from library import runXgb_opti as rnx
from library import runRf_opti as rnr


def feature_selection (df, tCol, featureList, ml_model, **params_dict):

    reducRate = 0.1
    THRESH_MSE = 0.1
    MAX_TURN  = 50
    MIN_FEATURE_NUM = 3
    tdata = df[tCol].values
    
    feature_list_tmp = copy.deepcopy(featureList)

    dfImp = pd.DataFrame({})
    lMse = []
    lRsq = []
    for i in range(MAX_TURN):
        if (i > 0):
            thre = np.percentile(importances[importances > 0].values, 100*reducRate)
            feature_list_tmp = list(importances[importances > thre].index)
            if (len(feature_list_tmp) < MIN_FEATURE_NUM):
                break
        xdata = df.loc[:,feature_list_tmp].values
        if (ml_model == 'xgb'):
            ml_fs = rnx.runXgb_opti('reg', xdata, tdata, **params_dict)
        elif (ml_model == 'rf'):
            ml_fs = rnr.runRf_opti( 'reg', xdata, tdata, **params_dict)
        ml_fs.runCv()
        importances = pd.Series(ml_fs.reg_best.feature_importances_, index = feature_list_tmp)
        #dfImp = pd.concat([dfImp, importances.to_frame(name='round%d' % i)], axis=1, join='outer', sort=True)
        dfImp = pd.concat([dfImp, importances.to_frame(name='round%d' % i)], axis=1, join='outer', sort=False)
        lMse.append(ml_fs.mse_best)
        lRsq.append(ml_fs.rsq_best)
        print('step%d' % i, len(feature_list_tmp), lMse[-1], lRsq[-1])
        print('\n')
        
        #check if break this roop
        if (len(lMse) > 1):
            if ( lMse[-2] > 0 ):
                if ( (lMse[-1] - lMse[-2])/lMse[-2] > THRESH_MSE ):
                    break
                
    dfMseRsq = pd.DataFrame({})
    dfMseRsq['Mse'] = np.array(lMse)
    dfMseRsq['Rsq'] = np.array(lRsq)
    dfMseRsq.index = list(dfImp.columns)                   
        
    return (dfImp, dfMseRsq)
