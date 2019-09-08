"""
Description : pipeline for ATI Risk Value prediction by using data science
Return      : ml model
"""

import pandas as pd
import numpy as np

import pickle
import copy
import re

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib import patches
import matplotlib.cm as cm
import probscale

import statsmodels.api as sm
from scipy import stats

from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import GridSearchCV

#import xgboost as xgb

import sys
sys.path.append('C:/dstools')
from params import ml4xti_const as xtiConst
import utils as ut
from library import feature_selection as fs
from library import runXgb_opti as rnx
from library import runRf_opti  as rnr

import os
sys.path.append(os.getcwd())
import ml_params as mlp

def pipeline_xti(tCol, sepTestData=True, doDropNoXti=True, doFeatSel=True, saveDir=None):
    #tCol           ... column name of target

    print('algorithm is: %s %s' % (mlp.algo['model'], mlp.algo['ftype']))

    #create a directly for saving data
    if (saveDir is not None):
        ut.mkdir(saveDir)
    else:
        saveDir = '.'
    
    #featureList
    featureList = mlp.featureList

    #data set
    df = mlp.dfData.reset_index(drop=True)
    #drop inf data
    with pd.option_context('mode.use_inf_as_na', True):
        nonNaIndex = list(df.loc[:,featureList].dropna().index)
        df = df.loc[nonNaIndex,:].reset_index(drop=True)
        
    #separate data to training and testing 
    if (sepTestData == True):
        np.random.seed(mlp.randSeed)
        n = df.shape[0]
        nTrain  = int(n * mlp.TrainDataRatio)
        id_all  = np.random.choice(n, n, replace=False)
        dfTrain = df.reset_index(drop=True).iloc[id_all[:nTrain],:]
        dfTrain.to_pickle('./%s/dfTrain.pkl' % saveDir)
        dfTest  = df.reset_index(drop=True).iloc[id_all[nTrain:],:]
        dfTest.to_pickle('./%s/dfTest.pkl' % saveDir)
        print('Number of Data for Training is %d' % nTrain)
        print('Number of Data for Testing is %d'  % (n-nTrain))
    else:
        dfTrain = df
        print('Number of Data for Training is %d' % df.shape[0])
        print('No Data for Testing')
    
    # drop non XTI head
    if (doDropNoXti == True ):
        dfTrain = drop_noxti(dfTrain, tCol, THRESH=np.log10(xtiConst.XTI_NMAX_THRESH))
    
    # feature selection
    if (doFeatSel == True):
        print('\n*** feature selection ***')
        params4FtSel_dict = mlp.params4FtSel_dict; #for feature selection
        print(params4FtSel_dict)
        dfImp, dfMseRsq  = fs.feature_selection(dfTrain, tCol, featureList, mlp.algo['model'], **params4FtSel_dict)
        bestRoundIndex  = list(dfMseRsq[dfMseRsq['Mse']==dfMseRsq['Mse'].min()].index)[0]
        print('\n%s is the best round\n' % bestRoundIndex)
        #if (saveDir is None):
        #    dfMseRsq.to_csv('dfMseRsq_%s.csv' % tCol)
        #    dfImp.to_csv('dfImp_%s.csv' % tCol)
        #else:
        #    #ut.mkdir(saveDir)
        #    dfMseRsq.to_csv('./%s/dfMseRsq_%s.csv' % (saveDir, tCol))
        #    dfImp.to_csv('./%s/dfImp_%s.csv' % (saveDir, tCol))
        dfMseRsq.to_csv('./%s/dfMseRsq_%s.csv' % (saveDir, tCol))
        dfImp.to_csv('./%s/dfImp_%s.csv' % (saveDir, tCol))
        featureListBest = list(dfImp.loc[dfImp[bestRoundIndex]>0, bestRoundIndex].dropna().index)
    else:
        featureListBest = featureList

    #if (saveDir is None):
    #    pd.DataFrame({'columns': featureListBest}).to_csv('./feature_list_best.csv' % saveDir, index=False)
    #else:
    #    pd.DataFrame({'columns': featureListBest}).to_csv('./%s/feature_list_best.csv' % saveDir, index=False)
    pd.DataFrame({'columns': featureListBest}).to_csv('./%s/feature_list_best.csv' % saveDir, index=False)
    
    
    # Main
    print('\n\n*** main ***')

    # Hyper parameters for main tuning
    params_dict       = mlp.params_dict;       #for main routine 
    xdata = dfTrain.loc[:, featureListBest].values
    tdata = dfTrain.loc[:, tCol].values
    if (mlp.algo['model'] == 'xgb'):
        mlModel = rnx.runXgb_opti('reg', xdata, tdata, **params_dict)
    elif (mlp.algo['model'] == 'rf'):
        mlModel = rnr.runRf_opti('reg', xdata, tdata, **params_dict)
    else:
        return ('invalid ML model')
    mlModel.runCv()

    pickle.dump(mlModel, open('%s/%s_%s_%s.pkl' % (saveDir, mlp.algo['model'], mlp.algo['ftype'], tCol), 'wb'))
    if (sepTestData == True):
        pred_w_test_data(dfTest, tCol, mlModel, featureListBest, saveDir)
        
    return (mlModel)


def drop_noxti(df, tCol, THRESH=np.log10(4294967294.0)):
    
    dropIndex = list(df.loc[df[tCol] >= THRESH, tCol].index)
    
    return(df.drop(dropIndex))


def pred_w_test_data(df, tCol, mlModel, feature_list, saveDir):

    xdata = np.array( list(df.loc[:,feature_list].values) )
    tdata = np.array( list(df.loc[:,tCol].values) )
    pred_test = mlModel.reg_best.predict(xdata)
    df['%s_pred' % tCol] = pred_test
    rvRatio = np.ceil(xtiConst.MULTI_DATI * xtiConst.XTI_NMAX_THRESH/np.power(10, pred_test)) / np.ceil(xtiConst.MULTI_DATI * xtiConst.XTI_NMAX_THRESH/np.power(10, tdata))
    df['%s_rvRatio_pred_over_act' % tCol] = rvRatio
    #if (saveDir is None):
    #    df.to_pickle('./dfTestResult_%s.pkl' % tCol)
    #else:
    #    df.to_pickle('./%s/dfTestResult_%s.pkl' % (saveDir, tCol))
    df.to_pickle('./%s/dfTestResult_%s.pkl' % (saveDir, tCol))

    
    ### plot probplot ###
    fig, ax = plt.subplots(1,1, sharex=False, sharey=False, figsize=(5*1+0.2,5*1))
    binsize=50
    #pltRange = (5e-1, 2)
    pltRange = (np.min(rvRatio), np.max(rvRatio))
    binsx = np.linspace(pltRange[0], pltRange[1], binsize)    
    rvRatio = df.loc[:,'%s_rvRatio_pred_over_act' % tCol].values    
    #ax.hist(rvRatio, bins=list(binsx)+[np.inf], normed=True, histtype='step', cumulative=True, label='>%d%%' % SampThreshPcnt)
    fig = probscale.probplot(rvRatio, 
                             ax=ax, 
                             plottype='prob', 
                             probax='y', 
                             bestfit=False,
                             estimate_ci=False,
                             datascale='linear', 
                             problabel='Probabilities (%)', 
                             datalabel='Predicted RV / Actual RV',
                             scatter_kws=dict(marker='.', markersize=1),
    )
    ax.axvline(1.0/np.sqrt(xtiConst.MULTI_DATI), c='red', linestyle='dashed')
    ax.axvline(1.0/xtiConst.MULTI_DATI, c='gray', linestyle='dashed')
    ax.set_xlim(pltRange)
    plt.tight_layout()
    #if (saveDir is None):
    #    plt.savefig('%s_probplot.png' % tCol, format='png')
    #else:
    #    plt.savefig('./%s/%s_probplot.png' % (saveDir, tCol), format='png')
    plt.savefig('./%s/%s_probplot.png' % (saveDir, tCol), format='png')
    ax.cla()


    ### xy plot ###
    fig, ax = plt.subplots(1,1, sharex=False, sharey=False, figsize=(5*1+0.2,5*1))    
    #pltRange = (1e1, 2000)
    xdata = np.ceil(xtiConst.MULTI_DATI * 30000/np.power(10, tdata))
    ydata = np.ceil(xtiConst.MULTI_DATI * 30000/np.power(10, pred_test))
    pltRange = (1, np.max([np.max(xdata), np.max(ydata)]))
    ax.scatter(xdata, ydata, s=10, alpha=0.5)
    ax.set_title('%s, R square=%0.4f' % (tCol, r2_score(tdata, pred_test)))
    ax.plot([pltRange[0], pltRange[1]], [pltRange[0], pltRange[1]], c='black', linestyle='solid', linewidth=0.5)
    ax.plot(np.arange(pltRange[0], pltRange[1]+1, 1), np.arange(pltRange[0]+100, pltRange[1]+100+1, 1), c='red', linestyle='dotted', label='y=x+100')
    ax.plot([pltRange[0], pltRange[1]], [pltRange[0]/np.sqrt(xtiConst.MULTI_DATI), pltRange[1]/np.sqrt(xtiConst.MULTI_DATI)],
            c='red', linestyle='dashed', label='y/x=%0.2f' % (1.0/np.sqrt(xtiConst.MULTI_DATI)))
    ax.grid(which='both', axis='both')
    ax.set_xlabel('Actual RV')
    ax.set_ylabel('Predicted RV')
    ax.set_xlim(pltRange)
    ax.set_ylim(pltRange)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.legend(loc='upper left', fontsize=8)
    plt.tight_layout()
    #if (saveDir is None):
    #    plt.savefig('%s_xyplot.png' % tCol, format='png')
    #else:
    #    plt.savefig('./%s/%s_xyplot.png' % (saveDir, tCol), format='png')
    plt.savefig('./%s/%s_xyplot.png' % (saveDir, tCol), format='png')
    ax.cla()


    ### feature importance ###
    fig, ax = plt.subplots(1,1, sharex=False, sharey=False, figsize=(5*1+0.2,12*1))
    importances = pd.Series(mlModel.reg_best.feature_importances_, index = feature_list)
    importances = importances.sort_values()
    importances[-30:].plot(kind = "barh", color='blue', ax=ax)
    ax.set_title("Feature Importance")
    plt.tight_layout()
    plt.savefig('./%s/%s_feature_importance.png' % (saveDir, tCol), format='png')
    ax.cla()
