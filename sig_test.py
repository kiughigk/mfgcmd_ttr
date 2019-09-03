"""
Description : check significance of difference of two samples
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


def sig_test (dfSampA, dfSampB, paramList):
    """
    Arguments
        dfSampA:   e2e data of sample A [dataframe]
        dfSampB:   e2e data of sample B [dataframe]
        paramList: parameter list to be tested [list]
    
    Return
        p value(KS), p value(T test), p value threshold [dataframe]
    """

    
    numParam = len(paramList)
    if (numParam < 20):
        pValThre = 0.05; #0.05 is a typical P value 
    else:
        pValThre = 1/(1.0*numParam)

    df = pd.DataFrame({}, columns=['pValKs', 'pValT', 'pValThresh'])
    for param in paramList:

        tmp = sig_test_by_param(dfSampA, dfSampB, param)
        df.loc[param] = [tmp[1], tmp[3], pValThre]

    return(df)
    

def sig_test_by_param( dfSampA, dfSampB, param ):

    isInvalid = check_dataframe(dfSampA, dfSampB, param)
    
    if (isInvalid == False):
        arrSampA = dfSampA[param].dropna().values
        arrSampB = dfSampB[param].dropna().values   
        ksVal = stats.ks_2samp(arrSampA, arrSampB)
        tVal  = stats.ttest_ind(arrSampA, arrSampB, axis=0, equal_var=False)
        rslt = [ksVal[0], ksVal[1], tVal[0], tVal[1], len(arrSampA), len(arrSampB)]
        return(rslt)
    else:
        return([np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])


def check_dataframe(dfSampA, dfSampB, param):

    isInvalid = False
    
    if not(param in list(dfSampA.columns)) or not(param in list(dfSampB.columns)):
        isInvalid = True
    elif ((dfSampA[param].dtype == 'object') or (dfSampB[param].dtype == 'object')):
        isInvalid = True
    else:
        nSampA, nSampB = dfSampA[param].dropna().values.shape[0], dfSampB[param].dropna().values.shape[0]
        if (nSampA * nSampB == 0):
            isInvalid = True

    return(isInvalid)
    

def plot_sig_test(dfSampA, dfSampB, paramList, legA=None, legB=None,  figName='twoHist.pdf'):
    """
    Arguments
        dfSampA:   e2e data of sample A [dataframe]
        dfSampB:   e2e data of sample B [dataframe]
        paramList: parameter list to be tested [list]
        figName:   name of histograms with pdf format
    Return
        
    """

    binSize = 50
    pp = PdfPages(figName)     
    for param in paramList:

        isInvalid = check_dataframe(dfSampA, dfSampB, param)
        
        if (isInvalid == False):
            
            arrSampA = dfSampA[param].dropna().values
            arrSampB = dfSampB[param].dropna().values
            ksVal = stats.ks_2samp(arrSampA, arrSampB)      
            tVal  = stats.ttest_ind(arrSampA, arrSampB, axis=0, equal_var=False)
            
            xmax = np.max([np.max(arrSampA), np.max(arrSampB)])
            xmin = np.min([np.min(arrSampA), np.min(arrSampB)])
            xrng = xmax - xmin
            if (xrng == 0):
                continue
            xmax += xrng*0.01
            xmin -= xrng*0.01
            xrng = xmax - xmin
            binArray = list(np.linspace(xmin, xmax, binSize+1))

            fig, ax = plt.subplots(1,1)
            ax1 = ax.twinx()
            ax.hist(arrSampA, bins=binArray, normed=False, histtype='stepfilled', label=legA, alpha=0.5, color='b')
            ax.hist(arrSampB, bins=binArray, normed=False, histtype='stepfilled', label=legB, alpha=0.5, color='red')
            ax.set_xlabel(param)
            ax.set_ylabel('Number')
            ax.set_title('Pval = %0.1e(KS-Test), %0.1e(T-Test)' % (float(ksVal[1]), float(tVal[1])))
            if (legA != None) or (legB != None):
                ax.legend(loc='best')
            ax.tick_params(labelsize=12)
            ax1.hist(arrSampA, bins=binArray+[np.inf], normed=True, histtype='step', cumulative=True, color='b')
            ax1.hist(arrSampB, bins=binArray+[np.inf], normed=True, histtype='step', cumulative=True, color='red')
            ax1.tick_params(labelsize=12)
            ax1.set_ylabel('CDF')

            plt.tight_layout()
            plt.savefig(pp, format='pdf')
            ax.cla()
        
    pp.close()
