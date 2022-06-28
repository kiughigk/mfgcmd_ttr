import pickle
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.cm as cm
import re
#from sklearn.metrics import mean_squared_error
#from sklearn.metrics import r2_score
#from sklearn.model_selection import GridSearchCV
#from sklearn.datasets import load_boston
#from sklearn.metrics import mean_squared_error

import probscale
import seaborn as sns

import sys
sys.path.append('C:/dstools')
import sig_test
import prob_ellip
from matplotlib.patches import Ellipse, Polygon


def plot_xy(df, colX, colY, xlabel, ylabel, title, figname, disLegend=False, hue=None, hue_order=None, logx=False, logy=False,
            xlim=None, ylim=None, divX=None, divY=None, addThresh=None, addXThresh=None, addYXLine=False, addRegLine=False,
            deg=1, addEllipse=False, sigma=2.0, palette='tab10', hatch=None, closefig=False, alpha=0.5, fail_hd_flag=True):

    #cmap = plt.get_cmap(palette)
    cmap = cm.get_cmap(palette)
    
    fig, ax = plt.subplots(1,1, figsize=(6,4.5))
    
    if hue != None:
        if hue_order==None:
            hue_order = list(np.sort(df.loc[:,hue].unique()))
            #if len(hue_order) > 4:
            #    hue = None   
    else:
        hue_order = ['All']
    print(hue_order)
    
    for i, hueVal in enumerate(hue_order):
        
        hueVal = hue_order[i]
        uniRec = False
        
        if hueVal == 'All':
            df2 = df.loc[:,[colX,colY]].dropna(axis=0)
        else:
            if (df.query("%s=='%s'"%(hue, hueVal)).shape[0] == 0):
                continue
            elif (df.query("%s=='%s'"%(hue, hueVal)).shape[0] == 1):
                uniRec = True
            df2 = df.query("%s=='%s'"%(hue, hueVal)).loc[:,[colX,colY]].dropna(axis=0)
            
        xdata = df2.loc[:,colX].values.astype(np.double)
        ydata = df2.loc[:,colY].values.astype(np.double)
        if (divX != None) and (divX != 0):
                xdata = xdata / divX
        if (divY != None) and (divY != 0):
                ydata = ydata / divY
    
        if uniRec == False or fail_hd_flag==False:
            ax.scatter(xdata, ydata, c=cmap(i), label=hueVal, alpha=alpha)
        else:
            ax.scatter(xdata, ydata, c=cmap(i), label=hueVal, alpha=1, edgecolor='white', zorder=df.shape[0])
        #add prob ellipse
        if addEllipse == True:
            mx, my, width, height, theta = prob_ellip.get_prob_ellip_param(xdata, ydata, sigma)
            #ell = Ellipse(xy=(mx, my), width=width, height=height, angle=theta, color='C%s'%str(i+1), alpha=0.3)
            ell = Ellipse(xy=(mx, my), width=width, height=height, angle=theta, fc=cmap(i), ec='white', alpha=0.3)
            #ell.set_facecolor('none')
            ax.add_artist(ell)
            
        
        if addRegLine == True:
            #deg = 1
            if len(xdata) > 1:
                poly  = np.poly1d(np.polyfit(xdata, ydata, deg))
                ax.plot(np.sort(xdata), poly(np.sort(xdata)), c='white', lw=3, ls='dashed', zorder=len(hue_order))
                ax.plot(np.sort(xdata), poly(np.sort(xdata)), c=cmap(i), lw=1, ls='dashed', label=poly, zorder=len(hue_order)+1)
    
    #ax.legend(loc='best')
    if disLegend==False:
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0, fontsize=10)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='both',which='major')
    ax.set_axisbelow(True)
    if logx==True:
        ax.set_xscale('log')
        ax.grid(axis='x',which='both')
    if logy==True:
        ax.set_yscale('log')
        ax.grid(axis='y',which='both')
    if xlim != None:
        ax.set_xlim((xlim[0], xlim[1]))
    if ylim != None:
        ax.set_ylim((ylim[0], ylim[1]))
    if addYXLine == True:
        xlim2 = ax.get_xlim()
        ylim2 = ax.get_ylim()
        minVal2 = np.min([xlim2[0], ylim2[0]])
        maxVal2 = np.max([xlim2[1], ylim2[1]])
        ax.plot([minVal2, maxVal2], [minVal2, maxVal2], c='black', linewidth=1, linestyle='dashed', zorder=len(hue_order)+2)
    if addThresh != None:
        xlim2 = ax.get_xlim()
        ylim2 = ax.get_ylim()
        minVal2 = np.min([xlim2[0], ylim2[0]])
        maxVal2 = np.max([xlim2[1], ylim2[1]])
        xdata = np.linspace(minVal2     , maxVal2, num=100)
        if not(isinstance(addThresh, list)):
            ax.plot(xdata, addThresh(xdata), c='black', linewidth=1, linestyle='dashed')
        else:
            for coefs in addThresh:
                ax.plot(xdata, coefs(xdata), c='black', linewidth=1, linestyle='dashed')
    if addXThresh != None:
        if not(isinstance(addXThresh, list)):
            ax.axvline(addXThresh, c='black', linewidth=1, linestyle='dashed')
        else:
            for thresh in addXThresh:
                ax.axvline(thresh, c='black', linewidth=1, linestyle='dashed')
                
    if hatch!=None:
        #hatch should be like [[0, 0], [4, 1.1], [6, 2.5], [2, 1.4]]
        ax.add_patch(Polygon(hatch, closed=True, fill=True, color='red', alpha=0.1))
        
    plt.tight_layout()
    plt.savefig(figname+'_xyplot.png', format='png')
    if closefig==True:
        plt.close()


def plot_cdf(df, col, xlabel, title, figname, 
             xlim=None, hue=None, hue_order=None, div=None, addThresh=None, binSize=50, addTable=False, palette='tab10'):

    #can not support multiple thresholds
    if isinstance(addTable, list):
        addTable==False
    
    #cmap = cm.get_cmap('rainbow')
    cmap = cm.get_cmap(palette)
    xdata = df.loc[:,col].dropna(axis=0).values
    if (div != None) and (div != 0):
        xdata = xdata / div
    
    if xlim == None:
        xmax = np.max(xdata)
        xmin = np.min(xdata)
    else:
        xmax = xlim[1]
        xmin = xlim[0]
    xrng = xmax - xmin
    if (xrng == 0):
        return ('single value')
    xmax += xrng*0.01
    xmin -= xrng*0.01
    xrng = xmax - xmin
    binArray = list(np.linspace(xmin, xmax, binSize+1))

    if hue != None:
        if hue_order == None:
            hue_order = list(np.sort(df.loc[:,hue].unique()))
            #if len(hue_order) > 4:
            #    hue = None
    else:
        hue_order = ['All']
    print(hue_order)
    
    if addTable == False:
        fig, ax = plt.subplots(1,1, figsize=(8,5))
    else:
        gridsize = (2, 3)
        fig = plt.figure(figsize=(8, 8))
        ax      = plt.subplot2grid(gridsize, (0, 0), colspan=3, rowspan=1)
        axTable = plt.subplot2grid(gridsize, (1, 1), colspan=2, rowspan=1)
    
    ax1 = ax.twinx()
    

    
    dfPerc = pd.DataFrame({})
    
    for i, hueVal in enumerate(hue_order):
        
        hueVal = hue_order[i]
        uniRec = False
        
        if hue != None:
            if (df.query("%s=='%s'"%(hue, hueVal)).shape[0] > 0):
                if (df.query("%s=='%s'"%(hue, hueVal)).shape[0] == 1):
                    uniRec = True
                xdata                       = df.query("%s=='%s'"%(hue, hueVal)).loc[:,col].dropna(axis=0).values
                dfPerc.loc['count', hueVal] = df.query("%s=='%s'"%(hue, hueVal)).loc[:,col].dropna(axis=0).count()
            else:
                xdata                       = np.nan
                dfPerc.loc['count', hueVal] = np.nan
                
        else:
            xdata = df.loc[:,col].dropna(axis=0).values
            dfPerc.loc['count', hueVal] = df.loc[:,col].dropna(axis=0).count()
            
        if (div != None) and (div != 0):
            xdata = xdata / div
          
        dfPerc.loc['mean', hueVal]   = np.mean(xdata)
        dfPerc.loc['std', hueVal]   = np.std(xdata)
        dfPerc.loc['max', hueVal]   = np.max(xdata)
        dfPerc.loc['+3$\sigma$', hueVal] = np.percentile(xdata, 100*stats.norm.cdf(3))
        dfPerc.loc['median', hueVal]     = np.percentile(xdata, 100*stats.norm.cdf(0))
        dfPerc.loc['-3$\sigma$', hueVal] = np.percentile(xdata, 100*stats.norm.cdf(-3))
        dfPerc.loc['min', hueVal]        = np.min(xdata)

        if dfPerc.loc['count', hueVal] > 0:
            if uniRec == False:
                print(xdata.dtype)
                ax.hist(xdata.astype(np.double), bins=binArray, density=False, histtype='stepfilled', alpha=0.3, 
                        edgecolor='white', color=cmap(i), label=hueVal)
                ax1.hist(xdata.astype(np.double), bins=binArray+[np.inf], density=True, histtype='step', cumulative=True, lw=2, color=cmap(i))
            else:
                ax.axvline(xdata, label=hueVal, lw=2, color=cmap(i))
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Number')
    ax.set_title(title)
    if addThresh != None:
        #ax.axvline(addThresh, c='red', ls='dashed')
        if not(isinstance(addThresh, list)):
            ax.axvline(addThresh, c='red', ls='dashed')
        else:
            for val in addThresh:
                ax.axvline(val, c='red', ls='dashed')
    ax1.tick_params(labelsize=12)
    ax1.set_ylabel('CDF')
    ax.grid(axis='both',which='major')
    """
    if addTable == True:
        ax.legend(loc='upper left')
    else:
        ax.legend(bbox_to_anchor=(1.2, 1), loc='upper left', borderaxespad=0, fontsize=10)
    """
    ax.legend(bbox_to_anchor=(1.2, 1), loc='upper left', borderaxespad=0, fontsize=10)
    ax.set_axisbelow(True)
    
    if addTable == True:
        axTable.axis('off')
        the_table = axTable.table(cellText=np.around(dfPerc.values.astype(np.double), decimals=2),
                      fontsize=20,
                      rowLoc='right',
                      #rowColours=colors, 
                      rowLabels=list(dfPerc.index),
                      #colWidths=[.5,.5], 
                      colLabels=list(dfPerc.columns),
                      cellLoc='center',
                      colLoc='center', 
                      loc='center right'
                     )
        the_table.scale(1.5, 1.5)
        the_table.set_fontsize(14)
        #plt.subplots_adjust(wspace=0.4)
        plt.tight_layout()
    else:
        ax.set_xlabel(xlabel)
        plt.tight_layout()
    
    plt.savefig(figname+'_cdf.png', format='png')


#qq plot
def plot_cdf2(df, col, xlabel, title, figname, addThresh, 
              xlim=None, ylim=None, hue=None, hue_order=None, div=None, binSize=50, addTable=False, logX=False, doBoxCox=False, palette='tab10'):

    #can not support multiple thresholds
    if isinstance(addThresh, list):
        addTable=False
    
    cmap = cm.get_cmap(palette)
    
    xdata = df.loc[:,col].dropna(axis=0).values
    if (div != None) and (div != 0):
        xdata = xdata / div
        if addThresh != False:
            addThresh = addThresh /div
        
    if logX == True:
        xdata = np.log10(xdata)
        if addThresh != False:
            addThresh = np.log10(addThresh)
    
    if doBoxCox == True:
        #xdata, lmd = stats.boxcox(xdata)
        #pt = PowerTransformer(method='yeo-johnson')
        pt = PowerTransformer(method='box-cox')
        pt.fit(xdata.reshape(-1,1))
        #print(pt.lambdas_)
        xdata = pt.transform(xdata.reshape(-1,1)).T[0]
        if addThresh != False:
            addThresh = pt.transform([[addThresh]])[0][0]
    
    
    if xlim == None:
        xmax = np.max(xdata)
        xmin = np.min(xdata)
    else:
        xmax = xlim[1]
        xmin = xlim[0]
    xrng = xmax - xmin
    if (xrng == 0):
        return ('single value')
    xmax += xrng*0.01
    xmin -= xrng*0.01
    xrng = xmax - xmin
    binArray = list(np.linspace(xmin, xmax, binSize+1))

    """
    if addTable == False:
        fig, ax = plt.subplots(1,1, figsize=(8,5))
    else:
        gridsize = (1, 8)
        fig = plt.figure(figsize=(12, 5))
        ax      = plt.subplot2grid(gridsize, (0, 0), colspan=4, rowspan=1)
        axTable = plt.subplot2grid(gridsize, (0, 6), colspan=2, rowspan=1)
    """
    if addTable == False:
        fig, ax = plt.subplots(1,1, figsize=(8,5))
    else:
        gridsize = (2, 3)
        fig = plt.figure(figsize=(8, 8))
        ax      = plt.subplot2grid(gridsize, (0, 0), colspan=3, rowspan=1)
        axTable = plt.subplot2grid(gridsize, (1, 1), colspan=2, rowspan=1)
    
    ax1 = ax.twinx()
    
    if hue != None:
        if hue_order == None:
            hue_order = list(np.sort(df.loc[:,hue].unique()))
            #if len(hue_order) > 4:
            #    hue = None     
    else:
        hue_order = ['All']
    print(hue_order)
    
    dfPerc = pd.DataFrame({})
    
    for i in range(len(hue_order)):
        
        hueVal = hue_order[i]
        uniRec = False
        
        if hue != None:
            if (df.query("%s=='%s'"%(hue, hueVal)).shape[0] > 0):
                if (df.query("%s=='%s'"%(hue, hueVal)).shape[0] == 1):
                    uniRec = True
                xdata                       = df.query("%s=='%s'"%(hue, hueVal)).loc[:,col].dropna(axis=0).values
                dfPerc.loc['count', hueVal] = df.query("%s=='%s'"%(hue, hueVal)).loc[:,col].dropna(axis=0).count()
            else:
                xdata                       = np.nan
                dfPerc.loc['count', hueVal] = np.nan
        else:
            xdata = df.loc[:,col].dropna(axis=0).values
            dfPerc.loc['count', hueVal] = df.loc[:,col].dropna(axis=0).count()
            
        if (div != None) and (div != 0):
            xdata = xdata / div
            
        if logX == True:
            xdata = np.log10(xdata)
            
        if doBoxCox == True:
            #xdata = stats.boxcox(xdata, lmd)
            xdata = pt.transform(xdata.reshape(-1,1)).T[0]
            #print(xdata)
            
        dfPerc.loc['mean', hueVal]  = np.mean(xdata)
        dfPerc.loc['std', hueVal]   = np.std(xdata)
        dfPerc.loc['max', hueVal]   = np.max(xdata)
        dfPerc.loc['+3$\sigma$', hueVal] = np.percentile(xdata, 100*stats.norm.cdf(3))
        dfPerc.loc['median', hueVal]     = np.percentile(xdata, 100*stats.norm.cdf(0))
        dfPerc.loc['-3$\sigma$', hueVal] = np.percentile(xdata, 100*stats.norm.cdf(-3))
        dfPerc.loc['min', hueVal]        = np.min(xdata)
        
        #can not support multiple thresholds
        if (addThresh != False) and (addThresh != None) and not(isinstance(addThresh, list)):
            dfPerc.loc['Distance', hueVal] = np.abs(np.percentile(xdata, 100*stats.norm.cdf(0)) - addThresh)/np.std(xdata)
        else:
            dfPerc.loc['Distance', hueVal] = np.nan
    
        if (uniRec == False) and (not np.isnan(np.mean(xdata))):
            ax.hist(xdata.astype(np.double), bins=binArray, density=False, histtype='stepfilled', alpha=0.3, 
                        edgecolor='white', color=cmap(i), label=hueVal)
            #ax1.hist(xdata.astype(np.double), bins=binArray+[np.inf], density=True, histtype='step', cumulative=True, lw=2, color='C%s'%str(i))
            probscale.probplot(xdata.astype(np.double), 
                               ax=ax1, 
                               plottype='prob', 
                               probax='y', 
                               bestfit=False,
                               estimate_ci=False,
                               datascale='linear', 
                               #problabel='Probabilities (%)', 
                               #datalabel='Min P-Value between KS-Test and T-Test',
                               scatter_kws=dict(marker='.', markersize=5, color=cmap(i), label=hueVal)
                              )
            if ylim != None:
                ax1.set_ylim(ylim[0],ylim[1])
        elif not np.isnan(np.mean(xdata)):
            ax.axvline(xdata, label=hueVal, lw=2, color=cmap(i))
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel('Number')
    ax.set_title(title)
    if (addThresh != False) and (addThresh != None):
        if not(isinstance(addThresh, list)):
            ax.axvline(addThresh, c='red', ls='dashed')
        else:
            for val in addThresh:
                ax.axvline(val, c='red', ls='dashed')
    ax.set_xlim(xmin, xmax)
    ax1.tick_params(labelsize=12)
    ax1.set_ylabel('CDF')
    ax.grid(axis='both',which='major')
    """
    if addTable == True:
        ax.legend(loc='upper left')
    else:
        ax.legend(bbox_to_anchor=(1.2, 1), loc='upper left', borderaxespad=0, fontsize=10)
    """
    ax.legend(bbox_to_anchor=(1.2, 1), loc='upper left', borderaxespad=0, fontsize=10)
    ax.set_axisbelow(True)
    
    if addTable == True:
        axTable.axis('off')
        the_table = axTable.table(cellText=np.around(dfPerc.values.astype(np.double), decimals=2),
                      fontsize=12,
                      rowLoc='right',
                      #rowColours=colors, 
                      rowLabels=list(dfPerc.index),
                      #colWidths=[.5,.5], 
                      colLabels=list(dfPerc.columns),
                      cellLoc='center',
                      colLoc='center', 
                      loc='center right'
                     )
        the_table.scale(1.5, 1.5)
        the_table.set_fontsize(12)
        #plt.subplots_adjust(wspace=0.4)
        plt.tight_layout()
    else:
        ax.set_xlabel(xlabel)
        plt.tight_layout()
    
    plt.savefig(figname+'_cdf.png', format='png')
