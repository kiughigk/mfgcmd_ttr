#Aligne roller profile with FFT
#
#
import pandas as pd
import numpy as np
from scipy import stats
import statsmodels.api as sm
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.cm as cm

#Sector Number of Roller SER is normalized to num

def thinout_align(df, colList, num=200, thinout='default',
                  ref=0, doAlign=True, doRmBaseLine=True,
                  savefile='df_ser_roller_align'):
    #df:         dataframe
    #colList:    columns of roller data: e.g., ser0, ser1, ser2, ser3, ...[
    #num:        number that normalized to
    #thinout:    max:     pick up max value
    #            min:     pick up min value
    #            mean:    pick up mean value
    #            defualt: pick up as is value
    #ref:        index of reference profile or reference profile in list or reference profile in array
    
    
    MAX_SPT = 720
    
    if num > MAX_SPT:
        print('num must be less than 720')
        return(-1)
    
    #data2d = df.loc[:,'ser0':'ser720'].values
    #data2d = df.loc[:,'itr1':'itr539'].values
    data2d = df.loc[:,colList].values
    n = data2d.shape[0]
    
    #count valid data number by head
    nValid = df.loc[:,colList].count(axis=1)
    
    #select indices to pick up by sector ser data
    idx2d  = [np.ceil(np.linspace(0,x-1,num,endpoint=True)).astype(int) for x in nValid]
    
    #thin out by sector ser data to num points
    data2dnormed = np.empty((0,num), float)
    for i in range(n):
    #    data2dnormed = np.append(data2dnormed, [data2d[i][idx2d[i]]], axis=0)
        vals = []
        if thinout == 'mean':
            for j in range(num):
                if j < (num-1):
                    vals.append(np.mean(data2d[i][idx2d[i][j]:idx2d[i][j+1]]))
                else:
                    vals.append(np.mean(data2d[i][idx2d[i][j]:nValid[i]]))
            data2dnormed = np.append(data2dnormed, [vals], axis=0)
        elif thinout == 'max':
            for j in range(num):
                if j < (num-1):
                    vals.append(np.max(data2d[i][idx2d[i][j]:idx2d[i][j+1]]))
                else:
                    vals.append(np.max(data2d[i][idx2d[i][j]:nValid[i]]))
            data2dnormed = np.append(data2dnormed, [vals], axis=0)
        elif thinout == 'min':
            for j in range(num):
                if j < (num-1):
                    vals.append(np.min(data2d[i][idx2d[i][j]:idx2d[i][j+1]]))
                else:
                    vals.append(np.min(data2d[i][idx2d[i][j]:nValid[i]]))
            data2dnormed = np.append(data2dnormed, [vals], axis=0)
        else:
            data2dnormed = np.append(data2dnormed, [data2d[i][idx2d[i]]], axis=0)
            
        
    temp = pd.DataFrame(data=data2dnormed, columns=["ser{}_pct".format(x) for x in range(num)])
    df = df.merge(temp, left_index=True, right_index=True, how='inner')
    #return(temp)
    
    #aligne shadow location
    if doAlign == True:
        #sc = shadow_comparison(df.loc[0,'ser0_pct':'ser%d_pct'%(num-1)].values, num=200)
        if isinstance(ref, int):
            #need casting for polynominal fitting
            ref_array = df.loc[ref,'ser0_pct':'ser%d_pct'%(num-1)].values.astype(np.float64)
            #sc = shadow_comparison(df.loc[ref,'ser0_pct':'ser%d_pct'%(num-1)].values, num)
        elif isinstance(ref, list) or ('numpy.ndarray' in str(type(np.array([10,11])))):
            if (len(ref) != num):
                print ('invalid profile')
                return(-1)
            else:
                if isinstance(ref, list):
                    ref_array = np.array(ref)
                else:
                    ref_array = ref
            #sc = shadow_comparison(ref, num)
        #if doRmBaseLine == True:
        #    deg = 6
        #    poly  = np.poly1d(np.polyfit(np.arange(len(ref_array)), ref_array, deg))
        #    ref_array = ref_array - poly(np.arange(len(ref_array)))
        sc = shadow_comparison(ref_array, num, doRmBaseLine)
        
        arrTemp = np.empty((0,num+1), float)
        for idx in range(0, df.shape[0]):
            shift = sc.calc_shift(data2dnormed[idx])
            arrTemp0 = np.append(data2dnormed[idx][(num>>1)-shift:],data2dnormed[idx][:(num>>1)-shift])
            arrTemp = np.append(arrTemp, [np.append(arrTemp0, np.array([shift]))], axis=0)
        
        temp = pd.DataFrame(data=arrTemp, columns=["ser{}_pct_sort".format(x) for x in range(num)]+['shift'])
        df = df.merge(temp, left_index=True, right_index=True, how='inner')
        
    df.loc[:,'nValid'] = nValid
    df = df.reset_index(drop=True)
    df.to_pickle(savefile+'.pkl')
    df.to_csv(savefile+'.csv')
    
    return(df)

#FFT
def fftcorr(x, y):
    N, X, Y, Rxy = len(x), np.fft.fft(x), np.fft.fft(y), []
    for i in range(N):
        Rxy.append(X[i].conjugate() * Y[i])
    return np.fft.ifft(Rxy)

#sp is fft result, complex number

def fourierComp(sp, minFreq=None, maxFreq=None):
    
    spComp = np.zeros(len(sp))
    if (minFreq == None):
        mn = 1
    else:
        mn = int(np.ceil(minFreq * len(sp)))
    if (maxFreq == None):
        mx = (len(sp) >> 1) + len(sp)%2
    else:
        mx = int(maxFreq * len(sp)) + 1
        
    for i in range(mn, mx, 1):
        spComp[i] = sp[i]
        spComp[-i] = sp[-i]
    spInv = np.fft.ifft(spComp)
    
    freq = np.fft.fftfreq(len(sp))
    lMinFreq = [freq[mn], freq[-mn]]
    lMaxFreq = [freq[mx-1], freq[-mx+1]]
    
    return ([spInv, lMinFreq, lMaxFreq])


class shadow_comparison:
    
    
    def __init__(self, refData, num, doRmBaseLine):

        self.doRmBaseLine  = doRmBaseLine
        if self.doRmBaseLine:
            deg = 6
            poly  = np.poly1d(np.polyfit(np.arange(len(refData)), refData, deg))
            refData = refData - poly(np.arange(len(refData)))
        self.refData       = refData
        self.num           = num
        self.refDataMean   = self.refData.mean()
        self.refDataSig    = self.refData.std()
        self.refDataNormed = refData - self.refDataMean
        
        #FFT and inverse FFT
        self.sp_ref        = np.fft.fft(list(self.refDataNormed))
        self.freq_ref      = np.fft.fftfreq(len(self.refDataNormed))
        self.spInv_ref, lMinFreq_ref, lMaxFreq_ref = fourierComp(self.sp_ref, minFreq=None, maxFreq=10/self.num)
        
        
    def calc_shift(self, data):

        if self.doRmBaseLine:
            deg = 6
            poly = np.poly1d(np.polyfit(np.arange(len(data)), data, deg))
            data = data - poly(np.arange(len(data)))
        dataMean   = data.mean()
        dataNormed = data - dataMean
        dataSig    = data.std()

        #FFT
        sp = np.fft.fft(list(dataNormed))
        freq = np.fft.fftfreq(len(dataNormed))
        spInv, lMinFreq, lMaxFreq = fourierComp(sp, minFreq=None, maxFreq=10/self.num)

        aCov = []
        crs_cor = fftcorr(list(dataNormed - spInv), list(self.refDataNormed - self.spInv_ref))
        #crs_cor = fftcorr(x[tm0-LNGTH:tm0], y[tm0-LNGTH:tm0])
        #mean_x = x.mean()
        #mean_y = y.mean()
        #sigm_x = x.std()
        N = len(data)
        cov1 = (crs_cor[N>>1:N].real / N - self.refDataMean * dataMean) / (self.refDataSig * dataSig)
        cov2 = (crs_cor[0:N>>1].real / N - self.refDataMean * dataMean) / (self.refDataSig * dataSig)
        cov  = np.concatenate([cov1, cov2])
        return(cov.argmax())


#plot roller SER profile

def plot_roller_ser_profile(df, numPoints, title, figname, ylabel='Roller Coaster SER (Symbol ER)',
                            logy=False, ylim=None, calcPerBit=False, hue=None, addFailSec=False, addLeg=False):
    
    fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(10, 6))
    
    cmap = plt.cm.get_cmap('tab20c')
    
    term4PerBit = 0.0
    if calcPerBit == True:
        term4PerBit = np.log10(12.0) 
        
    if hue != None:
        hueList = list(np.sort(df.loc[:,hue].unique()))
        #if len(hueList) > 4:
        #    hue = None   
    if hue == None:
        hueList = ['All']
    
    print(hueList)
    
    
    for i in range(len(hueList)):
        
        hueVal = hueList[i]
        uniRec = False
        
        if hueVal == 'All':
            ydata = (df.loc[:,'ser0_pct_sort':'ser%d_pct_sort'%(numPoints-1)].values - term4PerBit).T
            xdata = np.tile(np.arange(numPoints),(len(ydata[0]),1)).T
            ax.plot(xdata/numPoints, ydata, linewidth=0.5, c='C%s'%str(i), alpha=0.2)
        else:
            if (df.query("%s=='%s'"%(hue, hueVal)).shape[0] > 1):
                ydata = (df.query("%s=='%s'"%(hue, hueVal)).loc[:,'ser0_pct_sort':'ser%d_pct_sort'%(numPoints-1)].dropna(axis=0).values - term4PerBit).T
                xdata = np.tile(np.arange(numPoints),(len(ydata[0]),1)).T
                #ax.plot(xdata/numPoints, ydata, linewidth=0.5, c='C%s'%str(i), alpha=0.2)
                #ax.plot(xdata/numPoints, ydata, linewidth=0.5, c=cmap(i), alpha=0.2)
                ax.plot(xdata/numPoints, ydata, linewidth=0.5, c='C%s'%str(0), alpha=0.2)
            else:
                uniRec = True
                ydata = df.query("%s=='%s'"%(hue, hueVal)).loc[:,'ser0_pct_sort':'ser%d_pct_sort'%(numPoints-1)].values - term4PerBit
                xdata = np.arange(numPoints)
                #ax.plot(xdata/numPoints, ydata[0], linewidth=2, c='C%s'%str(i), alpha=1, zorder=df.shape[0]-1)
                if addLeg==True:
                    ax.plot(xdata/numPoints, ydata[0], linewidth=2, c=cmap(i), alpha=1, zorder=df.shape[0]-1)
                else:
                    ax.plot(xdata/numPoints, ydata[0], linewidth=0.5, c='red', alpha=0.2, zorder=df.shape[0]-1)
                if addFailSec == True:
                    sn, lhd, sct = hueVal.split('_')
                    lhd, sct =int(lhd), int(sct)
                    nValidSec = df.query("%s=='%s'"%(hue, hueVal)).loc[:,'nValid'].values[0]
                    shift     = df.query("%s=='%s'"%(hue, hueVal)).loc[:,'shift'].values[0]
                    #below commeted out lines are not good for comparison between SER profile and RV profile
                    """
                    #pick up a location where the value is larger (worse)
                    #1
                    failSecNormed1 = np.int(sct/nValidSec * numPoints) - ((numPoints>>1)-shift)
                    if failSecNormed1 >= numPoints:
                        failSecNormed1 = failSecNormed1 - numPoints
                    elif failSecNormed1 < 0:
                        failSecNormed1 = failSecNormed1 + numPoints
                    valAtFailSec1 = df.query("%s=='%s'"%(hue, hueVal)).loc[:,'ser%d_pct_sort'%(failSecNormed1)].values[0] - term4PerBit
                    #2
                    failSecNormed2 = np.int(sct/nValidSec * numPoints) - ((numPoints>>1)-shift) + 1
                    if failSecNormed2 >= numPoints:
                        failSecNormed2 = failSecNormed2 - numPoints
                    elif failSecNormed2 < 0:
                        failSecNormed2 = failSecNormed2 + numPoints
                    valAtFailSec2 = df.query("%s=='%s'"%(hue, hueVal)).loc[:,'ser%d_pct_sort'%(failSecNormed2)].values[0] - term4PerBit
                    if (valAtFailSec1 > valAtFailSec2):
                        valAtFailSec = valAtFailSec1
                        failSecNormed = failSecNormed1
                    else:
                        valAtFailSec = valAtFailSec2
                        failSecNormed = failSecNormed2
                    """
                    failSecNormed = np.int(sct/nValidSec * numPoints) - ((numPoints>>1)-shift)
                    if failSecNormed >= numPoints:
                        failSecNormed = failSecNormed - numPoints
                    elif failSecNormed < 0:
                        failSecNormed = failSecNormed + numPoints
                    valAtFailSec = df.query("%s=='%s'"%(hue, hueVal)).loc[:,'ser%d_pct_sort'%(failSecNormed)].values[0] - term4PerBit
                    #ax.scatter([failSecNormed/numPoints], [valAtFailsec], label='%s'%hueVal, zorder=df.shape[0], facecolor='white', edgecolor='C%s'%str(i), s=80)
                    if addLeg == True:
                        ax.scatter([failSecNormed/numPoints], [valAtFailSec], label='%s'%hueVal, zorder=df.shape[0], facecolor='white', edgecolor=cmap(i), s=80)
                    else:
                        ax.scatter([failSecNormed/numPoints], [valAtFailSec], zorder=df.shape[0], facecolor='white', edgecolor='red', s=80)
            
    ax.set_xlabel('Normalized Sector')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', which='major')
    ax.set_axisbelow(False)
    if addLeg == True:
        ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, fontsize=10)
    if ylim != None:
            #ax.set_ylim((-2.2, -0.4))
            ax.set_ylim((ylim[0],ylim[1]))
    if logy:
        ax.set_yscale('log')
            
    plt.tight_layout()
    plt.savefig(figname+'.png', format='png')

    

def plot_roller_ser_profile_by_target(df, numPoints, title, figname, ylabel='Roller Coaster SER (Symbol ER)',
                                      logy=False, ylim=None, calcPerBit=False,
                                      #addFailSec=False, #addLeg=False,
                                      target='mcw_nm'
                                       ):
    
    fig, ax = plt.subplots(1,1, sharex=True, sharey=True, figsize=(10, 6))

    df = df.reset_index()
    
    vmax = int(np.percentile(df.loc[:,target].values, 99.9))
    vmin = int(np.percentile(df.loc[:,target].values, 0.1))
    barRange = vmax - vmin

    term4PerBit = 0.0
    if calcPerBit == True:
        term4PerBit = np.log10(12.0) 

    #ydata = (df.loc[:,'ser0_pct_sort':'ser%d_pct_sort'%(numPoints-1)].values - term4PerBit).T
    #xdata = np.tile(np.arange(numPoints),(len(ydata[0]),1)).T
    #ax.plot(xdata/numPoints, ydata, linewidth=0.5, #c=cm.rainbow((vmax - df.loc[:,target].values)/barRange),
    #        alpha=0.2)
    
    xdata = np.arange(numPoints)
    for i in range(df.shape[0]):
        ydata = df.loc[i,'ser0_pct_sort':'ser%d_pct_sort'%(numPoints-1)].values - term4PerBit
        ax.plot(xdata/numPoints, ydata, linewidth=0.5, alpha=0.6, c=cm.rainbow((df.loc[i,target]-vmin)/barRange))
            
    ax.set_xlabel('Normalized Sector')
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis='y', which='major')
    ax.set_axisbelow(False)

    sm = cm.ScalarMappable(cmap=cm.rainbow, norm=plt.Normalize(vmin=vmin, vmax=vmax))
    #sm = cm.ScalarMappable(cmap=cm.rainbow, norm=plt.Normalize(vmax=vmax, vmin=vmin))
    sm._A = []
    fig.subplots_adjust(left=0.1, right=0.8, top=0.95, bottom=0.1)
    cbar_ax = fig.add_axes([0.81, 0.1, 0.05, 0.85])
    plt.colorbar(sm, cax = cbar_ax, label=target)
            
    #if addLeg == True:
    #    ax.legend(bbox_to_anchor=(1.01, 1), loc='upper left', borderaxespad=0, fontsize=10)
    if ylim != None:
            ax.set_ylim((ylim[0],ylim[1]))
    if logy:
        ax.set_yscale('log')
            
    #plt.tight_layout()
    plt.savefig(figname+'.png', format='png')

"""
    "    vmax = int(np.percentile(df.loc[:,target].values, 99))\n",
    "    vmin = int(np.percentile(df.loc[:,target].values, 1))\n",
    "    #vmin = 0\n",
    "    #vmax = 1000\n",
    "    #print(vmin, vmax)\n",
    "    barRange = vmax - vmin\n",
    "    fig, ax = plt.subplots(1,1, figsize=(6,6))\n",
    "    #ax.scatter(feature[:, 0], feature[:, 1], alpha=0.8, c=cm.rainbow((dfs.loc[:,target].values - dfs.loc[:,target].min())/barRange))\n",
    "    idxA = list(df.loc[df['hddtrial']=='M1Y2'].index)\n",
    "    idxB = list(df.loc[df['hddtrial']=='M1TZ'].index)\n",
    "    #ax.scatter(feature[:, 0], feature[:, 1], alpha=0.6, c=cm.rainbow((vmax - df.loc[:,target].values)/barRange))\n",
    "    ax.scatter(feature[idxA, 0], feature[idxA, 1], marker='o', alpha=0.6, label='M1Y2', c=cm.rainbow((vmax - df.loc[idxA,target].values)/barRange))\n",
    "    ax.scatter(feature[idxB, 0], feature[idxB, 1], marker='s', alpha=0.6, label='M1TZ', c=cm.rainbow((vmax - df.loc[idxB,target].values)/barRange))\n",
    "    #print((vmax - df.loc[:,target].values)/barRange)\n",
    "    ax.grid()\n",
    "    ax.set_title(target)\n",
    "    ax.set_xlabel(\"PC1\")\n",
    "    ax.set_ylabel(\"PC2\")\n",
    "    ax.legend(loc='best')\n",
    "    sm = cm.ScalarMappable(cmap=cm.rainbow, norm=plt.Normalize(vmin=vmin, vmax=vmax))\n",
    "    sm._A = []\n",
    "    fig.subplots_adjust(left=0.1, right=0.8, top=0.95, bottom=0.1)\n",
    "    cbar_ax = fig.add_axes([0.81, 0.1, 0.05, 0.85])\n",
    "    plt.colorbar(sm, cax = cbar_ax, label='ACC')    \n",
    "\n",
    "    #plt.tight_layout()\n",
    "    plt.savefig('%s.png' % figname)\n",
    "    ax.cla()\n",
"""    
    
    
   
