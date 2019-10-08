"""
Description : emulate CAti mfg command (only DATI RV calculation)
Return      : dataframe of ACRP
"""

import numpy as np
import pandas as pd
import pickle
import sys
sys.path.append('C:/dstools')
from params import ml4xti_const as xtiConst
import utils as ut


ATI_MAX_SIGNAL_CALC_NMAX    = 30000;       #CCB7-9972
ATI_S0_LO_RANGE             = -25.0;       #CCB7-16567
ATI_S0_HI_RANGE             = 9.676430102; #CCB7-16567 CCB7-17717
ATI_S0_TOLERANCE            = 0.01;        #CCB7-16567
ATI_SLOPE_LO_RANGE          = 1.0;         #CCB7-16567
ATI_SLOPE_HI_RANGE          = 30000.0;     #CCB7-16567
ATI_SLOPE_TOLERANCE         = 0.1;         #CCB7-16567
ATI_RELCYL_ATI = 1
DEFAULT_RV_SCALING_1 = 1.0
DATI_MULTIPLIER = [3.00, 1.36, 1.36, 1.36]
SATI_MULTIPLIER = [4.00, 3.00, 3.00, 3.00]
tempList = ['c', 'n', 'h', 's']

MUEC_NO_ERROR = 0


#appendix of nmax of prediction
apdx4pred = 'max_sknn_rf_xgb'
#prediction is func at even bands and srst at odd bands, then func_even is True
func_even = True


#main function
def make_acrp (df, lBand, usePred, saveDir):

    df = df.reset_index(drop=True)

    if (saveDir is not None):
        ut.mkdir(saveDir)
    else:
        saveDir = '.'

    dfAcrpRid = make_defAcrp(df)

    ### Func ###
    # Single ATI RV
    dfAcrp = setup4dati(df, 'func', tempUpdateFlag=0xf)

    n = df.shape[0]
    for idx in range(n):
        hddsn = df.loc[idx,'hddsn']
        lhd   = df.loc[idx,'lhd']
        print(idx, hddsn, lhd)
        for band in lBand:
            dfAcrp = update_rv4dati(df, dfAcrp, hddsn, lhd, band, 'func', usePred=usePred, tempUpdateFlag=0xf, debug=False)

    #band copy
    dfAcrp = do_rv_copy(dfAcrp, tempUpdateFlag=0xf, debug=False)

    #compare DRAM image and RID
    dfAcrpRid = compare_dram_rid(dfAcrp, dfAcrpRid)

    
    ### SRST ###
    # Single ATI RV
    dfAcrp = setup4dati(df, 'srst', tempUpdateFlag=0x9)
    
    for idx in range(n):
        hddsn = df.loc[idx,'hddsn']
        lhd   = df.loc[idx,'lhd']
        print(idx, hddsn, lhd)
        for band in lBand:
            dfAcrp = update_rv4dati(df, dfAcrp, hddsn, lhd, band, 'srst', usePred=usePred, tempUpdateFlag=0x9, debug=False)

    #band copy
    dfAcrp = do_rv_copy(dfAcrp, tempUpdateFlag=0x9, debug=False)

    #compare DRAM image and RID
    dfAcrpRid = compare_dram_rid(dfAcrp, dfAcrpRid)

    
    # save data to csv and pickle
    dfAcrpRid.to_csv('./%s/dfAcrp.csv' % saveDir)
    dfAcrpRid.to_pickle('./%s/dfAcrp.pkl' % saveDir)
    
    return(dfAcrpRid)



def make_acrp_table():
    
    #arrRv = [[0] * 4] * 3
    colList = ['hddsn', 'lhd']
    
    for temp in ['c', 'n', 'h', 's']:
        for np in ['n', 'p']:
            for band in range(64):
                colList.append('rv%s_%s_%d' % (np, temp, band))
    
    df = pd.DataFrame({}, columns = colList)
    
    return(df)



def make_defAcrp(dfNmax):
    
    dfAcrp = make_acrp_table()
    dfAcrp.loc[:, 'hddsn'] = dfNmax.loc[:,'hddsn'].values
    dfAcrp.loc[:, 'lhd']   = dfNmax.loc[:,'lhd'].values
    dfAcrp.loc[:,'rvn_c_0':'rvp_s_63'] = 1.0
    
    return(dfAcrp)



def setup4dati(dfNmax, proc, tempUpdateFlag=0xf):
    
    T = 30000; #force refresh
    
    dfAcrp = make_acrp_table()
    dfAcrp.loc[:, 'hddsn'] = dfNmax.loc[:,'hddsn'].values
    dfAcrp.loc[:, 'lhd']   = dfNmax.loc[:,'lhd'].values
    dfAcrp.loc[:,'rvn_c_0':'rvp_s_63'] = 1.0
    
    for tmprZone in range(0, 4, 1):
        
        if ( ((tempUpdateFlag >> tmprZone) & 0x1) == 0 ):
            continue
        
        tmpr = tempList[tmprZone]
        
        arrNmax0   = dfNmax.loc[:, 'wrnumminus_%d_%s'%(0, proc):'wrnumminus_%d_%s'%(63, proc)].values / SATI_MULTIPLIER[tmprZone]
        arrNmax100 = dfNmax.loc[:, 'wrnumplus_%d_%s'% (0, proc):'wrnumplus_%d_%s'% (63, proc)].values / SATI_MULTIPLIER[tmprZone]

        arrRVn = np.ceil(T/arrNmax0)
        arrRVp = np.ceil(T/arrNmax100)
        
        dfAcrp.loc[:,'rvn_%s_0'%(tmpr):'rvn_%s_63'%(tmpr)] = arrRVn
        dfAcrp.loc[:,'rvp_%s_0'%(tmpr):'rvp_%s_63'%(tmpr)] = arrRVp
        
    return (dfAcrp)
    


def update_rv4dati(dfNmax, dfAcrp, sn, hd, band, proc, usePred, tempUpdateFlag=0xf, debug=False):
    
    RvScaling = DEFAULT_RV_SCALING_1
    T = 30000; #force refresh
    mixIdxNear50 = 0xFF;
    deltaNear50 = 0xFF;
    mixNear50 = 50
    dMixNear50 = mixNear50 / 100.0
    
    debug_step = 0

    nmax50byband = 'nmax50byband_%d_%s' % (band, proc)
    if (usePred == True):
        if (func_even == True):
            if ((proc=='func') and (band%2 == 0)) or ((proc=='srst') and (band%2 == 1)):
                nmax50byband = 'nmax50byband_%d_%s_%s' % (band, proc, apdx4pred)
        else:
            if ((proc=='func') and (band%2 == 1)) or ((proc=='srst') and (band%2 == 0)):
                nmax50byband = 'nmax50byband_%d_%s_%s' % (band, proc, apdx4pred)
    
    for tmprZone in range(0, 4, 1):
        
        if ( ((tempUpdateFlag >> tmprZone) & 0x1) == 0 ):
            continue
        
        tmpr = tempList[tmprZone]

        #Calculate S0
        #data16 T = MccbCuPtr->GetForceRisk();

        # get nmax
        #Nmax0 and Nmax100 are linear scale, NmaxP is log scale on the data frame.
        Nmax0   = np.ceil(dfNmax.loc[(dfNmax['hddsn']==sn) & (dfNmax['lhd']==hd), 'wrnumminus_%d_%s' % (band, proc)].values / SATI_MULTIPLIER[tmprZone])
        Nmax100 = np.ceil(dfNmax.loc[(dfNmax['hddsn']==sn) & (dfNmax['lhd']==hd), 'wrnumplus_%d_%s'  % (band, proc)].values / SATI_MULTIPLIER[tmprZone])
        NmaxP   = np.ceil(np.power(10, dfNmax.loc[(dfNmax['hddsn']==sn) & (dfNmax['lhd']==hd), nmax50byband].values) / DATI_MULTIPLIER[tmprZone])
        if (debug == True):
            print(debug_step, Nmax0, Nmax100, NmaxP)
            debug_step += 1
        
        #Clip Nmax to T
        if (Nmax0   > ATI_MAX_SIGNAL_CALC_NMAX):
            Nmax0   = ATI_MAX_SIGNAL_CALC_NMAX
        if (Nmax100 > ATI_MAX_SIGNAL_CALC_NMAX):
            Nmax100 = ATI_MAX_SIGNAL_CALC_NMAX;
        if (NmaxP   > ATI_MAX_SIGNAL_CALC_NMAX):
            NmaxP   = ATI_MAX_SIGNAL_CALC_NMAX;

        #CCB7T-5857: Skip RV update when DATI result is better than that of SATI.
        NmaxPAve = Nmax0 + (Nmax100 - Nmax0) * mixNear50 / 100;
        if (NmaxP >= NmaxPAve):
            print('skip calc')
            continue;

        #class
        solves0 = SolveS0(T, Nmax0, Nmax100, NmaxP, dMixNear50);

        #float fz;
        #solveMuec = solves0.FindRootFP(&fz, AtiDefNs::ATI_S0_LO_RANGE, AtiDefNs::ATI_S0_HI_RANGE, 100, AtiDefNs::ATI_S0_TOLERANCE);
        solveMuec, fz = solves0.FindRootFP(ATI_S0_LO_RANGE, ATI_S0_HI_RANGE, 100, ATI_S0_TOLERANCE)
        if (solveMuec == 0):
            ds0 = T / (1.0 - (np.exp(fz)/T))
        if (debug == True):
            print(debug_step, fz, ds0)
            debug_step += 1
        
        # When no solution is found, expect that the DATI result was good enough where
        # the Nmax vs. WriteMix curve does not concave downward and set max Sx0.
        if (solveMuec == 'MUEC_CI_NO_CROSSING_FOUND_PLUS'):
            ds0 = 64000.0;
        elif ((solveMuec != 0) or (fz < -27.0) or (ds0 < 30001.0)):
            ds0 = 30001.0;


        # Calculate decay rates
        da0   = 1.0 / Nmax0 * np.log(ds0 / (ds0 - T))
        da100 = 1.0 / Nmax100 * np.log(ds0 / (ds0 - T))

        # Calculate slope
        # double dslope;
        ds0 = np.floor(ds0)
        if (ds0 > 30001.0):
            dslope = 1.0 / NmaxP * ((da100 * np.exp(-da100 * NmaxP * dMixNear50)) - (da0 * np.exp(-da0 * NmaxP * (1.0 - dMixNear50)))) / \
            ((da100 * dMixNear50 * np.exp(-da100 * NmaxP * dMixNear50)) + (da0 * (1.0 - dMixNear50) * np.exp(-da0 * NmaxP * (1.0 - dMixNear50))));
        else:

            #float fNdn, fNup;
            #MUECType mUECd = MUEC_NO_ERROR;
            #MUECType mUECu = MUEC_NO_ERROR;

            solveslope = SolveSlope(T, da0, da100, ds0);
            solveslope.setRatio(dMixNear50 - 0.01);
            mUECd, fNdn = solveslope.FindRootFP(ATI_SLOPE_LO_RANGE, ATI_SLOPE_HI_RANGE, 100, ATI_SLOPE_TOLERANCE);

            if (mUECd == MUEC_NO_ERROR): 
                solveslope.setRatio(dMixNear50 + 0.01);
                mUECu, fNup = solveslope.FindRootFP(ATI_SLOPE_LO_RANGE, ATI_SLOPE_HI_RANGE, 100, ATI_SLOPE_TOLERANCE);

            if ((mUECd == MUEC_NO_ERROR) and (mUECu == MUEC_NO_ERROR)):
                dslope = ((1.0 / fNup) - (1.0 / fNdn)) / 0.02
            else:
                dslope = 1.0 / NmaxP * ((da100 * exp(-da100 * NmaxP * dMixNear50)) - (da0 * np.exp(-da0 * NmaxP * (1.0 - dMixNear50)))) / \
                        ((da100 * dMixNear50 * np.exp(-da100 * NmaxP * dMixNear50)) + (da0 * (1.0 - dMixNear50) * exp(-da0 * NmaxP * (1.0 - dMixNear50))));

        # Debug data
        # *((sdata16*)(DumpTableDAtiValuePtr[tmprZone] + ((2 * measRange) + 1))) = fz * 100;
        # *(DumpTableDAtiValuePtr[tmprZone] + ((2 * measRange) + 2)) = ds0;

        # Calculate RVn and RVp
        dNmaxFit0   = 1.0 / ((1.0 / NmaxP) - (dslope * dMixNear50));
        dNmaxFit100 = 1.0 / ((1.0 / NmaxP) + (dslope * (1.0 - dMixNear50)));

        rvScalingSub = RvScaling;

        # CCB8-14877: If not legacy 6 location measurement mode, do not use RvScaling
        #if (MccbPtr->GetCmdOperationMode() != ATI_OPE_LEGACY_MEAS):
        #    rvScalingSub = DEFAULT_RV_SCALING_1;
        rvScalingSub = DEFAULT_RV_SCALING_1;

        RVn = np.ceil(T * rvScalingSub / dNmaxFit0);
        RVp = np.ceil(T * rvScalingSub / dNmaxFit100);

        # Save final RVs to all temperatures if new RVn and RVp is greater than the raw value.

        if (debug == True):
            print(debug_step, tmprZone, RVn, RVp)
            debug_step += 1
            
        if (debug == False):
            #if (*(DumpTableDAtiValuePtr[tmprZone]   + measRange - 1) < RVn) *(DumpTableDAtiValuePtr[tmprZone]   + measRange - 1) = RVn ;
            #if (*(DumpTableDAtiValuePtr[tmprZone]   + measRange + 1) < RVp) *(DumpTableDAtiValuePtr[tmprZone]   + measRange + 1) = RVp ;
            idx = dfAcrp.shape[0]
            if (len(dfAcrp.loc[(dfAcrp['hddsn']==sn) & (dfAcrp['lhd']==hd)].values) == 0):
                print('!!! acrp table is not initilized !!!')
                return(-1)
            
            curRVn = dfAcrp.loc[(dfAcrp['hddsn']==sn) & (dfAcrp['lhd']==hd), 'rvn_%s_%d' % (tmpr, band)].values[0]
            if (curRVn < RVn):
                dfAcrp.loc[(dfAcrp['hddsn']==sn) & (dfAcrp['lhd']==hd), 'rvn_%s_%d' % (tmpr, band)] = RVn
                    
            curRVp = dfAcrp.loc[(dfAcrp['hddsn']==sn) & (dfAcrp['lhd']==hd), 'rvp_%s_%d' % (tmpr, band)].values[0]
            if (curRVp < RVp):
                dfAcrp.loc[(dfAcrp['hddsn']==sn) & (dfAcrp['lhd']==hd), 'rvp_%s_%d' % (tmpr, band)] = RVp
                
            #print(tmpr, band, curRVn, curRVp, RVn, RVp)
                
    if (debug == False):
        return(dfAcrp)
    else:
        return(0)


    
def do_rv_copy(dfAcrp, tempUpdateFlag=0xf, debug=False):
    
    dfAcrp_ref = dfAcrp.copy(deep=True)
    
    for tmprZone in range(0, 4, 1):
        
        if ( ((tempUpdateFlag >> tmprZone) & 0x1) == 0 ):
            continue
        
        tmpr = tempList[tmprZone]
    
        band = 0
        dfAcrp.loc[:, 'rvn_%s_%d' % (tmpr, band)] = np.max(dfAcrp_ref.loc[:, 'rvn_%s_%d'%(tmpr, band):'rvn_%s_%d'%(tmpr, band+1)].values, axis=1)
        dfAcrp.loc[:, 'rvp_%s_%d' % (tmpr, band)] = np.max(dfAcrp_ref.loc[:, 'rvp_%s_%d'%(tmpr, band):'rvp_%s_%d'%(tmpr, band+1)].values, axis=1)
        
        for band in range(1, 63):
            
            #print(tmpr, band, dfAcrp.loc[:, 'rvn_%s_%d' % (tmpr, band)].values, np.max(dfAcrp_ref.loc[:, 'rvn_%s_%d'%(tmpr, band-1):'rvn_%s_%d'%(tmpr, band+1)].values, axis=1))
            dfAcrp.loc[:, 'rvn_%s_%d' % (tmpr, band)] = np.max(dfAcrp_ref.loc[:, 'rvn_%s_%d'%(tmpr, band-1):'rvn_%s_%d'%(tmpr, band+1)].values, axis=1)
            dfAcrp.loc[:, 'rvp_%s_%d' % (tmpr, band)] = np.max(dfAcrp_ref.loc[:, 'rvp_%s_%d'%(tmpr, band-1):'rvp_%s_%d'%(tmpr, band+1)].values, axis=1)
            
        band = 63
        dfAcrp.loc[:, 'rvn_%s_%d' % (tmpr, band)] = np.max(dfAcrp_ref.loc[:, 'rvn_%s_%d'%(tmpr, band-1):'rvn_%s_%d'%(tmpr, band)].values, axis=1)
        dfAcrp.loc[:, 'rvp_%s_%d' % (tmpr, band)] = np.max(dfAcrp_ref.loc[:, 'rvp_%s_%d'%(tmpr, band-1):'rvp_%s_%d'%(tmpr, band)].values, axis=1)
    
    
    return(dfAcrp)



def compare_dram_rid(dfAcrp, dfAcrpRID):
    
    dfAcrpRID.loc[:,'rvn_c_0':'rvp_s_63'] = np.max([dfAcrp.loc[:,'rvn_c_0':'rvp_s_63'].values, dfAcrpRID.loc[:,'rvn_c_0':'rvp_s_63'].values], axis=0)
    
    return(dfAcrpRID)



def interpolate_ati_profile_byband(dfAcrpRid, df):
    
    #normTemp = MccbCuPtr->GetFixedDriveTempNormal();
    #hotTemp  = MccbCuPtr->GetFixedDriveTempScorch();
    
    TargetTemp = [45, 55, 65]
    #intRv = static_cast<Ati::RiskType>(ceil(ridRvN * pow((ridRvS / ridRvN),
    #                                                                               ((targetTemp - fNormTemp) /
    #                                                                                (fHotTemp - fNormTemp)))));
    arrRidRvN = dfAcrpRid.loc[:,'rvn_n_0':'rvp_n_63'].values;
    arrRidRvH = dfAcrpRid.loc[:,'rvn_h_0':'rvp_h_63'].values;
    arrRidRvS = dfAcrpRid.loc[:,'rvn_s_0':'rvp_s_63'].values;
    arr1dTempN = df.loc[:,'interptemp_func'].values;
    arr1dTempH = df.loc[:,'interptemp_srst'].values;
    n = arr1dTempN .shape[0]
    
    dfAcrpRid.loc[:,'rvn_n_0':'rvp_n_63'] = np.ceil( arrRidRvN * np.power(arrRidRvS/arrRidRvN, (TargetTemp[0] - arr1dTempN.reshape(n,1))/(arr1dTempH.reshape(n,1) - arr1dTempN.reshape(n,1))))
    dfAcrpRid.loc[:,'rvn_h_0':'rvp_h_63'] = np.ceil( arrRidRvN * np.power(arrRidRvS/arrRidRvN, (TargetTemp[1] - arr1dTempN.reshape(n,1))/(arr1dTempH.reshape(n,1) - arr1dTempN.reshape(n,1))))
    dfAcrpRid.loc[:,'rvn_s_0':'rvp_s_63'] = np.ceil( arrRidRvN * np.power(arrRidRvS/arrRidRvN, (TargetTemp[2] - arr1dTempN.reshape(n,1))/(arr1dTempH.reshape(n,1) - arr1dTempN.reshape(n,1))))                                                
    
    return(dfAcrpRid)


class FindRoot:

    def FindRootFP(self, x1, x2, maxIter, tolerance):

        muec = 0
        root = None
    
        flo = self.eqToSolve(x1)
        fhi = self.eqToSolve(x2)
        #xlo;
        #xhi;

        if ((flo * fhi) <= 0.0): # make sure equation crosses x-axis, otherwise no solution
            if (flo < 0.0):
                xlo = x1;
                xhi = x2;
            else:
                xlo = x2;
                xhi = x1;
                swapTemp = flo;
                flo = fhi;
                fhi = swapTemp;

            for itr in range(0, maxIter, 1):
                rtflsp = (xlo + xhi) / 2;
                f = self.eqToSolve(rtflsp);
                if (f < 0.0):
                    delta = xlo - rtflsp;
                    xlo = rtflsp;
                    flo = f;
                else:
                    delta = xhi - rtflsp;
                    xhi = rtflsp;
                    fhi = f;

                if ((np.abs(delta) < tolerance) or (f == 0.0)):
                    root = rtflsp;
                    return (muec, root);

            muec = 'MUEC_CI_ARITHMETIC_OVERFLOW';

        else:
            # If the solution is outside the search range, determine whether the it is above or below the given search range.
            if(fhi > 0.0):
                muec = 'MUEC_CI_NO_CROSSING_FOUND_PLUS';
            else:
                muec = 'MUEC_CI_NO_CROSSING_FOUND_MINUS';

        return (muec, root)


    
class SolveS0(FindRoot):

    def __init__(self, T, Nmax0, Nmax100, NmaxP, p):
        self.fT       = T;
        self.fNmax0   = Nmax0;
        self.fNmax100 = Nmax100;
        self.fNmaxP   = NmaxP;
        self.fP       = p;
    

    def eqToSolve(self, fz):
        
        fs0 = self.fT / (1 - (np.exp(fz) / self.fT))
        
        return (2.0 - np.exp(np.log(1.0 - (self.fT / fs0)) * self.fNmaxP / self.fNmax0 * (1.0 - self.fP)) -
            np.exp(np.log(1.0 - (self.fT / fs0)) * self.fNmaxP / self.fNmax100 * self.fP) - (self.fT / fs0));
    

    
class SolveSlope(FindRoot):
    
    def __init__(self, fT, fa0, fa100, fs0):
        
        self.fT     = fT;
        self.fa0    = fa0
        self.fa100  = fa100
        #self.fratio = fratio
        self.fs0    = fs0
        
    def setRatio(self, finRatio):
        self.fratio = finRatio;
        
    def eqToSolve(self, x):
        return ((2.0 - np.exp(-self.fa0 * x * (1.0 - self.fratio)) - np.exp(-self.fa100 * x * self.fratio)) - (self.fT / self.fs0));
    
    
