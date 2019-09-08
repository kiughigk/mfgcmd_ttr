
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import RepeatedKFold

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

"""
output: MSE and R squared value of best hyper parameters:
        self.mse_best, self.rsq_best
        RF model with those hyper parameters:
        self.reg_best
        Hyper Parameter Tuning Log
        self.dfTuningLog
"""

class runRf_opti:

    
    def __init__(self, model, x_data, t_data, **params_dict):

        self.model = model
        self.x_data = x_data
        self.t_data = t_data
        self.key_idx = 0
        self.param_dict = {}
        self.params_dict = params_dict
        self.key_list = list(params_dict.keys())

        self.plot_train = np.array([])
        self.plot_test  = np.array([])

        #initial parameter
        self.param_dict_best = {}
        self.rsq_best = 0.0
        self.mse_best = 1e15
        self.fn_best  = 1e15

        #dataframe for parameter tuning log
        self.dfTuningLog = pd.DataFrame({})

        #declariation
        if (model == 'reg'):
            self.reg_best = RandomForestRegressor()
        elif (model == 'class'):
            self.reg_best = RandomForestClassifier()
        else:
            print('invalid model')

        
    def runCv(self):
        
        self._loopCv(self.key_list, **self.param_dict)
        
        print('***Best Hyper Parameter***')
        
        if (self.model == 'reg'):
            print(self.mse_best, self.rsq_best)
            
        else:
            print(self.mse_best, self.fn_best)

        print(self.param_dict_best)
        
        #refit with all data
        if (self.model == 'reg'):
            self.reg_best = RandomForestRegressor(**self.param_dict_best)
        else:
            self.reg_best = RandomForestClassifier(**self.param_dict_best)
        self.reg_best.fit(self.x_data, self.t_data)
        
        
    def _loopCv(self, key_list, **param_dict):
        
        key = key_list[self.key_idx]
        for val in self.params_dict[key]:
            param_dict[key] = val
            if len(param_dict) == len(self.params_dict):
                self.mdl_train(self.x_data, self.t_data, **param_dict)
            else:
                self.key_idx += 1
                self._loopCv(key_list, **param_dict)
    
        del param_dict[key]
        self.key_idx -= 1

        
    def mdl_train(self, x_data, t_data, **param_dict ):

        t_train, t_test = np.array([]), np.array([])
        y_train, y_test = np.array([]), np.array([])
        
        rkf = RepeatedKFold(n_splits=4, n_repeats=1, random_state=2652124)
        for train_index, test_index in rkf.split(x_data):
            x_train_batch = x_data[train_index]
            t_train_batch = t_data[train_index]
            x_test_batch  = x_data[test_index]
            t_test_batch  = t_data[test_index]
            if (self.model == 'reg'):
                self.reg = RandomForestRegressor(**param_dict)
            else:
                self.reg = RandomForestClassifier(**param_dict)
            self.reg.fit(x_train_batch, t_train_batch)
            t_train = np.insert( t_train, 0, t_train_batch )
            y_train = np.insert( y_train, 0, self.reg.predict(x_train_batch) )
            t_test  = np.insert( t_test,  0, t_test_batch )
            y_test  = np.insert( y_test,  0, self.reg.predict(x_test_batch) )
            #i += 1

        slope_tr,   intercept_tr,   r_value_tr,   p_value_tr,   std_err = stats.linregress(t_train, y_train)
        slope_test, intercept_test, r_value_test, p_value_test, std_err = stats.linregress(t_test, y_test)
        #rsq_tr   = r_value_tr**2
        #rsq_test = r_value_test**2
        rsq_tr   = r2_score(t_train, y_train)
        rsq_test = r2_score(t_test, y_test)
        mse_tr   = mean_squared_error(t_train, y_train)
        mse_test = mean_squared_error(t_test,  y_test)

        self.dfTuningLog = pd.concat([self.dfTuningLog, pd.DataFrame(param_dict, index=[0])], sort=False, ignore_index=True)
        idxTuningLog = self.dfTuningLog.shape[0]-1

        if (self.model == 'reg'):
            print ('%s: mse_train:%0.4f, slope_train:%0.4f, rsq_train:%0.4f, mse_test:%0.4f, slope_test:%0.4f, rsq_test:%0.4f' %
                   (param_dict, mse_tr, slope_tr, rsq_tr, mse_test, slope_test, rsq_test ))
            self.dfTuningLog.loc[idxTuningLog, 'mse_train']   = mse_tr
            self.dfTuningLog.loc[idxTuningLog, 'slope_train'] = slope_tr
            self.dfTuningLog.loc[idxTuningLog, 'rsq_train']   = rsq_tr
            self.dfTuningLog.loc[idxTuningLog, 'mse_test']    = mse_test
            self.dfTuningLog.loc[idxTuningLog, 'slope_test']  = slope_test
            self.dfTuningLog.loc[idxTuningLog, 'rsq_test']    = rsq_test
            
            #pick up if current setting is better than previous one
            if (rsq_test > self.rsq_best) and (mse_test < self.mse_best):
                self.rsq_best = rsq_test
                self.mse_best = mse_test
                self.param_dict_best = param_dict

                self.plot_train = np.append(self.plot_train, [rsq_tr])
                self.plot_test  = np.append(self.plot_test,  [rsq_test])

        else:
            confMtx = confusion_matrix(y_test, t_test);  #confMtx = [[true_neg, false_pos], [false_neg, true_pos]]
            print ('%s: mse:%0.4f, confustion_matrix:%0.4f %0.4f %0.4f %0.4f' %
                   (param_dict, mse_test, confMtx[0][0], confMtx[0][1], confMtx[1][0], confMtx[1][1]) )
            self.dfTuningLog.loc[idxTuningLog, 'mse_test']   = mse_test
            self.dfTuningLog.loc[idxTuningLog, 'tn'] = confMtx[0][0]
            self.dfTuningLog.loc[idxTuningLog, 'fp'] = confMtx[0][1]
            self.dfTuningLog.loc[idxTuningLog, 'fn'] = confMtx[1][0]
            self.dfTuningLog.loc[idxTuningLog, 'tp'] = confMtx[1][1]
            
            #metric is false negative
            if (confMtx[1][0] < self.fn_best):
                self.fn_best = confMtx[1][0]
                self.mse_best = mse_test
                self.param_dict_best = param_dict
                

    def plot_Rsq_score(self):

        fig, ax = plt.subplots(1,1, figsize=(6,4))
        ax.plot(np.arange(len(self.plot_train)), self.plot_train, label='training')
        ax.plot(np.arange(len(self.plot_test)),  self.plot_test, label='test')
        ax.legend(loc='upper right')
        ax.set_ylabel('R Square')
        plt.show()


    ### plot feature importance of the best model ###
    def plot_feature_importance(self, feature_list):
        
        importances = pd.Series(self.reg_best.feature_importances_, index = feature_list)
        importances = importances.sort_values()
        importances.plot(kind = "barh")
        plt.title("importance in the RF Model")
        plt.show()
