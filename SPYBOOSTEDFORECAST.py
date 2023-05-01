import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBRegressor as xgbr
from xgboost import XGBClassifier as xgbc
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection._split import _BaseKFold, indexable, _num_samples
import yfinance as yf
import time
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from fredapi import Fred 
#import scikitplot as skplt
# CV method override basekfold inhereit the parent class

class PurgedTimeSeriesSplitGroups(_BaseKFold):
    def __init__(self,groups, n_splits=5, purge_groups=0):
        super().__init__(n_splits, shuffle=False, random_state=None)
        self.purge_groups = purge_groups
        self.groups = groups

    def split(self, X, y=None,groups=None):
        # override groups with initalizer era dependent 
        X, y, self.groups = indexable(X, y, self.groups)
        n_samples = _num_samples(X)
        n_folds = self.n_splits + 1
        group_list = np.unique(self.groups)
        n_groups = len(group_list)
        if n_folds + self.purge_groups > n_groups:
            raise ValueError((f"Cannot have number of folds plus purged groups "
                              f"={n_folds+self.purge_groups} greater than the "
                              f"number of groups: {n_groups}."))
        indices = np.arange(n_samples)
        test_size = ((n_groups-self.purge_groups) // n_folds)
        test_starts = [n_groups-test_size*c for c in range(n_folds-1, 0, -1)]
        for test_start in test_starts:
            yield (indices[self.groups.isin(group_list[:test_start-self.purge_groups])],
                   indices[self.groups.isin(group_list[test_start:test_start + test_size])])
            

class SignalBoosted():
    def __init__(self,period = '2y',interval ='1d',perc = 0.90):

        # insert below
        # yfinance pipeline
        #how to properly implement 

        self.tickers = ['SPY']
        self.feat = ['Close','Volume']
        self.X = yf.download(tickers =self.tickers,period = period, interval = interval)[self.feat]
        fig, axs = plt.subplots(2,1,figsize = (12,12))
        
        axs[0].scatter(self.X.index,self.X.Close,c='r',label= 'SPY Closing Prices')
        axs[0].set_title('Plot of 5 min closing')
        axs[0].set_xlabel('Over Time(1Mo)')
        axs[0].set_ylabel('Closing Prices')
        
        axs[1].hist(self.X.Close, bins = 50 ,density=True,color='b',label = 'Histogram' )
        axs[1].set_title('Plot of 5 min closing bins')
        axs[1].set_xlabel('Closing Prices')
        axs[1].set_ylabel('Freq')
        

        axs[0].legend()
        axs[1].legend()


        self.X,self.X_og = self.LogChangeFracDiff()
        self.X = self.X.drop(self.feat,axis=1)
 
        self.X['era'] = np.arange(0,self.X.shape[0],1)
       
        trainendog = self.X
        self.newy = trainendog['5mRet_Close'][0:int(perc*self.X.shape[0])]
        self.testy = trainendog['5mRet_Close'][int(perc*self.X.shape[0]):self.X.shape[0]]
        self.newy_og = self.X_og['5mRet_Close'][0:int(perc*self.X.shape[0])]
        self.testy_og = self.X_og['5mRet_Close'][int(perc*self.X.shape[0]):self.X.shape[0]]
        self.newy.columns = ['5mRet_Close_y']

      
        lag1 = np.arange(trainendog['era'][0],trainendog['era'].max(),1)
        X1 = trainendog[ trainendog['era'].isin(lag1)].drop('era',axis =1).dropna()

       
        self.newX = X1.fillna(X1.mean())
        self.newy = self.newy.shift(1).fillna(self.newy.mean())

        self.newyc = np.sign(self.newy.values.reshape(-1,1))
        indc = np.where(self.newyc == -1)
        self.newyc[indc ] = 0
        self.newyc= pd.DataFrame(self.newyc)
        
        
        self.testyc = np.sign(self.testy.values.reshape(-1,1))
        self.testyc_og = self.testyc
        indc2 = np.where(self.testyc == -1)
        self.testyc[indc2] = 0
        self.testyc= pd.DataFrame(self.testyc)
        self.newX['date']= self.newX.index.astype('str')
    
        jawn=[]
        for row in range(0,self.newX.shape[0]):
            jawn.append(float(self.newX['date'][row].split('-')[1]))
        jawn = np.array(jawn).reshape(-1,1).astype('float')
        self.newX['eraDay'] = jawn

        key='bb358fb7479df683ef3c8fb6df7c3ebf'
        
        codes={'Corporate': 'BAA10Y','10Y':'DGS10','1Y': 'DGS1'}
       
        self.macro = self.downloadmacro( codes,key,self.X.index[0]).resample('MS').ffill()

        self.macro.columns=list(codes.keys())

        fig, ax = plt.subplots(2,1)

        self.macro['empl'] = Fred(key).get_series('PAYEMS', observation_start = self.X.index[0]) # seasonally ajusted non-fram employment in thousands 
        self.macro['gdp'] = Fred(key).get_series('GDPC1', observation_start = self.X.index[0])
        self.macro['unemp'] =Fred(key).get_series('UNRATE', observation_start = self.X.index[0])

     #   ax[0].plot(np.arange(0,38,1),self.macro.gdp.dropna().values,c= 'r')


        for o in self.macro.columns: 
            self.newX[o] = self.macro.loc[:,o]


        self.newX = self.Interp(self.newX,'gdp',o1=2,o2=1)
#
        self.newX = self.Interp(self.newX,'10Y',o1=2,o2=1)

        self.newX = self.Interp(self.newX,'1Y',o1=2,o2=1)

        self.newX = self.Interp(self.newX,'Corporate',o1=2,o2=1)

        self.newX = self.Interp(self.newX,'empl',o1=2,o2=1)

        self.newX = self.Interp(self.newX,'unemp',o1=2,o2=1)


        self.newX['eraDay']=StandardScaler().fit_transform(self.newX['eraDay'].values.reshape(-1,1))


      #  self.macro['Time']=((self.macro['10Y']-self.macro['1Y'])/((self.macro['10Y']).expanding().mean()))*100
      #  self.macro['Confidence']=((self.macro['10Y']-self.macro['Corporate'])/((self.macro['10Y']).expanding().mean()))*100
        
        self.newX = self.NormalizeMacro(self.macro.columns)
   

        self.newXtest = self.newX[int(perc *self.newX.shape[0] ):self.newX.shape[ 0]]
        self.newX = self.newX[0:int(perc*self.newX.shape[0])]

       # print(self.newX.isnull().sum())
    
    @staticmethod
    def Interp(macro,col,o1=2,o2=1) -> pd.DataFrame():
         
        macro[col] = macro[col].interpolate(method='spline',order =o1)

        rev = pd.DataFrame(macro[col].values.reshape(-1,1)[::-1])
        rev = rev.interpolate(method='spline',order =o2)
        rev= rev.values.reshape(-1,1)[::-1]

        macro[col] = rev

        return macro


    @staticmethod
    def downloadmacro(codes,key,date2)-> pd.DataFrame():
        key='bb358fb7479df683ef3c8fb6df7c3ebf'
        fred = Fred(api_key=key)
        data={}
        for i in codes.values():
            data[i]=fred.get_series(i,observation_start=date2)
        macro=pd.DataFrame.from_dict(data)
    
        return macro

    def NormalizeMacro(self,col) -> pd.DataFrame():
        
         self.newX[col]=StandardScaler().fit_transform(self.newX[col].values)
         return self.newX

    
    @staticmethod
    def OGfrac(x, d):
        if np.isnan(np.sum(x)):
            return None

        n = len(x)
        if n < 2:
            return None

        x = np.subtract(x, np.mean(x))

    # calculate weights
        weights = [0] * n
        weights[0] = -d
        for k in range(2, n):
            weights[k - 1] = weights[k - 2] * (k - 1 - d) / k

    # difference series
        ydiff = list(x)

        for i in range(0, n):
            dat = x[:i]
            w = weights[:i]
            ydiff[i] = x[i] + np.dot(w, dat[::-1])

        return ydiff
    
    def LogChangeFracDiff(self,d = 0.85):
        
        self.X['5mRet_'+'Close'] = np.log(self.X.Close) - np.log(self.X.Close.shift(1))

        self.X_og = self.X.copy()
        #self.X['5mRet_'+'Volume'] = np.log(self.X.Volume) - np.log(self.X.Volume.shift(1))

        self.X= self.X.dropna()

        self.InvTransform= pd.DataFrame(index = self.X.index)
        self.InvTransform['5mRet_'+'Close'] = self.X['5mRet_'+'Close'].values.reshape(-1,1) -  np.array(self.OGfrac(self.X['5mRet_'+'Close'].values.reshape(-1,1),d)).ravel().reshape(-1,1)
       # self.InvTransform['5mRet_'+'Volume'] = self.X['5mRet_'+'Volume'].values.reshape(-1,1) -  np.array(self.OGfrac(self.X['5mRet_'+'Volume'].values.reshape(-1,1),0.25)).ravel().reshape(-1,1)
        
        self.X['5mRet_'+'Close'] = np.array(self.OGfrac(self.X['5mRet_'+'Close'].values.reshape(-1,1),d)).ravel().reshape(-1,1)
        #self.X['5mRet_'+'Volume'] = StandardScaler().fit_transform(self.X['Volume'].values.reshape(-1,1))
        #self.X['5mRet_'+'Close'] = StandardScaler().fit_transform(self.X['5mRet_'+'Close'].values.reshape(-1,1))
        self.X[['5mRet_'+'Close','5mRet_'+'Volume' ]] = StandardScaler().fit_transform(self.X[['5mRet_'+'Close','Volume' ]].values)

        return self.X,self.X_og
    

    def TradingSignalXG(self,gridg):
        st_time = time.time()

        initgc = {
        "n_estimators" : 125,
       # "max_depth" : 5,
      #  "learning_rate" : 0.01,
        "eval_metric":"auc",
        "monotone_constraints":{'5mRet_Close' :0, 'eraDay': 0,'5mRet_'+'Volume':0},
      #  "objective":"binary:logistic"
       # "":20,
      # 'early_stopping_rounds' : 5
      #  "feature_selector":'greedy'
     #   "colsample_bytree" : 0.1,
       # "tree_method" : 'gpu_hist'
        }

      #  print(self.newX.head())

        self.predc =[]

        pred = []

        copx = self.newX.copy()
        copy= self.newyc.copy()
        window = int(self.newXtest.shape[0]/10)

        print(self.newyc.shape)
        print(self.newX.head())


        for o in range(0,self.newXtest.shape[0]-window):

            self.newX= copx

           # print(self.newyc.shape)
           # print(self.newX.shape)

            self.newyc = copy

            self.newX = pd.concat([self.newX,self.newXtest.iloc[0:o+window,:]],axis=0)

            self.newyc= pd.concat([self.newyc,self.testyc[0:o+window]])

            ptg =  PurgedTimeSeriesSplitGroups(groups = self.newX['eraDay'], n_splits=3)

            model_gc = xgbc(**initgc)
         
            bscvc = GridSearchCV(model_gc,gridg, cv = ptg)

            bscvc.fit(self.newX.drop('date',axis =1),self.newyc.iloc[0:self.newX.shape[0],0])

            optparamsc = bscvc.best_params_
#
            model_gc.set_params(**optparamsc)
       
            model_gc.fit(self.newX.drop('date',axis =1 ),self.newyc[0:self.newX.shape[0]],eval_set=[(self.newXtest.drop('date',axis =1).iloc[0+o:o+window,:],self.testyc[0+o:o+window])])

            self.predc.append(model_gc.predict_proba(self.newXtest.drop('date',axis = 1).iloc[0+o:o+window,:])[-1])

            pred.append(model_gc.predict(self.newXtest.drop('date',axis = 1).iloc[0+o:o+window,:])[-1])

       # print(self.predc)

        pred = np.array(pred)
        pred= pred.reshape(-1,1)
        
        self.predc = np.array(self.predc)
        self.predc = self.predc.reshape( -1,1)

        fpr, tpr, thresholds = metrics.roc_curve(self.testyc[0:pred.shape[0]], pred)
        print(' On testing set AUC: {} '.format(metrics.auc(fpr, tpr)))

        print(np.round(self.predc[:,0],0))

        fig, ax = plt.subplots(1,1)

      #  ax.plot(fpr,tpr,label = 'g')

        ax.set_xlabel('FPR')
        ax.set_ylabel('TPR')

        ax.legend()
 #   
       # plt.show()

        st_time2= time.time()
              
        print('Time taken to tune XGBR model with GridSearchCV {}'.format(st_time2 - st_time))

        

        return self.predc
    


    def TradingSignalSVC(self,gridg):
        initgc={
            'kernel' : 'rbf',
            'probability':True,
            

        }
        print(self.newX.head())

        model_gc = SVC(**initgc)

        ptg =  PurgedTimeSeriesSplitGroups(groups = self.newX['eraDay'], n_splits=5)

        bscvc = GridSearchCV(model_gc,gridg, cv = ptg)
        
        bscvc.fit(self.newX.drop('date',axis =1).values,self.newyc[0:self.newX.shape[0]].values.reshape(-1,1).ravel())

        optparamsc = bscvc.best_params_
#
        model_gc.set_params(**optparamsc)

        #print(self.newyc.shape)
       
        model_gc.fit(self.newX.drop('date',axis =1 ).values,self.newyc[0:self.newX.shape[0]].values.reshape(-1,1).ravel() )
                     #,eval_set=[(self.newXtest.drop('date',axis =1),self.testyc[0:self.newXtest.shape[0]])])

        self.predsvc = model_gc.predict_proba(self.newXtest.drop('date',axis = 1).values)

        pred = model_gc.predict(self.newXtest.drop('date',axis = 1).values)

        predind = np.where(pred == 0)
        predo = np.ones(pred.shape[0])
        predo[predind] = -1

        self.new_returns = predo.reshape(-1,1)* self.testy_og.values.reshape(-1,1)

        self.new_returns=self.new_returns.cumsum().reshape(-1,1)

        ep = np.mean(self.new_returns)

        risk = np.std(self.new_returns)

        equity = np.zeros(self.new_returns.shape[0])
        equity[0] = 100
        

        sharpe = ep/risk

        print('Sharpe Ratio of {}'.format(sharpe))
        print('shape of new returns {}'.format(self.new_returns.shape))
        for t in range(1,equity.shape[0]):
            equity[t] = equity[t-1]*np.exp(self.new_returns[t-1])


        fpr, tpr, thresholds = metrics.roc_curve(self.testyc, pred)
        print(' On testing set AUC: {} '.format(metrics.auc(fpr, tpr)))


        fig, ax = plt.subplots(2,1)

        ax[0].plot(fpr,tpr,label = 'g')
        ax[1].plot(self.newXtest[0:int(self.new_returns.shape[0])].index,equity,label = 'g')

        ax[0].set_xlabel('FPR')
        ax[0].set_ylabel('TPR')


        ax[1].set_xlabel('Time')
        ax[1].set_ylabel('Equity')

        for tick in ax[1].get_xticklabels():
            tick.set_rotation(90)

        ax[0].legend()
 #   
        plt.show()

        st_time2= time.time()

        return self.predsvc
    

    def TradingSignalEnsemble(self):
         
        gridg2  = {
    'learning_rate': [0.03, 0.01, 0.03, 0.009],
  #  'bagging_temperature': [0, 1,5,10],                                                                                                                                                                                                 
     'n_estimators':np.arange(100,300,50),
     'reg_alpha': [0.1, 0.5],
     'reg_lambda':[0, 1, 1.5],
  #   'num_leaves':np.arange(25,40,5),
        'alpha':[0,1],
        'max_depth':[2,3,4], 
        'min_child_weight':[1,2,3],
        #'gamma':np.arange(0,1.1,0.1)
    #    'colsample_bytree':[0.9,1]
     }    

        gridsvc= {'C':[0.1,0.2,0.3],
          'gamma':[0.2,0.3,0.4,0.5,0.6],
          } 

        self.predc = self.TradingSignalXG(gridg2)[:,0]

        self.predsvc = self.TradingSignalSVC(gridsvc)[:,0]

        superpos =  self.predc + self.predsvc

        superpos=np.round(superpos/2,0)

        fpr, tpr, thresholds = metrics.roc_curve(self.testyc, superpos)
        print(' On superimpose model the testing set AUC: {} '.format(metrics.auc(fpr, tpr)))



    def Boost(self,gridg):

        st_time = time.time()
        
        initg = {
        "n_estimators" : 125,
        "max_depth" : 5,
      #  "learning_rate" : 0.01,
        "eval_metric":"mape",
        "monotone_constraints":{'5mRet_Close' :0, 'eraDay': 0,'5mRet_'+'Volume':0},
      #'early_stopping_rounds' : 1
      #  "feature_selector":'greedy'
     #   "colsample_bytree" : 0.1,
       # "tree_method" : 'gpu_hist'
        }

        model_g = xgbr(**initg)
        ptg =  PurgedTimeSeriesSplitGroups(groups = self.newX['eraDay'], n_splits=3)

        bscv = GridSearchCV(model_g,gridg, cv = ptg)
        
        bscv.fit(self.newX.drop('date',axis =1),self.newy[0:self.newX.shape[0]])

        optparams = bscv.best_params_
#
        model_g.set_params(**optparams)

        model_g.fit(self.newX.drop('date',axis =1 ),self.newy[0:self.newX.shape[0]],eval_set=[(self.newXtest.drop('date',axis =1),self.testy[0:self.newXtest.shape[0]])])

        loss_curvey = model_g.evals_result()['validation_0']['mape']

        pred = model_g.predict(self.newXtest.drop('date',axis = 1))

        fig, ax = plt.subplots(2,1)

        ax[0].plot(np.arange(0,len(loss_curvey)),loss_curvey, c = 'b', label=' Loss Function RMSE over training iterations')
        ax[1].plot(self.newXtest.index,pred,c ='b', label='prediction')
     
        ax[1].plot(self.newXtest.index,self.testy,c ='r', label='actual')

        for tick in ax[1].get_xticklabels():
            tick.set_rotation(90)
     
        ax[0].set_xlabel('Iterations')
        ax[0].set_ylabel('RMSE loss curve')

        ax[0].legend()
        ax[1].legend()
    
        plt.show()

        st_time2= time.time()
              
        print('Time taken to tune XGBR model with GridSearchCV {}'.format(st_time2 - st_time))
  
sb = SignalBoosted()

gridg2  = {
    'learning_rate': [0.03, 0.01, 0.003, 0.009],
    #'bagging_temperature': [0, 1,5,10],                                                                                                                                                                                                 
   #  'n_estimators':np.arange(100,300,50),
     'reg_alpha': [0.1, 0.5],
     'reg_lambda':[0, 1, 1.5],
  #   'num_leaves':np.arange(25,40,5),
    #    'alpha':[0,1],
        'max_depth':[3,4,6,7,8], 
       'min_child_weight':[1,2,3],
        #'gamma':np.arange(0,1.1,0.1)
    #    'colsample_bytree':[0.9,1]
     }    

gridsvc= {'C':np.arange(0.01,1,0.01),
          'gamma':np.arange(0.01,1,0.01),
          

          } 


   
#sb.Boost(gridg2)
#sb.TradingSignalXG(gridg2)
sb.TradingSignalSVC(gridsvc)
#sb.TradingSignalEnsemble()