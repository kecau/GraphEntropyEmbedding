#Import packages#

import pandas as pd
import numpy as np
import pywt
from statsmodels.tsa.stattools import grangercausalitytests

#Loading the data#

def LoadingData(City):
    df = pd.read_csv(City).set_index('Date')
    return df

#Data preprocessing#
    
def Preprocessing(df):
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if df.iloc[i,j]>9000: #Since if the value is greater than 9000, it indicates the the data is missing#
                df.iloc[i,j] = 0 #The missing value is removed by using 0#
    return df

#Definition of wavelet transform#
def plot_signal_decomp(data, w):
    mode = pywt.Modes.smooth
    w = pywt.Wavelet(w)#Selecting the wavelet
    a = data
    ca = []#Approximate component
    cd = []#Detail component
    for i in range(4):
        (a, d) = pywt.dwt(a, w, mode)# Doing wavelet transform
        ca.append(a)
        cd.append(d)
 
    rec_a = []
    rec_d = []
 
    for i, coeff in enumerate(ca):
        coeff_list = [coeff, None] + [None] * i
        rec_a.append(pywt.waverec(coeff_list, w))#Reconstructing the signal
 
    for i, coeff in enumerate(cd):
        coeff_list = [None, coeff] + [None] * i
        if i ==3:
            print(len(coeff))
            print(len(coeff_list))
        rec_d.append(pywt.waverec(coeff_list, w))
    return np.array(rec_a),np.array(rec_d)

#Getting the component signal#
def getwavelet(data,w):
    data_rec_a,data_rec_d = plot_signal_decomp(data,w)
    return data_rec_a,data_rec_d

#Time interval selection#
def Timeinterval(df,w):
    Reca = []
    Recd = []
    for i in df.columns:
        Reca.append(getwavelet(df[i].values,w)[0])
        Recd.append(getwavelet(df[i].values,w)[1])
    Reca,Recd = np.array(Reca),np.array(Recd)
    i = 1
    diff = []
    while i <=Recd[0][3].shape[0]:
        dif = Recd[0][3][:i].dot(Recd[1][3][:i])
        i+=1
        diff = np.array(dif)
    rec_ad,rec_dd = plot_signal_decomp(diff,w)
    
    i = 1
    pt = []
    pt1 = []
    while i <=diff.shape[0]-2:
        if rec_ad[3][i-1]<rec_ad[3][i]>rec_ad[3][i+1]:
            pt.append(i)
        elif rec_ad[3][i-1]>rec_ad[3][i]<rec_ad[3][i+1]:
            pt.append(i)
            i+=1
        pt,pt1 = np.array(pt),np.array(pt1)
    return pt, pt1

#Causality calculation#
def grangers_causation_matrix(data,variables,test = 'ssr_chi2test', verbose = False):
    maxlag = 1
    test = 'ssr_chi2test'
    X_train = pd.DataFrame(np.zeros((len(variables),len(variables))),columns = variables, index = variables)
    for c in X_train.columns:
        for r in X_train.index:
            test_result = grangercausalitytests(data[[r,c]],maxlag = maxlag, verbose = False)
            #print(data[[r,c]])
            p_values = [round(test_result[i+1][0][test][1],4) for i in range(maxlag)]
            #print(p_values)
            if verbose: print(f'y={r},x={c}, P values = {p_values}')
            min_p_value = np.min(p_values)
            X_train.loc[r,c] = min_p_value
    X_train.columns = [var + '_x' for var in variables]
    X_train.index = [var + '_y' for var in variables]
    return X_train

def processingforcausality(data):
    causality = grangers_causation_matrix(data,data.columns,test = 'ssr_chi2test', verbose = False)
    return causality.fillna(0)

def patternofcausality(data):
    causality = processingforcausality(data)
    print(causality)
    for i in range(causality.shape[0]):
        for j in range(causality.shape[1]):
            if causality.iloc[i][j] >= 0.05:
                causality.iloc[i][j] = 1
    return causality

#Calculating the correlation coefficient#
def processingforcorr(data):
    Corr = data.corr().values
    Corr = np.abs(Corr)
    Corr = pd.DataFrame(Corr)
    return Corr.fillna(0).values

#Calculating the spurious correlation coefficient#
def NewMatrix(data):
    Causality = patternofcausality(data)
    Corr = processingforcorr(data)
    Mat = Corr
    for i in range(Corr.shape[0]):
        for j in range(Corr.shape[1]):
            Mat[i][j] = Causality.iloc[i][j] - Corr[i][j]
    for i in range(Mat.shape[0]):
        for j in range(Mat.shape[1]):
            if Mat[i][j] == 0:
                Mat[i][j] = 1
    return Mat

#Claculating the spurious correlation coefficient for each time interval#

def NewMatrix_EachTime(df,pt):
    X = []
    new = NewMatrix(df.iloc[:pt[1]])
    X.append(new)
    i = 1
    From = []
    End = []
    while i<=pt.shape[0]-2:
    #print(i)
        new = NewMatrix(df.iloc[pt[i]:pt[i+1]])
    #print(new)
        X.append(new)
        From.append(df.index[pt[i]])
        End.append(df.index[pt[i+1]])
        i=i+1
    #print(i)
    return np.array(X),From,End

#Saving data#
def Savedata(City,df,pt):
    X, From, End = NewMatrix_EachTime(df,pt)
    np.savetxt(City+'.csv', X.reshape(X.shape[0],18*18,delimiter = ','))
    Date = np.vstack([From,End]).T
    np.savetxt('Date_'+City+'.csv', Date, delimiter = ',',fmt = '%s')
    return None