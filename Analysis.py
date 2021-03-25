import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

LOF = pd.read_csv('ODBJLOF.csv')
IF = pd.read_csv('ODBJIF.csv')
BP = pd.read_csv('ODBJBP.csv')
Date = pd.read_csv('Date_Beijing.csv',header = None)
df = pd.read_csv('WeatherDatasetBeijing.csv').set_index('Date')

X = pd.concat([IF,Date],axis= 1)
a = X[X['Groud']==-1]

df.loc[X.iloc[145,-2]:X.iloc[145,-1]].shape

def evaluation(pred,groud):
    TP = 0
    FP = 0
    TN = 0
    FN = 0
    for i in range(pred.shape[0]):
        if pred[i] == groud[i] == -1:
            TP+=1
        elif pred[i] == groud[i] == 1:
            TN +=1
        elif pred[i] == -1 and  groud[i] == 1:
            FP+=1
        elif pred[i] == 1 and groud[i] == -1:
            FN +=1
    ACC = (TP + TN)/(TP+TN+FP+FN)
    Pre = (TP)/(TP+FP+0.1)
    Rec = TP/(TP+FN+0.1)
    F1 = (2*Pre*Rec)/(Pre+Rec+0.1)
    return ACC,Pre,Rec,F1

pred_LOF = LOF['Label'].values
groud_LOF = LOF['Groud'].values

pred_IF = IF['Label'].values
groud_IF = IF['Groud'].values

pred_BP = BP['Label'].values
groud_BP = BP['Groud'].values

ACC_LOF,Pre_LOF,Rec_LOF,F1_LOF = evaluation(pred_LOF,groud_LOF)

ACC_IF,Pre_IF,Rec_IF,F1_IF = evaluation(pred_IF,groud_IF)

ACC_Bp,Pre_BP,Rec_BP,F1_BP = evaluation(pred_BP,groud_BP)

plt.figure(figsize=(9,6))
plt.subplots_adjust(wspace =0, hspace =0.4)
plt.subplot(3,1,1)

plt.plot(x,y,color = 'k',ls = '--',label = '1 Year',marker = 'o')
plt.xlabel('Date',fontsize = 12)
plt.ylabel('Days',fontsize = 12)
plt.legend(loc = 'lower right')
plt.xticks(rotation=90)

plt.subplot(3,1,2)
plt.plot(x1,y1,color = 'k',ls = '--',label = '5 Year',marker = 'D')
plt.xlabel('Date',fontsize = 12)
plt.ylabel('Days',fontsize = 12)
plt.legend(loc = 'lower right')

plt.subplot(3,1,3)
plt.plot(x2,y2,color = 'k',ls = '--',label = '10 Year',marker = '^')
plt.xlabel('Date',fontsize = 12)
plt.ylabel('Days',fontsize = 12)
plt.legend(loc = 'lower right')

plt.savefig('p_bj.png',dpi=600,format='png')
plt.show()