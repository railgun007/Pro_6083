from pandas_datareader import data
import yfinance as yf
from statsmodels.tsa.holtwinters import ExponentialSmoothing, Holt
from scipy.interpolate import UnivariateSpline
from pandas_wrapper import l1tf, hp, l1ctf, l1tccf
import csv
from numpy import linspace, exp
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
from scipy import stats
# from Holt import additive,linear,multiplicative,newadditive
from mpl_toolkits import axisartist
import random
import gc
from  Seperate import *
start_date = '2018-01-02'
end_date = '2019-12-31'
df = yf.download('GC=F', start_date, end_date)
df['log Close']=np.log(df['Close'])
close_15to18=df['log Close'].loc["2019-01-02":"2019-04-30"]
close_2015 = np.log(df['Adj Close'].loc["2019-01-02":"2019-01-31"])
close_2016_Actual=np.array(df['Adj Close'].loc["2019-02-02":"2019-02-28"])
close_2016 = np.log(close_2016_Actual)
close_2017_Actual=np.array(df['Adj Close'].loc["2019-03-02":"2019-03-31"])
close_2017 = np.log(close_2017_Actual)
close_2018_Actual=np.array(df['Adj Close'].loc["2019-04-02":"2019-04-30"])
close_2018 =np.log(close_2018_Actual)
close_2019_Actual=np.array(df['Adj Close'].loc["2019-05-02":"2019-12-31"])
close_2019=np.log(close_2019_Actual)
time=[close_2015,close_2016_Actual,close_2017_Actual,close_2018_Actual,close_2019_Actual]
df.Close.plot()
plt.show()


def cross_validation_gold(df,lower_bound,higher_bound,step_length,leng_of_training,leng_of_test,type):
    Interval=10
    error = []
    for delta in np.linspace(lower_bound,higher_bound,step_length):
        oneerror = 0
        for count in range(1, 9):
            x1 = df[(Interval * (count - 1)):(Interval * (count - 1) + leng_of_training)]
            if type=="L1T":  filtered = l1tf(x1, (delta))
            elif type=="L1C": filtered = l1ctf(x1, (delta))
            else :filtered = hp(x1, (delta))
            estimate = filtered[(leng_of_training-leng_of_test): leng_of_training]
            y1 = list(df[(Interval * (count - 1) + leng_of_training):(Interval * (count - 1) + (leng_of_training+leng_of_test))])
            newerror = [pow(y1[i] - estimate[i], 2) for i in range(0, len(y1))]
            oneerror = oneerror + sum(newerror)
        error.append(oneerror)
    np.array(error)
    plt.plot(np.linspace(lower_bound,higher_bound,step_length),error)
    plt.legend([type],fontsize=20)
    plt.xlabel("lambda")
    plt.ylabel('Error')
    plt.title('CrossValidation')
    plt.show()
    return np.linspace(lower_bound,higher_bound,step_length)[error.index(min(error))]


leng_of_training=14
leng_of_test=5
best_lamda_L1T=cross_validation_gold(close_15to18,0,10,20,leng_of_training,leng_of_test,"L1T")
print("Best lamda for L1-T filter:"+str(best_lamda_L1T))#2.631578947368421

best_lamda_L1C=cross_validation_gold(close_15to18,0,10,20,leng_of_training,leng_of_test,"L1C")
print("Best lamda for L1-C filter:"+str(best_lamda_L1C)) #1.0526315789473684

best_lamda_L2=cross_validation_gold(close_15to18,0,2000,30,leng_of_training,leng_of_test,"L2")
print("Best lamda for L2 filter:"+str(best_lamda_L2))#282.7586206896552


alltype=["L1T","L1C","L2","L1TC"]
allbestlambda=[best_lamda_L1T,best_lamda_L1C,best_lamda_L2]

result1=calculate_return(df,alltype[0],allbestlambda[0],time)
result2=calculate_return(df,alltype[1],allbestlambda[1],time)
result3=calculate_return(df,alltype[2],allbestlambda[2],time)
result4=calculate_return(df,alltype[3],allbestlambda[0],time,allbestlambda[1])

newfilter1=np.array(result1[3])
newfilter2=np.array(result2[3])
newfilter3=np.array(result3[3])
newfilter4=np.array(result4[3])

close_15to17=df['log Close'].loc["2019-01-01":"2019-04-04"]
filtered = l1tf(close_15to17, best_lamda_L1T)
filtered2 = l1ctf(close_15to17, best_lamda_L1C)
gc.collect()
filtered3 = hp(close_15to17, best_lamda_L2)
filtered4 = l1tccf(close_15to17, best_lamda_L1T,best_lamda_L1C)


df["trend_L1T"]=np.hstack((filtered,newfilter1))
df["trend_L1C"]=np.hstack((filtered2,newfilter2))
df["trend_L2"]=np.hstack((filtered3,newfilter3))
df["trend_L1TC"]=np.hstack((filtered4,newfilter4))

plt.subplot(221)
plt.plot(df["trend_L1T"],linewidth = '2.5')
plt.plot(df["log Close"])
#plt.show()
plt.subplot(222)
plt.plot(df["trend_L1C"],linewidth = '2.5')
plt.plot(df["log Close"])
#plt.show()
plt.subplot(223)
plt.plot(df["trend_L2"],linewidth = '2.5')
plt.plot(df["log Close"])
#plt.show()
plt.subplot(224)
plt.plot(df["trend_L1TC"],linewidth = '2.5')
plt.plot(df["log Close"])
plt.show()

x = np.arange(4)
allresult=np.array([result1[0],result2[0],result3[0],result4[0]])
allresult2=np.array([result1[1],result2[1],result3[1],result4[1]])

y=np.array([4])
plt.bar(y,result1[2],label="benchmark",color="red")
plt.xticks(y,["benchmark"])

total_width, n = 0.8, 2
width = total_width / n
x = x - (total_width - width) / 2
plt.bar(x,allresult,width=width,label="method 1",color='orange')
plt.bar(x+width,np.array(allresult2),width=width,color= 'blue',label="method 2")
plt.yticks(range(0,170000,10000))
plt.xticks(x, (["L1T","L1C","L2","L1TC"]))
plt.legend()
plt.show()

