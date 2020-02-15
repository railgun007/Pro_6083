from pandas_wrapper import l1tf, hp, l1ctf, l1tccf
import csv
from numpy import linspace, exp
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import math
from mpl_toolkits import axisartist
import random
import gc
from pandas_datareader import data
import yfinance as yf


########################################################
#simulation
random.seed( 10 )
v=0.1
position=0
trend=[position]
steps = 1000
for i in range(steps):
    RandVal=random.randint(1,101)
    if(RandVal<=90):#p=0.9
        position+=v
        trend.append(position)
    else:
        v=0.5*(random.uniform(0,1)-0.5)
        position+=v
        trend.append(position)
fig = plt.figure()
plt.subplot(321)
plt.title("Real Trend")
plt.plot(trend)
plt.grid()
#Simulate Yt
walk=[]
for i in range(steps):
    step = np.random.normal(0,2)
    walk.append(trend[i]+step)

plt.subplot(322)
plt.plot(walk)
plt.grid()
plt.title("Real Value")
walk=np.array(walk)
delta=5
#L1-T
filtered = l1tf(walk, delta)
plt.subplot(323)
plt.title("L1-T filter")
plt.plot(filtered)
plt.grid()
#L1_C
filtered = l1ctf(walk, delta)
plt.subplot(324)
plt.title("L1-C filter")
plt.plot(filtered)
plt.grid()
#L1-tc
filtered = l1tccf(walk,5,10)
plt.subplot(325)
plt.title("L1-TC filter")
plt.plot(filtered)
plt.grid()
#L2
filtered = hp(walk, 2000)
plt.subplot(326)
plt.title("L2 filter")
plt.plot(filtered)
plt.grid()
plt.show()


###############################################
#simulation

position=40
trend=[]
steps = 1000
for i in range(steps):
    RandVal=random.randint(1,1001)
    if(RandVal<=995):#p=0.995
        trend.append(position)
    else:
        position=50*(random.uniform(0,1)-0.5)
        trend.append(position)
fig = plt.figure()
plt.subplot(321)
plt.title("Real Trend")
plt.plot(trend)
plt.grid()
#Simulate Yt
walk=[]
for i in range(steps):
    step = np.random.normal(0,16)
    walk.append(trend[i]+step)

plt.subplot(322)
plt.title("Real Value")
plt.plot(walk)
plt.grid()

walk=np.array(walk)
delta=5

#L1-T
filtered = l1tf(walk, delta)
plt.subplot(323)
plt.title("L1-T filter")
plt.plot(filtered)
plt.grid()
#L1_C
filtered = l1ctf(walk, delta)
plt.subplot(324)
plt.title("L1-C filter")
plt.plot(filtered)
plt.grid()

#L1-tc
filtered = l1tccf(walk,5,10)
plt.subplot(325)
plt.title("L1-TC filter")
plt.plot(filtered)
plt.grid()

#L2
filtered = hp(walk, 2000)
plt.subplot(326)
plt.title("L2 filter")
plt.plot(filtered)
plt.grid()
plt.show()




#########################################################################
#Download df
#df = pd.read_csv('GSPC3.csv', skiprows=ski)  # read data
start_date = '2009-01-05'
end_date = '2018-12-31'
df = yf.download('^GSPC', start_date, end_date)
#df = yf.download('000001.SS', start_date, end_date)

#close_15to18=np.log(df['Adj Close'].loc["2015-01-04":"2018-12-31"])
derivative = [(df.Close[i] - df.Close[i - 1]) for i in range(1, len(df))]
derivative.insert(0, derivative[0])
df["derivative"] = derivative
df["Actual_Close"]=df["Close"]
df["Close"] = np.log(df["Close"])




######################################################
#Cross validation
#Cv!!!!!!!!!!!!!!!!!!
def cross_validation(df,lower_bound,higher_bound,step_length,leng_of_training,leng_of_test,type):
    Interval=250
    error = []
    for delta in np.linspace(lower_bound,higher_bound,step_length):
        oneerror = 0
        for count in range(1,10):
            x1 = df["Close"][(Interval * (count - 1)):(Interval * (count - 1) + leng_of_training)]
            if type=="L1T":  filtered = l1tf(x1, (delta))
            elif type=="L1C": filtered = l1ctf(x1, (delta))
            else :filtered = hp(x1, (delta))
            estimate = filtered[(leng_of_training-leng_of_test): leng_of_training]
            y1 = list(df["Close"][(Interval * (count - 1) + leng_of_training):(Interval * (count - 1) + (leng_of_training+leng_of_test))])
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
leng_of_training=90
leng_of_test=30
best_lamda_L1T=cross_validation(df,0,10,20,leng_of_training,leng_of_test,"L1T")
print("Best lamda for L1-T filter:"+str(best_lamda_L1T))#1578947368421053

best_lamda_L1C=cross_validation(df,0,2,10,leng_of_training,leng_of_test,"L1C")
print("Best lamda for L1-C filter:"+str(best_lamda_L1C)) #:0.8888888888888888

best_lamda_L2=cross_validation(df,0,2000,20,leng_of_training,leng_of_test,"L2")
print("Best lamda for L2 filter:"+str(best_lamda_L2))#736.8421052631578

###############################################################################################################
# use L1-T to get filtered trend
filtered = l1tf(df["Close"], best_lamda_L1T)
filtered2 = l1ctf(df["Close"], best_lamda_L1C)
gc.collect()
filtered3 = l1tccf(df["Close"], best_lamda_L1T,best_lamda_L1C)
filtered4 = hp(df["Close"], best_lamda_L2)

plt.subplot(221)
filtered.plot(figsize=(15, 8), title='L1-T filter', fontsize=14,linewidth = '4')
plt.subplot(222)
filtered2.plot(figsize=(15, 8), title='L1-C filter', fontsize=14,linewidth = '4')
plt.subplot(223)
filtered3.plot(figsize=(15, 8), title='L1-TC filter', fontsize=14,linewidth = '4')
plt.subplot(224)
filtered4.plot(figsize=(15, 8), title='L2 filter', fontsize=14,linewidth = '4')
plt.show()
#show trend with very large lambda

filter_derivative = [(filtered[i] - filtered[i - 1]) for i in range(1, len(filtered))]
filter_derivative.insert(0, filter_derivative[0])
df["filtered_derivative"] = filter_derivative
average_filtered_derivative = [np.mean(df.filtered_derivative[i - 10:i]) for i in range(10, len(df))]
temp_lis = [0] * 10
average_filtered_derivative = temp_lis + average_filtered_derivative

def judge(a,b):
    if a>b :
        return 1
    if a<b :
        return -1
    if a==b:
        return 0

def trend_detection(n,dataset,average_filtered_derivative):
    # judge whether there is a trend
    st = [0] * (len(dataset)-n)
    for date in range(n, len(dataset)):
        total = 0
        for i in range(0, n - 1):
            for j in range(i + 1, n):
                value = judge(dataset["Close"][date - i], dataset["Close"][date - j])
                total = total + value
        st[date-n] = total
    average_filtered_derivative=average_filtered_derivative[n:]
    st_normal = list(map(lambda num: num * 2 / (n * n + 1), st))
    plt.plot(st_normal)
    plt.title("n= "+str(n)+" days")
    plt.plot()
    plt.show()
    std = math.sqrt((n * (n - 1) * (2 * n + 5)) / 18)
    zt = list(map(lambda num: num / std, st))
    print("90% confidence interval is" +str(len(list(filter(lambda num: num > 1.645 or num < -1.645, zt))) / len(df)))
    print("95% confidence interval is" +str(len(list(filter(lambda num: num > 1.96 or num < -1.96, zt))) / len(df)))
    print("99% confidence interval is" +str(len(list(filter(lambda num: num > 2.33 or num < -2.33, zt))) / len(df)))
    fig = plt.figure(figsize=(8, 8))
    ax = axisartist.Subplot(fig, 111)
    fig.add_axes(ax)
    ax.axis["x"] = ax.new_floating_axis(0, 0)
    ax.axis["y"] = ax.new_floating_axis(1, 0)
    plt.xlabel("derivative")
    plt.ylabel("trend detection")
    plt.plot(st_normal, average_filtered_derivative, 'o')
    plt.title("Trend detection versus trend filtering" + " n= "+str(n))
    plt.show()
trend_detection(10,df,average_filtered_derivative)
#trend_detection(60,df,average_filtered_derivative)
#trend_detection(100,df,average_filtered_derivative)

############################################

total_result=[]
for threshold in range(0,4):
    Positive_result = []

    for i in range(61, len(filtered) - 20):
        if filtered[i-1] - filtered[i - 61] > 0.025*threshold:
            Positive_result.append(df.Close[i + 20] - df.Close[i])
            total_result.append(df.Close[i + 20] - df.Close[i])
    plt.subplot(2, 2, threshold + 1)
    hist, bin_edges = np.histogram(Positive_result)
    width = (bin_edges[1] - bin_edges[0]) * 0.8
    plt.bar(bin_edges[1:], hist / max(hist), width=width, color='#5B9BD5')
    cdf = np.cumsum(hist / sum(hist))
    plt.plot(bin_edges[1:], cdf, '-*', color='#ED7D31')
    plt.xlim([min(Positive_result), max(Positive_result)])
    plt.title("positive threshold =" +str(round(0.025*threshold*100,3)) +"percent")
    plt.ylim([0, 1])
    plt.grid()

print("the mean return is "+str(np.mean(total_result)))
plt.show()

total_result=[]
for threshold in range(0,4):
        Negative_result = []
        for i in range(61, len(filtered) - 20):
            if filtered[i-1] - filtered[i - 61] < -0.025*threshold:
                Negative_result.append(df.Close[i + 20] - df.Close[i])
                total_result.append(df.Close[i + 20] - df.Close[i])
        plt.subplot(2, 2, threshold + 1)
        hist, bin_edges = np.histogram(Negative_result)
        width = (bin_edges[1] - bin_edges[0]) * 0.8
        plt.bar(bin_edges[1:], hist / max(hist), width=width, color='#5B9BD5')
        cdf = np.cumsum(hist / sum(hist))
        plt.title("negative threshold =" + str(round(0.025 * threshold * 100, 3)) + "percent")
        plt.plot(bin_edges[1:], cdf, '-*', color='#ED7D31')
        plt.xlim([min(Negative_result), max(Negative_result)])
        plt.ylim([0, 1])
        plt.grid()

print("the mean return is "+str(np.mean(total_result)))
plt.show()


