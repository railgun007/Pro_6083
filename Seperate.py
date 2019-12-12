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

def judge(a,b):
    if a>b :
        return 1
    if a<b :
        return -1
    if a==b:
        return 0

def calculate_revenue(list1,list2,holding_period):
    initial=100000
    list1=np.array(list1)
    list2=np.array(list2)
    days_to_long=0
    days_to_short=0
    money=100000
    money_trend=[money]
    for index, value in enumerate(list1):
        #print(money)
        if value==1:
            try:
                days_to_long+=1
                number_to_buy=int(money/list2[index-1])
                #print("long",money,list2[index+holding_period-1],list2[index-1])
                money+= number_to_buy*(list2[index-1+holding_period]-list2[index-1])

            except IndexError:
                continue
        if value==-1:
            try:
                days_to_short +=1
                number_to_sell = int(money / list2[index-1])
                #print("short",money,  list2[index-1],list2[index + holding_period-1])
                money += number_to_sell * (list2[index-1]-list2[index-1+holding_period])

            except IndexError:
                continue
        money_trend.append(money)
    #plt.plot(money_trend)
    #plt.show()
    #print(days_to_long,days_to_short)
    return money

def best_period_r(New_close, average_filtered_derivative, confidence):
    sum=0
    count=0
    # lis=[0]*40
    for i in range(1,30):
        action_list = strategy_r(average_filtered_derivative, 0, i,confidence)
        final_revenue=calculate_revenue(action_list, New_close, i)
        if final_revenue>sum:
                sum=final_revenue
                count=i
    print("Best holding period:"+str(count)+"return:"+str(sum))
    return count

def best_period(New_close, average_filtered_derivative, confidence):
    sum=0
    count=0
    # lis=[0]*40
    for i in range(1,30):
        action_list = strategy(average_filtered_derivative, 0, i,confidence)
        final_revenue=calculate_revenue(action_list, New_close, i)
        if final_revenue>sum:
                sum=final_revenue
                count=i
    print("Best holding period:"+str(count)+"return:"+str(sum))
    return count

def strategy_r( average_filtered_derivative,start_date,period,confidence):
    size = len(confidence)
    buy_or_sell_list = [0] * (size)
    for i in range(start_date, size - period, period):
        buy_or_sell_list[i] = buy_or_sell_r(average_filtered_derivative[i], confidence[i])
    return buy_or_sell_list

def strategy( average_filtered_derivative,start_date,period,confidence):
    size = len(confidence)
    buy_or_sell_list = [0] * (size)
    for i in range(start_date, size - period, period):
        buy_or_sell_list[i] = buy_or_sell(average_filtered_derivative[i], confidence[i])
    return buy_or_sell_list


def buy_or_sell_r(value, confidence):
    if value> 0 and confidence != 0:
        x = -1
    elif value < 0 and confidence !=0:
        x = 1
    else:
        x = 0
    return x

def buy_or_sell(value, confidence):
    if value> 0 and confidence != 0:
        x = 1
    elif value < 0 and confidence !=0:
        x = -1
    else:
        x = 0
    return x
# decide whether to buy or sell or no action



def strategy3_r(average_filtered_derivative,start_date,period):
    # judge whether there is a trend, n is the moving average day, eg. 10
    size = len(average_filtered_derivative)
    buy_or_sell_list = [0] * (size)
    for i in range(start_date, size - period, period):
        if average_filtered_derivative[i]>0:
         buy_or_sell_list[i] = -1
        elif average_filtered_derivative[i]<0:
            buy_or_sell_list[i] = 1
    return buy_or_sell_list

def strategy3(average_filtered_derivative,start_date,period):
    # judge whether there is a trend, n is the moving average day, eg. 10
    size = len(average_filtered_derivative)
    buy_or_sell_list = [0] * (size)
    for i in range(start_date, size - period, period):
        if average_filtered_derivative[i]>0:
         buy_or_sell_list[i] = 1
        elif average_filtered_derivative[i]<0:
            buy_or_sell_list[i] = -1
    return buy_or_sell_list


def best_period3_r(New_close, average_filtered_derivative):
    sum=0
    count=0
    # lis=[0]*40
    for i in range(1,30):
        action_list = strategy3_r(average_filtered_derivative,0, i)
        final_revenue=calculate_revenue(action_list, New_close, i)
        if final_revenue>sum:
            sum=final_revenue
            count=i
    print("Best holding period:"+str(count)+"return:"+str(sum))
    return count

def best_period3(New_close, average_filtered_derivative):
    sum=0
    count=0
    # lis=[0]*40
    for i in range(1,30):
        action_list = strategy3(average_filtered_derivative,0, i)
        final_revenue=calculate_revenue(action_list, New_close, i)
        if final_revenue>sum:
            sum=final_revenue
            count=i
    print("Best holding period:"+str(count)+"return:"+str(sum))
    return count

#Download df
start_date = '2015-01-05'
end_date = '2019-12-31'

# Use pandas_reader.data.DataReader to load the desired data.

# df = yf.download('000001.SS', start_date, end_date)
# df = yf.download('SPY', start_date, end_date)
df = yf.download('^GSPC', start_date, end_date)

close_2015 = np.log(df['Adj Close'].loc["2015-01-04":"2015-12-31"])
close_2016_Actual=np.array(df['Adj Close'].loc["2016-01-04":"2016-12-31"])
close_2016 = np.log(df['Adj Close'].loc["2016-01-04":"2016-12-31"])
close_2017_Actual=np.array(df['Adj Close'].loc["2017-01-04":"2017-12-31"])
close_2017 = np.log(df['Adj Close'].loc["2017-01-04":"2017-12-31"])
close_2018_Actual=np.array(df['Adj Close'].loc["2018-01-04":"2018-12-31"])
close_2018 = np.log(df['Adj Close'].loc["2018-01-04":"2018-12-31"])
close_2019_Actual=np.array(df['Adj Close'].loc["2019-01-04":"2019-12-31"])
close_2019=np.log(close_2019_Actual)



# #*******************************     For the first
start=close_2015
end=close_2016
close_all=np.hstack((start,end))
close_end_Actual=close_2016_Actual
lmd=10
inter=10
filtered = l1tf(start, lmd)
filter_derivative = [(filtered[i] - filtered[i - 1]) for i in range(1, len(filtered))]
filter_derivative.insert(0, filter_derivative[0])
average_filtered_derivative = [np.mean(filter_derivative[(i - inter):i]) for i in range(inter, len(filter_derivative))]
temp_lis = np.array([0] * inter)
average_filtered_derivative = np.hstack((temp_lis,average_filtered_derivative))

size_start=len(start)
size_end=len(end)
temp_lis2=np.array([0]*size_end)
average_filtered_derivative=np.hstack((average_filtered_derivative,temp_lis2))

for i in range(0,size_end):
    filtered_updating = l1tf(close_all[0:size_start+i+1], lmd)
    filter_derivative = [(filtered_updating[i] - filtered_updating[i - 1]) for i in range(1, len(filtered_updating))]
    target=np.mean(filter_derivative[-inter-1:-1])
    average_filtered_derivative[size_start+i]=target

#updating zt
n=60
size=len(close_all)
st = [0] * size
for date in range(n, size):
    total = 0
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            value = judge(close_all[date - i], close_all[date - j])
            total = total + value
        st[date] = total

std = math.sqrt((n * (n - 1) * (2 * n + 5)) / 18)
zt = list(map(lambda num: num / std, st))
confidence=[0]*size
for i in range(len(zt)):
    if(zt[i]>1.96) :confidence[i]=1
    elif (zt[i]<-1.96):confidence[i]=-1

average_filtered_derivative_end=average_filtered_derivative[-len(end):]
confidence_end=confidence[-len(end):]

bp11=best_period(close_end_Actual, average_filtered_derivative_end, confidence_end)
bp11_r=best_period_r(close_end_Actual, average_filtered_derivative_end, confidence_end)

bp31=best_period3(close_end_Actual, average_filtered_derivative_end)
bp31_r=best_period3_r(close_end_Actual, average_filtered_derivative_end)
#print(bp11,bp11_r,bp31,bp31_r)

# print(100000/close_2018_Actual[0]*close_2018_Actual[-1])


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Do for 2nd
start=close_2016
end=close_2017
close_all=np.hstack((start,end))
close_end_Actual=close_2017_Actual
filtered = l1tf(start, lmd)
filter_derivative = [(filtered[i] - filtered[i - 1]) for i in range(1, len(filtered))]
filter_derivative.insert(0, filter_derivative[0])
average_filtered_derivative = [np.mean(filter_derivative[(i - inter):i]) for i in range(inter, len(filter_derivative))]
temp_lis = np.array([0] * inter)
average_filtered_derivative = np.hstack((temp_lis,average_filtered_derivative))

size_start=len(start)
size_end=len(end)
temp_lis2=np.array([0]*size_end)
average_filtered_derivative=np.hstack((average_filtered_derivative,temp_lis2))

for i in range(0,size_end):
    filtered_updating = l1tf(close_all[0:size_start+i+1], lmd)
    filter_derivative = [(filtered_updating[i] - filtered_updating[i - 1]) for i in range(1, len(filtered_updating))]
    target=np.mean(filter_derivative[-inter-1:-1])
    average_filtered_derivative[size_start+i]=target

#updating zt
n=60
size=len(close_all)
st = [0] * size
for date in range(n, size):
    total = 0
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            value = judge(close_all[date - i], close_all[date - j])
            total = total + value
        st[date] = total
       # st_normal = list(map(lambda num: num * 2 / (n * n + 1), st))

std = math.sqrt((n * (n - 1) * (2 * n + 5)) / 18)
zt = list(map(lambda num: num / std, st))
confidence=[0]*size
for i in range(len(zt)):
    if(zt[i]>1.96) :confidence[i]=1
    elif (zt[i]<-1.96):confidence[i]=-1

average_filtered_derivative_end=average_filtered_derivative[-len(end):]
confidence_end=confidence[-len(end):]

bp12=best_period(close_end_Actual, average_filtered_derivative_end, confidence_end)
bp12_r=best_period_r(close_end_Actual, average_filtered_derivative_end, confidence_end)

bp32=best_period3(close_end_Actual, average_filtered_derivative_end)
bp32_r=best_period3_r(close_end_Actual, average_filtered_derivative_end)

#print(bp12,bp12_r,bp32,bp32_r)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Do for 3rd
start=close_2017
end=close_2018
close_all=np.hstack((start,end))
close_end_Actual=close_2018_Actual
filtered = l1tf(start, lmd)
filter_derivative = [(filtered[i] - filtered[i - 1]) for i in range(1, len(filtered))]
filter_derivative.insert(0, filter_derivative[0])
average_filtered_derivative = [np.mean(filter_derivative[(i - inter):i]) for i in range(inter, len(filter_derivative))]
temp_lis = np.array([0] * inter)
average_filtered_derivative = np.hstack((temp_lis,average_filtered_derivative))

size_start=len(start)
size_end=len(end)
temp_lis2=np.array([0]*size_end)
average_filtered_derivative=np.hstack((average_filtered_derivative,temp_lis2))

for i in range(0,size_end):
    filtered_updating = l1tf(close_all[0:size_start+i+1], lmd)
    filter_derivative = [(filtered_updating[i] - filtered_updating[i - 1]) for i in range(1, len(filtered_updating))]
    target=np.mean(filter_derivative[-inter-1:-1])
    average_filtered_derivative[size_start+i]=target

#updating zt
n=60
size=len(close_all)
st = [0] * size
for date in range(n, size):
    total = 0
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            value = judge(close_all[date - i], close_all[date - j])
            total = total + value
        st[date] = total
       # st_normal = list(map(lambda num: num * 2 / (n * n + 1), st))

std = math.sqrt((n * (n - 1) * (2 * n + 5)) / 18)
zt = list(map(lambda num: num / std, st))
confidence=[0]*size
for i in range(len(zt)):
    if(zt[i]>1.96) :confidence[i]=1
    elif (zt[i]<-1.96):confidence[i]=-1

average_filtered_derivative_end=average_filtered_derivative[-len(end):]
confidence_end=confidence[-len(end):]

bp13=best_period(close_end_Actual, average_filtered_derivative_end, confidence_end)
bp13_r=best_period_r(close_end_Actual, average_filtered_derivative_end, confidence_end)

bp33=best_period3(close_end_Actual, average_filtered_derivative_end)
bp33_r=best_period3_r(close_end_Actual, average_filtered_derivative_end)
# print(bp13,bp13_r,bp33,bp33_r)

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Do for 4th
start=close_2018
end=close_2019
close_all=np.hstack((start,end))
close_end_Actual=close_2019_Actual
filtered = l1tf(start, lmd)
filter_derivative = [(filtered[i] - filtered[i - 1]) for i in range(1, len(filtered))]
filter_derivative.insert(0, filter_derivative[0])
average_filtered_derivative = [np.mean(filter_derivative[(i - inter):i]) for i in range(inter, len(filter_derivative))]
temp_lis = np.array([0] * inter)
average_filtered_derivative = np.hstack((temp_lis,average_filtered_derivative))

size_start=len(start)
size_end=len(end)
temp_lis2=np.array([0]*size_end)
average_filtered_derivative=np.hstack((average_filtered_derivative,temp_lis2))

for i in range(0,size_end):
    filtered_updating = l1tf(close_all[0:size_start+i+1], lmd)
    filter_derivative = [(filtered_updating[i] - filtered_updating[i - 1]) for i in range(1, len(filtered_updating))]
    target=np.mean(filter_derivative[-inter-1:-1])
    average_filtered_derivative[size_start+i]=target

#updating zt
n=60
size=len(close_all)
st = [0] * size
for date in range(n, size):
    total = 0
    for i in range(0, n - 1):
        for j in range(i + 1, n):
            value = judge(close_all[date - i], close_all[date - j])
            total = total + value
        st[date] = total
       # st_normal = list(map(lambda num: num * 2 / (n * n + 1), st))

std = math.sqrt((n * (n - 1) * (2 * n + 5)) / 18)
zt = list(map(lambda num: num / std, st))
confidence=[0]*size
for i in range(len(zt)):
    if(zt[i]>1.96) :confidence[i]=1
    elif (zt[i]<-1.96):confidence[i]=-1

average_filtered_derivative_end=average_filtered_derivative[-len(end):]
confidence_end=confidence[-len(end):]


bp1_r=int(np.mean([bp11_r,bp12_r,bp13_r]))
bp3_r=int(np.mean([bp31_r,bp32_r,bp33_r]))
Ac_list1_r=strategy_r(average_filtered_derivative_end, 0,bp1_r, confidence_end)
Ac_list3_r=strategy3_r(average_filtered_derivative_end, 0,bp3_r)

bp1=int(np.mean([bp11,bp12,bp13]))
bp3=int(np.mean([bp31,bp32,bp33]))
Ac_list1=strategy(average_filtered_derivative_end, 0,bp1, confidence_end)
Ac_list3=strategy3(average_filtered_derivative_end, 0,bp3)


print("######################################################################")

print(bp11_r,bp12_r,bp13_r)
print(bp31_r,bp32_r,bp33_r)
print(bp11,bp12,bp13)
print(bp31,bp32,bp33)
print(bp1_r,bp3_r,bp1,bp3)

print(calculate_revenue(Ac_list1_r,close_2019_Actual,bp1_r))
print(calculate_revenue(Ac_list3_r,close_2019_Actual,bp3_r))
print(calculate_revenue(Ac_list1,close_2019_Actual,bp1))
print(calculate_revenue(Ac_list3,close_2019_Actual,bp3))
print(100000/close_2019_Actual[0]*close_2019_Actual[-1])
print(100000/close_2019_Actual[-1]*close_2019_Actual[0])

#
# print(calculate_revenue(Ac_list3_r,close_2019_Actual,59))
