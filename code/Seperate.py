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
import pywt

def wave(arr):
    return_ = np.diff(arr)
    # return_=return_[1:]
    # close2 / close

    # print(pywt.wavelist())

    #Do the Wavelet Transform for the return and inverse transformation
    #return_ is a dataframe series
    method='haar'
    mode_="soft"
    (ca, cd)=pywt.dwt(return_, method)
    cat = pywt.threshold(ca, 0.3*np.std(ca), mode=mode_)
    cdt = pywt.threshold(cd, 0.3*np.std(cd), mode=mode_)
    tx = pywt.idwt(cat, cdt, method,"smooth")
    # tx=pd.DataFrame(tx,index=return_.index)

    #Get back to the Stock price using denoising wavelet transform
    start_price=arr[0]
    # txx=tx.iloc[:,0]
    # txx=np.exp(tx)
    # txx=np.array(txx)
    temp=np.array([start_price])
    np.hstack((temp,tx))
    txx=np.cumsum(tx)
    # txx = pd.Series(txx, index=arr.index)
    # txx=pd.DataFrame(txx,index=return_.index)
    return txx


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


def strategy( average_filtered_derivative,start_date,period,confidence):
    size = len(confidence)
    buy_or_sell_list = [0] * (size)
    for i in range(start_date, size - period, period):
        buy_or_sell_list[i] = buy_or_sell(average_filtered_derivative[i], confidence[i])
    return buy_or_sell_list



def buy_or_sell(value, confidence):
    if value> 0 and confidence != 0:
        x = 1
    elif value < 0 and confidence !=0:
        x = -1
    else:
        x = 0
    return x
# decide whether to buy or sell or no action



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


def cross_validation(df,lower_bound,higher_bound,step_length,leng_of_training,leng_of_test,type):
    Interval=50
    error = []
    for delta in np.linspace(lower_bound,higher_bound,step_length):
        oneerror = 0
        for count in range(1, 19):
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



def calculate_return(df,TYPE,best_lambda1,time,best_lambda2=0):
    close_2015 = time[0]
    close_2016_Actual = time[1]
    close_2016 = np.log(close_2016_Actual)
    close_2017_Actual = time[2]
    close_2017 = np.log(close_2017_Actual)
    close_2018_Actual = time[3]
    close_2018 = np.log(close_2018_Actual)
    close_2019_Actual = time[4]
    close_2019 = np.log(close_2019_Actual)
    start = close_2015
    end = close_2016
    close_all = np.hstack((start, end))
    close_end_Actual = close_2016_Actual
    inter = 10
    if TYPE == "L1T":
        filtered = l1tf(start, best_lambda1)
    elif TYPE == "L1C":
        filtered = l1ctf(start, best_lambda1)
    elif TYPE == "L1TC":
        filtered = l1tccf(start, best_lambda1, best_lambda2)
    elif TYPE == "Wave":
        filtered = wave(start)
    else:
        filtered = hp(start, best_lambda1)
    filter_derivative = [(filtered[i] - filtered[i - 1]) for i in range(1, len(filtered))]
    filter_derivative.insert(0, filter_derivative[0])
    average_filtered_derivative = [np.mean(filter_derivative[(i - inter):i]) for i in
                                   range(inter, len(filter_derivative))]
    temp_lis = np.array([0] * inter)
    average_filtered_derivative = np.hstack((temp_lis, average_filtered_derivative))

    size_start = len(start)
    size_end = len(end)
    temp_lis2 = np.array([0] * size_end)
    average_filtered_derivative = np.hstack((average_filtered_derivative, temp_lis2))

    for i in range(0, size_end):
        if TYPE == "L1T":
            filtered_updating = l1tf(close_all[0:size_start + i + 1], best_lambda1)
        elif TYPE == "L1C":
            filtered_updating = l1ctf(close_all[0:size_start + i + 1], best_lambda1)
        elif TYPE == "L1TC":
            filtered_updating = l1tccf(close_all[0:size_start + i + 1], best_lambda1, best_lambda2)
        elif TYPE== "Wave":
            filtered_updating=wave(close_all[0:size_start + i + 1])
        else:
            filtered_updating = hp(close_all[0:size_start + i + 1], best_lambda1)
        filter_derivative = [(filtered_updating[i] - filtered_updating[i - 1]) for i in
                             range(1, len(filtered_updating))]
        target = np.mean(filter_derivative[-inter - 1:-1])
        try:
            average_filtered_derivative[size_start + i] = target
        except :
            continue

    # updating zt
    n = 60
    size = len(close_all)
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
    confidence = [0] * size
    for i in range(len(zt)):
        if (zt[i] > 1.96):
            confidence[i] = 1
        elif (zt[i] < -1.96):
            confidence[i] = -1

    average_filtered_derivative_end = average_filtered_derivative[-len(end):]
    confidence_end = confidence[-len(end):]
    bp11 = best_period(close_end_Actual, average_filtered_derivative_end, confidence_end)
    bp31 = best_period3(close_end_Actual, average_filtered_derivative_end)
    # print(bp11,bp11_r,bp31,bp31_r)

    # print(100000/close_2018_Actual[0]*close_2018_Actual[-1])

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Do for 2nd
    start = close_2016
    end = close_2017
    close_all = np.hstack((start, end))
    close_end_Actual = close_2017_Actual
    if TYPE == "L1T":
        filtered = l1tf(start, best_lambda1)
    elif TYPE == "L1C":
        filtered = l1ctf(start, best_lambda1)
    elif TYPE == "L1TC":
        filtered = l1tccf(start, best_lambda1, best_lambda2)
    elif TYPE == "Wave":
        filtered = wave(start)
    else:
        filtered = hp(start, best_lambda1)
    filter_derivative = [(filtered[i] - filtered[i - 1]) for i in range(1, len(filtered))]
    filter_derivative.insert(0, filter_derivative[0])
    average_filtered_derivative = [np.mean(filter_derivative[(i - inter):i]) for i in
                                   range(inter, len(filter_derivative))]
    temp_lis = np.array([0] * inter)
    average_filtered_derivative = np.hstack((temp_lis, average_filtered_derivative))

    size_start = len(start)
    size_end = len(end)
    temp_lis2 = np.array([0] * size_end)
    average_filtered_derivative = np.hstack((average_filtered_derivative, temp_lis2))

    for i in range(0, size_end):
        if TYPE == "L1T":
            filtered_updating = l1tf(close_all[0:size_start + i + 1], best_lambda1)
        elif TYPE == "L1C":
            filtered_updating = l1ctf(close_all[0:size_start + i + 1], best_lambda1)
        elif TYPE == "L1TC":
            filtered_updating = l1tccf(close_all[0:size_start + i + 1], best_lambda1, best_lambda2)
        elif TYPE== "Wave":
            filtered_updating=wave(close_all[0:size_start + i + 1])
        else:
            filtered_updating = hp(close_all[0:size_start + i + 1], best_lambda1)
        filter_derivative = [(filtered_updating[i] - filtered_updating[i - 1]) for i in
                             range(1, len(filtered_updating))]
        target = np.mean(filter_derivative[-inter - 1:-1])
        try:
            average_filtered_derivative[size_start + i] = target
        except :
            continue

    # updating zt
    n = 60
    size = len(close_all)
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
    confidence = [0] * size
    for i in range(len(zt)):
        if (zt[i] > 1.96):
            confidence[i] = 1
        elif (zt[i] < -1.96):
            confidence[i] = -1

    average_filtered_derivative_end = average_filtered_derivative[-len(end):]
    confidence_end = confidence[-len(end):]

    bp12 = best_period(close_end_Actual, average_filtered_derivative_end, confidence_end)
    bp32 = best_period3(close_end_Actual, average_filtered_derivative_end)

    # print(bp12,bp12_r,bp32,bp32_r)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Do for 3rd
    start = close_2017
    end = close_2018
    close_all = np.hstack((start, end))
    close_end_Actual = close_2018_Actual
    if TYPE == "L1T":
        filtered = l1tf(start, best_lambda1)
    elif TYPE == "L1C":
        filtered = l1ctf(start, best_lambda1)
    elif TYPE == "L1TC":
        filtered = l1tccf(start, best_lambda1, best_lambda2)
    elif TYPE == "Wave":
        filtered = wave(start)
    else:
        filtered = hp(start, best_lambda1)
    filter_derivative = [(filtered[i] - filtered[i - 1]) for i in range(1, len(filtered))]
    filter_derivative.insert(0, filter_derivative[0])
    average_filtered_derivative = [np.mean(filter_derivative[(i - inter):i]) for i in
                                   range(inter, len(filter_derivative))]
    temp_lis = np.array([0] * inter)
    average_filtered_derivative = np.hstack((temp_lis, average_filtered_derivative))

    size_start = len(start)
    size_end = len(end)
    temp_lis2 = np.array([0] * size_end)
    average_filtered_derivative = np.hstack((average_filtered_derivative, temp_lis2))

    for i in range(0, size_end):
        if TYPE == "L1T":
            filtered_updating = l1tf(close_all[0:size_start + i + 1], best_lambda1)
        elif TYPE == "L1C":
            filtered_updating = l1ctf(close_all[0:size_start + i + 1], best_lambda1)
        elif TYPE == "L1TC":
            filtered_updating = l1tccf(close_all[0:size_start + i + 1], best_lambda1, best_lambda2)
        elif TYPE == "Wave":
            filtered_updating=wave(close_all[0:size_start + i + 1])
        else:
            filtered_updating = hp(close_all[0:size_start + i + 1], best_lambda1)
        filter_derivative = [(filtered_updating[i] - filtered_updating[i - 1]) for i in
                             range(1, len(filtered_updating))]
        target = np.mean(filter_derivative[-inter - 1:-1])
        try:
            average_filtered_derivative[size_start + i] = target
        except :
            continue

    # updating zt
    n = 60
    size = len(close_all)
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
    confidence = [0] * size
    for i in range(len(zt)):
        if (zt[i] > 1.96):
            confidence[i] = 1
        elif (zt[i] < -1.96):
            confidence[i] = -1

    average_filtered_derivative_end = average_filtered_derivative[-len(end):]
    confidence_end = confidence[-len(end):]

    bp13 = best_period(close_end_Actual, average_filtered_derivative_end, confidence_end)
    bp33 = best_period3(close_end_Actual, average_filtered_derivative_end)
    # print(bp13,bp13_r,bp33,bp33_r)

    # %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Do for 4th
    start = close_2018
    end = close_2019
    close_all = np.hstack((start, end))
    close_end_Actual = close_2019_Actual
    if TYPE == "L1T":
        filtered = l1tf(start, best_lambda1)
    elif TYPE == "L1C":
        filtered = l1ctf(start, best_lambda1)
    elif TYPE == "L1TC":
        filtered = l1tccf(start, best_lambda1, best_lambda2)
    elif TYPE == "Wave":
        filtered = wave(start)
    else:
        filtered = hp(start, best_lambda1)
    filter_derivative = [(filtered[i] - filtered[i - 1]) for i in range(1, len(filtered))]
    filter_derivative.insert(0, filter_derivative[0])
    average_filtered_derivative = [np.mean(filter_derivative[(i - inter):i]) for i in
                                   range(inter, len(filter_derivative))]
    temp_lis = np.array([0] * inter)
    average_filtered_derivative = np.hstack((temp_lis, average_filtered_derivative))

    size_start = len(start)
    size_end = len(end)
    temp_lis2 = np.array([0] * size_end)
    average_filtered_derivative = np.hstack((average_filtered_derivative, temp_lis2))

    for i in range(0, size_end):
        if TYPE == "L1T":
            filtered_updating = l1tf(close_all[0:size_start + i + 1], best_lambda1)
        elif TYPE == "L1C":
            filtered_updating = l1ctf(close_all[0:size_start + i + 1], best_lambda1)
        elif TYPE == "L1TC":
            filtered_updating = l1tccf(close_all[0:size_start + i + 1], best_lambda1, best_lambda2)
        elif TYPE == "Wave":
            filtered_updating=wave(close_all[0:size_start + i + 1])
        else:
            filtered_updating = hp(close_all[0:size_start + i + 1], best_lambda1)
        filter_derivative = [(filtered_updating[i] - filtered_updating[i - 1]) for i in
                             range(1, len(filtered_updating))]
        target = np.mean(filter_derivative[-inter - 1:-1])
        try:
            average_filtered_derivative[size_start + i] = target
        except :
            continue

    # updating zt
    n = 60
    size = len(close_all)
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
    confidence = [0] * size
    for i in range(len(zt)):
        if (zt[i] > 1.96):
            confidence[i] = 1
        elif (zt[i] < -1.96):
            confidence[i] = -1

    average_filtered_derivative_end = average_filtered_derivative[-len(end):]
    confidence_end = confidence[-len(end):]

    bp1 = int(np.mean([bp11, bp12, bp13]))
    bp3 = int(np.mean([bp31, bp32, bp33]))
    Ac_list1 = strategy(average_filtered_derivative_end, 0, bp1, confidence_end)
    Ac_list3 = strategy3(average_filtered_derivative_end, 0, bp3)

    print("######################################################################")

    print(bp11, bp12, bp13)
    print(bp31, bp32, bp33)
    print(bp1, bp3)
    revenue1=calculate_revenue(Ac_list1, close_2019_Actual, bp1)
    revenue2=calculate_revenue(Ac_list3, close_2019_Actual, bp3)
    benchmark=100000 / close_2019_Actual[0] * close_2019_Actual[-1]
    print(calculate_revenue(Ac_list1, close_2019_Actual, bp1))
    print(calculate_revenue(Ac_list3, close_2019_Actual, bp3))
    print(100000 / close_2019_Actual[0] * close_2019_Actual[-1])
    return (revenue1,revenue2,benchmark,filtered_updating)


def main():
    # Download df
    start_date = '2015-01-05'
    end_date = '2019-12-31'

    # Use pandas_reader.data.DataReader to load the desired data.
    df = yf.download('000001.SS', start_date, end_date)
    # df = yf.download('^GSPC', start_date, end_date)
    # df = yf.download('GC=F', start_date, end_date)
    df['log Close'] = np.log(df['Close'])

    close_15to18 = df['log Close'].loc["2015-01-04":"2018-12-31"]

    close_2015 = df['log Close'].loc["2015-01-04":"2015-12-31"]
    close_2016_Actual = np.array(df['Adj Close'].loc["2016-01-04":"2016-12-31"])
    close_2016 = np.log(close_2016_Actual)
    close_2017_Actual = np.array(df['Adj Close'].loc["2017-01-04":"2017-12-31"])
    close_2017 = np.log(close_2017_Actual)
    close_2018_Actual = np.array(df['Adj Close'].loc["2018-01-04":"2018-12-31"])
    close_2018 = np.log(close_2018_Actual)
    close_2019_Actual = np.array(df['Adj Close'].loc["2019-01-04":"2019-12-31"])
    close_2019 = np.log(close_2019_Actual)
    leng_of_training = 80
    leng_of_test = 20
    best_lamda_L1T = cross_validation(close_15to18, 0, 10, 20, leng_of_training, leng_of_test, "L1T")
    print("Best lamda for L1-T filter:" + str(best_lamda_L1T))  # 2.631578947368421

    best_lamda_L1C = cross_validation(close_15to18, 0, 10, 20, leng_of_training, leng_of_test, "L1C")
    print("Best lamda for L1-C filter:" + str(best_lamda_L1C))  # 1.0526315789473684

    best_lamda_L2 = cross_validation(close_15to18, 50, 500, 30, leng_of_training, leng_of_test, "L2")
    print("Best lamda for L2 filter:" + str(best_lamda_L2))  # 282.7586206896552

    time = [close_2015, close_2016_Actual, close_2017_Actual, close_2018_Actual, close_2019_Actual]


    alltype=["L1T","L1C","L2","L1TC","Wave"]
    allbestlambda=[best_lamda_L1T,best_lamda_L1C,best_lamda_L2]

    result1=calculate_return(df,alltype[0],allbestlambda[0],time)
    result2=calculate_return(df,alltype[1],allbestlambda[1],time)
    result3=calculate_return(df,alltype[2],allbestlambda[2],time)
    result4=calculate_return(df,alltype[3],allbestlambda[0],time,allbestlambda[1])
    result5=calculate_return(df,alltype[4],0,time)

    #
    # newfilter1=np.array(result1[3])
    # newfilter2=np.array(result2[3])
    # newfilter3=np.array(result3[3])
    # newfilter4=np.array(result4[3])
    #
    # filtered = l1tf(close_15to17, best_lamda_L1T)
    # filtered2 = l1ctf(close_15to17, best_lamda_L1C)
    # gc.collect()
    # filtered3 = hp(close_15to17, best_lamda_L2)
    # filtered4 = l1tccf(close_15to17, best_lamda_L1T,best_lamda_L1C)
    #
    #
    #
    # df["trend_L1T"]=np.hstack((filtered,newfilter1))
    # df["trend_L1C"]=np.hstack((filtered2,newfilter2))
    # df["trend_L2"]=np.hstack((filtered3,newfilter3))
    # df["trend_L1TC"]=np.hstack((filtered4,newfilter4))
    #
    #
    # plt.plot(df["trend_L1T"])
    # plt.plot(df["Adj Close"])
    # plt.show()
    # plt.plot(df["trend_L1C"])
    # plt.plot(df["Adj Close"])
    # plt.show()
    # plt.plot(df["trend_L2"])
    # plt.plot(df["Adj Close"])
    # plt.show()
    # plt.plot(df["trend_L1TC"])
    # plt.plot(df["Adj Close"])
    # plt.show()

    x = np.arange(5)
    allresult=np.array([result1[0],result2[0],result3[0],result4[0],result5[0]])
    allresult2=np.array([result1[1],result2[1],result3[1],result4[1],result5[1]])

    y=np.array([5])
    plt.bar(y,result1[2],label="benchmark",color="red")
    plt.xticks(y,["benchmark"])

    total_width, n = 0.8, 2
    width = total_width / n
    x = x - (total_width - width) / 2
    plt.bar(x,allresult,width=width,label="method 1",color='orange')
    plt.bar(x+width,np.array(allresult2),width=width,color= 'blue',label="method 2")
    plt.yticks(range(0,170000,10000))
    plt.xticks(x, (["L1T","L1C","L2","L1TC","Wavelet"]))
    plt.legend()
    plt.show()

if __name__=="__main__":
    main()

