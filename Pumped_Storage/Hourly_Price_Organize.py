# This code organizes data and create price-duration curves. 
# Mustafa Dogan
### 10/06/2016
##
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')

# This part is all about data (hourly marginal price (wholesale) $/MWh)
##*****************************************************************************
# this function creates price-duration curves
def dur_curve(load, duration, time_period):
    data_raw = [] # empty list to store temporary data
    if duration == 'Monthly':
        month = months.index(time_period) + 1 # python starts from index 0
        print(month)
        for i in range(len(load)):
            if load.Month[i] == month: # Unit is $/MWh
                data_raw.append(load.Price[i])                
    elif duration == 'Annual':
        for i in range(len(load)):
            if load.Year[i] == float(time_period): # Unit is $/MWh
                data_raw.append(load.Price[i])                
    elif duration == 'Daily': # does not work for now
        y,m,d = time_period.split("-") # year, month, day
        for i in range(len(load)):
            if load.Year[i] == float(y):
                if load.Month[i] == float(m):
                    if load.Day[i] == float(d):
                        data_raw.append(load.Price[i])                
    else:
        print('please define correct duration and/or time period')
        return
        
    # after determining what duration and time period to use, create price-duration data
    data = np.sort(data_raw) # sort data
    rank = sp.rankdata(data, method='average') # calculate the rank
    rank = rank[::-1] # non-exceedance prob. Comment out to get exceedance prob
    prob = [100*(rank[i]/(len(data)+1)) for i in range(len(data))] # frequency data
    
    # save price-duration data
    col = ['Price', 'Frequency']
    pdur = [[],[]]
    pdur[0],pdur[1] = data, prob
    pdur = np.array(pdur)
    price_duration = pd.DataFrame(pdur.T, columns = col, dtype = 'float')
    name = 'price_duration_' + str(time_period) + '.csv'
    price_duration.to_csv(name)    
    
    plt.figure()
    plt.scatter(prob,data)
    plt.ylabel('Price')
    plt.xlabel('Duration (%)')
    plt.ylim([0,max(data)+10])
    plt.xlim([0,100])
    plt.title('Hourly Price-Duration Curve for ' + str(time_period))
    plt.show()
    return

# Load Price data from OASIS (CAISO) http://oasis.caiso.com/mrioasis/logon.do
name = 'PRC_LMP_DAM_2016.csv'
df = pd.read_csv(name, parse_dates=True).sort(columns= 'INTERVALSTARTTIME_GMT') # read data and sort by time (gmt)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep','Oct', 'Nov', 'Dec']

P = [[],[],[],[],[],[]] # empty list to store required data
columns = ['Year', 'Month', 'Day', 'Start_Hour', 'End_Hour', 'Price'] # headers for data frame

# We are only interested in , start time, end time and LMP
for i in range(len(df)):
    if df.LMP_TYPE[i] == "LMP": # Unit is $/MWh
        P[0].append(df.INTERVALSTARTTIME_GMT[i].split("T")[0].split("-")[0])
        P[1].append(df.INTERVALSTARTTIME_GMT[i].split("T")[0].split("-")[1])
        P[2].append(df.INTERVALSTARTTIME_GMT[i].split("T")[0].split("-")[2])
        P[3].append(df.INTERVALSTARTTIME_GMT[i].split("T")[1].split("-")[0].split(":")[0])
        P[4].append(df.INTERVALENDTIME_GMT[i].split("T")[1].split("-")[0].split(":")[0])
        P[5].append(df.MW[i])
 
P = np.array(P) # convert list to numpy array    
Date_Price = pd.DataFrame(P.T, columns = columns, dtype = 'float') # convert list to data frame

# Examples of 'dur_curve' function use
#print(dur_curve(Date_Price, 'Annual', '2016')) # annual example
print(dur_curve(Date_Price, 'Monthly', 'Jul')) # monthly example
#print(dur_curve(Date_Price, 'Daily', '2016-9-7')) # daily example ('year-month-day')

Date_Price.to_csv('price.csv')

##*****************************************************************************


