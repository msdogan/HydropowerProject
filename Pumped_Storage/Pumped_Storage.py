# This code optimizes operations of pumped-storage hydropower facilities. 
# Mustafa Dogan
### 10/06/2016
##
from __future__ import division
import numpy as np
import time 
import matplotlib.pyplot as plt
import scipy.stats as sp
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')

# this function creates price-duration curves
def dur_curve(p):
    data = np.sort(p)
    rank = sp.rankdata(data, method='average')
    rank = rank[::-1] # non-exceedance prob. Comment out to get exceedance prob
    prob = [100*(rank[i]/(len(data)+1)) for i in range(len(data))]
    
    plt.figure()
    plt.scatter(prob,data)
    plt.ylabel('Price')
    plt.xlabel('Duration (%)')
    plt.title('Price-Duration Curve')
    return

# create moving average curves
def moving_aver():
    return

# Load Price data from OASIS (CAISO) http://oasis.caiso.com/mrioasis/logon.do
name = '20160901_20161002_PRC_LMP_DAM_20161005_11_19_18_v1.csv'
df = pd.read_csv(name, parse_dates=True)
months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
#print(months.index('Oct'))

P = [[],[],[],[],[],[]]
#P = np.empty([1,6])
columns = ['Year', 'Month', 'Day', 'Start_Hour', 'End_Hour', 'Price']

# We are only interested in , start time, end time and LMP
for i in range(len(df)):
    if df.LMP_TYPE[i] == "LMP": # Unit is $/MWh
        P[0].append(df.INTERVALSTARTTIME_GMT[i].split("T")[0].split("-")[0])
        P[1].append(df.INTERVALSTARTTIME_GMT[i].split("T")[0].split("-")[1])
        P[2].append(df.INTERVALSTARTTIME_GMT[i].split("T")[0].split("-")[2])
        P[3].append(df.INTERVALSTARTTIME_GMT[i].split("T")[1].split("-")[0])
        P[4].append(df.INTERVALENDTIME_GMT[i].split("T")[1].split("-")[0])
        P[5].append(df.MW[i])
 
P = np.array(P)    
Date_Price = pd.DataFrame(P.T, columns = columns)

print(dur_curve(P[5]))
#plt.plot(P[5])
#plt.ylabel('LMP ($/MWh)')
#plt.title('Price Time-Series')
#plt.show()


