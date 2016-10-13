# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 10:50:15 2016
@author: msdogan
"""
# This code optimizes pump-storage hydropower facility operations. 
# Mustafa Dogan
### 10/12/2016
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
    return price_duration

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
price = pd.DataFrame(P.T, columns = columns, dtype = 'float') # convert list to data frame

# Examples of 'dur_curve' function use
#price_duration = dur_curve(price, 'Annual', '2016') # annual example
price_duration = dur_curve(price, 'Monthly', 'Jul') # monthly example
#price_duration = dur_curve(price, 'Daily', '2016-9-7') # daily example ('year-month-day')

price.to_csv('price.csv')

##*****************************************************************************

e_g = 0.85 # generation efficiency
e_p = 0.80 # pumping efficiency

# fit a curve
z = np.polyfit(price_duration.Frequency, price_duration.Price, 9)
f = np.poly1d(z)

# normal distribution (cumulative, exceedance)
y_norm = np.linspace(0, price.Price.max(), 50)
x_norm = (1-sp.norm(price_duration.Price.mean(), price_duration.Price.std()).cdf(y_norm))*100

# print price-duration data and curve fitting
plt.scatter(price_duration.Frequency, price_duration.Price)
plt.xlim([0,price_duration.Frequency.max()])
plt.ylim([0,price_duration.Price.max()])
plt.plot(x_norm, y_norm, 'cyan', label = 'Normal Dist.')
x_new = np.linspace(0, price_duration.Frequency.max(), 50)
y_new = f(x_new)
plt.plot(x_new, y_new, 'r', label = 'Curve fit')
plt.ylabel('hourly price $/MWh', fontsize = 14)
plt.xlabel('duration %', fontsize = 14)
plt.title('Optimal Generating and Pumping Hours for July', fontsize = 16)
plt.grid(False)

for item,x in enumerate(price_duration.Frequency):
    if round(f(price_duration.Frequency.max()-x)/f(x),2) == round(e_g * e_p,2):
        H_G = x
        
#for item,x in enumerate(price_duration.Frequency):
#    if (sp.norm(price.Price.mean(), price.Price.std()).cdf(price_duration.Frequency.max()-x)*100) + (sp.norm(price.Price.mean(), price.Price.std()).cdf(x)*100) >= 100: # total hour % can't exceed 100%
#        break
#    if round((sp.norm(price.Price.mean(), price.Price.std()).cdf(price_duration.Frequency.max()-x)*100)/(sp.norm(price.Price.mean(), price.Price.std()).cdf(x)*100),3) == round(e_g * e_p,2):
#        H_G = x
		
print('Optimal Operation at '+ str(round(H_G,2)) + ' % of Total Hours')

plt.axvline(x=H_G, ymin=0, ymax = price_duration.Price.max(), linewidth=2, color='k', label = 'Generate Power')
plt.axvline(x=price_duration.Frequency.max()-H_G, ymin=0, ymax = price_duration.Price.max(), linewidth=2, color='b', label = 'Pump')
plt.legend(fontsize = 11, loc=9)
plt.text(0.5,7, 'Generating Hours')
plt.text(79,7, 'Pumping Hours')
plt.text(25,150, 'Generating Price Threshold >= ' + str(round(f(H_G),2)) + ' $/MWh', fontsize = 11)
plt.text(25,125, 'Pumping Price Threshold <= ' + str(round(f(price_duration.Frequency.max()-H_G),2)) + ' $/MWh', fontsize = 11)
plt.savefig("figure_pd.pdf")
plt.show()

# create time-series plot
# NOT WORKING!!! Time-series data is not in correct order!!! 
#price.Price.plot(linewidth=0.75)
#plt.axhline(y=f(H_G), linewidth=2, color='k', label = 'Generate Power')
#plt.axhline(y=f(price_duration.Frequency.max()-H_G), linewidth=2, color='red', label = 'Pump')
#plt.legend(fontsize = 11, loc=9)
#plt.grid(False)
#plt.ylabel('hourly price $/MWh', fontsize = 14)
#plt.xlabel('hours', fontsize = 14)
#plt.savefig("figure_ts.pdf")
#plt.show()

