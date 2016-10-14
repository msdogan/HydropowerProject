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
import scipy as sp
from scipy.optimize import differential_evolution
from scipy.integrate import quad
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
    rank = sp.stats.rankdata(data, method='average') # calculate the rank
    rank = rank[::-1] 
    prob = [100*(rank[i]/(len(data)+1)) for i in range(len(data))] # frequency data
    
    # save price-duration data
    col = ['Price', 'Frequency']
    pdur = [[],[]]
    pdur[0],pdur[1] = data, prob
    pdur = np.array(pdur)
    price_duration = pd.DataFrame(pdur.T, columns = col, dtype = 'float')
    s_name = 'price_duration_' + str(time_period) + '.csv'
    price_duration.to_csv(s_name)    
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
# Annual Duration and Time
#duration = 'Annual'
#time = '2016'

# Monthly Duration and Time
duration = 'Monthly'
time = 'Aug'

# Daily Duration and Time
#duration = 'Daily'
#time = '2016-9-11'

price_duration = dur_curve(price, duration, time)

price.to_csv('price.csv')

##*****************************************************************************

# fit a curve
z = np.polyfit(price_duration.Frequency, price_duration.Price, 9)
f = np.poly1d(z)

# objective function to maximize - continuous
# DOES NOT WORK!!! 
def obj_func_cont(xx, optimizing = True):
    # parameters
    e_g = 0.85 # generation efficiency
    e_p = 0.80 # pumping efficiency
    g = 9.81 # m/s2 - acceleration of gravity
    rho = 1000 # kg/m3 - density of water
    Q_g = 100 # m3/s - water flow for turbine
    Q_p = 100 # m3/s - water flow for pumping
    head_g = 100 # m - generating head    
    head_p = 100 # m - pumping head
    H_T = price_duration.Frequency.max() # total duration (100%)
    integrand_G = lambda H: f(H)*e_g*rho*g*Q_g*head_g*H/1000000
    Power_Revenue = quad(integrand_G, 0, xx)
    integrand_P = lambda H: f(H)/e_p*rho*g*Q_p*head_p*H/1000000
    Pumping_Cost = quad(integrand_P, H_T-xx, H_T)
    z = Power_Revenue[0] - Pumping_Cost[0] # profit
    return -z if optimizing else z

# objective function to maximize - discrete
def obj_func_disc(xx, optimizing = True):
    # parameters
    e_g = 0.85 # generation efficiency
    e_p = 0.80 # pumping efficiency
    g = 9.81 # m/s2 - acceleration of gravity
    rho = 1000 # kg/m3 - density of water
    Q_g = 100 # m3/s - water flow for turbine
    Q_p = 100 # m3/s - water flow for pumping
    head_g = 100 # m - generating head    
    head_p = 100 # m - pumping head
    dH = 1 # discretization level
    H_T = int(price_duration.Frequency.max()) # total duration (100%)
    Power_Revenue = 0
    for gen in range(0,xx):
        Power_Revenue += f(gen)*e_g*rho*g*Q_g*head_g*dH/(10**6)
    Pumping_Cost = 0
    for pump in range(H_T-xx,H_T):
        Pumping_Cost += f(pump)/e_p*rho*g*Q_p*head_p*dH/(10**6)
    z = Power_Revenue - Pumping_Cost # profit
    return -z if optimizing else z

## objective function to maximize - discrete, no curve fitting
## Currently not working
#def obj_func_disc_nofit(xx, optimizing = True):
#    # parameters
#    e_g = 0.85 # generation efficiency
#    e_p = 0.80 # pumping efficiency
#    g = 9.81 # m/s2 - acceleration of gravity
#    rho = 1000 # kg/m3 - density of water
#    Q_g = 100 # m3/s - water flow for turbine
#    Q_p = 100 # m3/s - water flow for pumping
#    head_g = 100 # m - generating head    
#    head_p = 100 # m - pumping head
#    freq = np.sort(price_duration.Frequency)
#    prc = np.sort(price_duration.Price)
#    Power_Revenue = 0
#    Pumping_Cost = 0
#    for item,x in enumerate(freq):
#        while x < xx:
#            Power_Revenue = Power_Revenue + f(x)*e_g*rho*g*Q_g*head_g*x/(10**6)
#            print(x,xx)
#        else:
#            Pumping_Cost = Pumping_Cost+ f(x)/e_p*rho*g*Q_p*head_p*x/(10**6)
#    z = Power_Revenue - Pumping_Cost # profit
#    return -z if optimizing else z


x_new = np.linspace(0, price_duration.Frequency.max(), 50)
y_new = f(x_new)

# normal distribution (cumulative, exceedance)
y_norm = np.linspace(0, price_duration.Price.max(), 50)
x_norm = sp.stats.norm(price_duration.Price.mean(), price_duration.Price.std()).sf(y_norm)*100 # survival function

# print price-duration data and curve fitting
plt.scatter(price_duration.Frequency, price_duration.Price)
plt.xlim([0,price_duration.Frequency.max()])
plt.ylim([0,price_duration.Price.max()])
plt.plot(x_norm, y_norm, 'cyan', label = 'Normal Dist.', linewidth=2) # normal dist. plot
plt.plot(x_new, y_new, 'r', label = 'Curve fit') # curve fit plot
plt.ylabel('hourly price $/MWh', fontsize = 14)
plt.xlabel('duration %', fontsize = 14)
plt.title('Optimal Generating and Pumping Hours for ' + str(time), fontsize = 16)
plt.grid(False)

# Reduced Analytical solution without integration: e_g * e_p = P(1-H_G)/P(H_G) 
#e_g = 0.85 # generation efficiency
#e_p = 0.80 # pumping efficiency
#for item,i in enumerate(price_duration.Frequency):
#    if f(i) >= f(price_duration.Frequency.max()-i): # total proability cannot exceed 1 (100%)
#        if round(f(price_duration.Frequency.max()-i)/f(i),2) == round(e_g * e_p,2):
#            H_G = i

# differential evolution
result = differential_evolution(obj_func_disc, bounds=[(0,100)], maxiter=1000, seed = 1)
print(result)
H_G = result.x
          
plt.axvline(x=H_G, linewidth=2, color='k', label = 'Generate Power')
plt.axvline(x=price_duration.Frequency.max()-H_G, linewidth=2, color='b', label = 'Pump')
plt.legend(fontsize = 12, loc=9)
plt.text(H_G-3,(price_duration.Price.max()+price_duration.Price.min())/2, 'Generating Hours', color = 'k', rotation = 'vertical')
plt.text(price_duration.Frequency.max()-H_G+1,(price_duration.Price.max()+price_duration.Price.min())/2, 'Pumping Hours', color = 'b', rotation = 'vertical')
plt.text(25,(price_duration.Price.max()+price_duration.Price.min())/2, 'Generating Price Threshold >= ' + str(round(f(H_G),2)) + ' $/MWh', fontsize = 11)
plt.text(25,(price_duration.Price.max()+price_duration.Price.min())/2-12, 'Pumping Price Threshold <= ' + str(round(f(price_duration.Frequency.max()-H_G),2)) + ' $/MWh', fontsize = 11)
plt.savefig("figure_pd.pdf")
plt.show()

print('*******Optimal Operation at '+ str(round(H_G,2)) + ' % of Total Hours*******')

# enumeration
enum_h = np.arange(0, 100, 1)
simulation =np.zeros(len(enum_h))
for i,item in enumerate(enum_h):
    simulation[i] = obj_func_disc(i, optimizing = False)
index = np.where(simulation == simulation.max())[0]

plt.plot(enum_h, simulation, label = 'Net Profit (Gen-Pump)')
plt.axvline(x=enum_h[index], linewidth=2, color='k', label = 'Opt Gen. Duration')
plt.title('Enumeration Line', fontsize = 16)
plt.xlabel('duration %', fontsize = 14)
plt.ylabel('profit $/hour', fontsize = 14)
plt.legend(fontsize = 12, loc=1)
plt.grid(False)
plt.savefig("figure_enum.pdf")
plt.show


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

