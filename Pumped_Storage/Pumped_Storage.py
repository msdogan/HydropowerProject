# -*- coding: utf-8 -*-
"""
Created on Wed Oct 12 10:50:15 2016
@author: msdogan
"""
# This code optimizes pump-storage hydropower facility operations. 
# Mustafa Dogan
### 10/12/2016
##
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sp
import pandas as pd
import seaborn as sns
sns.set_style('whitegrid')

# import data
price = pd.read_csv('price.csv')
price_duration = pd.read_csv('price_duration_Jul.csv')

e_g = 0.85 # generation efficiency
e_p = 0.80 # pumping efficiency

# fit a curve
z = np.polyfit(price_duration.Frequency, price_duration.Price, 9)
f = np.poly1d(z)

# print price-duration data and curve fitting
plt.scatter(price_duration.Frequency, price_duration.Price)
plt.xlim([0,price_duration.Frequency.max()])
plt.ylim([0,price_duration.Price.max()])
y_norm = np.linspace(0, price.Price.max(), 50)
x_norm = (1-sp.norm(price_duration.Price.mean(), price_duration.Price.std()).cdf(y_norm))*100
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
#    if sp.norm(price.Price.mean(), price.Price.std()).cdf(price_duration.Frequency.max()-x) <= sp.norm(price.Price.mean(), price.Price.std()).cdf(x) == round(e_g * e_p,2):
#        break
#    if round(sp.norm(price.Price.mean(), price.Price.std()).cdf(price_duration.Frequency.max()-x)/sp.norm(price.Price.mean(), price.Price.std()).cdf(x),3) == round(e_g * e_p,2):
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

price.Price.plot(linewidth=0.75)
plt.axhline(y=f(H_G), linewidth=2, color='k', label = 'Generate Power')
plt.axhline(y=f(price_duration.Frequency.max()-H_G), linewidth=2, color='red', label = 'Pump')
plt.legend(fontsize = 11, loc=9)
plt.grid(False)
plt.ylabel('hourly price $/MWh', fontsize = 14)
plt.xlabel('hours', fontsize = 14)
plt.savefig("figure_ts.pdf")
plt.show()
