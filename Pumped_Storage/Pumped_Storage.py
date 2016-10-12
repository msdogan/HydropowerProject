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
e_p = 0.75 # pumping efficiency

# fit a curve
z = np.polyfit(price_duration.Frequency, price_duration.Price, 9)
f = np.poly1d(z)

# print price-duration data and curve fitting
plt.scatter(price_duration.Frequency, price_duration.Price)
plt.xlim([0,100])
plt.ylim([0,price_duration.Price.max()])
x_new = np.linspace(0, 100, 50)
y_new = f(x_new)
plt.plot(x_new, y_new, 'r', label = 'Curve fit')
plt.xlim([0, 100])
plt.ylabel('hourly price $/MWh', fontsize = 14)
plt.xlabel('duration %', fontsize = 14)
plt.title('Optimal Generating and Pumping Hours for July', fontsize = 16)
plt.grid(False)

for item,x in enumerate(price_duration.Frequency):
    if round(f(100-x)/f(x),2) == round(e_g * e_p,2):
        H_G = x
        print(H_G)

plt.axvline(x=H_G, ymin=0, ymax = price_duration.Price.max(), linewidth=2, color='k', label = 'Generate Power')
plt.axvline(x=100-H_G, ymin=0, ymax = price_duration.Price.max(), linewidth=2, color='b', label = 'Pump')
plt.legend(fontsize = 11, loc=9)
plt.savefig("figure.pdf")
plt.show()
