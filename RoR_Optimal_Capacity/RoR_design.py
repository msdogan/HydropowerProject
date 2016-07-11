# This code optimizes design flow rate of a run-of-river hydropower plant given river flow availability. 
# It assumes constant head and efficiency. Differential evolution is used to optimize the objective function.
# Mustafa Dogan
# 07/07/2016

from __future__ import division
import numpy as np 
import matplotlib.pyplot as plt
from scipy.integrate import quad, dblquad, trapz, simps
from scipy.stats import lognorm
from mpl_toolkits.mplot3d import Axes3D
# import seaborn as sns
# sns.set_style('whitegrid')

# Parameters
eff = 0.8 # overall efficiency
convert = 0.00046905 # Unit conversion factor from AF/month to m3/s
rho = 1000 # density of water, kg/m3
g = 9.81 # gravitational constant, m/s2
H = 5 # head, m
a = 27.75 # exponential cost constant a*Q**b
b = 3 # exponential cost constant a*Q**b
# r = (1+0.05)**(1/12)-1 # discount rate
r = 0.05 # discount rate
delta_t = 24*30 # 1 Day = 720 Hours

lb = 0
ub = 25
inc = 0.07

q = np.loadtxt('Cosumnes_monthly_divided.txt', skiprows=1)*convert # Cosumnes River Natural Flows from Oct to Sep
price = np.loadtxt('energy_prices.txt') # Average Retail Price of electricity in California from Oct 2013 to Sep 2014, cent/kWh
price = price / 100 # convert to $/kWh
# (Price Source:http://www.eia.gov/electricity/data/browser/#/topic/7?agg=2,0,1&geo=g&freq=M)

months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']

# define cost and benefit functions
def cost(a,b,Q):
  return a*Q**b

def power(rho,g,eff,H,q):
	return rho*g*eff*H*q/1000

def NPV(rho,g,eff,H,q_process,delta_t,p,Q):
	integrand = lambda t: np.exp(-t*r)*power(rho,g,eff,H,q_process)*delta_t*p
	PV = quad(integrand, 0, np.inf)
	NPVal = PV[0] - cost(a, b, Q)
	return NPVal

# declare matrices to store data
# simulation =np.zeros((len(enum_q),len(months)))
# NPV_opt = np.zeros(len(months))
# Q_opt = np.zeros(len(months))

