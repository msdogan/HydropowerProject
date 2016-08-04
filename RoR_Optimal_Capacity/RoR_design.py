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
inc = 0.7

q = np.loadtxt('Cosumnes_monthly_divided.txt', skiprows=1)*convert # Cosumnes River Natural Flows from Oct to Sep
price = np.loadtxt('energy_prices.txt') # Average Retail Price of electricity in California from Oct 2013 to Sep 2014, cent/kWh
price = price / 100 # convert to $/kWh
# (Price Source:http://www.eia.gov/electricity/data/browser/#/topic/7?agg=2,0,1&geo=g&freq=M)

months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']

# method = 0 is Deterministic, method = 0 is Probabilistic
method = 1

# define cost and benefit functions
def cost(a,b,Q):
  return a*Q**b

def power(rho,g,eff,H,q):
	return rho*g*eff*H*q/1000

def energy(rho,g,eff,H,q,delta_t):
	return power(rho,g,eff,H,q)*delta_t

def revenue(rho,g,eff,H,q,delta_t,p):
	return energy(rho,g,eff,H,q,delta_t)*p

def NPV(rho,g,eff,H,q_process,delta_t,p,Q,a,b):
	integrand = lambda t: np.exp(-r*t)*power(rho,g,eff,H,q_process)*delta_t*p
	PV = quad(integrand, 0, np.inf)
	NPVal = PV[0] - cost(a, b, Q)
	return NPVal

def lognorm_pdf(x,mean,std):
	dist_pdf = lognorm.pdf(x,std,0,mean)
	return dist_pdf

def lognorm_cdf(x,mean,std):
	dist_cdf = lognorm.cdf(x,std,0,mean)
	return dist_cdf

def EV_flow(mean,std,Q):
	EV = quad(lambda x: lognorm_pdf(x,mean,std)*x, 0, Q)
	EV_2 = (1 - lognorm_cdf(Q,mean,std))*Q
	EV_total = EV[0] + EV_2
	return EV_total

def NPV_prob(rho,g,eff,H,delta_t,p,Q,mean,std,a,b):
	t = np.arange(0,100000,0.5)
	PV = simps(np.exp(-t*r)*rho*g*eff*H*delta_t*p/1000*EV_flow(mean,std,Q), t)
	NPVal = PV - cost(a, b, Q)
	return NPVal

# declare matrices to store data
enum_q = np.arange(lb, ub, inc)

simulation =np.zeros((len(enum_q),len(months)))
NPV_opt = np.zeros(len(months))
Q_opt = np.zeros(len(months))
q_mean = np.zeros(len(months))
q_std = np.zeros(len(months))


for time in range(len(months)):

	q_mean[time] = np.mean(q[:,time]) # mean flow, m3/s
	q_std[time] = np.std(q[:,time]) # standard deviation, m3/s

	for i,item in enumerate(enum_q):

		if method == 0: # Deterministic Approach
			for j,itemx in enumerate(q[:,time]):
				if itemx < item:
					q_process = itemx
				else:
					q_process = item
				simulation[i,time] = NPV(rho,g,eff,H,q_process,delta_t,price[time],item,a,b) # Deterministic Approach
		else: # Probabilistic Approach
			simulation[i,time] = NPV_prob(rho,g,eff,H,delta_t,price[time],item,q_mean[time],q_std[time],a,b) # Probabilistic Approach

	NPV_opt[time] = np.max(simulation[:,time])
	Q_opt[time] = enum_q[np.argmax(simulation[:,time])]

	print('month: '+str(months[time])+', Q_optimal: '+str(Q_opt[time])+' m3/s'+', Q_mean: '+str(q_mean[time]))

	plt.plot(enum_q, simulation[:,time]/(10**6), label=months[time]+ ', Qdesign: '+str(Q_opt[time]))

plt.ylabel('Net Present Value (M$/m)', fontsize=16)
plt.xlabel('Design Flow Rate (m3/s)', fontsize=16)
plt.xticks(fontsize = 13)
plt.yticks(fontsize = 13)
plt.legend(loc='lower left', fontsize=13)
plt.show()