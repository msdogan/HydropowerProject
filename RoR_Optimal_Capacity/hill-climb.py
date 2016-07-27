from __future__ import division
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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

# months = ['Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep']
months = ['Oct']

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
  return -1*NPVal


NPV_opt = np.zeros(len(months))
Q_opt = np.zeros(len(months))
q_mean = np.zeros(len(months))
q_std = np.zeros(len(months))

d = 1 # dimension of decision variable space
s = 0.4 # stdev of normal noise (if this is too big, it's just random search!)

num_seeds = 10
max_NFE = 200000

ft = np.zeros((num_seeds, max_NFE))
x = np.zeros(d)
xt = np.zeros((num_seeds,d))
ft_best = np.zeros((num_seeds,d))

# Note: current model does not store best values for each time-step. I need to update this

for time in range(len(months)):

  q_mean[time] = np.mean(q[:,time]) # mean flow, m3/s
  q_std[time] = np.std(q[:,time]) # standard deviation, m3/s

  # hill climbing
  for seed in range(num_seeds):
    np.random.seed(seed)

    # random initial starting point
    x = np.random.uniform(lb, ub)
    
    bestf = NPV_prob(rho,g,eff,H,delta_t,price[time],x,q_mean[time],q_std[time],a,b)
    nfe = 0

    while nfe < max_NFE:

      trial_x = x + np.random.normal(0,s,d)
      trial_f = NPV_prob(rho,g,eff,H,delta_t,price[time],trial_x,q_mean[time],q_std[time],a,b)

      if trial_f > bestf:
        x = trial_x
        bestf = trial_f
      
      ft[seed,nfe] = bestf
      nfe += 1

    xt[seed,:] = x
    ft_best[seed,:] = bestf
    # for each trial print the result (but the traces are saved in ft)
    print(bestf)


# # Save objective results to a .csv file
# np.savetxt('xt.csv', xt, delimiter=",")
# np.savetxt('ft.csv', ft_best, delimiter=",")

plt.loglog(ft.T, color='steelblue', linewidth=1)
plt.xlabel('Iterations')
plt.ylabel('Objective Value')
plt.show()


