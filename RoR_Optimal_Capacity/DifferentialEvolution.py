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

# check the function, it does not look right!!!
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
num_seeds = 5

popsize = 50
CR = 0.9 # crossover probability
F = 0.8 # between 0 and 2, vector step
max_NFE = 200 # should be a multiple

xt = np.zeros((num_seeds,d))
ft_best = np.zeros((num_seeds,d))
ft = np.zeros((num_seeds, max_NFE/popsize))
P = np.zeros((popsize,d))


# Note: current model does not store best values for each time-step. I need to update this

for time in range(len(months)):

  q_mean[time] = np.mean(q[:,time]) # mean flow, m3/s
  q_std[time] = np.std(q[:,time]) # standard deviation, m3/s

  # differential evolution
  for seed in range(num_seeds):
    np.random.seed(seed)
    # random initial population (popsize x d matrix)
    P = np.random.uniform(lb, ub, (popsize,d))
    f = np.zeros(popsize) # we'll evaluate them later

    nfe = 0
    f_best, x_best = None, None

    while nfe < max_NFE:

      # for each member of the population ..
      for i,x in enumerate(P):
        
        # pick two random population members
        xb,xc = P[np.random.randint(0, popsize, 2), :]
        trial_x = np.copy(x)

        # for each dimension ..
        for j in range(d):
          if np.random.rand() < CR:
            trial_x[j] = x[j] + F*(xb[j]-xc[j])
                
        f[i] = NPV_prob(rho,g,eff,H,delta_t,price[time],x,q_mean[time],q_std[time],a,b)
        trial_f = NPV_prob(rho,g,eff,H,delta_t,price[time],trial_x,q_mean[time],q_std[time],a,b)
        nfe += 1

        # if this is better than the parent, replace
        if trial_f < f[i]:
          P[i,:] = trial_x
          f[i] = trial_f

      # keep track of best here
      if f_best is None or f.min() < f_best:
        f_best = f.min()
        x_best = P[f.argmin(),:]

      ft[seed,nfe/popsize-1] = f_best
    
    # for each trial print the result (but the traces are saved in ft)
    xt[seed,:] = x_best
    ft_best[seed,:] = -1*f_best
    print (-1*f_best)
    print(x_best)

# # Save objective results to a .csv file
# np.savetxt('xt.csv', xt, delimiter=",")
# np.savetxt('ft.csv', ft_best, delimiter=",")

ft = -1*ft
plt.semilogx(range(popsize,max_NFE+1,popsize), ft.T, color='steelblue', linewidth=1)
plt.xlabel('Iterations')
plt.ylabel('Objective Value')
plt.show()
