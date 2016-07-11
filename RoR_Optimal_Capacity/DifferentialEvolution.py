from __future__ import division
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
# import seaborn as sns
# sns.set_style('whitegrid')

# Note: This algorithm minimizes negative benefits but returns actual benefits in the end.

price = 10.21 # Average monthly retail price of electricity for May, cent/kWh
price = price / 100 # convert to $/kWh
# (Source:http://www.eia.gov/electricity/data/browser/#/topic/7?agg=2,0,1&geo=g&freq=M)
rho = 1000 # density of water, kg/m3
g = 9.81 # gravitational constant, m/s2
delta_t = 24*30 # 1 Day = 720 Hours
eff = 0.8 # overall efficiency
Dead_Pool = 116 # Dead pool storage,TAF
Res_Cap = 4000 # Reservoir capacity, TAF
Release_Cap = 18000 # Release/turbine capacity, cfs

# calculate elevation (m) as a function of storage (m3)
def elev(x):
  a = 0.3899
  b = 0.2697
  return a*x**b

# convert from TAF to m3
def TAFconvertm3(x):
  return x*1000*1233.481855
# convert from TAF/m to m3/s
def ReleaseConvert(x):
  return x*0.0283168

# objective function to minimize
def ObjFunc(x):
  Storage = TAFconvertm3(x[0])
  Head = elev(Storage) # Head in m
  Release = ReleaseConvert(x[1]) # Release in m3/s
  Revenue = rho*g*eff*Head*Release/1000 * delta_t * price # revenue, $
  f1 = Revenue
  a = 4000000
  b = 0.4
  f2 = a*Release**b
  return -1*(f1 + f2)

d = 2 # dimension of decision variable space
num_seeds = 10

popsize = 50
CR = 0.3 # crossover probability
F = 1.5 # between 0 and 2, vector step
max_NFE = 20000 # should be a multiple

xt = np.zeros((num_seeds,d))
ft_best = np.zeros((num_seeds,d))
ft = np.zeros((num_seeds, max_NFE/popsize))
P = np.zeros((popsize,d))

# differential evolution
for seed in range(num_seeds):
  np.random.seed(seed)
  # random initial population (popsize x d matrix)
  for i,item in enumerate(P):
    P[i,0] = np.random.uniform(Dead_Pool, Res_Cap)
    P[i,1] = np.random.uniform(0, Release_Cap)
  
  f = np.zeros(popsize) # we'll evaluate them later
  nfe = 0
  f_best, x_best = None, None

  while nfe < max_NFE:

    # for each member of the population ..
    for i,x in enumerate(P):
      
      # pick two random population members
      xb,xc = P[np.random.randint(0, popsize, d), :]
      trial_x = np.copy(x)

      # for each dimension ..
      for j in range(d):
        if np.random.rand() < CR:
          trial_x[j] = x[j] + F*(xb[j]-xc[j])
      if trial_x[1] > Release_Cap or trial_x[0] > Res_Cap:
        break
        
      f[i] = ObjFunc(x)
      trial_f = ObjFunc(trial_x)
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
  print ('Seed: '+ str(seed) + ', Stor_best: '+ str(x_best[0]) + ', Rel_best: ' + str(x_best[1]) + ', Obj: ' + str(-1*f_best))

# # Save objective results to a .csv file
# np.savetxt('xt.csv', xt, delimiter=",")
# np.savetxt('ft.csv', ft_best, delimiter=",")

ft = -1*ft
plt.semilogx(range(popsize,max_NFE+1,popsize), ft.T, color='steelblue', linewidth=1)
plt.xlabel('Iterations')
plt.ylabel('Objective Value')
plt.show()
