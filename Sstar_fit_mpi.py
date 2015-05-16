# -*- coding: utf-8 -*-
"""
Created on Wed May 13 11:42:42 2015

@author: dylan
"""

import sys
import time
import emcee
import triangle
import numpy as np
import matplotlib.pyplot as plt
from math import pi, sin, cos
from emcee.utils import MPIPool
#from numbapro import vectorize, float32
#@vectorize([float32(float32, float32)], target='parallel')

def RKF45(t0, y0, f, tmax, h0=1*24*60*60, atol=1e-8, rtol=1e-18,
          safety = 0.9, hfactor_min = 0.01, hfactor_max = 10):
    """Runge-Kutta-Fehlberg method to solve y' = f(t,y) with y(t0) = y0.
    Runge-Kutta-Fehlberg method is one of the Embedded methods which can
    estimate the local truncation error of a single Runge-Kutta step,
    and thus allow to control the error with adaptive stepsize. 
    Parameters:
      h0: initial step sizes, default 1 day
      tmax: Maximum integration time, default 1 period
      atol: float or sequence absolute tolerance for solution
      rtol: float or sequence relative tolerance for solution
      safety: float Safety factor on new step selection, default 0.9
      hfactor_min: float Minimum factor to change step size in one step
      hfactor_max: float Maximum factor to change step size in one step
    Warning: if atol or rtol is set to 0, the program will run for a very long
    time and I don't really understand why"""

    #Butcher Tableau  
    cc = np.array([0, 1/4, 3/8, 12/13, 1, 1/2])
    aa = np.array([[0        ,      0    ,     0     ,      0   ,     0  ],
                   [1/4      ,      0    ,     0     ,      0   ,     0  ],
                   [3/32     ,     9/32  ,     0     ,      0   ,     0  ],
                   [1932/2197, -7200/2197, 7296/2197 ,      0   ,     0  ],
                   [439/216  ,     -8    , 3680/513  , -845/4104,     0  ],
                   [-8/27    ,      2    , -3544/2565, 1859/4104,  -11/40]])
    bb5 = np.array([16/135, 0, 6656/12825, 28561/56430, -9/50, 2/55])
    bb4 = np.array([25/216, 0, 1408/2565,  2197/4104,   -1/5,   0  ])
    kk = np.zeros([6, 6])
    tt = [t0]; yy = [y0]
    t = t0;  y = y0; h=h0
    while t < tmax:
       #if t + h > tmax:
       #     h = tmax - t
        for k in range(6):
            kk[k] = h*f(t+cc[k]*h, y+aa[k].dot(kk[:5,:]))
        err = abs((bb5-bb4).dot(kk))
        scale = atol + rtol*abs(y)
        err = np.sqrt(sum((err/scale)**2)/len(err)) # simpler err=max(err)/atol
        if err <= 1:
            t = t + h
            y = y + bb4.dot(kk)
            tt.append(t)
            yy.append(y)
        try:
            h = h*min(max(safety*(1/err)**(1/5), hfactor_min), hfactor_max)
        except ZeroDivisionError:
            h = h * hfactor_max
    return (np.array(tt), np.array(yy))


def orbit(P, T0, e, i, omega, Omega, M_BH, distance, X, Y):
    G = 6.67e-11
    M_sun = 2.0e30
    
    # Parameters Preprocessing 
    P = P*365*24*60*60
    distance = distance*1e3*3e16  # distance of the black hole to the earth
    M_BH = M_BH*1e6*M_sun # Mass of black hole and star
    M_star = 10*M_sun
    mu = G*(M_BH+M_star)
    a = (mu/4/pi**2 * P**2)**(1/3)   # a^3/P^2 = G(M_BH+m)/4/pi^2

    # Solve the orbit
    rp = a*(1-e)    # pericenter distance
    vp = np.sqrt(2*mu/rp - mu/a)    # speed at pericenter
    r_star0 = [rp, 0, 0]
    v_star0 = [0, vp, 0]
    t0 = 0
    y0 = np.hstack([r_star0, v_star0])
    f = lambda t,y: np.hstack( [y[3:],
               -mu/np.linalg.norm(y[:3])**3 * y[:3] * (M_BH/(M_BH+M_star))**3] )
    t,y = RKF45(t0, y0, f, tmax=1*P)

    # coordinate transformation
    A = np.array([[cos(Omega), -sin(Omega), 0],
                  [sin(Omega),  cos(Omega), 0],
                  [  0,            0,       1]])
    B = np.array([[1,  0,         0  ],
                  [0, cos(i), -sin(i)],
                  [0, sin(i),  cos(i)]])
    C = np.array([[cos(omega), -sin(omega), 0],
                  [sin(omega),  cos(omega), 0],
                  [   0,          0,        1]])
    xyz_star = A.dot(B).dot(C).dot(y[:,:3].T)
    DEC_RA = np.arctan(xyz_star[:2,:]/distance) *180/pi *60*60 *1000
    return DEC_RA+[[X], [Y]], t


def orbit_interpolation(date, P, T0, e, i, omega, Omega, M_BH, distance, X, Y):
    date = date-T0 + (date<T0)*P
    date = date*365*24*60*60
    DEC_RA,t = orbit(P, T0, e, i, omega, Omega, M_BH, distance, X, Y)
    DEC_RA_theoretic = np.zeros((2,len(date)))
    for k in range(len(date)):
        temp = len(t[t<date[k]])
        DEC_RA_theoretic[:,k] = (DEC_RA[:,temp-1]+DEC_RA[:,temp])/2
    return DEC_RA_theoretic


def lnposterior (parameters, data):
    '''lnposterior = lnlikelihood + lnprior
       lnprior = 0 in the range of parameters, =-inf outside the range
       Parameters:
         parameters: The fitting parameters of the model,also the position
                     of a single walker (N-dim numpy array).'''
#    parameters1, parameters2 = parameters
#    (P_1, T0_1, e_1, i_1, omega_1, Omega_1, M_BH, distance, X, Y) = parameters1
#    (P_2, T0_2, e_2, i_2, omega_2, Omega_2, M_BH, distance, X, Y) = parameters2
    (M_BH, distance, X, Y,
     P_1, T0_1, e_1, i_1, omega_1, Omega_1,
     P_2, T0_2, e_2, i_2, omega_2, Omega_2) = parameters
    parameters1 = (P_1, T0_1, e_1, i_1, omega_1, Omega_1, M_BH, distance, X, Y)
    parameters2 = (P_2, T0_2, e_2, i_2, omega_2, Omega_2, M_BH, distance, X, Y)
    data1, data2 = data
    (date_1, RA_measured_1, err_RA_1, DEC_measured_1, err_DEC_1) = data1
    (date_2, RA_measured_2, err_RA_2, DEC_measured_2, err_DEC_2) = data2
    if 0.1<M_BH<10 and 7<distance<9 and -100<X<100 and -100<Y<100\
       and 14<P_1<16 and 2001.5<T0_1<2003 and 0<e_1<1\
       and 1<i_1<pi and 1<omega_1<2*pi and 1<Omega_1<2*pi\
       and 11<P_2<12 and 2009<T0_2<2010 and 0<e_2<1\
       and 0<i_2<pi and 0<omega_2<2*pi and 0<Omega_2<2*pi:
        DEC_1,RA_1 = orbit_interpolation(date_1, *parameters1)
        DEC_2,RA_2 = orbit_interpolation(date_2, *parameters2)
        return -1/2 * (sum(((RA_measured_1 - RA_1)/err_RA_1)**2+
                           ((DEC_measured_1 - DEC_1)/err_DEC_1)**2)+\
                       sum(((RA_measured_2 - RA_2)/err_RA_2)**2+
                           ((DEC_measured_2 - DEC_2)/err_DEC_2)**2))
    return -np.inf



# .........Load data.............................
#Data_path = '/home/dylan/Documents/Nutstore/Code/python/'
#Data = np.loadtxt(Data_path+"S0_2", delimiter="\t")
#Data = np.loadtxt(Data_path+"S0_102", delimiter="\t")
Data1 = np.loadtxt("S0_2", delimiter="\t")
date_1 = Data1[:,0]
RA_measured_1 = Data1[:,1]
err_RA_1 = Data1[:,2]
DEC_measured_1 = Data1[:,3]
err_DEC_1 = Data1[:,4]
data1 = (date_1, RA_measured_1, err_RA_1, DEC_measured_1, err_DEC_1)

Data2 = np.loadtxt("S0_102", delimiter="\t")
date_2 = Data2[:,0]
RA_measured_2 = Data2[:,1]
err_RA_2 = Data2[:,2]
DEC_measured_2 = Data2[:,3]
err_DEC_2 = Data2[:,4]
data2 = (date_2, RA_measured_2, err_RA_2, DEC_measured_2, err_DEC_2)

data = (data1,data2)


#.......Initialize the pool used for parallelization......
pool = MPIPool()

if not pool.is_master():
    # Wait for instructions from the master process.
    pool.wait()
    sys.exit(0)
#The point here is that:
#Only the master processes could directly interact with the sampler
#All other processes should only wait for master's instructions.


# ........Initialization Sampler.................
ndim = 16 #When the number of paramters changes, remembers to modify the output part accordingly.
nwalkers = 100
#parameters0_1 = [[15, 2002, 0.8, 2, 1, 4, 4, 8, 0, 0] + 1e-4*np.random.randn(ndim)
#               for i in range(nwalkers)]
#parameters0_2 = [[11.5, 2009.5, 0.7, 2, 3, 3, 4, 8, 0, 0] + 1e-4*np.random.randn(ndim) 
#               for i in range(nwalkers)]
#parameters0 = (parameters0_1, parameters0_2)
parameters0 = [[ 4, 8, 0, 0, 15, 2002, 0.8, 2, 1, 4, 
                             11.5, 2009.5, 0.7, 2, 3, 3] + 1e-4*np.random.randn(ndim)
               for i in range(nwalkers)]
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnposterior, args=[data], pool=pool)


#.........Sampling...............................
print('Sampling start at ' + time.strftime('%Y/%m/%d %I:%M%p'))
tstart = time.time()

# Run a burn-in.
pos, prob, state = sampler.run_mcmc(parameters0, 800)
sampler.reset()

# Sample and save the result
fn = "Result_MCMC"+time.strftime('-%Y-%m-%d')
with open(fn, "w") as f:
    f.write("# M_BH(10^6M_sun) distance(kpc) X(mas) Y(mas) \
               P_1(year) T0_1(year) e_1 i_1 omega_1 Omega_1\
               P_2(year) T0_2(year) e_2 i_2 omega_2 Omega_2 lnposterior\n")
#The symbol # here is used to tell the load function to ingnore this line when load data. 
for pos, lnprob, state in sampler.sample(pos, iterations=700, storechain=False):
    with open(fn, "a") as f:
        for p, lp in zip(pos, lnprob):
            f.write("{0} {1}\n".format(" ".join(map("{0}".format, p)), lp))
print('Sampling end at ' + time.strftime('%Y/%m/%d %I:%M%p'))
print("Took {0:.1f} hours".format((time.time() - tstart)/3600))
print("Acceptance fraction: {0:.3f}".format(np.mean(sampler.acceptance_fraction)))
samples = np.loadtxt(fn)[:,:-1]
#samples = sampler.chain[:, 500:, :].reshape((-1, ndim))

#...........Close the processes......................
pool.close()


#...........Output sampling result...................
# calculate reduced chi2 and parameters with 1 sigma error
parameters = np.median(samples,axis=0)  #mean(sampler.flatchain, axis = 0)
(M_BH, distance, X, Y,
 P_1, T0_1, e_1, i_1, omega_1, Omega_1,
 P_2, T0_2, e_2, i_2, omega_2, Omega_2) = parameters
parameters1 = (P_1, T0_1, e_1, i_1, omega_1, Omega_1, M_BH, distance, X, Y)
parameters2 = (P_2, T0_2, e_2, i_2, omega_2, Omega_2, M_BH, distance, X, Y)
DEC_fit_1,RA_fit_1 = orbit_interpolation(date_1, *parameters1)
DEC_fit_2,RA_fit_2 = orbit_interpolation(date_2, *parameters2)
chi2 = sum(((RA_measured_1 - RA_fit_1)/err_RA_1)**2 + ((DEC_measured_1 - DEC_fit_1)/err_DEC_1)**2)+\
 sum(((RA_measured_2 - RA_fit_2)/err_RA_2)**2+  ((DEC_measured_2 - DEC_fit_2)/err_DEC_2)**2)
chi2_red = chi2/(np.size(data1,1)+np.size(data2,1)-ndim)
print('Chi^2={:.3f}'.format(chi2_red))
samples *= [1,1,1,1, 1,1,1,180/pi,180/pi,180/pi, 1,1,1,180/pi,180/pi,180/pi]
M_BH_mcmc, distance_mcmc, X_mcmc, Y_mcmc,\
P_mcmc_1, T0_mcmc_1, e_mcmc_1, i_mcmc_1, omega_mcmc_1, Omega_mcmc_1,\
P_mcmc_2, T0_mcmc_2, e_mcmc_2, i_mcmc_2, omega_mcmc_2, Omega_mcmc_2 =\
    map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]), zip(*np.percentile(samples, [16, 50, 84], axis=0)) )
#50%(v[1]) is the mean value; 16%(v[0])~84%(v[2]) is the error range, corresponding to 1 sigma error(68%)
print('M_BH={}\nR={}\nX={}\nY={}'
       .format(M_BH_mcmc, distance_mcmc, X_mcmc, Y_mcmc))
print('P_1={}\nT0_1={}\ne_1={}\ni_1={}\nomega_1={}\nOmega_1={}'
       .format(P_mcmc_1, T0_mcmc_1, e_mcmc_1, i_mcmc_1, omega_mcmc_1, Omega_mcmc_1))
print('P_2={}\nT_0_2={}\ne_2={}\ni_2={}\nomega_2={}\nOmega_2={}'
       .format(P_mcmc_2, T0_mcmc_2, e_mcmc_2, i_mcmc_2, omega_mcmc_2, Omega_mcmc_2))

# Plot the distribution of parameters(triangle)
fig = triangle.corner(samples, 
                     labels=["$M_{BH}(10^6M_{sun})$","$R(kpc)$", "$X(mas)$", "$Y(mas)$",
"$P_1(year)$", "$T_{0_1}(year)$", "$e_1$", "$i_1(^o)$", "$\omega_1(^o)$","$\Omega_1(^o)$",
"$P_2(year)$", "$T_{0_2}(year)$", "$e_2$", "$i_2(^o)$", "$\omega_2(^o)$","$\Omega_2(^o)$"],
                     quantiles=(0.16, 0.5, 0.84)) #plot_datapoints=False
fig.savefig("triangle"+time.strftime('-%Y-%m-%d-%I:%M%p')+".png")

# Plot the data point and fitting orbit
parameters12 = (parameters1, parameters2)
labels = ('S0_2','S0_102')
fig = plt.figure(figsize=(7,7))
for i in (0,1): 
    (date, RA_measured, err_RA, DEC_measured, err_DEC) = data[i]
    (DEC,RA),Time = orbit(*parameters12[i])
    RA, DEC, RA_measured, DEC_measured, err_RA, err_DEC =\
        np.array((RA, DEC, RA_measured, DEC_measured, err_RA, err_DEC))/1000 # mas——>arcsec
    plt.errorbar(RA_measured, DEC_measured, err_DEC, err_RA, '*')
    [plt.text(x, y, ' '+str(t)) for x,y,t in zip(RA_measured,DEC_measured,date)];
    plt.plot(RA[np.arange(0,len(RA),100)], DEC[np.arange(0,len(DEC),100)],'--',label=labels[i]) #, 'b--'
plt.axis( [0.20, -0.20, -0.20, 0.20] )
plt.xlabel('$\Delta$RA(arsec)')
plt.ylabel('$\Delta$DEC(arsec)')
plt.title('fitting Orbit')
plt.legend(loc='upper left')
fig.savefig("orbit"+time.strftime('-%Y-%m-%d-%I:%M%p')+".png")







##........Test the "orbit_interpolation" function.................
##parameters = [11.5, 2009.5, 0.68, 151*pi/180, 185*pi/180, 175*pi/180, 4.1, 7.7, 0, 0]
#parameters = [15.32, 2002.4, 0.87, 48.4*pi/180, 60*pi/180, 142.6*pi/180, 4.0, 8.3, 0, 0]
#DEC,RA = orbit_interpolation(date, *parameters)
#RA, DEC, RA_measured, DEC_measured, err_RA, err_DEC =\
#np.array((RA, DEC, RA_measured, DEC_measured, err_RA, err_DEC))/1000 # mas——>arcsec
#plt.figure(figsize=(7,7))
#plt.errorbar(RA_measured, DEC_measured, err_RA, err_DEC, '*')
#[plt.text(x, y, ' '+str(t)) for x,y,t in zip(RA_measured,DEC_measured,date)]
#plt.plot(RA, DEC, 'b.--',label="fit orbit")
#plt.axis( [0.20, -0.20, -0.20, 0.20] )
#plt.show()


#for pos, lnprob, state in sampler.sample(pos, lnprob0=lnposterior, 
#                                         iterations=5, storechain=False):

#for result in sampler.sample(pos, iterations=5, storechain=False):
#    position = result[0]
#    with open(fn , "a") as f:
#        for k in range(position.shape[0]):
#            f.write("{0:4d} {1:s}\n".format(k, " ".join(position[k])))


#for i in range(ndim):
#    plt.figure()
#    plt.hist(samples[:,i], 10, color="k", histtype="step")
#    plt.title("Dimension {0:d}".format(i+1))
#plt.show()


#fig = triangle.corner(samples, labels=["$P(year)$", "$T_0(year)$", "$e$", "$i(^o)$", 
#                    "$\omega(^o)$", "$\Omega(^o)$", "$M_{BH}(M_{sun})$", "$distance(kpc)$"],
#                         quantiles=[0.16, 0.5, 0.84],
#                         show_titles=True, title_args={"fontsize": 12})
#fig.gca().annotate("A Title", xy=(0.5, 1.0), xycoords="figure fraction",
#                   xytext=(0, -5), textcoords="offset points", ha="center", va="top")
#fig.savefig("triangle"+time.strftime('-%Y-%m-%d-%I%p')+".png")
#triangle.corner的参数：truths参数的真值；