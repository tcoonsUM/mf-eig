#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 17:34:09 2025

@author: me-tcoons
"""
#%% imports
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
plt.rcParams['figure.dpi'] = 1200
plt.set_cmap("jet")

min_val = 3.389e-5
max_val = 6.074e-4

#%% parameters
n_nin = 41
n1 = np.linspace(25,4025,n_nin); n1[25]=2500
n2 = np.linspace(25,4025,n_nin); n2[25]=2500
w = np.array([1, 0.1, 0.01])

#%% load data
variances = np.load("variances.npy")
corrs12 = np.load("corrs12.npy")
corrs1 = np.load("corrs1.npy")
corrs2 = np.load("corrs2.npy")

#%% N_in search contour plot
maxInds = np.where(variances==variances.min())
print("Best estimator occurs at nIn_1 = "+str(n1[maxInds[0]])+" & nIn_2 = "+str(n1[maxInds[1]])+", var = "+str(np.min(variances)))
norm=colors.PowerNorm(gamma=1/3,vmin=min_val,vmax=max_val)
levels = norm.inverse(np.linspace(0,1,101))
cs = plt.contourf(n2,n1,variances.T,101,cmap='jet',levels=levels, norm=norm)
#cs = plt.pcolormesh(n2,n1,variances.T,cmap='jet', norm=norm)
plt.xlabel(r'$N_{\text{in},1}$')
plt.ylabel(r'$N_{\text{in},2}$')
cb = plt.colorbar(cs,ticks=norm.inverse(np.linspace(0,1,10)),format='%.2e')
#cb.ax.set_yticklabels(powspace(min_val, max_val, 3, 9),format='%.2e')
plt.savefig("vars_reuse.pdf")

#%% corrs12 contour plot

fig12, ax12 = plt.subplots()
norm=colors.PowerNorm(gamma=4,vmin=np.min(0),vmax=np.max(1))
levels = norm.inverse(np.linspace(0,1,101))
cs = plt.contourf(n2,n1,corrs12.T,51,cmap='jet',norm=norm,levels=levels)
fig12.colorbar(cs,ticks=norm.inverse(np.linspace(0,1,10)),format='%.2f')
plt.xlabel(r'$N_{\text{in},1}$')
plt.ylabel(r'$N_{\text{in},2}$')
plt.savefig("corrs12_reuse.pdf")

#%% corrcosts tradeoffs plot
igc, axc = plt.subplots()
twin1 = axc.twinx()
axc.plot(n1,corrs1[:,0],label='m=1')
axc.plot(n2,corrs2[0,:],label='m=2')
twin1.plot(n1,(n1+1)*w[1],'--',label='m=1')
twin1.plot(n2,(n2+1)*w[2],'--',label='m=2')
axc.set_xlabel(r'$N_{\text{in},m}$')
axc.set_ylabel(r'$\rho_{0,m}$')
axc.legend(loc=7)
twin1.set_ylabel(r'$w_{m}$')
plt.savefig("corrcosts_reuse.pdf")