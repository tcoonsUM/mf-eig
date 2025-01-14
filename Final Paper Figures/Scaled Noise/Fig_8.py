#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 12:01:36 2025

@author: me-tcoons
"""

#%% imports
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 1200


#%% load data/definitions from scaled noise runs
u_d = np.load("u_d_mult_big.npy")[:,:,1]
u_d_NMC = np.load("u_d_NMC_mult_big.npy")[:,:,1]
lb=0
ub=1
n_d=40
d_vals = np.linspace(lb, ub, n_d+1)
totalRuns = 50

#%% +/12 stdev u_d plot
avg_ud_NMC = np.average(u_d_NMC,axis=0)
std_ud_NMC =  np.std(u_d_NMC,axis=0)
avg_ud = np.average(u_d,axis=0)
std_ud = np.std(u_d,axis=0)
fig2, ax2 = plt.subplots()
#ax2.fill_between(d_vals,avg_r0-std_r0,avg_r0+std_r0, alpha=0.3, color='blue',label='no reuse')
#ax2.fill_between(d_vals,avg_r1-std_r1,avg_r1+std_r1, alpha=0.3, color='green',label='across-fid inner reuse')
ax2.fill_between(d_vals,avg_ud_NMC-2*std_ud_NMC,avg_ud_NMC+2*std_ud_NMC, alpha=0.2, color='red',label=r'NMC $\pm 2$ std')
ax2.fill_between(d_vals,avg_ud-std_ud,avg_ud+std_ud, alpha=0.3, color='green',label=r'MF-EIG $\pm 2$ std')
#ax2.plot(d_vals,u_d0,linewidth=.5,color='black',linestyle='dashed',label='DNMC, N_in=2.5K, N_out=10K')
ax2.plot(d_vals,avg_ud_NMC,color='red', linewidth=0.5, label='NMC mean')
ax2.plot(d_vals,avg_ud,color='green', linewidth=0.5, label='MF-EIG mean')
#ax2.plot(d_vals,avg_r1,color='green', linewidth=0.5)
#ax2.plot(d_vals,avg_r2,color='purple', linewidth=0.5)
#ax2.set_title('MF-EIG estimator')
#ax2.set_title('Single- vs. multi-fidelity estimator of EIG')
ax2.set_xlabel(r'$\xi$')
ax2.set_ylabel(r'$U$')
ax2.legend(loc=4)
plt.savefig("u_d_mult.pdf")

#%% Variance plot
var_ud_NMC = np.var(u_d_NMC,axis=0)
var_ud = np.var(u_d,axis=0)
fig4, ax4 = plt.subplots()
ax4.semilogy(d_vals, var_ud_NMC, color='red', label='NMC')
ax4.semilogy(d_vals, var_ud, color='green', label='MF-EIG')
ax4.set_xlabel(r'$\xi$')
ax4.set_ylabel(r'$\mathrm{\mathbb{V}}\text{ar}[U]$')
ax4.legend()
plt.savefig("var_across_d_mult.pdf")


