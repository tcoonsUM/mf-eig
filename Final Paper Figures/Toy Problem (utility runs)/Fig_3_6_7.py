#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 10:41:07 2025

@author: me-tcoons
"""

#%% import necessary tools
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 1200

#%% import data from u_d runs

u_d_opt_noreuse = np.load("u_d_optnin_noreuse.npy")
u_d_opt_reuse = np.load("u_d_optnin_reuse.npy")
u_d_naive_noreuse = np.load("u_d_naivenin_noreuse.npy")
u_d_naive_reuse = np.load("u_d_naivenin_reuse.npy")
u_d_nmc = np.load("u_d_NMC.npy")[:,:,0]

#%% perform operations to get averages, stdevs, variances
avg_ud_naive_noreuse = np.average(u_d_naive_noreuse,axis=0)
std_ud_naive_noreuse = np.std(u_d_naive_noreuse,axis=0)
var_ud_naive_noreuse = np.var(u_d_naive_noreuse,axis=0)

avg_ud_opt_noreuse = np.average(u_d_opt_noreuse,axis=0)
std_ud_opt_noreuse = np.std(u_d_opt_noreuse,axis=0)
var_ud_opt_noreuse = np.var(u_d_opt_noreuse,axis=0)

avg_ud_naive_reuse = np.average(u_d_naive_reuse,axis=0)
std_ud_naive_reuse = np.std(u_d_naive_reuse,axis=0)
var_ud_naive_reuse = np.var(u_d_naive_reuse,axis=0)

avg_ud_opt_reuse = np.average(u_d_opt_reuse,axis=0)
std_ud_opt_reuse = np.std(u_d_opt_reuse,axis=0)
var_ud_opt_reuse = np.var(u_d_opt_reuse,axis=0)

avg_ud_nmc = np.average(u_d_nmc,axis=0)
std_ud_nmc = np.std(u_d_nmc,axis=0)
var_ud_nmc = np.var(u_d_nmc,axis=0)

#%% Figure 3 (u_d +/-1 stdev plots, no reuse)

fig0,ax0 = plt.subplots()
totalRuns=50
n_d=40
d_vals = np.linspace(0, 1, n_d+1)

ax0.fill_between(d_vals,avg_ud_nmc-2*std_ud_nmc,avg_ud_nmc+2*std_ud_nmc, alpha=0.2, color='red',label='NMC $\pm \,2$ std')
ax0.fill_between(d_vals,avg_ud_opt_noreuse-2*std_ud_opt_noreuse,avg_ud_opt_noreuse+2*std_ud_opt_noreuse, alpha=0.2, color='green',label=r'Optimal-$\text{N}_{\text{in}} \pm 2$ std ') 
ax0.plot(d_vals,avg_ud_nmc,color='red', linewidth=1, label='NMC mean')
ax0.plot(d_vals,avg_ud_opt_noreuse,color='green', linewidth=1,label=r'Optimal-$\text{N}_{\text{in}}$ mean')
ax0.set_xlabel(r'$\xi$')
ax0.set_ylabel(r'$U$')
ax0.legend()
plt.savefig("u_d_noreuse.pdf")

#%% Figure 6 (u_d +/-1 stdev plots, with reuse)
fig1,ax1 = plt.subplots()
totalRuns=50
n_d=40
d_vals = np.linspace(0, 1, n_d+1)

ax1.fill_between(d_vals,avg_ud_nmc-2*std_ud_nmc,avg_ud_nmc+2*std_ud_nmc, alpha=0.2, color='red',label='NMC $\pm \,2$ std')
ax1.fill_between(d_vals,avg_ud_opt_reuse-2*std_ud_opt_reuse,avg_ud_opt_reuse+2*std_ud_opt_reuse, alpha=0.2, color='green',label=r'Optimal-$\text{N}_{\text{in}} \pm 2$ std ') 
ax1.plot(d_vals,avg_ud_nmc,color='red', linewidth=1, label='NMC mean')
ax1.plot(d_vals,avg_ud_opt_reuse,color='green', linewidth=1,label=r'Optimal-$\text{N}_{\text{in}}$ mean')
ax1.set_xlabel(r'$\xi$')
ax1.set_ylabel(r'$U$')
ax1.legend()
plt.savefig("u_d_reuse.pdf")

#%% Figure 7 (variance comparisons)
n_d=40
d_vals = np.linspace(0, 1, n_d+1)
fig2,ax2 = plt.subplots()
ax1.ticklabel_format(style='sci', axis='y',scilimits=(0,1))
ax1.yaxis.major.formatter._useMathText = True
ax2.semilogy(d_vals,var_ud_nmc,'r',label=r"NMC")
ax2.semilogy(d_vals,var_ud_naive_reuse,'b',label=r"$\text{Na}\ddot{\mathrm{\imath}}\text{ve}$-$\text{N}_{\text{in}}$ (reuse)")
ax2.semilogy(d_vals,var_ud_opt_reuse,'g',label=r"Optimal-$\text{N}_{\text{in}}$ (reuse)")
ax2.semilogy(d_vals,var_ud_naive_noreuse,'b',linestyle='dashed',label=r"$\text{Na}\ddot{\mathrm{\imath}}\text{ve}$-$\text{N}_{\text{in}}$ (no reuse)")
ax2.semilogy(d_vals,var_ud_opt_noreuse,'g',linestyle='dashed',label=r"Optimal-$\text{N}_{\text{in}}$ (no reuse)")
ax2.set_xlabel(r'$\xi$')
ax2.set_ylabel(r'$\mathrm{\mathbb{V}}\text{ar}[U]$')
ax2.legend(loc="center left", bbox_to_anchor=(0.,0.6))
plt.savefig("var_across_d_all.pdf")