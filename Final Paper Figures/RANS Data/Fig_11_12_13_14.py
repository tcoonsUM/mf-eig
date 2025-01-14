#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 12:13:59 2025

@author: me-tcoons
"""

#%% imports
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as colors
from matplotlib import colormaps
plt.rcParams['figure.dpi'] = 1200

#%% load data
uDSF = np.load('uFinals_sf.npy').transpose()
uDMF_naive = np.load('uFinals_naive.npy').transpose()
uDMF_opt = np.load('uFinals_opt.npy').transpose()
xy=np.load('xy.npy')
utilities = np.load('bigY.npy')

#%% problem parameters/definitions

nTrials = 21
nSensors = 875
n0 = 74
w=np.flip([1.9154183389782818, 27.39698237466692, 298.18460526719235, 1778.4654708378243])
maxpoints = np.zeros((2,nTrials))
n_LF = 3

#%% finding some statistics
# main results
means = np.mean(uDMF_opt,axis=1); meansSF = np.mean(uDSF,axis=1); meansNaive = np.mean(uDMF_naive,axis=1)
variances = np.var(uDMF_opt,axis=1); variancesSF = np.var(uDSF,axis=1); variancesNaive = np.var(uDMF_naive,axis=1)

# compute top sensors from MF-EIG (opt) results
order=np.argsort(means); order=np.flip(order)
topSensors = order[:8]

# best point
maxpoint=xy[np.argmax(means),:]
varAtOptimum = np.var(uDSF[np.argmax(means),:])

# correlations across models
covs_all = np.zeros((n_LF+1,n_LF+1,nSensors))
corrs_all = np.zeros((n_LF+1,n_LF+1,nSensors))
for sensor in range(nSensors): 
    covs_all[:,:,sensor] = np.cov(utilities[:,:,sensor]); 
    corrs_all[:,:,sensor] = np.corrcoef(utilities[:,:,sensor])
corr_avg = np.mean(corrs_all[:,0,:],axis=0)

#%% 11a - no logy EIG averages
# Assuming these variables are already defined: variances, corr_avg, means, xy, maxpoint
plotVariable = means  
fig1, ax1 = plt.subplots(dpi=1200)

# Use the updated Colormap API
cm_name = 'jet'
colormap = colormaps[cm_name]  # 'plasma' or 'viridis' can also be used
color = colormap(plotVariable)

# Create a ScalarMappable for the colorbar
sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0., vmax=1.5))
ax1.tricontourf(xy[:, 0], xy[:, 1], plotVariable, levels=200, cmap=cm_name, norm=plt.Normalize(vmin=0., vmax=1.5))

# Highlight the maximum point (optional)
cmax = colormap(plotVariable[np.argmax(means)])
# Uncomment the next line if you want to mark the maximum point
# plt.scatter(maxpoint[0], maxpoint[1], c='white', marker='*', s=50)

# Add the colorbar
plt.colorbar(sm, ax=ax1)

# Set scale, labels, and titles
ax1.set_xlabel(r'$x_0$')
ax1.set_ylabel(r'$x_1$')

# Set limits on the y-axis
ax1.set_ylim([10**-5, 0.1])

# Display the plot
plt.savefig("tricontour_means_nology.pdf")

#%% 11b - logy
plotVariable = means  
fig2, ax2 = plt.subplots(dpi=1200)

# Use the updated Colormap API
cm_name = 'jet'
colormap = colormaps[cm_name]  # 'plasma' or 'viridis' can also be used
color = colormap(plotVariable)

# Create a ScalarMappable for the colorbar
sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0., vmax=1.5))
ax2.tricontourf(xy[:, 0], xy[:, 1], plotVariable, levels=200, cmap=cm_name, norm=plt.Normalize(vmin=0., vmax=1.5))

# Highlight the maximum point (optional)
cmax = colormap(plotVariable[np.argmax(means)])
# Uncomment the next line if you want to mark the maximum point
# plt.scatter(maxpoint[0], maxpoint[1], c='white', marker='*', s=50)

# Add the colorbar
plt.colorbar(sm, ax=ax2)

# Set scale, labels, and titles
ax2.set_yscale('log')
ax2.set_xlabel(r'$x_0$')
ax2.set_ylabel(r'$x_1$')

# Set limits on the y-axis
ax2.set_ylim([10**-5, 0.1])

# Display the plot
plt.savefig("tricontour_means.pdf")

#%% 12a - VRRs of Naive-Nin
plotVariable = variancesSF/variancesNaive 
fig3, ax3 = plt.subplots(dpi=1200)

# Use the updated Colormap API
cm_name = 'jet'
colormap = colormaps[cm_name]  # 'plasma' or 'viridis' can also be used
color = colormap(plotVariable)

# Create a ScalarMappable for the colorbar
sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0., vmax=plotVariable.max()))
ax3.tricontourf(xy[:, 0], xy[:, 1], plotVariable, levels=200, cmap=cm_name, norm=plt.Normalize(vmin=0., vmax=plotVariable.max()))

# Highlight the maximum point (optional)
cmax = colormap(plotVariable[np.argmax(means)])
# Uncomment the next line if you want to mark the maximum point
# plt.scatter(maxpoint[0], maxpoint[1], c='white', marker='*', s=50)

# Add the colorbar
plt.colorbar(sm, ax=ax3)

# Set scale, labels, and titles
ax3.set_yscale('log')
ax3.set_xlabel(r'$x_0$')
ax3.set_ylabel(r'$x_1$')

# Set limits on the y-axis
ax3.set_ylim([10**-5, 0.1])

# Display the plot
plt.savefig("vrr_naive.pdf")

#%% 12b - VRRs of Naive-Nin
plotVariable = variancesSF/variances
fig4, ax4 = plt.subplots(dpi=1200)

# Use the updated Colormap API
cm_name = 'jet'
colormap = colormaps[cm_name]  # 'plasma' or 'viridis' can also be used
color = colormap(plotVariable)

# Create a ScalarMappable for the colorbar
sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0., vmax=plotVariable.max()))
ax4.tricontourf(xy[:, 0], xy[:, 1], plotVariable, levels=200, cmap=cm_name, norm=plt.Normalize(vmin=0., vmax=plotVariable.max()))

# Highlight the maximum point (optional)
cmax = colormap(plotVariable[np.argmax(means)])
# Uncomment the next line if you want to mark the maximum point
# plt.scatter(maxpoint[0], maxpoint[1], c='white', marker='*', s=50)

# Add the colorbar
plt.colorbar(sm, ax=ax2)

# Set scale, labels, and titles
ax4.set_yscale('log')
ax4.set_xlabel(r'$x_0$')
ax4.set_ylabel(r'$x_1$')

# Set limits on the y-axis
ax4.set_ylim([10**-5, 0.1])

# Display the plot
plt.savefig("vrr_opt.pdf")

#%% Figure 13 - average correlations to u_0

plotVariable = corr_avg
fig4, ax4 = plt.subplots(dpi=1200)

# Use the updated Colormap API
cm_name = 'jet'
colormap = colormaps[cm_name]  # 'plasma' or 'viridis' can also be used
color = colormap(plotVariable)

# Create a ScalarMappable for the colorbar
sm = plt.cm.ScalarMappable(cmap=colormap, norm=plt.Normalize(vmin=0., vmax=1.))
ax4.tricontourf(xy[:, 0], xy[:, 1], plotVariable, levels=200, cmap=cm_name, norm=plt.Normalize(vmin=0., vmax=1.))

# Highlight the maximum point (optional)
cmax = colormap(plotVariable[np.argmax(means)])
# Uncomment the next line if you want to mark the maximum point
# plt.scatter(maxpoint[0], maxpoint[1], c='white', marker='*', s=50)

# Add the colorbar
plt.colorbar(sm, ax=ax4)

# Set scale, labels, and titles
ax4.set_yscale('log')
ax4.set_xlabel(r'$x_0$')
ax4.set_ylabel(r'$x_1$')

# Set limits on the y-axis
ax4.set_ylim([10**-5, 0.1])

# Display the plot
plt.savefig("corrs_contour.pdf")

#%% Figure 14a - SF violin plots

fig5, ax5 = plt.subplots(dpi=1200)
ax5.violinplot(np.transpose(uDSF[topSensors,:]),showmeans=False)
labels=np.arange(1,9)
ax5.set_xticks(np.arange(1,len(labels)+1),labels=labels)
ax5.set_xlabel("Design Ranking")
ax5.set_ylabel(r'$U$')

plt.savefig("nmc_violin.pdf")

#%% figure 14b - Naive MF-EIG

fig6, ax6 = plt.subplots(dpi=1200)
ax6.violinplot(np.transpose(uDMF_naive[topSensors,:]),showmeans=False)
labels=np.arange(1,9)
ax6.set_xticks(np.arange(1,len(labels)+1),labels=labels)
ax6.set_xlabel("Design Ranking")
ax6.set_ylabel(r'$U$')

plt.savefig("naive_violin.pdf")

#%% figure 14b - Optimal MF-EIG

fig56, ax7 = plt.subplots(dpi=1200)
ax7.violinplot(np.transpose(uDMF_opt[topSensors,:]),showmeans=False)
labels=np.arange(1,9)
ax7.set_xticks(np.arange(1,len(labels)+1),labels=labels)
ax7.set_xlabel("Design Ranking")
ax7.set_ylabel(r'$U$')

plt.savefig("opt_violin.pdf")
