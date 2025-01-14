#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(utils.py function code)
----------
Bayesian Model-Based Optimal Experimental Design Code
Author: Thomas Coons

Date: Summer 2022
"""

import numpy as np
import random
from scipy import stats

def sample_prior(n_sample, n_param, lb, ub, seed=3142):   
    np.random.seed(seed)
    thetas = np.random.uniform(lb, ub, (n_sample, n_param))
    return thetas

def sample_likelihood(epsilons, g):
    y_vec = g*epsilons
    return y_vec

def evaluate_log_likelihood_mult(n_samples, y, g, mean=0, sd=1e-2):
    epsilons = np.divide(y-g,g)
    eps_pdf = stats.norm.logpdf(epsilons,mean,sd)
    return eps_pdf  

def evaluate_log_likelihood_mult_2d(n_samples, y, g, mean=[0,0], cov=1e-2*np.diag([1,1])):
    epsilons = np.divide(y,g)
    eps_pdf = stats.multivariate_normal.logpdf(epsilons,mean,cov)
    return eps_pdf  

def sample_epsilon(n_samples, mean=0, sd=1e-2, seed=3141):
    # note: sd is standard deviation of Gaussian noise term 
    np.random.seed(seed)
    epsilons = np.random.normal(mean,sd,n_samples)
    epsilons = np.reshape(epsilons,(-1,1))
    return epsilons

def sample_epsilon_2d(n_samples, mean=[0,0], cov=1e-2*np.diag([1,1]), seed=3141):
    np.random.seed(seed)
    epsilons = stats.multivariate_normal.rvs(mean,cov,n_samples,random_state=seed)
    #epsilons = np.reshape(epsilons,(-1,1))
    return epsilons

def evaluate_log_epsilon(epsilon, mean=0, sd=1e-2):
    eps_pdf = stats.norm.logpdf(epsilon,mean,sd)
    return eps_pdf

def evaluate_log_epsilon_2d(epsilon, mean=[0,0], cov=1e-2*np.diag([1,1])):
    eps_pdf =  stats.multivariate_normal.logpdf(epsilon,mean,cov)
    return eps_pdf

# def eig_eps(thetas_outer, epsilons, d, n_out, n_in, n_param, lb_theta, ub_theta, g_func):
#     u_d = 0
#     for i in range(n_out):
#         eps_log_pdf = evaluate_log_epsilon(epsilons[i])
#         evidence=0
        
#         for j in range(n_in):
#             thetas_inner = sample_prior(n_in, n_param, lb_theta, ub_theta, 31415)
#             if n_out==1:
#                 eps_inner = g_func(thetas_outer,d)+epsilons[i]-g_func(thetas_inner[j,:],d)
#             else:
#                 eps_inner = g_func(thetas_outer[i],d)+epsilons[i]-g_func(thetas_inner[j,:],d)
#             evidence+=np.exp(evaluate_log_epsilon(eps_inner))
            
#         evidence/=n_in
#         u_d+=eps_log_pdf - np.log(evidence)
#     u_d/=n_out
#     return u_d

def eig_eps(epsilons, n_out, n_in, y_inner, y_outer):
    u_d = 0
    for i in range(n_out):
        eps_log_pdf = evaluate_log_epsilon(epsilons[i])
            
        eps_inners=y_outer[i]+epsilons[i]-y_inner
        evidence=np.exp(evaluate_log_epsilon(eps_inners))
        evidence=np.sum(evidence)/n_in
        u_d+=eps_log_pdf - np.log(evidence)
        
    u_d/=n_out
    return u_d

def eig_mult(epsilons, n_out, n_in, g_inner, g_outer):
    u_d = 0
    y_outer = np.multiply(g_outer,epsilons)
    for i in range(n_out):
        llh_log_pdf = evaluate_log_likelihood_mult(1, y_outer[i], g_outer[i])
            
        evidence=np.exp(evaluate_log_likelihood_mult(1, y_outer[i], g_inner))
        evidence=np.sum(evidence)/n_in
        u_d+=llh_log_pdf - np.log(evidence)
        
    u_d/=n_out
    return u_d

from mpmath import mp, exp, fsum, log
import numpy as np
from multiprocessing import Pool
import os

def evaluate_iteration(args):
    """Helper function to compute one iteration of the eig_eps_mult or eig_eps_mult_vec loop."""
    i, epsilons, y_outer, g_inner, g_outer, n_in = args
    
    eps_log_pdf = evaluate_log_epsilon(epsilons[i])
    evidence = np.exp(evaluate_log_likelihood_mult(1, y_outer[i], g_inner)) / np.abs(g_inner)
    evidence = np.sum(evidence) / n_in
    result = eps_log_pdf - np.log(np.abs(g_outer[i])) - np.log(evidence)
    
    return result

def eig_eps_mult_mp(epsilons, n_out, n_in, g_inner, g_outer):
    y_outer = g_outer + np.multiply(g_outer, epsilons)
    
    # Prepare arguments for multiprocessing
    args = [(i, epsilons, y_outer, g_inner, g_outer, n_in) for i in range(n_out)]
    
    n_workers = os.cpu_count()-2
    # Use multiprocessing to compute results
    with Pool(processes=n_workers) as pool:
        results = pool.map(evaluate_iteration, args)
    
    # Aggregate results
    u_d = np.sum(results) / n_out
    return u_d

def eig_eps_mult_vec_mp(epsilons, n_out, n_in, g_inner, g_outer):
    y_outer = g_outer + np.multiply(g_outer, epsilons)
    
    # Prepare arguments for multiprocessing
    args = [(i, epsilons, y_outer, g_inner, g_outer, n_in) for i in range(n_out)]
    
    n_workers = os.cpu_count()-2
    # Use multiprocessing to compute results
    with Pool(processes=n_workers) as pool:
        results = pool.map(evaluate_iteration, args)
    
    # Convert results to a numpy array
    u_ds = np.array(results)
    return u_ds


def eig_eps_mult(epsilons, n_out, n_in, g_inner, g_outer):
    u_d = 0
    y_outer = g_outer + np.multiply(g_outer,epsilons)
    for i in range(n_out):
        eps_log_pdf = evaluate_log_epsilon(epsilons[i])
        #llh_log_pdf = np.log(np.exp(eps_log_pdf)/abs(g_outer[i]))
            
        evidence=np.exp(evaluate_log_likelihood_mult(1, y_outer[i], g_inner))/np.abs(g_inner)
        evidence=np.sum(evidence)/n_in
        u_d+=eps_log_pdf - np.log(np.abs(g_outer[i])) - np.log(evidence)
        # if evidence.all()==0.:
        #     mp.dps = 20
        #     evidence_list = []
        #     for j in range(n_in):
        #         evidence_list.append(exp(evaluate_log_likelihood_mult(1, y_outer[i], g_inner[j]))/np.abs(g_inner[j]))
        #     evidence = fsum(evidence_list)
        #     log_ev = float(log(evidence))
        #     u_d+=eps_log_pdf - np.log(np.abs(g_outer[i])) - log_ev
        # else:
        #     u_d+=eps_log_pdf - np.log(np.abs(g_outer[i])) - np.log(evidence)
        #u_d+=eps_log_pdf - np.log(evidence*g_outer[i])
        #u_d+=eps_log_pdf - np.log(evidence)
        #u_d+=llh_log_pdf - np.log(evidence)
        
    u_d/=n_out
    return u_d

def eig_eps_mult_vec(epsilons, n_out, n_in, g_inner, g_outer):
    u_ds = np.zeros((n_out,))
    u = 0
    y_outer = g_outer + np.multiply(g_outer,epsilons)
    for i in range(n_out):
        eps_log_pdf = evaluate_log_epsilon(epsilons[i])
        #llh_log_pdf = np.log(np.exp(eps_log_pdf)/abs(g_outer[i]))
            
        evidence=np.exp(evaluate_log_likelihood_mult(1, y_outer[i], g_inner))/np.abs(g_inner)
        evidence=np.sum(evidence)/n_in
        u+=eps_log_pdf - np.log(np.abs(g_outer[i])) - np.log(evidence)
        u_ds[i]=u
        u=0
        #u_d+=eps_log_pdf - np.log(evidence*g_outer[i])
        #u_d+=eps_log_pdf - np.log(evidence)
        #u_d+=llh_log_pdf - np.log(evidence)
        
    return u_ds

def eig_eps_mult_2d(epsilons, n_out, n_in, g_inner, g_outer):
    u_d = 0
    #ydim = np.shape(epsilons)[1]
    y_outer = np.multiply(g_outer,epsilons)
    for i in range(n_out):
        eps_log_pdf = evaluate_log_epsilon_2d(epsilons[i,:])
            
        evidence=np.exp(evaluate_log_likelihood_mult_2d(1, y_outer[i,:], g_inner))/abs(g_inner[:,0]*g_inner[:,1])
        evidence=np.sum(evidence)/n_in
        u_d+=eps_log_pdf - np.log(abs(g_outer[i,0]*g_outer[i,1])) - np.log(evidence)
        
    u_d/=n_out
    return u_d

def eig_mult_2d(epsilons, n_out, n_in, g_inner, g_outer):
    u_d = 0
    #ydim = np.shape(epsilons)[1]
    y_outer = np.multiply(g_outer,epsilons)
    for i in range(n_out):
        eps_log_pdf = evaluate_log_epsilon_2d(epsilons[i,:])
            
        evidence=np.exp(evaluate_log_likelihood_mult_2d(1, y_outer[i,:], g_inner))
        evidence=np.sum(evidence)/n_in
        u_d+=eps_log_pdf - np.log(evidence)
        
    u_d/=n_out
    return u_d
