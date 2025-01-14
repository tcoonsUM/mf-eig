#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
(Forward Model Functions)
-----------
Bayesian Model-Based Optimal Experimental Design Code
Author: Thomas Coons

Date: Summer 2022
"""
import numpy as np

#%% Xun function, n_param=1
def xuns_function(theta, d):
    first_term = theta**3*d**2
    second_term = theta*np.exp(-1*np.abs(0.2-d))
    return first_term + second_term

def xun2(theta, d):
    first_term = theta**3*d
    second_term = theta*np.exp(-1*np.abs(0.2-d))
    return first_term + second_term

def xun3(theta, d):
    first_term = theta**5*d
    second_term = theta*np.exp(-1*np.abs(0.2-d))
    return first_term + second_term

def xunL1(theta,d):
    first_term = theta**3*d**2
    second_term = theta*(0.81873 + 0.81873*d + 0.40936*(d**2) + 0.13645*(d**3))
    return first_term + second_term

def xunL2(theta,d):
    first_term = theta**3*d**2
    second_term = theta*(0.81873 + 0.81873*d + 0.40936*(d**2))
    return first_term + second_term

def xunL3(theta,d):
    first_term = theta**3*d**2
    second_term = theta*(0.81873 + 0.81873*d)
    return first_term + second_term

#%% Xun function, n_param=2
def xuns_function_2d(theta, d):
    z1 = theta[0]; z2=theta[1]
    first_term = z1**3*d**2
    second_term = z2*np.exp(-1*np.abs(0.2-d))
    return first_term + second_term

def xun2_2d(theta, d):
    z1 = theta[0]; z2=theta[1]
    first_term = z1**3*d
    second_term = z2*np.exp(-1*np.abs(0.2-d))
    return first_term + second_term

def xun3_2d(theta, d):
    z1 = theta[0]; z2=theta[1]
    first_term = z1**3*d
    second_term = z2
    return first_term + second_term

#%% Ishigami function, n_param=3
def ish1(theta, d):
    z1 = theta[0]; z2=theta[1]; z3=theta[2]
    y=(np.sin(z1) + 5*np.sin(z2**2) + 0.1*z3**4*np.sin(z1)) * d**4;
    return y

def ish2(theta, d):
    z1 = theta[0]; z2=theta[1]; z3=theta[2]
    y=(np.sin(z1) + 4.75*np.sin(z2**2) + 0.1*z3**4*np.sin(z1)) * d**4;
    return y

def ish3(theta, d):
    z1 = theta[0]; z2=theta[1]; z3=theta[2]
    y=(np.sin(z1) + 3*np.sin(z2**2) + 0.9*z3**2*np.sin(z1)) * d**4;
    return y