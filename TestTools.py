#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os.path
from scipy import *
from scipy import integrate as integrate
import numpy as np
from numpy import *
from numpy import linalg as LA
from scipy import linalg as LA2
import sympy as sympy
import sys as sys
import time
import matplotlib.pyplot as plt
import itertools as it
from IPython.core.display import HTML
import BasicTools as BT
import WaveTools as WT
import PlotTools as PT
import FFTTools as FFTT
import OperatorTools as OT
import GridTransferTools as GTT
import SolverTools as ST


# This function returns a Gaussian waveform with standard deviation `sigma` centered about `mu`. As the default, it returns the cell-averaged values of the Gaussian using Boole's Rule.

# In[2]:


def Gauss(omega, sigma, mu, cellAve = True):
    xCell = omega.xCell
    nh = omega.nh[::-1][0]
    if (cellAve):
        x = np.linspace(0., 1., num = (4 * nh) + 1)
    else:
        x = xCell
    gauss = np.exp(-((x - mu)**2) / (2. * (sigma**2)))
    if (cellAve):
        gauss = BoolesAve(gauss)
    return gauss


# This function uses Boole's Rule to return the cell average of some function `f`.

# In[3]:


def BoolesAve(f):
    errorLoc = 'ERROR:\nTestTools:\nBoolesAve:\n'
    if (len(f) % 4 != 1):
        sys.exit(errorLoc + 'f must be one more than integer multiple of four in length!')
    f_ave = (1. / 90.) * ((7 * f[:-1:4]) + (32 * f[1::4]) + (12 * f[2::4]) + (32 * f[3::4]) + (7 * f[4::4]))
    return f_ave


# This function calculates either the absolute or percent error between the theoretical and actual solutions.

# In[4]:


def CalcError(omega, theoretical, actual, errorType = 'absolute', tol = 1e-14):
    # Check size of theoretical and actual.
    nh = omega.nh_max
    error = abs(actual - theoretical)
    if (errorType == 'relative'):
        error = abs(error / (theoretical + tol))
    ks = np.linspace(0, (nh / 2) - 1, num = nh)
    return ks, error


# This function calculates the error given by an arbitrary time solver as compared to that given through Fourier analysis.

# In[5]:


def NormVersusCFL(func, omega, waves, u_0, const, CFL_0, nt_0, normType = 'max', errorType = 'absolute', plot = False):
    # Check size of waves and u_0, and check that nt_0 isn't negative?
    errorLoc = 'ERROR:\nTestTools:\nNormVersusCFL:\n'
    nt = nt_0
    CFL = CFL_0
    nh = omega.nh_max
    dx = omega.dx[0]
    FCoefs = FFTT.FourierCoefs(omega, waves, u_0)
    norms = []
    CFLs = []
    while (CFL > 0.1):
        calcCoefs, t = func(omega, waves, u_0, nt, const, CFL = CFL)
        if (CFL == CFL_0):
            t_0 = t
        propCoefs = FFTT.PropogateFCoefs(omega, FCoefs, const, t)
        ks, error = CalcError(omega, propCoefs, calcCoefs, errorType = errorType)
        if (plot):
            if (nt % 10 == 0):
                allCoefs = PT.Load(FCoefs, propCoefs, calcCoefs)
                title = 'CFL = ' + str(CFL)
                PT.PlotMixedWave(omega, waves, allCoefs, rescale = [2, 3], title = title, labels = [r'$u_{0} (x)$', r'Reference $u_{0} (x - c t)$', r'Solver $u_{0} (x - c t)$'])
        if (normType == 'max'):
            norm = max(error)
        else:
            norm = sum(error) / nh
        norms.append(norm)
        CFLs.append(CFL)
        nt = nt + 1
        CFL = (CFL * const * t_0) / ((CFL * dx) + (const * t_0))
        if (not np.isclose(t, t_0, atol = 1e-15, rtol = 0)):
            errorMess = 't does not match t_0!\nt_0 = ' + str(t_0) + '\nt = ' + str(t)
            sys.exit(errorLoc + errorMess)
    return norms, CFLs


# This function calculates either the amplitude error between the theoretical and actual solutions.

# In[6]:


def AmpError(omega, theoretical, actual, tol = 1e-20):
    # Check size of theoretical and actual.
    nh = omega.nh_max
    numKs = int((nh / 2) + 1)
    error = np.zeros(numKs, float)
    error[0] = 1 - abs(actual[0] / np.sqrt((theoretical[0]**2) + tol))
    error[::-1][0] = 1 - abs(actual[::-1][0] / np.sqrt((theoretical[::-1][0]**2) + tol))
    error[1:-1] = 1 - np.sqrt(((actual[1::2][:-1]**2) + (actual[::2][1:]**2)) / ((theoretical[1::2][:-1]**2) + (theoretical[::2][1:]**2) + tol))
    ks = np.arange(numKs)
    return ks, error


# In[ ]:
