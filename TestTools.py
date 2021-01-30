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

