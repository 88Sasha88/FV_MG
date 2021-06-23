#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os.path
from scipy import *
import numpy as np
from numpy import *
from numpy import linalg as LA
from scipy import linalg as LA2
import sys as sys
import time
import matplotlib.pyplot as plt
import itertools as it
from IPython.core.display import HTML
from Modules import BasicTools as BT
from Modules import WaveTools as WT
from Modules import PlotTools as PT
from Modules import FFTTools as FFTT


# This function creates an injection operator.

# In[2]:


def MakeInject(nh):
    n2h = int(nh / 2)
    inject = np.zeros((n2h, nh), float)
    for i in range(n2h):
        inject[i, (2 * i) + 1] = 1
    return inject


# This function creates a full-weighting operator.

# In[3]:


def MakeFullWeight(nh):
    n2h = int(nh / 2)
    fullWeight = np.zeros((n2h, nh), float)
    weights = [0.5, 0.5]
    for i in range(n2h):
        fullWeight[i, (2 * i):(2 * i) + 2] = weights
    return fullWeight


# This function creates a piecewise interpolation operator.

# In[4]:


def MakePiecewise(nh):
    nh2 = 2 * nh
    piecewise = np.zeros((nh2, nh), float)
    weights = [1, 1]
    for i in range(nh):
        piecewise[(2 * i):(2 * i) + 2, i] = weights
    return piecewise


# This function creates a linear interpolation operator.

# In[5]:


def MakeLinearInterp(nh):
    nh2 = 2 * nh
    linearInterp = np.zeros((nh2, nh), float)
    weights = [1, 1]
    for i in range(nh):
        linearInterp[(2 * i):(2 * i) + 2, i] = weights
    return linearInterp


# In[ ]:

def CoarsenOp(omega):
    hs = omega.h
    nh_max = omega.nh_max
    h_min = 1. / nh_max
    weights = h_min / hs
    sizes = np.asarray(1 / weights, int)
    matrices = [w * np.ones(s, float) for (w, s) in zip(weights, sizes)]
    CoarseOp = LA2.block_diag(*matrices)
    return CoarseOp

def BoundVals(order, x_0):
    if ((order + 1) % 3 == 0):
        n_c = int(np.floor((order + 1) / 3))
    else:
        n_c = int(np.floor(((order + 1) / 3) + 1))
    n_f = order + 1 - n_c
    bounds = np.linspace(-n_c, n_f / 2., num = (2 * n_c) + n_f + 1)
    rm = [(2 * k) + 1 for k in range(n_c)]
    bounds = np.delete(bounds, rm)
    if (x_0 > 0):
         bounds = -bounds[::-1]
    return bounds

# Put this in GTT.
def MomentVander(order, bounds, xVec):
    # Add error catchers!
    intCoefs = (np.arange(order + 1) + 1)[::-1]**-1.
    polyCoefs = np.diag(intCoefs)
    h = (bounds[1:] - bounds[:-1])**-1.
    hInv = np.diag(h)
    A = np.diag(bounds[1:]) @ np.vander(bounds[1:])
    B = np.diag(bounds[:-1]) @ np.vander(bounds[:-1])
    VanderMat = hInv @ (A - B) @ polyCoefs
    polyInterp = xVec @ LA2.inv(VanderMat)
    return polyInterp

def GhostCellStencil(order, x_0):
    intCoefs = (np.arange(order + 1) + 1)[::-1]**-1.
    polyCoefs = np.diag(intCoefs)
    xVec = np.polynomial.polynomial.polyvander(x_0, order)[0][::-1] @ polyCoefs
    bounds = BoundVals(order, x_0)
    polyInterp = MomentVander(order, bounds, xVec)
    return polyInterp




