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
import BasicTools as BT
import WaveTools as WT
import PlotTools as PT
import FFTTools as FFTT


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




