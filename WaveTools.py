#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os.path
from scipy import *
import numpy as np
from numpy import *
from numpy import linalg as LA
import sys as sys
import time
import matplotlib.pyplot as plt
import BasicTools as BT


# This function creates a matrix of cell-centered Fourier modes along with a linear space of the cell locations.

# In[2]:


def MakeWaves(nh, h):
    x, y = BT.MakeXY(nh)
    waves = np.zeros((nh, nh), float)
    xCell = x[0:nh] + (h / 2.)
    for k in range(int(nh / 2)):
        waves[:, (2 * k) + 1] = (1.0 / (2.0 * np.pi * (k + 1) * h)) * (cos(2 * np.pi * (k + 1) * x[0:nh]) - cos(2 * np.pi * (k + 1) * x[1:nh + 1]))
        if (k == 0):
            waves[:, 2 * k] = np.ones(nh, float)
        else:
            waves[:, 2 * k] = (1.0 / (2.0 * np.pi * k * h)) * (sin(2 * np.pi * k * x[1:nh + 1]) - sin(2 * np.pi * k * x[0:nh]))
    return xCell, waves


# This function creates a matrix of node-centered Fourier modes along with a linear space of the node locations.

# In[3]:


def MakeNodeWaves(nh, h):
    x = np.linspace(0, 1. - (1. / nh), num = nh)
    waves = np.zeros((nh, nh), float)
    for k in range(int(nh / 2)):
        waves[:, (2 * k) + 1] = np.sin(2 * np.pi * (k + 1) * x)
        if (k == 0):
            waves[:, 2 * k] = np.ones(nh, float)
        else:
            waves[:, 2 * k] = np.cos(2 * np.pi * k * x)
    return x, waves


# In[ ]:




