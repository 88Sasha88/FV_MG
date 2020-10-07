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


# This function checks to make sure that sizes match up appropriately.

# In[2]:


def CheckSize(n, matrica):
    dim = size(shape(matrica))
    problem = 0
    for i in range(dim):
        if (n != shape(matrica)[i]):
            problem = problem + 1
    return problem


# This function ensures that $n^{h}$ is an appropriate base-2 value.

# In[3]:


def CheckNumber(nh):
    check = nh
    while (check % 2 == 0):
        check = check / 2
    if (check != 1):
        sys.exit('ERROR:\nnh must be a base-2 integer!')
    return


# This function checks if a given matrix is diagonal.

# In[4]:


def CheckDiag(matrica):
    if (np.size(np.shape(matrica)) != 2):
        sys.exit('ERROR:\nBasicTools:\nCheckDiag:\nmatrica must be a rank-2 array!')
    if (np.shape(matrica)[0] != np.shape(matrica)[1]):
        sys.exit('ERROR:\nBasicTools:\nCheckDiag:\nmatrica must be a square array!')
    i, j = np.nonzero(matrica)
    if (np.all(i == j)):
        problem = 0
    else:
        problem = 1
    return problem


# This function checks if a given matrix is diagonal.

# In[5]:


def CheckBounds(nh_max, bounds):
    problem = 0
    loops = len(bounds)
    levels = int(np.log(nh_max) / np.log(2))
    if (loops != levels):
        problem = 1
    for i in range(loops):
        if (len(bounds[i]) != 2):
            problem = 2
#     if ((bounds > nh_max).any()):
#         problem = 3
    return problem


# This function outputs an $x$ array and a $y$ array of size $n^{h}$ + 1 of the locations of the tick marks.

# In[6]:


def MakeXY(nh_max, xBounds = []):
    if (xBounds == []):
        loops = int(np.log(nh_max) / np.log(2))
        x = np.linspace(0, 1, num = nh_max + 1)
    else:
        loops = len(xBounds)
        problem = CheckBounds(nh_max, xBounds)
        if (problem == 1):
            sys.exit('ERROR:\nBasicTools:\nMakeXY:\nnh_max does not match up with log of shape of xBounds!')
        if (problem == 2):
            sys.exit('ERROR:\nBasicTools:\nMakeXY:\nAll elements in xBounds must be of length 2!')
        if (problem == 3):
            sys.exit('ERROR:\nBasicTools:\nMakeXY:\nValues in xBounds must be less than nh_max!')
        x = []
        for i in range(loops):
            h = 2**-(i + 1)
            n = xBounds[i][1] - xBounds[i][0] + 1
            xMin = h * xBounds[i][0]
            xMax = h * xBounds[i][1]
            if (i > 0):
                n = n - 2
                xMin = xMin + h
                xMax = xMax - h
            xPiece = np.linspace(xMin, xMax, num = n)
            x = sorted(set(np.append(x, xPiece)))
    n_max = np.shape(x)[0]
    y = np.zeros(n_max, float)
    return x, y


# In[ ]:




