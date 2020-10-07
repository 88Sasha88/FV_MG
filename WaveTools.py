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
import OperatorTools as OT


# This function creates a matrix of cell-centered Fourier modes along with a linear space of the cell locations.

# In[2]:


def MakeWaves(nh, h):
    x, y = BT.MakeXY(nh)
    xCell = 0.5 * (x[0:nh] + x[1:nh + 1])
    waves = CellWaves(nh, x)
    hs = nh * np.eye(nh)
    hs[0, 0] = 1.
    waves = waves @ hs
    return xCell, waves


# This function creates a matrix of cell-centered Fourier modes along with a linear space of the cell locations.

# In[2]:


def CellWaves(nh_max, x):
    n = np.shape(x)[0] - 1
    waves = np.zeros((n, nh_max), float)
    for k in range(int(nh_max / 2)):
        waves[:, (2 * k) + 1] = (1.0 / (2.0 * np.pi * (k + 1))) * (cos(2 * np.pi * (k + 1) * x[0:n]) - cos(2 * np.pi * (k + 1) * x[1:n + 1]))
        if (k == 0):
            waves[:, 2 * k] = np.ones(n, float)
        else:
            waves[:, 2 * k] = (1.0 / (2.0 * np.pi * k)) * (sin(2 * np.pi * k * x[1:n + 1]) - sin(2 * np.pi * k * x[0:n]))
    return waves


# This function creates a matrix of node-centered Fourier modes along with a linear space of the node locations.

# In[4]:


def MakeNodeWaves(nh, h, nRes = 0):
    if (nRes == 0):
        nRes = nh
        xMax = 1. - (1. / nRes)
    else:
        xMax = 1.
    x = np.linspace(0, xMax, num = nRes)
    waves = np.zeros((nRes, nh), float)
    for k in range(int(nh / 2)):
        waves[:, (2 * k) + 1] = np.sin(2 * np.pi * (k + 1) * x)
        if (k == 0):
            waves[:, 2 * k] = np.ones(nRes, float)
        else:
            waves[:, 2 * k] = np.cos(2 * np.pi * k * x)
    return x, waves


# This function creates an array of all the $k$ values at their respective index locations.

# In[5]:


def MakeKs(nh):
    kVals = list(np.arange(nh))
    kVals = kVals
    ks = [int((i + 1) / 2) for i in kVals]
    return ks


# This function takes in a matrix of wave vectors and finds their respective eigenvalues as they relate to the Laplacian matrix. Note that this function assumes that the input matrix comprises Fourier modes but includes an error catcher in case the eigenvalues don't turn out as expected within a $10^{-14}$ tolerance.

# In[6]:


def FindLaplaceEigVals(nh, h, waves):
    problem = BT.CheckSize(nh, waves)
    if (problem != 0):
        sys.exit('ERROR:\nWaveTools:\nFindLaplaceEigVals:\nnh does not match size of waves!')
    wavesNorm = OT.NormalizeMatrix(nh, waves)
    wavesInv = wavesNorm.conj().T
    Laplacian = OT.MakeLaplacian1D(nh)
    eigMat = np.round(wavesInv @ Laplacian @ wavesNorm, 14)
    problem = BT.CheckDiag(eigMat)
    if (problem != 0):
        sys.exit('ERROR:\nWaveTools:\nFindLaplaceEigVals:\neigMat is not diagonal!')
    eigvals = np.diag(eigMat)
    ks = MakeKs(nh)
    eigvalsShould = [2 * (np.cos(2 * np.pi * h * k) - 1) for k in ks]
    if (np.isclose(eigvals, eigvalsShould, 1e-14).all()):
        pass
    else:
        print('Approximate Expected Eigenvalues:', eigvalsShould)
        print('Eigenvalues Found:', eigvals)
        sys.exit('ERROR:\nWaveTools:\nFindLaplaceEigVals:\nNot returning correct eigvals!')
    return eigvals


# In[ ]:




