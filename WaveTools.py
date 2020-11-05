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


# This function creates a linear space of evenly-spaced cell locations and passes a matrix of cell-centered Fourier modes along with a linear space of the cell locations.

# In[2]:


def MakeWaves(omega):
    nh_max = omega.nh_max
    x = omega.xNode
    n = len(x) - 1
    xCell = 0.5 * (x[0:n] + x[1:n + 1])
    hDiag = x[1:n + 1] - x[0:n]
    waves = CellWaves(nh_max, x)
    hs = np.zeros((n, n), float)
    np.fill_diagonal(hs, hDiag)
    zeroMat = np.zeros((n, nh_max - n), float)
    hMat = LA.inv(hs)
    wave1up = 1 * waves
    wave1up.T[::nh_max] = 0
    wave0 = waves - wave1up
    waves = (hMat @ wave1up) + wave0
    return waves


# This function takes in a linear space of node locations and creates a matrix of `nh_max` many cell-centered Fourier modes.

# In[3]:


def CellWaves(nh_max, x):
    n = len(x) - 1
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


def MakeNodeWaves(omega, nRes = 0):
    nh_max = omega.nh_max
    if (nRes == 0):
        nRes = nh_max
        xMax = 1. - (1. / nRes)
    else:
        xMax = 1.
    x = np.linspace(0, xMax, num = nRes)
    waves = NodeWaves(nh_max, x, nRes)
    return x, waves


# This function creates a matrix of node-centered Fourier modes along with a linear space of the node locations.

# In[5]:


def NodeWaves(nh_max, x, nRes):
    waves = np.zeros((nRes, nh_max), float)
    for k in range(int(nh_max / 2)):
        waves[:, (2 * k) + 1] = np.sin(2 * np.pi * (k + 1) * x)
        if (k == 0):
            waves[:, 2 * k] = np.ones(nRes, float)
        else:
            waves[:, 2 * k] = np.cos(2 * np.pi * k * x)
    return waves


# This function creates an array of all the $k$ values at their respective index locations.

# In[6]:


def MakeKs(nh):
    kVals = list(np.arange(nh))
    kVals = kVals
    ks = [int((i + 1) / 2) for i in kVals]
    return ks


# This function takes in a matrix of wave vectors and finds their respective eigenvalues as they relate to the Laplacian matrix. Note that this function assumes that the input matrix comprises Fourier modes but includes an error catcher in case the eigenvalues don't turn out as expected within a $10^{-14}$ tolerance.

# In[7]:


def FindLaplaceEigVals(nh, h, waves):
    errorLoc = 'ERROR:\nWaveTools:\nFindLaplaceEigVals:\n'
    errorMess = BT.CheckSize(nh, waves, nName = 'nh', matricaName = 'waves')
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
    wavesNorm = OT.NormalizeMatrix(nh, waves)
    wavesInv = wavesNorm.conj().T
    Laplacian = OT.MakeLaplacian1D(nh)
    eigMat = wavesInv @ Laplacian @ wavesNorm
    eigMat = OT.RoundDiag(eigMat)
    errorMess = BT.CheckDiag(eigMat, matricaName = 'eigMat')
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
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




