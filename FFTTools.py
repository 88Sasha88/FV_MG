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
import WaveTools as WT
import PlotTools as PT
import OperatorTools as OT


# This function creates a matrix of the coefficients for our wave functions at the respective locations of the FFT values.

# In[2]:


def ConstructFFTCoef(nh):
    kMax = int(nh / 2)
    FFTCoef = np.zeros((nh, nh), dtype = complex)
    FFTCoef[0, kMax] = 1 # k=0 mode in max_k+1 column
    for k in range(1, kMax):
        # sin mode
        FFTCoef[(2 * k) - 1, kMax - k] = .5*1j
        FFTCoef[(2 * k) - 1, kMax + k] = -.5*1j
        # cos mode
        FFTCoef[2 * k, kMax - k] = .5
        FFTCoef[2 * k, kMax + k] = .5
    FFTCoef[nh - 1, 0] = 1 # k=max_k mode in first column
    return FFTCoef


# This function finds our phase shifts (I think?)

# In[3]:


def ConstructPhaseShift(nh, waves):
    problem = BT.CheckSize(nh, waves)
    if (problem != 0):
        sys.exit('ERROR:\nFFTTools:\nConstructPhaseShift:\nnh does not match size of waves!')
    xhat = PerformFFT(waves)
    xhat = xhat.T
    FFTCoef = ConstructFFTCoef(nh)
    FFTFix = np.round(FFTCoef.T @ LA.inv(xhat), 16)
    FFTFixXhat = np.round(FFTFix @ xhat, 15)
    return FFTFixXhat


# This function performs an FFT on the input array.

# In[4]:


def PerformFFT(inputArray):
    outputArray = np.fft.fft(inputArray.T)
    outputArray = np.round(np.fft.fftshift(outputArray, axes = 1), 14)
    return outputArray


# This function takes in some vector of coefficients and expresses it in the basis of the input matrix.

# In[5]:


def ChangeBasis(nh, coefs, waves):
    problemCoef = BT.CheckSize(nh, coefs)
    problemWave = BT.CheckSize(nh, waves)
    if (problemWave != 0):
        sys.exit('ERROR:\nFFTTools:\nChangeBasis:\nnh does not match size of waves!')
    if (problemCoef != 0):
        sys.exit('ERROR:\nFFTTools:\nChangeBasis:\nnh does not match size of coefs!')
    linCombo = waves @ coefs
    return linCombo


# In[ ]:




