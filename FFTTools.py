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
    k_max = int(nh / 2)
    FFTCoef = np.zeros((nh, nh), dtype = complex)
    FFTCoef[0, k_max] = 1 # k=0 mode in max_k+1 column
    for k in range(1, k_max):
        # sin mode
        FFTCoef[(2 * k) - 1, k_max - k] = .5*1j
        FFTCoef[(2 * k) - 1, k_max + k] = -.5*1j
        # cos mode
        FFTCoef[2 * k, k_max - k] = .5
        FFTCoef[2 * k, k_max + k] = .5
    FFTCoef[nh - 1, 0] = 1 # k = k_max mode in first column
    return FFTCoef, k_max


# This function finds our phase shifts (I think?)

# In[3]:


def ConstructShift(nh, waves):
    problem = BT.CheckSize(nh, waves)
    if (problem != 0):
        sys.exit('ERROR:\nFFTTools:\nConstructPhaseShift:\nnh does not match size of waves!')
    xhat = PerformFFT(waves, axes = 1)
    xhat = xhat.T
    FFTCoef, k_max = ConstructFFTCoef(nh)
    FFTFix = np.round(FFTCoef.T @ LA.inv(xhat), 16)
    problem = BT.CheckDiag(FFTFix)
    if (problem != 0):
        sys.exit('ERROR:\nFFTTools:\nConstructShift:\nFFTFix is not diagonal!')
    phaseModes = (nh / np.pi) * np.angle(np.diag(FFTFix)) # This is what the k number for the phase shifts should be.
    phaseModeCheck = -np.linspace(-k_max, k_max - 1, nh)
    phaseModeCheck[0] = 0
    if (np.isclose(phaseModes, phaseModeCheck, 1e-15).all()):
        pass
    else:
        print('Approximate Expected Phase Modes:', phaseModeCheck)
        print('Phase Modes Found:', phaseModes)
        sys.exit('ERROR:\nFFTTools:\nConstructPhaseShift:\nPhase modes are incorrect!')
    PhaseCorrect = np.exp((np.pi * phaseModeCheck * 1j) / nh)
    AmpCorrect = np.diag(FFTFix) / PhaseCorrect
    return PhaseCorrect, AmpCorrect


# This function performs an FFT on the input array.

# In[4]:


def PerformFFT(inputArray, axes = 0):
    outputArray = np.fft.fft(inputArray.T)
    outputArray = np.round(np.fft.fftshift(outputArray, axes = axes), 14)
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




