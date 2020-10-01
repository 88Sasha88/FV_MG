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


# This function creates a matrix of the coefficients for our wave functions at the respective locations of the FFT values.

# In[2]:


def ConstructFFTCoef(nh):
    k_max = int(nh / 2)
    FFTCoef = np.zeros((nh, nh), dtype = complex)
    FFTCoef[0, k_max] = 1 # k = 0 mode in k_max + 1 column
    for k in range(1, k_max):
        # sin mode
        FFTCoef[(2 * k) - 1, k_max - k] = .5*1j
        FFTCoef[(2 * k) - 1, k_max + k] = -.5*1j
        # cos mode
        FFTCoef[2 * k, k_max - k] = .5
        FFTCoef[2 * k, k_max + k] = .5
    FFTCoef[nh - 1, 0] = 1 # k = k_max mode in first column
    return FFTCoef, k_max


# This function finds our phase and amplitude adjustments.

# In[3]:


def ConstructShift(nh, waves):
    problem = BT.CheckSize(nh, waves)
    if (problem != 0):
        sys.exit('ERROR:\nFFTTools:\nConstructShift:\nnh does not match size of waves!')
    xhat = PerformFFT(waves, axes = 1)
    xhat = xhat.T
    FFTCoef, k_max = ConstructFFTCoef(nh)
    FFTFix = np.round(FFTCoef.T @ LA.inv(xhat), 15)
    problem = BT.CheckDiag(FFTFix)
    if (problem != 0):
        print('FFTFix: \n', FFTFix)
        sys.exit('ERROR:\nFFTTools:\nConstructShift:\nFFTFix is not diagonal!')
    phaseModes = (nh / np.pi) * np.angle(np.diag(FFTFix)) # This is what the k number for the phase shifts should be.
    phaseModeCheck = -np.linspace(-k_max, k_max - 1, nh)
    phaseModeCheck[0] = 0
    if (np.isclose(phaseModes, phaseModeCheck, 1e-15).all()):
        pass
    else:
        print('Approximate Expected Phase Modes:', phaseModeCheck)
        print('Phase Modes Found:', phaseModes)
        sys.exit('ERROR:\nFFTTools:\nConstructShift:\nPhase modes are incorrect!')
    PhaseCorrect = np.exp((np.pi * phaseModeCheck * 1j) / nh)
    AmpCorrect = np.diag(FFTFix) / PhaseCorrect
    return PhaseCorrect, AmpCorrect


# This function performs an FFT on the input array.

# In[4]:


def PerformFFT(inputArray, axes = 0):
    outputArray = np.fft.fft(inputArray.T)
    outputArray = np.round(np.fft.fftshift(outputArray, axes = axes), 14)
    return outputArray


# This function performs an inverse FFT on the input array.

# In[5]:


def PerformIFFT(inputArray, axes = 0):
    outputArray = np.round(np.fft.ifftshift(inputArray, axes = axes), 14)
    outputArray = np.fft.ifft(outputArray.T)
    return outputArray


# This takes in some vector in $x$ space and returns the coefficients in $k$ space.

# In[6]:


def GetKSpaceCoefs(nh, coefs, waves):
    problemCoef = BT.CheckSize(nh, coefs)
    problemWave = BT.CheckSize(nh, waves)
    if (problemCoef != 0):
        sys.exit('ERROR:\nFFTTools:\nGetKSpaceCoefs:\nnh does not match size of coefs!')
    if (problemWave != 0):
        sys.exit('ERROR:\nFFTTools:\nGetKSpaceCoefs:\nnh does not match size of waves!')
    linCombo = OT.ChangeBasis(nh, coefs, waves)
    linComboFFT = PerformFFT(linCombo)
    phase, amp = ConstructShift(nh, waves)
    kCoefs = np.round(amp * phase * linComboFFT, 14)
    return kCoefs


# This takes in some vector in $k$ space and returns the coefficients in $x$ space.

# In[7]:


def GetXSpaceCoefs(nh, coefs, waves):
    problemCoef = BT.CheckSize(nh, coefs)
    problemWave = BT.CheckSize(nh, waves)
    if (problemCoef != 0):
        sys.exit('ERROR:\nFFTTools:\nGetXSpaceCoefs:\nnh does not match size of coefs!')
    if (problemWave != 0):
        sys.exit('ERROR:\nFFTTools:\nGetXSpaceCoefs:\nnh does not match size of waves!')
    phase, amp = ConstructShift(nh, waves)
    linCombo = np.round(coefs / (amp * phase), 14)
    linComboIFFT = PerformIFFT(linCombo)
    xCoefs = np.round(LA.inv(waves) @ linComboIFFT, 14)
    if (np.isreal(xCoefs).all()):
        xCoefs = np.real(xCoefs)
    else:
        print('xCoefs :', xCoefs)
        sys.exit('ERROR:\nFFTTools:\nGetXSpaceCoefs:\nx coefficients are not real!')
    return xCoefs


# In[ ]:




