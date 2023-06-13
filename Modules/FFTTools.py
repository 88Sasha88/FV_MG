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
from Modules import BasicTools as BT
from Modules import OperatorTools as OT
from Modules import WaveTools as WT
from Modules import GridTransferTools as GTT


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
    errorLoc = 'ERROR:\nFFTTools:\nConstructShift:\n'
    errorMess = BT.CheckSize(nh, waves, nName = 'nh', matricaName = 'waves')
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
    xhat = PerformFFT(waves, axes = 1)
    xhat = xhat.T
    FFTCoef, k_max = ConstructFFTCoef(nh)
    FFTFix = FFTCoef.T @ LA.inv(xhat)
    FFTFix = OT.RoundDiag(FFTFix) # 15
    errorMess = BT.CheckDiag(FFTFix, matricaName = 'FFTFix')
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
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


def GetKSpaceCoefs(omega, coefs, waves):
    nh = omega.nh_max
    errorLoc = 'ERROR:\nFFTTools:\nGetKSpaceCoefs:\n'
    errorMess = BT.CheckSize(nh, coefs, nName = 'nh', matricaName = 'coefs')
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
    errorMess = BT.CheckSize(nh, waves, nName = 'nh', matricaName = 'waves')
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
    linCombo = waves @ coefs
    linComboFFT = PerformFFT(linCombo)
    phase, amp = ConstructShift(nh, waves)
    kCoefs = np.round(amp * phase * linComboFFT, 14)
    return kCoefs


# This takes in some vector in $k$ space and returns the coefficients in $x$ space.

# In[7]:


def GetXSpaceCoefs(omega, coefs, waves):
    nh = omega.nh_max
    errorLoc = 'ERROR:\nFFTTools:\nGetXSpaceCoefs:\n'
    errorMess = BT.CheckSize(nh, coefs, nName = 'nh', matricaName = 'coefs')
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
    errorMess = BT.CheckSize(nh, waves, nName = 'nh', matricaName = 'waves')
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
    phase, amp = ConstructShift(nh, waves)
    linCombo = np.round(coefs / (amp * phase), 14)
    linComboIFFT = PerformIFFT(linCombo)
    xCoefs = np.round(LA.inv(waves) @ linComboIFFT, 14)
    if (np.isreal(xCoefs).all()):
        xCoefs = np.real(xCoefs)
    else:
        print('xCoefs :', xCoefs)
        sys.exit(errorLoc + 'x coefficients are not real!')
    return xCoefs









# This returns the Fourier coefficients of some waveform.

# In[8]:


def FourierCoefs(waves, waveform, printBool = False): # YOU GOT RID OF OMEGA HERE!!!
    errorLoc = 'ERROR:\nFFTTools:\nFourierCoefs:\n'
    errorMess = ''
    FTOp = OT.FourierTransOp(waves)
    waveDim = np.shape(waves)[0]
    formDim = np.shape(waveform)[0]
    if (formDim == waveDim):
        FCoefs = FTOp @ waveform
    else:
        if (formDim == int(2 * waveDim)):
            FTOpBlock = OT.Block(FTOp, var = 2)
            FCoefs = FTOpBlock @ waveform
        else:
            errorMess = 'The size of waves, ' + str(waveDim) + 'x' + str(waveDim = np.shape(waves)[1]) + ', does not match up with the size of waveform, ' + str(formDim) + '!'
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
    return FCoefs


# This returns the Fourier coefficients of a time propogated waveform.

# In[10]:


def PropogateFCoefs(omega, FCoefs, c, t, nullspace = []):
    errorLoc = 'ERROR:\nFFTTools:\nPropogateFCoefs:\n'
    nh = omega.nh_max
    if (nullspace != []):
        omega = BT.Grid(nh)
    degFreed = omega.degFreed# [::-1][0]
#     errorMess = BT.CheckSize(degFreed, FCoefs, nName = 'degFreed', matricaName = 'FCoefs')
#     if (errorMess != ''):
#         sys.exit(errorLoc + errorMess)
#     Cosine = lambda k: np.cos(2. * np.pi * k * c * t)
#     Sine = lambda k: np.sin(2. * np.pi * k * c * t)
#     RotMat = lambda k: np.asarray([Cosine(k), Sine(k), -Sine(k), Cosine(k)]).reshape(2, 2)
#     rotMats = [RotMat(k) for k in range(int(nh / 2) + 1)]
#     shift = LA2.block_diag(*rotMats)[1:-1, 1:-1]
#     shift[0, 0] = Cosine(0)
#     shift[::-1, ::-1][0, 0] = Cosine(nh / 2)
#     print(np.round(shift, 14))
#     print('shape shift before:', np.shape(shift))
    shift = OT.MakeRotMat(omega, c * t)
    if (nullspace != []):
        print(shift)
        shift = nullspace.transpose() @ shift @ nullspace
        print(np.round(shift, 14))
        print('shape shift after:', np.shape(shift))
    print('shift coefs before:', np.shape(FCoefs))
    propFCoefs = shift @ FCoefs
    print('shift coefs after:', np.shape(propFCoefs))
    return propFCoefs


def PropWaves(omega, waves, ct): # Change was made here!
    # Get all my attributes.
    nh2 = omega.nh_max
    nhs = omega.nh
    hs = omega.h
    levels = omega.levels
    refRatios = omega.refRatios
    
    # Create rotMat.
    rotMat = OT.MakeRotMat(omega, ct) # Change was made here!
#     print('rotMat:')
#     print(rotMat)
    backRotMat = rotMat[::-1, ::-1] + 0
#     print('backRotMat:')
#     print(backRotMat)
#     np.fill_diagonal(backRotMat[1:], np.diagonal(backRotMat, offset = 1))
#     np.fill_diagonal(backRotMat[:, 1:], -np.diagonal(backRotMat, offset = 1))
    compRotMat = rotMat + 0
    compRotMat[::-1, ::-1][0, 0] = 1
    h = 1. / nh2
    # aliasedWaves = int(nhs[0])
    fineSpots = np.where(hs == h)[0]
    wavesAlias = waves + 0
    # wavesAlias[:, :aliasedWaves] = 0
    wavesAlias[fineSpots, :] = 0
    workingWaves = waves - wavesAlias
    propMat = workingWaves @ rotMat
    refRatioTot = 1
    for q in range(levels):
        nh0 = nhs[::-1][q + 1]
        if (q == 0):
            nh1 = nh0
        refRatio = refRatios[::-1][q]
        refRatioTot = refRatioTot * refRatio
        h = 1. / nh0
        for p in range(refRatioTot - 1):
            s0 = 2 * (p % 2)
            startPoint = nh0 * (p + 1) - s0
            endPoint = startPoint + nh0 + s0
            compRotMat[startPoint + 1:endPoint - 1, startPoint + 1:endPoint - 1] = backRotMat[nh2 - nh0 - s0 + 1:nh2 - 1, nh2 - nh0 - s0 + 1:nh2 - 1]
        workingWaves = wavesAlias + 0
        fineSpots = np.where(hs != h)[0]
        workingWaves[fineSpots, :] = 0
        propMat = propMat + (workingWaves @ backRotMat)
        wavesAlias = wavesAlias - workingWaves
    return propMat


# input: waveform of fully refined grid
# output: propagated coef
def PropRestrictWaves(omega, waveformIn, ct, Hans = False): # Change was made here!
    nh_max = omega.nh_max
    degFreed = omega.degFreed
    waves = WT.MakeWaves(omega)
    nullspace = OT.FindNullspace(omega, waves, Hans = Hans)
    omegaF = BT.Grid(nh_max)
    wavesF = WT.MakeWaves(omegaF)
    waveform = waveformIn.copy()
    restrictOp = GTT.CoarsenOp(omega)
    
    # Find the Fourier coefficients for the initial condition on the completely refined grid.
    FCoefs = FourierCoefs(wavesF, waveform)
    
    # Propagate all of the Fourier modes on the completely refined grid.
    propWaves = PropWaves(omegaF, wavesF, ct) # Change was made here!
    
    # Find the propagated waveform coarsened down to AMR.
    propWaveform = restrictOp @ propWaves @ FCoefs
    
    # Find Fourier coefficients of priopagated solution on AMR grid.
    propFCoefs = FourierCoefs(waves @ nullspace, propWaveform)
    
    return propFCoefs
