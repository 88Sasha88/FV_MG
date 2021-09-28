#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os.path
from scipy import *
import numpy as np
from numpy import *
from numpy import linalg as LA
from scipy import linalg as LA2
import sympy as sympy
import sys as sys
import time
import matplotlib.pyplot as plt
import itertools as it
from IPython.core.display import HTML


sys.path.append('/Users/sashacurcic/SashasDirectory/ANAG/FV_MG/')
from Modules import BasicTools as BT

display(HTML("<style>pre { white-space: pre !important; }</style>"))
np.set_printoptions( linewidth = 10000, threshold = 100000)

# ----------------------------------------------------------------------------------------------------------------
# Function: Gauss
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# This function returns a Gaussian waveform with standard deviation sigma centered about `mu`. As the default, it
# returns the cell-averaged values of the Gaussian using Boole's Rule.
# ----------------------------------------------------------------------------------------------------------------
# Input:
#
# omega                   BT.Grid                 Grid object
# sigma                   real                    Standard deviation of Gaussian
# mu                      real                    Average of Gaussian
# (cellAve)               bool                    Switch set to find the cell average Gaussian values using
#                                                     Boole's rule
# (deriv)                 int                     Number of derivatives to take of waveform (DOESN'T WORK BEYOND FIRST!)
# ----------------------------------------------------------------------------------------------------------------
# Output:
#
# gauss                   np.ndarray              Gaussian waveform values on Grid omega
# ----------------------------------------------------------------------------------------------------------------

def Gauss(omega, sigma, mu, cellAve = True, deriv = 0):
    xCell = omega.xCell
    xNode = omega.xNode
    h = omega.h
    nh = omega.nh[-1]
    if (cellAve):
        x = xNode
        for k in range(1, 4):
            x = np.asarray(sorted(set(np.append(x, xNode[:-1] + (k * h) / 4.))))
    else:
        x = xCell
    gauss = np.exp(-((x - mu)**2) / (2. * (sigma**2)))
    for k in range(deriv):
        gauss = ((mu - x) * gauss) / (sigma ** 2)
    if (cellAve):
        gauss = BoolesAve(gauss)
    return gauss

# ----------------------------------------------------------------------------------------------------------------
# Function: BoolesAve
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# This function uses Boole's Rule to return the cell average values of some mathematical function f represented on
# some grid.
# ----------------------------------------------------------------------------------------------------------------
# Input:
#
# f                       np.ndarray              Mathematical function f evaluated at nodes of some grid
# ----------------------------------------------------------------------------------------------------------------
# Output:
#
# f_ave                   np.ndarray              Cell average values of mathematical function f represented on
#                                                     some grid
# ----------------------------------------------------------------------------------------------------------------

def BoolesAve(f):
    errorLoc = 'ERROR:\nTestTools:\nBoolesAve:\n'
    if (len(f) % 4 != 1):
        sys.exit(errorLoc + 'f must be one more than integer multiple of four in length!')
    f_ave = (1. / 90.) * ((7 * f[:-1:4]) + (32 * f[1::4]) + (12 * f[2::4]) + (32 * f[3::4]) + (7 * f[4::4]))
    return f_ave

# ----------------------------------------------------------------------------------------------------------------
# Function: WavePacket
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# This function returns a Gaussian wavepacket with standard deviation sigma centered about `mu` and with a
# sinusoidal part corresponding to modenumber. It uses Boole's Rule to approximate the cell-averaged values of the
# Gaussian itself.
# ----------------------------------------------------------------------------------------------------------------
# Input:
#
# omega                   BT.Grid                 Grid object
# sigma                   real                    Standard deviation of Gaussian
# mu                      real                    Average of Gaussian
# waves                   np.ndarray              Matrix containing cell-averaged values of waves (can be full set
#                                                     or restricted set)
# (deriv)                 int                     Number of derivatives to take of waveform (DOESN'T WORK BEYOND FIRST!)
# ----------------------------------------------------------------------------------------------------------------
# Output:
#
# packet                  np.ndarray              Gaussian wavepacket cell-average values on Grid omega
# ----------------------------------------------------------------------------------------------------------------

def WavePacket(omega, sigma, mu, modenumber, waves, deriv = 0):
    errorLoc = 'ERROR:\nWaveformTools:\nWavePacket:\n'
    nh_max = omega.nh_max
    if (modenumber > nh_max):
        errorMess = 'Modenumber out of range for grid resolution!'
        sys.exit(errorLoc + errorMess)
    packetAmp = Gauss(omega, sigma, mu)
    if (deriv == 0):
        print(waves[:, modenumber])
        packet = packetAmp * waves[:, modenumber]
    else:
        q = int(2 * ((modenumber % 2) - 0.5))
        print('q is', q)
        newAmp = Gauss(omega, sigma, mu, deriv = deriv)
        packet = (newAmp * waves[:, modenumber]) + (2 * np.pi * q * modenumber * (packetAmp * waves[:, modenumber + q]))
    return packet

