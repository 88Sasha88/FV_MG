#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os.path
from scipy import *
import scipy as sp
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
from Modules import OperatorTools as OT
from Modules import WaveTools as WT

display(HTML("<style>pre { white-space: pre !important; }</style>"))
np.set_printoptions( linewidth = 10000, threshold = 100000)

# ----------------------------------------------------------------------------------------------------------------
# Function: Gauss
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# This function returns a Gaussian waveform with standard deviation sigma centered about mu. As the default, it
# returns the cell-averaged values of the Gaussian using Boole's Rule.
# ----------------------------------------------------------------------------------------------------------------
# Input:
#
# omega                   BT.Grid                 AMR grid
# sigma                   real                    Standard deviation of Gaussian
# mu                      real                    Average of Gaussian
# (BooleAve)              bool                    Switch for whether of not to use Boole's rule to find cell-
#                                                     averaged values of Gaussian
# (deriv)                 bool                    Switch for whether or not to use Boole's rule to find the cell-
# ----------------------------------------------------------------------------------------------------------------
# Output:
#
# gauss                   np.ndarray              Gaussian waveform values on Grid omega in space-space
# ----------------------------------------------------------------------------------------------------------------

def Gauss(omega, sigma, mu, BooleAve = False, deriv = False, cellAve = True):
    xNode = omega.xNode
    
    # There is no exact calculation for the calculation of the cell-averaged derivative of a Gaussian; therefore,
    # Boole's Rule approximate average must be taken.
#     if (deriv):
#         BooleAve = True
    
    if (BooleAve):
        if (cellAve):
            if (not deriv):
                print('This is not the most accurate option for a cell-averaged Gaussian, and you shouldn\'t use it!')
            x = BoolesX(omega)
            gauss = np.exp(-((x - mu)**2) / (2. * (sigma**2)))
    else:
        if (cellAve):
            x = xNode
            const = sigma * np.sqrt(np.pi / 2.)
            hMat = OT.StepMatrix(omega)
            erf = sp.special.erf((x - mu) / (sigma * np.sqrt(2)))
            gauss = const * (hMat @ (erf[1:] - erf[:-1]))
        else:
            x = BoolesX(omega)
            gauss = np.exp(-((x - mu)**2) / (2. * (sigma**2)))

    if (deriv):
        gauss = ((mu - x) * gauss) / (sigma ** 2)
    if (BooleAve):
        gauss = BoolesAve(gauss)
    return gauss

# ----------------------------------------------------------------------------------------------------------------
# Function: BoolesX
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# This function creates a linear space of x values which will accommodate Boole's Rule evalutation of cell
# averages for calculation of an arbitrary waveform.
# ----------------------------------------------------------------------------------------------------------------
# Input:
#
# omega                   BT.Grid                 AMR grid
# ----------------------------------------------------------------------------------------------------------------
# Output:
#
# x                       np.ndarray              x values needed to calculate Boole's cell averages on AMR grid
# ----------------------------------------------------------------------------------------------------------------

def BoolesX(omega):
    xNode = omega.xNode
    x = xNode
    h = omega.h
    for k in range(1, 4):
        x = np.asarray(sorted(set(np.append(x, xNode[:-1] + (k * h) / 4.))))
    return x

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
# omega                   BT.Grid                 AMR grid
# sigma                   real                    Standard deviation of Gaussian
# mu                      real                    Average of Gaussian
# waves                   np.ndarray              Matrix containing cell-averaged values of waves (can be full set
#                                                     or selected subset)
# (deriv)                 int                     Number of derivatives to take of waveform (DOESN'T WORK BEYOND FIRST!)
# ----------------------------------------------------------------------------------------------------------------
# Output:
#
# packet                  np.ndarray              Gaussian wavepacket cell-average values on Grid omega
# ----------------------------------------------------------------------------------------------------------------

def WavePacket(omega, sigma, mu, modenumber, deriv = False):
    # YOU GOTTA CREATE A WAVES INSTANCE!
    errorLoc = 'ERROR:\nWaveformTools:\nWavePacket:\n'
    nh_max = omega.nh_max
    
    k = int((modenumber + 1) / 2)
    Cosine = lambda x: np.cos(2. * np.pi * k * x)
    Sine = lambda x: np.sin(2. * np.pi * k * x)
    x = BoolesX(omega)
    if (modenumber > nh_max):
        errorMess = 'Modenumber out of range for grid resolution!'
        sys.exit(errorLoc + errorMess)
#     else:
#         if (modenumber % 2 == 0):
#             wave = Cosine(x)
#         else:
#             wave = Sine(x)
    
    packetAmp = Gauss(omega, sigma, mu, cellAve = False)
    if (deriv):
        if (modenumber % 2 == 0):
            part1 = -2 * np.pi * k * packetAmp * Sine(x)
            part2 = Gauss(omega, sigma, mu, deriv = True, cellAve = False) * Cosine(x)
        else:
            part1 = 2 * np.pi * k * packetAmp * Cosine(x)
            part2 = Gauss(omega, sigma, mu, deriv = True, cellAve = False) * Sine(x)
        packet = part1 + part2
    else:
        if (modenumber % 2 == 0):
            packet = packetAmp * Cosine(x)
        else:
            packet = packetAmp * Sine(x)
    wavepacket = BoolesAve(packet)
    return wavepacket

# ----------------------------------------------------------------------------------------------------------------
# Function: GaussParams
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# This function returns parameters mu and sigma for a Gaussian waveform with sufficient dropoff at the specified
# boundaries. As a default, it is centered at a center with a sufficient dropoff definied to be 10-14. It does not
# account for the Shannon-Nyquist sampling rate (so it is possible for the distribution to be too sharp for an
# excessively coarse grid.)
# ----------------------------------------------------------------------------------------------------------------
# Input:
#
# (x_0)                   real                    Left boundary
# (x_1)                   real                    Right boundary
# (errOrd)                real                    Order of dropoff at defined boundaries
# ----------------------------------------------------------------------------------------------------------------
# Output:
#
# sigma                   real                    Standard deviation of Gaussian
# mu                      real                    Average of Gaussian
# ----------------------------------------------------------------------------------------------------------------

def GaussParams(x_0 = 0., x_1 = 1., errOrd = 14):
    mu = (x_0 + x_1) / 2.
    sigma = abs((x_1 - x_0) / np.sqrt(8 * errOrd * log(10)))
    return sigma, mu

