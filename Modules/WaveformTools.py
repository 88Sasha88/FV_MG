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
# (cellAve)               bool                    Switch for whether of not to find cell-averaged values of
#                                                     Gaussian in terms of scipy erf function
# (deriv)                 bool                    Switch for whether or not to use Boole's rule to find the cell-
# ----------------------------------------------------------------------------------------------------------------
# Output:
#
# gauss                   np.ndarray              Gaussian waveform values on Grid omega in space-space
# ----------------------------------------------------------------------------------------------------------------
def WaveEq(omega, physics, func, args, t, IRT = 'IRT', cellAve = True, BooleAve = False, deriv = False, field = 'EB'):
    
    xCell = omega.xCell
    cs = physics.cs
    x_s = physics.locs[0]
    mus = physics.mus
    
    IRT = IRT.upper()
    I = IRT.find('I') + 1
    R = IRT.find('R') + 1
    T = IRT.find('T') + 1
    
    index = np.where(xCell >= x_s)[0][0]
    
    waveFuncIT = 0
    waveFuncR = 0
    if (I or T):
        waveFuncIT = Advect(omega, physics, func, args, t, cellAve = cellAve, BooleAve = BooleAve, deriv = deriv)
        EFuncIT = waveFuncIT.copy()
        # Scale the T part.
        scale = (2 * cs[1]) / (cs[0] + cs[1]) # Switch numerator to cs[0].
#         if (field == 'B'):
#             scale = (mus[0] * scale) / mus[1] # Switch mus.
        EFuncIT[index:] = scale * EFuncIT[index:]
        BFuncIT = EFuncIT.copy()
        BFuncIT[index:] = BFuncIT[index:] / cs[1]
        BFuncIT[:index] = BFuncIT[:index] / cs[0]
        if (not I):
            print('BE AWARE THAT YOU HAVE ELECTED FOR THERE TO BE NO INCIDENT PART!')
            # Zero out the I part.
            EFuncIT[:index] = 0
            BFuncIT[:index] = 0
        else: # Is I, might be T.
            if (not T):
                # Zero out the T part.
                EFuncIT[index:] = 0
                BFuncIT[index:] = 0
                
    if (R):
        waveFuncR = Reflect(omega, physics, func, args, t, cellAve = cellAve, BooleAve = BooleAve, deriv = deriv)
        # Scale R part.
        EFuncR = waveFuncR.copy()
        scale = (cs[1] - cs[0]) / (cs[0] + cs[1]) # Remove negative sign out front.
        EFuncR = scale * EFuncR
        BFuncR = EFuncR.copy()
        BFuncR = -BFuncR / cs[0]
    EFunc = EFuncIT + EFuncR
    BFunc = BFuncIT + BFuncR
    if (field == 'E'):
        waveFunc = EFunc
    else:
        if (field == 'B'):
            waveFunc = BFunc
        else:
            waveFunc = np.append(EFunc, BFunc)
    return waveFunc

def Reflect(omega, physics, func, args, t, cellAve = True, BooleAve = False, deriv = False):
    xCell = omega.xCell
    x_s = physics.locs[0]
    
    index = np.where(xCell >= x_s)[0][0]
    if (func == GaussPacket):
        BooleAve = True
    if (BooleAve and cellAve):
        x = BoolesX(omega, physics, t) # REFLECTION HERE!!!
        # I set cellAve to False because that changes the function for the Gaussian, and I want the
        # Gaussian to be calculated directly if I'm using Boole's Rule.
        waveFuncPre = func(omega, x, *args, deriv = deriv, cellAve = False)
        waveFunc = BoolesAve(waveFuncPre)
    else:
        x = ShiftX(omega, physics, t, adv = False) # REFLECTION HERE!!!
        waveFunc = func(omega, x, *args, deriv = deriv, cellAve = cellAve)
    waveFunc[index:] = 0
    return waveFunc

def Advect(omega, physics, func, args, t, cellAve = True, BooleAve = False, deriv = False):
    if (t == 0):
        waveFunc = InitCond(omega, physics, func, args, cellAve = cellAve, BooleAve = BooleAve, deriv = deriv)
    else:
        if (func == GaussPacket):
            BooleAve = True
        if (BooleAve and cellAve):
            print('We\'re doing BoolesAve!')
            x = BoolesX(omega, physics, t)
            print('x is ' + str(len(x)) + ' long.')
            # I set cellAve to False because that changes the function for the Gaussian, and I want the
            # Gaussian to be calculated directly if I'm using Boole's Rule.
            waveFuncPre = func(omega, x, *args, deriv = deriv, cellAve = False)
            waveFunc = BoolesAve(waveFuncPre)
        else:
            x = ShiftX(omega, physics, t)
            waveFunc = func(omega, x, *args, deriv = deriv, cellAve = cellAve)
    return waveFunc

def InitCond(omega, physics, func, args, cellAve = True, BooleAve = False, deriv = False, field = 'EB'):
    xNode = omega.xNode
    xCell = omega.xCell
    x = xNode
    cMat = physics.cMat
    cMatInv = LA.inv(cMat)
    
    if (func == GaussPacket):
        BooleAve = True
        if (cellAve == False):
            x = xCell
    if (BooleAve and cellAve):
        print('We\'re doing BoolesAve!')
        x = BoolesX(omega, physics, t = 0)
        print('x is ' + str(len(x)) + ' long.')
        # I set cellAve to False because that changes the function for the Gaussian, and I want the
        # Gaussian to be calculated directly if I'm using Boole's Rule.
        EFuncPre = func(omega, x, *args, deriv = deriv, cellAve = False)
        EFunc = BoolesAve(EFuncPre)
    else:
        EFunc = func(omega, x, *args, deriv = deriv, cellAve = cellAve)
    
    BFunc = cMatInv @ EFunc
    
    if (field == 'E'):
        waveFunc = EFunc
    else:
        if (field == 'B'):
            waveFunc = BFunc
        else:
            waveFunc = np.append(EFunc, BFunc)
    
    return waveFunc

def Gauss(omega, x, sigma, mu, deriv, cellAve):
    # Unpack requisite attributes from omega and physics.
    xCell = omega.xCell
    
    # There is no exact calculation for the calculation of the cell-averaged derivative of a Gaussian; therefore,
    # Boole's Rule approximate average must be taken.
#     if (deriv):
#         BooleAve = True
    # If I use Boole's rule to calculate cell-averaged values of my gaussian or gaussian derivative.
    xL = x[:-1]
    xR = x[1:]
    hDiag = xR - xL
#     zeroIndex = np.where(xR == 0)[0]
#     for i in range(len(zeroIndex)):
#         if (xCell[i] > locs[i]):
#             hDiag[zeroIndex] = hDiag[zeroIndex + 1]
#         else:
#             hDiag[zeroIndex] = hDiag[zeroIndex - 1]
#     xR[zeroIndex] = xL[zeroIndex] + hDiag[zeroIndex] # THIS MAY BE KINDA JANK
    hMat = LA.inv(np.diag(hDiag))
    
    if (cellAve and (not deriv)):
        const = sigma * np.sqrt(np.pi / 2.)
        Erf = lambda x: sp.special.erf((x - mu) / (sigma * np.sqrt(2)))
        # (Divide by xR - xL)
        gauss = const * (hMat @ (Erf(xR) - Erf(xL)))
    else:
        gaussian = lambda x: np.exp(-((x - mu)**2) / (2. * (sigma**2)))
        gauss = gaussian(x)
        if (deriv):
            if (cellAve):
                gauss = 2 * hMat @ (gaussian(xR) - gaussian(xL))
            else:
                gauss = ((mu - x) * gaussian(x)) / (sigma ** 2)
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

def BoolesX(omega, physics, t):
    # BOOLES X MIGHT NOT BE EFFECTIVELY SET UP FOR SHIFTX FUNCTION!!!
    if (t == 0):
        xNode = omega.xNode
    else:
        xNode = ShiftX(omega, physics, t)
    x = xNode
#     h = omega.h
    xL = x[:-1]
    xR = x[1:]
    h = xR - xL
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
    errorLoc = 'ERROR:\nWaveformTools:\nBoolesAve:\n'
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
# Gauss(omega, x, sigma, mu, deriv, cellAve)
def GaussPacket(omega, x, sigma, mu, modenumber, deriv = False, cellAve = True):
    # cellAve is an inert argument here.
    # YOU GOTTA CREATE A WAVES INSTANCE!
    errorLoc = 'ERROR:\nWaveformTools:\nWavePacket:\n'
    nh_max = omega.nh_max
    
    k = int((modenumber + 1) / 2)
    Cosine = lambda x: np.cos(2. * np.pi * k * x)
    Sine = lambda x: np.sin(2. * np.pi * k * x)
    # x = BoolesX(omega, physics, t)
    if (modenumber > nh_max):
        errorMess = 'Modenumber out of range for grid resolution!'
        sys.exit(errorLoc + errorMess)
#     else:
#         if (modenumber % 2 == 0):
#             wave = Cosine(x)
#         else:
#             wave = Sine(x)
    
    packetAmp = Gauss(omega, x, sigma, mu, cellAve = False, deriv = False)
    if (deriv):
        if (modenumber % 2 == 0):
            part1 = -2 * np.pi * k * packetAmp * Sine(x)
            part2 = Gauss(omega, x, sigma, mu, deriv = True, cellAve = False) * Cosine(x)
        else:
            part1 = 2 * np.pi * k * packetAmp * Cosine(x)
            part2 = Gauss(omega, x, sigma, mu, deriv = True, cellAve = False) * Sine(x)
        packet = part1 + part2
    else:
        if (modenumber % 2 == 0):
            packet = packetAmp * Cosine(x)
        else:
            packet = packetAmp * Sine(x)
#     wavepacket = BoolesAve(packet)
    return packet

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


# ----------------------------------------------------------------------------------------------------------------
# Function: ShiftX
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# This function creates a vector of x values shifted by however much a wave would have moved in two media at their
# given respective material speeds and the given material surface location for the input amount of time. This
# vector can be passed into a function of the initial condition to return the wave propagated at the future time.
# ----------------------------------------------------------------------------------------------------------------
# Inputs:
#
# omega                   Grid                    Object containing all grid attributes
# physics                 PhysProps               Object containing all attributes describing the physical setup
# t                       float                   Amount of time that wave is propagated
# ----------------------------------------------------------------------------------------------------------------
# Outputs:
#
# xShift                  array                   Vector of appropritately shifted x values
# ----------------------------------------------------------------------------------------------------------------

def ShiftX1(omega, physics, t_in):
    # SWITCH XSHIFT TO CELL-CENTERED!!!
    degFreed = omega.degFreed
    x_0 = omega.xNode
    cs = physics.cs
    locs = physics.locs
    
    shiftNum = len(cs)
    xShift = np.zeros(degFreed + 1, float)
#     xShiftR = np.zeros(degFreed + 1, float)
#     xShiftL = np.zeros(degFreed + 1, float)
    
    xShifts = [[] for i in cs]
    

    t_per = (((1. / cs[0]) - (1. / cs[1])) * locs[0]) + (1. / cs[1])
    t = t_in % t_per
    for i in range(shiftNum):
        print(i, ': shift by ', (cs[i] * t))
        xShifts[i] = x_0 - (cs[i] * t)

    ix0 = np.where(xShifts[0] >= 0.)[0]
    ix1 = np.where(x_0 <= locs[0])[0]
    ix2 = np.where(x_0 >= locs[0])[0]
    ix3 = np.where(xShifts[1] <= locs[0])[0]
    #ix3 = np.where(x_0 <= 1.)[0]
    
    
#     print(ix0)
    print(ix1)
    print(ix2)
    print(xShifts[1])
    print(x_0)
    
    ixc0 = list(sorted(set(ix0).intersection(ix1)))
    ixc1 = list(sorted(set(ix2).intersection(ix3)))
    
#     ixc1 = list(sorted(set(ix2).intersection(ix3)))
    
    print('')
    print('ixc0:')
    print(ixc0)
    print('ixc1:')
    print(ixc1)
#     print(max(ixc1))
#     print(min(ixc0))
#     ixcR = [i + 1 for i in ixc]
#     ixcL = [i - 1 for i in ixc]
    
#     print('')
#     print(ixc)
#     print(ixcR)
#     print(ixcL)
#     print('')
    
    xShift[:min(ixc0)] = ((cs[1] * xShifts[0][:min(ixc0)]) + cs[0]) / cs[0]
    xShift[ixc0] = xShifts[0][ixc0]
    if (cs[1] * t <= 1. - locs[0]): # MAKE SURE YOU CHANGE THIS TO ACCOUNT FOR FASTER SPEED IN MEDIUM 1!!!
        xShift[ixc1] = ((cs[0] * xShifts[1][ixc1]) + ((cs[1] - cs[0]) * locs[0])) / cs[1]
        xShift[max(ixc1):] = xShifts[1][max(ixc1):]
    else:
        xShift[min(ixc1):] = xShifts[0][min(ixc1):] + ((cs[0]/ cs[1]) * (1. - locs[0])) + locs[0]
    
#     xShiftR[:min(ixcR)] = xShifts[0][:min(ixcR)]
#     xShiftR[max(ixcR):] = xShifts[1][max(ixcR):]
    
#     xShiftL[:min(ixcL)] = xShifts[0][:min(ixcL)]
#     xShiftL[max(ixcL):] = xShifts[1][max(ixcL):]

#     tCross = (xShifts[1][ixc1] - locs[0]) / cs[1]
#     tCrossR = (xShifts[1][ixcR] - locs[0]) / cs[1]
#     tCrossL = (xShifts[1][ixcL] - locs[0]) / cs[1]
    
    
#     xShift[ixc1] = x_0[ixc1] + (cs[0] * tCross) - (cs[1] * (t + tCross))
#     xShift[:min(ixc0)] = ((cs[1] / cs[0]) * xShifts[0][:min(ixc0)]) + 1.
#     if (ixc1 == []):
#         val = degFreed + 1
#     else:
#         val = max(ixc1)
#     xShift[max(ixc0):val] = ((cs[0] / cs[1]) * (xShifts[1][max(ixc0):val] - locs[0])) + locs[0]
    
#     xShift[ixc0] = ((((cs[1] * t) - 1) / (cs[0] * t)) * x_0[ixc0]) + 1 - (cs[1] * t)
#     xShift[ixc0] = ((cs[1] * x_0[ixc0]) / cs[0]) - (cs[1] * t)

#     xShiftR[ixcR] = x_0[ixcR] + (cs[0] * tCrossR) - (cs[1] * (t + tCrossR))
#     xShiftL[ixcL] = x_0[ixcL] + (cs[0] * tCrossL) - (cs[1] * (t + tCrossL))
    
#     print(xShift)
#     print(xShiftR)
#     print(xShiftL)
#     print('')
    
#     xShiftR = xShiftR[1:]
#     xShiftL = xShiftL[:-1]
    print(ixc1)
    
#     while (xShift[0] < 0):

#         addOn = xShift[-1] - xShift[0]
# #         print(xShift[xShift < 0])
#         print('You shifted by ' + str(addOn) +'.')
        
# #         xShift[ixc0] = xShift[ixc0] + addOn
#         xShift[xShift < 0] = xShift[xShift < 0] + addOn
    print('HERE:')
    print(xShift)
    
    return xShift#, xShiftL, xShiftR

# Something's wrong when t=0.

def ShiftX(omega, physics, t, adv = True):
    # SWITCH XSHIFT TO CELL-CENTERED!!!
    degFreed = omega.degFreed
    x_0 = omega.xNode
    cs = physics.cs
    locs = physics.locs
    
    shiftNum = len(cs)
    xShift = np.zeros(degFreed + 1, float)
#     xShiftR = np.zeros(degFreed + 1, float)
#     xShiftL = np.zeros(degFreed + 1, float)
    
    xShifts = [[] for i in cs]
    

    for i in range(shiftNum):
        xShifts[i] = x_0 - (cs[i] * t)
    if (locs == []):
        xShift = xShifts[0]
    else:
        ixc1 = np.where(x_0 > locs[0])[0]
        ixc2 = np.where(xShifts[1] <= locs[0])[0]

        ixc = list(set(ixc1).intersection(ixc2))
    #     ixcR = [i + 1 for i in ixc]
    #     ixcL = [i - 1 for i in ixc]

    #     print('')
    #     print(ixc)
    #     print(ixcR)
    #     print(ixcL)
    #     print('')

        if (adv):
            if (ixc == []):
                minimum = max(ixc1)
                maximum = min(ixc2)
            else:
                minimum = min(ixc)
                maximum = max(ixc)
            xShift[:minimum] = xShifts[0][:minimum]
            xShift[maximum:] = xShifts[1][maximum:]

        #     xShiftR[:min(ixcR)] = xShifts[0][:min(ixcR)]
        #     xShiftR[max(ixcR):] = xShifts[1][max(ixcR):]

        #     xShiftL[:min(ixcL)] = xShifts[0][:min(ixcL)]
        #     xShiftL[max(ixcL):] = xShifts[1][max(ixcL):]

            tCross = (xShifts[1][ixc] - locs[0]) / cs[1]
        #     tCrossR = (xShifts[1][ixcR] - locs[0]) / cs[1]
        #     tCrossL = (xShifts[1][ixcL] - locs[0]) / cs[1]


            xShift[ixc] = x_0[ixc] + (cs[0] * tCross) - (cs[1] * (t + tCross))
        #     xShiftR[ixcR] = x_0[ixcR] + (cs[0] * tCrossR) - (cs[1] * (t + tCrossR))
        #     xShiftL[ixcL] = x_0[ixcL] + (cs[0] * tCrossL) - (cs[1] * (t + tCrossL))

        #     print(xShift)
        #     print(xShiftR)
        #     print(xShiftL)
        #     print('')

        #     xShiftR = xShiftR[1:]
        #     xShiftL = xShiftL[:-1]
        else:
            xShift = (2 * locs[0]) - (cs[0] * t) - x_0
    
    return xShift#, xShiftL, xShiftR



def SquareWave(omega, x, width, center, deriv, cellAve):
    # Unpack requisite attributes from omega and physics.
    errorLoc = 'ERROR:\nWaveformTools:\nSquareWave:\n'
    degFreed = omega.degFreed
    xL = x[:-1]
    xR = x[1:]
    
    edgeL = center - (width / 2)
    edgeR = center + (width / 2)
    sqWave = np.zeros(degFreed, float)
    sqWave[xL >= edgeL] = 1
    sqWave[xR > edgeR] = 0
    if (cellAve):
        if ((edgeL >= 0) and (np.shape(np.where(xL == edgeL)[0])[0] == 0)):
            leftPtLInd = np.where(xL < edgeL)[0][-1]
            rightPtLInd = leftPtLInd + 1
            area = x[rightPtLInd] - edgeL
            dx = x[rightPtLInd] - x[leftPtLInd]
            sqWave[leftPtLInd] = area / dx
        if ((edgeR >= 0) and (np.shape(np.where(xR == edgeR)[0])[0] == 0)):
            leftPtRInd = np.where(xL < edgeR)[0][-1]
            rightPtRInd = leftPtRInd + 1
            area = edgeR - x[leftPtRInd]
            dx = x[rightPtRInd] - x[leftPtRInd]
            sqWave[leftPtRInd] = area / dx
    
    if (deriv):
        errorMess = 'This function includes no provision for the derivative!'
        sys.exit(errorLoc + errorMess)
    
    return sqWave