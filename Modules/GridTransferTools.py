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
import itertools as it
from IPython.core.display import HTML
from Modules import BasicTools as BT
from Modules import WaveTools as WT
from Modules import PlotTools as PT
from Modules import FFTTools as FFTT


# This function creates an injection operator.

# In[2]:


def MakeInject(nh):
    n2h = int(nh / 2)
    inject = np.zeros((n2h, nh), float)
    for i in range(n2h):
        inject[i, (2 * i) + 1] = 1
    return inject


# This function creates a full-weighting operator.

# In[3]:


def MakeFullWeight(nh):
    n2h = int(nh / 2)
    fullWeight = np.zeros((n2h, nh), float)
    weights = [0.5, 0.5]
    for i in range(n2h):
        fullWeight[i, (2 * i):(2 * i) + 2] = weights
    return fullWeight


# This function creates a piecewise interpolation operator.

# In[4]:


def MakePiecewise(nh):
    nh2 = 2 * nh
    piecewise = np.zeros((nh2, nh), float)
    weights = [1, 1]
    for i in range(nh):
        piecewise[(2 * i):(2 * i) + 2, i] = weights
    return piecewise


# This function creates a linear interpolation operator.

# In[5]:


def MakeLinearInterp(nh):
    nh2 = 2 * nh
    linearInterp = np.zeros((nh2, nh), float)
    weights = [1, 1]
    for i in range(nh):
        linearInterp[(2 * i):(2 * i) + 2, i] = weights
    return linearInterp


# In[ ]:

def CoarsenOp(omega):
    hs = omega.h
    nh_max = omega.nh_max
    h_min = 1. / nh_max
    weights = h_min / hs
    sizes = np.asarray(1 / weights, int)
    matrices = [w * np.ones(s, float) for (w, s) in zip(weights, sizes)]
    CoarseOp = LA2.block_diag(*matrices)
    return CoarseOp

# ----------------------------------------------------------------------------------------------------------------
# Function: BoundVals
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# This function 
# ----------------------------------------------------------------------------------------------------------------
# Inputs:
#
# order                   int                    Order of interpolation
# x_0                     float                  Farthest face from zero point of cell to interpolate
# ----------------------------------------------------------------------------------------------------------------
# Outputs:
#
# bounds                  array                  Face values around cells used for interpolation
# ----------------------------------------------------------------------------------------------------------------

def BoundVals(order, x_0):
    print('YOU\'RE USING THE UPDATE.')
    a = abs(int(x_0 / 0.5))
    cells = order + 1
    if (cells <= a):
        n_c = cells
    else:
        n_c = a
        cells_new = cells - a
        if ((cells_new + 1) % 3 == 0):
            n_c_add = int(np.floor((cells_new + 1) / 3) - 1)
        else:
            n_c_add = int(np.floor((cells_new + 1) / 3))
        n_c = n_c + n_c_add
    n_f = cells - n_c
    
    if (n_f != 0):
        bounds = np.linspace(-n_c, n_f / 2., num = (2 * n_c) + n_f + 1)
        rm = [(2 * k) + 1 for k in range(n_c)]
        bounds = np.delete(bounds, rm)
        if (x_0 > 0):
            bounds = -bounds[::-1]
    else:
        if (x_0 % 1 == 0):
            bounds = np.arange(n_c + 1) + (int(x_0) - int((n_c + 1) / 2))
        else:
            bounds = np.arange(n_c + 1) + (int(x_0) - int(n_c / 2))
    print('bounds:', bounds)
    return bounds, n_c, n_f

def BoundVals1(order, x_0):
    print('')
    print('START BoundVals() FUNC!')
    
    if ((order + 1) % 3 == 0):
        print('HERE!')
        n_c = int(np.floor((order + 1) / 3))
    else:
        n_c = int(np.floor(((order + 1) / 3) + 1))
    print('n_c:', n_c)
    
    n_f = order + 1 - n_c
    print('n_f:', n_f)
    bounds = np.linspace(-n_c, n_f / 2., num = (2 * n_c) + n_f + 1)
    print('bounds:', bounds)
    rm = [(2 * k) + 1 for k in range(n_c)]
    print('rm:', rm)
    bounds = np.delete(bounds, rm)
    print('bounds:', bounds)
    if (x_0 > 0):
         bounds = -bounds[::-1]
    print('bounds:', bounds)
    print('END BoundVals() FUNC!')
    print('')
    return bounds

# ----------------------------------------------------------------------------------------------------------------
# Function: MomentVander
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# This function 
# ----------------------------------------------------------------------------------------------------------------
# Inputs:
#
# order                   int                    Order of interpolation
# bounds                  array                  Face values around cells used for interpolation
# xVec                    float                  (Not sure)
# ----------------------------------------------------------------------------------------------------------------
# Outputs:
#
# polyInterp              array                   (Not sure)
# ----------------------------------------------------------------------------------------------------------------

def MomentVander(order, bounds, xVec):
#     print('')
#     print('START MomentVander() FUNC!')
    # Add error catchers!
    intCoefs = (np.arange(order + 1) + 1)[::-1]**-1.
#     print('intCoefs:', intCoefs)
    polyCoefs = np.diag(intCoefs)
#     print('polyCoefs:', polyCoefs)
    h = (bounds[1:] - bounds[:-1])**-1.
#     print('h:', h)
    hInv = np.diag(h)
#     print('h:', h)
    A = np.diag(bounds[1:]) @ np.vander(bounds[1:])
#     print('A:', A)
    B = np.diag(bounds[:-1]) @ np.vander(bounds[:-1])
#     print('B:', B)
    VanderMat = hInv @ (A - B) @ polyCoefs
#     print('VanderMat:', VanderMat)
    polyInterp = xVec @ LA2.inv(VanderMat)
#     print('polyInterp:', polyInterp)
#     print('END MomentVander() FUNC!')
#     print('')
    return polyInterp

# ----------------------------------------------------------------------------------------------------------------
# Function: GhostCellStencil
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# This function 
# ----------------------------------------------------------------------------------------------------------------
# Inputs:
#
# order                   int                    Order of interpolation
# x_0                     float                  Farthest face from zero point of cell to interpolate
# ----------------------------------------------------------------------------------------------------------------
# Outputs:
#
# polyInterp              array                   Polynomial interpolation of ghost cell
# ----------------------------------------------------------------------------------------------------------------

def GhostCellStencil(order, x_0):
    print(x_0)
    errorLoc = 'ERROR:\nGridTransferTools:\nGhostCellStencil:\n'
    errorMess = ''
#     print('')
#     print('START GhostCellStencil() FUNC!')
    intCoefs = (np.arange(order + 1) + 1)[::-1]**-1.
#     print(intCoefs)
#     print('intCoefs:', intCoefs)
    polyCoefs = np.diag(intCoefs)
#     print('polyCoefs:', polyCoefs)
    if (x_0 > 0):
        xValsR = np.polynomial.polynomial.polyvander(x_0, order + 1)[0][1:][::-1] / 0.5   
        xValsL = np.polynomial.polynomial.polyvander(x_0 - 0.5, order + 1)[0][1:][::-1] / 0.5
    else:
        if (x_0 < 0):
            xValsR = np.polynomial.polynomial.polyvander(x_0 + 0.5, order + 1)[0][1:][::-1] / 0.5
            xValsL = np.polynomial.polynomial.polyvander(x_0, order + 1)[0][1:][::-1] / 0.5
        else:
            errorMess = 'x_0 cannot be zero!'
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
    print('xValsR:', xValsR)
    print('xValsL:', xValsL)
    print(np.polynomial.polynomial.polyvander(x_0, order)[0][::-1])
    xVec = (xValsR - xValsL) @ polyCoefs
#     print(np.polynomial.polynomial.polyvander(x_0, order)[0][::-1])
#     print('xVec:', xVec)
    bounds, n_c, n_f = BoundVals(order, x_0)
#     print('bounds:', bounds)
    polyInterp = MomentVander(order, bounds, xVec)
#     print('polyInterp:', polyInterp)
#     print('END GhoseCellStencil() FUNC!')
    print('')
    return polyInterp, n_c, n_f




