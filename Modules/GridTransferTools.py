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
# This function creates an array of face values which flank the cells that will be used to interpolate the ghost
# cell up to the given order. It also outputs the number of coarse and fine cells used in the interpolation. It
# selects the order-plus-one many cells closest to the ghost cell location.
# ----------------------------------------------------------------------------------------------------------------
# Inputs:
#
# order                   int                    Order of interpolation
# x_0                     float                  Farthest face from zero point of cell to interpolate
# ----------------------------------------------------------------------------------------------------------------
# Outputs:
#
# bounds                  array                  (order+2) many face values around cells used for interpolation
# n_c                     int                    Number of coarse cells used in interpolation
# n_f                     int                    Number of fine cells used in interpolation
# ----------------------------------------------------------------------------------------------------------------

def BoundVals(order, x_0):
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
    return bounds, n_c, n_f


# ----------------------------------------------------------------------------------------------------------------
# Function: MomentVander
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# This function creates a finite-volume Vandermonde matrix from the bounds array and then multiplies its inverse
# with the xVec array of cell averaged polynomial ghost cell values to find the polynomial interpolation for that
# ghost cell.
# ----------------------------------------------------------------------------------------------------------------
# Inputs:
#
# order                   int                    Order of interpolation
# bounds                  array                  (order+2) many face values around cells used for interpolation
# xVec                    array                  Cell-averaged vector order+1-order polynomial of ghost cell
#                                                    values
# ----------------------------------------------------------------------------------------------------------------
# Outputs:
#
# polyInterp              array                  (order+1)-sized array of polynomial interpolation of ghost cell
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
#     print(hInv)
#     print(A)
#     print(B)
#     print(polyCoefs)
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
# This function takes the face value of the side of the ghost cell farthest from the central point to construct a
# finite-volume polynomial interpolation from the cells nearest to the location of the ghost cell. It also outputs
# the number of coarse and fine cells used in the interpolation.
# ----------------------------------------------------------------------------------------------------------------
# Inputs:
#
# order                   int                    Order of interpolation
# x_0                     float                  Farthest face from zero point of cell to interpolate
# ----------------------------------------------------------------------------------------------------------------
# Outputs:
#
# polyInterp              array                  (order+1)-sized array of polynomial interpolation of ghost cell
# n_c                     int                    Number of coarse cells used in interpolation
# n_f                     int                    Number of fine cells used in interpolation
# ----------------------------------------------------------------------------------------------------------------

def GhostCellStencil(order, x_0, h_rat = 1):
    errorLoc = 'ERROR:\nGridTransferTools:\nGhostCellStencil:\n'
    errorMess = ''
    
#     intCoefs = (np.arange(order + 1) + 1)[::-1]**-1.
#     polyCoefs = np.diag(intCoefs)
# #     print('polyCoefs:', polyCoefs)
#     if (x_0 % 0.5 != 0):
#         errorMess = 'x_0 must be multiple of 0.5!'
#     else:
#         if (x_0 > 0):
#             xValsR = np.polynomial.polynomial.polyvander(x_0, order + 1)[0][1:][::-1] / 0.5   
#             xValsL = np.polynomial.polynomial.polyvander(x_0 - 0.5, order + 1)[0][1:][::-1] / 0.5
#         else:
#             if (x_0 < 0):
#                 xValsR = np.polynomial.polynomial.polyvander(x_0 + 0.5, order + 1)[0][1:][::-1] / 0.5
#                 xValsL = np.polynomial.polynomial.polyvander(x_0, order + 1)[0][1:][::-1] / 0.5
#             else:
#                 errorMess = 'x_0 cannot be zero!'
#     if (errorMess != ''):
#         sys.exit(errorLoc + errorMess)

#     xVec = (xValsR - xValsL) @ polyCoefs

    xVec = InterpVec(order, x_0, h_rat = h_rat)
    
    bounds, n_c, n_f = BoundVals(order, x_0)
    bounds = h_rat * bounds

    polyInterp = MomentVander(order, bounds, xVec)
    
    return polyInterp, n_c, n_f


def InterpVec(order, x_0, h_rat = 1):
    errorLoc = 'ERROR:\nGridTransferTools:\nInterpVec:\n'
    errorMess = ''
    intCoefs = (np.arange(order + 1) + 1)[::-1]**-1.
    polyCoefs = np.diag(intCoefs)
#     print('polyCoefs:', polyCoefs)
    if (x_0 % 0.5 != 0):
        errorMess = 'x_0 must be multiple of 0.5!'
    else:
        if (x_0 > 0):
            xValsR = np.polynomial.polynomial.polyvander(h_rat * x_0, order + 1)[0][1:][::-1] / (h_rat * 0.5)
            xValsL = np.polynomial.polynomial.polyvander(h_rat * (x_0 - 0.5), order + 1)[0][1:][::-1] / (h_rat * 0.5)
        else:
            if (x_0 < 0):
                xValsR = np.polynomial.polynomial.polyvander(h_rat * (x_0 + 0.5), order + 1)[0][1:][::-1] / (h_rat * 0.5)
                xValsL = np.polynomial.polynomial.polyvander(h_rat * x_0, order + 1)[0][1:][::-1] / (h_rat * 0.5)
            else:
                errorMess = 'x_0 cannot be zero!'
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)

    xVec = (xValsR - xValsL) @ polyCoefs
    return xVec




# ----------------------------------------------------------------------------------------------------------------
# Function: CentGhost
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# This function centers the polynomial interpolation for a ghost cell about the appropriate coarse-fine or fine-
# coarse boundary. If the grid is uniform, it outputs an array of zeros.
# ----------------------------------------------------------------------------------------------------------------
# Inputs:
#
# omega                   Grid                   Object containing all grid attributes
# order                   int                    Order of interpolation
# x_0                     float                  Farthest face from zero point of cell to interpolate
# ----------------------------------------------------------------------------------------------------------------
# Outputs:
#
# fullStenc               array                  (degFreed)-sized array of ghost cell interpolation centered at
#                                                    patch boundary or zeros
# ----------------------------------------------------------------------------------------------------------------

def CentGhost(omega, order, x_0):
    errorLoc = 'ERROR:\nGridTransferTools:\nCentGhost:\n'
    errorMess = ''
    
    degFreed = omega.degFreed
    hs = omega.h
    
    h_min = min(hs)
    
    h_rat = h_min / 0.5
    
    spots = np.roll(hs, -1) - hs
    
    if (all(spots == 0)):
        fullStenc = np.zeros(degFreed, float)
        n_c = 0
        n_f = 0
    else:
        # Index before fine-coarse interface
        p = np.where(spots > 0)[0][0]
        # Index before coarse-fine interface
        q = np.where(spots < 0)[0][0]
    
        h_c = max(hs)
        h_f = min(hs)

        n_c_m = list(hs).count(h_c)
        n_f_m = list(hs).count(h_f)

        ghostCell, n_c, n_f = GhostCellStencil(order, x_0, h_rat = h_rat)

        if (n_c > n_c_m):
            errorMess = 'This grid has too few coarse cells for the order of the polynomial interpolation!'
        if (n_f > n_f_m):
            errorMess = 'This grid has too few fine cells for the order of the polynomial interpolation!'

        cells = n_c + n_f


        fullStenc = np.zeros(degFreed, float)

        if (x_0 > 0):
            for k in range(cells):
                index = (p - n_f + k + 1) % degFreed
                fullStenc[index] = ghostCell[k]
        else:
            if (x_0 < 0):
                for k in range(cells):
                    index = (q - n_c + k + 1) % degFreed
                    fullStenc[index] = ghostCell[k]


        if (errorMess != ''):
            sys.exit(errorLoc + errorMess)
    
    
    return fullStenc, n_c, n_f








































def CentGhostMaterial(omega, order, matInd, centCellInd, offDiagInd, revBounds = False):
    errorLoc = 'ERROR:\nGridTransferTools:\nCenterGhostCellMaterial:\n'
    errorMess = ''
    
    degFreed = omega.degFreed
    centCellInd = centCellInd % degFreed
    
    stenc = np.zeros(degFreed, float)
    
    ghostStenc = GhostCellMaterialStencil(omega, order, matInd, centCellInd, offDiagInd, revBounds)
    
    if ((type(offDiagInd) != int) or (offDiagInd == 0)):
        errorMess = 'offDiagInd must be a nonzero integer identifying the number of cells from material boundary!'
    else:
        if (revBounds or (offDiagInd > 0)):
            if (matInd == degFreed - 1):
                stenc[:order + 1] = ghostStenc
            else:
                stenc[matInd + 1:matInd + order + 2] = ghostStenc
        else:
            if (matInd == degFreed - 1):
                stenc[degFreed - order - 1:degFreed] = ghostStenc
            else:
                stenc[matInd - order:matInd + 1] = ghostStenc # [matInd - order:matInd + 1]
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
    
    return stenc


def GhostCellMaterialStencil(omega, order, matInd, centCellInd, offDiagInd, revBounds):
    errorLoc = 'ERROR:\nGridTransferTools:\nGhostCellMaterialStencil:\n'
    errorMess = ''
    
    bounds = MaterialInterpBounds(omega, order, matInd, offDiagInd, revBounds)

    xVec, xL, xR = MaterialInterpVec(omega, order, centCellInd, offDiagInd)
#     print('bounds =', bounds)
#     print('xVec =', xVec)

    polyInterp = MomentVander(order, bounds, xVec)
#     print('polyInterp is', polyInterp)
    
    return polyInterp


def MaterialInterpVec(omega, order, centCellInd, offDiagInd):
    xNode = omega.xNode
    xCell = omega.xCell
    degFreed = omega.degFreed
#     print('centCellInd is', centCellInd)
    
    intCoefs = (np.arange(order + 1) + 1)[::-1]**-1.
    polyCoefs = np.diag(intCoefs)
    
    bounds = xNode[centCellInd:centCellInd + 2]
    h = bounds[-1] - bounds[0]
    bounds = bounds - (offDiagInd * h) # Possible source of error.
    
#     print('bounds is', bounds, '(This isn\'t right yet.)')
    
    xL = bounds[0]
    xR = bounds[1]
    
    xValsL = np.polynomial.polynomial.polyvander(bounds[0], order + 1)[0][1:][::-1] / h
    xValsR = np.polynomial.polynomial.polyvander(bounds[-1], order + 1)[0][1:][::-1] / h  
    
    xVec = (xValsR - xValsL) @ polyCoefs
    
    return xVec, xL, xR


def MaterialInterpBounds(omega, order, matInd, offDiagInd, revBounds):
    errorLoc = 'ERROR:\nGridTransferTools:\nMaterialInterpBounds:\n'
    errorMess = ''
    
    xNode = omega.xNode
    degFreed = omega.degFreed
    
    if ((type(offDiagInd) != int) or (offDiagInd == 0)):
        errorMess = 'offDiagInd must be a nonzero integer identifying the number of cells from material boundary!'
    else:
        if (revBounds or (offDiagInd > 0)): # Possible source of error.
#             print('k should be positive, or this is an exception case.')
            if (matInd == degFreed - 1):
#                 print('This case.')
                bounds = xNode[:order+2]
            else:
                bounds = xNode[matInd+1:matInd+order+3] # [matInd:matInd+order+2]
        else:
#             print('k should be negative.')
            if (matInd == degFreed - 1):
#                 print('That case.')
                bounds = xNode[degFreed-order-1:] #[degFreed-order-2:degFreed]
            else:
                bounds = xNode[matInd-order-1:matInd + 1]
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
    
#     print('Bounds are:', bounds, '<=', xNode[matInd])
    
    return bounds