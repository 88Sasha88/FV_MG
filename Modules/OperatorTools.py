#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os.path
import scipy as sp
from scipy import *
import numpy as np
from numpy import *
import sympy as sympy
from numpy import linalg as LA
from scipy import linalg as LA2
import sys as sys
import time
import matplotlib.pyplot as plt
from fractions import Fraction
from Modules import BasicTools as BT
from Modules import GridTransferTools as GTT


# This function normalizes the vectors of a matrix. As the default, it normalizes the column vectors. To change to row vectors, set axis equal to 1.


# In[2]:


def NormalizeMatrix(n, matrica, axis = 0):
    errorLoc = 'ERROR:\nOperatorTools:\nNormalizeMatrix:\n'
    errorMess = BT.CheckSize(n, matrica)
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
    zeroMat = np.zeros((n, n), float)
    norms = LA.norm(matrica, axis = axis)
    np.fill_diagonal(zeroMat, norms)
    normMatrix = LA.inv(zeroMat)
    matricaNorm = matrica @ normMatrix
    return matricaNorm


# This function constructs a 1-D Laplacian operator.

# In[3]:


def Laplacian1D(n):
    Laplacian = np.zeros((n, n), float)
    np.fill_diagonal(Laplacian[1:], 1)
    np.fill_diagonal(Laplacian[:, 1:], 1)
    np.fill_diagonal(Laplacian, -2)
    Laplacian[n - 1, 0] = 1
    Laplacian[0, n - 1] = 1
    return Laplacian


# This function rounds the off-diagonal elements of what should be a diagonal matrix.

# In[4]:


def RoundDiag(matrica, places = 14):
    eigVals = np.diag(matrica)
    idealMat = np.zeros(np.shape(matrica), float)
    np.fill_diagonal(idealMat, eigVals)
    matrica = np.round(matrica - idealMat, places) + idealMat
    return matrica


# This function finds the nullspace of the oscillatory waves in the coarse grid region and outputs a matrix which, when multiplied to `waves`, will create the correct number of orthogonal wave modes (Should this be true 100% of the time?) which are zero in the coarse grid space.

# In[5]:


def FindNullspace(omega, waves, shift = False, Hans = False):
    errorLoc = 'ERROR:\nOperatorTools:\nFindNullspace:\n'
    levels = omega.levels
    alias = omega.alias
    nh_min = omega.nh_min
    nh_max = omega.nh_max
    nh = nh_max
    degFreed = omega.degFreed# [::-1][0]
    fixWaves = np.zeros((nh, degFreed), float)

    nh = nh_min
    hs = omega.h
    
    leftover = []
    oneMore = 0
    if (shift):
        oneMore = 0# 1
    if (np.all(hs[0] == hs)):# or (hs:
#         print('location 1: fixWaves is', np.shape(fixWaves))
        np.fill_diagonal(fixWaves, 1)
#         print('location 1.5: fixWaves is', np.shape(fixWaves))
        N = int(alias * nh_max)
        fixWaves = np.eye(N, N)
#         print('location 2: fixWaves is', np.shape(fixWaves))
    else:
        if (Hans):
#             print('location 3: fixWaves is', np.shape(fixWaves))
            np.fill_diagonal(fixWaves, 1)#[:, :-1], 1)
#             print('location 4: fixWaves is', np.shape(fixWaves))
            #fixWaves[-1, -1] = 1
        else:
            for q in range(levels):
                degFreed = omega.degFreeds[q + 1]
                refRatio = omega.refRatios[::-1][q]
                nh = omega.nh[q + 1]
                if (q == levels - 1):
                    errorMess = BT.CheckSize(nh, waves[0, :], nName = 'nh', matricaName = 'waves')
                    if (errorMess != ''):
                        sys.exit(errorLoc + errorMess)
                    errorMess = BT.CheckSize(degFreed, waves[:, 0], nName = 'degFreed', matricaName = 'waves')
                    if (errorMess != ''):
                        sys.exit(errorLoc + errorMess)

                h = refRatio / nh
                oscNum = int(nh / refRatio) - oneMore
                oscWaves = waves[:, oscNum:nh]
                allIndices = np.arange(oscNum, nh)
                if (shift):
                    indices = []
                else:
                    maxCos = int(np.floor(refRatio - ((2. * refRatio) / nh)))
                    cosKs = int(nh / (2. * refRatio)) * np.arange(1, maxCos + 1)
                    cosInd = 2 * cosKs
                    maxSin = int(np.floor(refRatio / 2))
                    sinKs = int(nh / refRatio) * np.arange(1, maxSin + 1)
                    sinInd = (2 * sinKs) - 1
                    indices = np.sort(np.append(cosInd, sinInd))


                    #
                    # print(otherIndices)
                    #
                    oscWaves = np.delete(oscWaves, indices-oscNum, 1)
                otherIndices = np.setdiff1d(allIndices, indices)
                fineSpots = np.where(hs < h)[0]
                oscWaves = np.delete(oscWaves, fineSpots, 0)
                # oscWaves = np.round(oscWaves, 15)
                nullspace = LA2.null_space(oscWaves)
                nullspace = np.asarray(sympy.Matrix(nullspace.transpose()).rref()[0].transpose())
                nullspace = np.round(nullspace.astype(np.float64), 14)
                nullspace = GramSchmidt(nullspace)
#                 print('Gram-Schmidt:')
#                 print(nullspace)
#                 print('')
                if (q == 0):
                    fixWaves[0:oscNum, 0:oscNum] = np.eye(oscNum, oscNum)
                    j = oscNum
                for i in leftover:
                    if (j < degFreed):
                        fixWaves[i, j] = 1
                        leftover = leftover[1:]
                        j = j + 1
                for i in indices:
                    if (j < degFreed):
                        fixWaves[i, j] = 1
                        j = j + 1
                    else:
                        leftover.append(i)
                i = 0
                while (j < degFreed):
                    fixWaves[otherIndices, j] = nullspace[:, i]
                    i = i + 1
                    j = j + 1
    fixWaves = np.round(fixWaves, 14)
#     print('location 5: fixWaves is', np.shape(fixWaves))
    return fixWaves


# Explain.

# In[6]:


def GramSchmidt(matrica):
    shape = np.shape(matrica)
    n = shape[1]
    v = matrica + 0
    q = np.zeros(shape, float)
#     for j in range(n):
#         vj = matrica[:, j]
    for j in range(n):
        q[:, j] = v[:, j] / LA.norm(v[:, j])
        for  k in range(j + 1, n):
            v[:, k] = v[:, k] - ((q[:, j].transpose() @ v[:, k]) * q[:, j])
#         print(q[:, j])
        q[:, j] = q[:, j] / LA.norm(q[:, j])
#         print(q[:, j])
#         print('')
#     print(matrica)
#     matrica = q
#     print(matrica)
#     n = np.shape(matrica)[1]
#     for i in range(n):
#         x = matrica[:, i]
#         for j in range(i):
#             constant = (x @ matrica[:, j].transpose()) / (LA.norm(matrica[:, j])**2)
#             term = constant * matrica[:, j]
#             matrica[:, i] = matrica[:, i] - term
#         norm = 1 / LA.norm(matrica[:, i])
#         matrica[:, i] = norm * matrica[:, i]
    return q

# ----------------------------------------------------------------------------------------------------------------
# Function: MakeRotMat
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# This function creates the Fourier rotation matrix.
# ----------------------------------------------------------------------------------------------------------------
# Inputs:
#
# omega                   Grid                    Object containing all grid attributes
# ct                      float                   Distance by which all modes must shift
# ----------------------------------------------------------------------------------------------------------------
# Outputs:
#
# shift                   array                   Fourier rotation matrix
# ----------------------------------------------------------------------------------------------------------------

def MakeRotMat(omega, ct):
    nh = omega.nh_max
    Cosine = lambda k: np.cos(2. * np.pi * k * ct)
    Sine = lambda k: np.sin(2. * np.pi * k * ct)
    RotMat = lambda k: np.asarray([Cosine(k), Sine(k), -Sine(k), Cosine(k)]).reshape(2, 2)
    rotMats = [RotMat(k) for k in range(int(nh / 2) + 1)]
    shift = LA2.block_diag(*rotMats)[1:-1, 1:-1]
    shift[0, 0] = Cosine(0)
    shift[-1, -1] = Cosine(int(nh / 2))
    return shift


# ----------------------------------------------------------------------------------------------------------------
# Function: StepMatrix
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# This function creates a diagonal matrix of step sizes for the given grid.
# ----------------------------------------------------------------------------------------------------------------
# Inputs:
#
# omega                   Grid                    Object containing all grid attributes
# ----------------------------------------------------------------------------------------------------------------
# Outputs:
#
# hMat                    array                   Diagonal matrix of step sizes
# ----------------------------------------------------------------------------------------------------------------

def StepMatrix(omega):
    h = omega.h
    n = omega.degFreed
    hs = np.zeros((n, n), float)
    np.fill_diagonal(hs, h)
    hMat = LA.inv(hs)
    return hMat

# In[ ]:

def Curl(omega, order, diff):
    derivOp = SpaceDeriv(omega, order, diff)
    
    return

def Grad(omega, order, diff):
    
    return


# ----------------------------------------------------------------------------------------------------------------
# Function: Block
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# This function creates either a block diagonal or block anti-diagonal matrix from some operator or list of
# operators. If the input matrica is a single operator, then a matrix of var many blocks of that operator is
# constructed. If matrica is input directly as a list of operators, then a block diagonal of all the operators
# within that list is constructed. If the parameter diag is set to False, then the output is changed to a block
# antidiagonal matrix.
# ----------------------------------------------------------------------------------------------------------------
# Inputs:
#
# matrica                 array/list              Operator or list of operators to be reconstructed in blocks
# (var)                   int                     Number of blocks if matrica entered as array
# (diag)                  boolean                 Switch paramater for diagonal or antidiagonal blocks
# ----------------------------------------------------------------------------------------------------------------
# Outputs:
#
# matrice                 array                   Block diagonal or antidiagonal matrix
# ----------------------------------------------------------------------------------------------------------------

def Block(matrica, var = 1, diag = True):
    errorLoc = 'ERROR:\nOperatorTools:\nBlock:\n'
    errorMess = ''
    if ((var < 1) or (type(var) != int)):
        errorMess = 'var must be integer value greater than 0!'
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
    if (type(matrica) == list):
        var = np.shape(matrica)[0]
        matricaList = matrica
    else:
        matricaList = [matrica for k in range(var)]
    if (not diag):
        matricaList = [M[:, ::-1] for M in matricaList]
    matrice = LA2.block_diag(*matricaList)
    if (not diag):
        matrice = matrice[:, ::-1]
    return matrice

# ----------------------------------------------------------------------------------------------------------------
# Function: FourierTransOp
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# This function creates a Fourier Transform (or pseudo-Fourier Transform) operator for a uniform (or AMR) grid.
# ----------------------------------------------------------------------------------------------------------------
# Inputs:
#
# waves                   array/list              Matrix of waves in domain for Fourier Transform operator
# ----------------------------------------------------------------------------------------------------------------
# Outputs:
#
# FTOp                    array                   Block diagonal or antidiagonal matrix
# ----------------------------------------------------------------------------------------------------------------

def FourierTransOp(waves):

    prenorm = waves.transpose() @ waves
    norm = LA.inv(prenorm)
    FTOp = norm.transpose() @ waves.transpose()
    
    return FTOp


# ----------------------------------------------------------------------------------------------------------------
# Function: ExactSpatDerivOp
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# This function generates an exact Fourier derivative operator D, which can be multiplied with the Fourier matrix
# F and the Fourier coefficients A like FDA to find the exact derivative of the operation FA.
# ----------------------------------------------------------------------------------------------------------------
# Inputs:
#
# omega                   Grid                    Object containing all grid attributes
# ----------------------------------------------------------------------------------------------------------------
# Outputs:
#
# SpatOp                  array                   nh_max x nh_max Fourier derivative operator
# ----------------------------------------------------------------------------------------------------------------

def ExactSpatDerivOp(omega):
    print('You are using ExactSpatDerivOp in OperatorTools module (which I believe is what you want)!')

    nh_max = omega.nh_max
    omegaF = BT.Grid(nh_max)

    subsuper = np.linspace(0.5, nh_max, num = 2 * nh_max)
    subsuper[::2] = 0
    Op = np.zeros((nh_max, nh_max), float)
    np.fill_diagonal(Op[1:], subsuper[:])
    np.fill_diagonal(Op[:, 1:], -subsuper)
    SpatOp = 2 * np.pi * Op
    print(Op)
    return SpatOp


# ----------------------------------------------------------------------------------------------------------------
# Function: ExactTimeDerivOp
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# This function performs an exact Fourier derivative on some input function u0, given in space-space.
# ----------------------------------------------------------------------------------------------------------------
# Inputs:
#
# omega                   Grid                    Object containing all grid attributes
# t                       float                   Inert parameter included for flexibility of use
# u0                      array                   Initial waveform in space-space of length degFreed
# c                       float                   Constant value
# order                   float                   Inert parameter included for flexibility of use
# ----------------------------------------------------------------------------------------------------------------
# Outputs:
#
# u                       array                   Derivative of initial waveform in space-space of length degFreed
# ----------------------------------------------------------------------------------------------------------------

def ExactTimeDerivOp(omega, physics, waves):
    print('You are using ExactSpatDeriv in OperatorTools module!')
    
    cMat = physics.cMat
    
    nullspace = FindNullspace(omega, waves)
    
    SpatOp = ExactSpatDerivOp(omega)
    FTOp = nullspace @ FourierTransOp(waves @ nullspace)
#     FCoefs = nullspace @ FFTT.FourierCoefs(waves @ nullspace, u0)
#     u = -cMat @ waves @ SpatOp @ FCoefs
    ETDerivOp = -cMat @ waves @ SpatOp @ FTop
    return ETDerivOp



def CDStencil(orderIn):
    if (orderIn % 2 == 0):
        order = orderIn
    else:
        order = int(orderIn + 1)
    
    loops = int(order / 2)
        
    coefs = np.zeros(loops)
    stenc = np.zeros(order + 1)
    terms = np.arange(order + 1)
    rCell = np.asarray([1 / sp.math.factorial(j) for j in terms])
    lCell = rCell + 0
    lCell[1::2] = -lCell[1::2]
    deltaXFunc = lambda k: k ** terms
    tExp = [[] for j in range(loops)]
    for k in range(loops):
        rCellNew = rCell * deltaXFunc(k + 1)
        lCellNew = lCell * deltaXFunc(k + 1)
        tExp[k] = rCellNew - lCellNew


    tExp = np.asarray(tExp).transpose()
    mat = tExp[1::2, :][1:, :-1]
    vec = tExp[1::2, :][1:, -1]
    vec = -vec
    coefs[-1] = 1
    coefs[:-1] = LA.inv(mat) @ vec
    stenc[:loops] = -coefs[::-1]
    stenc[loops + 1:] = coefs
    stenc = (-1)**(loops + 1) * stenc
    val = abs((tExp @ coefs)[1])
    stenc = stenc / val
    
    return stenc

def UDStencil1(order):
    coefs = np.zeros(order + 1)
    stenc = np.zeros(order + 1)
    terms = np.arange(order + 1)
    cell = np.asarray([1 / sp.math.factorial(j) for j in terms])
    deltaXFunc = lambda k: k ** terms
    tExp = [[] for j in range(order + 1)]
    for k in range(order + 1):
        cellNew = cell * deltaXFunc(k)
        tExp[k] = cellNew
    
    tExp = np.asarray(tExp).transpose()
    mat = np.zeros((order, order), float)
    vec = np.zeros(order, float)
    mat[0, :] = tExp[0, :order]
    mat[1:, :] = tExp[2:, :order]
    vec[0] = tExp[0, order]
    vec[1:] = tExp[2:, order]
    vec = -vec
    stenc[order] = 1
    stenc[:order] = LA.inv(mat) @ vec
    val = (tExp @ stenc)[1]
    
    stenc = stenc / val

    return stenc

def UDStencil(orderIn):
    errorLoc = 'ERROR:\nOperatorTools:\nUDStencil:\n'
    errorMess = ''
    if (orderIn % 2 == 0):
        order = int(orderIn + 1)
    else:
        order = orderIn
    
    stenc = np.zeros(order + 1)
    
    if (order == 1):
        faceR = np.asarray([0, 1])
    else:
        if (order == 3):
            faceR = (1. / 6.) * np.asarray([0, -1, 5, 2])
        else:
            if (order == 5):
                faceR = (1. / 60.) * np.asarray([0, 2, -13, 47, 27, -3])
            else:
                if (order == 7):
                    faceR = (1. / 420.) * np.asarray([0, -3, 25, -101, 319, 214, -38, 4])
                else:
                    if (order == 9):
                        faceR = (1. / 2520.) * np.asarray([0, 4, -41, 199, -641, 1879, 1375, -305, 55, -5])
                    else:
                        errorMess = 'This program is not designed to handle this order of accuracy for forward- and backward-difference operators.'
    
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
    
    faceL = np.roll(faceR, -1)
    stenc = faceR - faceL
    
    return stenc

def DDStencil(order):
    stenc = -UDStencil(order)[::-1]
    return stenc

def SpaceDeriv(omega, order, diff):
    errorLoc = 'ERROR:\nOperatorTools:\nMakeSpaceDeriv:\n'
    errorMess = ''
    if (diff == 'C' or diff == 'CD'):
        stenc = CDStencil(order)
        if (order % 2 == 0):
            orderStenc = order
        else:
            orderStenc = int(order + 1)
        off = int(orderStenc / 2)
        loBound = -off / 2.
        hiBound = off / 2.
    else:
        orderStenc = order
        if (order % 2 == 0):
            orderStenc = int(order + 1)
        else:
            orderStenc = order
        off = ((orderStenc + 1) / 2)
        if (diff == 'U' or diff == 'UD'):
            stenc = UDStencil(order)
            loBound = -off / 2.
            hiBound = (off - 1.) / 2.
        else:
            if (diff == 'D' or diff == 'DD'):
                stenc = DDStencil(order)
                off = int(off - 1)
                loBound = -off / 2.
                hiBound = (off + 1.) / 2.
            else:
                errorMess = 'Invalid entry for variable diff. Must be \'C\', \'U\', \'D\', \'CD\', \'UD\', or \'DD\'.'
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
    
#     stenc = np.ones(orderStenc + 1)
    
    degFreed = omega.degFreed
    hs = omega.h
    
    spots = np.roll(hs, -1) - hs
    
    if (all(spots == 0)):
        p = []
        q = []
        NU = False
    else:
        # Index before fine-coarse interface
        p = np.where(spots > 0)[0][0]
        # Index before coarse-fine interface
        q = np.where(spots < 0)[0][0]
        NU = True
    
    polyStencSet = [[] for i in range(orderStenc)]
    cellFaces = np.linspace(loBound, hiBound, num = orderStenc + 1)
    zeroLoc = np.where(cellFaces == 0)[0][0]
    cellFaces = np.delete(cellFaces, zeroLoc)

    for i in range(orderStenc):
        polyStencSet[i], n_c, n_f = GTT.CentGhost(omega, order, cellFaces[i])
    
    polyStencSet = np.asarray(polyStencSet)

    IMat = np.eye(degFreed, degFreed)
    
    # YOU'RE GONNA NEED THESE TO RESTRICT FOR HIGHER EVEN ORDERS, TOO.
    
    
    polyMatU = IMat + 0
    
    
    mat = np.zeros((degFreed, degFreed), float)
    derivOp = mat + 0
    
    for d in range(orderStenc + 1):
        s = int(off - d)
        
        derivMat = mat + 0
        np.fill_diagonal(derivMat, stenc[d])
        derivMat = np.roll(derivMat, s, axis = 0)
        
        polyMat = IMat + 0
        
        if (NU):
            if (s > 0):
                j = int(off - s)
                pAt = p
                pLow = (p - 1) % degFreed
                qAt = (q - s + 1) % degFreed #(q + 1) % degFreed
                for i in range (s):
                    polyMat[pAt, :] = 0
                    polyMat[pAt, pLow:pLow+2] = 0.5
                    polyMat[qAt, :] = polyStencSet[j, :]
                    pAt = (pAt - 1) % degFreed
                    pLow = (pLow - 2) % degFreed
                    qAt = (qAt + 1) % degFreed
                    j = int(j + 1)

            if (s < 0):
                j = int(off) # - s - 1
                qAt = (q + 1) % degFreed
                qLow = (q + 1) % degFreed
                pAt = (p + 1) % degFreed#p
                for i in range(abs(s)):
                    polyMat[qAt, :] = 0
                    polyMat[qAt, qLow:qLow+2] = 0.5
                    polyMat[pAt, :] = polyStencSet[j, :]
                    qAt = (qAt + 1) % degFreed
                    qLow = (qLow + 2) % degFreed
                    pAt = (pAt + 1) % degFreed
                    j = int(j + 1) # - 1
        
        matThis = derivMat @ polyMat
        
        derivOp = derivOp + matThis
    
    hMat = StepMatrix(omega)
    
    derivOp = hMat @ derivOp

        
    return derivOp



def UDFace(orderIn):
    errorLoc = 'ERROR:\nOperatorTools:\nUDFace:\n'
    errorMess = ''
    if (orderIn % 2 == 0):
        order = int(orderIn + 1)
    else:
        order = orderIn
    
    stenc = np.zeros(order + 1)
    
    if (order == 1):
        face = np.asarray([1])
    else:
        if (order == 3):
            face = (1. / 6.) * np.asarray([-1, 5, 2])
        else:
            if (order == 5):
                face = (1. / 60.) * np.asarray([2, -13, 47, 27, -3])
            else:
                if (order == 7):
                    face = (1. / 420.) * np.asarray([-3, 25, -101, 319, 214, -38, 4])
                else:
                    if (order == 9):
                        face = (1. / 2520.) * np.asarray([4, -41, 199, -641, 1879, 1375, -305, 55, -5])
                    else:
                        errorMess = 'This program is not designed to handle this order of accuracy for forward- and backward-difference operators.'
    
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
    
    return face

def DDFace(order):
    stenc = UDFace(order)[::-1]
    return stenc

def CDFace(orderIn):
    errorLoc = 'ERROR:\nOperatorTools:\nCDFace:\n'
    errorMess = ''
    if (orderIn % 2 == 0):
        order = orderIn
    else:
        order = int(orderIn + 1)
    
    stenc = np.zeros(order + 1)
    
    if (order == 2):
        face = (1. / 2.) * np.asarray([1, 1])
    else:
        if (order == 4):
            face = (1. / 12.) * np.asarray([-1, 7, 7, -1])
        else:
            if (order == 6):
                face = (1. / 60.) * np.asarray([1, -8, 37, 37, -8, 1])
            else:
                errorMess = 'This program is not designed to handle this order of accuracy for center-difference operators.'
    
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
    
    return face


def MomentMatrix(x, x0, h, ixs, P):
    ilo = min(ixs)
    ihi = max(ixs)
    xhi = (x[ilo+1:ihi+2] - x0)/h
    xlo = (x[ilo:ihi+1] - x0)/h
    A = np.zeros((np.shape(ixs)[0],P), float)
    for p in range(P):
        A[:,p] = (xhi**(p + 1) - xlo**(p + 1)) / ((p + 1) * (xhi - xlo))
    A = np.asarray(A)
    return A




# Fill in cell-average ghost cells using jump conditions

def GhostCellsJump(omega, physics, phiavg,Ng,P):
    
    dx = omega.h[0]
    xNode = omega.xNode
    matInd = physics.matInd
    loc = physics.locs[0]
    
#     print('My x:')
#     print(xNode[matInd-P:matInd+P+1] - loc)
    
    # Create the cell average interpolation matrix
    x = xNode[matInd-P:matInd+P+1] - loc # np.arange(-P, P + 1).transpose()*dx
#     print('Hans\' x:')
#     print(x)
    print('')
    x0 = 0
    ixs = np.arange(2*P).transpose()
    A = MomentMatrix(x,x0,dx,ixs,P)

    # Build up an interpolant using the jump condition
    ix = np.arange(P)
    phi1 = phiavg[int(matInd-P)+ix] # phi avg in domain 1
    ix2 = np.arange(P)+P # domain 2 entries
    phi2 = phiavg[int(matInd-P)+ix2] # phi avg in domain 2
    B = Block([A[ix,:], A[ix2,:]]) # add the fit to the matrix
    addOn = np.zeros(2 * P, float)
    addOn[0] = 1
    addOn[P] = -1
    
    # Add the jump cond constraint - constant coef is same at x=0
    B = np.vstack([B, addOn])
    
    # Solve it with LS
    phic = LA.pinv(B)@np.concatenate([phi1, phi2, np.zeros(1, float)])

    # Evaluate the phi1 ghost cell values
    ix = P+np.arange(Ng)
    phig1 = A[ix,:]@phic[:P]

    # Evaluate the phi2 ghost cell values
    ix = np.arange(P-Ng, P)
    phig2 = A[ix,:]@phic[P:2*P]
    
    return phig1, phig2


def GhostCellsJumpNew(omega, physics, phiavg,Ng,P):
    print('WARNING: YOU ARE NOT USING HANS\' GhostCellsJump() FUNCTION!!!')
    
    dx = omega.h[0]
    xNode = omega.xNode
    matInd = physics.matInd
    loc = physics.locs[0]
    
#     print('My x:')
#     print(xNode[matInd-P:matInd+P+1] - loc)
    
    # Create the cell average interpolation matrix
    x = xNode[matInd-P:matInd+P+1] - loc # np.arange(-P, P + 1).transpose()*dx
#     print('Hans\' x:')
#     print(x)
    print('')
    x0 = 0
    ixs = np.arange(2*P).transpose()
    A = MomentMatrix(x,x0,dx,ixs,P)

    # Build up an interpolant using the jump condition
    ix = np.arange(P)
    phi1 = phiavg[int(matInd-P)+ix] # phi avg in domain 1
    ix2 = np.arange(P)+P # domain 2 entries
    phi2 = phiavg[int(matInd-P)+ix2] # phi avg in domain 2
    B = Block([A[ix,:], A[ix2,:]]) # add the fit to the matrix
#     addOn = np.zeros(2 * P, float)
#     addOn[0] = 1
#     addOn[P] = -1
    
    # Add the jump cond constraint - constant coef is same at x=0
#     B = np.vstack([B, addOn])
    
    # Solve it with LS
    phic = LA.inv(B)@np.concatenate([phi1, phi2]) # , np.zeros(1, float)])

    # Evaluate the phi1 ghost cell values
    ix = P+np.arange(Ng)
    phig1 = A[ix,:]@phic[:P]

    # Evaluate the phi2 ghost cell values
    ix = np.arange(P-Ng, P)
    phig2 = A[ix,:]@phic[P:2*P]
    
    return phig1, phig2



def FaceOp(omega, order, diff, RL, Ng, otherFace = False, AMROverride = False):
    errorLoc = 'ERROR:\nOperatorTools:\nFaceOp:\n'
    errorMess = ''
    
    if (diff == 'C' or diff == 'CD'):
        stenc = CDFace(order)
        if (order % 2 == 0):
            orderStenc = int(order - 1) # order
        else:
            orderStenc = order # int(order + 1)
        off = int(orderStenc / 2)
        hiBound = (off + 1.) / 2. # off / 2.
        if (RL == 'L'):
            off = int(off + 1)
            hiBound = (off - 1.) / 2. # off / 2.
        else:
            if (RL != 'R'):
                errorMess = 'Variable RL must be set either to \'R\' for right-moving wave or \'L\' for left-moving wave.'
        loBound = -off / 2.
    else:
        orderStenc = order
        if (order % 2 == 0):
            orderStenc = order # int(order + 1)
        else:
            orderStenc = int(order - 1) # order
        off = int(orderStenc / 2) # int((orderStenc + 1) / 2)
        if (diff == 'U' or diff == 'UD'):
            loBound = -off / 2.
            hiBound = off / 2. #(off - 1.) / 2.
            stenc = UDFace(order)
            if (RL == 'L'):
                stenc = stenc[::-1]
            else:
                if (RL != 'R'):
                    errorMess = 'Variable RL must be set either to \'R\' for right-moving wave or \'L\' for left-moving wave.'
        else:
            errorMess = 'Invalid entry for variable diff. Must be \'C\' or \'CD\', for center-difference, or \'U\' or \'UD\', for upwind-difference.'
    if (Ng > off + 1):
        errorMess = 'Too many ghost cells for this order of face approximation!'
    
    val = Ng + off
    
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
    
#     stenc = np.ones(orderStenc + 1)
    
    np.set_printoptions(precision=24, suppress=True)
    
    degFreed = omega.degFreed
    hs = omega.h
    
    halfDeg = int(degFreed / 2)
      
    spots = np.roll(hs, -1) - hs
    
    if ((all(spots == 0)) or AMROverride):
        p = []
        q = []
        NU = False
        print('THIS OPERATOR IS UNIFORM!')
    else:
        # Index before fine-coarse interface
        p = np.where(spots > 0)[0][0]
        # Index before coarse-fine interface
        q = np.where(spots < 0)[0][0]
        print('p:', p)
        print('q:', q)
        NU = True

    if (otherFace):
        val = val - 1
        if (RL == 'R'): # Then everything shifts left.
            off = off + 1
            loBound = loBound - 0.5
            hiBound = hiBound - 0.5
        else:
            off = off - 1
            loBound = loBound + 0.5
            hiBound = hiBound + 0.5
    
    cellFaces = np.linspace(loBound, hiBound, num = orderStenc + 1)
    zeroLoc = np.where(cellFaces == 0)[0]
    if (len(zeroLoc) != 0):
        cellFaces = np.delete(cellFaces, zeroLoc[0])

    polys = np.shape(cellFaces)[0]
    polyStencSet = [[] for i in range(polys)]
    
    addZeros = np.zeros((polys, Ng), float)

    for i in range(polys):
        polyStencSet[i], n_c, n_f = GTT.CentGhost(omega, order, cellFaces[i])
    
    polyStencSet = np.asarray(polyStencSet)
    if ((Ng != 0) and (polys != 0)):
        polyStencSet = np.hstack([addZeros, polyStencSet, addZeros])
    
    IMat = np.eye(degFreed + 2 * Ng, degFreed + 2 * Ng) # np.eye(degFreed, degFreed)
    
    # YOU'RE GONNA NEED THESE TO RESTRICT FOR HIGHER EVEN ORDERS, TOO.
    
    
    polyMatU = IMat + 0
    
    
    mat = np.zeros((degFreed, degFreed + 2 * Ng), float) # np.zeros((degFreed, degFreed), float)
    faceOp = mat + 0
    
    for d in range(orderStenc + 1):
        s = int(off - d)
        
        derivMat = mat + 0
        np.fill_diagonal(derivMat[:, Ng:Ng+degFreed], stenc[d]) # np.fill_diagonal(derivMat, stenc[d])
        derivMat = np.roll(derivMat, -s, axis = 1) # np.roll(derivMat, s, axis = 0)
        
        polyMat = IMat + 0
        
        if (NU):
            if (s > 0):
                j = int(off - s)
                pAt = (p + Ng) % (degFreed + 2 * Ng) # p
                pLo = (p + Ng - 1) % (degFreed + 2 * Ng) # (p - 1) % degFreed
                qAt = (q - s + Ng + 1) % (degFreed + 2 * Ng) # (q - s + 1) % degFreed #(q + 1) % degFreed
                for i in range (s):
                    polyMat[pAt, :] = 0
                    polyMat[pAt, pLo:pLo+2] = 0.5
                    polyMat[qAt, :] = polyStencSet[j, :]
                    pAt = (pAt - 1) % (degFreed + 2 * Ng) # (pAt - 1) % degFreed
                    pLo = (pLo - 2) % (degFreed + 2 * Ng) # (pLo - 2) % degFreed
                    qAt = (qAt + 1) % (degFreed + 2 * Ng) # (qAt + 1) % degFreed
                    j = int(j + 1)

            if (s < 0):
                j = int(off) # - s - 1
                qAt = (q + Ng + 1) % (degFreed + 2 * Ng) # (q + 1) % degFreed
                qLo = (q + Ng + 1) % (degFreed + 2 * Ng) # (q + 1) % degFreed
                pAt = (p + Ng + 1) % (degFreed + 2 * Ng) # (p + 1) % degFreed
                for i in range(abs(s)):
                    polyMat[qAt, :] = 0
                    polyMat[qAt, qLo:qLo+2] = 0.5
                    polyMat[pAt, :] = polyStencSet[j, :]
                    qAt = (qAt + 1) % (degFreed + 2 * Ng) # (qAt + 1) % degFreed
                    qLo = (qLo + 2) % (degFreed + 2 * Ng) # (qLo + 2) % degFreed
                    pAt = (pAt + 1) % (degFreed + 2 * Ng) # (pAt + 1) % degFreed
                    j = int(j + 1) # - 1
        
        matThis = derivMat @ polyMat
        
        faceOp = faceOp + matThis
    
    finRow = np.zeros((1, halfDeg + 2 * Ng), float)
    finRowMaj = np.zeros((1, degFreed + 2 * Ng), float)
    
    
    faceOp1 = faceOp[:halfDeg, :halfDeg + 2 * Ng]
    faceOp2 = faceOp[-halfDeg:, -halfDeg - 2 * Ng:]
    
    if (RL == 'R'):
        if ((order > 1) and (val > 0)):
            finRow[0, :val] = stenc[-val:]
            finRowMaj[0, :val] = stenc[-val:]
        faceOp1 = np.concatenate((finRow, faceOp1), axis = 0)
        faceOp2 = np.concatenate((finRow, faceOp2), axis = 0)
        faceOp = np.concatenate((finRowMaj, faceOp), axis = 0)
    else:
        if ((order > 1) and (val > 0)):
            finRow[0, -val:] = stenc[:val]
            finRowMaj[0, -val:] = stenc[:val]
        faceOp1 = np.concatenate((faceOp1, finRow), axis = 0)
        faceOp2 = np.concatenate((faceOp2, finRow), axis = 0)
        faceOp = np.concatenate((faceOp, finRowMaj), axis = 0)
        
    return faceOp1, faceOp2, faceOp

