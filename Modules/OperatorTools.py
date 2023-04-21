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

# NOT IN USE!

def Upwind1D(omega):
    n = omega.degFreed
    hs = omega.h
    B = hs - np.roll(hs, 1)
    B[B > 0] = 0.5
    B[B < 0] = 2. / 3.
    C = -np.roll(B, -1)
    B[B < 2. / 3.] = 1.
    C[C == 0] = -1.
    D = np.roll(C, -1)
    D[D != -0.5] = 0
    Deriv = np.zeros((n, n), float)
    np.fill_diagonal(Deriv, B)
    np.fill_diagonal(Deriv[1:], C)
    np.fill_diagonal(Deriv[2:], D)
    Deriv[0, n - 1] = C[::-1][0]
    Deriv[0, n - 2] = D[::-1][1]
    Deriv[1, n - 1] = D[::-1][0]
    hMat = StepMatrix(omega)
    Deriv = hMat @ Deriv
    return Deriv

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

# NOT IN USE!

def CenterDiff1D(omega):
    # A is the main diagonal; C is the subdiagonal; G is the sub-subdiagonal; E is the superdiagonal; H is the super-superdiagonal.
    n = omega.degFreed
    hs = omega.h
    A = hs - np.roll(hs, 1)
    B = A + 0
    F = np.roll(A, -1)
    F[F > 0] = 1. / 3.
    F[F != 1. / 3.] = 0
    A[A < 0] = -1. / 3.
    A[A != -1. / 3.] = 0
    A = A - F
    B[B > 0] = 0.5
    B[B < 0] = 2. / 3.
    C = -np.roll(B, -1)
    B[B < 2. / 3.] = 1.
    C[C == 0] = -1.
    D = np.roll(C, -1)
    D[D != -0.5] = 0
    E = -C
    E[E == 0.5] = 4. /3.
    E[E == 2. / 3.] = 0.5
    G = np.roll(C, -1)
    G[G != -0.5] = 0
    H = E + 0
    H[H != 0.5] = 0
    Deriv = np.zeros((n, n), float)
    np.fill_diagonal(Deriv, A)
    np.fill_diagonal(Deriv[1:], C)
    np.fill_diagonal(Deriv[:, 1:], E)
    np.fill_diagonal(Deriv[2:], G)
    np.fill_diagonal(Deriv[:, 2:], H)
    Deriv[0, n - 1] = C[::-1][0]
    Deriv[0, n - 2] = G[::-1][1]
    Deriv[1, n - 1] = G[::-1][0]

    Deriv[n - 1, 0] = E[::-1][0]
    Deriv[n - 2, 0] = H[::-1][1]
    Deriv[n - 1, 1] = H[::-1][0]
    hMat = 0.5 * StepMatrix(omega)
#     print(Deriv)
    Deriv = hMat @ Deriv
    return Deriv


# In[ ]:

def Curl(omega, order, diff):
    derivOp = SpaceDeriv(omega, order, diff)
    
    return

def Grad(omega, order, diff):
    
    return

# # DO NOT CHANGE THIS ONE!!!
# def SpaceDeriv0(omega, order, diff):
    
#     # Extract info from omega.
#     hs = omega.h
#     degFreed = omega.degFreed
    
#     # Create empty matrizes to fill later.
#     blockMat = np.zeros((degFreed, degFreed), float)
#     zeroMat = np.zeros(degFreed, float)
    
#     # Interpolate ghost cells.
#     ghostStencL = GTT.GhostCellStencil(order, -0.5)
#     ghostStencR = GTT.GhostCellStencil(order, 0.5)
    
#     # Create common constituents of row pieces.
#     rightCell = zeroMat + 0
#     leftCell = zeroMat + 0
#     leftCell[-1] = 1
#     ghostL = zeroMat + 0
#     ghostR = zeroMat + 0
#     ghostL[:order+1] = ghostStencL
#     ghostR[:order+1] = ghostStencR
#     nrollL = int(np.ceil((order + 1) / 3.))
#     nrollR = int(order - nrollL)
#     ghostL = np.roll(ghostL, -nrollL)
#     ghostR = np.roll(ghostR, -nrollR)
    
#     # Define distinct row pieces between upwind and center difference.
#     if (diff == 'U'):
#         rightCell[0] = 1
        
#         cf1v2 = rightCell + 0
#         cf1v1 = leftCell + 0
        
#         fc1v2 = rightCell + 0
#         fc1v1 = leftCell + 0
        
#         hMat = StepMatrix(omega)
#     else:
#         rightCell[1] = 1
        
#         cf1v2 = zeroMat + 0
#         cf1v2[1] = 0.5
#         cf1v2[2] = 0.5
#         cf1v1 = leftCell + 0
        
#         fc1v2 = ghostR
#         fc1v1 = leftCell + 0
        
#         hMat = 0.5 * StepMatrix(omega)
    
#     # Define common row pieces between upwind and center difference.
#     cf2v2 = rightCell + 0
#     cf2v1 = ghostL
    
#     fc2v2 = rightCell + 0
#     fc2v1 = zeroMat + 0
#     fc2v1[-1] = 0.5
#     fc2v1[-2] = 0.5
    
#     # Define rows.
#     default = rightCell - leftCell
#     cf1 = cf1v2 - cf1v1
#     cf2 = cf2v2 - cf2v1
#     fc1 = fc1v2 - fc1v1
#     fc2 = fc2v2 - fc2v1
    
#     # Create vector containing intergrid boundary locations.
#     spots = np.roll(hs, -1) - hs
    
#     # Fill matrix.
#     for k in range(degFreed):
#         if (np.roll(spots, 1 - k)[0] > 0):
#             row = fc2
#         else:
#             if (np.roll(spots, -k)[0] < 0): # This one's okay.
#                 row = cf1
                
#             else:
#                 if (np.roll(spots, 1 - k)[0] < 0): # This one's okay.
#                     row = cf2 #fc1
                    
#                 else:
#                     if (spots[k] > 0): # This one's okay.
#                         row = fc1 #cf2
#                     else:
#                         row = default
#         blockMat[k, :] = np.roll(row, k)
# #     print('1/2dx =')
# #     print(hMat)
# #     print('')
# #     print('difference matrix =')
# #     print(blockMat)
# #     print('')
#     blockMat = hMat @ blockMat
# #     print('derivative matrix =')
# #     print(blockMat)
# #     print('')
#     return blockMat

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


def SpaceDeriv1(omega, order, diff):
    
    # Extract info from omega.
    hs = omega.h
    degFreed = omega.degFreed
    
    # Create empty matrizes to fill later.
    blockMat = np.zeros((degFreed, degFreed), float)
    zeroMat = np.zeros(degFreed, float)
    
    # Interpolate ghost cells.
    ghostStencL, n_c, n_f = GTT.GhostCellStencil(order, -0.5)
    ghostStencR, n_c, n_f = GTT.GhostCellStencil(order, 0.5)
    
    # Create common constituents of row pieces.
    rightCell = zeroMat + 0
    leftCell = zeroMat + 0
    ghostL = zeroMat + 0
    ghostR = zeroMat + 0
    ghostL[:order+1] = ghostStencL
    ghostR[:order+1] = ghostStencR
    nrollL = int(np.ceil((order + 1) / 3.))
    nrollR = int(order - nrollL)
    ghostL = np.roll(ghostL, -nrollL)
    ghostR = np.roll(ghostR, -nrollR)
    
    # Define distinct row pieces between upwind and center difference.
    if (diff == 'U'):
        rightCell[0] = 1
        leftCell[-1] = 1
        
        cf1v2 = rightCell + 0
        # cf1v1 = leftCell + 0
        cf2v1 = ghostL
        
        fc1v2 = rightCell + 0
        # fc1v1 = leftCell + 0
        fc2v1 = zeroMat + 0
        fc2v1[-1] = 0.5
        fc2v1[-2] = 0.5
        fc2v2 = rightCell + 0
        
        hMat = StepMatrix(omega)
    else:
        if (diff == 'D'):
            rightCell[1] = 1
            leftCell[0] = 1
            
            cf1v2 = zeroMat + 0
            cf1v2[1] = 0.5
            cf1v2[2] = 0.5
            cf2v1 = leftCell + 0
            
            fc1v2 = ghostR
            fc2v1 = leftCell + 0
            
            hMat = StepMatrix(omega)
        else:
            rightCell[1] = 1
            leftCell[-1] = 1

            cf1v2 = zeroMat + 0
            cf1v2[1] = 0.5
            cf1v2[2] = 0.5
            # cf1v1 = leftCell + 0
            cf2v1 = ghostL

            fc1v2 = ghostR
            # fc1v1 = leftCell + 0
            # fc2v2 = rightCell + 0
            fc2v1 = zeroMat + 0
            fc2v1[-1] = 0.5
            fc2v1[-2] = 0.5

            hMat = 0.5 * StepMatrix(omega)
    
    # Define common row pieces between upwind and center difference.
    cf1v1 = leftCell + 0 # New add.
    cf2v2 = rightCell + 0
    # cf2v1 = ghostL
    
    fc1v1 = leftCell + 0 # New add.
    fc2v2 = rightCell + 0
#     fc2v1 = zeroMat + 0
#     fc2v1[-1] = 0.5
#     fc2v1[-2] = 0.5
    
    # Define rows.
    default = rightCell - leftCell
    cf1 = cf1v2 - cf1v1
    cf2 = cf2v2 - cf2v1
    fc1 = fc1v2 - fc1v1
    fc2 = fc2v2 - fc2v1
    
    # Create vector containing intergrid boundary locations.
    spots = np.roll(hs, -1) - hs
    
    # Fill matrix.
    for k in range(degFreed):
        if (np.roll(spots, 1 - k)[0] > 0):
            row = fc2
        else:
            if (np.roll(spots, -k)[0] < 0): # This one's okay.
                row = cf1
                
            else:
                if (np.roll(spots, 1 - k)[0] < 0): # This one's okay.
                    row = cf2 #fc1
                    
                else:
                    if (spots[k] > 0): # This one's okay.
                        row = fc1 #cf2
                    else:
                        row = default
        blockMat[k, :] = np.roll(row, k)
#     print('1/2dx =')
#     print(hMat)
#     print('')
#     print('difference matrix =')
#     print(blockMat)
#     print('')
    blockMat = hMat @ blockMat
#     print('derivative matrix =')
#     print(blockMat)
#     print('')
    return blockMat



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

def SpaceDeriv(omega, order, diff, matInd0 = -1):
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
                errorMess = 'Invalid entry for variable diff. Must be \'C\', \'U\', \'D\' \'CD\', \'UD\', or \'DD\'.'
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

    IMat = np.eye(degFreed, degFreed)
    
    # YOU'RE GONNA NEED THESE TO RESTRICT FOR HIGHER EVEN ORDERS, TOO.
    
    
    polyMatU = IMat + 0
    
    
    mat = np.zeros((degFreed, degFreed), float)
    derivOp = mat + 0
    
    
    # CHANGE MADE HERE!
    
    if (matInd0 >= 0):
        if ((order >= matInd0) or (order > degFreed - matInd0 - 2)):
            errorMess = 'order is too high for given patch boundary and material boundary locations!'
        else:
            materialOverwrite = True
            matIndVec = [matInd0, degFreed - 1]
    else:
        materialOverwrite = False
        matIndVec = []
    
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
    
    n_c_max = abs(off) # THIS MIGHT NOT BE RIGHT FOR UD AND/OR DD!!!!
    for i in range(orderStenc):
        polyStencSet[i], n_c, n_f = GTT.CentGhost(omega, order, cellFaces[i])
        for matInd in (matIndVec):
            if ((abs(p - matInd) < n_c) and (cellFaces[i] > 0) and (n_c > n_c_max)):
                n_c_max = n_c
    
    polyStencSet = np.asarray(polyStencSet)

    
    np.set_printoptions(suppress=True)
    # END CHANGE MADE!
    
    
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
                pHi = (p + 1) % degFreed
                qAt = (q - s + 1) % degFreed #(q + 1) % degFreed
                for i in range(s):
                    polyMat[pAt, :] = 0
                    polyMat[pAt, pLow:pHi] = 0.5
                    polyMat[qAt, :] = polyStencSet[j, :]
                    pAt = (pAt - 1) % degFreed
                    pLow = (pLow - 2) % degFreed
                    pHi = (pHi - 2) % degFreed
                    qAt = (qAt + 1) % degFreed
                    j = int(j + 1)
                
                # CHANGE MADE HERE!
            
                if (materialOverwrite):
                    for matInd in matIndVec:
                        if ((matInd <= p) and (p - matInd <= s)): # RIGHT HERE IS WHERE YOU SHOULD BE LOOKING FOR THE PROBLEM!!!
                            for i in range(matInd-s+1, matInd+2): # (matInd+s+1, matInd+(2*s)+2):
                                j = i % degFreed
                                polyMat[j, :] = GTT.CentGhostMaterial(omega, order, matInd, i+s, s)
                        else:
                            if ((matInd <= q) and (q - matInd <= s)):
                                for i in range(matInd-s+1, q+1): # (matInd+s+1, q+(2*s)+1):
                                    j = i % degFreed
                                    polyMat[j, :] = GTT.CentGhostMaterial(omega, order, matInd, i+s, s)
                            else:
                                for i in range(matInd-s+1, matInd+1): # (matInd+s+1, matInd+(2*s)+1):
                                    j = i % degFreed
                                    polyMat[j, :] = GTT.CentGhostMaterial(omega, order, matInd, i+s, s)

            # END CHANGE MADE!

            if (s < 0):
                j = int(off) # - s - 1
                qAt = (q + 1) % degFreed
                qLow = (q + 1) % degFreed
                qHi = (q + 3) % degFreed
                pAt = (p + 1) % degFreed#p
                for i in range(abs(s)):
                    polyMat[qAt, :] = 0
                    polyMat[qAt, qLow:qHi] = 0.5
                    polyMat[pAt, :] = polyStencSet[j, :]
                    qAt = (qAt + 1) % degFreed
                    qLow = (qLow + 2) % degFreed
                    qHi = (qHi + 2) % degFreed
                    pAt = (pAt + 1) % degFreed
                    j = int(j + 1) # - 1
                    
             
                # CHANGE MADE HERE!
            
                if (materialOverwrite):
                    for matInd in matIndVec:
                        if ((matInd >= p) and (matInd - p <= abs(s))): # This is where you changed an inequality.  # RIGHT HERE IS WHERE YOU SHOULD BE LOOKING FOR THE PROBLEM!!!
                            for i in range(p+1, matInd-s+1): # (p+(2*s)+1, matInd+s+1):
                                j = i % degFreed
                                polyMat[j, :] = GTT.CentGhostMaterial(omega, order, matInd, i+s, s)
                        else:
                            for i in range(matInd+1, matInd-s+1): # (matInd+(2*s)+1, matInd+s+1):
                                j = i % degFreed
                                polyMat[j, :] = GTT.CentGhostMaterial(omega, order, matInd, i+s, s)
                            if ((matInd < p) and (p - matInd <= n_c_max)): # This is where you changed an inequality. You also changed abs(s) to n_c_max.
                                for i in range(matInd-s+1, p-s+1): # (matInd+s+1, p+s+1):
                                    j = i % degFreed
                                    polyMat[j, :] = GTT.CentGhostMaterial(omega, order, matInd, i+s, s, revBounds = True)

            # END CHANGE MADE!
        
        
        matThis = derivMat @ polyMat
        
        derivOp = derivOp + matThis
    
    hMat = StepMatrix(omega)
    
    derivOp = hMat @ derivOp
        
    return derivOp


# def SpaceDeriv2(omega, order, diff, matInd0 = -1):
#     errorLoc = 'ERROR:\nOperatorTools:\nMakeSpaceDeriv:\n'
#     errorMess = ''
#     if (diff == 'C' or diff == 'CD'):
#         stenc = CDStencil(order)
#         if (order % 2 == 0):
#             orderStenc = order
#         else:
#             orderStenc = int(order + 1)
#         off = int(orderStenc / 2)
#         loBound = -off / 2.
#         hiBound = off / 2.
#     else:
#         orderStenc = order
#         if (order % 2 == 0):
#             orderStenc = int(order + 1)
#         else:
#             orderStenc = order
#         off = ((orderStenc + 1) / 2)
#         if (diff == 'U' or diff == 'UD'):
#             stenc = UDStencil(order)
#             loBound = -off / 2.
#             hiBound = (off - 1.) / 2.
#         else:
#             if (diff == 'D' or diff == 'DD'):
#                 stenc = DDStencil(order)
#                 off = int(off - 1)
#                 loBound = -off / 2.
#                 hiBound = (off + 1.) / 2.
#             else:
#                 errorMess = 'Invalid entry for variable diff. Must be \'C\', \'U\', \'D\' \'CD\', \'UD\', or \'DD\'.'
#     if (errorMess != ''):
#         sys.exit(errorLoc + errorMess)
    
# #     stenc = np.ones(orderStenc + 1)
    
#     degFreed = omega.degFreed
#     hs = omega.h
    
#     spots = np.roll(hs, -1) - hs
#     # Index before fine-coarse interface
#     p = np.where(spots > 0)[0][0]
#     # Index before coarse-fine interface
#     q = np.where(spots < 0)[0][0]
    
#     polyStencSet = [[] for i in range(orderStenc)]
#     cellFaces = np.linspace(loBound, hiBound, num = orderStenc + 1)
#     zeroLoc = np.where(cellFaces == 0)[0][0]
#     cellFaces = np.delete(cellFaces, zeroLoc)
    
    

#     IMat = np.eye(degFreed, degFreed)
    
#     # YOU'RE GONNA NEED THESE TO RESTRICT FOR HIGHER EVEN ORDERS, TOO.
    
    
    
#     polyMatU = IMat + 0
    
    
#     mat = np.zeros((degFreed, degFreed), float)
#     derivOp = mat + 0
    
#     # CHANGE MADE HERE!
    
    
#     if (matInd0 >= 0):
#         if ((order >= matInd0) or (order > degFreed - matInd0 - 2)):
#             errorMess = 'order is too high for given patch boundary and material boundary locations!'
#         else:
#             materialOverwrite = True
#             matIndVec = [matInd0, degFreed - 1]
#     else:
#         materialOverwrite = False
#         matIndVec = []
    
#     if (errorMess != ''):
#         sys.exit(errorLoc + errorMess)
    
#     n_c_max = abs(off)
#     for i in range(orderStenc):
#         polyStencSet[i], n_c, n_f = GTT.CentGhost(omega, order, cellFaces[i])
#         for matInd in (matIndVec):
#             if ((abs(p - matInd) < n_c) and (cellFaces[i] > 0) and (n_c > n_c_max)):
#                 n_c_max = n_c
    
#     polyStencSet = np.asarray(polyStencSet)
    
#     # END CHANGE MADE!
    
#     for d in range(orderStenc + 1):
#         s = int(off - d)
        
#         derivMat = mat + 0
#         np.fill_diagonal(derivMat, stenc[d])
#         derivMat = np.roll(derivMat, s, axis = 0)
        
#         polyMat = IMat + 0

#         if (s > 0):
#             j = int(off - s)
#             pAt = p
#             pLow = (p - 1) % degFreed
#             pHi = (p + 1) % degFreed
#             qAt = (q - s + 1) % degFreed #(q + 1) % degFreed
#             for i in range(s):
#                 polyMat[pAt, :] = 0
#                 polyMat[pAt, pLow:pHi] = 0.5
#                 polyMat[qAt, :] = polyStencSet[j, :]
#                 pAt = (pAt - 1) % degFreed
#                 pLow = (pLow - 2) % degFreed
#                 pHi = (pHi - 2) % degFreed
#                 qAt = (qAt + 1) % degFreed
#                 j = int(j + 1)
                
#             # CHANGE MADE HERE!
            
#             if (materialOverwrite):
#                 for matInd in matIndVec:
#                     if ((matInd <= p) and (p - matInd <= s)):
#                         for i in range(matInd-s+1, matInd+2): # (matInd+s+1, matInd+(2*s)+2):
#                             j = i % degFreed
#                             polyMat[j, :] = GTT.CentGhostMaterial(omega, order, matInd, i+s, s)
#                     else:
#                         if ((matInd <= q) and (q - matInd <= s)):
#                             for i in range(matInd-s+1, q+1): # (matInd+s+1, q+(2*s)+1):
#                                 j = i % degFreed
#                                 polyMat[j, :] = GTT.CentGhostMaterial(omega, order, matInd, i+s, s)
#                         else:
#                             for i in range(matInd-s+1, matInd+1): # (matInd+s+1, matInd+(2*s)+1):
#                                 j = i % degFreed
#                                 polyMat[j, :] = GTT.CentGhostMaterial(omega, order, matInd, i+s, s)
                    
#             # END CHANGE MADE!
        
#         if (s < 0):
#             j = int(off) # - s - 1
#             qAt = (q + 1) % degFreed
#             qLow = (q + 1) % degFreed
#             qHi = (q + 3) % degFreed
#             pAt = (p + 1) % degFreed#p
#             for i in range(abs(s)):
#                 polyMat[qAt, :] = 0
#                 polyMat[qAt, qLow:qHi] = 0.5
#                 polyMat[pAt, :] = polyStencSet[j, :]
#                 qAt = (qAt + 1) % degFreed
#                 qLow = (qLow + 2) % degFreed
#                 qHi = (qHi + 2) % degFreed
#                 pAt = (pAt + 1) % degFreed
#                 j = int(j + 1) # - 1
            
#             # CHANGE MADE HERE!
            
#             if (materialOverwrite):
#                 for matInd in matIndVec:
#                     if ((matInd >= p) and (matInd - p <= abs(s))):
#                         for i in range(p+1, matInd-s+1): # (p+(2*s)+1, matInd+s+1):
#                             j = i % degFreed
#                             polyMat[j, :] = GTT.CentGhostMaterial(omega, order, matInd, i+s, s)
#                     else:
#                         for i in range(matInd+1, matInd-s+1): # (matInd+(2*s)+1, matInd+s+1):
#                             j = i % degFreed
#                             polyMat[j, :] = GTT.CentGhostMaterial(omega, order, matInd, i+s, s)
#                         if ((matInd < p) and (p - matInd <= n_c_max)):
#                             for i in range(matInd-s+1, p-s+1): # (matInd+s+1, p+s+1):
#                                 j = i % degFreed
#                                 polyMat[j, :] = GTT.CentGhostMaterial(omega, order, matInd, i+s, s, revBounds = True)

#                 # END CHANGE MADE!
        
#         matThis = derivMat @ polyMat

        
#         derivOp = derivOp + matThis
    
#     hMat = StepMatrix(omega)
    
#     derivOp = hMat @ derivOp
        
#     return derivOp