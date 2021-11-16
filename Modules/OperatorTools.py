#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os.path
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

def StepMatrix(omega):
    h = omega.h
    n = omega.degFreed
    hs = np.zeros((n, n), float)
    np.fill_diagonal(hs, h)
    hMat = LA.inv(hs)
    return hMat

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
    print(Deriv)
    Deriv = hMat @ Deriv
    return Deriv


# In[ ]:


def SpaceDeriv(omega, order, diff):
    
    # Extract info from omega.
    hs = omega.h
    degFreed = omega.degFreed
    
    # Create empty matrizes to fill later.
    blockMat = np.zeros((degFreed, degFreed), float)
    zeroMat = np.zeros(degFreed, float)
    
    # Interpolate ghost cells.
    ghostStencL = GTT.GhostCellStencil(order, -0.5)
    ghostStencR = GTT.GhostCellStencil(order, 0.5)
    
    # Create common constituents of row pieces.
    rightCell = zeroMat + 0
    leftCell = zeroMat + 0
    leftCell[-1] = 1
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
        
        cf1v2 = rightCell + 0
        cf1v1 = leftCell + 0
        
        fc1v2 = rightCell + 0
        fc1v1 = leftCell + 0
        
        hMat = StepMatrix(omega)
    else:
        rightCell[1] = 1
        
        cf1v2 = zeroMat + 0
        cf1v2[1] = 0.5
        cf1v2[2] = 0.5
        cf1v1 = leftCell + 0
        
        fc1v2 = ghostR
        fc1v1 = leftCell + 0
        
        hMat = 0.5 * StepMatrix(omega)
    
    # Define common row pieces between upwind and center difference.
    cf2v2 = rightCell + 0
    cf2v1 = ghostL
    
    fc2v2 = rightCell + 0
    fc2v1 = zeroMat + 0
    fc2v1[-1] = 0.5
    fc2v1[-2] = 0.5
    
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







