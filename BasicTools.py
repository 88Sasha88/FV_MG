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


# This function checks to make sure that sizes match up appropriately.

# In[2]:


def CheckSize(n, matrica, nName = 'n', matricaName = 'matrica'):
    dim = size(shape(matrica))
    message = ''
    for i in range(dim):
        if (n != shape(matrica)[i]):
            message = '%s does not match size of %s!' %(nName, matricaName)
    return message


# This function ensures that $n^{h}$ is an appropriate base-2 value.

# In[3]:


def CheckNumber(nh, nhName = 'nh'):
    check = nh
    message = ''
    while (check % 2 == 0):
        check = check / 2
    if (check != 1):
        message = '%s must be a base-2 integer!' %nhName
    return message


# This function checks if a given matrix is diagonal.

# In[4]:


def CheckDiag(matrica, matricaName = 'matrica'):
    message = ''
    if (np.size(np.shape(matrica)) != 2):
        message = '%s must be a rank-2 array!' %matricaName
    if (np.shape(matrica)[0] != np.shape(matrica)[1]):
        message = '%s must be a square array!' %matricaName
    i, j = np.nonzero(matrica)
    if (np.all(i == j)):
        message = ''
    else:
        message = '%s is not diagonal!' %matricaName
    return message


# This function checks if a given matrix is diagonal.

# In[5]:


def CheckBounds(nh_min, loops, bounds):
    problem = CheckNumber(nh_min, showMess = False)
    if (bounds[0][0] != 0):
        problem = 2
    for i in range(loops):
        if (len(bounds[i]) != 2):
            problem = 3
        if (bounds[i][0] > bounds[i][1]):
            problem = 4
        if ((bounds[i][0] > nh_min * (2**(i + 1))) or (bounds[i][1] > nh_min * (2**(i + 1)))):
            problem = 5
    return problem


# This function outputs an $x$ array and a $y$ array of size $n^{h}$ + 1 of the locations of the tick marks.

# In[6]:


def MakeXY(xBounds):
    nh_min = xBounds[0][1]
    loops = len(xBounds)
    nh_max = nh_min * (2**(loops - 1))
#     problem = CheckBounds(nh_min, loops, xBounds)
#     if (problem == 1):
#         sys.exit('ERROR:\nBasicTools:\nMakeXY:\nnh_min must be base_2 integer!')
#     if (problem == 2):
#         sys.exit('ERROR:\nBasicTools:\nMakeXY:\nFirst value in first boundary pair must be 0!')
#     if (problem == 3):
#         sys.exit('ERROR:\nBasicTools:\nMakeXY:\nAll elements in xBounds must be of length 2!')
#     if (problem == 4):
#         sys.exit('ERROR:\nBasicTools:\nMakeXY:\nLower-bound values in xBounds greater than upper bounds!')
#     if (problem == 5):
#         sys.exit('ERROR:\nBasicTools:\nMakeXY:\nValues in xBounds must correspond to the grid spacing of their respective level!')
    x = []
    for i in range(loops):
        h = (2**-i) / nh_min
        n = xBounds[i][1] - xBounds[i][0] + 1
        xMin = h * xBounds[i][0]
        xMax = h * xBounds[i][1]
        if (i > 0):
            n = n - 2
            xMin = xMin + h
            xMax = xMax - h
        xPiece = np.linspace(xMin, xMax, num = n)
        x = sorted(set(np.append(x, xPiece)))
    x = np.asarray(x)
    n_max = np.shape(x)[0]
    y = np.zeros(n_max, float)
    return x, y


class Grid:
    patches = list([])
    xNode = []
    xPatches = [[]]
    levels = 0
    cells = list([])
    def __init__(self, nh):
        errorLoc = 'ERROR:\nBasicTools:\nGrid:\n__init__:\n'
        self.AddCell(nh)
        self.nh_min = nh
        errorMess = CheckNumber(self.nh_min, nhName = 'nh_min')
        if (errorMess != ''):
            sys.exit(errorLoc + errorMess)
    class Patch:
        def __init__(self, nh, refRatio, cell):
            nh = nh * refRatio
            h = 1. / nh
            if (cell == []):
                regions = 1
                cellPieces = [[0, nh - 1]]
                xPatch = [[]]
                cell = [0]
            else:
                spots = len(cell)
                if (spots == 1):
                    regions = spots
                    cellPieces = [[cell[0], cell[0]]]
                else:
                    upper = np.asarray(cell[1:spots])
                    lower = np.asarray(cell[0:spots - 1])
                    cutWhere = np.append(np.where(lower != upper - 1)[0], spots - 1)
                    regions = len(cutWhere)
                    cellPieces = [[] for j in range(regions)]
                    first = 0
                    for j in range(regions):
                        last = cutWhere[j]
                        cellPieces[j] = [cell[first], cell[last]]
                        first = last + 1
                xPatch = [[] for j in range(regions)]
            bounds = [[] for j in range(regions)]
            for j in range(regions):
                rangeVal = cellPieces[j][1] - cellPieces[j][0] + 1
                n = rangeVal * refRatio
                xMin = h * refRatio * cellPieces[j][0]
                xMax = h * refRatio * (cellPieces[j][1] + 1)
                xPatch[j] = np.linspace(xMin, xMax, num = n + 1)
                bounds[j] = np.asarray([xMin, xMax])
            self.nh = nh
            self.xPatch = xPatch
            self.cell = cell
            self.bounds = bounds
        def getNh(self):
            return self.nh
        def getRefRatio(self):
            return self.refRatio
        def getCell(self):
            return self.cell
        def getXPatch(self):
            return self.xPatch
    def AddCell(self, nh, refRatio = 1, cell = []):
        self.levels = self.levels + 1
        patch0 = self.Patch(nh, refRatio, cell)

        # ERROR CHECKS:

        errorLoc = 'ERROR:\nBasicTools:\nGrid:\nAddCell:\n'
        if (cell != []):
            errorMess = CheckNumber(self.refRatio, nhName = 'refRatio')
            if (errorMess != ''):
                sys.exit(errorLoc + errorMess)
            for patchBound in patch0.bounds:
                both = False
                j = 0
                while ((not both) and (j < len(self.bounds))):
                    bound = self.bounds[j]
                    aboveLower = np.all(patchBound >= bound[0])
                    belowUpper = np.all(patchBound <= bound[1])
                    both = np.all([aboveLower, belowUpper])
                    j = j + 1
                if (not both):
                    errorMess = 'cell values out of range of previous patch!'
                    sys.exit(errorLoc + errorMess)
        else:
            refRatio = 1
        if (cell != sorted(cell)):
            errorMess = 'cell must be in number order!'
            sys.exit(errorLoc + errorMess)
        if (len(cell) != len(set(cell))):
            errorMess = 'cell contains repeats!'
            sys.exit(errorLoc + errorMess)

        # END OF ERROR CHECKS

        self.patches.append(patch0)
        xPatchesFiller = [[] for i in range(self.levels)]
        cellsFiller = [[] for i in range(self.levels)]
        for xPatch in patch0.xPatch:
            self.xNode = np.asarray(sorted(set(np.append(self.xNode, xPatch))))
        for i in range(self.levels):
            if (i == self.levels - 1):
                xPatchesFiller[i] = patch0.xPatch
                cellsFiller[i] = patch0.cell
            else:
                xPatchesFiller[i] = self.xPatches[i]
                cellsFiller[i] = self.cells[i]
        n = len(self.xNode)
        self.xPatches = xPatchesFiller
        self.cells = cellsFiller
        self.nh_max = patch0.nh
        self.refRatio = refRatio
        self.y = np.zeros(n, float)
        self.xCell = 0.5 * (self.xNode[0:n - 1] + self.xNode[1:n])
        self.bounds = patch0.bounds


# In[ ]:




