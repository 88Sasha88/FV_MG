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


# This function ensures that `n` is an appropriate base-two value.

# In[3]:


def CheckNumber(n, nName = 'n'):
    message = ''
    check = np.log(n) / np.log(2)
    if ((check < 1) or (check != int(check))):
        message = '%s must be a base-two integer greater than one!' %nName
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


# This class creates an object which encompasses all the information one needs from an AMR grid. This includes `xNode`, an array containing all tick locations along the $x$ axis; `xCell`, an array of all cell center locations along the $x$ axis; `xPatches`, a list of all the smaller grids at their respective levels; `levels`, the total number of refinement levels; `cells`, a list of all the grid locations at their respective locations using the indexing of the previous level; 'nh_min', the step number at the coarsest level; 'nh_max', the step number at the finest level; `h`; an array of step-sizes at their respective locations; and `y`, an array of zeros corresponding to the locations of all of the tick marks.

# In[6]:


class Grid:
    patches = []
    xNode = []
    xPatches = []
    cells = []
    levels = 0
    nh_max = 1
    refRatios = []
    degFreed = []
    strings = []
    nh = []
    def __init__(self, nh):
        errorLoc = 'ERROR:\nBasicTools:\nGrid:\n__init__:\n'
        self.nh_min = nh
        self.nh_max = nh
        self.AddCell()
        errorMess = CheckNumber(self.nh_min, nName = 'nh_min')
        if (errorMess != ''):
            sys.exit(errorLoc + errorMess)
    class Patch:
        def __init__(self, nh, refRatio, cell):
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
            self.xPatch = xPatch
            self.cell = cell
            self.bounds = bounds
    def AddCell(self, refRatio = 1, cell = []):
        print('what the hell', self.nh)
        self.nh_max = self.nh_max * refRatio
        print('even earlier', self.nh)
        patch0 = self.Patch(self.nh_max, refRatio, cell)

        # ERROR CHECKS:

        errorLoc = 'ERROR:\nBasicTools:\nGrid:\nAddCell:\n'
        if (cell != []):
            self.levels = self.levels + 1
            errorMess = CheckNumber(refRatio, nName = 'refRatio')
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
            if (self.levels > 0):
                errorMess = 'You must enter a list of cell locations!'
                sys.exit(errorLoc + errorMess)
        if (cell != sorted(cell)):
            errorMess = 'cell must be in number order!'
            sys.exit(errorLoc + errorMess)
        if (len(cell) != len(set(cell))):
            errorMess = 'cell contains repeats!'
            sys.exit(errorLoc + errorMess)

        # END OF ERROR CHECKS
        
        if (refRatio == 1):
            self.refRatios.append(self.nh_min)
        else:
            self.refRatios.append(refRatio)
        self.patches.append(patch0)
        xPatchesFiller = [[] for i in range(self.levels + 1)]
        cellsFiller = [[] for i in range(self.levels + 1)]
        for xPatch in patch0.xPatch:
            self.xNode = np.asarray(sorted(set(np.append(self.xNode, xPatch))))
        for i in range(self.levels + 1):
            if (i == self.levels):
                xPatchesFiller[i] = patch0.xPatch
                cellsFiller[i] = patch0.cell
            else:
                xPatchesFiller[i] = self.xPatches[i]
                cellsFiller[i] = self.cells[i]
        strings = []
        for k in range(self.nh_max):
            if (k % 2 == 0):
                if (k == 0):
                    name = '$' '\\' + 'frac{a_{0}}{2}$'
                else:
                    number = str(k)
                    name = '$a_{' + number + '}$cos' + number + '$' + '\\' + 'pi x$'
            else:
                number1 = str(k)
                number2 = str(k + 1)
                name = '$a_{' + number1 + '}$sin' + number2 + '$' + '\\' + 'pi x$'
            strings = np.append(strings, name)
        n = len(self.xNode)
        self.xPatches = xPatchesFiller
        self.cells = cellsFiller
        self.refRatio = refRatio
        self.y = np.zeros(n, float)
        self.h = self.xNode[1:n] - self.xNode[0:n - 1]
        self.xCell = 0.5 * (self.xNode[0:n - 1] + self.xNode[1:n])
        self.bounds = patch0.bounds
        self.degFreed.append(n - 1)
        blah = self.nh_max
        print('before', self.nh)
        self.nh.append(self.nh_max)
        print('after', self.nh)
        self.strings = strings


# In[ ]:




