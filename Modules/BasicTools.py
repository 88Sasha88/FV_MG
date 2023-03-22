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
    
# ----------------------------------------------------------------------------------------------------------------
# Function: CheckSize
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# This function checks that the matrix input into it matches the number given. It assumes that the matrix size
# will match along each dimension. If all is well, the output string is ''. If not, the output is a message
# describing the problem.
# ----------------------------------------------------------------------------------------------------------------
# Input:
#
# n                       real                    Ostenisble size of matrica
# matrica                 real                    Matrix of arbitrary rank, each dimension with ostensible size n
# (nName)                 string                  Name of variable n for use in output message
# (matricaName)           string                  Name of variable matrica for use in output message
# ----------------------------------------------------------------------------------------------------------------
# Output:
#
# message                 string                  Feedback message
# ----------------------------------------------------------------------------------------------------------------

def CheckSize(n, matrica, nName = 'n', matricaName = 'matrica'):
    dim = size(shape(matrica))
    message = ''
    for i in range(dim):
        if (n != shape(matrica)[i]):
            message = '%s does not match size of %s!' %(nName, matricaName)
    return message

# ----------------------------------------------------------------------------------------------------------------
# Function: CheckNumber
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# This function ensures that n is a base-two value greater than 1. If all is well, the output string is ''. If
# not, the output is a message describing the problem.
# ----------------------------------------------------------------------------------------------------------------
# Input:
#
# n                       real                    Ostenisble base-2 value
# (nName)                 string                  Name of variable n for use in output message
# ----------------------------------------------------------------------------------------------------------------
# Output:
#
# message                 string                  Feedback message
# ----------------------------------------------------------------------------------------------------------------

def CheckNumber(n, nName = 'n'):
    message = ''
    check = np.log(n) / np.log(2)
    if ((check < 1) or (check != int(check))):
        message = '%s must be a base-two integer greater than one!' %nName
    return message

# ----------------------------------------------------------------------------------------------------------------
# Function: CheckDiag
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# THIS FUNCTION APPEARS NOT TO BE IN USE?
# ----------------------------------------------------------------------------------------------------------------
# Input:
#
# matrica                 real                    Ostensible square matrix to be evaluated for diagonality
# (matricaName)           string                  Name of variable matrica for use in output message
# ----------------------------------------------------------------------------------------------------------------
# Output:
#
# message                 string                  Feedback message
# ----------------------------------------------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------------------------------------------
# Function: CheckBounds
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# This function checks if a given matrix is rank-2 and diagonal. If all is well, the output string is ''. If not,
# the output is a message describing the problem.
# ----------------------------------------------------------------------------------------------------------------
# Input:
#
# matrica                 real                    Ostensible square matrix to be evaluated for diagonality
# (matricaName)           string                  Name of variable matrica for use in output message
# ----------------------------------------------------------------------------------------------------------------
# Output:
#
# message                 string                  Feedback message
# ----------------------------------------------------------------------------------------------------------------

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

# ----------------------------------------------------------------------------------------------------------------
# Class: Grid
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# This class stores all the attributes of an AMR grid. The base grid is created upon instantiation of an object
# from inputs nh, the degrees of freedom on the base grid, and alias, a numerical value set to 1 as the default in
# case the user wishes to plot examples of aliasing. Additional patches are added using the AddPatch function,
# which takes in arguments refRatio and cell, which are both overloaded with 1 and [], respectively, for the
# automatic call at the instantiation of the base level. Data stored in the elements of all list-type attributes,
# except strings, correspond to each level from lowest to highest refinement, respectively. All other attributes
# are assumed to apply to the current Grid object (at its highest refinement,) unless otherwise stated.
# ----------------------------------------------------------------------------------------------------------------
# Subclasses: Patch
# ----------------------------------------------------------------------------------------------------------------
# Functions: AddPatch
# ----------------------------------------------------------------------------------------------------------------
# Inputs:
#
# nh                      int                     Base grid degrees of freedom (base-2 value)
# (alias)                 int                     Scaling factor by which to multiply number of string outputs if
#                                                     user chooses to plot aliasing waves (base-2 value)
# ----------------------------------------------------------------------------------------------------------------
# Attributes:
#
# alias                   int                     Scaling factor by which to multiply number of string outputs if
#                                                     user chooses to plot aliasing waves (base-2 value)
# bounds                  int                     x values of edges of most recent patch
# cells                   list                    Lists of locations of cells to be refined using indexing for
#                                                     uniform grid of previous levels, recursively (integer
#                                                     values)
# degFreed                int                     Number of degrees of freedom on current AMR grid
# degFreeds               list                    List of AMR degrees of freedom corresponding to each level
# h                       array                   Array of length intervals corresponding to each cell on current
#                                                     AMR grid
# levels                  int                     Total number of of refinements
# nh                      list                    List of uniform degrees of freedom corresponding to each level
# nh_max                  int                     Number of degrees of freedom on current uniform grid
# nh_min                  int                     Number of degrees of freedom on base-level grid
# patches                 list                    List of Patch objects created for each level, including base
#                                                     level
# refRatios               list                    Refinement ratios at each level, including the base level
# strings                 string                  List of TeX-ready string headers for each mode to be created
#                                                     with the Grid object
# xCell                   array                   Values of x locations averaged over each cell
# xNode                   array                   Values of x locations at each node, including right edge
# xPatch                  list                    List of of lists of patch x values at the nodes corresponding to
#                                                     each level
# y (RECODE Y OUT OF THIS!)
# ----------------------------------------------------------------------------------------------------------------

class Grid:
#     patches = []
#     xNode = []
#     xCell = []
#     xPatches = []
#     cells = []
#     levels = 0
#     nh_max = 1
#     refRatios = []
#     degFreeds = []
#     strings = []
#     nh = []
    def __init__(self, nh, alias = 1):
        errorLoc = 'ERROR:\nBasicTools:\nGrid:\n__init__:\n'
        self.alias = alias
        # Add error check for alias! It must be base-2!
        self.patches = []
        self.xNode = []
        self.xCell = []
        self.h = []
        self.dx = []
        self.xPatches = []
        self.cells = []
        self.levels = 0
        self.degFreed = 0
        self.refRatios = []
        self.degFreeds = []
        self.strings = []
        self.nh = []
        self.nh_min = nh
        self.nh_max = nh
        
        self.AddPatch()
        errorMess = CheckNumber(self.nh_min, nName = 'nh_min')
        if (errorMess != ''):
            sys.exit(errorLoc + errorMess)
    
# ----------------------------------------------------------------------------------------------------------------
# Class: Patch (< Class: Grid)
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# This class generates and assimilates a number of attributes needed for the creation of a patch on an AMR grid
# using the cell locations (by index, with respect to the level on top of which the new patch(es) is/are being
# created) of the cells to be refined. Namely, these attributes are the x locations of the nodes at the edges of
# each patch; the original input of the cell locations; and the x node values within the new patch(es), boundaries
# included.
# ----------------------------------------------------------------------------------------------------------------
# Inputs:
#
# nh                      int                     Degrees of freedom for uniform grid of previous level (base-2
#                                                     value)
# refRatio                int                     Refinement ratio between previous and current level (base-2
#                                                     value)
# cell                    list                    Locations of cells to be refined using indexing for uniform grid
#                                                     of previous level (integer values)
# ----------------------------------------------------------------------------------------------------------------
# Attributes:
#
# bounds                  list                    x values at patch edges
# cell                    list                    Locations of cells to be refined using indexing for uniform grid
#                                                     of previous level (integer values)
# xPatch                  list                    Lists of arrays of x values at nodes of new patches being
#                                                     created, bounds included
# ----------------------------------------------------------------------------------------------------------------

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
            self.bounds = bounds
            self.cell = cell
            self.xPatch = xPatch
    
# ----------------------------------------------------------------------------------------------------------------
# Function: AddPatch (< Class: Grid)
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# This function modifies the attributes of the current Grid object. Its inputs are the refinement ratio and the
# locations of the cells to be refined, with respect to the level preceeding. The inputs are overloaded with
# values of 1 and the nullset, respectively, so that the attributes can be instantiated for the special case of
# the zeroth level. It first redefines the current uniform degrees of freedom attribute on the Grid object and
# then creates a new Patch instance. Using the attributes from the current Grid object and the new Patch object,
# respectively, it checks that conflicts do not exist between any of the input arguments and the attributes of the
# current Grid object and then proceeds to update all the attributes in the Grid to accomodate the new level.
# ----------------------------------------------------------------------------------------------------------------
# Input:
#
# self                    Grid                    Current Grid object
# (refRatio)              int                     Refinement ratio for new patch to be made (base-2 value)
# (cell)                  list                    Name of variable matrica for use in output message
# ----------------------------------------------------------------------------------------------------------------
    
    def AddPatch(self, refRatio = 1, cell = []):
#         print('what the hell', self.nh)
        self.nh_max = self.nh_max * refRatio
#         print('even earlier', self.nh)
        patch0 = self.Patch(self.nh_max, refRatio, cell)

        # ERROR CHECKS:

        errorLoc = 'ERROR:\nBasicTools:\nGrid:\nAddPatch:\n'
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
        kRange = self.nh_max
        kRange = int(self.alias * kRange)
        for k in range(kRange):
            if (k % 2 == 0):
                if (k == 0):
                    name = '$a_{0}$'
                else:
                    number = str(k)
                    name = '$a_{' + number + '}$cos' + number + '$' + '\\' + 'pi$'
            else:
                number1 = str(k)
                number2 = str(k + 1)
                name = '$a_{' + number1 + '}$sin' + number2 + '$' + '\\' + 'pi$'
            strings = np.append(strings, name)
        n = len(self.xNode)
        self.degFreed = n - 1
        self.xPatches = xPatchesFiller
        self.cells = cellsFiller
        self.refRatio = refRatio
        self.y = np.zeros(n, float)
        self.xCell = 0.5 * (self.xNode[:-1] + self.xNode[1:])
        self.h = self.xNode[1:] - self.xNode[:-1]
        self.dx = self.xCell[1:] - self.xCell[:-1]
        self.bounds = patch0.bounds
        self.degFreeds.append(self.degFreed)
#         print('before', self.nh)
        self.nh.append(self.nh_max)
#         print('after', self.nh)
        self.strings = strings

# ----------------------------------------------------------------------------------------------------------------
# Class: PhysProps
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# This class stores all the attributes of an AMR grid. The base grid is created upon instantiation of an object
# from inputs nh, the degrees of freedom on the base grid, and alias, a numerical value set to 1 as the default in
# case the user wishes to plot examples of aliasing. Additional patches are added using the AddPatch function,
# which takes in arguments refRatio and cell, which are both overloaded with 1 and [], respectively, for the
# automatic call at the instantiation of the base level. Data stored in the elements of all list-type attributes,
# except strings, correspond to each level from lowest to highest refinement, respectively. All other attributes
# are assumed to apply to the current Grid object (at its highest refinement,) unless otherwise stated.
# ----------------------------------------------------------------------------------------------------------------
# Inputs:
#
# omega                   Grid                    Grid object for AMR
# epsilons                list                    Relative electric permittivity of materials in order from left
# mus                     list                    Relative magnetic permeability of materials in order from left
# (locs)                  list                    Locations of material jumps
# (L)                     float                   Physical length in meters of entire area simulated
# ----------------------------------------------------------------------------------------------------------------
# Attributes:
#
# epsilons_r              list                    Unitless relative electric permittivity of materials in order
#                                                     from left
# mus_r                   list                    Unitless relative magnetic permeability of materials in order
#                                                     from left
# locs                    list                    Locations of material jumps
# L                       float                   Physical length of entire area simulated
# epsilons                list                    Electric permittivity in SI of materials in order from left
# mus                     list                    Magnetic permeability in SI of materials in order from left
# matInd                  int                     Index of material boundary location
# ----------------------------------------------------------------------------------------------------------------

class PhysProps:
    def __init__(self, omega, epsilons, mus, locs = [], L = 1):
        # Add error check for locs outside of range of L!
        # Add error check for right number of mus, epsilons, and locs!
        self.omega = omega
        self.mus_r = mus
        self.epsilons_r = epsilons
        self.locs = locs
        self.L = L
        
        x = omega.xNode
        
        epsilon_0 = 8.85418782e-12
        mu_0 = 1.25663706e-6
        self.epsilons = list(epsilon_0 * np.asarray(epsilons))
        self.mus = list(mu_0 * np.asarray(mus))
        
        locs = np.asarray(sorted(set(np.append(locs, [1.]))))
        iters = len(locs)
        
        x = omega.xNode
        degFreed = omega.degFreed
        
        cVec = np.ones(degFreed, float)
        cs = np.ones(iters, float)# 1. / (L * np.sqrt(epsilons * mus))
        
        indexOld = 0
        for i in range(iters):
            distance = locs[i] - x
            minDist = min(abs(distance))
            indexNew = np.where(distance == minDist)[0][0]
            c = 1. / (L * np.sqrt(epsilons[i] * mus[i]))
            cVec[indexOld:indexNew] = c
            cs[i] = c
            indexOld = indexNew

        self.cVec = cVec.transpose()
        self.cMat = np.diag(cVec)
        self.cs = cs
        if (locs is []):
            self.matInd = -1
        else:
            self.matInd = max(np.where(x[:-1] < locs[0])[0])

