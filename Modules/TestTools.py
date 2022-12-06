#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os.path
import scipy as sp
from scipy import *
from scipy import integrate as integrate
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
from Modules import BasicTools as BT
from Modules import WaveTools as WT
from Modules import PlotTools as PT
from Modules import FFTTools as FFTT
from Modules import OperatorTools as OT
from Modules import GridTransferTools as GTT
from Modules import SolverTools as ST


# This function calculates either the absolute or percent error between the theoretical and actual solutions.

# In[2]:


def CalcError(omega, theoretical, actual, errorType = 'absolute', tol = 1e-14):
    # Check size of theoretical and actual.
    nh = omega.nh_max
    error = abs(actual - theoretical)
    if (errorType == 'relative'):
        error = abs(error / (theoretical + tol))
    ks = np.linspace(0, (nh / 2) - 1, num = nh)
    return ks, error


# This function calculates the error given by an arbitrary time solver as compared to that given through Fourier analysis.

# In[5]:


def NormVersusCFL(func, omega, waves, u_0, const, CFL_0, nt_0, normType = 'max', errorType = 'absolute', plot = False, CFL_f = 0.1):
    # Check size of waves and u_0, and check that nt_0 isn't negative?
    errorLoc = 'ERROR:\nTestTools:\nNormVersusCFL:\n'
    nt = nt_0
    CFL = CFL_0
    nh = omega.nh_max
    dx = omega.dx[0]
    FCoefs = FFTT.FourierCoefs(omega, waves, u_0)
    norms = []
    CFLs = []
    while (CFL > CFL_f):
        calcCoefs, t = func(omega, waves, u_0, nt, const, CFL = CFL) # THIS IS GONNA NEED TO CHANGE!!!
        if (CFL == CFL_0):
            t_0 = t
        propCoefs = FFTT.PropogateFCoefs(omega, FCoefs, const, t)
        ks, error = CalcError(omega, propCoefs, calcCoefs, errorType = errorType)
        if (plot):
            if (nt % 10 == 0):
                allCoefs = PT.Load(FCoefs, propCoefs, calcCoefs)
                title = 'CFL = ' + str(CFL)
                PT.PlotMixedWave(omega, waves, allCoefs, rescale = [2, 3], title = title, labels = [r'$u_{0} (x)$', r'Reference $u_{0} (x - c t)$', r'Solver $u_{0} (x - c t)$'])
        if (normType == 'max'):
            norm = max(error)
        else:
            norm = sum(error) / nh
        norms.append(norm)
        CFLs.append(CFL)
        nt = nt + 1
        CFL = (CFL * const * t_0) / ((CFL * dx) + (const * t_0))
        if (not np.isclose(t, t_0, atol = 1e-15, rtol = 0)):
            errorMess = 't does not match t_0!\nt_0 = ' + str(t_0) + '\nt = ' + str(t)
            sys.exit(errorLoc + errorMess)
    return norms, CFLs


# This function calculates either the amplitude error between the theoretical and actual solutions.

# In[6]:


def AmpError(omega, theoreticalIn, actualIn, tol = 1e-20, printOut = False):
    # Check size of theoretical and actual.
    nh = omega.nh_max
    numKs = int((nh / 2) + 1)
    actual = np.round(actualIn, 15)
    theoretical = np.round(theoreticalIn, 15)
    error = np.zeros(numKs, float)
    error[0] = 1 - np.sqrt(((actual[0] ** 2) + tol) / ((theoretical[0]**2) + tol))
    error[::-1][0] = 1 - np.sqrt(((actual[::-1][0] ** 2) + tol) / ((theoretical[::-1][0]**2) + tol))
    error[1:-1] = 1 - np.sqrt(((actual[1::2][:-1]**2) + (actual[::2][1:]**2) + tol) / ((theoretical[1::2][:-1]**2) + (theoretical[::2][1:]**2) + tol))
    ks = np.arange(numKs)
    if (printOut):
        print('Actual:')
        print(actual)
        print('Theoretical:')
        print(theoretical)
        print('Error:')
        print(error)
        print('')
    return ks, error

def Upwind(omega, t, u0, c, order):
    derivMat = OT.SpaceDeriv(omega, order, 'U')
    print(derivMat)
    spatOp = -c @ derivMat
    u = spatOp @ u0
    return u

def CenterDiff(omega, t, u0, c, order):
    derivMat = OT.SpaceDeriv(omega, order, 'CD')
    spatOp = -c @ derivMat
    u = spatOp @ u0
    return u

# This function checks that your polynomial interpolations produce outputs up to the appropriate order of accuracy.

def TestPoly(order, x_0, const = 2, tol = 1e-15):
    
    # Create vector of cell bounds.
    bounds = GTT.BoundVals(order, x_0)
    
    # Find intervals of cell bounds.
    h = bounds[:-1] - bounds[1:]
    
    # Create stencil.
    polyInterp = GTT.GhostCellStencil(order, x_0)
    
    # Iterate through monomials up to appropriate order of accuracy to test stencil.
    for k in range(order + 2):
        coefs = np.zeros(k + 1, float)
        coefs[0] = const
        p = np.poly1d(coefs)
        P = np.polyint(p)
        v = (P(bounds[:-1]) - P(bounds[1:])) / h
        print('Order ' + str(k) + ':')

        theor = P(x_0) / x_0
        act = v.transpose() @ polyInterp
        print(theor, act)
        print('')
        if (k < order + 1):
            assert(np.isclose(act, theor, rtol = 0, atol = tol))
    return
    
# ----------------------------------------------------------------------------------------------------------------
# Function: DerivPolyTest
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# This function runs a dest on a differential operator which uses some order of polynomial interpolation to
# determine ghost-cell values at the coarse-fine and/or fine-coarse interfaces. The screen outputs for the actual
# and theoretical values should match if the derivative operator is good to the given order. The accuracy of the
# operation is limited to the order of both the polynomial interpolation used as well as the finite derivative
# operator itself.
# ----------------------------------------------------------------------------------------------------------------
# Inputs:
#
# omega                   Grid                    Object containing AMR grid attributes
# deriv                   str                     Switch which modulates finite difference function to be used
# order                   int                     Order of polynomial function to be tested
# (coefs)                 list                    Coefficients of polynomial function to be tested (defaulted to
#                                                     all ones
# ----------------------------------------------------------------------------------------------------------------

def DerivPolyTest(omega, diff, order, coefs = []):
    errorLoc = 'ERROR:\nTestTools:\nSpacePoly:\n'
    degFreed = omega.degFreed
    if (coefs == []):
        coefs = np.ones(order + 1, float)
    else:
        errorMess = BT.CheckSize(order, coefs, nName = 'order', matricaName = 'coefs')
        if (errorMess != ''):
            sys.exit(errorLoc + errorMess)
    nh_max = omega.nh_max
    waves = WT.MakeWaves(omega)
    x = omega.xCell
    P = np.poly1d(coefs)
    waveform = P(x)
    p = np.polyder(P)
    
    const = -np.eye(degFreed)
    derivOp = OT.SpaceDeriv(omega, order, diff)# DiffFunc(omega, 0, waveform, const, order)
    wavederiv = derivOp @ waveform
    print('x:')
    print(x)
    print('')
    print('Polynomial Function:')
    print('p(x) = ', P)
    print('p(x) =\n', waveform)
    print('')
    print('Polynomial Derivative:')
    print('dp(x)/dx =', p)
    print('Theoretical:')
    print('dp(x)/dx =\n', p(x))
    print('Actual:')
    print('dp(x)/dx =\n', wavederiv)
    print('')
    return


def VectorNorm(v, normType = 'L2'):
    n = len(v)
    if (normType == 'max'):
        shape = np.shape(v)
        dim = len(shape)
        if (dim == 1):
            norm = max(v)
        else:
            size = shape[1]
            norm = np.zeros(size)
            for i in range(size):
                norm[i] = max(v[:, i])
    else:
        if (normType == 'L1'):
            norm = sum(v) / n
        else:
            norm = np.sqrt(sum(v ** 2, axis = 0))
    return norm


def SolverAmpTheoretical(omega, RK, deriv, CFL):
    nh_max = omega.nh_max
    ks = np.arange((nh_max / 2) + 1)
    theta = (2 * np.pi * ks) / nh_max
    if (deriv == 'U'):
        print('Upwind', RK)
        x = CFL * (np.exp(1j * theta) - 1) # CFL * (1 - np.exp(-1j * theta))
    else:
        x = 0.5 * CFL * (np.exp(1j * theta) - np.exp(-1j * theta))
    coefs = np.arange(RK + 1)[::-1]
    coefs = sp.special.factorial(coefs)**-1
    # coefs[1::2] = -coefs[1::2]
    p = np.poly1d(coefs)
    print(p)
    amps = p(x)
    return ks, amps

# ----------------------------------------------------------------------------------------------------------------
# Function: SolverSwitch
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# This function streamlines switching between different combinations of RK and finite-difference operators.
# ----------------------------------------------------------------------------------------------------------------
# Inputs:
#
# deriv                   str                     Switch which modulates finite difference function to be used
# RK                      int                     Switch which modulates RK scheme to be used
# ----------------------------------------------------------------------------------------------------------------
# Outputs:
#
# TimeIntFunc             function                Time integration RK scheme to be used
# DiffFunc                function                Finite difference spatial derivative function to be used (Perhaps change this to a direct calculation of the operator.
# ----------------------------------------------------------------------------------------------------------------

def SolverSwitch(deriv, RK): # Changed RK to non-overloaded variable.
    errorLoc = 'ERROR:\nTestTools:\nSolverSwitch:\n'
    errorMess = ''
    if (RK == 1):
        TimeIntFunc = ST.ForwardEuler
    else:
        if (RK == 2):
            TimeIntFunc = ST.MidpointMeth
        else:
            if (RK == 4):
                TimeIntFunc = ST.RK4
            else:
                errorMess = 'RK solvers are only available to order 1, 2, or 4!'

    if (deriv == 'U'):
        # DiffMatFunc = OT.Upwind1D
        DiffFunc = Upwind#ST.Upwind
    else:
        if (deriv == 'CD'):
            # DiffMatFunc = OT.CenterDiff1D
            DiffFunc = CenterDiff#ST.CenterDiff
        else:
            errorMess = 'Finite difference derivative entry only valid as \'U\' (Upwind) or \'CD\' (Center-Difference)!'
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
    return TimeIntFunc, DiffFunc

# ----------------------------------------------------------------------------------------------------------------
# Function: ExactSpatOp
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

def ExactSpatOp(omega):
    print('You are using ExactSpatOp in TestTools module!')
    nh_max = omega.nh_max
    omegaF = BT.Grid(nh_max)
#     waves = WT.MakeWaves(omega)
#     wavesF = WT.MakeWaves(omegaF)
#     nullspace = OT.FindNullspace(omega, waves)
    subsuper = np.linspace(0.5, nh_max, num = 2 * nh_max)
    subsuper[::2] = 0
    Op = np.zeros((nh_max, nh_max), float)
    np.fill_diagonal(Op[1:], subsuper[:])
    np.fill_diagonal(Op[:, 1:], -subsuper)
    Op = 2 * np.pi * Op
    SpatOp = Op # wavesF @ Op # @ nullspace
    return SpatOp

# ----------------------------------------------------------------------------------------------------------------
# Function: ExactSpatDeriv
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

def ExactSpatDeriv(omega, t, u0, c, order):
    print('You are using ExactSpatDeriv in TestTools module!')
#     nh_max = omega.nh_max
    waves = WT.MakeWaves(omega)
    nullspace = OT.FindNullspace(omega, waves)
#     subsuper = np.linspace(0.5, nh_max, num = 2 * nh_max)
#     subsuper[::2] = 0
#     Op = np.zeros((nh_max, nh_max), float)
#     np.fill_diagonal(Op[1:], subsuper[:])
#     np.fill_diagonal(Op[:, 1:], -subsuper)
#     print(Op)
#     Op = -2 * np.pi * c * Op
#     SpatOp = waves @ Op
#     FCoefs = FFTT.FourierCoefs(omega, waves @ nullspace, u0)
#     u = SpatOp @ FCoefs
    
    SpatOp = ExactSpatOp(omega)  
    FCoefs = nullspace @ FFTT.FourierCoefs(waves @ nullspace, u0)
    u = -c * waves @ SpatOp @ FCoefs
    return u

# In[ ]:
