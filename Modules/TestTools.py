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
        calcCoefs, t = func(omega, waves, u_0, nt, const, CFL = CFL)
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
    spatOp = -c * derivMat
    u = spatOp @ u0
    return u

def CenterDiff(omega, t, u0, c, order):
    derivMat = OT.SpaceDeriv(omega, order, 'CD')
    spatOp = -c * derivMat
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

# This function runs a polynomial test on a derivative operator.

def DerivPolyTest(omega, DiffFunc, order, coefs = []):
    errorLoc = 'ERROR:\nTestTools:\nSpacePoly:\n'
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
    
    wavederiv = DiffFunc(omega, 0, waveform, -1, order)
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
    return wavederiv


def VectorNorm(v, normType = 'L2'):
    n = len(v)
    if (normType == 'max'):
            norm = max(v)
    else:
        if (normType == 'L1'):
            norm = sum(v) / n
        else:
            norm = np.sqrt(sum(v ** 2))
    return norm


def SolverAmpTheoretical(omega, RK, deriv, CFL):
    nh_max = omega.nh_max
    ks = np.arange((nh_max / 2) + 1)
    theta = (2 * np.pi * ks) / nh_max
    if (deriv == 'U'):
        print('Upwind', RK)
        x = CFL * (1 - np.exp(-1j * theta))
    else:
        x = 0.5 * CFL * (np.exp(1j * theta) - np.exp(-1j * theta))
    coefs = np.arange(RK + 1)[::-1]
    coefs = sp.special.factorial(coefs)**-1
    coefs[1::2] = -coefs[1::2]
    p = np.poly1d(coefs)
    amps = p(x)
    return ks, amps


def SolverSwitch(deriv, RK = 0):
    if (RK == 1):
        TimeIntegratorFunc = ST.ForwardEuler
    else:
        if (RK == 2):
            TimeIntegratorFunc = ST.MidpointMeth
        else:
            TimeIntegratorFunc = ST.RK4

    if (deriv == 'U'):
        # DiffMatFunc = OT.Upwind1D
        DiffFunc = Upwind#ST.Upwind
    else:
        # DiffMatFunc = OT.CenterDiff1D
        DiffFunc = CenterDiff#ST.CenterDiff
    return TimeIntegratorFunc, DiffFunc

# In[ ]:
