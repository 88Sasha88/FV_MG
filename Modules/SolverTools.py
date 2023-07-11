#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os.path
from scipy import *
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
from Modules import FFTTools as FFTT
from Modules import OperatorTools as OT

display(HTML("<style>pre { white-space: pre !important; }</style>"))
np.set_printoptions( linewidth = 1000)


# In[2]:

def FindDxDt(omega, CFL, c):
    dx = omega.dx
    dx_min = min(dx)
    if (np.shape(c) == ()):
        c_max = c
    else:
        c_max = max(np.diag(c))
    dt = CFL * dx_min / c_max
    return dx_min, dt

# You MUST pass op as an argument or creating a switch for the curl operator will be a pain in the ass!!!

def RungeKutta(omega, physics, waves, u0, nt, CFL, RK, op = [], left = True):
    errorMess = ''
    if (RK == 1):
        Scheme = ForwardEuler
    else:
        if (RK == 2):
            Scheme = MidpointMeth
        else:
            if (RK == 4):
                Scheme = RK4
            else:
                errorMess = str(RK) + ' is not a valid RK entry!'
    cMat = physics.cMat
    dx, dt = FindDxDt(omega, CFL, cMat)
    
#     if (order != 0):
#         if (func != TimePoly):
#             print('Spatial derivative method has been overridden in favor of TimePoly()!')
#         func = TimePoly # CHANGE THIS PROBABLY! AND FOR ALL THE OTHER RKS!
    
    u = u0.copy()
    t = 0
    for n in range(nt):
        u, t = Scheme(u, t, dt, op, waves, left)
    uCoefs = LA.inv(waves) @ u
    return uCoefs
    

def ForwardEuler(u0, t, dt, op, waves, left = True, charOp = []): #(omega, waves, u0, nt, const, CFL, func, order = 0):
    if (left):
        func = LeftMult
    else:
        func = FDeriv

    u = u0 + (dt * func(t, u0, op, waves, charOp))
    t = t + dt
#         if (func == TimePoly):
#             if (n == nt - 1):
#                 val = Operate(t, u, op) # func(omega, t, u, const, order + 1, deriv = False)
#                 if (order < 2):
#                     midstring = ' should be equal to '
#                 else:
#                     midstring = ' does not necessarily need to equal '
#                 print(str(u[0]) + midstring + str(val) + '.')
    return u, t


def CalcTime(omega, CFL, c, nt = 0, t = 0):
    errorLoc = 'ERROR:\nSolverTools:\nCalcTime:\n'
    errorMess = ''
    
    dx, dt = FindDxDt(omega, CFL, c)
    
    if (nt <= 0):
        nt = int(t / dt)
        t = nt * dt
        if (t <= 0):
            errorMess = 'There must be a greater-than-zero input for either nt or t!'
    else:
        if (t > 0):
            errorMess = 'There can only be an input for either nt or t!'
        if (errorMess != ''):
            sys.exit(errorLoc + errorMess)
        t = nt * dt
    return t, nt



def MidpointMeth(u, t, dt, op, waves, left = True, charOp = []): #(omega, waves, u0, nt, const, CFL, func, order = 0):
    if (left):
        func = LeftMult
    else:
        func = FDeriv
    
    k1 = func(t, u, op, waves, charOp)
    k2 = func(t + (dt / 2.), u + ((dt / 2.) * k1), op, waves, charOp)
    u = u + (dt * k2)
    t = t + dt
    return u, t

def RK4(u, t, dt, op, waves, left = True, charOp = []): # (omega, waves, u0, nt, const, CFL, func, order = 0):
    if (left):
        func = LeftMult
    else:
        func = FDeriv
    
    k1 = func(t, u, op, waves, charOp)
    k2 = func(t + (dt / 2.), u + ((dt / 2.) * k1), op, waves, charOp)
    k3 = func(t + (dt / 2.), u + ((dt / 2.) * k2), op, waves, charOp)
    k4 = func(t + dt, u + (dt * k3), op, waves, charOp)
    u = u + ((dt / 6.) * (k1 + (2. * k2) + (2. * k3) + k4))
    t = t + dt
    return u, t

# ----------------------------------------------------------------------------------------------------------------
# Function: TimePoly
# ----------------------------------------------------------------------------------------------------------------
# By: Sasha Curcic
#
# This function creates calculates a polynomial at the given time input. As a default, it calculates the
# derivative of a polynomial of arbitrary order using the power rule.
# ----------------------------------------------------------------------------------------------------------------
# Inputs: # INPUTS NEED TO BE CHANGED!!!
#
# matrica                 array/list              Operator or list of operators to be reconstructed in blocks
# (var)                   int                     Number of blocks if matrica entered as array
# (diag)                  boolean                 Switch paramater for diagonal or antidiagonal blocks
# ----------------------------------------------------------------------------------------------------------------
# Outputs:
#
# poly                    real                    Value of polynomial at time t
# ----------------------------------------------------------------------------------------------------------------

def TimePoly(omega, t, u, const, order, deriv = True):
    # add error checker for negative order or zero
    poly = 0
    for n in range(0, order):
        if (deriv):
            poly = poly + ((n + 1) * (t**n))
        else:
            poly = poly + (t**n)
    return poly



def LeftMult(t, u0, op, waves, charOp):
    if (charOp == []):
        u = op @ u0
    else:
        charOpInv = LA.inv(charOp)
        v0 = charOpInv @ u0
        v = op @ v0
        u = charOp @ v
    return u

def FDeriv(t, u0, op, waves, charOp):
    fcoefs = FFTT.FourierCoefs(waves, u0)
    fcoefsderiv = op @ fcoefs
    u = waves @ fcoefsderiv
    return u

















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

def ExactTimeDerivOp(omega, waves, cMat):
    print('You are using ExactSpatDeriv in SolverTools module!')
    nullspace = OT.FindNullspace(omega, waves)

    
    SpatOp = OT.ExactSpatDerivOp(omega)
    FTOp = nullspace @ OT.FourierTransOp(waves @ nullspace)
#     FCoefs = nullspace @ FFTT.FourierCoefs(waves @ nullspace, u0)
#     u = -cMat @ waves @ SpatOp @ FCoefs
    ETDerivOp = -cMat @ waves @ SpatOp @ FTop
    return ETDerivOp

# In[ ]:








def CenterDiff(omega, t, u0, c, order):
    print('You are using CenterDiff in SolverTools module!')
    derivMat = OT.SpaceDeriv(omega, order, 'CD')
    spatOp = -c @ derivMat
