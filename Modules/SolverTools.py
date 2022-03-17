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

def RungeKutta(omega, physics, waves, u0, nt, CFL, RK, op = []):
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
        u, t = Scheme(u, t, dt, op)
    uCoefs = LA.inv(waves) @ u
    return uCoefs
    

def ForwardEuler(u0, t, dt, op): #(omega, waves, u0, nt, const, CFL, func, order = 0):

    u = u0 + (dt * LeftMult(t, u0, op))
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
        nt = int(t / nt)
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

# NOT IN USE!

def Upwind(omega, t, u0, const, order):
    degFreed = omega.degFreed
    dx = omega.h
    B = dx - np.roll(dx, 1)
    B[B > 0] = 0.5
    # B[B > 0] = 0
    B[B < 0] = 2. / 3.
    C = B + 0 # np.roll(B, -1)
    # D = np.roll(B, -1)
#     D[D != 2. / 3.] = 0
#     C = C - D
    B[B < 2. / 3.] = 1.
    C[C == 0] = 1.
    D = C + 0
    D[D != 0.5] = 0
    print('')
    print('Start:')
    print(B)
    print(dx)
    print('')
    print(C)
    print(np.roll(dx, 1))
    print('')
    print(D)
    print(np.roll(dx, 2))
    print('')
    f = -(const / dx) @ ((B * u0) - (C * np.roll(u0, 1)) - (D * np.roll(u0, 2)))
    return f

# NOT IN USE!

# def CenterDiff(omega, t, u0, const, order):
#     degFreed = omega.degFreed
#     dx = omega.h
    
#     # A is the main diagonal; C is the subdiagonal; G is the sub-subdiagonal; E is the superdiagonal; H is the super-superdiagonal.
#     A = dx - np.roll(dx, 1)
#     B = A + 0
#     F = np.roll(A, -1)
#     F[F > 0] = 1. / 3.
#     F[F != 1. / 3.] = 0
#     A[A < 0] = -1. / 3.
#     A[A != -1. / 3.] = 0
#     A = A - F
#     B[B > 0] = 0.5
#     B[B < 0] = 2. / 3.
#     C = -B
#     B[B < 2. / 3.] = 1.
#     C[C == 0] = -1.
#     D = C + 0
#     D[D != -0.5] = 0
#     E = -C
#     E[E == 0.5] = 4. /3.
#     E[E == 2. / 3.] = 0.5
#     G = C + 0
#     G[G != -0.5] = 0
#     H = E + 0
#     H[H != 0.5] = 0
    
#     print('')
#     print('Start:')
#     print(H)
#     print(np.roll(dx, -2))
#     print('')
#     print(E)
#     print(np.roll(dx, -1))
#     print('')
#     print(A)
#     print(dx)
#     print('')
#     print(C)
#     print(np.roll(dx, 1))
#     print('')
#     print(G)
#     print(np.roll(dx, 2))
#     print('')
    
#     f = -(const / (2 * dx)) @ ((H * np.roll(dx, -2)) + (E * np.roll(dx, -1)) + (A * dx) + (C * np.roll(dx, 1)) + (G * np.roll(dx, 2)))
#     return f

def MidpointMeth(u, t, dt, op): #(omega, waves, u0, nt, const, CFL, func, order = 0):
    k1 = LeftMult(t, u, op)
    k2 = LeftMult(t + (dt / 2.), u + ((dt / 2.) * k1), op)
    u = u + (dt * k2)
    t = t + dt
    return u, t

def RK4(u, t, dt, op): # (omega, waves, u0, nt, const, CFL, func, order = 0):
    k1 = LeftMult(t, u, op)
    k2 = LeftMult(t + (dt / 2.), u + ((dt / 2.) * k1), op)
    k3 = LeftMult(t + (dt / 2.), u + ((dt / 2.) * k2), op)
    k4 = LeftMult(t + dt, u + (dt * k3), op)
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



def LeftMult(t, u0, op):
    u = op @ u0
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
