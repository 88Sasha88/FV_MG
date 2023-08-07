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

def RungeKutta(omega, physics, u0, CFL, nt, RK, order, diff, func):

    u = u0.copy()
    
    cMat = physics.cMat
    dx, dt = FindDxDt(omega, CFL, cMat)
    
    waves = WT.MakeWaves(omega)
    nullspace = OT.FindNullspace(omega, waves)
    waves = waves @ nullspace
    
    if (func == WaveEqRHS):
        waves = OT.Block(waves, var = 2)
    
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
            
    if (errorMess != ''):
        sys.exit(errorMess)
    
    np.set_printoptions(suppress = True)
    
    t = 0
    for n in range(nt):
        u, t = Scheme(omega, physics, u0, t, dt, order, diff, func)
    uCoefs = LA.inv(waves) @ u
    
    return uCoefs

# def RungeKutta(omega, physics, waves, u0, nt, CFL, RK, op = [], left = True):
#     errorMess = ''
#     if (RK == 1):
#         Scheme = ForwardEuler
#     else:
#         if (RK == 2):
#             Scheme = MidpointMeth
#         else:
#             if (RK == 4):
#                 Scheme = RK4
#             else:
#                 errorMess = str(RK) + ' is not a valid RK entry!'
#     cMat = physics.cMat
#     dx, dt = FindDxDt(omega, CFL, cMat)
    
# #     if (order != 0):
# #         if (func != TimePoly):
# #             print('Spatial derivative method has been overridden in favor of TimePoly()!')
# #         func = TimePoly # CHANGE THIS PROBABLY! AND FOR ALL THE OTHER RKS!
    
#     u = u0.copy()
#     t = 0
#     for n in range(nt):
#         u, t = Scheme(u, t, dt, op, waves, left)
#     uCoefs = LA.inv(waves) @ u
#     return uCoefs


def ForwardEuler(omega, physics, u0, t0, dt, order, diff, func): #(omega, waves, u0, nt, const, CFL, func, order = 0):

    u = u0 + (dt * func(omega, physics, u0, t0, order, diff))
    t = t0 + dt
#         if (func == TimePoly):
#             if (n == nt - 1):
#                 val = Operate(t, u, op) # func(omega, t, u, const, order + 1, deriv = False)
#                 if (order < 2):
#                     midstring = ' should be equal to '
#                 else:
#                     midstring = ' does not necessarily need to equal '
#                 print(str(u[0]) + midstring + str(val) + '.')
    return u, t


# def ForwardEuler(u0, t, dt, op, waves, left = True): #(omega, waves, u0, nt, const, CFL, func, order = 0):
#     if (left):
#         func = LeftMult
#     else:
#         func = FDeriv

#     u = u0 + (dt * func(t, u0, op, waves))
#     t = t + dt
# #         if (func == TimePoly):
# #             if (n == nt - 1):
# #                 val = Operate(t, u, op) # func(omega, t, u, const, order + 1, deriv = False)
# #                 if (order < 2):
# #                     midstring = ' should be equal to '
# #                 else:
# #                     midstring = ' does not necessarily need to equal '
# #                 print(str(u[0]) + midstring + str(val) + '.')
#     return u, t


def CalcTime(omega, CFL, c, nt = 0, t = 0):
    errorLoc = 'ERROR:\nSolverTools:\nCalcTime:\n'
    errorMess = ''
    
    dx, dt = FindDxDt(omega, CFL, c)
    print('dt:', dt)
    
    if (nt <= 0):
        print('This is what\'s happening.')
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

def MidpointMeth(omega, physics, u0, t0, dt, order, diff, func): #(omega, waves, u0, nt, const, CFL, func, order = 0):
    
    k1 = func(omega, physics, u0, t0, order, diff)
    k2 = func(omega, physics, u0 + ((dt / 2.) * k1), t0 + (dt / 2.), order, diff)
    u = u0 + (dt * k2)
    t = t0 + dt
    
    return u, t

# def MidpointMeth(u, t, dt, op, waves, left = True): #(omega, waves, u0, nt, const, CFL, func, order = 0):
#     if (left):
#         func = LeftMult
#     else:
#         func = FDeriv
    
#     k1 = func(t, u, op, waves)
#     k2 = func(t + (dt / 2.), u + ((dt / 2.) * k1), op, waves)
#     u = u + (dt * k2)
#     t = t + dt
#     return u, t


def RK4(omega, physics, u0, t0, dt, order, diff, func): # (omega, waves, u0, nt, const, CFL, func, order = 0):
    
    k1 = func(omega, physics, u0, t0, order, diff)
    k2 = func(omega, physics, u0 + ((dt / 2.) * k1), t0 + (dt / 2.), order, diff)
    k3 = func(omega, physics, u0 + ((dt / 2.) * k2), t0 + (dt / 2.), order, diff)
    k4 = func(omega, physics, u0 + (dt * k3), t0 + dt, order, diff)
    u = u0 + ((dt / 6.) * (k1 + (2. * k2) + (2. * k3) + k4))
    t = t0 + dt
    
    return u, t

# def RK4(u, t, dt, op, waves, left = True): # (omega, waves, u0, nt, const, CFL, func, order = 0):
#     if (left):
#         func = LeftMult
#     else:
#         func = FDeriv
    
#     k1 = func(t, u, op, waves)
#     k2 = func(t + (dt / 2.), u + ((dt / 2.) * k1), op, waves)
#     k3 = func(t + (dt / 2.), u + ((dt / 2.) * k2), op, waves)
#     k4 = func(t + dt, u + (dt * k3), op, waves)
#     u = u + ((dt / 6.) * (k1 + (2. * k2) + (2. * k3) + k4))
#     t = t + dt
#     return u, t

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



def LeftMult(t, u0, op, waves):
    u = op @ u0
    return u

def FDeriv(t, u0, op, waves):
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






## Calculate the RHS for E,B in Maxwell's equations using 5th-order upwind
def WaveEqRHS(omega, physics, u0, t, orderIn, diff):
    
    degFreed = omega.degFreed
    cs = physics.cVec
    matInd = physics.matInd
    
    c1 = cs[0]
    c2 = cs[-1]
    
    E = u0[:degFreed]
    B = u0[degFreed:]
    
    hMat = OT.StepMatrix(omega)
    
    # Fill in ghost cells for left, right domain - need 3 for up5
    if ((diff == 'CD') or (diff == 'C')):
        if (orderIn % 2 == 0):
            order = orderIn
        else:
            order = int(orderIn + 1)
        Ng = int(order / 2)
    else:
        if (orderIn % 2 == 0):
            order = int(orderIn + 1)
        else:
            order = orderIn
        Ng = int((order + 1) / 2)
    
    Eg1, Eg2 = OT.GhostCellsJump(omega, physics, E, Ng, order)
    Bg1, Bg2 = OT.GhostCellsJump(omega, physics, B, Ng, order)

    # Transform to the computational vars w/ eigen xform
    E1 = np.concatenate((E[:matInd], Eg1))  # with ghost cell values at the jump
    B1 = np.concatenate((B[:matInd], Bg1))
    phil1 = 0.5 * (E1 - c1 * B1)  # characteristic variable
    phir1 = 0.5 * (E1 + c1 * B1)
    E2 = np.concatenate((Eg2, E[matInd:]))  # with ghost cell values at the jump
    B2 = np.concatenate((Bg2, B[matInd:]))
    phil2 = 0.5 * (E2 - c2 * B2)
    phir2 = 0.5 * (E2 + c2 * B2)

    # Regular Face stencil from cell-averages
    
    faceOp1L, faceOp2L, faceOpL = OT.FaceOp(omega, order, diff, 'L', Ng)
    faceOp1R, faceOp2R, faceOpR = OT.FaceOp(omega, order, diff, 'R', Ng)

    # Face values from upwind (index 1:N for faces left of cell + 1 for mat)
    phil1f = faceOp1L @ np.concatenate((np.full(Ng, phil1[0]), phil1))  # outflow bc's on left
    phir1f = faceOp1R @ np.concatenate((np.zeros(Ng), phir1))  # 0 at leftmost face for inflow boundary conditions
    phil2f = faceOp2L @ np.concatenate((phil2, np.zeros(Ng))).transpose()  # 0 at rightmost face for inflow boundary conditions
    phir2f = faceOp2R @ np.concatenate((phir2, np.full(Ng, phir2[matInd + Ng - 1])))  # outflow bc's on right

    # Correct values at material interface with jump conditions
    T1 = 2 * c1 / (c1 + c2)
    R1 = (c2 - c1) / (c1 + c2)
    phil1f[matInd] = T1 * phil2f[0] + R1 * phir1f[matInd]
    T2 = 2 * c2 / (c1 + c2)
    R2 = (c1 - c2) / (c1 + c2)
    phir2f[0] = R2 * phil2f[0] + T2 * phir1f[matInd]

    # Transform back to E,B on faces
    E1f = phil1f + phir1f
    B1f = (-phil1f + phir1f) / c1
    E2f = phil2f + phir2f
    B2f = (-phil2f + phir2f) / c2
    
    faceOp1r, faceOp2r, faceOpr = OT.FaceOp(omega, 1, 'U', 'R', 1)
    faceOp1l, faceOp2l, faceOpl = OT.FaceOp(omega, 1, 'U', 'R', 1, True)
    
    derivOp1 = (faceOp1r - faceOp1l)[1:, :-1]
    derivOp2 = (faceOp2r - faceOp2l)[1:, :-1]

    # Calculate the RHS for E, B
    rhsE = hMat @ np.append(-c1**2*derivOp1 @ B1f, -c2**2*derivOp2 @ B2f)
    rhsB = hMat @ np.append(-1*derivOp1 @ E1f, -1* derivOp2 @ E2f)
    
    rhs = np.append(rhsE, rhsB)

    return rhs
