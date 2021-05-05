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

display(HTML("<style>pre { white-space: pre !important; }</style>"))
np.set_printoptions( linewidth = 1000)


# In[2]:


def ForwardEuler(omega, waves, u0, nt, const, CFL, func):
    dx = omega.h
    dx_min = np.min(dx)
    dt = CFL * dx_min / const
    u = u0.copy()
    t = 0
    for n in range(nt):
        u = u + (dt * func(omega, t, u, const))
        t = t + dt
    uCoefs = LA.inv(waves) @ u
    return uCoefs


def CalcTime(omega, CFL, const, nt = 0, t = 0):
    errorLoc = 'ERROR:\nSolverTools:\nCalcTime:\n'
    errorMess = ''
    dx = omega.dx
    dx_min = np.min(dx)
    dt = CFL * dx_min / const
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

def Upwind(omega, t, u0, const):
    degFreed = omega.degFreed
    dx = omega.h
    f = -(const / dx) * (u0 - np.roll(u0, 1))
    return f

def CenterDiff(omega, t, u0, const):
    degFreed = omega.degFreed
    dx = omega.h
    f = -(const / (2 * dx)) * (np.roll(u0, -1) - np.roll(u0, 1))
    return f

def MidpointMeth(omega, waves, u0, nt, const, CFL, func):
    dx = omega.dx
    dx_min = np.min(dx)
    dt = CFL * dx_min / const
    u = u0.copy()
    t = 0
    for n in range(nt):
        k1 = func(omega, t, u, const)
        k2 = func(omega, t + (dt / 2.), u + ((dt / 2.) * k1), const)
        u = u + (dt * k2)
        t = t + dt
    uCoefs = LA.inv(waves) @ u
    return uCoefs

def RK4(omega, waves, u0, nt, const, CFL, func):
    dx = omega.dx
    dx_min = np.min(dx)
    dt = CFL * dx_min / const
    u = u0.copy()
    t = 0
    for n in range(nt):
        k1 = func(omega, t, u, const)
        k2 = func(omega, t + (dt / 2.), u + ((dt / 2.) * k1), const)
        k3 = func(omega, t + (dt / 2.), u + ((dt / 2.) * k2), const)
        k4 = func(omega, t + dt, u + (dt * k3), const)
        u = u + ((dt / 6.) * (k1 + (2. * k2) + (2. * k3) + k4))
        t = t + dt
    uCoefs = LA.inv(waves) @ u
    return uCoefs

# def PolyTime(omega, t, u0, const):
    
#     return
