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


def ForwardEuler(omega, waves, u0, nt, const, CFL, periodic = True):
    degFreed = omega.degFreed# [::-1][0]
    x = omega.xCell
    dx = omega.dx
    dx_min = np.min(dx)
    dx_0 = 1 - x[::-1][0] + x[0]
    dt = CFL * dx_min / const
    dt_0 = CFL * dx_0 / const
    t = nt * dt
    u = u0.copy()
    for n in range(nt):
        u_f = u[::-1][0]
        u[1:] = u[1:] - (const * (dt / dx) * (u[1:] - u[:-1]))
        if (periodic == True):
            u[0] = u[0] - (const * (dt_0 / dx_0) * (u[0] - u_f))
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
