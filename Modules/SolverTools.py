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


def ForwardEuler(omega, waves, u0, nt, const, CFL, func, order = 0):
    dx = omega.h
    dx_min = np.min(dx)
    dt = CFL * dx_min / const
    if (order != 0):
        if (func != TimePoly):
            print('Spatial derivative method has been overridden in favor of TimePoly()!')
        func = TimePoly
    u = u0.copy()
    t = 0
    for n in range(nt):
        u = u + (dt * func(omega, t, u, const, order))
        t = t + dt
        if (func == TimePoly):
            if (n == nt - 1):
                val = func(omega, t, u, const, order + 1, deriv = False)
                if (order < 2):
                    midstring = ' should be equal to '
                else:
                    midstring = ' does not need to equal '
                print(str(u[0]) + midstring + str(val) + '.')
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
    f = -(const / dx) * ((B * u0) - (C * np.roll(u0, 1)) - (D * np.roll(u0, 2)))
    return f

def CenterDiff(omega, t, u0, const, order):
    degFreed = omega.degFreed
    dx = omega.h
    
    # A is the main diagonal; C is the subdiagonal; G is the sub-subdiagonal; E is the superdiagonal; H is the super-superdiagonal.
    A = dx - np.roll(dx, 1)
    B = A + 0
    F = np.roll(A, -1)
    F[F > 0] = 1. / 3.
    F[F != 1. / 3.] = 0
    A[A < 0] = -1. / 3.
    A[A != -1. / 3.] = 0
    A = A - F
    B[B > 0] = 0.5
    B[B < 0] = 2. / 3.
    C = -B
    B[B < 2. / 3.] = 1.
    C[C == 0] = -1.
    D = C + 0
    D[D != -0.5] = 0
    E = -C
    E[E == 0.5] = 4. /3.
    E[E == 2. / 3.] = 0.5
    G = C + 0
    G[G != -0.5] = 0
    H = E + 0
    H[H != 0.5] = 0
    
    print('')
    print('Start:')
    print(H)
    print(np.roll(dx, -2))
    print('')
    print(E)
    print(np.roll(dx, -1))
    print('')
    print(A)
    print(dx)
    print('')
    print(C)
    print(np.roll(dx, 1))
    print('')
    print(G)
    print(np.roll(dx, 2))
    print('')
    
    f = -(const / (2 * dx)) * ((H * np.roll(dx, -2)) + (E * np.roll(dx, -1)) + (A * dx) + (C * np.roll(dx, 1)) + (G * np.roll(dx, 2)))
    return f

def MidpointMeth(omega, waves, u0, nt, const, CFL, func, order = 0):
    dx = omega.dx
    dx_min = np.min(dx)
    dt = CFL * dx_min / const
    if (order > 0):
        if (func != TimePoly):
            print('Spatial derivative method has been overridden in favor of TimePoly()!')
        func = TimePoly
    if (order != 0):
        func = TimePoly
    u = u0.copy()
    t = 0
    for n in range(nt):
        k1 = func(omega, t, u, const, order)
        k2 = func(omega, t + (dt / 2.), u + ((dt / 2.) * k1), const, order)
        u = u + (dt * k2)
        t = t + dt
        if (func == TimePoly):
            if (n == nt - 1):
                val = func(omega, t, u, const, order + 1, deriv = False)
                if (order < 3):
                    midstring = ' should be equal to '
                else:
                    midstring = ' does not need to equal '
                print('k1 = ' + str(k1))
                print('k2 = ' + str(k2))
                print(str(u[0]) + midstring + str(val) + '.')
    uCoefs = LA.inv(waves) @ u
    return uCoefs

def RK4(omega, waves, u0, nt, const, CFL, func, order = 0):
    dx = omega.dx
    dx_min = np.min(dx)
    dt = CFL * dx_min / const
    if (order > 0):
        if (func != TimePoly):
            print('Spatial derivative method has been overridden in favor of TimePoly()!')
        func = TimePoly
    u = u0.copy()
    t = 0
    for n in range(nt):
        k1 = func(omega, t, u, const, order)
        k2 = func(omega, t + (dt / 2.), u + ((dt / 2.) * k1), const, order)
        k3 = func(omega, t + (dt / 2.), u + ((dt / 2.) * k2), const, order)
        k4 = func(omega, t + dt, u + (dt * k3), const, order)
        u = u + ((dt / 6.) * (k1 + (2. * k2) + (2. * k3) + k4))
        t = t + dt
        if (func == TimePoly):
            if (n == nt - 1):
                val = func(omega, t, u, const, order + 1, deriv = False)
                if (order < 5):
                    midstring = ' should be equal to '
                else:
                    midstring = ' does not need to equal '
                print('k1 = ' + str(k1))
                print('k2 = ' + str(k2))
                print('k3 = ' + str(k3))
                print('k4 = ' + str(k4))
                print(str(u[0]) + midstring + str(val) + '.')
    uCoefs = LA.inv(waves) @ u
    return uCoefs

def TimePoly(omega, t, u, const, order, deriv = True):
    # add error checker for negative order or zero
    poly = 0
    for n in range(0, order):
        if (deriv):
            poly = poly + ((n + 1) * (t**n))
        else:
            poly = poly + (t**n)
    return poly
