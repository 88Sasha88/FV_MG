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
import BasicTools as BT


# This function just allows me to have easy control over the prettier plotting colors.

# In[2]:


def ColorDefault(k):
    if (k == 0):
        color = '#1f77b4'  # blue
    else:
        if (k % 9 == 0):
            color = '#17becf'  # cyan
        else:
            if (k % 8 == 0):
                color = '#bcbd22'  # sickly greenish tan
            else:
                if (k % 7 == 0):
                    color = '#7f7f7f'  # grey
                else:
                    if (k % 6 == 0):
                        color = '#e377c2'  # pink
                    else:
                        if (k % 5 == 0):
                            color = '#8c564b'  # brown
                        else:
                            if (k % 4 == 0):
                                color = '#9467bd'  # purple
                            else:
                                if (k % 3 == 0):
                                    color = '#d62728'  # red
                                else:
                                    if (k % 2 == 0):
                                        color = '#2ca02c'  # green
                                    else:
                                        color = '#ff7f0e'  # orange
    return color


# This function returns a tick mark of height `h` at location (`xCenter`, `yCenter`).

# In[3]:


def DrawLine(xCenter, yCenter, h):
    x = xCenter * np.ones(2)
    y = linspace(yCenter - (h / 2), yCenter + (h / 2), num = 2)
    return (x, y)


# This function plots a number line with $n^{h}$ cells onto `ax`.

# In[4]:


def TickPlot(nh, ax):
    xAxis, yAxis = BT.MakeXY(nh)
    for (xi, yi) in zip(xAxis, yAxis):
        if ((xi == 0) or (xi == 1)):
            h = 0.5
        else:
            h = 0.25
        (xs, ys) = DrawLine(xi, yi, h)
        ax.plot(xs, ys, color = 'k', zorder = 1)
    ax.plot(xAxis, yAxis, color = 'k', zorder = 0)
    plt.xlim([-0.1, 1.25])
    plt.ylim([-2.5, 2.5])
    plt.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)
    plt.tick_params(axis = 'y', which = 'both', left = False, right = False, labelleft = False)
    return


# This function plots out the piecewise cell averages.

# In[5]:


def PiecePlot(nh, numPoints, X, pieces):
    problemX = BT.CheckSize(numPoints, X)
    problemAve = BT.CheckSize(nh, pieces)
    if (problemX != 0):
        sys.exit('ERROR:\nPiecePlot:\nnumPoints does not match size of X!')
    if (problemAve != 0):
        sys.exit('ERROR:\nPiecePlot:\nnh does not match size of pieces!')
    x, y = BT.MakeXY(nh)
    cellVals = np.ones(numPoints, float)
    lowIndex = 0
    for k in range(nh):
        highIndex = np.where(X <= x[k + 1])[0][::-1][0] + 1
        cellVals[lowIndex:highIndex] = pieces[k] * cellVals[lowIndex:highIndex]
        plt.plot(X[lowIndex:highIndex], cellVals[lowIndex:highIndex], color = ColorDefault(3), zorder = 3)
        lowIndex = highIndex
    return


# This function allows for convenient control over ubiquitous plotting parameters and objects so that they don't have to be constantly passed around all over.

# In[6]:


def UsefulPlotVals(nh):
    numPoints = 129
    X = linspace(0, 1, num = numPoints)
    font = 15
    return numPoints, X, font


# This function overlays the piecewise cell average plots onto plots of their respective continuous wave functions alongside written labels of the equations they should each represent. It also gives the option of plotting the node point values. It also allows you to save those plots if desired. As the default, these two features are subdued.

# In[7]:


def PlotWaves(nh, waveCell, x, waveNode, plotNode = False, save = False):
    problemCell = BT.CheckSize(nh, waveCell)
    problemX = BT.CheckSize(nh, x)
    problemNode = BT.CheckSize(nh, waveNode)
    if (problemCell != 0):
        sys.exit('ERROR:\nPlotWaves:\nnh does not match size of waveCell!')
    if (problemX != 0):
        sys.exit('ERROR:\nPlotWaves:\nnh does not match size of x!')
    if (problemNode != 0):
        sys.exit('ERROR:\nPlotWaves:\nnh does not match size of waveNode!')
    numPoints, X, font = UsefulPlotVals(nh)
    for k in range(nh):
        fig, ax = plt.subplots(figsize = (5, 2.5))
        ax.set_aspect(aspect = 4)
        ax = plt.axes(frameon = False)
        if (k % 2 == 0):
            fX = np.cos(np.pi * k * X)
            if (k == 0):
                plt.text(1.1, 0, r'$\frac{a_{0}}{2}$', fontsize = font)
            else:
                plt.text(1.1, 0, r'$a_{%d}$' %(k / 2) + 'cos' + r'$%d \pi x$' %(k), fontsize = font)
        else:
            fX = np.sin(np.pi * (k + 1) * X)
            plt.text(1.1, 0, r'$b_{%d}$' %((k / 2) + 1) + 'sin' + r'$%d \pi x$' %(k + 1), fontsize = font)
        PiecePlot(nh, numPoints, X, waveCell[:, k])
        TickPlot(nh, ax)
        plt.plot(X, fX, color = ColorDefault(0), zorder = 2)
        if (plotNode):
            plt.scatter(x, waveNode[:, k], color = ColorDefault(2), s = 10, zorder = 4)
        if (save):
            fig.savefig(savePath + 'FourierModes' + str(nh - k) + '.png', bbox_inches = 'tight', dpi = 600, transparent = True)
        plt.show()
    return


# This function just plots the real and imaginary parts of an arbitrary matrix of functions alongside written notes of each of the wave equations to which they should correspond. It also allows you to save those plots and select their file names if desired. As the default, that feature is subdued.

# In[8]:


def PlotGeneralWaves(nh, x, waves, save = False, saveName = 'PlotOutputs'):
    problemX = BT.CheckSize(nh, x)
    problemWave = BT.CheckSize(nh, waves)
    if (problemX != 0):
        sys.exit('ERROR:\nPlotGeneralWaves:\nnh does not match size of x!')
    if (problemWave != 0):
        sys.exit('ERROR:\nPlotGeneralWaves:\nnh does not match size of waves!')
    numPoints, X, font = UsefulPlotVals(nh)
    for k in range(nh):
        fig, ax = plt.subplots(figsize = (5, 2.5))
        ax.set_aspect(aspect = 4)
        ax = plt.axes(frameon = False)
        if (k % 2 == 0):
            if (k == 0):
                plt.text(1.1, 0, r'$\frac{a_{0}}{2}$', fontsize = font)
            else:
                plt.text(1.1, 0, r'$a_{%d}$' %(k / 2) + 'cos' + r'$%d \pi x$' %(k), fontsize = font)
        else:
            plt.text(1.1, 0, r'$b_{%d}$' %((k / 2) + 1) + 'sin' + r'$%d \pi x$' %(k + 1), fontsize = font)
        plt.plot(x, waves[:, k].real, color = ColorDefault(0), zorder = 2)
        plt.plot(x, waves[:, k].imag, color = ColorDefault(3), zorder = 3)
        TickPlot(nh, ax)
        if (save):
            fig.savefig(savePath + saveName + str(nh - k) + '.png', bbox_inches = 'tight', dpi = 600, transparent = True)
        plt.show()
    return


# In[ ]:





# In[ ]:




