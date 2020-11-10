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
import WaveTools as WT
import OperatorTools as OT


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


def DrawLine(xCenter, yCenter, tickHeight):
    x = xCenter * np.ones(2)
    y = linspace(yCenter - (tickHeight / 2), yCenter + (tickHeight / 2), num = 2)
    return (x, y)


# This function plots a number line marking off the grid onto `ax`.

# In[4]:


def TickPlot(omega, ax, tickHeight):
    xAxis = omega.xNode
    yAxis = omega.y
    for (xi, yi) in zip(xAxis, yAxis):
        if ((xi == 0) or (xi == 1)):
            height = tickHeight
        else:
            height = tickHeight / 2
        (xs, ys) = DrawLine(xi, yi, height)
        ax.plot(xs, ys, color = 'k', zorder = 1)
    ax.plot(xAxis, yAxis, color = 'k', zorder = 0)
    plt.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)
    plt.tick_params(axis = 'y', which = 'both', left = False, right = False, labelleft = False)
    return


# This function plots out the piecewise cell averages.

# In[5]:


def PiecePlot(omega, numPoints, X, pieces):
    errorLoc = 'ERROR:\nPlotTools:\nPiecePlot:\n'
    errorMess = BT.CheckSize(numPoints, X, nName = 'numPoints', matricaName = 'X')
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
    x = omega.xNode
    n = len(x) - 1
    cellVals = np.ones(numPoints, float)
    lowIndex = 0
    for k in range(n):
        highIndex = np.where(X <= x[k + 1])[0][::-1][0] + 1
        cellVals[lowIndex:highIndex] = pieces[k] * cellVals[lowIndex:highIndex]
        plt.plot(X[lowIndex:highIndex], cellVals[lowIndex:highIndex], color = ColorDefault(3), zorder = 3)
        lowIndex = highIndex
    return


# This function allows for convenient control over ubiquitous plotting parameters and objects so that they don't have to be constantly passed around all over.

# In[6]:


def UsefulPlotVals():
    numPoints = 129
    font = 15
    X = np.linspace(0, 1, num = numPoints)
    return numPoints, font, X


# This function iterates through the modes and overlays the piecewise cell average plots onto plots of their respective continuous wave functions alongside written labels of the equations they should each represent. It also gives the option of plotting the node point values. It also allows you to save those plots if desired. As the default, these two features are subdued.

# In[7]:


def PlotWaves(omega, waveCell, waveNode, plotNode = False, save = False):
    nh = omega.nh_max
    x = omega.xNode
    n = len(x) - 1
    numPoints, font, X = UsefulPlotVals()
    waveCont = WT.MakeNodeWaves(omega, nRes = numPoints)
    for k in range(nh):
        saveName = 'FourierModes' + str(nh - k)
        yMin, yMax, tickHeight = GetYBound(waveCont[:, k], 0.5, sym = True)
        PlotWave(omega, numPoints, tickHeight, X, waveCell[:, k], waveCont[:, k], save, saveName = saveName)
        if (plotNode):
            plt.scatter(x[:n], waveNode[:, k], color = ColorDefault(2), s = 10, zorder = 4)
        plt.xlim([-0.1, 1.25])
        plt.ylim([yMin, yMax])
        if (k % 2 == 0):
            if (k == 0):
                plt.text(1.1, 0, r'$\frac{a_{0}}{2}$', fontsize = font)
            else:
                plt.text(1.1, 0, r'$a_{%d}$' %(k / 2) + 'cos' + r'$%d \pi x$' %(k), fontsize = font)
        else:
            plt.text(1.1, 0, r'$b_{%d}$' %((k / 2) + 1) + 'sin' + r'$%d \pi x$' %(k + 1), fontsize = font)
        plt.show()
    return


# This function just plots the real and imaginary parts of an arbitrary matrix of functions alongside written notes of each of the wave equations to which they should correspond. It also allows you to save those plots and select their file names if desired. As the default, that feature is subdued.

# In[8]:


def PlotGeneralWaves(nh, x, waves, save = False, saveName = 'PlotOutputs'):
    errorLoc = 'ERROR:\nPlotTools:\nPlotGeneralWaves:\n'
    errorMess = BT.CheckSize(nh, x, nName = 'nh', matricaName = 'x')
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
    errorMess = BT.CheckSize(nh, waves, nName = 'nh', matricaName = 'waves')
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
    numPoints, font = UsefulPlotVals()
    for k in range(nh):
        fig, ax = plt.subplots(figsize = (5, 2.5))
        ax.set_aspect(aspect = 4)
        ax = plt.axes(frameon = False)
        plt.xlim([-0.1, 1.25])
        plt.ylim([-2.5, 2.5])
        if (k % 2 == 0):
            if (k == 0):
                plt.text(1.1, 0, r'$\frac{a_{0}}{2}$', fontsize = font)
            else:
                plt.text(1.1, 0, r'$a_{%d}$' %(k / 2) + 'cos' + r'$%d \pi x$' %(k), fontsize = font)
        else:
            plt.text(1.1, 0, r'$b_{%d}$' %((k / 2) + 1) + 'sin' + r'$%d \pi x$' %(k + 1), fontsize = font)
        plt.plot(x, waves[:, k].real, color = ColorDefault(0), zorder = 2)
        plt.plot(x, waves[:, k].imag, color = ColorDefault(3), zorder = 3)
        TickPlot(omega, ax, 0.5)
        if (save):
            fig.savefig(savePath + saveName + str(nh - k) + '.png', bbox_inches = 'tight', dpi = 600, transparent = True)
        plt.show()
    return


# This function overlays a particular piecewise cell average plot onto a plot of its continuous wave function. It also allows you to save this plot if desired. As the default, that feature is subdued.

# In[9]:


def PlotWave(omega, numPoints, tickHeight, X, waveCell, fX, save, saveName = 'WavePlot'):
    fig, ax = plt.subplots(figsize = (5, 2.5))
    ax = plt.axes(frameon = False)
    PiecePlot(omega, numPoints, X, waveCell)
    TickPlot(omega, ax, tickHeight)
    plt.plot(X, fX, color = ColorDefault(0), zorder = 2)
    if (save):
        fig.savefig(savePath + saveName + '.png', bbox_inches = 'tight', dpi = 600, transparent = True)
    return


# This function overlays a piecewise cell average plot of a linear combination of wave vectors onto a plot of its continuous wave function. It also allows you to save this plot if desired. As the default, that feature is subdued.

# In[10]:


def PlotMixedWave(omega, waveCell, waveCoef, save = False):
    nh = omega.nh_max
    numPoints, font, X = UsefulPlotVals()
    waveCont = WT.MakeNodeWaves(omega, nRes = numPoints)
    fXCell = OT.ChangeBasis(nh, waveCoef, waveCell)
    fXCont = OT.ChangeBasis(nh, waveCoef, waveCont)
    saveName = 'MixedWave'
    yMin, yMax, tickHeight = GetYBound(fXCont, 0.25)
    PlotWave(omega, numPoints, tickHeight, X, fXCell, fXCont, save, saveName = saveName)
    plt.xlim([-0.1, 1.1])
    plt.ylim([yMin, yMax])
    return


# This function outputs the $y$ limits for a graph along with their respective tick height.

# In[10]:


def GetYBound(inputArray, scaleParam, sym = False):
    yMin = np.min(inputArray)
    yMax = np.max(inputArray)
    totRange = yMax - yMin
    if (totRange == 0 and yMin != yMax):
        yMax = 0.1
        yMin = -0.1
    if (sym):
        yMax = np.max((np.abs(yMax), np.abs(yMin)))
        yMin = -yMax
    else:
        if (yMin > -yMax / 19):
            yMin = -np.abs(yMax / 19)
        if (yMax < -yMin / 19):
            yMax = np.abs(yMin / 19)
    totRange = yMax - yMin
    yMin = yMin - (scaleParam * totRange)
    yMax = yMax + (scaleParam * totRange)
    totRange = yMax - yMin
    tickHeight = totRange / 10
    return yMin, yMax, tickHeight


# In[ ]:




