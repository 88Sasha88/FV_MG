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


# This function plots a number line with $n^{h}$ cells onto `ax`.

# In[4]:


def TickPlot(nh, ax, tickHeight):
    xAxis, yAxis = BT.MakeXY(nh)
    for (xi, yi) in zip(xAxis, yAxis):
        if ((xi == 0) or (xi == 1)):
            height = tickHeight
        else:
            height = tickHeight / 2
        (xs, ys) = DrawLine(xi, yi, height)
        ax.plot(xs, ys, color = 'k', zorder = 1)
    ax.plot(xAxis, yAxis, color = 'k', zorder = 0)
#     plt.xlim([-0.1, 1.25])
#     plt.ylim([-2.5, 2.5])
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
    font = 15
    return numPoints, font


# This function iterates through the modes and overlays the piecewise cell average plots onto plots of their respective continuous wave functions alongside written labels of the equations they should each represent. It also gives the option of plotting the node point values. It also allows you to save those plots if desired. As the default, these two features are subdued.

# In[7]:


def PlotWaves(nh, h, waveCell, x, waveNode, plotNode = False, save = False):
    problemCell = BT.CheckSize(nh, waveCell)
    problemX = BT.CheckSize(nh, x)
    problemNode = BT.CheckSize(nh, waveNode)
    if (problemCell != 0):
        sys.exit('ERROR:\nPlotTools:\nPlotWaves:\nnh does not match size of waveCell!')
    if (problemX != 0):
        sys.exit('ERROR:\nPlotTools:\nPlotWaves:\nnh does not match size of x!')
    if (problemNode != 0):
        sys.exit('ERROR:\nPlotTools:\nPlotWaves:\nnh does not match size of waveNode!')
    numPoints, font = UsefulPlotVals(nh)
    X, waveCont = WT.MakeNodeWaves(nh, h, nRes = numPoints)
    for k in range(nh):
        saveName = 'FourierModes' + str(nh - k)
        yMin, yMax, tickHeight = GetYBound(waveCont[:, k], 0.5, sym = True)
        PlotWave(nh, numPoints, tickHeight, X, waveCell[:, k], waveCont[:, k], save, saveName = saveName)
        if (plotNode):
            plt.scatter(x, waveNode[:, k], color = ColorDefault(2), s = 10, zorder = 4)
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
    problemX = BT.CheckSize(nh, x)
    problemWave = BT.CheckSize(nh, waves)
    if (problemX != 0):
        sys.exit('ERROR:\nPlotTools:\nPlotGeneralWaves:\nnh does not match size of x!')
    if (problemWave != 0):
        sys.exit('ERROR:\nPlotTools:\nPlotGeneralWaves:\nnh does not match size of waves!')
    numPoints, font = UsefulPlotVals(nh)
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
        TickPlot(nh, ax, 0.5)
        if (save):
            fig.savefig(savePath + saveName + str(nh - k) + '.png', bbox_inches = 'tight', dpi = 600, transparent = True)
        plt.show()
    return


# This 

# In[9]:


def PlotWave(nh, numPoints, tickHeight, X, waveCell, fX, save, saveName = 'WavePlot'):
    fig, ax = plt.subplots(figsize = (5, 2.5))
    ax = plt.axes(frameon = False)
    PiecePlot(nh, numPoints, X, waveCell)
    TickPlot(nh, ax, tickHeight)
    plt.plot(X, fX, color = ColorDefault(0), zorder = 2)
    if (save):
        fig.savefig(savePath + saveName + '.png', bbox_inches = 'tight', dpi = 600, transparent = True)
    return


# This 

# In[10]:


def PlotMixedWave(nh, h, waveCell, waveCoef, save = False):
    numPoints, font = UsefulPlotVals(nh)
    X, waveCont = WT.MakeNodeWaves(nh, h, nRes = numPoints)
    fXCell = OT.ChangeBasis(nh, waveCoef, waveCell)
    fXCont = OT.ChangeBasis(nh, waveCoef, waveCont)
    saveName = 'MixedWave'
    yMin, yMax, tickHeight = GetYBound(fXCont, 0.25)
    PlotWave(nh, numPoints, tickHeight, X, fXCell, fXCont, save, saveName = saveName)
    plt.xlim([-0.1, 1.1])
    plt.ylim([yMin, yMax])
    return


# This 

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





# In[ ]:




