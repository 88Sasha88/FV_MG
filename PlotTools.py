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


def PiecePlot(omega, numPoints, X, pieces, color = 3, linestyle = '-'):
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
        plt.plot(X[lowIndex:highIndex], cellVals[lowIndex:highIndex], color = ColorDefault(color), linestyle = linestyle, zorder = 3)
        lowIndex = highIndex
    return


# This function allows for convenient control over ubiquitous plotting parameters and objects so that they don't have to be constantly passed around all over.

# In[6]:


def UsefulPlotVals():
    numPoints = 357
    font = 15
    X = np.linspace(0, 1, num = numPoints)
    savePath = '/Users/sashacurcic/SashasDirectory/ANAG/FV_MG/Figures/'
    return numPoints, font, X, savePath


# This function iterates through the modes and overlays the piecewise cell average plots onto plots of their respective continuous wave functions alongside written labels of the equations they should each represent. It also gives the option of plotting the node point values. It also allows you to save those plots if desired. As the default, these two features are subdued.

# In[7]:


def PlotWaves(omega, waves, waveNode = [], waveTrans = [], save = False, rescale = 1, nullspace = []):
    nh = omega.nh_max
    x = omega.xNode
    n = omega.degFreed[::-1][0]
    N = nh
    numPoints, font, X, savePath = UsefulPlotVals()
    waveCont = WT.MakeNodeWaves(omega, nRes = numPoints)
    if (nullspace == []):
        nullspace = np.eye(nh, nh)
        strings = omega.strings
    else:
        N = n
        strings = FixStrings(omega, nullspace)
    if (waveNode != []):
        waveNodes = waveNode @ nullspace
    waveCell = waves @ nullspace
    waveCont = waveCont @ nullspace
    for k in range(N):
        if (waveTrans != []):
            if (k < np.shape(waveTrans)[1]):
                waveTransfer = waveTrans[:, k]
        else:
            waveTransfer = []
        fig = PlotWave(omega, numPoints, X, waveCell[:, k], waveCont[:, k], rescale, waveTrans = waveTransfer)
        if (waveNode != []):
            plt.scatter(x[:n], waveNodes[:, k], color = ColorDefault(2), s = 10, zorder = 4)
        plt.xlim([-0.1, 1.25])
        plt.text(1.1, 0, strings[k], fontsize = font)
        plt.show()
        if (save):
            saveName = savePath + 'FourierModes' + str(k + 1)
            fig.savefig(saveName + '.png', bbox_inches = 'tight', dpi = 600, transparent = True)
            print('This image has been saved under ' + saveName + '.')
    return


# This function overlays a particular piecewise cell average plot onto a plot of its continuous wave function. It also allows you to save this plot if desired. As the default, that feature is subdued.

# In[8]:


def PlotWave(omega, numPoints, X, waveCell, fX, rescale, waveTrans = [], sym = True):
    errorLoc = 'ERROR:\nPlotTools:\nPlotWave:\n'
    yMin, yMax, tickHeight = GetYBound(fX, sym = sym)
    if np.any(np.asarray(rescale) <= 0):
        errorMess = 'All values of rescale must be greater than 0!'
        sys.exit(errorLoc + errorMess)
    if (np.shape(rescale) == ()):
        size = [5 * rescale, 2.5 * rescale]
        tickHeight = tickHeight / rescale
    else:
        if (np.shape(rescale) == (2,)):
            size = [5 * rescale[0], 2.5 * rescale[1]]
            tickHeight = tickHeight / rescale[1]
        else:
            errorMess = 'Invalid shape of rescale object entered!'
            sys.exit(errorLoc + errorMess)
    fig, ax = plt.subplots(figsize = size)
    ax = plt.axes(frameon = False)
    PiecePlot(omega, numPoints, X, waveCell)
    if (waveTrans != []):
        PiecePlot(omega, numPoints, X, waveTrans, color = 3)
    TickPlot(omega, ax, tickHeight)
    plt.plot(X, fX, color = ColorDefault(0), zorder = 2) # 1
    plt.ylim([yMin, yMax])
    return fig


# This function overlays a piecewise cell average plot of a linear combination of wave vectors onto a plot of its continuous wave function. It also allows you to save this plot if desired. As the default, that feature is subdued.

# In[9]:


def PlotMixedWave(omega, waveCell, waveCoef, rescale = 1, sym = False, save = False):
    nh = omega.nh_max
    numPoints, font, X, savePath = UsefulPlotVals()
    waveCont = WT.MakeNodeWaves(omega, nRes = numPoints)
    
    fXCell = waveCell @ waveCoef
    fXCont = waveCont @ waveCoef
    saveName = savePath + 'MixedWave'
    fig = PlotWave(omega, numPoints, X, fXCell, fXCont, rescale, sym = sym)
    plt.xlim([-0.1, 1.1])
    if (save):
        fig.savefig(saveName + '.png', bbox_inches = 'tight', dpi = 600, transparent = True)
        print('This image has been saved under ' + saveName + '.')
    return


# This function outputs the $y$ limits for a graph along with their respective tick height. `scaleParam` is the percentage of the range which will be neutral space.

# In[11]:


def GetYBound(inputArray, scaleParam = 0.25, sym = False):
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
# If the positive range is less than a 20th of the total range (That is, it's less than a 19th of the negative range.) then the positive half of the tick mark won't fully show, and vice versa.
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


# This function fixes a list of strings such that linear combinations are represented appropriately.

# In[12]:


def FixStrings(omega, nullspace):
    errorLoc = 'ERROR:\nPlotTools:\nFixStrings:\n'
    strings = omega.strings
    degFreed = omega.degFreed[::-1][0]
    errorMess = BT.CheckSize(degFreed, nullspace[0, :], nName = 'degFreed', matricaName = 'nullspace')
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
    locations = np.where(nullspace != 0)
    stringsNew = ['' for i in range(degFreed)]
    j = 0
    for i in locations[1]:
        if (stringsNew[i] == ''):
            stringsNew[i] = strings[locations[0][j]]
        else:
            stringsNew[i] = stringsNew[i] + '+' + strings[locations[0][j]]
        j = j + 1
    return stringsNew


# In[ ]:




