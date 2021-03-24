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


def TickPlot(omega, ax, tickHeight, label = False):
    ax = plt.axes(frameon = False)
    xAxis = omega.xNode
    yAxis = omega.y
    nh = omega.nh_max
    shiftX = 0.02
    i = 0
    for (xi, yi) in zip(xAxis, yAxis):
        if ((xi == 0) or (xi == 1)):
            height = tickHeight
            shiftY = tickHeight
            if (label):
                plt.text(xi - shiftX, yi + shiftY, int(xi), fontsize = 12)
        else:
            height = tickHeight / 2
        (xs, ys) = DrawLine(xi, yi, height)
        ax.plot(xs, ys, color = 'k', zorder = 1)
        if (label):
            print(i)
            if ((i < 3) or (i > nh - 2)):
                prestring = r'$j = $'
                istring = prestring + str(i)
                shiftExtra = 2 * shiftX
                if (i == nh - 1):
                    print('hi')
                    istring = prestring + r'$n - 1$'
                    shiftExtra = 4 * shiftX
                if (i == nh):
                    print('ho')
                    istring = prestring + r'$n$'
                    shiftExtra = shiftX
                plt.text(xi - shiftX - shiftExtra, yi - (1.5 * shiftY), istring, fontsize = 12)
        i = i + 1
    ax.plot(xAxis, yAxis, color = 'k', zorder = 0)
    plt.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = False)
    plt.tick_params(axis = 'y', which = 'both', left = False, right = False, labelleft = False)
    return


# This function plots out the piecewise cell averages.

# In[5]:


def PiecePlot(omega, numPoints, X, pieces, color = 3, label = [], linestyle = '-'):
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
        if ((k == 0) and (label != [])):
            plt.plot(X[lowIndex:highIndex], cellVals[lowIndex:highIndex], color = ColorDefault(color), linestyle = linestyle, zorder = 3, label = label)
        else:
            plt.plot(X[lowIndex:highIndex], cellVals[lowIndex:highIndex], color = ColorDefault(color), linestyle = linestyle, zorder = 3)
        lowIndex = highIndex
    return


# This function allows for convenient control over ubiquitous plotting parameters and objects so that they don't have to be constantly passed around all over.

# In[6]:


def UsefulPlotVals():
    numPoints = 1025
    font = 15
    X = np.linspace(0, 1, num = numPoints)
    savePath = '/Users/sashacurcic/SashasDirectory/ANAG/FV_MG/Figures/'
    return numPoints, font, X, savePath


# This function iterates through the modes and overlays the piecewise cell average plots onto plots of their respective continuous wave functions alongside written labels of the equations they should each represent. It also gives the option of plotting the node point values. It also allows you to save those plots if desired. As the default, these two features are subdued.

# In[7]:


def PlotWaves(omega, waves = [], waveNode = [], waveTrans = [], save = False, saveName = '', rescale = 1, nullspace = [], dpi = 600):
    nh = omega.nh_max
    x = omega.xNode
    n = omega.degFreed
    if (omega.alias):
        nh = int(2 * nh)
    N = nh
    numPoints, font, X, savePath = UsefulPlotVals()
    if (saveName != ''):
        save = True
    else:
        saveName = 'FourierMode'
    waveCont = WT.MakeNodeWaves(omega, nRes = numPoints)
    if (nullspace == []):
        nullspace = np.eye(nh, nh)
        strings = omega.strings
    else:
        N = n
        strings = FixStrings(omega, nullspace)
    if (waveNode != []):
        waveNodes = waveNode @ nullspace
    if (waves == []):
        waveCell = np.asarray([[[] for i in range(N)] for j in range(n)])
    else:
        waveCell = waves @ nullspace
    waveCont = waveCont @ nullspace
    for k in range(N):
        if (waveTrans != []):
            if (k < np.shape(waveTrans)[1]):
                waveTransfer = waveTrans[:, k]
        else:
            waveTransfer = []
        fig = PlotWave(omega, numPoints, X, rescale, waveCell[:, k], waveCont[:, k], waveTrans = waveTransfer)
        if (waveNode != []):
            plt.scatter(x[:], waveNodes[:, k], color = ColorDefault(2), s = 10, zorder = 4)
        plt.xlim([-0.1, 1.25])
        plt.text(1.1, 0, strings[k], fontsize = font)
        plt.show()
        if (save):
            saveString = savePath + saveName + str(k)
            Save(fig, saveString, dpi + '.png', bbox_inches = 'tight', dpi = 600, transparent = True)
#             print('This image has been saved under ' + saveName + '.')
    return


# This function overlays a particular piecewise cell average plot onto a plot of its continuous wave function. It also allows you to save this plot if desired. As the default, that feature is subdued.

# In[8]:


def PlotWave(omega, numPoints, X, rescale, waveCell = [], fX = [], title = '', labels = [], waveTrans = [], sym = True):
    errorLoc = 'ERROR:\nPlotTools:\nPlotWave:\n'
    errorMess = ''
    if (fX != []):
        yMin, yMax, tickHeight = GetYBound(fX, sym)
        numGraphs = np.ndim(fX)
        if (waveCell != []):
            if (numGraphs == 1):
                if (numGraphs != np.ndim(waveCell)):
                    errorMess = 'Dimensions of waveCell and fX do not match!'
            else:
                numGraphs = np.shape(waveCell[0, :])[0]
                if (np.ndim(fX) == 1):
                    errorMess = 'Dimensions of waveCell and fX do not match!'
                else:
                    if (numGraphs != np.shape(fX[0, :])[0]):
                        errorMess = 'Dimensions of waveCell and fX do not match!'
    else:
        if (waveCell != []):
            yMin, yMax, tickHeight = GetYBound(waveCell, sym)
            numGraphs = np.shape(waveCell[0, :])[0]
        else:
            errorMess = 'Must have argument for either fX or waveCell!'
    if (labels != []):
        if (len(labels) != numGraphs):
            errorMess = 'Dimensions of input graph(s) do(es) not match size of labels!'
            sys.exit(errorLoc + errorMess)
        else:
            labelsOut = labels
    else:
        labelsOut = [str(i + 1) for i in range(numGraphs)]
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
    size, tickHeight = Resize(rescale, tickHeight)
    fig, ax = plt.subplots(figsize = size)
    if (waveTrans != []):
        PiecePlot(omega, numPoints, X, waveTrans, color = 3)
    TickPlot(omega, ax, tickHeight)
    if (numGraphs == 1):
        if (fX != []):
            plt.plot(X, fX, color = ColorDefault(0), zorder = 2, label = labelsOut[0])
            pieceLabel = []
        else:
            pieceLabel = labelsOut[0]
        if (waveCell != []):
            PiecePlot(omega, numPoints, X, waveCell, label = pieceLabel)
    else:
        i = 0
        for j in range(numGraphs):
            if (fX != []):
                plt.plot(X, fX[:, j], color = ColorDefault(i), zorder = 2, label = labelsOut[j])
                pieceColor = 3
                pieceLabel = []
            else:
                pieceColor = j
                pieceLabel = labelsOut[j]
            if (waveCell != []):
                PiecePlot(omega, numPoints, X, waveCell[:, j], color = pieceColor, label = pieceLabel)
            i = i + 1
            if (j == 2):
                i = i + 1
        if (labels != []):
            plt.legend()
            print('Are you *sure* your labels are ordered correctly?')
    if (title != ''):
        plt.title(title)
    plt.ylim([yMin, yMax])
    return fig


# This function overlays a piecewise cell average plot of a linear combination of wave vectors onto a plot of its continuous wave function. It also allows you to save this plot if desired. As the default, that feature is subdued.

# In[9]:


def PlotMixedWave(omega, waves, waveCoef, title = '', labels = [], rescale = 1, plotCont = True, sym = False, save = False, saveName = '', dpi = 600):
    nh = omega.nh_max
    numPoints, font, X, savePath = UsefulPlotVals()
    
    if (saveName != ''):
        save = True
    else:
        saveName = 'MixedWave'
    saveString = savePath + saveName
    
    
    numGraphs = np.ndim(waveCoef)
    
    fXCell = waves @ waveCoef
    if (plotCont):
        waveCont = WT.MakeNodeWaves(omega, nRes = numPoints)
        fXCont = waveCont @ waveCoef
    else:
        fXCont = []
    fig = PlotWave(omega, numPoints, X, rescale, fXCell, fXCont, title = title, sym = sym, labels = labels)
    plt.xlim([-0.1, 1.1])
    if (save):
        Save(fig, saveString, dpi)
#         fig.savefig(saveName + '.png', bbox_inches = 'tight', dpi = 600, transparent = True)
#         print('This image has been saved under ' + saveName + '.')
    return


# This function outputs the $y$ limits for a graph along with their respective tick height. `scaleParam` is the percentage of the range which will be neutral space.

# In[11]:


def GetYBound(inputArray, sym, scaleParam = 0.25):
    yMin = np.min(inputArray)
    yMax = np.max(inputArray)
    totRange = yMax - yMin
    if (totRange == 0 and yMin == 0):
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
    degFreed = omega.degFreed# [::-1][0]
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


# This function zips together several vectors so that they may be graphed simultaneously.

# In[13]:


def Load(*vecs):
    errorLoc = 'ERROR:\nPlotTools:\nLoad:\n'
    i = 0
    for vec in vecs:
        i = i + 1
        if (len(vec) != len(vecs[0])):
            if (i % 10 == 1):
                appendage = 'st'
            else:
                if (i % 10 == 2):
                    appendage = 'nd'
                else:
                    if (i % 10 == 3):
                        appendage = 'rd'
                    else:
                        appendage = 'th'
            indexString = str(i) + appendage
            errorMess = '%s vector\'s size does not match size of 1st vector!' %indexString
            sys.exit(errorLoc + errorMess)
    loadedVecs = np.asarray(list(zip(*vecs)))
    return loadedVecs


# In[ ]:

def Resize(rescale, tickHeight):
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
    return size, tickHeight


def PlotGrid(omega, rescale = 1, save = False, saveName = '', dpi = 600):
    numPoints, font, X, savePath = UsefulPlotVals()
    if (saveName != ''):
        save = True
    else:
        saveName = 'Grid'
    saveString = savePath + saveName
    yMin, yMax, tickHeight = GetYBound(0, True)
    size, tickHeight = Resize(rescale, tickHeight)
    fig, ax = plt.subplots(figsize = size)
    TickPlot(omega, ax, tickHeight, label = True)
    plt.ylim([yMin, yMax])
    plt.show()
    if (save):
        Save(fig, saveString, dpi)
    return


def Save(fig, saveString, dpi):
    fig.savefig(saveString + '.png', bbox_inches = 'tight', dpi = dpi, transparent = True)
    print('This image has been saved under ' + saveString + '.')
    return


