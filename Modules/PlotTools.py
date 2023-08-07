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
from Modules import BasicTools as BT
from Modules import OperatorTools as OT
from Modules import WaveTools as WT


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


def DrawLine(xCenter, yCenter, tickHeight, center = True):
    x = xCenter * np.ones(2)
    y = linspace(yCenter, yCenter + tickHeight, num = 2)
    if (center):
        y = y - (tickHeight / 2.)
    return (x, y)


# This function plots a number line marking off the grid onto `ax`.

# In[4]:


def TickPlot(omega, ax, tickHeight, xGrid, yGrid, label = False, u = [], labelsize = 10, linewidth = 1.5, matVis = False):
#     if (enlarge):
#         labelsize = 25
#         linewidth = 4
#     else:
#         labelsize = 10
#         linewidth = 1.5
    ax = plt.axes(frameon = False)
    if (yGrid):
        ax.grid(True, axis = 'y', zorder = 0)
    if (xGrid):
        ax.grid(True, axis = 'x', zorder = 0)
    xAxis = omega.xNode
    yAxis = omega.y
    xCell = omega.xCell
    nh = omega.nh_max
    shiftX = 0.02
    shiftY = tickHeight
    
    if (matVis):
        var = r'F'
        ind = r'h'
    else:
        var = r'v'
        ind = r'j'
    
    
    if (u != []):
        label = False
        xAxis = xAxis[1:4]
        yAxis = yAxis[1:4]
        shiftX = shiftX / 4
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
        ax.plot(xs, ys, color = 'k', zorder = 1, linewidth = linewidth)
        if (label):
            print(i)
            if ((i < 3) or (i > nh - 2)):
                prestring = r'$j = $'
                istring = prestring + str(i)
                shiftExtra = 2 * shiftX
                if (i == nh - 1):
                    istring = prestring + r'$n - 1$'
                    shiftExtra = 4 * shiftX
                if (i == nh):
                    istring = prestring + r'$n$'
                    shiftExtra = shiftX
                plt.text(xi - shiftX - shiftExtra, yi - (1.5 * shiftY), istring, fontsize = 12)
        if ( u != []):
            if (i == 0):
                color = 2
                topString = r'$' + var + r'_{' + ind + r' - 1}$'
                botString = r'$x_{' + ind + r' - 1}$'
                midString = r'$\left<x\right>_{' + ind + r' - 1}$'
                if (matVis):
                    midString2 = r'$\left<x\right>_{' + ind + r' - 2}$'
                    plt.text(xCell[i] - shiftX, yi - shiftY, midString2, fontsize = 12)
            else:
                if (i == 1):
                    if (matVis):
                        color = 0
                    shiftX = shiftX / 2
                    topString = r'$' + var + r'_{' + ind + r'}$'
                    botString = r'$x_{' + ind + r'}$'
                    midString = r'$\left<x\right>_{' + ind + r'}$'
                else:
                    color = 2
                    shiftX = 2 * shiftX
                    topString = r'$' + var + r'_{' + ind + r' + 1}$'
                    botString = r'$x_{' + ind + r' + 1}$'
                    midString = r'$\left<x\right>_{' + ind + r' + 1}$'
            (xs, ys) = DrawLine(xi, yi, u[i + 1], center = False)
            ax.plot(xs, ys, color = ColorDefault(color), zorder = 2, linestyle = ':')
            plt.text(xi - shiftX, u[i + 1] + (shiftY / 2), topString, fontsize = 12)
            plt.text(xi - shiftX, yi - shiftY, botString, fontsize = 12)
            plt.text(xCell[i + 1] - shiftX, yi - shiftY, midString, fontsize = 12)
        i = i + 1
    if (u == []):
        ax.plot(xAxis, yAxis, color = 'k', zorder = 0, linewidth = linewidth)
    plt.tick_params(axis = 'x', which = 'both', bottom = False, top = False, labelbottom = xGrid, labelsize = labelsize)
    plt.tick_params(axis = 'y', which = 'both', left = False, right = False, labelleft = yGrid, labelsize = labelsize)
    return


# This function plots out the piecewise cell averages.

# In[5]:


def PiecePlot(omega, numPoints, X, pieces, color = 3, label = [], linestyle = '-', tickHeight = 0, linewidth = 1.5, matVis = False):
    errorLoc = 'ERROR:\nPlotTools:\nPiecePlot:\n'
    errorMess = BT.CheckSize(numPoints, X, nName = 'numPoints', matricaName = 'X')
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
    x = omega.xNode
    xCell = omega.xCell
    n = len(x) - 1
    
    if (tickHeight != 0):
        label = []
        n = 4
        shiftX = 0.005
        shiftY = tickHeight / 2
    cellVals = np.ones(numPoints, float)
    lowIndex = 0
    
    if (matVis):
        var1 = r'\phi_{1}'
        var2 = r'\phi_{2}'
        ind = r'h'
    else:
        var1 = r'v'
        var2 = r'v'
        ind = r'j'
    
    for k in range(n):
        highIndex = np.where(X <= x[k + 1])[0][::-1][0] + 1
        cellVals[lowIndex:highIndex] = pieces[k] * cellVals[lowIndex:highIndex]
        if ((k == 0) and (label != [])):
            plt.plot(X[lowIndex:highIndex], cellVals[lowIndex:highIndex], color = ColorDefault(color), linestyle = linestyle, zorder = 3, label = label, linewidth = linewidth)
        else:
            if ((k != 0) or (tickHeight == 0) or matVis):
                plt.plot(X[lowIndex:highIndex], cellVals[lowIndex:highIndex], color = ColorDefault(color), linestyle = linestyle, zorder = 3, linewidth = linewidth)
            if (tickHeight != 0): # ((k != 0) and (tickHeight != 0)):
                if (k == 0):
                    if (matVis):
                        topString = r'$\left<' + var1 + r'\right>_{' + ind + r'- 2}$'
                    else:
                        topString = ''
                else:
                    if (k == 1):
                        topString = r'$\left<' + var1 + r'\right>_{' + ind + r' - 1}$'
                    else:
                        if (k == 2):
                            shiftX = shiftX / 2
                            topString = r'$\left<' + var2 + r'\right>_{' + ind + r'}$'
                        else:
                            shiftX = 2 * shiftX
                            topString = r'$\left<' + var2 + r'\right>_{' + ind + r' + 1}$'
                plt.text(xCell[k] - shiftX, pieces[k] + shiftY, topString, fontsize = 12)
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


def PlotWaves(omega, physics, waves = [], waveNode = [], nullspace = [], waveTrans = [], ct = 0, save = False, saveName = '', rescale = 1, dpi = 600, enlarge = False):
    nh = omega.nh_max
    x = omega.xNode
    n = omega.degFreed
    alias = omega.alias
    nh = int(alias * nh)
    N = nh
    numPoints, font, X, savePath = UsefulPlotVals()
    if (saveName != ''):
        save = True
    else:
        saveName = 'FourierMode'
    waveCont = WT.MakeNodeWaves(omega, nRes = numPoints)
    if (ct != 0):
        omega2 = BT.Grid(nh)
        rotMat = OT.MakeRotMat(omega2, ct)
        shift = True
    else:
        rotMat = np.eye(nh, nh)
        shift = False
    strings = FixStrings(omega, nullspace, shift)
    if (nullspace == []):
        nullspace = np.eye(nh, nh)
#         strings = omega.strings
    else:
        N = n
    
    
    if (waveNode != []):
        waveNodes = waveNode @ rotMat @ nullspace
    if (waves == []):
        waveCell = np.asarray([[[] for i in range(N)] for j in range(n)])
    else:
        waveCell = waves @ nullspace
    waveCont = waveCont @ rotMat @ nullspace
    for k in range(N):
        if (waveTrans != []):
            if (k < np.shape(waveTrans)[1]):
                waveTransfer = waveTrans[:, k]
        else:
            waveTransfer = []
        fig = PlotWave(omega, physics, numPoints, X, rescale, waveCell[:, k], waveCont[:, k], waveTrans = waveTransfer, xGrid = False, yGrid = False)
        if (waveNode != []):
            plt.scatter(x[:], waveNodes[:, k], color = ColorDefault(2), s = 10, zorder = 4)
        plt.xlim([-0.1, 1.25])
        plt.text(1.1, 0, strings[k], fontsize = font)
        plt.show()
        if (save):
            saveString = savePath + saveName + str(k)
            Save(fig, saveString, dpi)
    return


# This function overlays a particular piecewise cell average plot onto a plot of its continuous wave function. It also allows you to save this plot if desired. As the default, that feature is subdued.

# In[8]:


def PlotWave(omega, physics, numPoints, X, rescale, waveCell = [], fX = [], title = '', labels = [], waveTrans = [], sym = True, xGrid = False, yGrid = False, enlarge = False):
    errorLoc = 'ERROR:\nPlotTools:\nPlotWave:\n'
    errorMess = ''
    
    if (enlarge):
        linewidth = 4
        fontsize = 35
        labelsize = 25
    else:
        linewidth = 1.5
        fontsize = 25
        labelsize = 10
    
    if (fX != []):
        yMin, yMax, tickHeight = GetYBound(fX, sym)
        numGraphs = np.ndim(fX)
        if (waveCell is not []):
            if (numGraphs == 1):
                if (numGraphs != np.ndim(waveCell)):
                    errorMess = 'Dimensions of waveCell and fX do not match!'
            else:
                if ((numGraphs == 2) and (np.shape(fX)[1] == 1)):
                    print((numGraphs == 2) and (np.shape(fX)[1] == 1))
                    numGraphs = 1
                    fX = fX[:, 0]
                    waveCell = waveCell[:, 0]
#                     print('')
#                     print(fX)
#                     print('')
#                     print(waveCell)
#                     print('')
                        
                else:
                    numGraphs = np.shape(waveCell[0, :])[0]
                    if (np.ndim(fX) == 1):
                        errorMess = 'Dimensions of waveCell and fX do not match!'
                    else:
                        if (numGraphs != np.shape(fX[0, :])[0]):
                            errorMess = 'Dimensions of waveCell and fX do not match!'
    else:
#         print('Here is the info you seek.')
#         print('shape of wavecell:', np.shape(waveCell))
#         print('shape of shape:', np.shape(np.shape(waveCell)))
#         print('index you want:', np.shape(waveCell[0, :])[0])
        waveCellDim = np.shape(waveCell)
        if (np.shape(waveCellDim)[0] == 1):
            numGraphs = 1
        else:
            if (np.shape(waveCellDim)[0] == 2):
                numGraphs = waveCellDim[1]
            else:
                errorMess = 'The rank of waveCell is too high!'
#         numGraphs = np.shape(waveCell[0, :])[0]
        if (waveCell != []):
            yMin, yMax, tickHeight = GetYBound(waveCell, sym)
        else:
            errorMess = 'Must have argument for either fX or waveCell!'
    if (labels != []):
        if (len(labels) != numGraphs):
            errorMess = 'Dimensions of input graph(s) do(es) not match size of labels!'
            print('labels:', len(labels))
            print('graphs:', numGraphs)
            sys.exit(errorLoc + errorMess)
        else:
            labelsOut = labels
    else:
        labelsOut = [str(i + 1) for i in range(numGraphs)]
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
    size, tickHeight, labelfont = Resize(rescale, tickHeight)
    fig, ax = plt.subplots(figsize = size)
    if (waveTrans != []):
        PiecePlot(omega, numPoints, X, waveTrans, color = 3, linewidth = linewidth)
    TickPlot(omega, ax, tickHeight, xGrid, yGrid, linewidth = linewidth, labelsize = labelsize)
    if (numGraphs == 1):
        if (fX != []):
            plt.plot(X, fX, color = ColorDefault(0), zorder = 2, label = labelsOut[0], linewidth = linewidth) # Fuck with this when you have time to worry about the line thickness of the analytic solution for a single plot.
            pieceLabel = []
        else:
            pieceLabel = labelsOut[0]
        if (waveCell != []):
            PiecePlot(omega, numPoints, X, waveCell, label = pieceLabel, linewidth = linewidth)
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
                PiecePlot(omega, numPoints, X, waveCell[:, j], color = pieceColor, label = pieceLabel, linewidth = linewidth)
            i = i + 1
            if (j == 2):
                i = i + 1
        if (labels != []):
            plt.legend(fontsize = labelfont)
            print('Are you *sure* your labels are ordered correctly?')
    if (title != ''):
        plt.title(title, fontsize = fontsize)
    locs = physics.locs
    for loc in locs:
        locx = loc * np.ones(2)
        locy = np.linspace(yMin, yMax, num = 2)
        plt.plot(locx, locy, color = 'k', zorder = 1, linewidth = linewidth)
    plt.ylim([yMin, yMax])
    return fig


# This function overlays a piecewise cell average plot of a linear combination of wave vectors onto a plot of its continuous wave function. It also allows you to save this plot if desired. As the default, that feature is subdued.

# In[9]:


def PlotMixedWave(omega, physics, FCoefs, waves = [], title = '', labels = [], rescale = 1, plotCont = True, sym = False, save = False, saveName = '', dpi = 600, ct = 0, xGrid = False, yGrid = False, enlarge = False):
    errorLoc = 'ERROR:\nPlotTools:\nPlotMixedWave:\n'
    nh = omega.nh_max
    degFreed = omega.degFreed
    numPoints, font, X, savePath = UsefulPlotVals()
    lenFCoefs = np.shape(FCoefs)[0]
    if (lenFCoefs % nh != 0):
        errorMess = 'FCoefs must have length which is integer multiple of nh_max! Currently, FCoefs is ' + str(lenFCoefs) + ' long, and nh_max is ' + str(nh) + '!'
    else:
        errorMess = ''
        numPlots = int(lenFCoefs / nh)
    if (errorMess != ''):
        sys.exit(errorLoc + errorMess)
    
    
    
    if (saveName != ''):
        save = True
    else:
        saveName = 'MixedWave'
    saveString1 = savePath + saveName
    
    numGraphs = np.ndim(FCoefs)
    
    if (plotCont):
        waveCont = WT.MakeNodeWaves(omega, nRes = numPoints)
        if (ct != 0):
            omega2 = BT.Grid(nh)
            rotMat = OT.MakeRotMat(omega2, ct)
            waveCont = waveCont @ rotMat
    
    for k in range(numPlots):
        if (k == 0):
            title1 = title + r' $E$ Field'
        else:
            title1 = title + r' $B$ Field'
        if (numGraphs == 1):
            FCoef = FCoefs[k * nh:(k + 1) * nh]
            title1 = title
        else:
            FCoef = FCoefs[k * nh:(k + 1) * nh, :]
            
        if (waves != []):
            fXCell = waves[:nh, :nh] @ FCoef
        else:
            fXCell = []
        if (plotCont):
            fXCont = waveCont @ FCoef
        else:
            fXCont = []
        fig = PlotWave(omega, physics, numPoints, X, rescale, fXCell, fXCont, title = title1, sym = sym, labels = labels, xGrid = xGrid, yGrid = yGrid, enlarge = enlarge)
        plt.xlim([-0.1, 1.1])
        if (save):
            if (numPlots > 1):
                if (k == 0):
                    extraPiece = 'E'
                else:
                    extraPiece = 'B'
                saveString2 = saveString1
                saveString = saveString2 + extraPiece
            else:
                saveString = saveString1
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


def FixStrings(omega, nullspace, shift):
    errorLoc = 'ERROR:\nPlotTools:\nFixStrings:\n'
    strings = omega.strings
    degFreed = omega.degFreed# [::-1][0]
    nh = omega.nh_max
    alias = omega.alias
    if (nullspace == []):
        N = int(alias * nh)
        location = np.arange(N)
        locations = [np.asarray(location), np.asarray(location)]
    else:
        errorMess = BT.CheckSize(degFreed, nullspace[0, :], nName = 'degFreed', matricaName = 'nullspace')
        if (errorMess != ''):
            sys.exit(errorLoc + errorMess)
        N = degFreed
        locations = np.where(nullspace != 0)
    stringsNew = ['' for i in range(N)]
    if (shift):
        x = '$(x - c t)$'
    else:
        x = '$x$'
    j = 0
    for i in locations[1]:
        if (stringsNew[i] == ''):
            if ((i == 0) and (j == 0)):
                stringsNew[i] = strings[locations[0][j]]
            else:
                stringsNew[i] = strings[locations[0][j]] + x
        else:
            stringsNew[i] = stringsNew[i] + '+' + strings[locations[0][j]] + x
        j = j + 1
    return stringsNew


# This function zips together several vectors so that they may be graphed simultaneously.

# In[13]:


def Load(*vecs):
    errorLoc = 'ERROR:\nPlotTools:\nLoad:\n'
    i = 0
    if (len(vecs) == 1):
        for vec in vecs:
            loadedVecs = vec
    else:
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
    errorLoc = 'ERROR:\nPlotTools:\nResize:\n'
    labelfont = 10
    if np.any(np.asarray(rescale) <= 0):
        errorMess = 'All values of rescale must be greater than 0!'
        sys.exit(errorLoc + errorMess)
    if (np.shape(rescale) == ()):
        size = [5 * rescale, 2.5 * rescale]
        tickHeight = tickHeight / rescale
        labelfont = int(rescale) + labelfont
    else:
        if (np.shape(rescale) == (2,)):
            size = [5 * rescale[0], 2.5 * rescale[1]]
            tickHeight = tickHeight / rescale[1]
            labelfont = int(min(rescale)) + labelfont
        else:
            errorMess = 'Invalid shape of rescale object entered!'
            sys.exit(errorLoc + errorMess)
    return size, tickHeight, labelfont


def PlotGrid(omega, rescale = 1, save = False, saveName = '', dpi = 600, enlarge = False):
    numPoints, font, X, savePath = UsefulPlotVals()
    if (saveName != ''):
        save = True
    else:
        saveName = 'Grid'
    
    if (enlarge):
        linewidth = 4
        fontsize = 35
        labelsize = 25
    else:
        linewidth = 1.5
        fontsize = 25
        labelsize = 10
    
    saveString = savePath + saveName
    yMin, yMax, tickHeight = GetYBound(0, True)
    size, tickHeight, labelfont = Resize(rescale, tickHeight)
    fig, ax = plt.subplots(figsize = size)
    TickPlot(omega, ax, tickHeight, False, False, label = True, labelsize = labelsize, linewidth = linewidth)
    plt.ylim([yMin, yMax])
    plt.show()
    if (save):
        Save(fig, saveString, dpi)
    return


def Save(fig, saveString, dpi):
    fig.savefig(saveString + '.png', bbox_inches = 'tight', dpi = dpi, transparent = True)
    print('This image has been saved under ' + saveString + '.')
    return

def DivergVis(save = False, saveName = '', dpi = 600, enlarge = False, matVis = False):
    if (saveName != ''):
        save = True
    else:
        saveName = 'DivergenceVisual'
    
    if (enlarge):
        linewidth = 4
        fontsize = 35
        labelsize = 25
    else:
        linewidth = 1.5
        fontsize = 25
        labelsize = 10
    
    nh = 32
    omega = BT.Grid(nh)
    h = omega.h[0]
    x = omega.xNode
    length = 2 * h
    k = 2
    Cosine = lambda x: np.cos(2. * np.pi * k * x)
    Sine = lambda x: np.sin(2. * np.pi * k * x)
    factor = 1. / (2 * pi * k * h)
    uNode = Sine(x)
    uCell = factor * (Cosine(x[:-1]) - Cosine(x[1:]))
    xCell = omega.xCell
    fig, ax = plt.subplots()
    numPoints, font, X, savePath = UsefulPlotVals()
    yMin, yMax, tickHeight = GetYBound(uNode[1:4], False)
    TickPlot(omega, ax, tickHeight, False, False, u = uNode, labelsize = labelsize, linewidth = linewidth, matVis = matVis)
    if (matVis):
        plt.scatter(x[1], uNode[1], s = 10, color = ColorDefault(2))
        plt.scatter(x[2], uNode[2], s = 10, color = ColorDefault(0))
        plt.scatter(x[3], uNode[3], s = 10, color = ColorDefault(2))
    else:
        plt.scatter(x[1:4], uNode[1:4], s = 10, color = ColorDefault(2))
    PiecePlot(omega, numPoints, X, uCell, tickHeight = tickHeight, linewidth = linewidth, matVis = matVis)
    plt.quiver([length], [0], [length], [0], color = ['k', 'k'], angles = 'xy', scale_units = 'xy', scale = 1, width = 0.005, headwidth = 8, headlength = 8)
    plt.quiver([length], [0], [-length], [0], color = ['k', 'k'], angles = 'xy', scale_units = 'xy', scale = 1, width = 0.005, headwidth = 8, headlength = 8)
    plt.xlim([-0.1 * length, 2.1 * length])
    plt.ylim([yMin, yMax])
    plt.show()
    if (save):
        saveString = savePath + saveName
        Save(fig, saveString, dpi)
    return


