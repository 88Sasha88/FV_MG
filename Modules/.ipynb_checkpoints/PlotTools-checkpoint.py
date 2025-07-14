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
    if (k == 0.5):
        color = 'k'
    else:
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


def TickPlot(omega, ax, tickHeight, xGrid, yGrid, label = False, u = [], labelsize = 10, linewidth = 1.5, matVis = False, fill = False, var = var, ghost = '', something = True):
#     if (enlarge):
#         labelsize = 25
#         linewidth = 4
#     else:
#         labelsize = 10
#         linewidth = 1.5
    fontsize = 12
    # ax = plt.axes(frameon = False # THIS WAS REMOVED AFTER PYTHON UPDATE!!!
    if (yGrid):
        ax.grid(True, axis = 'y', zorder = 0)
    if (xGrid):
        ax.grid(True, axis = 'x', zorder = 0)
    ax.set_axisbelow(True)
    # fig.canvas.draw()
    xAxis = omega.xNode
    yAxis = omega.y

    xCell = omega.xCell
    nh = omega.nh_max
    degFreed = omega.degFreed
    levels = omega.levels
    bigN = len(omega.cells[-1])
    shiftX = 0.025
    shiftY = tickHeight
    extraShift = 0
    
    if (matVis != ''):
        ind = r'h'
    else:
        ind = r'j'
    color = 2
    
    
    if (not BT.Empty(u)):
        print('IT\'S HAPPENING!!!')
        level = ''
        if (ghost != ''):
            level = r'^{(l - 1)}'
            if (ghost == 'G1'):
                n = 6
            else:
                if (ghost == 'G2'):
                    n = 7
                else:
                    n = 5
            
        else:
            n = 4
        label = False
        xAxis = xAxis[1:n + 1]
        yAxis = yAxis[1:n + 1]
        shiftX = shiftX / 4
    i = 0
    j = 0
    
    LS = ':'
    
    L1R1bot = 0
    scootch = 0
    side = ''

    if ((matVis == 'R1') or (matVis == 'R2')):
        print('CHECK 1')
        if (var == r'L_{1}'):
            var = 'L'
        else:
            var = 'R'
        side = '1'
        if (matVis == 'R2'):
            side = '2'
        print('CHECK 2')
    else:
        if ((matVis == 'L1') or (matVis == 'L2')):
            if (matVis == 'L2'):
                if (var == r'R_{2}'):
                    var = 'R'
                else:
                    var = 'L'
                side = '2'
            else:
                var = 'L'
                side = '1'
    print('CHECK 3')
    
    for (xi, yi) in zip(xAxis, yAxis):
        j = j + 1
        if ((xi == 0) or (xi == 1)):
            height = tickHeight
            shiftY = tickHeight
            if ((label) and (levels == 0)):
                plt.text(xi - shiftX/2, yi + shiftY, int(xi), fontsize = fontsize)
        else:
            height = tickHeight / 2
#             print('height:', height)
        if (((j != 1) or (matVis != 'L2')) and ((j != 4) or (matVis != 'R1'))):
            (xs, ys) = DrawLine(xi, yi, height)
#             if (((j == 4) and (matVis == 'L1')) or ((j == 1) and (matVis == 'R2')) or ((j == 3) and (ghost == 'G2'))):
# #                 ax.plot(xs, ys, color = ColorDefault(1), zorder = 0, linewidth = linewidth)
#                 ax.plot(xs, ys, color = 'k', zorder = 2, linewidth = linewidth, linestyle = '--', dashes = [2.75, 2.5], dash_capstyle = 'projecting')
#             else:
            ax.plot(xs, ys, color = 'k', zorder = 2, linewidth = linewidth)
            
        if (label):
            if (levels == 0):
                imax = 3
                imin = degFreed - 2
            else:
                imax = 5
                imin = degFreed - 1
            if ((i < imax) or (i > imin)):
                prestring = r'$j = $'
                istring = prestring + str(i)
                shiftExtra = shiftX
                if (levels > 0):
                    if (i != 0):
                        if (i == imax - 1):
                            istring = prestring + r'$r N^{(1)}$'
                        else:
                            if (i == degFreed):
                                istring = prestring + r'$\overline{n}^{(1)}$'
                            else:
                                istring = ''
                else:
                    if (i == degFreed - 1):
                        istring = prestring + r'$n - 1$'
                        shiftExtra = 3 * shiftX
                    if (i == degFreed):
                        istring = prestring + r'$n$'
                        shiftExtra = -0.5 * shiftX
                        # shiftExtra = shiftX
                plt.text(xi - shiftX - shiftExtra, yi - (1.5 * shiftY), istring, fontsize = fontsize)
        if (not BT.Empty(u)):
            if (i == 0):
                color = 2
                topString = r'$' + var + r'_{' + side + ind + r' - 1}$'
                if (matVis == 'L2'):
                    botString = ''
                    topString = ''
                else:
                    if ((matVis == 'R1') or (matVis == 'L1') or (ghost != '')):
                        L1R1bot = -0.001
                        botString = r'$x_{' + ind + r' - 2}' + level + r'$'
                        topString = r'$' + var + r'_{' + side + ind + r' - 2}$'
                    else:
                        botString = r'$x_{' + ind + r' - 1}$'
                        if (matVis == 'R2'):
                            topString = r'$' + var + r'^{*}_{' + side + ind + r' - 1}$'
                if (fill or (matVis != '')):
                    if (matVis == 'L2'):
                        midString = ''
                    else:
                        if ((matVis == 'R1') or (matVis == 'L1')):
                            extraShift = 0.002
                            midString = r'$\Delta x_{' + ind + r' - 2}$'
                        else:
                            midString = r'$\Delta x_{' + ind + r' - 1}$'
                else:
                    midString = r'$\left<x\right>_{' + ind + r' - 1}$'
            else:
                if (i == 1):
                    
                    if ((matVis == 'R2') or (matVis == 'L2')):
                        LS = '-'
                        color = 0.5
                    shiftX = shiftX / 3
                    extraShift = 0.002
                    topString = r'$' + var + r'_{' + side + ind + r'}$'
                    if ((matVis == 'R1') or (matVis == 'L1') or (ghost != '')):
                        extraShift = extraShift + 0.002
                        L1R1bot = -0.004
                        if (ghost != ''):
                            L1R1bot = -0.005
                        if (ghost == 'G2'):
                            level = r'^{(l)}'
                            botString = r'$x_{2' + ind + r' - 2}' + level + r'$'
                        else:
                            botString = r'$x_{' + ind + r' - 1}' + level + r'$'
                    else:
                        botString = r'$x_{' + ind + r'}$'
                
                    if (fill or (matVis != '')):
                        if ((matVis == 'R1') or (matVis == 'L1')):
                            midString = r'$\Delta x_{' + ind + r' - 1}$'
                            topString = r'$' + var + r'_{' + side + ind + r' - 1}$'
                        else:
                            midString = r'$\Delta x_{' + ind + r'}$'
                    else:
                        midString = r'$\left<x\right>_{' + ind + r'}$'
                else:
#                     color = 2
                    extraShift = 0
                    topString = r'$' + var + r'_{' + side + ind + r' + 1}$'
                    if (i == 2):
                        shiftX = 3 * shiftX
                        if ((matVis == 'R1') or (matVis == 'L1')):# or (ghost == 'G1')):
                            LS = '-'
                            color = 0.5
                            L1R1bot = 0.004
                            botString = r'$x_{' + ind + r'}$'
                        else:
                            botString = r'$x_{' + ind + r' + 1}$'
                        
                        if (ghost != ''):
                            if (ghost == 'G2'):
                                botString = r'$x_{2' + ind + r' - 1}' + level + r'$'
                            else:
                                LS = '-.'
                                L1R1bot = 0.001
                                if (ghost == 'G1'):
                                    # L1R1bot = 0.001
                                    level = r'^{(l)}'
                                    botString = r'$x_{2' + ind + r'}' + level + r'$'
                            
                                else:
                                    botString = r'$x_{' + ind + r'}' + level + r'$'
                        
                        
                        if (fill or (matVis != '')):
                            if (matVis == 'R1'):
                                midString = ''
                                topString = r'$' + var + r'_{' + side + ind + r'}$'
                            else:
                                if (matVis == 'L1'):
                                    midString = r'$\Delta x_{' + ind + r'}$'
                                    topString = r'$' + var + r'_{' + side + ind + r'}$'
                                else:
                                    midString = r'$\Delta x_{' + ind + r' + 1}$'
                        else:
                            midString = r'$\left<x\right>_{' + ind + r' + 1}$'
                            
                    if (i == 3):
                        shiftX = (2 * shiftX) / 5
                        scootch = -0.002
                        topString = r'$' + var + r'_{' + side + ind + r' + 2}$'
                        if (matVis == 'R1'):
                            L1R1bot = -0.003
                            botString = ''
                            topString = ''
                        else:
                            if (matVis == 'L1'):
                                L1R1bot = -0.003
                                botString = r'$x_{' + ind + r' + 1}$'
                                topString = r'$' + var + r'^{*}_{' + side + ind + r' + 1}$'
                            else:
                                L1R1bot = 0#-0.001
                                botString = r'$x_{' + ind + r' + 2}$'
                        if (ghost != ''):
                            level = r'^{(l)}'
                            if (ghost == 'G1'):
                                L1R1bot = -0.0055
                                botString = r'$x_{2' + ind + r' + 1}' + level + r'$'
                            else:
                                L1R1bot = -0.00275
                                if (ghost == 'G2'):
                                    botString = r'$x_{2' + ind + r'}' + level + r'$'
                                    LS = '-.'
                                else:
                                    L1R1bot = -0.004
                                    botString = r'$x_{2' + ind + r' + 2}' + level + r'$'

                        midString = ''
                    if (i == 4):
                        # L1R1bot = -0.001
                        if (ghost == 'G1'):
                            L1R1bot = -0.004
                            botString = r'$x_{2' + ind + r' + 2}' + level + r'$'
                        else:
                            if (ghost == 'G2'):
                                L1R1bot = -0.0055
                                botString = r'$x_{2' + ind + r' + 1}' + level + r'$'
                            else:
                                L1R1bot = -0.002
                                botString = r'$x_{2' + ind + r' + 3}' + level + r'$'
                    if (i == 5):
                        if (ghost == 'G1'):
                            L1R1bot = -0.002
                            botString = r'$x_{2' + ind + r' + 3}' + level + r'$'
                        else:
                            L1R1bot = -0.004
                            botString = r'$x_{2' + ind + r' + 2}' + level + r'$'
                    if (i == 6):
                        L1R1bot = -0.002
                        botString = r'$x_{2' + ind + r' + 3}' + level + r'$'
            if (not fill):
                if (((i != 0) or (matVis != 'L2')) and ((i != 3) or (matVis != 'R1')) and something):
                    (xs, ys) = DrawLine(xi, yi, u[i + 1], center = False)
                    ax.plot(xs, ys, color = ColorDefault(color), zorder = 1, linestyle = LS)
                    color = 2
                LS = ':'
                if ((ghost == '') and something):
                    plt.text(xi - shiftX + scootch + L1R1bot, u[i + 1] + (shiftY / 3), topString, fontsize = fontsize, zorder = 6)
                plt.text(xi - shiftX + L1R1bot, yi - 0.85 * shiftY, botString, fontsize = fontsize)
                print('iter =', i, 'xLoc =', xi - shiftX + L1R1bot, L1R1bot)
                L1R1bot = 0
            if ((i < 3) and (ghost == '')):
                plt.text(xCell[i + 1] - shiftX - extraShift, yi - 0.85 * shiftY, midString, fontsize = fontsize)
        i = i + 1
    if (BT.Empty(u)):
        ax.plot(xAxis, yAxis, color = 'k', zorder = 2, linewidth = linewidth)
    plt.tick_params(reset = True, axis = 'x', which = 'both', bottom = False, top = False, labelbottom = xGrid, labelsize = labelsize)
    plt.tick_params(reset = True, axis = 'y', which = 'both', left = False, right = False, labelleft = yGrid, labelsize = labelsize)
    ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False) # THIS WAS ADDED AFTER PYTHON UPDATE!!!
    if (xGrid): # THIS WAS ADDED AFTER PYTHON UPDATE!!!
        ax.grid(visible = xGrid, zorder = -1, axis = 'x') # THIS WAS ADDED AFTER PYTHON UPDATE!!!
    if (yGrid): # THIS WAS ADDED AFTER PYTHON UPDATE!!!
        ax.grid(visible = yGrid, zorder = -1, axis = 'y') # THIS WAS ADDED AFTER PYTHON UPDATE!!!
    return


# This function plots out the piecewise cell averages.

# In[5]:


def PiecePlot(omega, numPoints, X, pieces, color = 3, label = [], linestyle = '-', tickHeight = 0, linewidth = 1.5, matVis = False, fill = False, var = '', ghost = ''):
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
        shiftY = tickHeight / 3
        level = ''
        levShift = 0.003
        if (ghost != ''):
            n = 6
            level = r'^{(l - 1)}'
            levShift = 0.01#75
        if (ghost == 'G2'):
            n = 7
        else:
            if (ghost == 'G3'):
                n = 5
    cellVals = np.ones(numPoints, float)
    lowIndex = 0
    fontsize = 12
    LS = linestyle
    
#     if (matVis):
#         var1 = r'\phi_{1}'
#         var2 = r'\phi_{2}'
#         ind = r'h'
#     else:
#         var1 = r'v'
#         var2 = r'v'
#         if (fill):
#             var1 = r'u'
#             var2 = r'u'
    if (matVis == ''):
        ind = r'j'
    else:
        ind = r'h'
    
    
    for k in range(n):
        # print('iter:', k)
        highIndex = np.where(X <= x[k + 1])[0][::-1][0] + 1
        cellVals[lowIndex:highIndex] = pieces[k] * cellVals[lowIndex:highIndex]
        if ((k == 0) and (not BT.Empty(label))):
            plt.plot(X[lowIndex:highIndex], cellVals[lowIndex:highIndex], color = ColorDefault(color), linestyle = LS, zorder = 3, label = label, linewidth = linewidth)
        else:
            if ((k != 0) or (tickHeight == 0)): # or (matVis != '')):
                if (fill):
                    plt.fill_between(X[lowIndex-1:highIndex], 0, cellVals[lowIndex-1:highIndex], color = ColorDefault(color), alpha = 0.1)
                if (tickHeight != 0):
                    if (k == 1):
                        if (matVis == 'L2'):
                            topString = ''
                            LS = ''
                        else:
                            if (matVis == 'R2'):
                                topString = r'$\left<' + var + r'^{*}\right>_{' + ind + r' - 1}$'
                                LS = '--'
                            else:
                                if ((matVis == 'L1') or (matVis == 'R1') or (ghost != '')):
                                    topString = r'$\left<' + var + level + r'\right>_{' + ind + r' - 2}$'
                                else:
                                    topString = r'$\left<' + var + r'\right>_{' + ind + r' - 1}$'
                    else:
                        if (k == 2):
                            if ((matVis == 'L1') or (matVis == 'R1') or (ghost != '')):
                                if (ghost != ''):
                                    levShift = 0.009
                                if (ghost == 'G2'):
                                    shiftX = 1.5 * shiftX
                                    shiftY = shiftY - 0.015
                                    level = r'^{(l) *}'
                                    levShift = 0.15 * levShift# / 5
                                    topString = r'$\left<' + var + level + r'\right>_{2' + ind + r' - 2}$' # I can't star this probably because of level.
                                    LS = '--'
                                else:
                                    topString = r'$\left<' + var + level + r'\right>_{' + ind + r' - 1}$'
                            else:
                                levShift = 0.002
                                shiftX = shiftX / 2
                                topString = r'$\left<' + var + r'\right>_{' + ind + r'}$'
                        else:
                            if (k == 3):
                                if (matVis != 0):
                                    levShift = 0.002
                                if (matVis == 'R1'):
                                    topString = ''
                                    LS = ''
                                else:
                                    if (matVis == 'L1'):
                                        levShift = 0.003
                                        shiftX = shiftX / 2
                                        topString = r'$\left<' + var + r'^{*}\right>_{' + ind + r'}$'
                                        LS = '--'
                                    else:
                                        if (ghost != ''):
                                            if (ghost == 'G2'):
                                                levShift = levShift / 200
                                                shiftY = shiftY + 0.015
                                                level = r'^{(l) *}'
                                                topString = r'$\left<' + var + level + r'\right>_{2' + ind + r' - 1}$' # I can't star this probably because of level.
                                                LS = '--'
                                                level = r'^{(l)}'
                                            else:
                                                if (ghost == 'G1'):
                                                    level = r'^{(l)}'
                                                    levShift = 0.003
                                                    topString = r'$\left<' + var + level + r'\right>_{2' + ind + r'}$'
                                                else:
                                                    levShift = 0.005
                                                    topString = r'$\left<' + var + level + r'\right>_{' + ind + r'}$'
                                        else:
                                            shiftX = 2 * shiftX
                                            topString = r'$\left<' + var + r'\right>_{' + ind + r' + 1}$'
                            if (k == 4):
#                                 shiftX = 1.5 * shiftX
                                if (ghost == 'G2'):
                                    shiftX = shiftX / 1.5
                                    levShift = 0.003
                                    topString = r'$\left<' + var + level + r'\right>_{2' + ind + r'}$'
                                else:
                                    if (ghost == 'G3'):
                                        level = r'^{(l)}'
                                        levShift = 0.005
                                        topString = r'$\left<' + var + level + r'\right>_{2' + ind + r' + 2}$'
                                    else:
                                        levShift = 0.005
                                        topString = r'$\left<' + var + level + r'\right>_{2' + ind + r' + 1}$'
                            if (k == 5):
                                if (ghost == 'G1'):
                                    topString = r'$\left<' + var + level + r'\right>_{2' + ind + r' + 2}$'
                                else:
                                    if (ghost == 'G2'):
                                        levShift = 0.005
                                    topString = r'$\left<' + var + level + r'\right>_{2' + ind + r' + 1}$'
                            if (k == 6):
                                topString = r'$\left<' + var + level + r'\right>_{2' + ind + r' + 2}$'
                    plt.text(xCell[k] - shiftX - levShift, pieces[k] + shiftY, topString, fontsize = fontsize, zorder = 5)
                    # print('xLoc =', xCell[k] - shiftX - levShift, levShift)
#                     print(k, xCell[k] - shiftX - levShift, topString)#, pieces[k])
                plt.plot(X[lowIndex:highIndex], cellVals[lowIndex:highIndex], color = ColorDefault(color), linestyle = LS, zorder = 3, linewidth = linewidth)
                LS = linestyle
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


def PlotWaves(omega, physics, waves = [], waveNode = [], nullspace = [], waveTrans = [], ct = 0, save = False, saveName = '', rescale = 1, dpi = 600, enlarge = False, alias = False):
    warnLoc = 'WARNING:\nPlotTools:\nPlotWaves:\n'
    nh = omega.nh_max
    x = omega.xNode
    n = omega.degFreed
    aliases = omega.alias
    if (aliases < 2):
        if (alias):
            warnMess = 'alias cannot be True! There are no waves to alias.'
            print(warnLoc + warnMess)
        alias = False
    NA = nh
    nh = int(aliases * nh)
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
    if (BT.Empty(nullspace)):
        nullspace = np.eye(nh, nh)
#         strings = omega.strings
    else:
        if (aliases > 1):
            N = np.shape(nullspace)[1]
        else:
            N = n
    
    
    if (not BT.Empty(waveNode)):
        waveNodes = waveNode @ rotMat @ nullspace
    if (BT.Empty(waves)):
        waveCell = np.asarray([[[] for i in range(N)] for j in range(n)])
    else:
        waveCell = waves @ nullspace
    waveCont = waveCont @ rotMat @ nullspace
    for k in range(N):
        if (not BT.Empty(waveTrans)):
            if (k < np.shape(waveTrans)[1]):
                waveTransfer = waveTrans[:, k]
        else:
            waveTransfer = []
        fig = PlotWave(omega, physics, numPoints, X, rescale, waveCell[:, k], waveCont[:, k], waveTrans = waveTransfer, xGrid = False, yGrid = False, enlarge = enlarge)
        if (alias):
            print('alias is', alias)
            if (k >= NA):
                kf = k
                kc = (2 * NA) - kf
                nodeFact = 1
                cellFact = -kc / kf
                if (k % 2 == 1):
                    kc = kc - 2
                    nodeFact = -1
                    cellFact = -cellFact
                aliasNode = nodeFact * waveCont[:, kc]
                aliasCell = cellFact * waveCont[:, kc]
                plt.plot(X, aliasNode, linestyle = ':', color = ColorDefault(2), zorder = 4)
                plt.plot(X, aliasCell, linestyle = '--', color = ColorDefault(3), zorder = 3)
        
        if (not BT.Empty(waveNode)):
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


def PlotWave(omega, physics, numPoints, X, rescale, waveCell = [], fX = [], title = '', labels = [], waveTrans = [], sym = True, xGrid = False, yGrid = False, enlarge = False, newBounds = []):
    errorLoc = 'ERROR:\nPlotTools:\nPlotWave:\n'
    errorMess = ''
    
    # if (enlarge):
    #     linewidth = 4
    #     fontsize = 35
    #     labelsize = 25
    # else:
    #     linewidth = 1.5
    #     fontsize = 25
    #     labelsize = 10

    linewidth, fontsize, labelsize = Enlarge(enlarge)
    
    if (not BT.Empty(fX)):
        if (BT.Empty(newBounds)):
            yMin, yMax, tickHeight = GetYBound(fX, sym)
        else:
            yMin, yMax, tickHeight = GetYBound(newBounds, sym)
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
        if (not BT.Empty(waveCell)):
            if (BT.Empty(newBounds)):
                yMin, yMax, tickHeight = GetYBound(waveCell, sym)
            else:
                yMin, yMax, tickHeight = GetYBound(newBounds, sym)
        else:
            errorMess = 'Must have argument for either fX or waveCell!'
    if (not BT.Empty(labels)):
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
    if (not BT.Empty(waveTrans)):
        PiecePlot(omega, numPoints, X, waveTrans, color = 3, linewidth = linewidth)
    TickPlot(omega, ax, tickHeight, xGrid, yGrid, linewidth = linewidth, labelsize = labelsize)
    if (numGraphs == 1):
        if (not BT.Empty(fX)):
            plt.plot(X, fX, color = ColorDefault(0), zorder = 2, label = labelsOut[0], linewidth = linewidth) # Fuck with this when you have time to worry about the line thickness of the analytic solution for a single plot.
            pieceLabel = []
        else:
            pieceLabel = labelsOut[0]
        if (not BT.Empty(waveCell)):
            PiecePlot(omega, numPoints, X, waveCell, label = pieceLabel, linewidth = linewidth)
    else:
        i = 0
        for j in range(numGraphs):
            if (not BT.Empty(fX)):
                plt.plot(X, fX[:, j], color = ColorDefault(i), zorder = 2, label = labelsOut[j], linewidth = linewidth)
                pieceColor = 3
                pieceLabel = []
            else:
                pieceColor = j
                pieceLabel = labelsOut[j]
            if (not BT.Empty(waveCell)):
                PiecePlot(omega, numPoints, X, waveCell[:, j], color = pieceColor, label = pieceLabel, linewidth = linewidth)
            i = i + 1
            if (j == 2):
                i = i + 1
        if (not BT.Empty(labels)):
            plt.legend(fontsize = labelfont)
            print('Are you *sure* your labels are ordered correctly?')
    if (title != ''):
        plt.title(title, fontsize = fontsize)
    locs = physics.locs
    for loc in locs:
        locx = loc * np.ones(2)
        locy = np.linspace(yMin, yMax, num = 2)
        plt.plot(locx, locy, color = ColorDefault(0.5), zorder = 1.5, linewidth = linewidth)
    plt.ylim([yMin, yMax])
    return fig


# This function overlays a piecewise cell average plot of a linear combination of wave vectors onto a plot of its continuous wave function. It also allows you to save this plot if desired. As the default, that feature is subdued.

# In[9]:


def PlotMixedWave(omega, physics, FCoefs, waves = [], title = '', labels = [], rescale = 1, plotCont = True, sym = False, save = False, saveName = '', dpi = 600, ct = 0, xGrid = False, yGrid = False, enlarge = False, newBounds = []):
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
            
        if (not BT.Empty(waves)):
            fXCell = waves[:nh, :nh] @ FCoef
        else:
            fXCell = []
        if (plotCont):
            fXCont = waveCont @ FCoef
        else:
            fXCont = []
        fig = PlotWave(omega, physics, numPoints, X, rescale, fXCell, fXCont, title = title1, sym = sym, labels = labels, xGrid = xGrid, yGrid = yGrid, enlarge = enlarge, newBounds = newBounds)
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
    if (BT.Empty(nullspace)):
        N = int(alias * nh)
        location = np.arange(N)
        locations = [np.asarray(location), np.asarray(location)]
    else:
        errorMess = BT.CheckSize(degFreed, nullspace[0, :], nName = 'degFreed', matricaName = 'nullspace')
        if (errorMess != ''):
            if (alias < 2):
                sys.exit(errorLoc + errorMess)
        N = alias * degFreed
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


def PlotGrid(omega, rescale = 1, save = False, saveName = '', dpi = 600, enlarge = False, label = True):
    numPoints, font, X, savePath = UsefulPlotVals()
    if (saveName != ''):
        save = True
    else:
        saveName = 'Grid'
    
    # if (enlarge):
    #     linewidth = 4
    #     fontsize = 35
    #     labelsize = 25
    # else:
    #     linewidth = 1.5
    #     fontsize = 25
    #     labelsize = 10

    linewidth, fontsize, labelsize = Enlarge(enlarge)
    
    saveString = savePath + saveName
    yMin, yMax, tickHeight = GetYBound(0, True)
    size, tickHeight, labelfont = Resize(rescale, tickHeight)
    fig, ax = plt.subplots(figsize = size)
    TickPlot(omega, ax, tickHeight, False, False, label = label, labelsize = labelsize, linewidth = linewidth)
    plt.ylim([yMin, yMax])
    plt.show()
    if (save):
        Save(fig, saveString, dpi)
    return


def Save(fig, saveString, dpi):
    fig.savefig(saveString + '.png', bbox_inches = 'tight', dpi = dpi, transparent = True)
    print('This image has been saved under ' + saveString + '.')
    return

def DivergVis(save = False, saveName = '', dpi = 600, enlarge = False, matVis = '', fill = False, ghost = '', var = 'v', something = True):
    if (saveName != ''):
        save = True
    else:
        saveName = 'DivergenceVisual'
    
    # if (enlarge):
    #     linewidth = 4
    #     fontsize = 35
    #     labelsize = 25
    # else:
    #     linewidth = 1.5
    #     fontsize = 25
    #     labelsize = 10

    linewidth, fontsize, labelsize = Enlarge(enlarge)
    
    if (fill):
        var = r'u'
    
    if (not something):
        var = ''
    
    
    
    nh = 32
    omega = BT.Grid(nh)
    if (ghost == ''):
        n = 4
    else:
        refRatio = 2
        if (ghost == 'G1'):
            n = 6
            off = 3
        else:
            if (ghost == 'G3'):
                n = 5
                off = 4
            else:
                n = 7
                off = 2
        cells = list(np.arange(int(nh - off)) + off)
        omega.AddPatch(refRatio, cells)
        
    
    hs = omega.h
    h = hs[0]
    x = omega.xNode
    length = 2.5 * h
    k = 2
    move = -0.01
    up = 0.03 # 0.3
    fact = 0.5
    
    
    gBlackX = length + np.linspace(-h/2, h/2, 2)
    gBlackY = np.zeros(2, float)
    
    
    
    numPoints, font, X, savePath = UsefulPlotVals()
    cellVals = np.ones(numPoints, float)
    

    
    
    
    
    Cosine = lambda x: fact * np.cos(2. * np.pi * k * (x + move))
    Sine = lambda x: fact * np.sin(2. * np.pi * k * (x + move))
    factor = 1. / (2 * pi * k)
    
    uNode = x * Sine(x) + up
    U = X * Sine(X) + up
    term1 = (factor ** 2) * (Sine(x[1:]) - Sine(x[:-1]))
    term2 = factor * (x[:-1] * (Cosine(x[:-1])) - (x[1:] * Cosine(x[1:])))
    uCell = ((term1 + term2) / hs) + up
    if (something):
        yMin, yMax, tickHeight = GetYBound(uNode[1:n + 1], False) # This started out as just n.
    else:
        yMin, yMax, tickHeight = GetYBound(0, True)
    
    if ((matVis == 'R2') or (matVis == 'L2')):
        uNode = uNode[1:]
        uCell = uCell[1:]
        newXInds = np.where(X > h)[0]
        U = U[newXInds]
        numPoints = len(newXInds)
        X = X[:numPoints]
#         print('U:', len(U))
#         print('X:', len(X))
#         print(X)
#         print('gBlackX Before:', gBlackX / h)
#         gBlackX = gBlackX + h
#         print('gBlackX After:', gBlackX / h)
    
    
#     factor = 1. / (2 * pi * k * omega.h)
    
#     uNode = Sine(x) + up
#     uCell = factor * (Cosine(x[:-1]) - Cosine(x[1:])) + up
#     U = Sine(X) + up
    
    
#     xCell = omega.xCell
    fig, ax = plt.subplots()
#     numPoints, font, X, savePath = UsefulPlotVals()
    
    TickPlot(omega, ax, tickHeight, False, False, u = uNode, labelsize = labelsize, linewidth = linewidth, matVis = matVis, fill = fill, var = var, ghost = ghost, something = something)
    if (matVis == 'R1'):
        if (var != r'L_{1}'):
            var = r'R_{1}'
        matInd = np.where(X <= gBlackX[-1])[0][-1]
        XL = X[:matInd]
        UL = U[:matInd]
        if (something):
            plt.plot(XL, UL, color = ColorDefault(0), zorder = 0, linewidth = linewidth)
            plt.scatter(x[1:n], uNode[1:n], s = 20, color = ColorDefault(2), zorder = 4)
        plt.quiver([length + (h / 2)], [0], [-length - (h / 2)], [0], color = ['k', 'k'], angles = 'xy', scale_units = 'xy', scale = 1, width = 0.005, headwidth = 8, headlength = 8)
    else:
        if (matVis == 'L2'):
            if (var != r'R_{2}'):
                var = r'L_{2}'
            matInd = np.where(X >= gBlackX[0])[0][0]
            XR = X[matInd:]
            UR = U[matInd:]
            if (something):
                plt.plot(XR, UR, color = ColorDefault(0), zorder = 0, linewidth = linewidth)
                plt.scatter(x[2:n + 1], uNode[2:n + 1], s = 20, color = ColorDefault(2), zorder = 4)
            plt.quiver([length - (h / 2)], [0], [length + (h / 2)], [0], color = ['k', 'k'], angles = 'xy', scale_units = 'xy', scale = 1, width = 0.005, headwidth = 8, headlength = 8)
            
        else:
            if (matVis == 'R2'):
                var = r'R_{2}'
                matInd = np.where(X >= gBlackX[0])[0][0]
                XL = X[:matInd]
                XR = X[matInd:]
                UL = U[:matInd]
                UR = U[matInd:]
                gBlackX = gBlackX - h
                if (something):
                    plt.plot(XL, UL, color = ColorDefault(0), zorder = 0, linewidth = linewidth, linestyle = '--')
                    plt.plot(XR, UR, color = ColorDefault(0), zorder = 0, linewidth = linewidth)
                    plt.plot(gBlackX, gBlackY, color = 'k', linestyle = '--', zorder = 0)
                    plt.scatter(x[2:n + 1], uNode[2:n + 1], s = 20, color = ColorDefault(2), zorder = 4)
                    plt.scatter(x[1], uNode[1], s = 20, facecolors = 'none', edgecolors = ColorDefault(2), zorder = 4)
                
                plt.quiver([length - (h / 2)], [0], [length + (h / 2)], [0], color = ['k', 'k'], angles = 'xy', scale_units = 'xy', scale = 1, width = 0.005, headwidth = 8, headlength = 8)
                
            else:
                if (matVis == 'L1'):
                    var = r'L_{1}'
                    matInd = np.where(X <= gBlackX[-1])[0][-1]
                    XL = X[:matInd]
                    XR = X[matInd:]
                    UL = U[:matInd]
                    UR = U[matInd:]
                    gBlackX = gBlackX + h
                    if (something):
                        plt.plot(XL, UL, color = ColorDefault(0), zorder = 0, linewidth = linewidth)
                        plt.plot(XR, UR, color = ColorDefault(0), zorder = 0, linewidth = linewidth, linestyle = '--')
                        plt.plot(gBlackX, gBlackY, color = 'k', linestyle = '--', zorder = 0)
                        plt.scatter(x[1:n], uNode[1:n], s = 20, color = ColorDefault(2), zorder = 4)
                        plt.scatter(x[n], uNode[n], s = 20, facecolors = 'none', edgecolors = ColorDefault(2), zorder = 4)
                    plt.quiver([length + (h / 2)], [0], [-length - (h / 2)], [0], color = ['k', 'k'], angles = 'xy', scale_units = 'xy', scale = 1, width = 0.005, headwidth = 8, headlength = 8)
                    
                else:
                    if (ghost == 'G2'):
                        g = 2
                        gVal = 0.5 * (uCell[g] + uCell[g + 1])
                        lowIndex = np.where(X <= x[g])[0][::-1][0] + 1
                        highIndex = np.where(X <= x[g + 2])[0][::-1][0] + 1
                        cellVals[lowIndex:highIndex] = gVal * cellVals[lowIndex:highIndex]
                        topString = r'$\left<' + var + r'^{(l - 1)}\right>_{j - 1}$'
                        shiftY = tickHeight / 3
                        xLoc = X[lowIndex] - 0.013
                        yLoc = gVal + shiftY
                        matInd1 = np.where(X >= gBlackX[0])[0][0]
                        matInd2 = np.where(X <= gBlackX[-1])[0][-1]
                        XL = X[:matInd1]
                        XM = X[matInd1:matInd2]
                        XR = X[matInd2:]
                        UL = U[:matInd1]
                        UM = U[matInd1:matInd2]
                        UR = U[matInd2:]
                        if (something):
                            plt.plot(XL, UL, color = ColorDefault(0), zorder = 0, linewidth = linewidth)
                            plt.plot(XM, UM, color = ColorDefault(0), zorder = 0, linewidth = linewidth, linestyle = '--')
                            plt.plot(XR, UR, color = ColorDefault(0), zorder = 0, linewidth = linewidth)
                            #                         plt.plot(gBlackX, gBlackY, color = 'k', linestyle = '--', zorder = 2)
    #                         plt.quiver([length + (h / 2)], [0], [length - (h / 2)], [0], color = ['k', 'k'], angles = 'xy', scale_units = 'xy', scale = 1, width = 0.005, headwidth = 8, headlength = 8)
    #                         plt.quiver([length - (h / 2)], [0], [-length + (h / 2)], [0], color = ['k', 'k'], angles = 'xy', scale_units = 'xy', scale = 1, width = 0.005, headwidth = 8, headlength = 8)
                            plt.plot(X[lowIndex:highIndex], cellVals[lowIndex:highIndex], color = ColorDefault(3), linestyle = '-', zorder = 3, linewidth = linewidth)
                            plt.scatter(x[4:n + 1], uNode[4:n + 1], s = 20, color = ColorDefault(2), zorder = 4)
                            plt.scatter(x[3], uNode[3], s = 20, facecolors = 'none', edgecolors = ColorDefault(2), zorder = 4)
                            plt.scatter(x[1:3], uNode[1:3], s = 20, color = ColorDefault(2), zorder = 4)
                            plt.text(xLoc, yLoc, topString, fontsize = 12, zorder = 5)
                    
                        plt.quiver([length], [0], [length], [0], color = ['k', 'k'], angles = 'xy', scale_units = 'xy', scale = 1, width = 0.005, headwidth = 8, headlength = 8)
                        plt.quiver([length], [0], [-length], [0], color = ['k', 'k'], angles = 'xy', scale_units = 'xy', scale = 1, width = 0.005, headwidth = 8, headlength = 8)

                    else:
                        plt.quiver([length], [0], [length], [0], color = ['k', 'k'], angles = 'xy', scale_units = 'xy', scale = 1, width = 0.005, headwidth = 8, headlength = 8)
                        plt.quiver([length], [0], [-length], [0], color = ['k', 'k'], angles = 'xy', scale_units = 'xy', scale = 1, width = 0.005, headwidth = 8, headlength = 8)
                        if (something):
                            plt.plot(X, U, color = ColorDefault(0), zorder = 0, linewidth = linewidth)
                            plt.scatter(x[1:n + 1], uNode[1:n + 1], s = 20, color = ColorDefault(2), zorder = 4)
    
    if (something):
        PiecePlot(omega, numPoints, X, uCell, tickHeight = tickHeight, linewidth = linewidth, matVis = matVis, fill = fill, var = var, ghost = ghost)
    if (fill and something):
        plt.fill_between(X, 0, U, color = ColorDefault(0), alpha = 0.1)
    
    plt.xlim([-0.1 * length, 2 * length])
    plt.ylim([yMin, yMax])
    plt.show()
    if (save):
        saveString = savePath + saveName
        Save(fig, saveString, dpi)
    return


def Enlarge(enlarge):
    if (enlarge):
        linewidth = 4
        fontsize = 35
        labelsize = 30
    else:
        linewidth = 1.5
        fontsize = 25
        labelsize = 10
    return linewidth, fontsize, labelsize


