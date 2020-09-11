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


# This function checks to make sure that sizes match up appropriately.

# In[2]:


def CheckSize(nh, checkMatrix):
    dim = size(shape(checkMatrix))
    problem = 0
    for i in range(dim):
        if (nh != shape(checkMatrix)[i]):
            problem = problem + 1
    return problem


# This function outputs an $x$ array and a $y$ array of size $n^{h}$ + 1 of the locations of the tick marks.

# In[3]:


def MakeXY(nh):
    x = np.linspace(0, 1, num = nh + 1)
    y = np.zeros(nh + 1, float)
    return x, y


# In[ ]:




