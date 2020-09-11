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


def CheckSize(n, checkMatrix):
    dim = size(shape(checkMatrix))
    problem = 0
    for i in range(dim):
        if (n != shape(checkMatrix)[i]):
            problem = problem + 1
    return problem


# This function ensures that $n^{h}$ is an appropriate base-2 value.

# In[3]:


def CheckNumber(nh):
    check = nh
    while (check % 2 == 0):
        check = check / 2
    if (check != 1):
        sys.exit('ERROR:\nnh must be a base-2 integer!')
    return


# This function outputs an $x$ array and a $y$ array of size $n^{h}$ + 1 of the locations of the tick marks.

# In[4]:


def MakeXY(nh):
    x = np.linspace(0, 1, num = nh + 1)
    y = np.zeros(nh + 1, float)
    return x, y


# In[ ]:




