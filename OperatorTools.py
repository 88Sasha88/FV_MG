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


# This function constructs a 1-D Laplacian operator.

# In[3]:


def MakeLaplacian1D(nh, h):
    Laplacian = np.zeros((nh, nh), float)
    fill_diagonal(Laplacian[1:], -1)
    fill_diagonal(Laplacian[:, 1:], -1)
    fill_diagonal(Laplacian, 2)
    Laplacian = (1.0 / (h ** 2)) * Laplacian
    return Laplacian


# In[ ]:




