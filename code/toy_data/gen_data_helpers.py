# -*- coding: utf-8 -*-
"""
Created on Tue Jan 14 16:26:07 2020

@author: cmccurley
"""


"""
***********************************************************************
    *  File:  gen_data_helpers.py
    *
    *  Desc:  This file contains helper functions for generating toy
    *         manifold learning data.
    *
    *  Written by:  Connor H. McCurley
    *
    *  Latest Revision:  2020-01-13
    *
**********************************************************************
"""

######################################################################
######################### Import Packages ############################
######################################################################

# General packages
import numpy as np
from sklearn import manifold, datasets
import scipy.io
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Custom packages
import toy_data_params

######################################################################
######################## Function Definitions ########################
######################################################################
def samplegrid(f0_min, f0_max, f1_min, f1_max, n):
    """
    ******************************************************************
        *
        *  Func:    samplegrid()
        *
        *  Desc:    Returns list of indices sampled from 2D grid.
        *
        *  Inputs:
        *          n - number of points to samples 
        *        
        *          f0_min - minimum value in feature 0
        *        
        *          f0_max - maximum value in feature 0
        *        
        *          f1_min - minimum value in feature 1
        *        
        *          f1_max - minimum value in feature 1
        *
        *          n - number of samples
        *
        *  Outputs:
        *          X - [nx2] matrix of sampled coordinates in grid
        *
        *
    ******************************************************************
    """
    
    X = np.empty((n,2))
    
    f0 = np.random.uniform(f0_min, f0_max, n)
    f1 = np.random.uniform(f1_min, f1_max, n)
    
    X[:,0] = f0
    X[:,1] = f1
    
    
    return X

def gen_quadratic_surface(f2_min, f2_max, weight_0, weight_1, grid_samples):
    """
    ******************************************************************
        *
        *  Func:    gen_quadratic_surface()
        *
        *  Desc:    Returns matrix of a 3D quadratic manifold.  This
        *           manifold is an elliptic parabloid with the bowl up,
        *           meaning it has a global minimum.
        *
        *  Inputs: 
        *        
        *          f2_min - minimum value in feature 2
        *        
        *          f2_max - maximum value in feature 2 (currently not used)
        *
        *          weight_0 - weighting on first term of quad surface
        *
        *          weight_1 - weighting on second term of quad surface
        *
        *          grid_samples - [nx2] matrix of sampled coordinates in grid
        *
        *  Outputs:
        *          X - [nx3] matrix of quadratic surface manifold
        *
        *
    ******************************************************************
    """
    
    n = grid_samples.shape[0]
    X = np.empty((n,3))
    
    X[:,0] = grid_samples[:,0]
    X[:,1] = grid_samples[:,1]
    
    for samp in range(0,n):
        X[samp,2] = f2_min + (weight_0*X[samp,0]**2) + (weight_1*X[samp,1]**2)
    
    return X

def plot_3D_manifolds(X_c0, X_c1, title):
    """
    ******************************************************************
        *
        *  Func:    plot_3D_manifolds()
        *
        *  Desc:    Makes a 3D scatter plot displaying two manifolds of 
        *           different classes.
        *
        *  Inputs: 
        *        
        *          X_c0 - [nx3] matrix of manifold samples for class 0
        *        
        *          X_c1 - [nx3] matrix of manifold samples for class 1 
        *
        *          title - string of figure title
        *
    ******************************************************************
    """
    
    ## Create new figure
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    ## Plot manifold for class 0
    ax.scatter(X_c0[:,0], X_c0[:,1], X_c0[:,2], color='blue')
    
    ## Plot manifold for class 1
    ax.scatter(X_c1[:,0], X_c1[:,1], X_c1[:,2], color='red')
    
    ax.set_xlabel('f0', fontsize = 14)
    ax.set_ylabel('f1', fontsize = 14)
    ax.set_zlabel('f2', fontsize = 14)
    plt.title(title, fontsize = 14)
    plt.show()    
    
    return