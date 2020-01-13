# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 14:13:08 2020

@author: cmccurley
"""

"""
***********************************************************************
    *  File:  toy_data_params.py
    *
    *  Desc:  This file defines the parameters for toy data generation.
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

## General packages
import numpy as np
from sklearn import manifold, datasets

import scipy.io

## Custom packages


######################################################################
############################## Set Parameters ########################
######################################################################
class SetParams(object):
    """
    ******************************************************************
        *
        *  Func:    SetParams()
        *
        *  Desc:    Class definition for the parameters object.  
        *           This object defines that parameters for toy manifold
        *           learning data script.
        *
    ******************************************************************
    """
    
    ###################### General parameters #########################
    
    num_c0 = 200 # number of samples in background class
    num_c1 = 200 # number of samples in target class
    
    
    ###################### Quadratic surfaces #########################
    
    f0_range = np.arange(-2,2)
    f1_range = np.arange(-2,2)
    
    ######################### Swiss Roll ##############################
    
    
    ########################### Faces #################################
  
    
    ########################## MNIST Digits ###########################
