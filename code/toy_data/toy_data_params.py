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

    num_c0 = 1000 # number of samples in background class
    num_c1 = 1000 # number of samples in target class
    num_input_dims = 3 # number of features in the input space

    manifold_type = 'quadratic_surfaces_overlap'


    ###################### Quadratic surfaces #########################

    f0_min = -4
    f0_max = 4
    f1_min = -4
    f1_max = 4
    f2_min = 0
    f2_max = 4

    weight_0 = 4 # weighting on first term of quadratic surface (must be positive)
    weight_1 = 4 # weighting on second term of quadratic surface (must be positive)

    ######################### Swiss Roll ##############################


    ########################### Faces #################################


    ########################## MNIST Digits ###########################
