# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
***********************************************************************
    *  File:  genDataMain.py
    *
    *  Desc:  This file generates a variety of toy discriminative
    *         manifold learning datasets.
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
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import manifold, datasets
import scipy.io
import random
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Custom packages
import toy_data_params
from gen_data_helpers import *

######################################################################
############################## Main ##################################
######################################################################
if __name__ == "__main__":

    ######################################################################
    ######################### Set Parameters #############################
    ######################################################################
    ## Import parameters
    parameters = toy_data_params.SetParams()

    ######################################################################
    ######################## Create Datasets #############################
    ######################################################################

    ## Initialize data structures and variables
    num_c0 = parameters.num_c0
    num_c1 = parameters.num_c1
    class_00 = np.empty((num_c0,parameters.num_input_dims))
    class_01 = np.empty((num_c1,parameters.num_input_dims))
    f0_min = parameters.f0_min
    f0_max = parameters.f0_max
    f1_min = parameters.f1_min
    f1_max = parameters.f1_max
    f2_min = parameters.f2_min
    f2_max = parameters.f2_max
    manifold_type = parameters.manifold_type

    ## Create manifolds
    if (manifold_type == 'quadratic_surfaces_separable'):

        ## Generate grids of samples
        samples_f0 = samplegrid(f0_min, f0_max, f1_min, f1_max, num_c0)
        samples_f1 = samplegrid(f0_min, f0_max, f1_min, f1_max, num_c1)

        ## Generate Class 0 manifold
        f2_min = 0
        weight_0 = 1
        weight_1 = 1
        X_c0 = gen_quadratic_surface(f2_min, f2_max, weight_0, weight_1, samples_f0)

        ## Generate Class 1 manifold
        f2_min = 10
        weight_0 = 2
        weight_1 = 2
        X_c1 = gen_quadratic_surface(f2_min, f2_max, weight_0, weight_1, samples_f1)

        # ## Scale the data
        # scaler_c0 = StandardScaler()
        # scaler_c0.fit(X_c0)
        # X_c0 = scaler_c0.transform(X_c0)

        # scaler_c1 = StandardScaler()
        # scaler_c1.fit(X_c1)
        # X_c1 = scaler_c1.transform(X_c1)

        ## Plot manifolds
        plot_3D_manifolds(X_c0, X_c1, 'Linearly Separable Quadradic Surfaces')


    elif (manifold_type == 'quadratic_surfaces_overlap'):
        ## Generate grids of samples
        samples_f0 = samplegrid(f0_min, f0_max, f1_min, f1_max, num_c0)

        ## These will control the make value in feature 3 (which is important
        ## for controlling visualization when increasing steepness)
        f0_min = -0.5
        f0_max = 0.5
        f1_min = -0.5
        f1_max = 0.5
        samples_f1 = samplegrid(f0_min, f0_max, f1_min, f1_max, num_c1)

        ## Generate Class 0 manifold
        f2_min = 0
        weight_0 = 1
        weight_1 = 1
        X_c0 = gen_quadratic_surface(f2_min, f2_max, weight_0, weight_1, samples_f0)

        ## Generate Class 1 manifold
        f2_min = -4
        weight_0 = 5
        weight_1 = 5
        X_c1 = gen_quadratic_surface(f2_min, f2_max, weight_0, weight_1, samples_f1)

        X_c1[:,2] = X_c1[:,2]

        ## Plot manifolds
        plot_3D_manifolds(X_c0, X_c1, 'Overlapping Quadradic Surfaces')

    # elif (manifold_type == 's_curves'):
    #     ## S-curve data set
    #     SCurve, SCurveColor = datasets.make_s_curve(n_points, random_state=0)
    #     SCurve = SCurve.T
    #     SCurveColor = SCurveColor.T

    # elif (manifold_type == 'swiss_rolls'):
    #     ## Swiss roll data set
    #     swissRoll, swissRollColor = datasets.make_swiss_roll(n_samples=n_points, noise=0.0, random_state=0)
    #     swissRoll = swissRoll.T
    #     swissRollColor = swissRollColor.T







    ######################################################################
    ########################## Save Datasets #############################
    ######################################################################


# ============================= Save data sets ================================

##numpy array
#np.save('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Algorithms/Manifold/sampleData/SCurve.npy',SCurve)
#np.save('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Algorithms/Manifold/sampleData/SCurveColor.npy',SCurveColor)
#np.save('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Algorithms/Manifold/sampleData/swissRoll.npy',swissRoll)
#np.save('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Algorithms/Manifold/sampleData/swissRollColor.npy',swissRollColor)

#mat file
#scipy.io.savemat('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Algorithms/Manifold/sampleData/SCurve.mat', mdict={'SCurve': SCurve})
#scipy.io.savemat('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Algorithms/Manifold/sampleData/SCurveColor.mat', mdict={'SCurveColor': SCurveColor})
#scipy.io.savemat('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Algorithms/Manifold/sampleData/swissRoll.mat', mdict={'swissRoll': swissRoll})
#scipy.io.savemat('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Algorithms/Manifold/sampleData/swissRollColor.mat', mdict={'swissRollColor': swissRollColor})