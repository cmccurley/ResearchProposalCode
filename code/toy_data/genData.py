# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

"""
***********************************************************************
    *  File:  genData.py
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
from sklearn import manifold, datasets

import scipy.io

# Custom packages
from toy_data_params import *

#number of data points to generate
n_points = 3000

#S-curve data set
SCurve, SCurveColor = datasets.samples_generator.make_s_curve(n_points, random_state=0)
SCurve = SCurve.T
SCurveColor = SCurveColor.T

#Swiss roll data set
swissRoll, swissRollColor = datasets.samples_generator.make_swiss_roll(n_samples=n_points, noise=0.0, random_state=0)
swissRoll = swissRoll.T
swissRollColor = swissRollColor.T


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