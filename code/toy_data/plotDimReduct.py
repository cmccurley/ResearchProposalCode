# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 08:15:50 2018

@author: cmccurley
"""


"""
***********************************************************************
    *  File:  plotDimReduct.py
    *  Name:  Connor McCurley
    *  Date:  2018-11-14
    *  Desc:  
**********************************************************************
"""

import h5py
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import offsetbox


def plot_components(proj, imageIndex, images=None,  ax=None,cmap='gray'):
        
    """
    ***********************************************************************
        *  File:  plot_components
        *  Name:  Connor McCurley
        *  Date:  2018-11-14
        *  Desc:  
        *  Inputs: data: nSamples by nFeatures, proj: nSamples by 2, 
                    imageIndex: list of data point indices corresponding to images
                    images: MxNxnumImages
        *  Outputs:
    **********************************************************************
    """
    ax = ax or plt.gca()
    
    ax.plot(proj[:, 0], proj[:, 1], '.k')
    
    
    for i in range(images.shape[2]):
        imagebox = offsetbox.AnnotationBbox(
            offsetbox.OffsetImage(images[:,:,i],zoom=0.25, cmap=cmap),
                                  proj[int(imageIndex[i,0]-1)])
        ax.add_artist(imagebox)

            
if __name__== "__main__":
    
    #=================== Load data for mat files ==============================
#    #2D embedded data
#    projStruct = scipy.io.loadmat('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Results/2018_11_Manifold/embData_Isomap_6.mat')
#    proj = projStruct["embData_Isomap_6"].T
#    
#    #Feature images
#    imageStruct = scipy.io.loadmat('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Results/2018_11_Manifold/Images_Isomap_6.mat')
#    images = imageStruct["Images_Isomap_6"]
#    
#    #indices of images
#    imageIndex = scipy.io.loadmat('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Results/2018_11_Manifold/imageIndex_Isomap_6.mat')
#    index = imageIndex["imageIndex_Isomap_6"]
#
    #==============================================================================
    #==============================================================================
    #==============================================================================
    #==============================================================================
    #==============================================================================
    #==============================================================================
    #==============================================================================
    #==============================================================================
    #==============================================================================
    #==============================================================================
    #==============================================================================
    #==============================================================================
    
    
#    #================== Load data for v7.3 mat files ==========================
#    #2D embedded data
#    f = h5py.File('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Results/2018_11_Manifold/embData_Isomap_6.mat')
#    arrays = {}
#    for k,v in f.items():
#         arrays[k] = np.array(v)
#    projData = arrays["embData_Isomap_6"]
#    
#    #Feature images
#    f = h5py.File('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Results/2018_11_Manifold/Images_Isomap_6.mat')
#    arrays = {}
#    for k,v in f.items():
#         arrays[k] = np.array(v)
#    images = arrays["Images_Isomap_6"].T
#    
#    #indices of images
#    f = h5py.File('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Results/2018_11_Manifold/imageIndex_Isomap_6.mat')
#    arrays = {}
#    for k,v in f.items():
#         arrays[k] = np.array(v)
#    index = arrays["imageIndex_Isomap_6"].T
#    
#    
#    #================== Plot nodes over scatter plot ==========================
#    #plot nodes over scatter plot
#    fig, ax = plt.subplots(figsize=(10, 10))
#    plot_components(projData, index, images=images[::2, ::2, :])
#    fig.suptitle('Isomap Embedding NN=6')
#    
#    #2D embedded data
#    f = h5py.File('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Results/2018_11_Manifold/embData_Isomap_12.mat')
#    arrays = {}
#    for k,v in f.items():
#         arrays[k] = np.array(v)
#    projData = arrays["embData_Isomap_12"]
#    
#    #Feature images
#    f = h5py.File('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Results/2018_11_Manifold/Images_Isomap_12.mat')
#    arrays = {}
#    for k,v in f.items():
#         arrays[k] = np.array(v)
#    images = arrays["Images_Isomap_12"].T
#    
#    #indices of images
#    f = h5py.File('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Results/2018_11_Manifold/imageIndex_Isomap_12.mat')
#    arrays = {}
#    for k,v in f.items():
#         arrays[k] = np.array(v)
#    index = arrays["imageIndex_Isomap_12"].T
#    
#    #plot nodes over scatter plot
#    fig, ax = plt.subplots(figsize=(10, 10))
#    plot_components(projData, index, images=images[::2, ::2, :])
#    fig.suptitle('Isomap Embedding NN=12')
#    
#    #2D embedded data
#    f = h5py.File('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Results/2018_11_Manifold/embData_Isomap_20.mat')
#    arrays = {}
#    for k,v in f.items():
#         arrays[k] = np.array(v)
#    projData = arrays["embData_Isomap_20"]
#    
#    #Feature images
#    f = h5py.File('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Results/2018_11_Manifold/Images_Isomap_20.mat')
#    arrays = {}
#    for k,v in f.items():
#         arrays[k] = np.array(v)
#    images = arrays["Images_Isomap_20"].T
#    
#    #indices of images
#    f = h5py.File('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Results/2018_11_Manifold/imageIndex_Isomap_20.mat')
#    arrays = {}
#    for k,v in f.items():
#         arrays[k] = np.array(v)
#    index = arrays["imageIndex_Isomap_20"].T
#    
#    #plot nodes over scatter plot
#    fig, ax = plt.subplots(figsize=(10, 10))
#    plot_components(projData, index, images=images[::2, ::2, :])
#    fig.suptitle('Isomap Embedding NN=20')
    
    #==============================================================================
    #==============================================================================
    #==============================================================================
    #==============================================================================
    #==============================================================================
    #==============================================================================
    #==============================================================================
    #==============================================================================
    #==============================================================================
    #==============================================================================
    #==============================================================================
    #==============================================================================
    

    
     #2D embedded data
    f = h5py.File('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Results/2018_11_Manifold/embData_LLE_12.mat')
    arrays = {}
    for k,v in f.items():
         arrays[k] = np.array(v)
    projData = arrays["embData_LLE_12"]
    
    #Feature images
    f = h5py.File('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Results/2018_11_Manifold/Images_LLE_12.mat')
    arrays = {}
    for k,v in f.items():
         arrays[k] = np.array(v)
    images = arrays["Images_LLE_12"].T
    
    #indices of images
    f = h5py.File('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Results/2018_11_Manifold/imageIndex_LLE_12.mat')
    arrays = {}
    for k,v in f.items():
         arrays[k] = np.array(v)
    index = arrays["imageIndex_LLE_12"].T
    
    #plot nodes over scatter plot
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_components(projData, index, images=images[::2, ::2, :])
    fig.suptitle('LLE Embedding NN=12')
    
#    #2D embedded data
#    f = h5py.File('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Results/2018_11_Manifold/embData_LLE_20.mat')
#    arrays = {}
#    for k,v in f.items():
#         arrays[k] = np.array(v)
#    projData = arrays["embData_LLE_20"]
#    
#    #Feature images
#    f = h5py.File('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Results/2018_11_Manifold/Images_LLE_20.mat')
#    arrays = {}
#    for k,v in f.items():
#         arrays[k] = np.array(v)
#    images = arrays["Images_LLE_20"].T
#    
#    #indices of images
#    f = h5py.File('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Results/2018_11_Manifold/imageIndex_LLE_20.mat')
#    arrays = {}
#    for k,v in f.items():
#         arrays[k] = np.array(v)
#    index = arrays["imageIndex_LLE_20"].T
#    
#    #plot nodes over scatter plot
#    fig, ax = plt.subplots(figsize=(10, 10))
#    plot_components(projData, index, images=images[::2, ::2, :])
#    fig.suptitle('LLE Embedding NN=20')
#    
#    #2D embedded data
#    f = h5py.File('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Results/2018_11_Manifold/embData_LLE_30.mat')
#    arrays = {}
#    for k,v in f.items():
#         arrays[k] = np.array(v)
#    projData = arrays["embData_LLE_30"]
#    
#    #Feature images
#    f = h5py.File('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Results/2018_11_Manifold/Images_LLE_30.mat')
#    arrays = {}
#    for k,v in f.items():
#         arrays[k] = np.array(v)
#    images = arrays["Images_LLE_30"].T
#    
#    #indices of images
#    f = h5py.File('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Results/2018_11_Manifold/imageIndex_LLE_30.mat')
#    arrays = {}
#    for k,v in f.items():
#         arrays[k] = np.array(v)
#    index = arrays["imageIndex_LLE_30"].T
#    
#    #plot nodes over scatter plot
#    fig, ax = plt.subplots(figsize=(10, 10))
#    plot_components(projData, index, images=images[::2, ::2, :])
#    fig.suptitle('LLE Embedding NN=30')
