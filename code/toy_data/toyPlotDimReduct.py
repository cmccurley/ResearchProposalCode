# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 11:46:50 2018

@author: cmccurley
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 08:15:50 2018

@author: cmccurley
"""


"""
***********************************************************************
    *  File:  toyPlotDimReduct.py
    *  Name:  Connor McCurley
    *  Date:  2018-11-14
    *  Desc:  
**********************************************************************
"""


import numpy as np
from sklearn import manifold, datasets

import scipy.io
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding
import matplotlib.pyplot as plt
from matplotlib import offsetbox



#plotting function- node images over scatter plot

#Use this one for the Faces and MNIST data sets
def plot_components(data, proj, images=None, ax=None,
                    thumb_frac=0.05, cmap='gray'):
    ax = ax or plt.gca()
    
    ax.plot(proj[:, 0], proj[:, 1], '.k')
    
    if images is not None:
        min_dist_2 = (thumb_frac * max(proj.max(0) - proj.min(0))) ** 2
        shown_images = np.array([2 * proj.max(0)])
        for i in range(data.shape[0]):
            dist = np.sum((proj[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, proj[i]])
            imagebox = offsetbox.AnnotationBbox(
                offsetbox.OffsetImage(images[i],zoom=1, cmap=cmap),
                                      proj[i])
            ax.add_artist(imagebox)
            

            
if __name__== "__main__":
    

   #=================== Load and plot faces dataset ======================
    projStruct = scipy.io.loadmat('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Algorithms/Manifold/sampleData/IsomapFaces12.mat')
    proj = projStruct["IsomapFaces12"]
    projStruct = scipy.io.loadmat('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Algorithms/Manifold/sampleData/LLEFaces20.mat')
    proj = projStruct["LLEFaces20"]
    
    from sklearn.datasets import fetch_lfw_people
    faces = fetch_lfw_people(min_faces_per_person=30)
        
    #Plot a few example faces
    fig, ax = plt.subplots(4, 8, subplot_kw=dict(xticks=[], yticks=[]))
    for i, axi in enumerate(ax.flat):
        axi.imshow(faces.images[i], cmap='gray')
    fig.suptitle('Example Faces')

    #Reduce data dimensionality using Isomap
    model = Isomap(n_components=2)
    proj = model.fit_transform(faces.data)
    proj.shape
    
    #plot nodes over scatter plot
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_components(faces.data, proj, images=faces.images[:, ::2, ::2])
    fig.suptitle('LLE Embedding with kNN=20')
    
    
#    #=================== Load and plot MNIST dataset ======================
#    digits = datasets.load_digits(n_class=10)
#    data = digits["data"]
#    projStruct = scipy.io.loadmat('R:/Army/Users/Connor/Code/MLSLArmyRepoServer/Algorithms/Manifold/sampleData/digits40x40_isomap_12.mat')
#    proj = projStruct["digits40x40_isomap_12"]
#    
#
#    
#    #Upsample images
#    import scipy.ndimage
#    up = 40
#    upData = np.zeros((data.shape[0], up*up))
#    upImages = np.zeros((data.shape[0],up,up))
#    
#    for i in range (0, data.shape[0]):
#        currentImage = np.reshape(data[i,:],(8,8))
#        J = scipy.ndimage.zoom(currentImage, (up/8), order=1) 
#        K = np.reshape(J,(1,up*up))
#        upData[i,:] = K
#        upImages[i,:,:] = J
#        
#    digits40x40 = upData
#        
#    fig, ax = plt.subplots(6, 8, subplot_kw=dict(xticks=[], yticks=[]))
#    for i, axi in enumerate(ax.flat):
#        axi.imshow(upData[i].reshape(up, up), cmap='gray_r')
#    fig.suptitle('Example MNIST Digits')
#    
#    #plot nodes over scatter plot
#    fig, ax = plt.subplots(figsize=(10, 10))
#    plot_components(upData, proj, imageIndex, images=MNImages[::2, ::2, :])
#    fig.suptitle('Isomap Embedding with kNN=20')
    
   
    