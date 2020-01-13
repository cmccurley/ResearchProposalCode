# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
'''==============================================='''
'''==============================================='''
'''==============================================='''
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import offsetbox

from PIL import Image
import cv2
from skimage.transform import resize

from sklearn import datasets
from sklearn.manifold import Isomap
from sklearn.manifold import LocallyLinearEmbedding

import scipy.ndimage

#from gtm import GTM

from minisom import MiniSom  

import matplotlib  

#import ugtm

'''==============================================='''
'''============= Function Definitions ============'''
'''==============================================='''

#plotting function- node images over scatter plot

#Use this one for the Faces and MNIST data sets
def plot_components(data, proj, images=None, ax=None,
                    thumb_frac=0.1, cmap='plasma'):
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
            
'''==============================================='''
'''=================== Main ======================'''
'''==============================================='''   
if __name__== "__main__":

    '''==============================================='''
    '''==================== MNIST  ==================='''
    '''==============================================='''

    #Load the dataset
    digits = datasets.load_digits(n_class=10)
    data = digits["data"]
    labels = digits["target"]
    
    #Upsample images
    
    up = 40
    upData = np.zeros((data.shape[0], up*up))
    upImages = np.zeros((data.shape[0],up,up))
    
    for i in range (0, data.shape[0]):
        currentImage = np.reshape(data[i,:],(8,8))
        J = scipy.ndimage.zoom(currentImage, (up/8), order=3) 
        K = np.reshape(J,(1,up*up))
        upData[i,:] = K
        upImages[i,:,:] = J
        
    digits40x40 = upData
    
    '''==============================================='''
    '''============== Plot MNIST Images =============='''
    '''===============================================''' 
#    #Plot a few upsampled images
#    fig, ax = plt.subplots(6, 8, subplot_kw=dict(xticks=[], yticks=[]))
#    for i, axi in enumerate(ax.flat):
#        axi.imshow(upData[i].reshape(up, up), cmap='plasma')
#    fig.suptitle('Example MNIST Digits')

    '''==============================================='''
    '''=============== MNIST SOM ====================='''
    '''==============================================='''
#    print('Running self-organizing map...')
#    
#    mapSize = 10
#    
#    #Initialize SOM
#    model = MiniSom(mapSize, mapSize, upData.shape[1], sigma=0.8, learning_rate=0.8)
#    
##    model.pca_weights_init(upData)
#    
#    #plot initial node examples
#    fig, axarr = plt.subplots(mapSize, mapSize)
#    
#    for row in range(mapSize):
#        for col in range(mapSize):
#            axarr[row,col].imshow(model._weights[row,col].reshape(up,up), cmap='plasma')
#            plt.setp(axarr[row,col],xticks=[], yticks=[])
#            
#    fig.suptitle('Initial SOM')
#
#    #Train SOM
#    model.train_random(upData, 5000)
#    
#    #plot initial node examples
#    fig, axarr = plt.subplots(mapSize, mapSize)
#    
#    for row in range(mapSize):
#        for col in range(mapSize):
#            axarr[row,col].imshow(model._weights[row,col].reshape(up,up), cmap='plasma')
#            plt.setp(axarr[row,col],xticks=[], yticks=[])
#            
#    fig.suptitle('Trained SOM')

    '''==============================================='''
    '''=========== Handwritten Characters ============'''
    '''==============================================='''
    
#    #Load the dataset
#    data = np.load('E:/University of Florida/Research/Presentations/2019_01_23_Lab_Group/Test/Hard_Test_Images.npy')
#
#    up = 40
#    
#    upData = np.zeros((data.shape[0], up, up))
#    
#    #Resize images to have the same dimensions
#    for ind in range(0,data.shape[0]):
#        upData[ind, :, :] = resize(data[ind],(up, up), anti_aliasing=True)
#    
#    
#    '''==============================================='''
#    '''============== Plot Character Images =========='''
#    '''===============================================''' 
#    #Plot a few upsampled images
#    fig, ax = plt.subplots(6, 8, subplot_kw=dict(xticks=[], yticks=[]))
#    for i, axi in enumerate(ax.flat):
#        axi.imshow(upData[i], cmap='plasma')
#    fig.suptitle('Example Handwritten Characters')

    '''==============================================='''
    '''=============== Character SOM ================='''
    '''==============================================='''
#    print('Running self-organizing map...')
#   
#    mapSize = 10
#    
#    #Initialize SOM
#    model = MiniSom(mapSize, mapSize, upData.shape[1], sigma=0.8, learning_rate=0.8)
##    
##    #plot initial node examples
##    fig, axarr = plt.subplots(mapSize, mapSize)
##    
##    for row in range(mapSize):
##        for col in range(mapSize):
##            axarr[row,col].imshow(model._weights[row,col].reshape(up,up), cmap='plasma')
##            plt.setp(axarr[row,col],xticks=[], yticks=[])
##            
##    fig.suptitle('Initial SOM')
##
#    #Train SOM
#    model.train_random(upData, 5000)
#    
#    #plot initial node examples
#    fig, axarr = plt.subplots(mapSize, mapSize)
#    
#    for row in range(mapSize):
#        for col in range(mapSize):
#            axarr[row,col].imshow(model._weights[row,col].reshape(up,up), cmap='plasma')
#            plt.setp(axarr[row,col],xticks=[], yticks=[])
#            
#    fig.suptitle('Trained SOM')

    '''==============================================='''
    '''================ Isomap ======================='''
    '''===============================================''' 
#    print('Running Isomap...')
#    #Reduce data dimensionality using Isomap
#    model = Isomap(n_neighbors=20, n_components=2)
#    proj = model.fit_transform(upData)
#    
#    
#    #plot nodes over scatter plot
#    fig, ax = plt.subplots(figsize=(10, 10))
#    plot_components(upData, proj, images=upData.reshape((upData.shape[0], up, up)))
#    fig.suptitle('Isomap Embedding with kNN=20')
    
    '''==============================================='''
    '''=================== LLE ======================='''
    '''==============================================='''
#    print('Running LLE...')
#    #Reduce data dimensionality using Isomap
#    model = LocallyLinearEmbedding(n_neighbors=20, n_components=2)
#    proj = model.fit_transform(upData)
#    
#    
#    #plot nodes over scatter plot
#    fig, ax = plt.subplots(figsize=(10, 10))
#    plot_components(upData, proj, images=upData.reshape((upData.shape[0], up, up)))
#    fig.suptitle('LLE Embedding with kNN=20')

    '''==============================================='''
    '''=================== GTM ======================='''
    '''==============================================='''
    print('Running GTM...')
    
#    from sklearn.preprocessing import StandardScaler
#    from sklearn.pipeline import make_pipeline
    
#    model = make_pipeline(StandardScaler(),GTM(n_components=2))
#    embedding = model.fit_transform(upData)
    
    #Reduce data dimensionality using Isomap
#    model = GTM(n_components=2, max_iter=50, tol=1e-2, verbose=True)
#    proj = model.fit_transform(upData.T)
    
    
#    #plot nodes over scatter plot
#    fig, ax = plt.subplots(figsize=(10, 10))
#    plot_components(upData, proj, images=upData.reshape((upData.shape[0], up, up)))
#    fig.suptitle('GTM Embedding')
    
    
#    gtm = ugtm.runGTM(data=upData, verbose=True)
##    gtm.plot_multipanel(output="testout2",labels=labels,discrete=True,pointsize=20)
#    proj = gtm.matMeans
#    np.save('E:/University of Florida/Research/Presentations/2019_01_23_Lab_Group/Test/gtmEmbeddedData.npy',proj)

    proj = np.load('E:/University of Florida/Research/Presentations/2019_01_23_Lab_Group/Test/gtmEmbeddedData.npy')
    
    #plot nodes over scatter plot
    fig, ax = plt.subplots(figsize=(10, 10))
    plot_components(upData, proj, images=upData.reshape((upData.shape[0], up, up)))
    fig.suptitle('GTM Embedding')
