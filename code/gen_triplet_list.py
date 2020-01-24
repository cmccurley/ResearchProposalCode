# -*- coding: utf-8 -*-
"""
Created on Tue Jan 21 17:30:52 2020

@author: cmccurley
"""

"""
***********************************************************************
    *  File:  gen_triplet_list.py
    *
    *  Desc:  This file generates sets of training triplets from lists of 
    *         file paths and corresponding binary image-level labels.
    *
    *  Written by:  Connor H. McCurley
    *
    *  Latest Revision:  2020-01-21
    *
**********************************************************************
"""

######################################################################
######################### Import Packages ############################
######################################################################

# General packages
import numpy as np
from sklearn.model_selection import train_test_split
import scipy.io
import random
import math
import matplotlib.pyplot as plt
from PIL import Image
from itertools import combinations, combinations_with_replacement 
from skimage.feature import hog
from skimage import data, exposure
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn import (manifold, datasets, decomposition, ensemble,
                     discriminant_analysis, random_projection, neighbors)

# Custom packages

######################################################################
######################## Function Definitions ########################
######################################################################
def partition_triplets(data_paths):
    """
    ******************************************************************
        *
        *  Func:    partition_triplets()
        *
        *  Desc:    Returns list of training triplets as file paths.
        *
        *  Inputs:
        *          data_paths - full paths to image patches and 
        *                       image-level labels
        *
        *          e.g. "\path_to_image\image.png 0" 
        *          to denote the full image path and binary class label
        *
        *  Outputs:
        *          triplets_all - list of triplets as (anchor, pos, neg).
        *            The triplets are full file paths to samples with known
        *            bag-level labels.
        *
        *
        *          triplets_all_labels - binary label of anchor point
        *            in triplet.  Meaning the anchor and positive (pos. 1 & 2)
        *            have the same label (0 or 1), and the 3rd position has the
        *            opposite class label.
        *
        *
    ******************************************************************
    """

    X = []
    Y = []
    triplets_all = []
    triplets_all_labels = []

    ## Open file of sample paths and corresponding weak labels
    data = open(data_paths, 'r')
    
    ## Parse the text file
    for line in data:
        
        tempData = line.rstrip('\n').split(" ") ## Separate file path and label
        X.append(tempData[0])
        Y.append(tempData[1])
        
        y = np.array(Y)
        
    ## Partition files into all possible triplet combinations
        
    ## Get indicies of target and bg files
    ind_target =np.where(y=='1')[0]
    ind_bg = np.where(y=='0')[0]
    
    ## split files into target and bg data sets
    X_target = [X[idx] for idx in ind_target]  
    X_bg =  [X[idx] for idx in ind_bg]   
    
    ## randomly sample anchors, positives and negatives
    triplets_target = []
    triplets_target_labels = []
    triplets_bg = []
    triplets_bg_labels = []
    
    print('Generating target anchor triplets...')
    n_samples_target = math.floor(len(X_target)/3)
    
    anchors = random.sample(X_target, n_samples_target)
    pos = random.sample(X_target, n_samples_target)
    neg = random.sample(X_bg, n_samples_target)
    
    for idx in range(0,n_samples_target):
        triplet_temp = [anchors[idx], pos[idx], neg[idx]]
        triplets_target.append(triplet_temp)
        triplets_target_labels.append(1)
    
    
    print('Generating background anchor triplets...')
    n_samples_bg = math.floor(len(X_bg)/3)
    
    anchors = random.sample(X_bg, n_samples_bg)
    pos = random.sample(X_bg, n_samples_bg)
    neg = random.sample(X_target, n_samples_bg)
    
    for idx in range(0,n_samples_bg):
        triplet_temp = [anchors[idx], pos[idx], neg[idx]]
        triplets_bg.append(triplet_temp)
        triplets_bg_labels.append(0)
        
        
    ## Concatenate lists of target and background triplets
    triplets_all = triplets_target + triplets_bg
    triplets_all_labels = triplets_target_labels + triplets_bg_labels

    return triplets_all, triplets_all_labels

def plot_triplets(triplets_all, triplets_all_labels, num_figs):
    """
    ******************************************************************
        *
        *  Func:    plot_triplets()
        *
        *  Desc:    Plots sample tri-plots of anchor, pos and neg triplets.
        *
        *  Inputs:
        *          triplets_all - list of triplets as (anchor, pos, neg).
        *            The triplets are full file paths to samples with known
        *            bag-level labels.
        *
        *
        *          triplets_all_labels - binary label of anchor point
        *            in triplet.  Meaning the anchor and positive (pos. 1 & 2)
        *            have the same label (0 or 1), and the 3rd position has the
        *            opposite class label.
        *
        *          num_figs - number of random triplets to plot
        *
        *
        *  Outputs:
        *          Plots sample tri-plots of anchor, pos and neg triplets.
        *
    ******************************************************************
    """
    
    ## Randomly sample triplets to plot
    triplets = random.sample(triplets_all, num_figs)
    
    ## Plot triplets
    for idx in range(0, num_figs):
        
        triplet_temp = triplets[idx]
        
        ## Load samples
        anchor = Image.open(triplet_temp[0])
        pos = Image.open(triplet_temp[1])
        neg = Image.open(triplet_temp[2])
        
        ## Create blank figure
        f, (ax0, ax1, ax2) = plt.subplots(1,3,sharey=True)
        ax0.imshow(anchor, cmap='gray')
        ax1.imshow(pos, cmap='gray')
        ax2.imshow(neg, cmap='gray')
        
        ## Set figure titles
        f.suptitle("Sample MIL Triplet")
        ax0.set_title("Anchor")
        ax1.set_title("Positive")
        ax2.set_title("Negative")
        
        ## Remove ticks
        ax0.get_xaxis().set_visible(False)
        ax1.get_xaxis().set_visible(False)
        ax2.get_xaxis().set_visible(False)
        
        ax0.get_yaxis().set_visible(False)
        ax1.get_yaxis().set_visible(False)
        ax2.get_yaxis().set_visible(False)

    return


def extract_features(triplets_all, triplets_all_labels):
    """
    ******************************************************************
        *
        *  Func:    extract_features()
        *
        *  Desc:    Returns feature vectors for a list of triplets.
        *
        *  Inputs:
        *          triplets_all - list of triplets as (anchor, pos, neg).
        *            The triplets are full file paths to samples with known
        *            bag-level labels.
        *
        *
        *          triplets_all_labels - binary label of anchor point
        *            in triplet.  Meaning the anchor and positive (pos. 1 & 2)
        *            have the same label (0 or 1), and the 3rd position has the
        *            opposite class label.
        *
        *  Outputs:
        *          X_triplets - list of feature vectors for triplet list.
        *
    ******************************************************************
    """
    print('Extracting features...')
    
    X_triplets = []
    
    ## Extract features on triplets
    # for idx in range(700, 704):
    for idx in range(0, len(triplets_all)):
        
        if not(idx % 50):
            print(f'Triplet {idx} of {len(triplets_all)}')
        
        triplet_temp = triplets_all[idx]
        
        ## Load samples
        anchor = Image.open(triplet_temp[0])
        pos = Image.open(triplet_temp[1])
        neg = Image.open(triplet_temp[2])
        
        ## Extract HoG features on each image
        X_anchor, _ = hog(anchor, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)
        
        X_pos, _ = hog(pos, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)
        
        X_neg, _ = hog(neg, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)
        
        X_triplets.append([X_anchor, X_pos, X_neg])
                
        ######################################################################
        
        # ## Plot HoG features
        
        # X_anchor, image_anchor = hog(anchor, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)
        
        # X_pos, image_pos = hog(pos, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)
        
        # X_neg, image_neg = hog(neg, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True, multichannel=True)
        
        # X_triplets.append([X_anchor, X_pos, X_neg])
        
        
        # fig, ax = plt.subplots(3, 2, figsize=(8, 4), sharex=True, sharey=True)

        # ax[0,0].axis('off')
        # ax[0,0].imshow(anchor, cmap=plt.cm.gray)
        # ax[0,0].set_title('Anchor image')
        
        # # Rescale histogram for better display
        # image_anchor_rescaled = exposure.rescale_intensity(image_anchor, in_range=(0, 10))
        
        # ax[0,1].axis('off')
        # ax[0,1].imshow(image_anchor_rescaled, cmap=plt.cm.gray)
        # ax[0,1].set_title('Histogram of Oriented Gradients')
        # plt.show()
        
        # ax[1,0].axis('off')
        # ax[1,0].imshow(pos, cmap=plt.cm.gray)
        # ax[1,0].set_title('Positive Example Image')
        
        # # Rescale histogram for better display
        # image_pos_rescaled = exposure.rescale_intensity(image_pos, in_range=(0, 10))
        
        # ax[1,1].axis('off')
        # ax[1,1].imshow(image_pos_rescaled, cmap=plt.cm.gray)
        # ax[1,1].set_title('Histogram of Oriented Gradients')
        # plt.show()
        
        # ax[2,0].axis('off')
        # ax[2,0].imshow(neg, cmap=plt.cm.gray)
        # ax[2,0].set_title('Negative Example Image')
        
        # # Rescale histogram for better display
        # image_neg_rescaled = exposure.rescale_intensity(image_neg, in_range=(0, 10))
        
        # ax[2,1].axis('off')
        # ax[2,1].imshow(image_neg_rescaled, cmap=plt.cm.gray)
        # ax[2,1].set_title('Histogram of Oriented Gradients')
        # plt.show()
        
        # fig.suptitle("HoG Features from Triplets")
    
    

    return X_triplets

def triplet_lda(X_triplets, X_labels):
    """
    ******************************************************************
        *
        *  Func:    triplet_lda()
        *
        *  Desc:    Performs traditional supervised linear discriminant analysis 
        *           by extrating pos/neg images out of triplets.
        *
        *  Inputs:
        *          X_triplets - list of triplets as (anchor, pos, neg).
        *            The triplets are feature vectors with known
        *            bag-level labels.
        *
        *
        *          X_labels - binary label of anchor point
        *            in triplet.  Meaning the anchor and positive (pos. 1 & 2)
        *            have the same label (0 or 1), and the 3rd position has the
        *            opposite class label.
        *
        *  Outputs:
        *          X_triplets - list of feature vectors for triplet list.
        *
    ******************************************************************
    """
    print('Separating triplets into classes...')
    
    X_target = []
    X_bg = []
    
    for idx, triplet in enumerate(X_triplets):
    
        if X_labels[idx]:
            X_target.append(triplet[0])
            X_target.append(triplet[1])
            X_bg.append(triplet[2])
        else:
            X_bg.append(triplet[0])
            X_bg.append(triplet[1])
            X_target.append(triplet[2])
                
    labels_target = np.ones(len(X_target))
    labels_bg = 2*np.ones(len(X_bg))
    
    y = np.concatenate((labels_target,labels_bg))
    
    X = np.asarray(X_target+X_bg)
    
    print('Fitting LDA...')
    ## Fit LDA
    clf = LinearDiscriminantAnalysis().fit_transform(X, y) 
    
    for i in range(X.shape[0]):
        plt.scatter(X[i, 0], X[i, 1],color=plt.cm.Set1(y[i] / 10.))
    
    # for i in range(X.shape[0]):
    #     plt.text(X[i, 0], X[i, 1], str(y[i]),
    #             color=plt.cm.Set1(y[i] / 10.),
    #             fontdict={'weight': 'bold', 'size': 9})

    return clf,X,y


def triplet_iso(X_triplets, X_labels):
    """
    ******************************************************************
        *
        *  Func:    triplet_iso()
        *
        *  Desc:    Performs traditional, unsupervised isomap
        *           by extrating pos/neg images out of triplets.
        *
        *  Inputs:
        *          X_triplets - list of triplets as (anchor, pos, neg).
        *            The triplets are feature vectors with known
        *            bag-level labels.
        *
        *
        *          X_labels - binary label of anchor point
        *            in triplet.  Meaning the anchor and positive (pos. 1 & 2)
        *            have the same label (0 or 1), and the 3rd position has the
        *            opposite class label.
        *
        *  Outputs:
        *          X_triplets - list of feature vectors for triplet list.
        *
    ******************************************************************
    """
    print('Separating triplets into classes...')
    
    X_target = []
    X_bg = []
    
    for idx, triplet in enumerate(X_triplets):
    
        if X_labels[idx]:
            X_target.append(triplet[0])
            X_target.append(triplet[1])
            X_bg.append(triplet[2])
        else:
            X_bg.append(triplet[0])
            X_bg.append(triplet[1])
            X_target.append(triplet[2])
                
    labels_target = np.ones(len(X_target))
    labels_bg = 2*np.ones(len(X_bg))
    
    y = np.concatenate((labels_target,labels_bg))
    
    X = np.asarray(X_target+X_bg)
    
    print('Fitting manifold...')
    ## Fit Isomap
    n_neighbors = 30
    # X_manifold = manifold.Isomap(n_neighbors, n_components=2).fit_transform(X)
    
    
    # # lle
    # clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2,
    #                                   method='standard')
    # X_manifold = clf.fit_transform(X)
    
    
    # pca
    # X_manifold = decomposition.TruncatedSVD(n_components=2).fit_transform(X)
    
    # ## lda
    # X2 = X.copy()
    # X2.flat[::X.shape[1] + 1] += 0.01  # Make X invertible
    # X_manifold = discriminant_analysis.LinearDiscriminantAnalysis(n_components=2).fit_transform(X2, y)
    
    # ## ltsa
    # clf = manifold.LocallyLinearEmbedding(n_neighbors, n_components=2, method='ltsa')
    # X_manifold = clf.fit_transform(X)
    
    # ## mds
    # clf = manifold.MDS(n_components=2, n_init=1, max_iter=100)
    # X_manifold = clf.fit_transform(X)
    
    # ## LE
    # embedder = manifold.SpectralEmbedding(n_components=2, random_state=0, eigen_solver="arpack")
    # X_manifold = embedder.fit_transform(X)
    
    # ## tsne
    # tsne = manifold.TSNE(n_components=2, init='pca', random_state=0)
    # X_iso = tsne.fit_transform(X)
    
    ## nca
    nca = neighbors.NeighborhoodComponentsAnalysis(init='random',n_components=2, random_state=0)
    X_manifold = nca.fit_transform(X, y)
    
    # ## Plot embedding in 2D
    # for i in range(X_manifold.shape[0]):
    #     plt.scatter(X_manifold[i, 0], X_manifold[i, 1],color=plt.cm.Set1(y[i] / 10.))
    
    # ## Plot embedding in 3D
    # from mpl_toolkits.mplot3d import Axes3D
    # fig = plt.figure()
    # ax = fig.add_subplot(111, projection='3d')
    # for i in range(X_manifold.shape[0]):
    #     ax.scatter(X_manifold[i, 0], X_manifold[i, 1],X_manifold[i, 2],color=plt.cm.Set1(y[i] / 10.))

    return

"""
%=====================================================================
%============================ Main ===================================
%=====================================================================
"""

if __name__== "__main__":
    
    print('================= Running Main =================')
    
    ##################################################################
    ####################### Generate Triplet List ####################
    ##################################################################
    
    data_list = 'train.txt'
    
    # ## Generate triplets
    # triplet_list, triplet_labels = partition_triplets(data_list)
    
    # ## Plot random triplets
    # num_figs = 3
    # plot_triplets(triplet_list, triplet_labels, num_figs)
    
    
    # ## Extract HoG features
    # X_triplets = extract_features(triplet_list, triplet_labels)
    
    print('================= Loading Data =================')
    ## Load training triplets
    data = np.load("training_triplets.npy",allow_pickle=True).item()
    X_triplets = data["data"]
    X_labels = data["labels"]
    
    # clf,X,y = triplet_lda(X_triplets, X_labels)
    
    triplet_iso(X_triplets, X_labels)
    