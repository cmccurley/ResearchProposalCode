# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
from sklearn import manifold, datasets

import scipy.io

import matplotlib.pyplot as plt

# This import is needed to modify the way figure behaves
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import Isomap

#number of data points to generate
n_points = 3000

#S-curve data set
SCurve, SCurveColor = datasets.make_s_curve(n_points, random_state=0)
SCurve = SCurve.T
SCurveColor = SCurveColor.T

#Swiss roll data set
swissRoll, swissRollColor = datasets.make_swiss_roll(n_samples=n_points, noise=0.0, random_state=0)
swissRoll = swissRoll.T
swissRollColor = swissRollColor.T

# ============================= Plot data sets ================================

X = SCurve.T
color = SCurveColor
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)

ax.set_title("S-Curve Data Manifold",fontsize=16)
plt.show()

X = swissRoll.T
color = swissRollColor
fig = plt.figure()

ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.bwr)

ax.set_title("Swiss Roll Data Manifold",fontsize=16)
plt.show()

# ============================= Embed and Plot ================================
embedding = Isomap(n_components=2,n_neighbors=30)
Z = embedding.fit_transform(X)

plt.figure()
plt.scatter(Z[:, 0], Z[:, 1], c=color, cmap=plt.cm.bwr)
plt.title("Unrolled Swiss Roll Data Manifold", fontsize=16)


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