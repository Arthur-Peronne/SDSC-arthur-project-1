# src/pca_testiris.py
"""
Script to perform a simple PCA on the IRIS dataset
"""

from sklearn.datasets import load_iris

import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import numpy as np

from paths import *

# Load IRIS raw data
iris = load_iris(as_frame=True)
print(iris.keys())

# Plot the pair plot
iris.frame["target"] = iris.target_names[iris.target]
_ = sns.pairplot(iris.frame, hue="target")
_.savefig(path_resultsfolder+"pairplot.png") 

#  Perform PCA
pca_1 = PCA(n_components=4)
X_reduced = pca_1.fit_transform(iris.data)

# Explicative power
explained_variance = pca_1.explained_variance_ratio_
print("explicative power of each component :", explained_variance)
print("cumulative explicative power:", explained_variance.cumsum())

# Explicative power plot
def plot_explipower():
    plt.figure(figsize=(8, 4))
    plt.bar(range(4), explained_variance, alpha=0.7, align='center',
        label='Individual explicative power', color='b')

    plt.step(range(4), explained_variance.cumsum(), where='mid',
            label='Cumulative explicative power', color='r')

    plt.ylabel('Proportion explained variace')
    plt.xlabel('Principal components')
    plt.xticks(range(4), ['PC1', 'PC2', 'PC3', 'PC4'])
    plt.legend(loc='best')
    plt.savefig(path_resultsfolder+"explained_variance.png")
plot_explipower()

# PCA plot: components 1&2 (top) and components 3&4 (bottom)
def plot_pca_2d(X):
    fig, ax = plt.subplots(2,1) 
    ax1, ax2 = ax[0], ax[1]

    scatter1 = ax1.scatter(
        X[:, 0],
        X[:, 1],
        c=iris.target,
        s=40,
    )

    scatter2 = ax2.scatter(
        X[:, 2],
        X[:, 3],
        c=iris.target,
        s=40,
    )

    ax1.set(
        # title="First two principal components",
        xlabel="1st Principal Component",
        ylabel="2nd Principal Component")
    ax1.xaxis.set_ticklabels([])
    ax1.yaxis.set_ticklabels([])

    ax2.set(
        # title="First two principal components",
        xlabel="3rd Principal Component",
        ylabel="4th Principal Component")
    ax2.xaxis.set_ticklabels([])
    ax2.yaxis.set_ticklabels([])

    # Add a legend
    legend1 = ax1.legend(
        scatter1.legend_elements()[0],
        iris.target_names.tolist(),
        loc="upper right",
        title="Classes",
    )
    ax1.add_artist(legend1)

    legend2 = ax2.legend(
        scatter2.legend_elements()[0],
        iris.target_names.tolist(),
        loc="upper right",
        title="Classes",
    )
    ax2.add_artist(legend2)

    plt.savefig(path_resultsfolder+'PC_2dx2.png')

plot_pca_2d(X_reduced)









# Create PCA -> from 4D to 3D
# X_reduced = PCA(n_components=3).fit_transform(iris.data)

# # Plot 3D
# fig = plt.figure(1, figsize=(8, 6))
# ax = fig.add_subplot(111, projection="3d", elev=-150, azim=110)

# scatter = ax.scatter(
#     X_reduced[:, 0],
#     X_reduced[:, 1],
#     X_reduced[:, 2],
#     c=iris.target,
#     s=40,
# )

# ax.set(
#     title="First three principal components",
#     xlabel="1st Principal Component",
#     ylabel="2nd Principal Component",
#     zlabel="3rd Principal Component",
# )
# ax.xaxis.set_ticklabels([])
# ax.yaxis.set_ticklabels([])
# ax.zaxis.set_ticklabels([])

# # Add a legend
# legend1 = ax.legend(
#     scatter.legend_elements()[0],
#     iris.target_names.tolist(),
#     loc="upper right",
#     title="Classes",
# )
# ax.add_artist(legend1)

# plt.show()
# plt.savefig('blabla.png')
