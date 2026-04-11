# src/vizualization/plot.py
"""
Various plots, not direct MRI representations
"""


import matplotlib.pyplot as plt

from src.config import RESULTS_FOLDER

def plot_centroid_distributions(list3D, datatype="Centroid"):

    fig, axes = plt.subplots(1,3, figsize=(12,4))

    axes[0].hist(list3D[:,0], bins=20)
    axes[0].set_title(datatype + " X")

    axes[1].hist(list3D[:,1], bins=20)
    axes[1].set_title(datatype + " Y")

    axes[2].hist(list3D[:,2], bins=20)
    axes[2].set_title(datatype + " Z")

    plt.tight_layout()
    # plt.show()
    plt.savefig(RESULTS_FOLDER / f"{datatype}_masks.png")
