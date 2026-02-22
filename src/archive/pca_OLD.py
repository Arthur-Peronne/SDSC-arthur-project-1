# src/pca.py
"""
Script to perform PCA
"""

# import nibabel as nib
# import numpy as np 

# from sklearn.decomposition import PCA
# from sklearn.feature_selection import VarianceThreshold
# from sklearn.preprocessing import StandardScaler

# import matplotlib.pyplot as plt


# # Extract data from one patient: find .nii file
# data_dir = "/home/renku/work/s3-bucket/ACDC/testing/"
# patient_name1 = "patient102"
# file_name1 = "4d"
# file_name_total = data_dir + patient_name1 + "/" + patient_name1 + "_" + file_name1 + ".nii.gz"

# # Extract .nii data
# data_extracted1= nib.load(file_name_total)
# # print(file_name_total)
# # print(data_extracted1.shape)
# # print(type(data_extracted1))
# data_array = data_extracted1.get_fdata()
# # print("Shape of the data array:", data_array.shape)

# # PCA 1: each 3D image as a sample (how voxels co-vary over time, temporal dynamics) -> 30 lines, >100 000 columns (dimensions).

# # Reshape data
# data_transposed = np.transpose(data_array, (3, 0, 1, 2))
# X = data_transposed.reshape(30, -1)
# # print("Shape of X:", X.shape)  # Should be (30, >100000)

# # print("NaNs:", np.isnan(X).sum())
# # print("Infs:", np.isinf(X).sum())
# # print("Max value:", np.max(X))
# # print("Min value:", np.min(X))
# # print("Constant features:", np.sum(np.var(X, axis=0) == 0))

# # Clean data to remove constant features
# selector = VarianceThreshold()
# X_filtered = selector.fit_transform(X)
# # print("Shape after removing constant features:", X_filtered.shape)  # Should be (30, 442368 - 10645)

# # Standardize data
# # X = (X - X.mean(axis=0)) / X.std(axis=0)
# scaler = StandardScaler()
# X_scaled = scaler.fit_transform(X_filtered)

# # Perform PCA 
# # ipca = IncrementalPCA(n_components=30)
# # X_reduced = ipca.fit_transform(X_scaled)
# pca = PCA(n_components=30)  # Reduce to 30 components
# X_reduced = pca.fit_transform(X)
# # print(X_reduced)
# # print("Explained variance ratio:", pca.explained_variance_ratio_)

# # Plot PCA 
# # fig
# fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))
# fig.suptitle(patient_name1 + ': variance explained by principal components') # Title
# # plot top: explained variance
# ax1.plot(pca.explained_variance_ratio_, marker='o', linestyle='--')
# ax1.set_xlabel('Number of principal components')
# ax1.set_ylabel('Explained variance')
# ax1.set_ylim(0, 0.5)
# ax1.grid(True)
# # plot bot: cumulative explained variance
# cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
# ax2.plot(cumulative_variance, marker='o', linestyle='--')
# ax2.set_xlabel('Number of principal components')
# ax2.set_ylabel('Cumulative explained variance')
# ax2.set_ylim(0, 1.05)
# ax2.grid(True)
# #save fig
# plt.savefig("images/PCA1_explainedvariance.png")

# PCA 2: each voxel as a sample (spatial modes, spatial patterns) -> >100 000 lines, 30 columns (dimensions).





# all_volumes = []

# for i in range(50):
#     path_4d = data_dir + "/patient" + repr(i+101) + "/patient" + repr(i+101) + "_4d.nii.gz"
#     data_extracted = nib.load(path_4d)
#     all_volumes.append(data_extracted)

# print(len(all_volumes))
# print(all_volumes[0].shape)

