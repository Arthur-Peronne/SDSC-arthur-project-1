# # src/visualize_data.py
# """
# Script pour visualize data
# """

# # import matplotlib
# # matplotlib.use('Agg')  # Force le backend non-interactif
# # import matplotlib.pyplot as plt

# import nibabel as nib # to import the files

# from nilearn import plotting
# from nilearn import image
# from nilearn.plotting import plot_stat_map, show

# # Directories 
# # data_dir = "/home/renku/work/s3-bucket/ACDC/testing/"
# # images_path = "/home/renku/work/SDSC-arthur-project-1/images/"

# # Parameters : 
# patient_name1 = "patient102"
# file_name1 = "4d"

# # # Load the .nii.gz file
# # file_name_total = data_dir + patient_name1 + "/" + patient_name1 + "_" + file_name1 + ".nii.gz"
# # data_extracted1= nib.load(file_name_total)
# # print(file_name_total)
# # print(data_extracted1.shape)

# # Extract first temporal volume
# data1_t1 = image.index_img(data_extracted1, 0)
# print(data1_t1.shape)

# # Plotting all t
# def plot_epoch(data_extracted, patient_name):
#     n_epochs = data_extracted.shape[3]
#     for t in range(n_epochs):
#         vol = image.index_img(data_extracted, t)
#         plot_stat_map(
#             vol,
#             title=patient_name + f" – epoch {t}",
#             output_file=images_path+patient_name + "_epochs_"+repr(t)+".png"
#         )
#         print(f"Saved epoch {t}")

# plot_epoch(data_extracted1, patient_name1)

# Plotting others
# file_name2 = "frame01"
# data_extracted2= nib.load(data_dir + patient_name1 + "/" + patient_name1 + "_" + file_name2 + ".nii.gz")
# print(data_extracted2.shape)
# plot_stat_map(
#     data_extracted2,
#     title=patient_name1 + file_name2 ,
#     output_file=images_path+patient_name1 + "_" + file_name2 +".png"
# )

