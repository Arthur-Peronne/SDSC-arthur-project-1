import glob
import numpy as np

import importdata_functions as idf
from paths import *



training_files = glob.glob(path_datadir + "training/patient*/Info.cfg")
testing_files = glob.glob(path_datadir + "testing/patient*/Info.cfg")
all_files = training_files + testing_files # and optional: all_files.sort() 

# print(all_files)
group_list, height_list, weight_list = [], [], []
for file in all_files:
    dic = idf.read_info_cfg(file)
    group_list.append(dic["Group"])
    height_list.append(dic["Height"])
    weight_list.append(dic["Weight"])

botlimit, toplimit = sorted(height_list)[10],  sorted(height_list)[-10]
print(botlimit, toplimit)
print(min(height_list), max(height_list))
# a = np.linspace(min(weight_list), max(weight_list), 6)
# print(a)

# print(group_list)
# print(height_list)
# print(weight_list)
# print(set(group_list))
# print(min(height_list), max(height_list))
# print(min(weight_list), max(weight_list))

# scale = [(weight - min(weight_list))/(max(weight_list) - min(weight_list))  for weight in weight_list]
# print(scale)
# pf.plot_pcvalues_2d(X_pca, pc_n1, pc_n2, "allpatients_epoch0", "_pc_in_eigenbase", scale_str ='Patient number', segments=False) #axisscale_fixed=False