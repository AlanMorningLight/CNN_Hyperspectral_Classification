"""
Created on Fri Sep 21 23:05:02 2018

@author: abhi

This file is for creating patches which will be fed as input to the neural network.
The patches will then be labelled on their first index , and reshaped to become a 4D Keras Input tensor. 
"""

import scipy.interpolate as sip
import sklearn.preprocessing as sp
import scipy.io as sio
import numpy as np
import os
import matplotlib.pyplot as plt

# Set the dimension of convolution kernel(Only 2 or 3 is supported)
conv_d = 2

# Set patch size, which have to be same as that of unlabelled data.
img_size = 8

# Set number of bands adopted in experiments
num_bands = 200

# Path of output label value
#path_ip_labels = os.path.join(path_ip_base, "label.npy")

# Set path of output data
#path_scaled_ip = r'/home/abhi/Documents/Hyper/Dataset_Hyperspectral/Indian_pines_corrected_interp_scaled.npy'
#path_ip_patches = r"/home/abhi/Documents/Hyper/Dataset_Hyperspectral/patches_ip_2d.npy"

# Read original data
original_image =np.load('/home/abhi/Documents/Hyper/Dataset_Hyperspectral/numpy_open_data/indianpines.npy')
I_gt = np.load('/home/abhi/Documents/Hyper/Dataset_Hyperspectral/Ground_truths/Indian_pines_gt.npy')

# Transpose the array to "channel first"
original_image = original_image.transpose((2, 0, 1))

# Get spectral information, positions that have spectral values
list_ip_points = []
for i_row in (0,8):
    list_ip_points.append(float(i_row[0]))
array_ip_points=np.array(list_ip_points)

list_interp_points = []
for i_row in (0,8):
    list_interp_points.append(float(i_row[0]))
array_interp_points=np.array(list_interp_points)

# Spectral resampling

interp_image = np.zeros((len(array_interp_points), original_image.shape[1],original_image.shape[2]))
for r in range(0,original_image.shape[1]):
    for c in range(0, original_image.shape[2]):
        ip_spec_array = original_image[:,r,c]
        f = sip.interp1d(array_ip_points, ip_spec_array)
        interp_image[:,r,c] = f(array_interp_points)

# Expand the array for scale
array_expand = interp_image[:,0,:]
for i_row in range(1, interp_image.shape[1]):
    tempmatirx = interp_image[:,i_row,:]
    array_expand = np.hstack((array_expand,tempmatirx))
        
# Data normalization
array_expand_scaled = sp.scale(array_expand.T)

########################################################################################################################################
# Patch creation part

array_scaled = np.zeros_like(interp_image, dtype = float)
for i_row in range(0, array_scaled.shape[1]):
    array_scaled[:,i_row,:] = array_expand_scaled[i_row*array_scaled.shape[2]:
        (i_row+1)*array_scaled.shape[2],:].T


# change this equation if change img_size
mar_size = int(img_size/2)

# fill the marginal area with values in borders
arr_scal_spes = array_scaled.shape[0]
arr_scal_rows = array_scaled.shape[1]
arr_scal_cols = array_scaled.shape[2]
array_larger = np.zeros((arr_scal_spes, arr_scal_rows + mar_size*2, arr_scal_cols + mar_size*2))
array_larger[:, mar_size: arr_scal_rows + mar_size, mar_size: arr_scal_cols + mar_size] = array_scaled
for p in range(0, mar_size):
    array_larger[:, p, mar_size: arr_scal_cols + mar_size] = array_larger[:, mar_size, mar_size: arr_scal_cols + mar_size]
    array_larger[:, arr_scal_rows + mar_size + p, mar_size: arr_scal_cols + mar_size] = array_larger[:, arr_scal_rows + mar_size - 1, mar_size: arr_scal_cols + mar_size]

for q in range(0, mar_size):
    array_larger[:, 0 : arr_scal_rows + mar_size*2, q] = array_larger[:, 0 : arr_scal_rows + mar_size*2, mar_size]
    array_larger[:, 0 : arr_scal_rows + mar_size*2, arr_scal_cols + mar_size + q] = array_larger[:, 0 : arr_scal_rows + mar_size*2, arr_scal_cols + mar_size - 1]
    

larger_rows = array_larger.shape[1]
larger_cols = array_larger.shape[2]


#draw the picture
mx_data = array_larger[0, 0: larger_rows, 0: larger_cols]
#mx_data = array_larger[80, i:i+5, j:j+5]
fig = plt.figure(figsize=(20,20))
ax = fig.add_subplot(223)
cmap=plt.cm.hot
im=ax.imshow(mx_data,cmap=cmap)  
plt.colorbar(im)



# construct pixel patchs
list_sub_image = []
npy_cnt = 0
for i in range(0, larger_rows):
    for j in range(0, larger_cols):
        if i < larger_rows - img_size:
            if j < larger_cols - img_size:
                
                # np.array() can't be lost, otherwise array_larger's values will be modified
                sub_image = np.array(array_larger[:, i:i+img_size, j:j+img_size])
                
                t_vector = sub_image[:, int(img_size/2) -1, int(img_size/2) - 1]
                t_cube = np.array([[t_vector, t_vector], [t_vector, t_vector]]).transpose((2, 0, 1))
                sub_image[:, int(img_size/2) -1: int(img_size/2) +1, int(img_size/2) - 1:int(img_size/2) +1] = t_cube
                
                if conv_d == 2:
                    list_sub_image.append(sub_image)
                else:
                    list_sub_image.append(sub_image.reshape((1, num_bands, img_size, img_size)))
                   
# output the last list to file 
np.save(path_ip_patches, np.array(list_sub_image))

# Convert labeled data
# Read original data
mat_contents_lab = sio.loadmat(path_lab_mat)
original_array_lab = mat_contents_lab['indian_pines_gt']

list_lab = []
for i_row_lab in np.arange(0, 145):
    for i_col_lab in np.arange(0, 145):
        list_lab.append(original_array_lab[i_row_lab, i_col_lab])
    


