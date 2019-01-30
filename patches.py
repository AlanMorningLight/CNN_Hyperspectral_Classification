#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 14:41:17 2018

@author: abhi
"""
import numpy as np

I_vect =np.load('/home/abhi/Documents/Hyper/Dataset_Hyperspectral/numpy_open_data/indianpines.npy')

array_expand = I_vect[:,0,:]
for i_row in range(1, I_vect.shape[1]):
    tempmatirx = I_vect[:,i_row,:]
    array_expand = np.hstack((array_expand,tempmatirx))