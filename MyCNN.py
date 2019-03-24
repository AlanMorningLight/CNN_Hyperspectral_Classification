#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 19 14:55:08 2019

@author: abhi
"""
  
import keras
import numpy as np
from keras.utils import to_categorical
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential,Input,Model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from scipy.io import loadmat 
img=loadmat('/home/abhi/Desktop/Hyperspectral Datasets/mat_files/Indian_pines_corrected.mat')
img=img['indian_pines_corrected']
imgt=loadmat('/home/abhi/Desktop/Hyperspectral Datasets/mat_files/Indian_pines_gt.mat')
gt=imgt['indian_pines_gt']
plt.imshow(gt,cmap='gnuplot2')

img=img.astype('float32')
indx=img.shape
im=img/np.max(img)

imr=im.reshape(indx[0]*indx[1],indx[2],order='F')
gtr=gt.reshape(indx[0]*indx[1],order='F')


indd=np.unique(gtr)
indd=indd+1
trts=dict()
for i in indd[:-1]:
    gtx=gtr.copy()
    gtx[gtr!=i]=0
    ind=np.nonzero(gtx)[0]
    trts[i]=train_test_split(ind, train_size=0.1, random_state=42)

xtrain=np.zeros((1,indx[2]))
xtest=np.zeros((1,indx[2]))
trainid=0
testid=0
ytrain=0
ytest=0
for j in indd[:-1]:
    xtrainn=imr[trts[j][0],:]
    xtestt=imr[trts[j][1],:]
    inx=xtrainn.shape
    iny=xtestt.shape
    ytrainn=np.zeros(inx[0])+j
    ytestt=np.zeros(iny[0])+j          
    xtrain=np.vstack((xtrain,xtrainn))
    xtest=np.vstack((xtest,xtestt))
    ytrain=np.append(ytrain,ytrainn)
    ytest=np.append(ytest,ytestt)
    trainidd=trts[j][0]
    testidd=trts[j][1]
    trainid=np.append(trainid,trainidd)
    testid=np.append(testid,testidd)
xtrain=xtrain[1::,:]
xtest=xtest[1::,:]
ytrain=ytrain[1::]
ytest=ytest[1::]
trainid=trainid[1::]
testid=testid[1::]

xtrainm = xtrain.mean(axis=0)
xtrainstd = xtrain.std(axis=0)

xtrain-=xtrainm
xtrain/=xtrainstd
xtest-=xtrainm
xtest/=xtrainstd


classes = np.unique(ytrain)
nClasses = len(classes)

print('Total number of classes:', nClasses)
print('Classes: ', classes)


xtrain = xtrain.reshape(-1,200,1)
xtest = xtest.reshape(-1,200,1)
ytrain = ytrain.reshape(-1,200,1)


#batch_size = 64
epochs=2
num_classes= 16

model = Sequential()
model.add(Conv1D(50, kernel_size=(3),activation='linear',input_shape=(1018,1),padding='same'))

'''
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D((2, 2),padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))
model.add(Conv2D(64, (3, 3), activation='linear',padding='same'))
model.add(LeakyReLU(alpha=0.1))
model.add(MaxPooling2D(pool_size=(2, 2),padding='same'))

model.add(Dense(128, activation='linear'))
model.add(LeakyReLU(alpha=0.1))
model.add(Flatten())
'''

model.add(Dense(num_classes, activation='softmax'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# Fit the model
model.fit(xtrain,ytrain, epochs=2)


# evaluate the model
scores = model.evaluate(xtrain,xtest)
print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
