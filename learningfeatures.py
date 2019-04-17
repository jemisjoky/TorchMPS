#!/usr/bin/env python
# coding: utf-8
from __future__ import print_function, division
import os
import torch
from skimage import io, transform 
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, utils
import random

def loadMnist(batch_size=1):
    """ Load MNIST dataset"""

    #transform input/output to tensor
    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    #train set
    train_set = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
                     dataset=train_set,
                     batch_size=batch_size,
                     shuffle=False)

    #test set
    test_set = torchvision.datasets.MNIST(root='./data', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=batch_size,
                    shuffle=False)
    return [train_loader, test_loader] 


def local_feature_vectors(vector):
    """ Transform a vector representing an image to a matrix where the first row=[1,1,...,1] 
        and the elements of the second row are the elements of the vector  """

    N = vector.shape[1]
    phi = torch.ones(2, N)
    phi[1,:] = vector
    return torch.transpose(phi, 0, 1)

def custom_feature(data_loader):
    """ For each image: 
            Transform each pixel of each image to a vector of dimension 2 """
    
    #dimensions of feature tensor Phi
    dim1 = len(data_loader) #number of images
    
    #get height and width of an image
    for batch_idx, (x, target) in enumerate(data_loader):
        [h,w] = x[0,0,:,:].shape
        break 
    
    dim2 = h*w
    dim3 = 2 
    
    Phi = torch.zeros(dim1, dim2, dim3)
    
    for batch_idx, (x, target) in enumerate(data_loader):
        image = x[0,0,:,:]
        image.resize_(1, dim2) #vectorize the image
        image = local_feature_vectors(image)
        Phi[batch_idx, :, :] = image

    return Phi

def reduced_covariance(Phi, s1, s2):
    """ Compute the reduced covariance matrix given the two position (s1,s2) in feature matrix of an image.
        Example: to compute the reduced covariance matrix ro34, s1=2 and s2=3"""

    Nt = Phi.shape[0]      #number of images
    N = Phi.shape[1]       #number of local features vectors in Phi
    # d = Phi[0,:,s1].shape[-1] #dimension of the local feature vector
    d = 2
    
    ro = torch.zeros(d**2,d**2)
    print(ro.shape)
    for j in range(Nt):
        #get the two local feature vectors 
        phi1 = Phi[j, s1, :]
        phi2 = Phi[j, s2, :]

        #trace over all the indices except s1 and s2 
        traceProd = 1
        for s in range(N):
            if s != s1 and s != s2:
                x = Phi[j, s, :]
                outer_product = np.outer(x, np.transpose(x)) 
                traceProd *= torch.trace(outer_product)

        #compute the order 4 tensor
        mat1 = phi1[:, None] @ phi1[None,:]
        mat2 = phi2[:, None] @ phi2[None,:]

        ro_j = mat1[:, :, None] @ mat2[:, None, :]
        ro +=  traceProd * ro_j.resize_(d**2,d**2) #reshape the reduced covariance as matrix 
        
        if j == 1000: #compute the reduced covariance matrix using 1000 images
            break

    return ro



################################### Test ###############################################################

#test load mnist
train_loader, test_loader = loadMnist()

print('==>>> total trainning batch number: {}'.format(len(train_loader)))
print('==>>> total testing batch number: {}'.format(len(test_loader)))

print('==>>> Plot some images:')
for batch_idx, (x, target) in enumerate(train_loader):
    print(batch_idx, x.shape, target)
    plt.imshow(x[0,0,:,:])
    plt.show()
    if batch_idx == 1:
        break

#test feature map
Phi = custom_feature(train_loader)
print(Phi.shape)

#compute the reduced covariance matrix ro12
ro = reduced_covariance(Phi, 0, 1)
ro = reduced_covariance(Phi, 2, 3)
print(ro.shape)