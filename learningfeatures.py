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
import math


N_FAKE_IMGS = 100
HEIGHT = 32
WIDTH = 32

HEIGHT = 8
WIDTH = 8

FEATURE_MAP_D = 2
TRUNCATION_EPS = 1e-3

def loadMnist(batch_size=1):
    """ Load MNIST dataset"""

    #transform input/output to tensor
    transform = transforms.Compose([
        transforms.Pad(2), # Makes 32x32, Log2(32) = 5  
        transforms.ToTensor(),  
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

    N = HEIGHT * WIDTH
    phi = np.ones((2, N))
    phi[1, :] = np.squeeze(vector)

    return phi.T


def custom_feature(data_loader, fake_img=True):
    """ For each image: 
            Transform each pixel of each image to a vector of dimension 2 """
    
    #dimensions of feature tensor Phi
    dim1 = N_FAKE_IMGS # len(data_loader) #number of images

    if not fake_img:
        dim1 = len(data_loader)

    dim2 = HEIGHT * WIDTH
    dim3 = 2 
    
    Phi = np.zeros((dim1, dim2, dim3))
   
    for batch_idx, (x, target) in enumerate(data_loader):
        if fake_img:
            # Expand
            x = x[None, :]

        image = x[0, 0, :, :]
        image = image.flatten() #vectorize the image
        image = local_feature_vectors(image)
        Phi[batch_idx, :, :] = image

    return Phi

def reduced_covariance(Phi, s1, s2):
    """ Compute the reduced covariance matrix given the two position (s1,s2) in feature matrix of an image.
        Example: to compute the reduced covariance matrix ro34, s1=2 and s2=3"""

    Nt = Phi.shape[0]      #number of images
    N = Phi.shape[1]       #number of local features vectors in Phi
    d = FEATURE_MAP_D
    
    ro = np.zeros((d**2,d**2))

    n_images = 0
    print(Phi.shape)

    for j in range(Nt):
        if j == 1000: #compute the reduced covariance matrix using 1000 images
            break

        n_images += 1
        #get the two local feature vectors 
        phi1 = Phi[j, s1, :]
        phi2 = Phi[j, s2, :]

        #trace over all the indices except s1 and s2 
        trace_tracker = 1
        for s in range(N):
            if s != s1 and s != s2:
                x = Phi[j, s, :]
                outer_product = np.outer(x, x.T) 
                # Again define ρ as a sum of outer products of the training data feature vectors as before in Eq. 10 ... 

                # TODO: If not sum, eigenvalues blow up!
                trace_tracker += np.trace(outer_product)

                # OLD: trace_tracker *= np.trace(outer_product)

        #compute the order 4 tensor
        mat1 = np.outer(phi1, phi1.T)
        mat2 = np.outer(phi2, phi2.T)

        # TODO: @Adel; Not equal!
        ro_j = np.outer(mat1, mat2)
        # ro_j2 = np.kron(mat1, mat2)

        # TODO: Should this also now be a plus?
        ro += trace_tracker * ro_j
        # OLD: ro += trace_tracker * ro_j

    return ro / n_images


################################### Test ###############################################################

# #test load mnist
# train_loader, test_loader = loadMnist()

# print('==>>> total trainning batch number: {}'.format(len(train_loader)))
# print('==>>> total testing batch number: {}'.format(len(test_loader)))

# Phi = custom_feature(train_loader, fake_img=False)
# print(Phi.shape)

# # Fake images for faster testing
train_loader = np.random.random((N_FAKE_IMGS, 1, HEIGHT, WIDTH))
#test feature map
Phi = custom_feature(zip(train_loader, np.random.random(N_FAKE_IMGS)))
print(Phi.shape)

tree_depth = int(math.log2(HEIGHT * WIDTH))
iterates = HEIGHT * WIDTH

tree_tensor = dict()

for layer in range(tree_depth):
    iterates = iterates // 2
    next_phi = []
    for i in range(iterates):
        if i % 2 != 0: continue 

        ind1 = i
        ind2 = i + 1

        ro = reduced_covariance(Phi, ind1, ind2)
        u, s, v = np.linalg.svd(ro)
        print("({}, {})\nU\n{}\nS{}\nV{}\n".format(ind1, ind2, u, s, v))

        eigenvalues = s**2
        trace = np.sum(eigenvalues)

        truncation_sum = 0
        # Gross notation, but makes indexing nicer
        first_truncated_eigenvalue = 0

        for eig_idx, e in enumerate(eigenvalues):
            truncation_sum += e
            first_truncated_eigenvalue += 1

            if (truncation_sum / trace) > (1 - TRUNCATION_EPS):
                break

        truncated_U = u[:, :first_truncated_eigenvalue] # keep first r cols of U
        tree_tensor[layer, ind1, ind2] = truncated_U

        z = np.dot(np.outer(x, y).flatten(), truncated_U)
        next_phi.append(z)

    next_phi = np.concatenate(next_phi)
    Phi = next_phi






"""

For each pair of indices,

calculate ro
svd
truncation 
store U

"""