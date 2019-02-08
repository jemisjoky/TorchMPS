#!/usr/bin/env python3
import sys
import time
import torch
import torchvision
import torch.nn as nn
from torch_mps import MPSModule
from misc import convert_to_onehot, joint_shuffle

# Experimental parameters
length = 28
size = length**2
num_train_imgs = 100
num_test_imgs = 100
D = 10
epochs = 10
batch_size = 100            # Size of minibatches
num_labels = 10             # Always 10 for MNIST
loss_type = 'crossentropy'  # Either 'mse' or 'crossentropy'
args = {'bc': 'open',
        'weight_init_method': 'random_eye',
        'weight_init_scale': 0.01,
        'train_mode': 'dynamic'}

batches = num_train_imgs // batch_size
if batches == 0:
    raise ValueError("Batch size < # of training images")

# Initialize timing variables
start_point = time.time()
forward_time = 0.
back_time = 0.
diag_time = 0.
torch.manual_seed(23)

# Get and set GPU-related parameters
if len(sys.argv) == 1:
    want_gpu = True
else:
    want_gpu = False if sys.argv[1]=='--no_gpu' else True
use_gpu = want_gpu and torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")
torch.set_default_tensor_type('torch.cuda.FloatTensor'
              if use_gpu else 'torch.FloatTensor')
print("Using device:", device)

# Load the training and test sets
transform = torchvision.transforms.ToTensor()
train_set = torchvision.datasets.MNIST(root='./mnist',
            train=True, download=True, transform=transform)
test_set = torchvision.datasets.MNIST(root='./mnist',
            train=False, download=True, transform=transform)

train_imgs = torch.stack([data[0].view(size) for data in train_set])
train_lbls = torch.stack([data[1] for data in train_set])
train_imgs, train_lbls = train_imgs[:num_train_imgs], train_lbls[:num_train_imgs]

test_imgs = torch.stack([data[0].view(size) for data in test_set])
test_lbls = torch.stack([data[1] for data in test_set])
test_imgs, test_lbls = test_imgs[:num_test_imgs], test_lbls[:num_test_imgs]

print("Training on {0} images of size "
      "{1}x{1} for {2} epochs".format(
                               num_train_imgs, length, epochs))
print("Using bond dimension D =", D)
print()

# Build our MPS classifier using our chosen parameters
classifier = MPSModule(size=size, D=D, output_dim=num_labels, args=args)

if use_gpu:
    classifier = classifier.cuda(device=device)
    train_imgs = train_imgs.cuda(device=device)
    train_lbls = train_lbls.cuda(device=device)
    test_imgs = test_imgs.cuda(device=device)
    test_lbls = test_lbls.cuda(device=device)

if loss_type == 'mse':
    loss_f = nn.MSELoss()
elif loss_type == 'crossentropy':
    loss_f = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(classifier.parameters(), lr=1E-3)
init_time = time.time() - start_point

# Compute and print out the initial accuracy
diag_point = time.time()
train_correct = classifier.num_correct(train_imgs,
                           train_lbls, batch_size)
train_acc = train_correct / num_train_imgs
test_correct = classifier.num_correct(test_imgs,
                          test_lbls, batch_size)
test_acc = test_correct / num_test_imgs
diag_time += time.time() - diag_point

print("### before training ###")
print("training accuracy = {:.4f}".format(train_acc.item()))
print("test accuracy = {:.4f}".format(test_acc.item()))
print()

# Start the actual training
for epoch in range(epochs):
    av_loss = 0.

    for batch in range(batches):
        # Get data for this batch
        batch_imgs = train_imgs[batch*batch_size:
                               (batch+1)*batch_size]
        batch_lbls = train_lbls[batch*batch_size:
                               (batch+1)*batch_size]
        batch_vecs = convert_to_onehot(batch_lbls, num_labels)

        # Compute the loss and add it to the running total
        forward_point = time.time()
        scores = classifier(batch_imgs)
        if loss_type == 'mse':
            loss = loss_f(scores, batch_vecs.float())
        elif loss_type == 'crossentropy':
            loss = loss_f(scores, batch_lbls)
        av_loss += loss
        forward_time += time.time() - forward_point

        # Get the gradients and take an optimization step
        back_point = time.time()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        back_time += time.time() - back_point

    # Compute the training information, repeat
    loss = av_loss / batches

    print("### epoch", epoch, "###")
    print("average loss = {:.4e}".format(loss.item()))

    # Every so often, calculate the diagnostic information
    if epoch % 1 == 0:
        diag_point = time.time()
        train_correct = classifier.num_correct(train_imgs, train_lbls)
        train_acc = train_correct / num_train_imgs
        test_correct = classifier.num_correct(test_imgs, test_lbls)
        test_acc = test_correct / num_test_imgs
        diag_time += time.time() - diag_point

        current_point = time.time()
        run_time = current_point - start_point
        print("training accuracy = {:.4f}".format(train_acc.item()))
        print("test accuracy = {:.4f}".format(test_acc.item()))
        print("  forward time  = {0:.2f} sec".format(forward_time))
        print("  backprop time = {0:.2f} sec".format(back_time))
        print("  error time    = {0:.2f} sec".format(diag_time))
        print("  ---------------------------")
        print("  runtime so far = {0:.2f} sec".format(run_time))
    print()

    # Shuffle our training data for the next epoch
    train_imgs, train_lbls = joint_shuffle(train_imgs, train_lbls)

# Get relevant runtimes
run_time = time.time() - start_point

print("loading time  = {0:.2f} sec".format(init_time))
print("forward time  = {0:.2f} sec".format(forward_time))
print("backprop time = {0:.2f} sec".format(back_time))
print("error time    = {0:.2f} sec".format(diag_time))
print("---------------------------")
print("total runtime = {0:.2f} sec".format(run_time))