#!/usr/bin/env python3
import time
import torch
from cores import MPS
from torchvision import transforms, datasets

# MPS parameters
num_pixels = 28**2
bond_dim = 10
num_labels = 10
dynamic_mode = False
periodic_bc = False

# Training parameters
num_train = 1000
num_test = 1000
batch_size = 100
num_epochs = 10

# Miscellaneous initialization
torch.set_default_tensor_type('torch.FloatTensor')
torch.manual_seed(23)
start_time = time.time()

# Get the training and test sets
transform = transforms.ToTensor()
train_set = datasets.MNIST('./mnist', download=True, transform=transform)
test_set = datasets.MNIST('./mnist', download=True, transform=transform, 
                          train=False)

# Put MNIST data into dataloaders
samplers = {'train': torch.utils.data.SubsetRandomSampler(range(num_train)),
            'test': torch.utils.data.SubsetRandomSampler(range(num_test))}
loaders = {name: torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
           sampler=samplers[name], drop_last=True) for (name, dataset) in 
           [('train', train_set), ('test', test_set)]}
num_batches = {name: total_num // batch_size for (name, total_num) in
               [('train', num_train), ('test', num_test)]}

# Initialize our MPS module
mps = MPS(input_dim=num_pixels, output_dim=num_labels, bond_dim=bond_dim, 
          dynamic_mode=dynamic_mode, periodic_bc=periodic_bc)

# Set loss function and optimizer
loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mps.parameters(), lr=1E-3)

print(f"Training on {num_train} MNIST images for {num_epochs} epochs")
print(f"Maximum MPS bond dimension = {bond_dim}\n")

# Let's start training!
for epoch_num in range(1, num_epochs+1):
    running_loss = 0.
    running_acc = 0.

    for inputs, labels in loaders['train']:
        inputs, labels = inputs.view([batch_size, num_pixels]), labels.data

        # Call our MPS to get logit scores and predictions
        scores = mps(inputs)
        _, preds = torch.max(scores, 1)

        # Compute the loss and accuracy, add them to the running totals
        loss = loss_fun(scores, labels)
        accuracy = torch.sum(preds == labels) / batch_size
        with torch.no_grad():
            running_loss += loss
            running_acc += accuracy

        # Backpropagate and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"### Epoch {epoch_num} ###")
    print(f"Average loss:           {running_loss / num_batches['train']}")
    print(f"Average train accuracy: {running_acc / num_batches['train']}")

    # Evaluate accuracy of MPS classifier on the test set
    with torch.no_grad():
        running_acc = 0.

        for inputs, labels in loaders['test']:
            inputs, labels = inputs.view([batch_size, num_pixels]), labels.data

            # Call our MPS to get logit scores and predictions
            scores = mps(inputs)
            _, preds = torch.max(scores, 1)
            running_acc += torch.sum(preds == labels) / batch_size

    print(f"Test accuracy:          {running_acc / num_batches['test']}")
    print(f"Runtime so far:         {int(time.time()-start_time)} sec\n")
