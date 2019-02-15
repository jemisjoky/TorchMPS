#!/usr/bin/env python3
import time
import torch
from cores import MPS
from torchvision import transforms, datasets

# MPS parameters
bond_dim = 10
dynamic_mode = False
periodic_bc = True

# Training parameters
num_train = 2000
num_test = 1000
batch_size = 100
num_epochs = 15
lr = 1e-3
l2_reg = 0.

# Initialize the MPS module
mps = MPS(input_dim=28**2, output_dim=10, bond_dim=bond_dim, 
          dynamic_mode=dynamic_mode, periodic_bc=periodic_bc)

# Set loss function and optimizer
loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mps.parameters(), lr=lr, weight_decay=l2_reg)

# Miscellaneous initialization
torch.set_default_tensor_type('torch.FloatTensor')
torch.manual_seed(0)
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

print(f"Training on {num_train} MNIST images \n"
      f"(Testing on {num_test}) for {num_epochs} epochs")
print(f"Maximum MPS bond dimension  = {bond_dim}")
print(f"Using Adam w/ learning rate = {lr:.1e}")
if l2_reg > 0:
    print(f"    and L2 regularization   = {l2_reg:.2e}")
print(f"{'Periodic' if periodic_bc else 'Open'} boundary conditions")
print(f"{'Dynamic' if dynamic_mode else 'Static'} training mode\n")

# Let's start training!
for epoch_num in range(1, num_epochs+1):
    running_loss = 0.
    running_acc = 0.

    for inputs, labels in loaders['train']:
        inputs, labels = inputs.view([batch_size, 28**2]), labels.data

        # Call our MPS to get logit scores and predictions
        scores = mps(inputs)
        _, preds = torch.max(scores, 1)

        # Compute the loss and accuracy, add them to the running totals
        loss = loss_fun(scores, labels)
        with torch.no_grad():
            accuracy = torch.sum(preds == labels).item() / batch_size
            running_loss += loss
            running_acc += accuracy

        # Backpropagate and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"### Epoch {epoch_num} ###")
    print(f"Average loss:           {running_loss / num_batches['train']:.3f}")
    print(f"Average train accuracy: {running_acc / num_batches['train']:.3f}")

    # Evaluate accuracy of MPS classifier on the test set
    with torch.no_grad():
        running_acc = 0.

        for inputs, labels in loaders['test']:
            inputs, labels = inputs.view([batch_size, 28**2]), labels.data

            # Call our MPS to get logit scores and predictions
            scores = mps(inputs)
            _, preds = torch.max(scores, 1)
            running_acc += torch.sum(preds == labels).item() / batch_size

    print(f"Test accuracy:          {running_acc / num_batches['test']:.3f}")
    print(f"Runtime so far:         {int(time.time()-start_time)} sec\n")
