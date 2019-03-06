#!/usr/bin/env python3
import sys
import time
import torch
import argparse
from torchvision import transforms, datasets
sys.path.append('/home/jemis/torch_mps')
from torchmps import MPS

# Get parameters for testing
parser = argparse.ArgumentParser(description='Hyperparameter tuning')
parser.add_argument('--lr', type=float, default=1e-4, metavar='LR',
                    help='learning rate (default: 1e-4)')
parser.add_argument('--init_std', type=float, default=1e-6, metavar='STD',
                    help='size of noise in initialization (default: 1e-6)')

parser.add_argument('--l2_reg', type=float, default=0., metavar='WD',
                    help='L2 regularization (default: 0.)')
parser.add_argument('--num_train', type=int, default=1000, metavar='NT',
                    help='how many MNIST images to train on')
parser.add_argument('--batch_size', type=int, default=100, metavar='BS',
                    help='minibatch size for training')

parser.add_argument('--bond_dim', type=int, default=15, metavar='BD',
                    help='bond dimension for our MPS')
parser.add_argument('--num_epochs', type=int, default=10, metavar='NE',
                    help='number of epochs to train for')
parser.add_argument('--num_test', type=int, default=5000, metavar='NTE',
                    help='how many MNIST images to test on')
parser.add_argument('--periodic_bc', type=int, default=1, metavar='BC',
                    help='sets boundary conditions')
parser.add_argument('--dynamic_mode', type=int, default=0, metavar='DM',
                    help='sets if our bond dimensions change dynamically')
parser.add_argument('--threshold', type=int, default=2000, metavar='TH',
                    help='sets how often we change our merge state')
parser.add_argument('--cutoff', type=float, default=1e-10, metavar='CO',
                    help='sets our SVD truncation')

args = parser.parse_args()

# MPS parameters
bond_dim = args.bond_dim
dynamic_mode = bool(args.dynamic_mode)
periodic_bc = bool(args.periodic_bc)
init_std = args.init_std
threshold = args.threshold
cutoff = args.cutoff

# Training parameters
num_train = args.num_train
num_test = args.num_test
batch_size = args.batch_size
num_epochs = args.num_epochs
lr = args.lr
l2_reg = args.l2_reg

print("THIS TRIAL'S ALL PARAMETERS")
print("bond_dim =", bond_dim)
print("dynamic_mode =", dynamic_mode)
print("periodic_bc =", periodic_bc)
print("init_std =", init_std)
print("num_train =", num_train)
print("num_test =", num_test)
print("batch_size =", batch_size)
print("num_epochs =", num_epochs)
print("learning_rate =", lr)
print("l2_reg =", l2_reg)
print("threshold =", threshold)
print("cutoff =", cutoff)
print()
sys.stdout.flush()

# Initialize the MPS module
mps = MPS(input_dim=28**2, output_dim=10, bond_dim=bond_dim, 
          dynamic_mode=dynamic_mode, periodic_bc=periodic_bc,
          threshold=threshold)

# Set loss function and optimizer
loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mps.parameters(), lr=lr, weight_decay=l2_reg)

# Miscellaneous initialization
torch.set_default_tensor_type('torch.FloatTensor')
torch.manual_seed(0)
start_time = time.time()

# Get the training and test sets
transform = transforms.ToTensor()
train_set = datasets.MNIST('/home/jemis/torch_mps/mnist', download=True, 
                           transform=transform)
test_set = datasets.MNIST('/home/jemis/torch_mps/mnist', download=True, 
                          transform=transform, train=False)

# Put MNIST data into dataloaders
offset = 2323
samplers = {'train': torch.utils.data.SubsetRandomSampler(range(num_train)),
            'test': torch.utils.data.SubsetRandomSampler(range(num_test))}
loaders = {name: torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
           sampler=samplers[name], drop_last=True) for (name, dataset) in 
           [('train', train_set), ('test', test_set)]}
num_batches = {name: total_num // batch_size for (name, total_num) in
               [('train', num_train), ('test', num_test)]}

# Let's start training!
for epoch_num in range(1, num_epochs+1):
    running_loss = 0.
    train_acc = 0.

    for inputs, labels in loaders['train']:
        inputs, labels = inputs.view([batch_size, 28**2]), labels.data

        # Call our MPS to get logit scores and predictions
        scores = mps(inputs)
        _, preds = torch.max(scores, 1)

        # If our system encounters numerical instability, just halt there
        if torch.any(torch.isnan(scores)):
            print("-1, -1, -1")
            quit()

        # Compute the loss and accuracy, add them to the running totals
        loss = loss_fun(scores, labels)
        with torch.no_grad():
            accuracy = torch.sum(preds == labels).item() / batch_size
            running_loss += loss
            train_acc += accuracy

        # Backpropagate and update parameters
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        running_loss = running_loss / num_batches['train']
        train_acc = train_acc / num_batches['train']

    print(f"### Epoch {epoch_num} ###")
    print(f"Average loss for epoch: {running_loss:.3f}")
    print(f"Average train accuracy: {train_acc:.3f}")
    print(f"Runtime so far:         {int(time.time()-start_time)} sec\n")
    sys.stdout.flush()

# Evaluate accuracy of MPS classifier on the test set (only in last epoch)
with torch.no_grad():
    test_acc = 0.

    for inputs, labels in loaders['test']:
        inputs, labels = inputs.view([batch_size, 28**2]), labels.data

        scores = mps(inputs)
        _, preds = torch.max(scores, 1)
        test_acc += torch.sum(preds == labels).item() / batch_size
    test_acc /= num_batches['test']
print(f"Total runtime:          {int(time.time()-start_time)} sec\n")
            
# Finally, print the smoothed loss, training accuracy, and test accuracy for
# the last epoch. This is used by the calling hyperparameter tuner
print("Final loss, Final training accuracy, Final test accuracy")
print(f"{float(running_loss)}, {train_acc}, {test_acc}")