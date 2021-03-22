#!/usr/bin/env python3
import sys
import time
import torch
import argparse
import numpy as np
from torchvision import transforms, datasets

torchmps_dir = ".."
sys.path.append(torchmps_dir)

from torchmps import MPS
from utils import joint_shuffle

# Get parameters for testing
parser = argparse.ArgumentParser(description="Hyperparameter tuning")
parser.add_argument(
    "--lr", type=float, default=1e-4, metavar="LR", help="learning rate (default: 1e-4)"
)
parser.add_argument(
    "--init_std",
    type=float,
    default=1e-9,
    metavar="STD",
    help="size of noise in initialization (default: 1e-9)",
)

parser.add_argument(
    "--l2_reg",
    type=float,
    default=0.0,
    metavar="WD",
    help="L2 regularization (default: 0.)",
)
parser.add_argument(
    "--num_train",
    type=int,
    default=1000,
    metavar="NT",
    help="how many MNIST images to train on",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=100,
    metavar="BS",
    help="minibatch size for training",
)

parser.add_argument(
    "--bond_dim", type=int, default=20, metavar="BD", help="bond dimension for our MPS"
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=10,
    metavar="NE",
    help="number of epochs to train for",
)
parser.add_argument(
    "--num_test",
    type=int,
    default=1000,
    metavar="NTE",
    help="how many MNIST images to test on",
)
parser.add_argument(
    "--periodic_bc", type=int, default=0, metavar="BC", help="sets boundary conditions"
)
parser.add_argument(
    "--adaptive_mode",
    type=int,
    default=0,
    metavar="DM",
    help="sets if our bond dimensions change dynamically",
)
parser.add_argument(
    "--merge_threshold",
    type=int,
    default=2000,
    metavar="TH",
    help="sets how often we change our merge state",
)
parser.add_argument(
    "--cutoff", type=float, default=1e-10, metavar="CO", help="sets our SVD truncation"
)

parser.add_argument(
    "--use_gpu",
    type=int,
    default=0,
    metavar="GPU",
    help="Whether we use a GPU (if available)",
)
parser.add_argument(
    "--random_path",
    type=int,
    default=0,
    metavar="PATH",
    help="Whether to set our MPS up along a random path",
)

args = parser.parse_args()

# MPS parameters
input_dim = 28 ** 2
output_dim = 10
bond_dim = args.bond_dim
adaptive_mode = bool(args.adaptive_mode)
periodic_bc = bool(args.periodic_bc)
init_std = args.init_std
merge_threshold = args.merge_threshold
cutoff = args.cutoff

# Training parameters
num_train = args.num_train
num_test = args.num_test
batch_size = args.batch_size
num_epochs = args.num_epochs
lr = args.lr
l2_reg = args.l2_reg

# GPU settings
use_gpu = bool(args.use_gpu) and torch.cuda.is_available()
device = torch.device("cuda:0" if use_gpu else "cpu")
torch.set_default_tensor_type(
    "torch.cuda.FloatTensor" if use_gpu else "torch.FloatTensor"
)

# Random path
random_path = bool(args.random_path)
path = list(np.random.permutation(input_dim)) if random_path else None

print("THIS TRIAL'S ALL PARAMETERS")
print("bond_dim =", bond_dim)
print("adaptive_mode =", adaptive_mode)
print("periodic_bc =", periodic_bc)
print("init_std =", init_std)
print("num_train =", num_train)
print("num_test =", num_test)
print("batch_size =", batch_size)
print("num_epochs =", num_epochs)
print("learning_rate =", lr)
print("l2_reg =", l2_reg)
print("merge_threshold =", merge_threshold)
print("cutoff =", cutoff)
print("Using device:", device)
print()
print("path =", path)
print()
sys.stdout.flush()

# Initialize the MPS module
mps = MPS(
    input_dim=input_dim,
    output_dim=output_dim,
    bond_dim=bond_dim,
    adaptive_mode=adaptive_mode,
    periodic_bc=periodic_bc,
    merge_threshold=merge_threshold,
    path=path,
)

# Set loss function and optimizer
loss_fun = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(mps.parameters(), lr=lr, weight_decay=l2_reg)

# Miscellaneous initialization
torch.set_default_tensor_type("torch.FloatTensor")
torch.manual_seed(0)
start_time = time.time()

# Get the training and test sets
transform = transforms.ToTensor()
train_set = datasets.MNIST(torchmps_dir + "/mnist", download=True, transform=transform)
test_set = datasets.MNIST(
    torchmps_dir + "/mnist", download=True, transform=transform, train=False
)

# Put MNIST data into Pytorch tensors
train_inputs = torch.stack([data[0].view(input_dim) for data in train_set])
test_inputs = torch.stack([data[0].view(input_dim) for data in test_set])
train_labels = torch.stack([data[1] for data in train_set])
test_labels = torch.stack([data[1] for data in test_set])

# Get the desired number of input data
train_inputs, train_labels = train_inputs[:num_train], train_labels[:num_train]
test_inputs, test_labels = test_inputs[:num_test], test_labels[:num_test]

num_batches = {
    name: total_num // batch_size
    for (name, total_num) in [("train", num_train), ("test", num_test)]
}
# samplers = {'train': torch.utils.data.SubsetRandomSampler(range(num_train)),
#             'test': torch.utils.data.SubsetRandomSampler(range(num_test))}
# loaders = {name: torch.utils.data.DataLoader(dataset, batch_size=batch_size,
#            sampler=samplers[name], drop_last=True) for (name, dataset) in
#            [('train', train_set), ('test', test_set)]}

# Move everything to GPU (if we're using it)
if use_gpu:
    torch.set_default_tensor_type("torch.cuda.FloatTensor")
    mps = mps.cuda(device=device)
    train_inputs = train_inputs.cuda(device=device)
    train_labels = train_labels.cuda(device=device)
    test_inputs = test_inputs.cuda(device=device)
    test_labels = test_labels.cuda(device=device)

# Let's start training!
for epoch_num in range(1, num_epochs + 1):
    running_loss = 0.0
    train_acc = 0.0

    for batch in range(num_batches["train"]):
        # for inputs, labels in loaders['train']:
        # inputs, labels = inputs.view([batch_size, 28**2]), labels.data
        inputs, labels = (
            train_inputs[batch * batch_size : (batch + 1) * batch_size],
            train_labels[batch * batch_size : (batch + 1) * batch_size],
        )

        # Call our MPS to get logit scores and predictions
        scores = mps(inputs)
        _, preds = torch.max(scores, 1)

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

    running_loss = running_loss / num_batches["train"]
    train_acc = train_acc / num_batches["train"]

    print(f"### Epoch {epoch_num} ###")
    print(f"Average loss for epoch: {running_loss:.4f}")
    print(f"Average train accuracy: {train_acc:.4f}")
    sys.stdout.flush()

    # Shuffle our training data for the next epoch
    train_inputs, train_labels = joint_shuffle(train_inputs, train_labels)

    # Evaluate accuracy of MPS classifier on the test set
    with torch.no_grad():
        test_acc = 0.0

        for batch in range(num_batches["test"]):
            inputs, labels = (
                test_inputs[batch * batch_size : (batch + 1) * batch_size],
                test_labels[batch * batch_size : (batch + 1) * batch_size],
            )

            scores = mps(inputs)
            _, preds = torch.max(scores, 1)
            test_acc += torch.sum(preds == labels).item() / batch_size

        test_acc /= num_batches["test"]

    print(f"Test accuracy:          {test_acc:.4f}")
    print(f"Runtime so far:         {int(time.time()-start_time)} sec\n")
    sys.stdout.flush()
