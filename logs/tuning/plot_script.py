#!/usr/bin/env python3
import sys
import csv
import scipy
import numpy as np
import matplotlib.pyplot as plt

### SPECIFY YOUR PLOT HERE ###

# Plot all records satisfying the specifications in given in record_spec
if len(sys.argv) == 2:
    record_spec = {'experiment_name': sys.argv[1].upper()}
else:
    # record_spec = {'dynamic_mode': 1, 'periodic_bc': 1}
    record_spec = {'experiment_name': 'base_random'.upper()}

# Parameters that that are varied in our plot (choose at most two)
plot_params = ['lr', 'init_std']
assert len(plot_params) in [1, 2]

# Name of output variable we want to plot
output_var = 'train_acc'
# Range of values for output_var (values outside this are considered outliers)
output_range = [0, 1]
# Output value we assign to broken/unstable trials
error_val = -1/3

# Specifies if we want log axes, one for each parameter in plot_params
log_axes = [True, True]

# Type of plot. The following plots are available: 'tricontourf'
plot_type = 'tricontourf'

### FILE INFORMATION ###

# Location of the CSV file with all our records
csv_file = "logs/+data.csv"

### THE FOLLOWING MAKES THE PLOTS, DON'T CUSTOMIZE ###

pretty_names = {'lr': 'Learning rate', 'init_std': 'Initialization scale',
                'l2_reg': 'L2 regularization', 'bond_dim': 'Bond dimension', 
                'loss': 'Crossentropy loss', 'train_acc': 'Training accuracy', 
                'test_acc': 'Testing accuracy'}

# Convert our CSV file into a list of parameter-keyed dicts
all_records = []
with open(csv_file, 'r') as file:
    reader = csv.reader(file, delimiter=',')
    for line_num, line in enumerate(reader):
        line = [field.strip() for field in line]

        # Use the first line in csv_file to make a legend
        if line_num == 0:
            legend = line
            continue

        # Later lines store records which are added to all_records
        record = {param: val for (param, val) in zip(legend, line)}
        all_records.append(record)

# Get relevant data from the records we're interested in
data = [[] for _ in plot_params]
data.append([])
for record in all_records:
    # Only look at records which satisfy all the criteria in record_spec
    if all([record[p] == str(v) for (p, v) in record_spec.items()]):
        for i, param in enumerate(plot_params):
            data[i].append(float(record[param]))

        # Modify output values to deal with outlier outputs
        output = float(record[output_var])
        if output == -1:
            output = error_val
        elif output < output_range[0]:
            output = output_range[0]
        elif output > output_range[1]:
            output = output_range[1]
        data[-1].append(output)

if plot_type == 'tricontourf':

    # Rescale the data to log scale where needed
    for i, stream in enumerate(data[:len(plot_params)]):
        if log_axes[i]:
            data[i] = np.log10(stream)

    # Plot the actual data and markers
    plt.tricontourf(*data)
    plt.colorbar()
    plt.plot(*data[:2], 'ko', ms=5)

    # Add title...
    experiment_note = f"{record_spec['experiment_name']}, " \
                      if 'experiment_name' in record_spec else ""
    plt.title(experiment_note + pretty_names[output_var])

    # ...and axis labels
    log_note = " (log10 scale)"
    axis_labels = [pretty_names[param] for param in plot_params]
    x_name, y_name = [name + log_note if log else '' for name, log in 
                      zip(axis_labels, log_axes)]
    plt.xlabel(x_name)
    plt.ylabel(y_name)

    plt.show()