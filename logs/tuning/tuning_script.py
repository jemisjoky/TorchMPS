#!/usr/bin/env python3
import os
import sys
import csv
import numpy as np
"""
The following carries out a random parameter search, where `variables` 
specifies the search parameters and num_trials specifies how many times we want
to run our experiment for. Experiments are interruptible, and if you try to
rerun a completed experiment under the same name, the script won't do anything

The experiment logs are kept in the directory logs, one for each trial, while 
the training and test errors, as well as the smoothed loss, are kept in 
logs/+data.csv. This should allow for easy data visualization later on.
"""

### SPECIFY YOUR EXPERIMENT HERE ###

# experiment_name will form part of our logfile name, so don't include any 
# special characters or (ideally) spaces
experiment_name = 'dyn_periodic'
experiment_name = experiment_name.upper()

# The random parameters we want to search over
# OPEN_BC
# variables = {'init_std': ('lognormal', np.log(1e-13), np.log(1e4)),
#              'lr': ('lognormal', np.log(5e-4), np.log(1e2)),
#              'periodic_bc': 0}
# PERIODIC_BC
# variables = {'init_std': ('lognormal', np.log(1e0), np.log(1e5)),
#              'lr': ('lognormal', np.log(1e-4), np.log(1e2))}
# DYN_PERIODIC
# variables = {'init_std': ('lognormal', np.log(1e-6), np.log(1e6)),
#              'lr': ('lognormal', np.log(1e-3), np.log(5e1)),
#              'dynamic_mode': 1}
# DYN_OPEN
# variables = {'init_std': ('lognormal', np.log(1e-6), np.log(1e6)),
#              'lr': ('lognormal', np.log(1e-4), np.log(5e1)),
#              'periodic_bc': 0, 'dynamic_mode': 1}
variables = {'init_std': ('lognormal', np.log(1e-5), np.log(1e5)),
             'lr': 10**(-3.2),
             'dynamic_mode': 1}

# Number of random parameter choices we want to search over
num_trials = 80

# Number of output values returned by our experiment script
num_outputs = 3

### FILE INFORMATION ###

# The name of the script we invoke for each experimental run
# This should print its output value(s) to stdout immediately before exiting
exp_script = "./train_script.py"

# The CSV file we store our condensed experiment records
csv_file = "logs/+data.csv"

### THE FOLLOWING RUNS THE EXPERIMENTS, DON'T CUSTOMIZE ###

defaults = {'lr': 1e-4,
            'init_std': 1e-6,
            'l2_reg': 0.,
            'num_train': 1000,
            'batch_size': 100,
            'bond_dim': 15,
            'num_epochs': 10,
            'num_test': 5000,
            'dynamic_mode': 0,
            'periodic_bc': 1,
            'threshold': 2000,
            'cutoff': 1e-10
            }

# The parameters fed to our script, along with shorthand versions
all_params = {'lr': 'lr', 'init_std': 'std', 'l2_reg': 'wd', 'num_train': 'nt',
              'batch_size': 'bs', 'bond_dim': 'bd', 'num_epochs': 'ne', 
              'num_test': 'nte', 'dynamic_mode': 'dm', 'periodic_bc': 'bc', 
              'threshold': 'thr', 'cutoff': 'cut'}

# Check to see if we've previously run this experiment
with open(csv_file, 'r') as file:
    csv_reader = csv.reader(file, delimiter=',')
    exp_count = 1
    for i, record in enumerate(csv_reader):
        # Get the legend for the different entries of CSV records
        if i == 0:
            legend = [param.strip() for param in record]
            assert legend[0] == 'experiment_name'

            legend = legend[1: len(legend)-num_outputs]
            assert len(legend) == len(all_params)
            assert set(legend) == set(all_params.keys())
            assert list(legend) == list(all_params.keys())
            continue

        if record[0] == experiment_name:
            exp_count += 1

# Run the random trials
print(f"BEGINNING TRIAL {exp_count} OF EXPERIMENT:")
print(f"  '{experiment_name}'")
while exp_count <= num_trials:

    # Get the values of all our parameters for this run
    these_params = {}
    print("THIS TRIAL'S SPECIAL PARAMETERS")
    for param in all_params.keys():
        if param in variables:
            param_spec = variables[param]
            # param_spec is either a tuple with initialization function and 
            # parameters, or it's the value we want for param
            if isinstance(param_spec, tuple) or isinstance(param_spec, list):
                rand_fun = getattr(np.random, param_spec[0])
                these_params[param] = rand_fun(*param_spec[1:])
                print(f"{param} = {these_params[param]}")
            else:
                these_params[param] = param_spec
        else:
            these_params[param] = defaults[param]
    print()

    # Build the command line string for our invocation of train_script.py
    call_str = f"{exp_script} "
    for param in all_params.keys():
        call_str += f"--{param} {these_params[param]} "

    # Build the name of where we're storing the runtime log for this experiment
    log_name = f"logs/{experiment_name}"
    for param in variables.keys():
        log_name += f"_{all_params[param]}"
        if param in ['lr', 'init_std', 'l2_reg']:
            log_name += f"_{these_params[param]:.2e}"
        else:
            log_name += f"_{these_params[param]}"

    # Make sure we're not overwriting another log file
    base_name, suffix = log_name, 0
    while os.path.isfile(log_name):
        suffix += 1
        log_name = base_name + f"_{suffix}"
    log_name += '.log'

    # Make the actual system call, wait for it to finish, and get return code
    # NOTE: Might need to modify this for non-Unix systems
    return_code = os.system(f"{call_str} | tee {log_name}")

    # For successful run, get our values from the log file
    if return_code == 0:
        with open(log_name, 'r') as file:
            # Strip whitespace from lines
            all_lines = ["".join(l.split()) for l in file.read().splitlines()]
            for line in all_lines[::-1]:
                if line == '':
                    continue
                else:
                    # Outputs are on the last non-empty line, comma-separated
                    exp_outputs = line.split(',')
                    break
    # If the user manually exited train_script, then end the search
    elif return_code == 2:
        os.system(f"rm {log_name}")
        print("\nENDING EXPERIMENT EARLY\n")
        quit()
    # For other errors, just put down a -1 for all our outputs
    else:
        print(return_code)
        exp_outputs = [-1] * num_outputs

    # Build the line we're printing to our CSV file
    data_line = experiment_name
    for param in legend:
        data_line += f", {these_params[param]}"
    for output in exp_outputs:
        data_line += f", {output}"

    # Append our new line to the CSV file
    with open(csv_file, 'a') as file:
        file.write(f"{data_line}\n")

    # Increment our experiment number and go again
    print(f"\nTRIAL {exp_count} COMPLETE\n")
    exp_count += 1

print("EXPERIMENT COMPLETE\n")