#!/usr/bin/env python3
import os
batch_script_name = 'batch_script.sh'

# The header for our shell script
print("#!/bin/sh")
print("#SBATCH --gres=gpu:1")
# print("#SBATCH --ntasks=1")
# print("#SBATCH --cpus-per-task=6")
print()

### SPECIFY YOUR EXPERIMENT HERE ###

# experiment_name will form part of our logfile name, so don't include any 
# special characters or (ideally) spaces
experiment_name = 'adaptive'
experiment_name = experiment_name.upper()

# The parameters we want to search over
num_trials = 6
variables = {'bond_dim': [10, 20, 40, 60, 80, 100],
             'adaptive_mode': 1}

### FILE INFORMATION ###

# The name of the script we invoke for each experimental run
exp_script = "train_script.py"

# The srun call we use to pass exp_script to SLURM
# srun_call = "srun --exclusive --cpu-bind=cores -c1"
srun_call = "srun"

### THE FOLLOWING RUNS THE EXPERIMENTS, DON'T CUSTOMIZE ###

defaults = {'lr': 1e-4,
            'init_std': 1e-9,
            'l2_reg': 0.,
            'num_train': 60000,
            'batch_size': 100,
            'bond_dim': 20,
            'num_epochs': 30,
            'num_test': 10000,
            'adaptive_mode': 0,
            'periodic_bc': 0,
            'merge_threshold': 2000,
            'cutoff': 1e-10,
            'use_gpu': 1,
            'random_path': 0
            }

# The parameters fed to our script, along with shorthand versions
all_params = {'lr': 'lr', 'init_std': 'std', 'l2_reg': 'wd', 'num_train': 'nt',
              'batch_size': 'bs', 'bond_dim': 'bd', 'num_epochs': 'ne', 
              'num_test': 'nte', 'adaptive_mode': 'dm', 'periodic_bc': 'bc', 
              'merge_threshold': 'thr', 'cutoff': 'cut', 'use_gpu': 'gpu',
              'random_path': 'path'}

# Print some metadata about the experiment
print(f'echo "Running experiment {experiment_name}, with:"')
for param in variables:
    print(f'echo "  {param}={variables[param]}"')
print()

# Print the calls for our different trials
exp_count = 0
while exp_count < num_trials:

    # Get the values of all our parameters for this run
    these_params = {}
    for param in all_params.keys():
        if param in variables:

            # param_spec is a list of values, or a single value for param
            param_spec = variables[param]
            if isinstance(param_spec, list):
                these_params[param] = param_spec[exp_count]
                print(f"# {param} = {these_params[param]}")
            else:
                these_params[param] = param_spec
                print(f"# {param} = {these_params[param]}")

        else:
            these_params[param] = defaults[param]

    # Build the command line string for our invocation of train_script.py
    call_str = f"{exp_script} "
    for param in all_params.keys():
        call_str += f"--{param} {these_params[param]} "

    # Build the name of where we're storing the runtime log for this experiment
    log_name = experiment_name
    for param in variables.keys():
        log_name += f"_{all_params[param]}"
        if param in ['lr', 'init_std', 'l2_reg']:
            log_name += f"_{these_params[param]:.2e}"
        else:
            log_name += f"_{these_params[param]}"

    # Make sure we're not overwriting another log file
    # base_name, suffix = log_name, 0
    # while os.path.isfile(log_name):
    #     suffix += 1
    #     log_name = base_name + f"_{suffix}"
    log_name += '.log'

    # Make the actual system call, which happens using srun
    # call_str = f"{srun_call} {call_str} >> {log_name} &"
    call_str = f"{srun_call} {call_str} >> {log_name}"

    # Print our system call
    print(call_str)
    print(f'echo "Done with experiment {exp_count+1}"')
    print()
    
    exp_count += 1

# print("wait")
print('echo "Done with all experiments"')

os.system(f"chmod +x {batch_script_name}")
