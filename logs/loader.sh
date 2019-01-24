#!/usr/bin/env bash
export HOME=`getent passwd $USER | cut -d':' -f6`
source ~/.bashrc
export THEANO_FLAGS=...
export PYTHONUNBUFFERED=1
echo Running pwdon $HOSTNAME
source activate torch_mps

# Actual script below
echo "Few images, no L2 regularization, but"
echo "compression after every epoch"
echo "------------- WITH GPU -------------"
./torch_mps.py
