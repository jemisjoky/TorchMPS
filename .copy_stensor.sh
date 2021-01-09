#!/bin/sh

# TODO: Call this inside make-type file for calling all system-level 
# commands (e.g. sphinx doc, setting up envs)

# Move everything from STensor folder to here
cp -ru ~/stensor .

# Remove random stuff I don't want
rm stensor/{.gitignore,README.md,test_case.py}
rm -rf stensor/.git
rm -rf stensor/__pycache__/