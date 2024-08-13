#!/bin/bash -l

# Load git
module load git

# Load lair dev environment
conda activate lair-dev

# Run the make_docs.py script
python ~/.lair-docs/docs/make_docs.py
