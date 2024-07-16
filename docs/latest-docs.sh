#!/bin/bash -l

# Change to lair docs directory
cd ~/.lair-docs

# Hosting directory for the docs
HOST_DIR=~/public_html/lair

# Pull the latest changes from the git repo
status=$(git pull)

# Check if there are any changes
if [[ $status == *"Already up to date."* ]]; then
    echo "No changes detected."
else
    echo Changes detected! Rebuilding docs...
    cd docs
    make html
    echo "Docs rebuilt successfully!"

    # Copy the built docs to the hosting directory
    mkdir -p $HOST_DIR
    cp -r build/html/* $HOST_DIR
fi
