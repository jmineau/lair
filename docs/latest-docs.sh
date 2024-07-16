#!/bin/bash -l

# Change to lair docs directory
cd ~/.lair-docs

# Build directory for the docs
BUILD_DIR=~/public_html/lair

# Pull the latest changes from the git repo
status=$(git pull)

# Check if there are any changes
if [[ $status == *"Already up to date."* ]]; then
    echo "No changes detected."
else
    echo Changes detected! Rebuilding docs...
    sphinx-build docs/source $BUILD_DIR
    echo "Docs rebuilt successfully!"
fi
