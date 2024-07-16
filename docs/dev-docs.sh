#!/bin/bash -l

# This script is used to generate the developer documentation for the project.

export LAIR_DOCS_VERSION='dev'

HOST_DIR=~/public_html/lair/dev

cd ~/lair/docs

echo "Building developer documentation..."
make html

echo "Copying developer documentation to hosting directory..."
mkdir -p $HOST_DIR
cp -r build/html/* $HOST_DIR
