#!/bin/bash -l

# This script is used to generate the developer documentation for the project.

export LAIR_DOCS_VERSION='dev'

sphinx-build ~/lair/docs/source ~/public_html/lair/dev
