#!/bin/bash

python3 -m isort --profile black --skip venv --skip archived --skip pykan_editable .


python3 -m autoflake -r --in-place --remove-unused-variables --remove-all-unused-imports --ignore-init-module-imports --exclude=venv,pykan_editable .