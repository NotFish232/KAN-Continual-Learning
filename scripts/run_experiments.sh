#!/bin/bash

for file in ./experiments/*; do
    if [[ $file == *.py ]]; then
        python3 -m "experiments.$(basename $file .py)"
    fi
done