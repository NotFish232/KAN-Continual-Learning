#!/bin/bash

for file in ./experiments/*; do
    python3 -m "experiments.$(basename $file .py)"
done