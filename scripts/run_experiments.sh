#!/bin/bash

for dir in ./experiments/*; do
    python3 -m "experiments.$(basename $dir).main"
done