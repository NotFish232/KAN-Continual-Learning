#!/bin/bash

for file in ./results/*; do
    if [ -d $file ]; then
        rm -r $file
    fi
done