#!/bin/bash

for file in ./examples/*.py;
    do python3 -m "examples.$(basename $file .py)"
done