#!/bin/bash

BOLD_BLUE="\033[1;34m"
RESET="\033[0m"

for file in ./experiments/*; do
    if [[ $file == *.py ]]; then
        echo -e "${BOLD_BLUE}Running $file${RESET}"
        python3 -m "experiments.$(basename $file .py)"
        echo -e "\n"
    fi
done