#!/bin/bash
python -m unittest discover -v > test_results.txt 2>&1
cat test_results.txt