#!/bin/bash -x

rm -f save/*
./model.py && ./plotter.py 2>&1 | pygmentize -l py3tb
