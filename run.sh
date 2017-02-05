#!/bin/bash -x

# rm -f save/*
case "$OSTYPE" in
  darwin*)  PLATFORM="OSX" ;;
  linux*)   PLATFORM="linux" ;;
  *)        PLATFORM="unknown: $OSTYPE" ;;
esac

if [[ $PLATFORM == "linux" ]]; then
  ipython model.py && ipython plotter.py 2>&1 && feh save/ModelError.png && feh save/predictions.png
elif [[ $PLATFORM == "OSX" ]]; then
   ./model.py && ./plotter.py && open save/*.png
  #  ./model.py && ./plotter.py 2>&1 && open save/*.png
   #| pygmentize -l py3tb
fi
