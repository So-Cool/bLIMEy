#!/bin/bash

#if [ 1 -eq "$(echo "${val} = ${min}" | bc)" ]

RESIZE_DIR=/mnt/data/shared/ks1591/imagenet_val
mkdir -p $RESIZE_DIR

FILES=/mnt/data/imagenet/ILSVRC/Data/CLS-LOC/val/*
for f in $FILES
do
  #echo "Processing $f file..."
  #size=$(convert $f -print "%w/%h\n" /dev/null)
  size=$(identify -format "%w/%h" $f)
  size_a=$(identify -format "%w" $f)
  size_b=$(identify -format "%h" $f)

  colour=$(identify -format "%r" $f)

  #ratio=$(echo $size | bc -l)
  ratio=$(echo $size == 1 | bc -l)

  if [[ "$colour" == *"Gray"* ]]; then
    echo "It's there."
  else
    if [ "$size_a" -gt "255" ]; then
      if [ "$size_b" -gt "255" ]; then
        if [ "$ratio" -eq "1" ]; then
          echo $f
          echo $size
          #echo $ratio
          convert $f -resize 256x256\> $RESIZE_DIR/$(basename $f)
        fi
      fi
    fi
  fi
done
