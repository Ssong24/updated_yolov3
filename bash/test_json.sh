#!/bin/bash
# YOLO to COCO using yolo file path list
python ../../Sonk_Program/Converter/YOLO2COCO.py --img-shape $1 --data-format $2 --n-classes $3

# Test with COCOAPI
#python test.py

