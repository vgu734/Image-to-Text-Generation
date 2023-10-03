#!/bin/bash
#python train.py 2 0.00001 0 &&
#python train.py 2 0.00001 .001 &&
#python train.py 2 0.00001 .005 &&
#python train.py 2 0.00001 .01 &&

python train.py 2 0.000005 0 &&
python train.py 2 0.000005 .001 &&
python train.py 2 0.000005 .005 &&
python train.py 2 0.000005 .01 &&

python train.py 2 0.000001 0 &&
python train.py 2 0.000001 .001 &&
python train.py 2 0.000001 .005 &&
python train.py 2 0.000001 .01 &&

#python train.py 4 0.00001 0 &&
#python train.py 4 0.00001 .001 &&
#python train.py 4 0.00001 .005 &&
#python train.py 4 0.00001 .01 &&

python train.py 4 0.000005 0 &&
python train.py 4 0.000005 .001 &&
python train.py 4 0.000005 .005 &&
python train.py 4 0.000005 .01 &&

python train.py 4 0.000001 0 &&
python train.py 4 0.000001 .001 &&
python train.py 4 0.000001 .005 &&
python train.py 4 0.000001 .01