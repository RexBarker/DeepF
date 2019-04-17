#!/bin/bash
for i in {1..10}
do
    python train_frcnn.py --hf -o simple -p ../data/Anno/list_clothing_bbox_class.txt --input_weight_path model_frcnn.hdf5 &>> Log.txt
    sleep 30
done
