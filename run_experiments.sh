#!/bin/bash

python_path="D:/Anaconda3/envs/pytorch/python.exe"
python_file="PipeSFLV1_ResNet50_HAM10000_heartbeat_interval_serial_opt.py"

for epochs in 30 50 100; do
    for disconnect_prob in 0 0.01; do
        echo "Running with epochs=$epochs and disconnect_prob=$disconnect_prob"
        $python_path $python_file --epochs $epochs --disconnect_prob $disconnect_prob
    done
done