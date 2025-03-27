#!/bin/bash

python_path="D:/Anaconda3/envs/pytorch/python.exe"
python_file="CIFAR-10_corr_long_offline_new1.py"
second_python_file="CIFAR-10_no_corr_long_offline.py"

#for epochs in 100; do
#    for disconnect_prob in 0.25 0.20 0.15 0.30 0.35 0.40; do
#        echo "Running with epochs=$epochs and disconnect_prob=$disconnect_prob"
#        $python_path $python_file --epochs $epochs --disconnect_prob $disconnect_prob
#    done
#done

for epochs in 100; do
    for disconnect_prob in 0.50 0.60 0.70 0.80 0.90 ; do
        echo "Running with epochs=$epochs and disconnect_prob=$disconnect_prob"
        $python_path $python_file --epochs $epochs --disconnect_prob $disconnect_prob
    done
done

#for epochs in 100; do
#    for disconnect_prob in 0.25 0.20 0.15 0.30 0.35 0.40; do
#        echo "Running with epochs=$epochs and disconnect_prob=$disconnect_prob"
#        $python_path $second_python_file --epochs $epochs --disconnect_prob $disconnect_prob
#    done
#done

for epochs in 100; do
    for disconnect_prob in 0.50 0.60 0.70 0.80 0.90; do
        echo "Running with epochs=$epochs and disconnect_prob=$disconnect_prob"
        $python_path $second_python_file --epochs $epochs --disconnect_prob $disconnect_prob
    done
done