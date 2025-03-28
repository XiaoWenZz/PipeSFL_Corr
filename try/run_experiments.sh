#!/bin/bash

python_path="D:/Anaconda3/envs/pytorch/python.exe"
python_file="CIFAR-10_corr_long_offline_new1.py"
second_python_file="CIFAR-10_no_corr_long_offline.py"

#for disconnect_round in 1 2 3 5; do
#    for epochs in 50; do
#        for disconnect_prob in 0.50 0.60 0.70 0.80; do
#            echo "Running with epochs=$epochs, disconnect_prob=$disconnect_prob and disconnect_round=$disconnect_round"
#            $python_path $python_file --epochs $epochs --disconnect_prob $disconnect_prob --disconnect_round $disconnect_round
#        done
#    done
#done

for disconnect_round in 1 2 3 5; do
    for epochs in 50; do
        for disconnect_prob in 0.50 0.60 0.70 0.80; do
            for correction_rate in 0.7 0.8 0.9 1.0; do
                echo "Running with epochs=$epochs, disconnect_prob=$disconnect_prob, disconnect_round=$disconnect_round and correction_rate=$correction_rate"
                $python_path $python_file --epochs $epochs --disconnect_prob $disconnect_prob --disconnect_round $disconnect_round --correction_rate $correction_rate
            done
        done
    done
done

#for disconnect_round in 1 2 3 5; do
#    for epochs in 50; do
#        for disconnect_prob in 0.50 0.60 0.70 0.80; do
#            echo "Running with epochs=$epochs, disconnect_prob=$disconnect_prob and disconnect_round=$disconnect_round"
#            $python_path $second_python_file --epochs $epochs --disconnect_prob $disconnect_prob --disconnect_round $disconnect_round
#        done
#    done
#done