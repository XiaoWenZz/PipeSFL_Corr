#!/bin/bash

python_path="D:/Anaconda3/envs/pytorch/python.exe"
python_file_cifar="CIFAR_corr_long_offline_new2.py"
second_python_file_cifar="CIFAR_no_corr_long_offline.py"
python_file_fmnist="FMNIST_corr_long_offline_new1.py"
second_python_file_fmnist="FMNIST_no_corr_long_offline.py"


for disconnect_round in 1; do
    for epochs in 20; do
        for disconnect_prob in 0.40; do
            for correction_rate in 0.4 0.5 0.6 0.7 0.8 0.9 1.0; do
                for local_ep in 5; do
                  for lr in 0.001; do
                      echo "CIFAR Running with epochs=$epochs, disconnect_prob=$disconnect_prob, disconnect_round=$disconnect_round, correction_rate=$correction_rate and local_ep=$local_ep"
                      $python_path $python_file_cifar --epochs $epochs --disconnect_prob $disconnect_prob --disconnect_round $disconnect_round --correction_rate $correction_rate --local_ep $local_ep --lr $lr
                      echo "CIFAR Running with epochs=$epochs, disconnect_prob=$disconnect_prob, disconnect_round=$disconnect_round and local_ep=$local_ep"
                      $python_path $second_python_file_cifar --epochs $epochs --disconnect_prob $disconnect_prob --disconnect_round $disconnect_round --local_ep $local_ep --lr $lr
                      echo "FMNIST Running with epochs=$epochs, disconnect_prob=$disconnect_prob, disconnect_round=$disconnect_round, correction_rate=$correction_rate and local_ep=$local_ep"
                      $python_path $python_file_fmnist --epochs $epochs --disconnect_prob $disconnect_prob --disconnect_round $disconnect_round --correction_rate $correction_rate --local_ep $local_ep --lr $lr
                      echo "FMNIST Running with epochs=$epochs, disconnect_prob=$disconnect_prob, disconnect_round=$disconnect_round and local_ep=$local_ep"
                      $python_path $second_python_file_fmnist --epochs $epochs --disconnect_prob $disconnect_prob --disconnect_round $disconnect_round --local_ep $local_ep --lr $lr
                  done
                done
            done
        done
    done
done
