#!/bin/bash

python_path="D:/Anaconda3/envs/pytorch/python.exe"
python_file="CIFAR-10_corr_long_offline_new1.py"
second_python_file="CIFAR-10_no_corr_long_offline.py"

#for disconnect_round in 1; do
#    for epochs in 20; do
#        for disconnect_prob in 0.40 0.50 0.60; do
#            for correction_rate in 1.0; do
#                for local_ep in 10; do
#                  for lr in 0.0005 0.001 0.003; do
#                      echo "Running with epochs=$epochs, disconnect_prob=$disconnect_prob, disconnect_round=$disconnect_round, correction_rate=$correction_rate and local_ep=$local_ep"
#                      $python_path $python_file --epochs $epochs --disconnect_prob $disconnect_prob --disconnect_round $disconnect_round --correction_rate $correction_rate --local_ep $local_ep --lr $lr
#                  done
#                done
#            done
#        done
#    done
#done

for disconnect_round in 1; do
    for epochs in 20; do
        for disconnect_prob in 0.40 0.50 0.60; do
            for local_ep in 10; do
              for lr in 0.0005 0.001 0.003; do
                  echo "Running with epochs=$epochs, disconnect_prob=$disconnect_prob, disconnect_round=$disconnect_round and local_ep=$local_ep"
                  $python_path $second_python_file --epochs $epochs --disconnect_prob $disconnect_prob --disconnect_round $disconnect_round --local_ep $local_ep --lr $lr
              done
            done
        done
    done
done