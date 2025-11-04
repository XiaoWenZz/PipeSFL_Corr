#!/bin/bash

python_path="/home/conda/bin/python"
python_file_cifar="CIFAR_corr_apt_alpha.py"
second_python_file_cifar="CIFAR_corr_constant_alpha.py"
third_python_file_cifar="CIFAR_no_corr_new.py"
fourth_python_file_cifar="CIFAR_no_corr_new_v1.py"


for disconnect_round in 3; do
    for epochs in 50; do
        for disconnect_prob in 0.40; do
            for local_ep in 3; do
                for lr in 0.001; do
                    echo "CIFAR Running with epochs=$epochs, disconnect_prob=$disconnect_prob, disconnect_round=$disconnect_round, correction_rate=$correction_rate and local_ep=$local_ep"
                    $python_path $python_file_cifar --epochs $epochs --disconnect_prob $disconnect_prob --disconnect_round $disconnect_round --correction_rate $correction_rate --local_ep $local_ep --lr $lr --lr_decay 1
                    echo "CIFAR Running with epochs=$epochs, disconnect_prob=$disconnect_prob, disconnect_round=$disconnect_round and local_ep=$local_ep"
                    $python_path $second_python_file_cifar --epochs $epochs --disconnect_prob $disconnect_prob --disconnect_round $disconnect_round --local_ep $local_ep --lr $lr --lr_decay 1
                    echo "CIFAR Running no correction with epochs=$epochs, disconnect_prob=$disconnect_prob, disconnect_round=$disconnect_round and local_ep=$local_ep"
                    $python_path $third_python_file_cifar --epochs $epochs --disconnect_prob $disconnect_prob --disconnect_round $disconnect_round --local_ep $local_ep --lr $lr --lr_decay 1
                    echo "CIFAR Running no correction v1 with epochs=$epochs, disconnect_prob=$disconnect_prob, disconnect_round=$disconnect_round and local_ep=$local_ep"
                    $python_path $fourth_python_file_cifar --epochs $epochs --disconnect_prob $disconnect_prob --disconnect_round $disconnect_round --local_ep $local_ep --lr $lr --lr_decay 1
                done
            done
        done
    done
done