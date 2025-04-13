import matplotlib.pyplot as plt
import numpy as np

# 根据提供的map key 为离线率p value 为 acc 绘制图像
map_of_corr_fmnist = {
    0.3: 0.62,
    0.4: 0.59,
    0.5: 0.57,
    0.6: 0.55,
    0.7: 0.54,
}

map_without_corr_fmnist = {
    0.3: 0.59,
    0.4: 0.56,
    0.5: 0.54,
    0.6: 0.52,
    0.7: 0.51,
}

map_of_corr_cifar10 = {
    0.3: 0.48,
    0.4: 0.45,
    0.5: 0.44,
    0.6: 0.43,
    0.7: 0.43,
}

map_without_corr_cifar10 = {
    0.3: 0.45,
    0.4: 0.43,
    0.5: 0.42,
    0.6: 0.41,
    0.7: 0.39,
}

output_path_fmnist = "p_rate_fmnist.png"
output_path_cifar10 = "p_rate_cifar10.png"


def plot_map_fmnist():
    plt.figure(figsize=(10, 6))
    plt.plot(list(map_of_corr_fmnist.keys()), list(map_of_corr_fmnist.values()),
             label='With Correction', marker='o')
    plt.plot(list(map_without_corr_fmnist.keys()), list(map_without_corr_fmnist.values()),
             label='Without Correction', marker='o')
    plt.title('Offline Rate vs Accuracy on FMNIST', fontsize=16)
    plt.xlabel('Offline Rate (p)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(list(map_of_corr_fmnist.keys()), fontsize=14)
    plt.yticks(np.arange(0.5, 0.65, 0.02), fontsize=14)
    plt.legend()
    plt.grid()
    plt.savefig(output_path_fmnist)
    plt.show()


def plot_map_cifar10():
    plt.figure(figsize=(10, 6))
    plt.plot(list(map_of_corr_cifar10.keys()), list(map_of_corr_cifar10.values()),
             label='Corr', marker='o')
    plt.plot(list(map_without_corr_cifar10.keys()), list(map_without_corr_cifar10.values()),
             label='No Corre', marker='o')
    plt.title('Offline Rate vs Accuracy on CIFAR-10', fontsize=16)
    plt.xlabel('Offline Rate (p)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(list(map_of_corr_cifar10.keys()), fontsize=14)
    plt.yticks(np.arange(0.35, 0.5, 0.02), fontsize=14)
    plt.legend()
    plt.grid()
    plt.savefig(output_path_cifar10)
    plt.show()


if __name__ == "__main__":
    plot_map_fmnist()
    plot_map_cifar10()
