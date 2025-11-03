
import matplotlib.pyplot as plt
import numpy as np

# 根据提供的map key 为修正率cr value 为 acc 绘制图像
map_of_corr_fmnist = {
    0.4: 0.53,
    0.5: 0.54,
    0.6: 0.55,
    0.7: 0.55,
    0.8: 0.57,
    0.9: 0.58,
    1.0: 0.60,
}

map_without_corr_fmnist = {
    0.4: 0.53,
    0.5: 0.53,
    0.6: 0.53,
    0.7: 0.53,
    0.8: 0.53,
    0.9: 0.53,
    1.0: 0.53,
}

map_of_corr_cifar10 = {
    0.4: 0.43,
    0.5: 0.44,
    0.6: 0.44,
    0.7: 0.43,
    0.8: 0.43,
    0.9: 0.45,
    1.0: 0.46,
}

map_without_corr_cifar10 = {
    0.4: 0.42,
    0.5: 0.42,
    0.6: 0.42,
    0.7: 0.42,
    0.8: 0.42,
    0.9: 0.42,
    1.0: 0.42,
}

output_path_fmnist = "corr_rate_fmnist.png"
output_path_cifar10 = "corr_rate_cifar10.png"

def plot_map_cifar10():
    plt.figure(figsize=(10, 6))
    plt.plot(list(map_of_corr_cifar10.keys()), list(map_of_corr_cifar10.values()),
             label='Corr', marker='o')
    plt.plot(list(map_without_corr_cifar10.keys()), list(map_without_corr_cifar10.values()),
             label='No Corr', marker='o')
    plt.title('Correction Rate vs Accuracy on CIFAR10', fontsize=16)
    plt.xlabel('Correction Rate (cr)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(list(map_of_corr_cifar10.keys()), fontsize=14)
    plt.yticks(np.arange(0.4, 0.5, 0.01), fontsize=14)
    plt.legend()
    plt.grid()
    plt.savefig(output_path_cifar10)
    plt.show()

def plot_map_fmnist():
    plt.figure(figsize=(10, 6))
    plt.plot(list(map_of_corr_fmnist.keys()), list(map_of_corr_fmnist.values()),
             label='Corr', marker='o')
    plt.plot(list(map_without_corr_fmnist.keys()), list(map_without_corr_fmnist.values()),
             label='No Corr', marker='o')
    plt.title('Correction Rate vs Accuracy on FMNIST', fontsize=16)
    plt.xlabel('Correction Rate (cr)', fontsize=14)
    plt.ylabel('Accuracy', fontsize=14)
    plt.xticks(list(map_of_corr_fmnist.keys()), fontsize=14)
    plt.yticks(np.arange(0.4, 0.6, 0.01), fontsize=14)
    plt.legend()
    plt.grid()
    plt.savefig(output_path_fmnist)
    plt.show()


if __name__ == "__main__":
    plot_map_fmnist()
    plot_map_cifar10()