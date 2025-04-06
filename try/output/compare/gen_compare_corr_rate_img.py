
import matplotlib.pyplot as plt
import numpy as np

# 根据提供的map key 为修正率cr value 为 acc 绘制图像
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

output_path_fmnist = "corr_rate_fmnist.png"