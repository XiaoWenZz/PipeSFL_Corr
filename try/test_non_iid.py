from torchvision import datasets
from torchvision.transforms import transforms


def draw_data_distribution(dict_users, dataset, num_users, save_path='data_distribution.png'):
    import matplotlib.pyplot as plt
    import numpy as np

    # 统计每个客户端的类别分布
    client_dist = {i: [0]*10 for i in range(num_users)}
    for client_idx, indices in dict_users.items():
        labels = [dataset[idx][1] for idx in indices]
        for label in labels:
            client_dist[client_idx][label] += 1

    # 绘制子图
    fig, axes = plt.subplots(nrows=num_users, ncols=1, figsize=(12, 3*num_users))
    for i in range(num_users):
        ax = axes[i]
        ax.bar(range(10), client_dist[i], color='skyblue')
        ax.set_title(f'Client {i} Data Distribution')
        ax.set_xlabel('Class')
        ax.set_ylabel('Count')
        ax.set_xticks(range(10))
        ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def dataset_non_iid(dataset, num_users, class_distribution):
    """
    该函数用于将数据集按照指定的类别分布划分给不同的客户端，每个客户端持有独特类别的一半数据，
    且对于重复持有的类别，不同客户端持有不重复的一半。
    :param dataset: 输入的数据集
    :param num_users: 客户端的数量
    :param class_distribution: 每个客户端应持有的类别标记列表，是一个二维列表，每个子列表长度为10，元素为0或1
    :return: 一个字典，键为客户端编号，值为该客户端持有的样本索引集合
    """
    # 检查类别分布的合理性
    if len(class_distribution) != num_users or any(len(cd) != 10 for cd in class_distribution):
        raise ValueError("类别分布列表的长度必须等于客户端数量，且每个子列表长度必须为10。")

    # 获取数据集中的标签列表
    labels = [label for _, label in dataset]
    # 统计每个类别的样本索引
    class_idxs = {i: [] for i in range(10)}
    for idx, label in enumerate(labels):
        class_idxs[label].append(idx)

    dict_users = {i: set() for i in range(num_users)}
    # 记录每个类别的已分配索引
    used_indices = {i: [] for i in range(10)}

    for user in range(num_users):
        for cls in range(10):
            if class_distribution[user][cls] == 1:
                class_indices = class_idxs[cls]
                half_len = len(class_indices) // 2
                available_indices = [idx for idx in class_indices if idx not in used_indices[cls]]
                if len(available_indices) >= half_len:
                    assigned_indices = available_indices[:half_len]
                    dict_users[user].update(assigned_indices)
                    used_indices[cls].extend(assigned_indices)
                else:
                    # 如果可用数据不足一半，将剩余数据全部分配
                    dict_users[user].update(available_indices)
                    used_indices[cls].extend(available_indices)

    return dict_users

def verify_different_data(dict_users, class_distribution, num_classes=10):
    num_clients = len(dict_users)
    for cls in range(num_classes):
        clients_with_class = []
        for client in range(num_clients):
            if class_distribution[client][cls] == 1:
                clients_with_class.append(client)

        if len(clients_with_class) > 1:
            for i in range(len(clients_with_class)):
                for j in range(i + 1, len(clients_with_class)):
                    client1 = clients_with_class[i]
                    client2 = clients_with_class[j]
                    intersection = dict_users[client1].intersection(dict_users[client2])
                    if len(intersection) == 0:
                        print(f"客户端 {client1} 和客户端 {client2} 持有的类别 {cls} 的数据没有交集，符合要求。")
                    else:
                        print(f"客户端 {client1} 和客户端 {client2} 持有的类别 {cls} 的数据有交集，不符合要求。交集数据索引为: {intersection}")



mean = [0.2860]  # FMNIST均值
std = [0.3530]  # FMNIST标准差

train_transforms = transforms.Compose([
        transforms.RandomCrop(28, padding=2),  # 适配28x28尺寸
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

dataset_train = datasets.FashionMNIST(
        root='./data/fmnist',
        train=True,
        download=True,
        transform=train_transforms
    )

class_distribution = [
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        [1, 1, 1, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 1, 1, 1]
    ]
dict_users = dataset_non_iid(dataset_train, 4, class_distribution)
draw_data_distribution(dict_users, dataset_train, 4,
                           save_path='output/data_distribution.png')
verify_different_data(dict_users, class_distribution)