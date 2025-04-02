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



def dataset_iid(dataset, num_users):
    # 获取数据集中的标签列表
    labels = [label for _, label in dataset]
    # 统计每个类别的样本索引
    class_idxs = {i: [] for i in range(10)}  # CIFAR-10有10个类别
    for idx, label in enumerate(labels):
        class_idxs[label].append(idx)

    dict_users = {}
    num_per_class = len(class_idxs[0]) // num_users  # 假设每个类别的样本数相同

    for user in range(num_users):
        dict_users[user] = set()
        for class_idx in class_idxs:
            start = user * num_per_class
            end = (user + 1) * num_per_class
            dict_users[user].update(class_idxs[class_idx][start:end])

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

dict_users = dataset_iid(dataset_train, 4)
draw_data_distribution(dict_users, dataset_train, 4,
                           save_path='output/data_distribution.png')