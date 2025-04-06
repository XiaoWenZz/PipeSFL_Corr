import os
import pandas as pd
import matplotlib.pyplot as plt


def extract_params(file_name):
    """
    从文件名中提取参数值，如 ep, dp, dr, le, lr, 数据集名称
    """
    parts = file_name.split('_')
    ep = None
    dp = None
    dr = None
    le = None
    lr = None
    for part in parts:
        if part.startswith('ep'):
            ep = part[2:]
        elif part.startswith('dp'):
            dp = part[2:]
        elif part.startswith('dr'):
            dr = part[2:]
        elif part.startswith('le'):
            le = part[2:]
        elif part.startswith('lr'):
            lr = part[2:]
    # 提取数据集名称，跳过 'no' 和 'Corr'
    try:
        acc_index = parts.index('Acc')
        dataset_start = acc_index + 1
        # 跳过 'no' 和 'Corr'
        while dataset_start < len(parts) and parts[dataset_start] in ['no', 'Corr']:
            dataset_start += 1
        dataset_end = None
        for i, part in enumerate(parts):
            if part.startswith(('ep', 'dp', 'dr', 'le', 'lr')):
                dataset_end = i
                break
        if dataset_end is not None:
            dataset = '_'.join(parts[dataset_start:dataset_end])
        else:
            dataset = '_'.join(parts[dataset_start:])
    except ValueError:
        dataset = None
    return ep, dp, dr, le, lr, dataset


def get_corr_files(corr_folder, client_or_server, ep, dp, dr, le, lr, dataset):
    """
    根据 no_corr 文件的参数，查找对应的 corr 文件
    """
    corr_files = []
    if dataset is None:
        return corr_files
    corr_file_prefix = f'{client_or_server}_Acc_Corr_{dataset}'
    # 构建匹配字符串
    match_str = f'_ep{ep}_dp{dp}_dr{dr}'
    le_lr_part = ""
    if le:
        le_lr_part += f'_le{le}'
    if lr:
        le_lr_part += f'_lr{lr}'
    # 考虑 cr 参数在中间的情况
    for f in os.listdir(corr_folder):
        if f.startswith(corr_file_prefix) and f.endswith('.csv'):
            if match_str in f and le_lr_part in f:
                corr_files.append(os.path.join(corr_folder, f))
    return corr_files


def generate_title(client_or_server, ep, dp, dr, le, lr):
    """
    根据参数生成图表标题
    """
    if client_or_server == 'Client':
        title_prefix = 'train_acc'
    else:
        title_prefix = 'test_acc'
    if le and lr:
        title = f'{title_prefix}_ep{ep}_dp{dp}_dr{dr}_le{le}_lr{lr}'
    elif le:
        title = f'{title_prefix}_ep{ep}_dp{dp}_dr{dr}_le{le}'
    elif lr:
        title = f'{title_prefix}_ep{ep}_dp{dp}_dr{dr}_lr{lr}'
    else:
        title = f'{title_prefix}_ep{ep}_dp{dp}_dr{dr}'
    return title


def plot_comparison(no_corr_file, corr_files, output_file):
    """
    绘制 no_corr 和 corr 文件数据的对比图
    """
    try:
        # 加载 no_corr 文件数据
        no_corr_data = pd.read_csv(no_corr_file, header=None).squeeze()
    except FileNotFoundError:
        print(f"Error: No_corr file {no_corr_file} not found.")
        return
    except Exception as e:
        print(f"Error reading no_corr file {no_corr_file}: {e}")
        return

    # 确定是 client 还是 server 文件
    if 'Client' in no_corr_file:
        y_label = 'Train Acc'
    else:
        y_label = 'Test Acc'

    epochs = range(len(no_corr_data))

    # 绘制 no_corr 数据
    plt.plot(epochs, no_corr_data, label='No Corr')

    # 绘制 corr 数据
    print(f"Found {len(corr_files)} corr files for {no_corr_file}: {corr_files}")
    for corr_file in corr_files:
        try:
            corr_data = pd.read_csv(corr_file, header=None).squeeze()
        except FileNotFoundError:
            print(f"Error: Corr file {corr_file} not found.")
            continue
        except Exception as e:
            print(f"Error reading corr file {corr_file}: {e}")
            continue
        cr_value = corr_file.split('_cr')[1].split('_')[0]
        plt.plot(epochs, corr_data, label=f'Corr (cr={cr_value})')

    # 生成标题
    client_or_server = 'Client' if 'Client' in no_corr_file else 'Server'
    ep, dp, dr, le, lr, _ = extract_params(no_corr_file)
    title = generate_title(client_or_server, ep, dp, dr, le, lr)

    # 设置图表属性
    plt.xlabel('Epochs')
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    # 保存图表
    try:
        plt.savefig(output_file)
    except Exception as e:
        print(f"Error saving figure to {output_file}: {e}")
    plt.close()


# 定义文件夹路径
no_corr_folder = 'csvs/no_corr/fmnist'
corr_folder = 'csvs/corr/fmnist'
output_folder = 'imgs/fmnist'

# 创建输出文件夹（如果不存在）
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# 获取 no_corr 文件夹下的所有文件
try:
    no_corr_files = [f for f in os.listdir(no_corr_folder) if f.endswith('.csv')]
except FileNotFoundError:
    print(f"Error: No_corr folder {no_corr_folder} not found.")
else:
    # 处理每个 no_corr 文件
    for no_corr_file in no_corr_files:
        # 构建 no_corr 文件的完整路径
        no_corr_file_path = os.path.join(no_corr_folder, no_corr_file)
        # 提取参数
        ep, dp, dr, le, lr, dataset = extract_params(no_corr_file)
        if any(param is None for param in [ep, dp, dr, dataset]):
            print(f"Skipping {no_corr_file} due to incomplete parameters.")
            continue
        client_or_server = 'Client' if 'Client' in no_corr_file else 'Server'

        # 查找对应的 corr 文件
        corr_files = get_corr_files(corr_folder, client_or_server, ep, dp, dr, le, lr, dataset)

        # 生成输出文件名
        title = generate_title(client_or_server, ep, dp, dr, le, lr)
        output_file = os.path.join(output_folder, f'{title}.png')

        # 绘制对比图
        plot_comparison(no_corr_file_path, corr_files, output_file)
