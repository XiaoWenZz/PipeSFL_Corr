import os

# 指定目录 为当前脚本所在目录
directory = os.path.dirname(os.path.abspath(__file__))
print(directory)

# 遍历目录中的所有文件
for filename in os.listdir(directory):
    # 检查文件是否为CSV文件且名称中包含"FMNIST"
    if filename.endswith('.csv') and 'FMNIST' in filename:
        # 构建新的文件名
        new_filename = filename.replace('FMNIST', 'CIFAR')
        # 获取完整的文件路径
        old_file = os.path.join(directory, filename)
        new_file = os.path.join(directory, new_filename)
        # 重命名文件
        os.rename(old_file, new_file)
        print(f'Renamed: {old_file} to {new_file}')

print('All applicable files have been renamed.')