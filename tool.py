import os
import shutil

# 定义源目录和目标目录
source_dir = '/home/s02009/data/hsnet_date'
target_dir = '/home/s02009/data/hsnet_date/VOC_Val/'

# 遍历源目录下的所有文件
for file_name in os.listdir(source_dir):
    # 如果文件名包含"CAM_VOC_Train"
    if "CAM_VOC_Val" in file_name:
        # 拼接完整的文件路径
        source_file = os.path.join(source_dir, file_name)
        target_file = os.path.join(target_dir, file_name)

        # 移动文件到目标目录
        shutil.move(source_file, target_file)