"""
将数据集分为训练集和测试集
"""
import pandas as pd
import numpy as np
import random
import csv
import os

data = dict()
data[0] = []
data[1] = []
data[2] = []
data[3] = []

for filename in os.listdir('./dataset/image'):
    if filename.endswith('.png'):
        if filename.startswith('NL'):
            data[0].append(filename)  # Normal retina
        elif filename.startswith('ca'):  # Cataract
            data[1].append(filename)
        elif filename.startswith('Gl'):  # Glaucoma
            data[2].append(filename)
        elif filename.startswith('Re'):  # Retina Disease
            data[3].append(filename)

train_dict = dict()
train_dict[0] = []
train_dict[1] = []
train_dict[2] = []
train_dict[3] = []

val_dict = dict()
val_dict[0] = []
val_dict[1] = []
val_dict[2] = []
val_dict[3] = []
# 遍历5个类型
for k, v in data.items():
    sample = random.sample(range(0, len(v)), round(len(v) * 0.2))
    sample.sort()
    j = 0
    for i in range(len(v)):
        if j < len(sample) and i == sample[j]:
            val_dict[k].append(v[i])
            j += 1
        elif j == len(sample):
            train_dict[k].append(v[i])
        elif i != sample[j]:
            train_dict[k].append(v[i])

with open('./dataset/train.csv', 'w', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerow(['image', 'level'])
    for k, v in train_dict.items():
        for i in range(len(v)):
            writer.writerow([v[i], k])

with open('./dataset/val.csv', 'w', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerow(['image', 'level'])
    for k, v in val_dict.items():
        for i in range(len(v)):
            writer.writerow([v[i], k])





# # 读取文件，转换格式
# file_name = './dataset/remove_bad.csv'
#
# df = pd.read_csv(file_name)
# ls = list(df.values)
# # 初始化格式，使用字典存储每个row
# data = dict()
# new_data = dict()
# data[0] = []
# data[1] = []
# data[2] = []
# data[3] = []
# data[4] = []
# new_data[0] = []
# new_data[1] = []
# new_data[2] = []
# new_data[3] = []
# new_data[4] = []
# # 将每个row放到对应的桶
# # w mi
#
# for l in ls:
#     data[l[1]].append(l)
#
#
# # 将数据集分开为训练集和测试集
# # num_trian = [1225, 700, 980, 315, 280]
# train_list = []
# test_list = []
#
# # 遍历5个类型
# for k, v in data.items():
#     sample = random.sample(range(0, len(v)), round(len(v) * 0.1))
#     sample.sort()
#     j = 0
#     for i in range(len(v)):
#         if j < len(sample) and i == sample[j]:
#             test_list.append(v[i])
#             j += 1
#         elif j == len(sample):
#             train_list.append(v[i])
#         elif i != sample[j]:
#             train_list.append(v[i])
#
#
#
# # 保存train_list_jpeg和test_list_jpeg
# with open('./dataset/32000train.csv', 'w', newline='') as fp:
#     writer = csv.writer(fp)
#     writer.writerow(['image', 'level'])
#     writer.writerows(train_list)
# with open('./dataset/32000test.csv', 'w', newline='') as fp:
#     writer = csv.writer(fp)
#     writer.writerow(['image', 'level'])
#     writer.writerows(test_list)




