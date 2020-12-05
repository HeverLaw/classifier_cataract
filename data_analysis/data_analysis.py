"""
将数据集分为训练集和测试集（将双眼视为一对数据，不拆分）
比例为7:3（0.428），训练集每个level约2450,1400,1960,630,560
这是每个level的比值
0.4322 0.4062 0.4322 0.4260 0.4296
注意：在样本中删除了只有单眼的数据
"""
import pandas as pd
import numpy as np
import random
import csv
# 读取文件，转换格式
file_name = '../dataset/trainLabels_cropped.csv'

df = pd.read_csv(file_name)
ls = list(df.values)
# 初始化格式，使用字典存储每个row
data = dict()
new_data = dict()
data[0] = []
data[1] = []
data[2] = []
data[3] = []
data[4] = []
new_data[0] = []
new_data[1] = []
new_data[2] = []
new_data[3] = []
new_data[4] = []
# 将每个row放到对应的桶
# 25802，2438，5288，872，708
tup = []
cat = 0
for l in ls:
    if l[0].find('left') > 0:
        tup = [l]
        cat = l[1]
    if l[0].find('right') > 0:
        tup.append(l)
        cat = l[1] if l[1] > cat else cat
        data[cat].append(tup)

# for l in ls:
#     data[l[1]].append(l)

# 新数据集的规模
num = [1750, 1000, 1400, 450, 400]

# 按序随机sample，放到new_data
for k, v in data.items():
    index = random.sample(range(0, len(v)), num[k])
    index.sort()
    # new_data[k] = data[k][index]
    for i in index:
        new_data[k].append(data[k][i])

# 将数据集分开为训练集和测试集
num_trian = [1225, 700, 980, 315, 280]
train_list = []
test_list = []

# 遍历5个类型
for k, v in new_data.items():
    sample = random.sample(range(0, len(v)), num_trian[k])
    sample.sort()
    # 使用双指针
    point_train = 0
    for i in range(0, len(v)):
        if point_train == len(sample):
            test_list.append(v[i])
        elif sample[point_train] == i:
            train_list.append(v[i])
            point_train += 1
        else:
            test_list.append(v[i])

# 将数据的shape变为(-1,1)
train_list_one = []
test_list_one = []
for a in train_list:
    train_list_one.append(a[0])
    train_list_one.append(a[1])
for a in test_list:
    test_list_one.append(a[0])
    test_list_one.append(a[1])

# 替换train_list和test_list
train_list = train_list_one
test_list = test_list_one

# 统计train和test的各个桶的数量
sta_train=[0]*5
sta_test=[0]*5
for l in train_list:
    sta_train[l[1]] += 1
for l in test_list:
    sta_test[l[1]] += 1
for a,b in zip(sta_train,sta_test):
    print(b/a)

# # train_list和test_list加上后缀.jpeg
train_list_jpeg = train_list.copy()
test_list_jpeg = test_list.copy()
for i in range(0, len(train_list_jpeg)):
    train_list_jpeg[i][0] += '.jpeg'
for i in range(0, len(test_list_jpeg)):
    test_list_jpeg[i][0] += '.jpeg'
train_list = train_list_jpeg
test_list = test_list_jpeg

# 保存train_list_jpeg和test_list_jpeg
with open('../dataset/train.csv', 'w', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerow(['image', 'level'])
    writer.writerows(train_list)
with open('../dataset/test.csv', 'w', newline='') as fp:
    writer = csv.writer(fp)
    writer.writerow(['image', 'level'])
    writer.writerows(test_list)

# 把图像放到train和test里面
import shutil
import os
if not os.path.exists('../dataset/train'):
    os.mkdir('../dataset/train')
if not os.path.exists('../dataset/test'):
    os.mkdir('../dataset/test')
for name in train_list:
    shutil.copyfile(os.path.join('./dataset/resized_train_cropped/resized_train_cropped', name[0]),
                    os.path.join('./dataset/train', name[0]))
for name in test_list:
    shutil.copyfile(os.path.join('./dataset/resized_train_cropped/resized_train_cropped', name[0]),
                    os.path.join('./dataset/test', name[0]))


