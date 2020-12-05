"""
检查图像的模糊度，并删除3000张模糊的图像列表，用于分配训练集和测试集
"""
import os
import cv2
import glob
import csv
import pandas as pd
import numpy as np


def variance_of_laplacian(image):
    # compute the Laplacian of the image and then return the focus
    # measure, which is simply the variance of the Laplacian
    return cv2.Laplacian(image, cv2.CV_64F).var()

def blur_check(PATH):
    image_dict = {}
    list_of_images = os.listdir(PATH)
    for im in list_of_images:
        image = cv2.imread(PATH+im)
        image = cv2.resize(image, (448, 448))
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        fm = variance_of_laplacian(gray)
        #plt.imshow(image);
        #plt.show()
        image_dict[im]= fm
    return image_dict


def takeSecond(elem):
    return elem[1]


if __name__ == '__main__':
    # all_images = glob.glob('./dataset/resized_train_cropped/resized_train_cropped/*.jpeg')
    # image_dict = blur_check('./dataset/resized_train_cropped/resized_train_cropped/')
    #
    # image_list = [[k, v] for k, v in image_dict.items()]
    # 以上的流程已经跑完了，用读取文件代替
    df = pd.read_csv('./dataset/blur.csv')
    image_list = list(df.values)
    image_list.sort(key=takeSecond)

    train_filename = './dataset/trainLabels.csv'
    df = pd.read_csv(train_filename)
    ls = list(df.values)
    dic = dict(df.values)
    # 把list里面有的拿出来
    dic_exist = {}
    for l in image_list:
        if l[0].split('.')[0] in dic:
            dic_exist[l[0]] = dic[l[0].split('.')[0]]

    count_sum = [0] * 5
    for k, v in dic_exist.items():
        count_sum[v] += 1

    count_3000 = [0] * 5
    for i in range(3000):
        image_name = image_list[i][0]
        count_3000[dic_exist[image_name]] += 1

    count_6000 = [0] * 5
    for i in range(6000):
        image_name = image_list[i][0]
        count_6000[dic_exist[image_name]] += 1

    # 3000以前的图像比较模糊，难以分辨
    with open('./dataset/remove_bad.csv', 'w', newline='') as fp:
         writer = csv.writer(fp)
         writer.writerow(['image', 'level'])
         for i, l in enumerate(image_list):
             if i > 3000:
                writer.writerow([l[0], dic_exist[l[0]]])
