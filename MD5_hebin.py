# MD5加密转换文件名


import cv2 as cv
import numpy as np
import time
import hashlib
import os
import shutil

img_path = '。/datasets/nead2/'
unormalimg_path = './vscode-project/python/yolov5-6.1/datasets/panbie/traindata/images/'
normalimg_path = './vscode-project/python/yolov5-6.1/datasets/panbie/traindata/normalimages/'
label_path = './datasets/label/'
unormallabel_path = './vscode-project/python/yolov5-6.1/datasets/panbie/traindata/labels/'

def md5value_lower(key):
    input_name = hashlib.md5()
    input_name.update(key.encode("utf-8"))
    # print("小写的16位" + (input_name.hexdigest())[8:-8].lower())
    return input_name.hexdigest()[8:-8].lower()

def md5value_upper(key):
    input_name = hashlib.md5()
    input_name.update(key.encode("utf-8"))
    # print("大写的16位" + (input_name.hexdigest())[4:-12].lower())
    return input_name.hexdigest()[4:-12].lower()


img = os.listdir(img_path)  
unormal_img = []
normal_img = []
label = os.listdir(label_path)
label_name = [i for i in label]
md5 = os.listdir(unormalimg_path)
md5_list = [i.split('.jpg')[0] for i in md5]

for i in img:
    if len(i.split('_'))!=2:
        normal_img.append(i)
    elif len(i.split('_'))==2:
        unormal_img.append(i)
print(len(unormal_img),len(normal_img))

for index_u in unormal_img:
    index_n = index_u.split('.jpg')[0]+'_noraml.jpg'
    index_l = index_u.split('.jpg')[0]+'.txt'
    print('开始转换:',index_u)
    unormal_name = md5value_lower(index_u) 
    if unormal_name not in md5_list:
        shutil.copyfile(img_path+index_u, unormalimg_path+unormal_name+'.jpg')
        shutil.copyfile(img_path+index_n, normalimg_path+unormal_name+'.jpg')
        print('%s转换成功'%index_l)
        shutil.copyfile(label_path+index_l, unormallabel_path+unormal_name+'.txt')
    elif unormal_name in md5_list:
        print('文件名重复')
        unormal_name = md5value_upper(index_u)
        shutil.copyfile(img_path+index_u, unormalimg_path+unormal_name+'.jpg')
        shutil.copyfile(img_path+index_n, normalimg_path+unormal_name+'.jpg')
        print('%s转换成功'%index_l)
        shutil.copyfile(label_path+index_l, unormallabel_path+unormal_name+'.txt')